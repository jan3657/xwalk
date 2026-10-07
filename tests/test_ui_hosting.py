"""`xwalk ui --public`: private sessions, the shared pre-parsed library, visitors' own
keys, and the refusals that keep a public server from fetching what a visitor names.

Offline: ontologies come from a local HTTP server, no model endpoint is called (a client
is built and inspected, never used).
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from xwalk.config import LLMSpec
from xwalk.ui import make_server
from xwalk.ui.api import ApiError, Workspace, dispatch
from xwalk.ui.hosting import (
    PublicRefusal,
    Sessions,
    check_public_url,
    endpoint_client,
    valid_session_id,
)
from xwalk.ui.library import Library
from xwalk.ui.server import UIServer

ROOT = Path(__file__).resolve().parent.parent
TOY_OBO = (
    "format-version: 1.2\n\n"
    '[Term]\nid: TOY:0001\nname: Myocardial infarction\nsynonym: "heart attack" EXACT []\n\n'
    '[Term]\nid: TOY:0002\nname: Seizure\nsynonym: "seizures" EXACT []\n'
)


def _build_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_ontology_library", ROOT / "scripts" / "build_ontology_library.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def served(tmp_path: Path) -> Iterator[str]:
    www = tmp_path / "www"
    www.mkdir()
    (www / "toy.obo").write_text(TOY_OBO, encoding="utf-8")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(www), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def library_dir(tmp_path: Path, served: str) -> tuple[Path, Path]:
    """A pre-built library (and its index cache), as the image build makes it."""
    out, cache = tmp_path / "library", tmp_path / "cache"
    code = _build_script().main(
        ["--out", str(out), "--cache", str(cache), "--only", "--url", f"toy={served}/toy.obo"]
    )
    assert code == 0
    return out, cache


# --- the build script ---------------------------------------------------------------------


def test_the_build_script_parses_indexes_and_keeps(
    library_dir: tuple[Path, Path], served: str, capsys: pytest.CaptureFixture[str]
) -> None:
    out, cache = library_dir
    entry = Library(tmp := out.parent / "own", shared=[out]).get("toy")
    assert entry.kind == "hosted" and entry.count == 2 and entry.source.endswith("/toy.obo")
    assert not tmp.exists()
    assert list(cache.glob("*/projects/*/index/bm25"))
    capsys.readouterr()
    build = _build_script()
    assert build.main(["--out", str(out), "--only", "--url", f"toy={served}/toy.obo"]) == 0
    assert "kept" in capsys.readouterr().out
    assert build.main(["--out", str(out), "--only", "--url", f"gone={served}/gone.obo"]) == 1
    with pytest.raises(SystemExit):
        build.main(["--out", str(out), "--only", "nope"])


def test_shared_entries_are_read_only(library_dir: tuple[Path, Path], tmp_path: Path) -> None:
    out, _ = library_dir
    library = Library(tmp_path / "own", shared=[out])
    assert [o.kind for o in library.all()][:1] == ["hosted"]
    from xwalk.ui.library import LibraryError

    with pytest.raises(LibraryError, match="shared"):
        library.delete("toy")
    assert (out / "toy").is_dir()


# --- URL and key rules ----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",
        "https://localhost/v1",
        "https://127.0.0.1/v1",
        "https://169.254.169.254/latest",
        "https://10.0.0.5/v1",
        "https://[::1]/v1",
        "ftp://example.com",
        "https:///nohost",
    ],
)
def test_non_public_endpoints_are_refused(url: str) -> None:
    with pytest.raises(PublicRefusal):
        check_public_url(url)


def test_a_name_resolving_to_a_private_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        address = "10.1.2.3" if host == "sneaky.example.com" else "93.184.215.14"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(PublicRefusal, match="non-public"):
        check_public_url("https://sneaky.example.com/v1")
    check_public_url("https://api.example.com/v1")


def test_the_endpoint_client_takes_the_visitor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 443))],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    spec = LLMSpec(model="m", base_url="https://api.example.com/v1", api_key_env="OPENAI_API_KEY")
    with pytest.raises(PublicRefusal, match="Settings"):
        endpoint_client(spec, {})  # the server's own key is never used for a visitor
    client = endpoint_client(spec, {"OPENAI_API_KEY": "visitor-key"})
    assert client.model == "m"
    assert "visitor-key" in repr(vars(client)) and "server-key" not in repr(vars(client))


# --- sessions -----------------------------------------------------------------------------


def test_sessions_are_private_and_expire(tmp_path: Path) -> None:
    sessions = Sessions(tmp_path / "sessions", make_workspace=lambda d: Workspace(d), ttl=60)
    (tmp_path / "sessions" / "left-from-before").mkdir()
    first, second = sessions.create(), sessions.create()
    assert valid_session_id(first) and first != second
    assert not (tmp_path / "sessions" / "left-from-before").exists()  # swept on create
    one, two = sessions.get(first), sessions.get(second)
    assert one is not None and two is not None and one.root != two.root
    assert sessions.get("not-a-session") is None and sessions.get(None) is None
    assert sessions.get("A" * 43) is None  # well formed, never issued

    import time

    removed = sessions.sweep(now=time.time() + 120, force=True)
    assert set(removed) == {first, second}
    assert sessions.get(first) is None and not (tmp_path / "sessions" / first).exists()


def test_sessions_are_bounded(tmp_path: Path) -> None:
    sessions = Sessions(tmp_path / "s", make_workspace=lambda d: Workspace(d), max_sessions=1)
    sessions.create()
    with pytest.raises(PublicRefusal, match="busy"):
        sessions.create()


# --- the public server --------------------------------------------------------------------


@pytest.fixture
def public(tmp_path: Path, library_dir: tuple[Path, Path]) -> Iterator[UIServer]:
    out, cache = library_dir
    root = tmp_path / "site"
    root.mkdir()
    server = make_server(
        str(root),
        host="127.0.0.1",
        port=0,
        token="t",
        public=True,
        library=[str(out)],
        lookup_cache=str(cache),
        max_tasks=1,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class Visitor:
    def __init__(self, server: UIServer) -> None:
        self.server = server
        self.cookie = ""

    def page(self) -> dict[str, str]:
        request = urllib.request.Request(self.server.url, headers=self._headers())
        with urllib.request.urlopen(request, timeout=10) as response:
            headers = dict(response.headers)
        if "Set-Cookie" in headers:
            self.cookie = headers["Set-Cookie"].split(";")[0]
        return headers

    def _headers(self) -> dict[str, str]:
        headers = {"X-Xwalk-Token": "t"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    def call(self, path: str, body: Any = None, *, raw: bytes | None = None) -> tuple[int, Any]:
        headers = self._headers()
        data = raw
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if raw is not None:
            headers["Content-Type"] = "application/octet-stream"
        request = urllib.request.Request(
            self.server.url.rstrip("/") + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())


def test_each_visitor_gets_a_private_workspace(public: UIServer) -> None:
    alice, bob = Visitor(public), Visitor(public)
    headers = alice.page()
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Lax" in headers["Set-Cookie"]
    bob.page()
    assert alice.cookie != bob.cookie
    assert "Set-Cookie" not in alice.page()  # a known session is kept

    status, stored = alice.call("/api/upload?name=mine.txt", raw=b"heart attack\n")
    assert status == 200
    assert bob.call("/api/files")[1]["files"] == []
    assert alice.call("/api/files")[1]["files"][0]["name"] == "mine.txt"
    status, error = bob.call(f"/api/inspect?path={stored['path']}")
    assert status == 404
    status, error = bob.call(
        "/api/inspect?path=../" + alice.cookie.split("=")[1] + "/uploads/mine.txt"
    )
    assert status == 403 and "workspace" in error["error"]["message"]
    assert str(public.workspace.root) not in json.dumps(error)

    stranger = Visitor(public)
    status, error = stranger.call("/api/info")
    assert status == 401 and error["error"]["code"] == "session_expired"
    info = alice.call("/api/info")[1]
    assert info["public"] is True and str(public.workspace.root) not in info["root"]


def test_visitors_share_the_parsed_library_and_its_index(public: UIServer) -> None:
    alice = Visitor(public)
    alice.page()
    entries = {o["slug"]: o for o in alice.call("/api/library")[1]["ontologies"]}
    assert entries["toy"]["kind"] == "hosted" and entries["toy"]["read_only"]
    status, found = alice.call(
        "/api/quickmap", {"terms": "heart attack", "target": {"libraries": ["toy"]}}
    )
    assert status == 200 and found["indexes"] == {"bm25": "opened"}  # built with the image
    assert found["rows"][0]["candidates"][0]["id"] == "TOY:0001"
    assert alice.call("/api/library/delete", {"slug": "toy"})[0] == 422


def test_public_refusals(public: UIServer, served: str) -> None:
    alice = Visitor(public)
    alice.page()
    status, error = alice.call("/api/library/download", {"url": f"{served}/toy.obo"})
    assert status == 403 and error["error"]["code"] == "not_allowed_here"
    assert alice.call("/api/library/catalog")[1]["allow_download"] is False

    status, _ = alice.call("/api/credentials", {"env": "OPENAI_API_KEY", "value": "sk-visitor"})
    assert status == 200 and os.environ.get("OPENAI_API_KEY") != "sk-visitor"
    bob = Visitor(public)
    bob.page()
    keys = {c["env"]: c["set"] for c in bob.call("/api/credentials")[1]["credentials"]}
    assert keys["OPENAI_API_KEY"] is False

    project = alice.call(
        "/api/projects",
        {
            "source": {"text": "heart attack"},
            "target": {"libraries": ["toy"]},
            "model": {
                "preset": "custom",
                "model": "m",
                "base_url": "http://169.254.169.254/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
    )[1]
    status, error = alice.call(
        "/api/tasks",
        {
            "job": project["job_path"],
            "out": project["suggested_out"],
            "model": "endpoint",
            "max_calls": 3,
        },
    )
    assert status == 403 and "https" in error["error"]["message"]


def test_runs_are_bounded_across_visitors(public: UIServer) -> None:
    alice, bob = Visitor(public), Visitor(public)
    alice.page()
    bob.page()
    project = alice.call(
        "/api/projects", {"source": {"text": "heart attack"}, "target": {"libraries": ["toy"]}}
    )[1]
    alice_ws = public.sessions.get(alice.cookie.split("=")[1])  # type: ignore[union-attr]
    assert alice_ws is not None
    release = threading.Event()

    from xwalk import ops
    from xwalk.ui.api import Task

    async def blocked() -> ops.OpResult:
        import asyncio

        while not release.is_set():
            await asyncio.sleep(0.01)
        return ops.OpResult("match")

    held = Task(
        id="held",
        kind="match",
        job="j",
        out="o",
        out_path=str(alice_ws.root / "o"),
        model="offline",
        max_calls=None,
        limit=None,
    )
    alice_ws.tasks.start(held, "match", blocked)
    try:
        status, error = alice.call(
            "/api/tasks", {"job": project["job_path"], "out": project["suggested_out"]}
        )
        assert status == 429 and error["error"]["code"] == "busy"
        bob_project = bob.call(
            "/api/projects", {"source": {"text": "seizures"}, "target": {"libraries": ["toy"]}}
        )[1]
        status, error = bob.call(
            "/api/tasks", {"job": bob_project["job_path"], "out": bob_project["suggested_out"]}
        )
        assert status == 429  # max_tasks=1 across the site
    finally:
        release.set()
    alice_ws.tasks.wait("held")


def test_health_and_framing(tmp_path: Path) -> None:
    server = make_server(
        str(tmp_path), port=0, public=True, frame_ancestors="https://huggingface.co"
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(server.url + "healthz", timeout=10) as response:
            assert response.read() == b"ok"
        with urllib.request.urlopen(server.url, timeout=10) as response:
            headers = dict(response.headers)
        assert "frame-ancestors https://huggingface.co" in headers["Content-Security-Policy"]
        assert "X-Frame-Options" not in headers
        assert "SameSite=None; Secure" in headers["Set-Cookie"]
    finally:
        server.shutdown()
        server.server_close()


def test_local_mode_is_unchanged(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    assert ws.public is False
    result = dispatch(ws, "GET", "/api/info", {})
    assert isinstance(result, dict) and result["root"] == str(tmp_path.resolve())
    with pytest.raises(ApiError):
        ws.path("../elsewhere")

"""`xwalk ui` at /app: the React workbench's page and the read routes it relies on.

The page itself is built from `web/` and committed into the package; these tests pin the
server side: what /app serves (token, policy, assets), the run fields the Runs screen
reads, source context on mapping rows, and the review overlay read route.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from tests.test_ui import _request
from xwalk import ops
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.ui import make_server
from xwalk.ui.api import ApiError, Workspace, dispatch
from xwalk.ui.server import TOKEN_HEADER, UIServer

APP = resources.files("xwalk.ui") / "static" / "app"


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root, max_calls_cap=50)


@pytest.fixture
def demo(ws: Workspace) -> str:
    envelope = ws.init({"dest": "demo"})
    assert envelope["status"] == "ok"
    return str(envelope["data"]["job_path"])


def _uncertain_model(request: LLMRequest) -> str:
    """First candidate, confidence 0.5: between review_floor and accept_at."""
    return json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.5, "decision": "support", "queries": []}
    )


@pytest.fixture
def review_run(ws: Workspace, demo: str) -> str:
    ops.run(ws.root / demo, ws.root / "rev", llm=FakeLLM(handler=_uncertain_model))
    ws._remember_job(ws.root / "rev", ws.root / demo)
    return "rev"


def _apply(ws: Workspace, demo: str, run: str, source_id: str, **decision: str) -> None:
    rows = {r["source_id"]: r for r in ws.review_rows({"dir": run})["data"]["rows"]}
    envelope = ws.review_apply(
        {
            "dir": run,
            "job": demo,
            "reviewer": decision.pop("reviewer", "jan"),
            "decisions": [{"result_key": rows[source_id]["result_key"], **decision}],
        }
    )
    assert envelope["status"] == "ok", envelope["errors"]


# --- read routes ----------------------------------------------------------------------


def test_workspace_runs_carry_usage_and_fingerprint(ws: Workspace, review_run: str) -> None:
    run = ws.workspace({})["runs"][0]
    assert run["dir"] == "rev"
    assert run["run_fingerprint"] == ws.run({"dir": "rev"})["inspect"]["run"]["run_fingerprint"]
    assert run["usage"]["calls"] > 0
    # Not started from the page, so the UI does not know which model answered.
    assert run["answered_by"] is None


def test_run_reports_targets_shared_by_several_sources(ws: Workspace, review_run: str) -> None:
    # The quickstart's s1 and s2 both name glucose.
    assert ws.run({"dir": review_run})["duplicate_targets"] == {"CHEBI:17234": ["s1", "s2"]}


def test_mapping_rows_carry_the_rendered_source_context(ws: Workspace, review_run: str) -> None:
    rows = dispatch(ws, "GET", "/api/mapping", {"dir": review_run})["data"]["rows"]
    first = rows[0]
    assert first["source_text"] == "glucose"
    # The quickstart's context template: "{{ context_left }} [{{ mention }}] {{ ... }}"
    assert first["source_context"] == "blood [glucose] levels were elevated"


def test_reviews_route_reports_history_states_and_final_answers(
    ws: Workspace, demo: str, review_run: str
) -> None:
    empty = dispatch(ws, "GET", "/api/reviews", {"dir": review_run})
    assert empty["status"] == "ok"
    assert empty["data"] == {"history": [], "final": {}}

    _apply(ws, demo, review_run, "s1", decision="accept")
    _apply(ws, demo, review_run, "s2", decision="defer")
    _apply(
        ws,
        demo,
        review_run,
        "s2",
        decision="replace",
        corrected_target_id="CHEBI:17992",
        review_note="checked",
        reviewer="ana",
    )
    data = dispatch(ws, "GET", "/api/reviews", {"dir": review_run})["data"]

    history = [(h["source_id"], h["decision"], h["state"]) for h in data["history"]]
    assert history == [
        ("s1", "accept", "applied"),
        ("s2", "defer", "superseded"),
        ("s2", "replace", "applied"),
    ]
    latest = data["history"][-1]
    assert latest["reviewer"] == "ana" and latest["review_note"] == "checked"
    assert latest["corrected_target_id"] == "CHEBI:17992" and latest["result_key"]

    final = data["final"]
    assert set(final) == {latest["result_key"], data["history"][0]["result_key"]}
    assert final[latest["result_key"]] == {
        "source_id": "s2",
        "final_matched_id": "CHEBI:17992",
        "final_status": "matched",
    }


def test_reviews_route_refuses_a_directory_that_is_not_a_run(ws: Workspace, demo: str) -> None:
    with pytest.raises(ApiError) as caught:
        dispatch(ws, "GET", "/api/reviews", {"dir": "demo"})
    assert caught.value.code == "run_not_found"


# --- the page -------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path) -> Iterator[UIServer]:
    root = tmp_path / "ws"
    root.mkdir()
    instance = make_server(str(root), port=0, token="test-token")
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()


def _assets(html: str) -> list[str]:
    return re.findall(r'(?:src|href)="(/app/assets/[^"]+)"', html)


def test_the_built_workbench_is_packaged() -> None:
    html = (APP / "index.html").read_text(encoding="utf-8")
    assert "__XWALK_TOKEN__" in html
    assert _assets(html), "web/ has not been built into src/xwalk/ui/static/app"


@pytest.mark.parametrize("path", ["/app", "/app/", "/app/index.html"])
def test_app_page_carries_the_token_and_the_strict_policy(server: UIServer, path: str) -> None:
    status, headers, body = _request(server, path)
    assert status == 200
    assert b'content="test-token"' in body and b"__XWALK_TOKEN__" not in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY"


def test_app_assets_are_served_with_their_types(server: UIServer) -> None:
    html = _request(server, "/app/")[2].decode("utf-8")
    types: dict[str, Any] = {}
    for asset in _assets(html):
        status, headers, _ = _request(server, asset)
        assert status == 200, asset
        types[asset.rsplit(".", 1)[-1]] = headers["Content-Type"]
    assert types["js"].startswith("text/javascript")
    assert types["css"].startswith("text/css")
    fonts = list((APP / "assets").iterdir())
    woff2 = next(f.name for f in fonts if f.name.endswith(".woff2"))
    status, headers, _ = _request(server, f"/app/assets/{woff2}")
    assert status == 200 and headers["Content-Type"] == "font/woff2"


@pytest.mark.parametrize(
    "path",
    [
        "/app/assets/missing.js",
        # Real files with served extensions, reached by walking out of assets/.
        "/app/assets/../../app.js",
        "/app/assets/../../app.css",
        "/app/assets/..\\..\\app.js",
        "/app/assets/.hidden.js",
        "/app/assets/../index.html",
        "/app/assets/%2e%2e/index.html",
        "/app/../api.py",
        "/app/assets/",
        "/app/assets/x.py",
    ],
)
def test_app_refuses_anything_but_its_own_assets(server: UIServer, path: str) -> None:
    assert _request(server, path)[0] == 404


def test_app_api_calls_use_the_same_token(server: UIServer) -> None:
    status, _, body = _request(server, "/api/info", headers={TOKEN_HEADER: "test-token"})
    assert status == 200 and json.loads(body)["version"]

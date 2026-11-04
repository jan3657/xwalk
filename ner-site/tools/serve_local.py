"""Serve the NER demo page locally: ``public/`` plus ``/api/*``.

Usage (from the repository root, with the venv active)::

    python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures
    python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures --scenario degraded
    python ner-site/tools/serve_local.py --offline
    python ner-site/tools/serve_local.py            # live: needs OPENROUTER_API_KEY

``--canned DIR`` answers from JSON fixtures and never imports ``ner_site``:
``GET /api/health`` -> ``health[-SCENARIO].json``; ``POST /api/ner`` ->
``ner[-SCENARIO].json``, or ``error.json`` with HTTP 400 when the posted text contains
``ERROR`` (to exercise the error UI). Without ``--canned``, requests go through
``ner_site.http.handle`` exactly as the Vercel functions do; ``--offline`` sets
``NER_SITE_OFFLINE=1`` so the stand-in extractor replaces the LLM.

The response headers from ``ner-site/vercel.json`` are added to every response, so the
page runs under the same Content-Security-Policy locally as on Vercel.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
PUBLIC = PROJECT / "public"
VERCEL_JSON = PROJECT / "vercel.json"
MAX_READ_BYTES = 1 << 20  # the service itself rejects bodies over 64 KB

Response = tuple[int, dict[str, str], bytes]
Backend = Callable[[str, str, dict[str, str], bytes], Response]
HeaderRules = list[tuple[re.Pattern[str], list[tuple[str, str]]]]


def _json_response(status: int, data: object) -> Response:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return status, {"Content-Type": "application/json; charset=utf-8"}, body


def _envelope(status: int, code: str, message: str) -> Response:
    return _json_response(status, {"ok": False, "error": {"code": code, "message": message}})


def load_header_rules(path: Path = VERCEL_JSON) -> HeaderRules:
    """``headers`` from vercel.json. Sources are used as regular expressions, which is
    exact for the ``/(.*)``-style sources this project uses (not full path-to-regexp)."""
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    rules: HeaderRules = []
    for rule in config.get("headers", []):
        pairs = [(str(h["key"]), str(h["value"])) for h in rule["headers"]]
        rules.append((re.compile(str(rule["source"])), pairs))
    return rules


def canned_backend(directory: Path, scenario: str) -> Backend:
    suffix = f"-{scenario}" if scenario else ""
    names = (f"health{suffix}.json", f"ner{suffix}.json", "error.json")
    fixtures: dict[str, object] = {}
    for name in names:
        file = directory / name
        if not file.is_file():
            raise SystemExit(f"serve_local: fixture {file} not found")
        fixtures[name] = json.loads(file.read_text(encoding="utf-8"))
    health, ner, error = (fixtures[n] for n in names)

    def backend(method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        if path == "/api/health":
            return _json_response(200, health) if method == "GET" else _method_not_allowed()
        if path != "/api/ner":
            return _envelope(404, "not_found", f"no such endpoint: {path}")
        if method != "POST":
            return _method_not_allowed()
        if headers.get("x-requested-with") != "xwalk-ner":
            return _envelope(400, "bad_header", "missing X-Requested-With: xwalk-ner")
        try:
            request = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _envelope(400, "bad_json", "the request body is not valid JSON")
        text = request.get("text") if isinstance(request, dict) else None
        if isinstance(text, str) and "ERROR" in text:
            return _json_response(400, error)
        return _json_response(200, ner)

    return backend


def _method_not_allowed() -> Response:
    return _envelope(405, "method_not_allowed", "method not allowed")


def service_backend(*, offline: bool) -> Backend:
    """Route through ``ner_site.http.handle`` (WS-D). Imported here, not at module load."""
    if offline:
        os.environ["NER_SITE_OFFLINE"] = "1"
    sys.path.insert(0, str(PROJECT))
    try:
        http_mod = importlib.import_module("ner_site.http")
        service_mod = importlib.import_module("ner_site.service")
    except ModuleNotFoundError as exc:
        if exc.name in {"ner_site.http", "ner_site.service"}:
            raise SystemExit(
                f"serve_local: {exc.name} is not available yet (it is written by WS-D). "
                "Use --canned ner-site/tools/fixtures to serve fixture responses instead."
            ) from exc
        raise
    handle: Callable[..., tuple[int, dict[str, str], bytes]] = http_mod.handle
    settings = service_mod.Settings.from_env(os.environ)

    def backend(method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        status, out_headers, out_body = handle(method, path, headers, body, settings=settings)
        return status, dict(out_headers), out_body

    return backend


def make_handler(
    backend: Backend,
    rules: HeaderRules,
    delay: float,
) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        server_version = "xwalk-ner-local"
        sys_version = ""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._sent: set[str] = set()
            super().__init__(*args, directory=str(PUBLIC), **kwargs)

        def send_header(self, keyword: str, value: str) -> None:
            self._sent.add(keyword.lower())
            super().send_header(keyword, value)

        def end_headers(self) -> None:
            path = urlsplit(self.path).path
            for pattern, pairs in rules:
                if pattern.fullmatch(path):
                    for key, value in pairs:
                        if key.lower() not in self._sent:
                            self.send_header(key, value)
            self._sent = set()
            super().end_headers()

        def _api(self, method: str) -> None:
            raw_length = (self.headers.get("Content-Length") or "0").strip()
            # ASCII digits only, as the Vercel shim: "²" passes str.isdigit(), not int().
            length = int(raw_length) if re.fullmatch(r"[0-9]{1,18}", raw_length) else -1
            if length < 0:
                status, headers, body = _envelope(400, "bad_header", "bad Content-Length")
                self.close_connection = True
            elif length > MAX_READ_BYTES:
                status, headers, body = _envelope(413, "body_too_large", "request body too large")
                self.close_connection = True
            else:
                raw = self.rfile.read(length) if length > 0 else b""
                lowered = {k.lower(): v for k, v in self.headers.items()}
                if delay > 0 and method == "POST":
                    time.sleep(delay)
                try:
                    status, headers, body = backend(method, urlsplit(self.path).path, lowered, raw)
                except Exception as exc:  # noqa: BLE001 - a local dev server reports and goes on
                    self.log_error("backend raised %r", exc)
                    status, headers, body = _envelope(500, "internal", "internal error")
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() != "content-length":
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(body)

        def _is_api(self) -> bool:
            path = urlsplit(self.path).path
            return path == "/api" or path.startswith("/api/")

        def do_GET(self) -> None:
            if self._is_api():
                self._api("GET")
            else:
                super().do_GET()

        def do_HEAD(self) -> None:
            if self._is_api():
                self._api("HEAD")
            else:
                super().do_HEAD()

        def do_POST(self) -> None:
            if self._is_api():
                self._api("POST")
            else:
                self.send_error(405)

        def do_OPTIONS(self) -> None:
            if self._is_api():
                self._api("OPTIONS")
            else:
                self.send_error(405)

        def do_PUT(self) -> None:
            if self._is_api():
                self._api("PUT")
            else:
                self.send_error(405)

        def do_DELETE(self) -> None:
            if self._is_api():
                self._api("DELETE")
            else:
                self.send_error(405)

    Handler.extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }
    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--offline", action="store_true", help="set NER_SITE_OFFLINE=1")
    parser.add_argument("--canned", type=Path, metavar="DIR", help="serve fixture JSON from DIR")
    parser.add_argument(
        "--scenario",
        default="",
        help="with --canned: use health-NAME.json and ner-NAME.json (e.g. 'degraded')",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="seconds to wait before answering a POST"
    )
    args = parser.parse_args(argv)

    if args.canned is not None:
        if args.offline:
            parser.error("--offline has no effect with --canned")
        backend = canned_backend(args.canned, args.scenario)
        mode = f"canned fixtures from {args.canned}" + (
            f" (scenario {args.scenario})" if args.scenario else ""
        )
    else:
        if args.scenario:
            parser.error("--scenario needs --canned")
        backend = service_backend(offline=args.offline)
        mode = "ner_site.http" + (" (offline)" if args.offline else " (live)")

    handler = make_handler(backend, load_header_rules(), args.delay)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as exc:
        print(f"serve_local: cannot bind 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        return 1
    print(f"serving http://127.0.0.1:{args.port}/ ({mode}); Ctrl+C stops", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

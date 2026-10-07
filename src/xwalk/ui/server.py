"""`xwalk ui`: a local web app over `xwalk.ops`, served by the standard library.

    xwalk ui                         # serve the current directory on http://127.0.0.1:8765
    xwalk ui --root ~/mappings --port 9000 --no-browser
    xwalk ui --offline-only          # never call the job's endpoint, whatever is clicked

No dependency beyond the base install: `http.server` serves a static single-page app
(`xwalk/ui/static/`) and the JSON API in `xwalk.ui.api`.

It is a local tool, not a multi-user service. The defences are aimed at other web pages
open in the same browser, not at other users of the machine:

- it listens on 127.0.0.1 unless `--host` says otherwise;
- every API request must carry the per-process token embedded in the page
  (`X-Xwalk-Token`), which a page from another origin cannot read; the custom header also
  forces a CORS preflight, which this server never grants;
- the `Host` header must name the address the server listens on (DNS rebinding);
- every path a request names must lie inside the workspace root.
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from xwalk.ui.api import (
    DEFAULT_MAX_CALLS_CAP,
    DEFAULT_MAX_UPLOAD_MB,
    ApiError,
    Download,
    Workspace,
    dispatch,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY = 4 * 1024 * 1024
TOKEN_HEADER = "X-Xwalk-Token"
TOKEN_PLACEHOLDER = "__XWALK_TOKEN__"

_STATIC_TYPES = {
    "app.js": "text/javascript; charset=utf-8",
    "workbench.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "favicon.svg": "image/svg+xml",
}
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def _static(name: str) -> bytes:
    return (resources.files("xwalk.ui") / "static" / name).read_bytes()


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        workspace: Workspace,
        *,
        token: str | None = None,
        quiet: bool = True,
    ) -> None:
        super().__init__(address, _Handler)
        self.workspace = workspace
        self.token = token or secrets.token_urlsafe(24)
        self.quiet = quiet

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        shown = host if isinstance(host, str) else host.decode()
        if ":" in shown:
            shown = f"[{shown}]"
        return f"http://{shown}:{port}/"

    def allowed_hosts(self) -> set[str] | None:
        """Host header values to accept; None (any) when listening on every interface."""
        host = str(self.server_address[0])
        if host in ("0.0.0.0", "::"):
            return None
        port = self.server_address[1]
        names = {host} | (_LOOPBACK_NAMES if host in _LOOPBACK_NAMES else set())
        allowed = set()
        for name in names:
            shown = f"[{name}]" if ":" in name else name
            allowed |= {shown, f"{shown}:{port}"}
        return allowed


class _Handler(BaseHTTPRequestHandler):
    server: UIServer
    server_version = "xwalk-ui"
    protocol_version = "HTTP/1.1"

    # --- plumbing ------------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        if not self.server.quiet:
            sys.stderr.write(f"xwalk ui: {format % args}\n")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, error: ApiError) -> None:
        self._json(error.status, error.to_dict())

    def _host_ok(self) -> bool:
        allowed = self.server.allowed_hosts()
        return allowed is None or (self.headers.get("Host") or "") in allowed

    # --- methods -------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        self._handle("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def _handle(self, method: str) -> None:
        if not self._host_ok():
            self._error(ApiError(421, "bad_host", "unexpected Host header"))
            return
        parts = urlsplit(self.path)
        path = parts.path
        if path == "/api/upload":
            # A refused upload leaves its body unread; never reuse the connection.
            self.close_connection = True
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        try:
            if path.startswith("/api/"):
                self._api(method, path, query)
            elif method == "GET":
                self._page(path)
            else:
                raise ApiError(405, "method_not_allowed", f"{method} {path}")
        except ApiError as error:
            self._error(error)
        except Exception as exc:  # never a traceback to the browser
            self._error(ApiError(500, "server_error", f"{type(exc).__name__}: {exc}"))

    def _page(self, path: str) -> None:
        if path in ("/", "/index.html"):
            html = _static("index.html").decode("utf-8")
            body = html.replace(TOKEN_PLACEHOLDER, self.server.token).encode("utf-8")
            self._send(
                200,
                body,
                "text/html; charset=utf-8",
                extra={"Content-Security-Policy": _CSP, "X-Frame-Options": "DENY"},
            )
            return
        name = path.removeprefix("/static/")
        if path.startswith("/static/") and name in _STATIC_TYPES:
            self._send(200, _static(name), _STATIC_TYPES[name])
            return
        raise ApiError(404, "not_found", f"no page {path}")

    def _api(self, method: str, path: str, query: dict[str, str]) -> None:
        from xwalk.ui.workbench import DOWNLOAD_ROUTES

        token = self.headers.get(TOKEN_HEADER)
        if token is None and method == "GET" and path in DOWNLOAD_ROUTES:
            # A download is a plain link, which cannot carry a header.
            token = query.pop("token", None)
        if token is None or not secrets.compare_digest(token, self.server.token):
            raise ApiError(403, "bad_token", "missing or wrong API token; reload the page")
        params: dict[str, Any] = dict(query)
        if method == "POST" and path == "/api/upload":
            # The file is the body, streamed to disk by the handler, never held whole.
            params["_body"] = self.rfile
            params["_length"] = self._length()
        elif method == "POST":
            params.update(self._body())
        result = dispatch(self.server.workspace, method, path, params)
        if isinstance(result, Download):
            disposition = f"attachment; filename*=UTF-8''{quote(result.filename)}"
            self._send(
                200, result.body, result.content_type, extra={"Content-Disposition": disposition}
            )
        else:
            self._json(HTTPStatus.OK, result)

    def _length(self) -> int:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "bad_request", "invalid Content-Length") from None
        if length < 0:
            raise ApiError(400, "bad_request", "invalid Content-Length")
        return length

    def _body(self) -> dict[str, Any]:
        length = self._length()
        if length > MAX_BODY:
            raise ApiError(413, "too_large", "request body too large")
        if not length:
            return {}
        if "json" not in (self.headers.get("Content-Type") or ""):
            raise ApiError(415, "unsupported_media_type", "send application/json")
        try:
            data = json.loads(self.rfile.read(length))
        except ValueError:
            raise ApiError(400, "bad_request", "the body is not valid JSON") from None
        if not isinstance(data, dict):
            raise ApiError(400, "bad_request", "the body must be a JSON object")
        return data


def make_server(
    root: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_calls_cap: int = DEFAULT_MAX_CALLS_CAP,
    offline_only: bool = False,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    token: str | None = None,
    quiet: bool = True,
) -> UIServer:
    workspace = Workspace(
        root,
        max_calls_cap=max_calls_cap,
        allow_endpoint=not offline_only,
        max_upload_mb=max_upload_mb,
    )
    return UIServer((host, port), workspace, token=token, quiet=quiet)


def serve(
    root: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_calls_cap: int = DEFAULT_MAX_CALLS_CAP,
    offline_only: bool = False,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    open_browser: bool = True,
    verbose: bool = False,
) -> None:
    """Serve until interrupted (`xwalk ui`). Status lines go to stderr."""
    server = make_server(
        root,
        host=host,
        port=port,
        max_calls_cap=max_calls_cap,
        offline_only=offline_only,
        max_upload_mb=max_upload_mb,
        quiet=not verbose,
    )
    print(f"xwalk ui: serving {server.workspace.root}", file=sys.stderr)
    print(f"xwalk ui: open {server.url}  (Ctrl-C to stop)", file=sys.stderr)
    if host not in _LOOPBACK_NAMES:
        print(
            "xwalk ui: warning: listening beyond this machine; anyone who can reach "
            f"{server.url} can read and write the workspace",
            file=sys.stderr,
        )
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(server.url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "UIServer", "make_server", "serve"]

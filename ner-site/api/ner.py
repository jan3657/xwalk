"""POST /api/ner on Vercel: a shim over `ner_site.http` (docs/ner-site/PLAN.md section 2).

Vercel loads the top-level class named `handler` (a `BaseHTTPRequestHandler`
subclass) from each `api/*.py` file. `ner_site` is imported on the first request,
after the project root is put on `sys.path`; `ner_site.http.serve` reads the body
(refusing one over the cap before reading it), calls `handle`, and writes the status,
headers and body. Every route goes through `handle`, so a request that reaches the
wrong file still gets the right 404 or 405.
"""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel looks up this exact name
    def _serve(self) -> None:
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from ner_site.http import serve

        serve(self)

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _serve

    def log_message(self, format: str, *args: object) -> None:
        """Silenced: `ner_site.http` writes the one log line per request."""

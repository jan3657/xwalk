"""Public mode: `xwalk ui --public`, for running the explorer as a website.

The local UI trusts its one user with the workspace and the machine. A public site does
not, so public mode changes four things:

- **Sessions.** Every visitor gets a private workspace (`<root>/sessions/<id>/`) named by
  a random cookie. Nobody can list, open or download another visitor's files, and a
  session left alone for `session_ttl` is deleted with everything in it.
- **A shared, read-only library.** The operator parses ontologies once
  (`scripts/build_ontology_library.py`, usually while building the image); every visitor
  maps onto them, and their search indexes are built once and shared.
- **Visitors' own model keys.** A key typed on the Settings page is kept in that
  session's memory and handed to the model client directly. It never enters the
  process environment (which every visitor shares) and is never written to disk.
- **No requests to places the visitor picks inside the host's network.** The server
  never downloads a URL a visitor gives, and a model endpoint must be an `https` URL
  whose host resolves only to public addresses (no localhost, private ranges or the
  cloud metadata address).

Plus bounds: one running task per visitor, `max_tasks` across all of them.
"""

from __future__ import annotations

import ipaddress
import re
import secrets
import shutil
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from xwalk.config import LLMSpec
from xwalk.llm.base import LLMClient

SESSION_COOKIE = "xwalk_session"
DEFAULT_SESSION_TTL = 24 * 3600
DEFAULT_MAX_SESSIONS = 500
DEFAULT_MAX_TASKS = 4
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
_SWEEP_EVERY = 600.0


class PublicRefusal(ValueError):
    """Something public mode does not allow. The message says why, for the visitor."""


def check_public_url(url: str) -> None:
    """Refuse a URL a public server must not call: not https, or a host that resolves
    to a loopback, private, link-local or otherwise non-public address."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise PublicRefusal("on the hosted version a model endpoint must be an https:// URL")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError:
        raise PublicRefusal(f"cannot resolve {parts.hostname}") from None
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise PublicRefusal(
                f"{parts.hostname} resolves to a non-public address; the hosted version "
                "only calls public endpoints"
            )


def endpoint_client(spec: LLMSpec, keys: Mapping[str, str]) -> LLMClient:
    """The job's model client with the visitor's key, after the public-URL check."""
    from xwalk.llm.openai_compat import OpenAICompatClient

    if spec.kind != "openai_compat":
        raise PublicRefusal("the hosted version only calls OpenAI-compatible endpoints")
    check_public_url(spec.base_url or "")
    key = None
    if spec.api_key_env:
        key = keys.get(spec.api_key_env)
        if not key:
            raise PublicRefusal(
                f"set {spec.api_key_env} on the Settings page first (it stays in this "
                "browser session only)"
            )
    return OpenAICompatClient(
        base_url=spec.base_url or "",
        model=spec.model,
        api_key=key,
        profile=spec.profile,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        seed=spec.seed,
    )


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def valid_session_id(value: str | None) -> bool:
    return bool(value) and _SESSION_ID.match(value or "") is not None


class Sessions:
    """Visitor workspaces under `root`, created on first visit, deleted when idle."""

    def __init__(
        self,
        root: Path,
        *,
        make_workspace: Any,
        ttl: float = DEFAULT_SESSION_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_tasks: int = DEFAULT_MAX_TASKS,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._make = make_workspace
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.max_tasks = max_tasks
        self._open: dict[str, Any] = {}
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def create(self) -> str:
        self.sweep()
        with self._lock:
            if len(self._open) >= self.max_sessions:
                raise PublicRefusal("the site is busy; try again later")
            session = new_session_id()
            (self.root / session).mkdir()
            self._open[session] = self._make(self.root / session)
            self._seen[session] = time.time()
            return session

    def get(self, session: str | None) -> Any | None:
        """The visitor's workspace, or None for an unknown or expired session."""
        if not valid_session_id(session):
            return None
        assert session is not None
        with self._lock:
            workspace = self._open.get(session)
            if workspace is None:
                return None
            self._seen[session] = time.time()
            return workspace

    def running_tasks(self) -> int:
        with self._lock:
            workspaces = list(self._open.values())
        return sum(1 for ws in workspaces for task in ws.tasks.all() if task.state == "running")

    def sweep(self, *, now: float | None = None, force: bool = False) -> list[str]:
        """Delete sessions idle for longer than `ttl` (and stray directories)."""
        now = time.time() if now is None else now
        if not force and now - self._last_sweep < _SWEEP_EVERY:
            return []
        self._last_sweep = now
        removed: list[str] = []
        with self._lock:
            for session, seen in list(self._seen.items()):
                ws = self._open[session]
                busy = any(t.state == "running" for t in ws.tasks.all())
                if now - seen > self.ttl and not busy:
                    del self._open[session], self._seen[session]
                    shutil.rmtree(self.root / session, ignore_errors=True)
                    removed.append(session)
            known = set(self._open)
        for child in self.root.iterdir():
            # A directory left by an earlier process: its sessions cannot be resumed.
            if child.is_dir() and child.name not in known:
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
        return removed


def library_dirs(values: Sequence[str]) -> list[Path]:
    dirs = [Path(v).resolve() for v in values]
    for directory in dirs:
        if not directory.is_dir():
            raise ValueError(f"library directory {directory} does not exist")
    return dirs


__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_TASKS",
    "DEFAULT_SESSION_TTL",
    "SESSION_COOKIE",
    "PublicRefusal",
    "Sessions",
    "check_public_url",
    "endpoint_client",
    "library_dirs",
    "valid_session_id",
]

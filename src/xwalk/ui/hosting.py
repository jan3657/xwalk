"""Public mode: `xwalk ui --public`, for running the explorer as a website.

The local UI trusts its one user with the workspace and the machine. A public site does
not, so public mode changes four things:

- **Sessions.** Every visitor gets a private workspace (`<root>/sessions/<key>/`) named by
  a random cookie. Nobody can list, open or download another visitor's files, and a
  session left alone for `session_ttl` is deleted with everything in it. A page view
  only hands out a signed key; the workspace is made on the first API request that
  carries it, at most `new_per_client` times per client address in `RATE_WINDOW`, and
  one nobody has put anything in is dropped after `empty_ttl` (or sooner, when the pool
  is full).
- **A shared, read-only library.** The operator parses ontologies once
  (`scripts/build_ontology_library.py`, usually while building the image); every visitor
  maps onto them, and their search indexes are built once and shared.
- **Visitors' own model keys.** A key typed on the Settings page is kept in that
  session's memory and handed to the model client directly. It never enters the
  process environment (which every visitor shares) and is never written to disk.
- **No requests to places the visitor picks inside the host's network.** The server
  never downloads a URL a visitor gives, and a model endpoint must be an `https` URL
  whose host resolves only to public addresses (no localhost, private ranges or the
  cloud metadata address). The name is resolved once and the client connects to the
  addresses that were checked (`xwalk.ui.egress`), so DNS rebinding cannot swap in a
  private address between the check and the connection.

Plus bounds: one running task per visitor, `max_tasks` across all of them.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import shutil
import threading
import time
from collections import deque
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
# A session nobody has put anything in (no file, task or key) goes after this long idle.
DEFAULT_EMPTY_TTL = 600.0
# New sessions one client address may open per RATE_WINDOW seconds (0: no limit).
DEFAULT_NEW_PER_CLIENT = 20
RATE_WINDOW = 600.0
# How long an issued key may wait for its first API request before it is refused.
_KEY_WINDOW = 3600.0
# A key: random id, issue time (hex seconds), HMAC of both under the process secret.
_SESSION_KEY = re.compile(r"^[A-Za-z0-9_-]{43}\.[0-9a-f]{1,12}\.[A-Za-z0-9_-]{22}$")
_SWEEP_EVERY = 60.0


class PublicRefusal(ValueError):
    """Something public mode does not allow. The message says why, for the visitor."""


class SessionRateLimited(PublicRefusal):
    """One client address asked for too many new sessions in a short time."""


def check_public_url(url: str) -> tuple[str, list[str]]:
    """Refuse a URL a public server must not call: not https, or a host that resolves
    to a loopback, private, link-local or otherwise non-public address.

    Returns the host as it goes on the wire and the addresses it resolved to, every one
    of them public: the caller connects to these, not to the name."""
    from xwalk.ui import egress

    parts = urlsplit(url)
    host = egress.wire_host(url) if parts.scheme == "https" else ""
    if not host:
        raise PublicRefusal("on the hosted version a model endpoint must be an https:// URL")
    try:
        addresses = egress.resolve(host, parts.port or 443)
    except (OSError, ValueError):
        raise PublicRefusal(f"cannot resolve {parts.hostname}") from None
    if not addresses or not all(egress.is_public(address) for address in addresses):
        raise PublicRefusal(
            f"{parts.hostname} resolves to a non-public address; the hosted version "
            "only calls public endpoints"
        )
    return host, addresses


def endpoint_client(spec: LLMSpec, keys: Mapping[str, str]) -> LLMClient:
    """The job's model client with the visitor's key, after the public-URL check."""
    from xwalk.llm.openai_compat import OpenAICompatClient
    from xwalk.ui.egress import PinnedAsyncTransport

    if spec.kind != "openai_compat":
        raise PublicRefusal("the hosted version only calls OpenAI-compatible endpoints")
    host, addresses = check_public_url(spec.base_url or "")
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
        # Connect only to the addresses just checked; no proxy from the environment
        # (it would resolve the name again), and redirects stay off (httpx default).
        transport=PinnedAsyncTransport(host, addresses),
        trust_env=False,
    )


def valid_session_id(value: str | None) -> bool:
    """Whether `value` has the shape of a session key (not whether it is genuine)."""
    return bool(value) and _SESSION_KEY.match(value or "") is not None


def _in_use(workspace: Any, directory: Path) -> bool:
    """Whether a visitor has put anything in this session worth keeping."""
    if workspace.tasks.all() or getattr(workspace, "session_keys", None):
        return True
    auth = getattr(workspace, "chatgpt_auth", None)
    if auth is not None and auth.holds_state():
        return True
    try:
        return any(directory.iterdir())
    except OSError:
        return False


class Sessions:
    """Visitor workspaces under `root`, made on first use, deleted when idle.

    `issue()` (a page view) only signs a fresh key; nothing is stored. `open()` (an API
    request) turns a genuine, recent key into a workspace, once, rate limited per client
    address. A crawler that fetches the page and leaves costs nothing.
    """

    def __init__(
        self,
        root: Path,
        *,
        make_workspace: Any,
        ttl: float = DEFAULT_SESSION_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_tasks: int = DEFAULT_MAX_TASKS,
        empty_ttl: float = DEFAULT_EMPTY_TTL,
        new_per_client: int = DEFAULT_NEW_PER_CLIENT,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._make = make_workspace
        self.ttl = ttl
        self.empty_ttl = min(empty_ttl, ttl)
        self.max_sessions = max_sessions
        self.max_tasks = max_tasks
        self.new_per_client = new_per_client
        self._secret = secrets.token_bytes(32)
        self._open: dict[str, Any] = {}
        self._seen: dict[str, float] = {}
        self._made: dict[str, deque[float]] = {}
        # Requests in flight per session: a leased session is never evicted or swept,
        # so a request that has its workspace keeps it until it is done.
        self._leases: dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    # keys

    def _sign(self, body: str) -> str:
        digest = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest[:16]).decode("ascii").rstrip("=")

    def issue(self, *, now: float | None = None) -> str:
        """A new signed session key for a cookie. Allocates nothing."""
        issued = int(time.time() if now is None else now)
        body = f"{secrets.token_urlsafe(32)}.{issued:x}"
        return f"{body}.{self._sign(body)}"

    def genuine(self, key: str | None, *, now: float | None = None) -> bool:
        """Whether `key` was issued by this process recently enough to be opened."""
        if not valid_session_id(key):
            return False
        assert key is not None
        body, _, signature = key.rpartition(".")
        if not hmac.compare_digest(signature, self._sign(body)):
            return False
        issued = int(body.rpartition(".")[2], 16)
        now = time.time() if now is None else now
        return -60 <= now - issued <= min(_KEY_WINDOW, self.ttl)

    # sessions

    def count(self) -> int:
        with self._lock:
            return len(self._open)

    def create(self, *, client: str = "") -> str:
        """Issue a key and open its session at once (tests, scripts)."""
        key = self.issue()
        self.open(key, client=client)
        return key

    def open(self, key: str | None, *, client: str, lease: bool = False) -> Any | None:
        """The session for `key`, made now if it is new; None for a key that is not
        genuine (forged, malformed, or issued too long ago and never used).

        With `lease`, the session is also held for the caller (atomically with finding
        or making it) until `release(key)`: nothing evicts or sweeps it meanwhile.

        Raises `SessionRateLimited` when `client` has opened too many sessions lately,
        `PublicRefusal` when the pool is full of sessions in use."""
        if not valid_session_id(key):
            return None
        assert key is not None
        with self._lock:
            workspace = self._open.get(key)
            if workspace is not None:
                self._touch(key, lease)
                return workspace
        if not self.genuine(key):
            return None
        self.sweep()
        with self._lock:
            workspace = self._open.get(key)  # another request of the same page won
            if workspace is not None:
                self._touch(key, lease)
                return workspace
            now = time.time()
            self._check_rate(client, now)
            if len(self._open) >= self.max_sessions and not self._evict_one_empty():
                raise PublicRefusal("the site is busy; try again later")
            directory = self.root / key
            directory.mkdir(exist_ok=True)
            workspace = self._make(directory)
            self._open[key] = workspace
            self._touch(key, lease)
            if self.new_per_client > 0:
                self._made.setdefault(client, deque()).append(now)
            return workspace

    def _touch(self, key: str, lease: bool) -> None:
        """Mark `key` seen now, and leased if asked. Lock held."""
        self._seen[key] = time.time()
        if lease:
            self._leases[key] = self._leases.get(key, 0) + 1

    def release(self, key: str) -> None:
        """End one lease taken by `open(..., lease=True)`; idle time counts from now."""
        with self._lock:
            held = self._leases.get(key, 0) - 1
            if held > 0:
                self._leases[key] = held
            else:
                self._leases.pop(key, None)
            if key in self._open:
                self._seen[key] = time.time()

    def _check_rate(self, client: str, now: float) -> None:
        if self.new_per_client <= 0:
            return
        if len(self._made) > 10_000:  # forget quiet clients; the map stays small
            for name in [c for c, times in self._made.items() if now - times[-1] > RATE_WINDOW]:
                del self._made[name]
        times = self._made.get(client)
        if times is None:
            return
        while times and now - times[0] > RATE_WINDOW:
            times.popleft()
        if len(times) >= self.new_per_client:
            raise SessionRateLimited(
                "too many new sessions from your address; try again in a few minutes"
            )

    def _evict_one_empty(self) -> bool:
        """Drop the longest-idle session nobody has used and no request holds. Lock held."""
        for key in sorted(self._seen, key=self._seen.__getitem__):
            if key not in self._leases and not _in_use(self._open[key], self.root / key):
                del self._open[key], self._seen[key]
                shutil.rmtree(self.root / key, ignore_errors=True)
                return True
        return False

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
        """Delete sessions idle for longer than `ttl`, unused ones idle for longer than
        `empty_ttl`, and stray directories."""
        now = time.time() if now is None else now
        if not force and now - self._last_sweep < _SWEEP_EVERY:
            return []
        self._last_sweep = now
        removed: list[str] = []
        with self._lock:
            for session, seen in list(self._seen.items()):
                ws = self._open[session]
                busy = any(t.state == "running" for t in ws.tasks.all())
                idle = now - seen
                expired = idle > self.ttl or (
                    idle > self.empty_ttl and not _in_use(ws, self.root / session)
                )
                if expired and not busy and session not in self._leases:
                    del self._open[session], self._seen[session]
                    shutil.rmtree(self.root / session, ignore_errors=True)
                    removed.append(session)
            # Under the lock, so a session opened meanwhile is never taken for a stray.
            for child in self.root.iterdir():
                # A directory left by an earlier process: its sessions cannot be resumed.
                if child.is_dir() and child.name not in self._open:
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
    "DEFAULT_EMPTY_TTL",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_TASKS",
    "DEFAULT_NEW_PER_CLIENT",
    "DEFAULT_SESSION_TTL",
    "SESSION_COOKIE",
    "PublicRefusal",
    "SessionRateLimited",
    "Sessions",
    "check_public_url",
    "endpoint_client",
    "library_dirs",
    "valid_session_id",
]

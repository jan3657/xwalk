"""The UI's JSON API, independent of HTTP: a `Workspace` and its route table.

Every handler takes the request parameters (the query string of a GET, the JSON body of a
POST) and returns a JSON-able dict or a `Download`. Handlers that perform an operation
return its `OpResult.envelope()` -- the same schema 1 object `--json` prints -- also when
the operation failed (`status` is then ``error``), exactly as the MCP server does. Misuse
of the API itself (a missing parameter, a path outside the workspace) raises `ApiError`.

Confinement: every path a request names is resolved against the workspace root and
refused when it points outside it (`..`, an absolute path elsewhere, a symlink out).

Model calls: a match or cluster task runs either with an offline stand-in
(`xwalk.ui.offline`, no endpoint, no spend) or with the job's own endpoint; the latter
requires `max_calls`, at most the server's cap, enforced by the same `CallBudget` the CLI
uses. Tasks run in a background thread each and report progress for polling.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import itertools
import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xwalk import __version__, ops
from xwalk.ui import offline

DEFAULT_MAX_CALLS_CAP = 500
MAX_PREVIEW = 100
MAX_EVENTS = 20_000
MAX_SCAN_DEPTH = 5
MAX_SCAN_ENTRIES = 400
MAX_YAML_BYTES = 256 * 1024
MAX_CLUSTER_DECISIONS = 50
UI_DIR = ".xwalk-ui"

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    UI_DIR,
}
_STATUSES = ("matched", "needs_review", "unmatched", "failed", "pending")
_DECISIONS = ("accept", "reject", "replace", "no_match", "defer")


class ApiError(Exception):
    """A request the API cannot serve. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass(frozen=True)
class Download:
    """A file to send as an attachment instead of JSON."""

    filename: str
    content_type: str
    body: bytes


Params = Mapping[str, Any]
Response = dict[str, Any] | Download


# --- parameters -----------------------------------------------------------------------


def _str(params: Params, name: str, *, required: bool = True, default: str = "") -> str:
    value = params.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ApiError(400, "missing_parameter", f"parameter {name!r} is required")
        return default
    if not isinstance(value, str):
        raise ApiError(400, "invalid_parameter", f"parameter {name!r} must be a string")
    return value.strip()


def _int(
    params: Params, name: str, *, default: int | None, low: int, high: int | None = None
) -> int | None:
    value = params.get(name)
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ApiError(400, "invalid_parameter", f"parameter {name!r} must be an integer") from None
    if isinstance(value, bool) or number < low or (high is not None and number > high):
        bounds = f"between {low} and {high}" if high is not None else f">= {low}"
        raise ApiError(400, "invalid_parameter", f"parameter {name!r} must be {bounds}")
    return number


def _bool(params: Params, name: str, *, default: bool) -> bool:
    value = params.get(name)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("1", "true", "yes", "on"):
        return True
    if isinstance(value, str) and value.lower() in ("0", "false", "no", "off"):
        return False
    raise ApiError(400, "invalid_parameter", f"parameter {name!r} must be a boolean")


def _envelope(operation: str, work: Callable[[], ops.OpResult]) -> dict[str, Any]:
    try:
        result = work()
    except Exception as exc:  # reported in the envelope, never as a server error
        result = ops.failure_result(operation, exc)
    return result.envelope()


# --- background tasks -----------------------------------------------------------------


@dataclass
class Task:
    """One match or cluster invocation running in its own thread."""

    id: str
    kind: str
    job: str
    out: str  # as shown: relative to the workspace root
    out_path: str  # absolute, for the operations
    model: str
    max_calls: int | None
    limit: int | None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    state: str = "running"  # running, done, cancelled
    events: list[dict[str, Any]] = field(default_factory=list)
    dropped_events: int = 0
    result: dict[str, Any] | None = None
    cancel_requested: bool = False
    _loop: asyncio.AbstractEventLoop | None = None
    _future: asyncio.Future[ops.OpResult] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            if len(self.events) < MAX_EVENTS:
                self.events.append(event)
            else:
                self.dropped_events += 1

    def view(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            events = self.events[since:]
            return {
                "id": self.id,
                "kind": self.kind,
                "job": self.job,
                "out": self.out,
                "model": self.model,
                "max_calls": self.max_calls,
                "limit": self.limit,
                "state": self.state,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "event_count": len(self.events),
                "dropped_events": self.dropped_events,
                "events": events,
                "since": since,
                "result": self.result,
            }

    def cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True
            loop, future = self._loop, self._future
        if loop is not None and future is not None and not loop.is_closed():
            loop.call_soon_threadsafe(future.cancel)


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def running_for(self, out: str) -> Task | None:
        with self._lock:
            return next(
                (t for t in self._tasks.values() if t.out_path == out and t.state == "running"),
                None,
            )

    def get(self, task_id: str) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ApiError(404, "task_not_found", f"no task {task_id!r}")
        return task

    def all(self) -> list[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.started_at, reverse=True)

    def start(
        self,
        task: Task,
        operation: str,
        work: Callable[[], Coroutine[Any, Any, ops.OpResult]],
    ) -> Task:
        with self._lock:
            self._tasks[task.id] = task

        def body() -> None:
            loop = asyncio.new_event_loop()
            state = "done"
            try:
                future = loop.create_task(work())
                with task._lock:
                    task._loop, task._future = loop, future
                    cancelled = task.cancel_requested
                if cancelled:
                    future.cancel()
                result = loop.run_until_complete(future)
                envelope = result.envelope()
            except asyncio.CancelledError:
                state = "cancelled"
                envelope = ops.interrupted(operation, task.out_path).envelope()
            except Exception as exc:
                envelope = ops.failure_result(operation, exc).envelope()
            finally:
                with task._lock:
                    task._loop = task._future = None
                loop.close()
            with task._lock:
                task.result = envelope
                task.state = state
                task.finished_at = time.time()

        threading.Thread(target=body, name=f"xwalk-ui-{task.id}", daemon=True).start()
        return task

    def wait(self, task_id: str, timeout: float = 60.0) -> Task:
        """Block until a task finishes (tests and scripts)."""
        task = self.get(task_id)
        deadline = time.monotonic() + timeout
        while task.state == "running":
            if time.monotonic() > deadline:
                raise TimeoutError(f"task {task_id} still running after {timeout}s")
            time.sleep(0.02)
        return task


# --- the workspace ----------------------------------------------------------------------


def _job_kind(path: Path) -> str | None:
    """``match`` or ``cluster`` when the YAML file looks like an xwalk job, else None."""
    try:
        if path.stat().st_size > MAX_YAML_BYTES:
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("kind") == "cluster" and "source" in data:
        return "cluster"
    if {"templates", "target", "source"} <= set(data):
        return "match"
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _run_kind(manifest: Mapping[str, Any]) -> str:
    return "cluster" if manifest.get("operation") == "cluster" else "match"


def _csv_rows(path: Path) -> Iterator[dict[str, str]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


class Workspace:
    """The directory the UI may read and write, with the API handlers over it."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_calls_cap: int = DEFAULT_MAX_CALLS_CAP,
        allow_endpoint: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace root {self.root} is not a directory")
        if max_calls_cap < 1:
            raise ValueError(f"max_calls_cap must be >= 1, got {max_calls_cap}")
        self.max_calls_cap = max_calls_cap
        self.allow_endpoint = allow_endpoint
        self.tasks = TaskManager()
        self._state_lock = threading.Lock()

    # paths

    def path(self, value: str, *, must_exist: bool = False) -> Path:
        """`value` resolved inside the workspace, or `ApiError` 403."""
        raw = Path(os.path.expanduser(value))
        resolved = (raw if raw.is_absolute() else self.root / raw).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ApiError(
                403, "outside_workspace", f"{value!r} is outside the workspace {self.root}"
            )
        if must_exist and not resolved.exists():
            raise ApiError(404, "not_found", f"{value!r} does not exist")
        return resolved

    def rel(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError:
            return str(path)
        return relative.as_posix() or "."

    def _ui_dir(self) -> Path:
        return self.root / UI_DIR

    def _run_jobs(self) -> dict[str, str]:
        data = _read_json(self._ui_dir() / "runs.json")
        return {str(k): str(v) for k, v in (data or {}).items()}

    def _remember_job(self, run_dir: Path, job: Path) -> None:
        with self._state_lock:
            jobs = self._run_jobs()
            jobs[self.rel(run_dir)] = self.rel(job)
            self._ui_dir().mkdir(exist_ok=True)
            path = self._ui_dir() / "runs.json"
            path.write_text(json.dumps(jobs, indent=1, sort_keys=True), encoding="utf-8")

    # info and discovery

    def info(self, params: Params) -> Response:
        return {
            "version": __version__,
            "root": str(self.root),
            "max_calls_cap": self.max_calls_cap,
            "allow_endpoint": self.allow_endpoint,
            "offline_model": offline.OFFLINE_MODEL_NAME,
            "page_size": ops.MAX_PAGE_SIZE,
            "search_limit": ops.MAX_SEARCH_LIMIT,
        }

    def workspace(self, params: Params) -> Response:
        """Job files, run directories and gold-label files under the root (bounded)."""
        jobs: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        golds: list[str] = []
        truncated = False
        seen = 0
        run_jobs = self._run_jobs()
        for directory, dirnames, filenames in os.walk(self.root):
            here = Path(directory)
            depth = len(here.relative_to(self.root).parts)
            names = set(filenames)
            if "manifest.json" in names:
                manifest = _read_json(here / "manifest.json")
                if manifest is not None and "run_fingerprint" in manifest:
                    rel = self.rel(here)
                    runs.append(
                        {
                            "dir": rel,
                            "kind": _run_kind(manifest),
                            "job": manifest.get("job"),
                            "job_path": run_jobs.get(rel),
                            "model": manifest.get("model"),
                            "run_state": manifest.get("run_state"),
                            "counts": manifest.get("counts") or {},
                            "modified": (here / "manifest.json").stat().st_mtime,
                        }
                    )
                    dirnames[:] = []  # a run directory holds no jobs (its index is internal)
                    continue
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".") and depth < MAX_SCAN_DEPTH
            )
            for name in sorted(filenames):
                seen += 1
                if seen > MAX_SCAN_ENTRIES * 20:
                    truncated = True
                    dirnames[:] = []
                    break
                path = here / name
                if name.endswith((".yaml", ".yml")):
                    kind = _job_kind(path)
                    if kind is not None:
                        jobs.append({"path": self.rel(path), "kind": kind, "name": path.stem})
                elif name.endswith(".csv") and "gold" in name.lower():
                    golds.append(self.rel(path))
            if len(jobs) + len(runs) > MAX_SCAN_ENTRIES:
                truncated = True
                break
        runs.sort(key=lambda r: r["modified"], reverse=True)
        return {
            "root": str(self.root),
            "jobs": jobs,
            "runs": runs,
            "gold": golds,
            "truncated": truncated,
        }

    def init(self, params: Params) -> Response:
        dest = self.path(_str(params, "dest"))
        envelope = _envelope("init", lambda: ops.init(dest))
        if envelope["status"] == "ok":
            envelope["data"]["job_path"] = self.rel(dest / "job.yaml")
        return envelope

    # jobs

    def job(self, params: Params) -> Response:
        path = self.path(_str(params, "path"), must_exist=True)
        if not path.is_file():
            raise ApiError(400, "not_a_file", f"{self.rel(path)} is not a file")
        if path.stat().st_size > MAX_YAML_BYTES:
            raise ApiError(413, "too_large", f"{self.rel(path)} is larger than a job file")
        text = path.read_text(encoding="utf-8", errors="replace")
        kind = _job_kind(path)
        runs = [run for run, job in self._run_jobs().items() if job == self.rel(path)]
        return {
            "path": self.rel(path),
            "kind": kind,
            "text": text,
            "runs": runs,
            "suggested_out": self._suggest_out(path),
        }

    def _suggest_out(self, job: Path) -> str:
        base = job.parent / "runs"
        for n in itertools.count(1):
            candidate = base / f"ui-{n}"
            if not candidate.exists():
                return self.rel(candidate)
        raise AssertionError("unreachable")  # pragma: no cover

    def validate(self, params: Params) -> Response:
        job = self.path(_str(params, "job"), must_exist=True)
        check = _bool(params, "check_credentials", default=False)
        scan = _bool(params, "scan_records", default=True)
        return _envelope(
            "validate", lambda: ops.validate(job, check_credentials=check, scan_records=scan)
        )

    def preview(self, params: Params) -> Response:
        """The first records of one collection, as the job's templates render them."""
        job = self.path(_str(params, "job"), must_exist=True)
        role = _str(params, "role", default="source", required=False)
        if role not in ("source", "target"):
            raise ApiError(400, "invalid_parameter", "role must be 'source' or 'target'")
        limit = _int(params, "limit", default=20, low=1, high=MAX_PREVIEW) or 20
        offset = _int(params, "offset", default=0, low=0) or 0
        if _job_kind(job) == "cluster":
            from xwalk.cluster.operation import load_job as load_cluster

            if role != "source":
                raise ApiError(400, "invalid_parameter", "a clustering job has no target")
            spec: Any = load_cluster(job, "preview")
            renders = {"query": "render_query"}
        else:
            spec = ops.load_valid_job(job, operation="preview")
            renders = (
                {"query": "render_query", "context": "render_context"}
                if role == "source"
                else {"doc": "render_doc", "candidate": "render_candidate"}
            )
        templates = spec.build_templates()
        builder = spec.build_source_records if role == "source" else spec.build_target_records
        rows: list[dict[str, Any]] = []
        more = False
        try:
            records = builder()
            for record in itertools.islice(records, offset, offset + limit + 1):
                if len(rows) == limit:
                    more = True
                    break
                rendered: dict[str, str] = {}
                for label, method in renders.items():
                    try:
                        rendered[label] = getattr(templates, method)(record)
                    except Exception as exc:
                        rendered[label] = f"<template error: {exc}>"
                rows.append(
                    {"id": record.id, "fields": _jsonable(record.fields), "rendered": rendered}
                )
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(
                422, "records_unreadable", f"{role}: {type(exc).__name__}: {exc}"
            ) from None
        return {
            "role": role,
            "offset": offset,
            "limit": limit,
            "rows": rows,
            "next_offset": offset + limit if more else None,
        }

    def search(self, params: Params) -> Response:
        job = self.path(_str(params, "job"), must_exist=True)
        query = _str(params, "query")
        limit = _int(params, "limit", default=10, low=1, high=ops.MAX_SEARCH_LIMIT) or 10
        rebuild = _bool(params, "rebuild", default=False)
        index_dir = self._search_index(job)

        def work() -> ops.OpResult:
            if rebuild:
                ops.index(job, index_dir, rebuild=True)
            return ops.search(job, query, index_dir=index_dir, limit=limit)

        envelope = _envelope("search", work)
        envelope["data"]["index_dir"] = self.rel(index_dir)
        return envelope

    def _search_index(self, job: Path) -> Path:
        digest = hashlib.sha256(str(job).encode("utf-8")).hexdigest()[:16]
        return self._ui_dir() / "index" / digest

    # tasks

    def start_task(self, params: Params) -> Response:
        job = self.path(_str(params, "job"), must_exist=True)
        out = self.path(_str(params, "out"))
        if out == self.root:
            raise ApiError(400, "invalid_parameter", "the run directory cannot be the root")
        kind = _job_kind(job)
        if kind is None:
            raise ApiError(400, "not_a_job", f"{self.rel(job)} is not an xwalk job file")
        model = _str(params, "model", default="offline", required=False)
        if model not in ("offline", "endpoint"):
            raise ApiError(400, "invalid_parameter", "model must be 'offline' or 'endpoint'")
        max_calls = _int(params, "max_calls", default=None, low=1, high=self.max_calls_cap)
        limit = _int(params, "limit", default=None, low=1)
        resume = _bool(params, "resume", default=True)
        if model == "endpoint":
            if not self.allow_endpoint:
                raise ApiError(
                    403,
                    "endpoint_disabled",
                    "this server was started with --offline-only; use the offline model",
                )
            if max_calls is None:
                raise ApiError(
                    400,
                    "missing_parameter",
                    "max_calls is required with the job's endpoint (every call is billed)",
                )
        if self.tasks.running_for(str(out)) is not None:
            raise ApiError(409, "task_running", f"a task is already writing {self.rel(out)}")
        if kind == "cluster" and limit is not None:
            raise ApiError(400, "invalid_parameter", "limit is not supported for clustering")

        task = Task(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            job=self.rel(job),
            out=self.rel(out),
            out_path=str(out),
            model=model,
            max_calls=max_calls,
            limit=limit,
        )
        self._remember_job(out, job)

        if kind == "cluster":

            def cluster_progress(source_id: str, outcome: str) -> None:
                task.add_event({"source_id": source_id, "status": outcome})

            async def cluster_work() -> ops.OpResult:
                llm = offline.cluster_llm() if model == "offline" else None
                result = await ops.cluster_async(
                    job, out, max_calls=max_calls, llm=llm, progress=cluster_progress
                )
                return _mark_offline(result, model)

            return self.tasks.start(task, "cluster", cluster_work).view()

        def match_progress(result: Any) -> None:
            task.add_event(
                {
                    "source_id": result.source_id,
                    "status": result.status.value,
                    "matched_id": result.matched_id,
                    "confidence": result.confidence,
                }
            )

        async def match_work() -> ops.OpResult:
            llm = decider = None
            if model == "offline":
                from xwalk.decide.fake import FakeDecider

                llm = offline.match_llm()
                decider = FakeDecider(model=offline.OFFLINE_MODEL_NAME)
            result = await ops.run_async(
                job,
                out,
                resume=resume,
                limit=limit,
                max_calls=max_calls,
                llm=llm,
                decider=decider,
                progress=match_progress,
            )
            return _mark_offline(result, model)

        return self.tasks.start(task, "match", match_work).view()

    def list_tasks(self, params: Params) -> Response:
        return {"tasks": [{**t.view(), "events": []} for t in self.tasks.all()]}

    def task(self, task_id: str, params: Params) -> Response:
        since = _int(params, "since", default=0, low=0) or 0
        return self.tasks.get(task_id).view(since)

    def cancel_task(self, task_id: str, params: Params) -> Response:
        task = self.tasks.get(task_id)
        task.cancel()
        return {"id": task.id, "cancel_requested": True, "state": task.state}

    # runs

    def _run_dir(self, params: Params) -> tuple[Path, dict[str, Any]]:
        run_dir = self.path(_str(params, "dir"), must_exist=True)
        manifest = _read_json(run_dir / "manifest.json")
        if manifest is None:
            raise ApiError(404, "run_not_found", f"{self.rel(run_dir)} is not a run directory")
        return run_dir, manifest

    def run(self, params: Params) -> Response:
        run_dir, manifest = self._run_dir(params)
        kind = _run_kind(manifest)
        job = self._run_jobs().get(self.rel(run_dir))
        if kind == "cluster":
            return {
                "kind": kind,
                "dir": self.rel(run_dir),
                "job_path": job,
                "manifest": manifest,
            }
        envelope = _envelope("inspect", lambda: ops.inspect(run_dir))
        return {"kind": kind, "dir": self.rel(run_dir), "job_path": job, "inspect": envelope}

    def results(self, params: Params) -> Response:
        run_dir, _ = self._run_dir(params)
        offset = _int(params, "offset", default=0, low=0) or 0
        limit = _int(params, "limit", default=50, low=1, high=ops.MAX_PAGE_SIZE) or 50
        status = _str(params, "status", required=False) or None
        if status is not None and status not in _STATUSES:
            raise ApiError(400, "invalid_parameter", f"status must be one of {_STATUSES}")
        return _envelope(
            "results",
            lambda: ops.list_results(run_dir, offset=offset, limit=limit, status=status),
        )

    def explain(self, params: Params) -> Response:
        run_dir, _ = self._run_dir(params)
        source_id = _str(params, "source_id")
        return _envelope("explain", lambda: ops.explain(run_dir, source_id, full=True))

    def review_rows(self, params: Params) -> Response:
        """The review worksheet (current `needs_review` rows) as JSON."""
        run_dir, _ = self._run_dir(params)
        with tempfile.TemporaryDirectory() as scratch:
            sheet = Path(scratch) / "review.csv"
            envelope = _envelope("review export", lambda: ops.review_export(run_dir, sheet))
            rows = list(_csv_rows(sheet)) if envelope["status"] != "error" else []
        envelope["data"] = {"rows": rows, "decisions": list(_DECISIONS)}
        envelope["artifacts"] = {}
        return envelope

    def review_apply(self, params: Params) -> Response:
        """Record reviewer decisions through `ops.review_apply` (all-or-nothing)."""
        run_dir, _ = self._run_dir(params)
        job = self.path(_str(params, "job"), must_exist=True)
        reviewer = _str(params, "reviewer")
        decisions = params.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ApiError(400, "missing_parameter", "decisions must be a non-empty list")
        with tempfile.TemporaryDirectory() as scratch:
            sheet = Path(scratch) / "review.csv"
            exported = _envelope("review export", lambda: ops.review_export(run_dir, sheet))
            if exported["status"] == "error":
                return exported
            worksheet = {row["result_key"]: row for row in _csv_rows(sheet)}
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            filled: list[dict[str, str]] = []
            for item in decisions:
                if not isinstance(item, dict):
                    raise ApiError(400, "invalid_parameter", "each decision must be an object")
                key = str(item.get("result_key") or "")
                if key not in worksheet:
                    raise ApiError(
                        409,
                        "not_reviewable",
                        f"{key!r} is not a current needs_review result; reload the review",
                    )
                decision = str(item.get("decision") or "")
                if decision not in _DECISIONS:
                    raise ApiError(400, "invalid_parameter", f"unknown decision {decision!r}")
                filled.append(
                    {
                        **worksheet[key],
                        "decision": decision,
                        "corrected_target_id": str(item.get("corrected_target_id") or ""),
                        "reviewer": reviewer,
                        "review_note": str(item.get("review_note") or ""),
                        "reviewed_at": stamp,
                    }
                )
            reviewed = Path(scratch) / "reviewed.csv"
            with reviewed.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(next(iter(worksheet.values()))))
                writer.writeheader()
                writer.writerows(filled)
            return _envelope("review apply", lambda: ops.review_apply(run_dir, reviewed, job))

    def evaluate(self, params: Params) -> Response:
        """Score the run's current view against a gold CSV (`source_id,gold_ids`)."""
        run_dir, manifest = self._run_dir(params)
        gold = self.path(_str(params, "gold"), must_exist=True)
        if _run_kind(manifest) != "match":
            raise ApiError(400, "invalid_parameter", "evaluation needs a matching run")

        def work() -> ops.OpResult:
            from xwalk.evaluate.gold import load_gold_csv
            from xwalk.evaluate.report import evaluate, render_report
            from xwalk.ledger import Ledger

            labels = load_gold_csv(gold)
            ledger = Ledger.open(run_dir / "ledger.sqlite")
            try:
                report = evaluate(ledger, str(manifest["run_fingerprint"]), labels)
            finally:
                ledger.close()
            return ops.OpResult(
                "eval",
                data={"report": report.as_dict(), "text": render_report(report)},
            )

        return _envelope("eval", work)

    def export(self, params: Params) -> Response:
        run_dir, manifest = self._run_dir(params)
        view = _str(params, "view")
        if _run_kind(manifest) == "cluster":
            names = {
                "members": "members.csv",
                "clusters": "clusters.csv",
                "unresolved": "unresolved.csv",
                "decisions": "decisions.jsonl",
            }
            if view not in names:
                raise ApiError(400, "invalid_parameter", f"view must be one of {sorted(names)}")
            path = run_dir / names[view]
            if not path.is_file():
                raise ApiError(404, "not_found", f"{names[view]} has not been written")
            return Download(names[view], _content_type(path.name), path.read_bytes())
        if view not in ops.EXPORT_VIEWS:
            raise ApiError(400, "invalid_parameter", f"view must be one of {ops.EXPORT_VIEWS}")
        suffix = "jsonl" if view == "history" else "csv"
        filename = f"{run_dir.name}-{view}.{suffix}"
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / filename
            try:
                ops.export(run_dir, view, out)
            except Exception as exc:
                result = ops.failure_result("export", exc)
                message = "; ".join(e.message for e in result.errors)
                raise ApiError(422, "export_failed", message) from None
            return Download(filename, _content_type(filename), out.read_bytes())

    # clustering runs

    def clusters(self, params: Params) -> Response:
        run_dir, manifest = self._run_dir(params)
        if _run_kind(manifest) != "cluster":
            raise ApiError(400, "invalid_parameter", f"{self.rel(run_dir)} is not a cluster run")
        offset = _int(params, "offset", default=0, low=0) or 0
        limit = _int(params, "limit", default=50, low=1, high=ops.MAX_PAGE_SIZE) or 50
        min_size = _int(params, "min_size", default=1, low=1) or 1
        clusters: list[dict[str, Any]] = []
        for row in _csv_rows(run_dir / "clusters.csv"):
            size = int(row.get("size") or 0)
            if size < min_size:
                continue
            lines = (row.get("representation") or "").splitlines()
            clusters.append(
                {
                    "cluster_id": row.get("cluster_id"),
                    "size": size,
                    "status": row.get("status"),
                    "seed_id": row.get("seed_id"),
                    "provenance": row.get("mint_provenance"),
                    "member_ids": [m for m in (row.get("member_ids") or "").split("|") if m],
                    "labels": [line[4:] for line in lines if line.startswith("  - ")],
                    "hidden": next(
                        (line.strip() for line in lines if line.strip().startswith("(+")), None
                    ),
                }
            )
        clusters.sort(key=lambda c: (-int(c["size"]), str(c["cluster_id"])))
        page = clusters[offset : offset + limit]
        end = offset + len(page)
        return {
            "dir": self.rel(run_dir),
            "total": len(clusters),
            "offset": offset,
            "limit": limit,
            "clusters": page,
            "next_offset": end if end < len(clusters) else None,
            "unresolved": list(itertools.islice(_csv_rows(run_dir / "unresolved.csv"), 500)),
        }

    def cluster_member(self, params: Params) -> Response:
        """One source's membership row and the decisions made about it (bounded)."""
        run_dir, manifest = self._run_dir(params)
        source_id = _str(params, "source_id")
        member = next(
            (r for r in _csv_rows(run_dir / "members.csv") if r.get("source_id") == source_id),
            None,
        )
        decisions: list[dict[str, Any]] = []
        path = run_dir / "decisions.jsonl"
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if entry.get("subject_id") == source_id:
                        decisions.append(entry)
                        if len(decisions) >= MAX_CLUSTER_DECISIONS:
                            break
        if member is None and not decisions:
            raise ApiError(404, "source_not_found", f"{source_id!r} is not in this run")
        return {"source_id": source_id, "member": member, "decisions": decisions}


def _mark_offline(result: ops.OpResult, model: str) -> ops.OpResult:
    if model == "offline":
        result.warnings.append(
            ops.OpMessage(
                "offline_model",
                "run with the UI's offline stand-in model: answers are scripted or lexical, "
                "not a model judgement",
            )
        )
    return result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _content_type(name: str) -> str:
    if name.endswith(".csv"):
        return "text/csv; charset=utf-8"
    if name.endswith(".jsonl"):
        return "application/x-ndjson; charset=utf-8"
    return "application/octet-stream"


# --- routing --------------------------------------------------------------------------

Handler = Callable[[Workspace, Params], Response]
TaskHandler = Callable[[Workspace, str, Params], Response]

ROUTES: dict[tuple[str, str], Handler] = {
    ("GET", "/api/info"): Workspace.info,
    ("GET", "/api/workspace"): Workspace.workspace,
    ("POST", "/api/init"): Workspace.init,
    ("GET", "/api/job"): Workspace.job,
    ("POST", "/api/validate"): Workspace.validate,
    ("GET", "/api/preview"): Workspace.preview,
    ("POST", "/api/search"): Workspace.search,
    ("GET", "/api/tasks"): Workspace.list_tasks,
    ("POST", "/api/tasks"): Workspace.start_task,
    ("GET", "/api/run"): Workspace.run,
    ("GET", "/api/results"): Workspace.results,
    ("GET", "/api/explain"): Workspace.explain,
    ("GET", "/api/review"): Workspace.review_rows,
    ("POST", "/api/review"): Workspace.review_apply,
    ("POST", "/api/eval"): Workspace.evaluate,
    ("GET", "/api/export"): Workspace.export,
    ("GET", "/api/clusters"): Workspace.clusters,
    ("GET", "/api/cluster-member"): Workspace.cluster_member,
}
TASK_ROUTES: dict[tuple[str, str], TaskHandler] = {
    ("GET", ""): Workspace.task,
    ("POST", "/cancel"): Workspace.cancel_task,
}


def dispatch(workspace: Workspace, method: str, path: str, params: Params) -> Response:
    """Route one request. Raises `ApiError` for an unknown route or bad parameters."""
    handler = ROUTES.get((method, path))
    if handler is not None:
        return handler(workspace, params)
    prefix = "/api/tasks/"
    if path.startswith(prefix):
        task_id, _, rest = path[len(prefix) :].partition("/")
        task_handler = TASK_ROUTES.get((method, f"/{rest}" if rest else ""))
        if task_handler is not None and task_id:
            return task_handler(workspace, task_id, params)
    if any(p == path for _, p in ROUTES):
        raise ApiError(405, "method_not_allowed", f"{method} is not allowed on {path}")
    raise ApiError(404, "not_found", f"no API route {path}")


__all__ = [
    "DEFAULT_MAX_CALLS_CAP",
    "ApiError",
    "Download",
    "Task",
    "TaskManager",
    "Workspace",
    "dispatch",
]

"""The clustering run's SQLite store: its own schema, one transaction per step.

Everything is append-only except the one-row-per-key tables (`meta`, `plans`):

- `sources`: the ordered snapshot the run was fingerprinted on.
- `steps`: one row per completed unit of work. A step's decisions, cluster revisions,
  assignment changes and merge outcomes are written in the same transaction as its
  `steps` row, so an interruption leaves either all of a step or none of it.
- `plans`: a phase's work list, fixed when the phase starts (resume must not recompute
  it from a state the phase has already changed).
- `decisions`: every model interaction (and every model-free mint), with the clusters
  retrieved and the clusters shown as `(cluster_id, revision)` pairs, the exact prompt
  and the raw response.
- `cluster_revisions`: each cluster's members and rendered representation at every
  revision. With `decisions`, the pool state behind any decision can be rebuilt.
- `assignments`: each change of a source's outcome; the latest row is current.
- `state_revisions`: the complete state at the end of every iteration; exports read the
  selected one.

Rows carry no timestamps (those live in `invocations`), so a resumed run and a clean
run can be compared row for row.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xwalk.records import Record

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
    order_index INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    source_hash TEXT NOT NULL,
    fields TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    seq INTEGER PRIMARY KEY,
    step_key TEXT NOT NULL UNIQUE,
    phase TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    subject TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS plans (
    phase TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    items TEXT NOT NULL,
    PRIMARY KEY (phase, iteration)
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY,
    step_key TEXT NOT NULL,
    phase TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    subject_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    retrieved TEXT NOT NULL,
    shown TEXT NOT NULL,
    system TEXT,
    prompt TEXT,
    raw TEXT,
    result TEXT NOT NULL,
    confidence REAL,
    error TEXT,
    usage TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cluster_revisions (
    cluster_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    seed_id TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    members TEXT NOT NULL,
    representation TEXT NOT NULL,
    live INTEGER NOT NULL,
    merged_into TEXT,
    mint_provenance TEXT NOT NULL,
    step_key TEXT NOT NULL,
    decision_id INTEGER,
    PRIMARY KEY (cluster_id, revision)
);
CREATE TABLE IF NOT EXISTS assignments (
    seq INTEGER PRIMARY KEY,
    step_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    cluster_id TEXT,
    reason TEXT NOT NULL,
    proposal_cluster_id TEXT,
    confidence REAL,
    decision_id INTEGER
);
CREATE INDEX IF NOT EXISTS assignments_source ON assignments (source_id, seq);
CREATE TABLE IF NOT EXISTS merges (
    seq INTEGER PRIMARY KEY,
    step_key TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    left_id TEXT NOT NULL,
    right_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    winner_id TEXT,
    decision_id INTEGER
);
CREATE TABLE IF NOT EXISTS state_revisions (
    revision INTEGER PRIMARY KEY,
    iteration INTEGER NOT NULL,
    state_hash TEXT NOT NULL,
    unresolved INTEGER NOT NULL,
    assignments TEXT NOT NULL,
    clusters TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invocations (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    run_state TEXT NOT NULL,
    usage TEXT,
    errors TEXT
);
"""

# Tables whose rows a resumed run must reproduce exactly (`invocations` holds clocks).
DETERMINISTIC_TABLES = (
    "meta",
    "sources",
    "steps",
    "plans",
    "decisions",
    "cluster_revisions",
    "assignments",
    "merges",
    "state_revisions",
)


class ClusterStoreError(RuntimeError):
    """The store belongs to another run, or to an unknown schema version."""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class StepWrites:
    """What one step will write. Committed together with its `steps` row, or not at all."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    merges: list[dict[str, Any]] = field(default_factory=list)
    state_revision: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


class ClusterStore:
    def __init__(self, conn: sqlite3.Connection, path: Path) -> None:
        self._conn = conn
        self.path = path

    @classmethod
    def open(cls, path: str | Path) -> ClusterStore:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript(_SCHEMA)
        store = cls(conn, path)
        version = store.get_meta("schema_version")
        if version is None:
            store.set_meta("schema_version", SCHEMA_VERSION)
        elif version != SCHEMA_VERSION:
            conn.close()
            raise ClusterStoreError(
                f"{path} has cluster store schema {version}; this xwalk reads {SCHEMA_VERSION}"
            )
        return store

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # --- meta -----------------------------------------------------------------------

    def get_meta(self, key: str) -> Any:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, _dumps(value)),
        )

    # --- snapshot -------------------------------------------------------------------

    def bind(self, run_fingerprint: str, sources: Sequence[tuple[Record, str]]) -> None:
        """Record the run identity and ordered snapshot, or check they match."""
        stored = self.get_meta("run_fingerprint")
        if stored is not None:
            if stored != run_fingerprint:
                raise ClusterStoreError(
                    f"{self.path} belongs to run {stored}, not {run_fingerprint}"
                )
            return
        with self.transaction() as conn:
            self.set_meta("run_fingerprint", run_fingerprint)
            conn.executemany(
                "INSERT INTO sources (order_index, source_id, source_hash, fields) "
                "VALUES (?, ?, ?, ?)",
                [
                    (index, record.id, digest, _dumps(dict(record.fields)))
                    for index, (record, digest) in enumerate(sources)
                ],
            )

    def sources(self) -> list[tuple[int, str, str, dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT order_index, source_id, source_hash, fields FROM sources ORDER BY order_index"
        )
        return [
            (r["order_index"], r["source_id"], r["source_hash"], json.loads(r["fields"]))
            for r in rows
        ]

    # --- steps ------------------------------------------------------------------------

    def completed_steps(self) -> set[str]:
        return {r["step_key"] for r in self._conn.execute("SELECT step_key FROM steps")}

    def max_decision_id(self) -> int:
        row = self._conn.execute("SELECT MAX(decision_id) AS m FROM decisions").fetchone()
        return int(row["m"] or 0)

    def commit_step(
        self, step_key: str, phase: str, iteration: int, subject: str | None, writes: StepWrites
    ) -> None:
        with self.transaction() as conn:
            for d in writes.decisions:
                conn.execute(
                    "INSERT INTO decisions (decision_id, step_key, phase, iteration, subject_id, "
                    "kind, retrieved, shown, system, prompt, raw, result, confidence, error, "
                    "usage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        d["decision_id"],
                        step_key,
                        phase,
                        iteration,
                        d["subject_id"],
                        d["kind"],
                        _dumps(d["retrieved"]),
                        _dumps(d["shown"]),
                        d.get("system"),
                        d.get("prompt"),
                        d.get("raw"),
                        _dumps(d["result"]),
                        d.get("confidence"),
                        d.get("error"),
                        _dumps(d["usage"]),
                    ),
                )
            for r in writes.revisions:
                conn.execute(
                    "INSERT INTO cluster_revisions (cluster_id, revision, seed_id, created_seq, "
                    "members, representation, live, merged_into, mint_provenance, step_key, "
                    "decision_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        r["cluster_id"],
                        r["revision"],
                        r["seed_id"],
                        r["created_seq"],
                        _dumps(r["members"]),
                        r["representation"],
                        int(r["live"]),
                        r["merged_into"],
                        r["mint_provenance"],
                        step_key,
                        r["decision_id"],
                    ),
                )
            for a in writes.assignments:
                conn.execute(
                    "INSERT INTO assignments (step_key, source_id, outcome, cluster_id, reason, "
                    "proposal_cluster_id, confidence, decision_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        step_key,
                        a["source_id"],
                        a["outcome"],
                        a["cluster_id"],
                        a["reason"],
                        a["proposal_cluster_id"],
                        a["confidence"],
                        a["decision_id"],
                    ),
                )
            for m in writes.merges:
                conn.execute(
                    "INSERT INTO merges (step_key, iteration, left_id, right_id, outcome, "
                    "winner_id, decision_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        step_key,
                        iteration,
                        m["left_id"],
                        m["right_id"],
                        m["outcome"],
                        m["winner_id"],
                        m["decision_id"],
                    ),
                )
            if writes.state_revision is not None:
                s = writes.state_revision
                conn.execute(
                    "INSERT INTO state_revisions (revision, iteration, state_hash, unresolved, "
                    "assignments, clusters) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        s["revision"],
                        s["iteration"],
                        s["state_hash"],
                        s["unresolved"],
                        _dumps(s["assignments"]),
                        _dumps(s["clusters"]),
                    ),
                )
            for key, value in writes.meta.items():
                self.set_meta(key, value)
            conn.execute(
                "INSERT INTO steps (step_key, phase, iteration, subject, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (step_key, phase, iteration, subject, writes.note),
            )

    # --- plans ------------------------------------------------------------------------

    def get_plan(self, phase: str, iteration: int) -> list[str] | None:
        row = self._conn.execute(
            "SELECT items FROM plans WHERE phase = ? AND iteration = ?", (phase, iteration)
        ).fetchone()
        return None if row is None else list(json.loads(row["items"]))

    def put_plan(self, phase: str, iteration: int, items: Sequence[str]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO plans (phase, iteration, items) VALUES (?, ?, ?)",
                (phase, iteration, _dumps(list(items))),
            )

    # --- reads ------------------------------------------------------------------------

    def latest_revisions(self) -> dict[str, dict[str, Any]]:
        """Each cluster's most recent revision."""
        rows = self._conn.execute("SELECT * FROM cluster_revisions ORDER BY cluster_id, revision")
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            latest[row["cluster_id"]] = self._revision(row)
        return latest

    def revision(self, cluster_id: str, revision: int) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM cluster_revisions WHERE cluster_id = ? AND revision = ?",
            (cluster_id, revision),
        ).fetchone()
        if row is None:
            raise KeyError(f"{cluster_id} revision {revision}")
        return self._revision(row)

    @staticmethod
    def _revision(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["members"] = json.loads(data["members"])
        data["live"] = bool(data["live"])
        return data

    def current_assignments(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM assignments ORDER BY seq")
        return {row["source_id"]: dict(row) for row in rows}

    def merges(self, iteration: int | None = None) -> list[dict[str, Any]]:
        if iteration is None:
            rows = self._conn.execute("SELECT * FROM merges ORDER BY seq")
        else:
            rows = self._conn.execute(
                "SELECT * FROM merges WHERE iteration = ? ORDER BY seq", (iteration,)
            )
        return [dict(r) for r in rows]

    def state_revisions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM state_revisions ORDER BY revision")
        out = []
        for row in rows:
            data = dict(row)
            data["assignments"] = json.loads(data["assignments"])
            data["clusters"] = json.loads(data["clusters"])
            out.append(data)
        return out

    def decisions(self) -> Iterator[dict[str, Any]]:
        for row in self._conn.execute("SELECT * FROM decisions ORDER BY decision_id"):
            data = dict(row)
            for key in ("retrieved", "shown", "result", "usage"):
                data[key] = json.loads(data[key])
            yield data

    def dump(self, table: str) -> list[tuple[Any, ...]]:
        """Every row of a deterministic table, for comparing runs."""
        if table not in DETERMINISTIC_TABLES:
            raise ValueError(table)
        return [tuple(r) for r in self._conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2")]

    # --- invocations ------------------------------------------------------------------

    def begin_invocation(self) -> int:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE invocations SET run_state = 'interrupted' WHERE run_state = 'running'"
            )
            cursor = conn.execute(
                "INSERT INTO invocations (started_at, run_state) VALUES (?, 'running')",
                (_now(),),
            )
        return int(cursor.lastrowid or 0)

    def finish_invocation(
        self,
        invocation: int,
        run_state: str,
        usage: Mapping[str, Any],
        errors: Sequence[Mapping[str, Any]],
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE invocations SET finished_at = ?, run_state = ?, usage = ?, errors = ? "
                "WHERE id = ?",
                (_now(), run_state, _dumps(dict(usage)), _dumps(list(errors)), invocation),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "DETERMINISTIC_TABLES",
    "SCHEMA_VERSION",
    "ClusterStore",
    "ClusterStoreError",
    "StepWrites",
]

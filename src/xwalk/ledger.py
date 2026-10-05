"""The run ledger: transactional, resumable execution state.

Every completed result is committed as it finishes, so a batch run never loses work.
JSONL/CSV/manifest files are exports regenerated from here, never the source of truth.

Two views of a run are kept apart (docs/claude-upgrade/CONTRACTS.md section 2):

- The **current view** has exactly one entry per source in the run's latest source
  snapshot: the result for that source's current content, or a pending entry when it
  has not been processed yet. Every export and report reads this view.
- **History** is every result ever committed for the run, including superseded source
  versions, removed sources and earlier revisions of a recomputed result. Nothing in it
  is deleted.

Schema versions: version 1 is the 0.1.1 layout (no ``ledger_meta`` table). Opening a
version 1 ledger copies the file aside first and then upgrades it additively; a newer
version than this code knows is refused without being touched.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xwalk.fingerprint import hash_value
from xwalk.records import MatchResult, MatchStatus
from xwalk.serde import result_from_dict, result_to_dict

LEDGER_SCHEMA_VERSION = 2

# Tables every 0.1.1 ledger has. Their presence without `ledger_meta` identifies v1.
_V1_TABLES = frozenset({"results", "llm_cache", "manifests", "reviews"})

_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    result_key       TEXT PRIMARY KEY,
    run_fingerprint  TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    source_hash      TEXT NOT NULL,
    status           TEXT NOT NULL,
    reason           TEXT NOT NULL,
    matched_id       TEXT,
    confidence       REAL,
    blob             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_by_run ON results(run_fingerprint, source_id);
CREATE INDEX IF NOT EXISTS results_by_target ON results(run_fingerprint, matched_id);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manifests (
    run_fingerprint TEXT PRIMARY KEY,
    blob            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    result_key        TEXT NOT NULL,
    run_fingerprint   TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_hash       TEXT NOT NULL,
    proposed_target_id TEXT,
    decision          TEXT NOT NULL,
    corrected_target_id TEXT,
    reviewer          TEXT NOT NULL,
    review_note       TEXT NOT NULL DEFAULT '',
    reviewed_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reviews_by_run ON reviews(run_fingerprint, result_key);
"""

# Version 2 is version 1 plus these statements. Every one is additive: a column with a
# default, or a new table. Nothing existing is rewritten or dropped.
_V2_ADDITIONS = (
    "CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    # `revision` counts commits of one result_key; `commit_seq` orders all commits.
    "ALTER TABLE results ADD COLUMN revision INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE results ADD COLUMN commit_seq INTEGER NOT NULL DEFAULT 0",
    # v1 rows: insertion order is the only commit order there is.
    "UPDATE results SET commit_seq = rowid",
    "CREATE INDEX results_by_source ON results(run_fingerprint, source_id, source_hash)",
    """CREATE TABLE result_history (
        history_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        result_key      TEXT NOT NULL,
        run_fingerprint TEXT NOT NULL,
        source_id       TEXT NOT NULL,
        source_hash     TEXT NOT NULL,
        status          TEXT NOT NULL,
        reason          TEXT NOT NULL,
        matched_id      TEXT,
        confidence      REAL,
        blob            TEXT NOT NULL,
        revision        INTEGER NOT NULL,
        commit_seq      INTEGER NOT NULL,
        superseded_at   TEXT NOT NULL
    )""",
    "CREATE INDEX result_history_by_run ON result_history(run_fingerprint, source_id)",
    # The revision a review was made against. v1 reviews were made against revision 1.
    "ALTER TABLE reviews ADD COLUMN result_revision INTEGER NOT NULL DEFAULT 1",
    """CREATE TABLE snapshots (
        snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        run_fingerprint TEXT NOT NULL,
        digest          TEXT NOT NULL,
        size            INTEGER NOT NULL,
        created_at      TEXT NOT NULL
    )""",
    """CREATE TABLE snapshot_entries (
        snapshot_id INTEGER NOT NULL,
        position    INTEGER NOT NULL,
        source_id   TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, position)
    )""",
    """CREATE TABLE invocations (
        invocation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        run_fingerprint TEXT NOT NULL,
        library_version TEXT NOT NULL,
        started_at      TEXT NOT NULL,
        finished_at     TEXT,
        run_state       TEXT NOT NULL,
        resume          INTEGER NOT NULL,
        record_limit    INTEGER,
        snapshot_id     INTEGER,
        usage           TEXT,
        errors          TEXT
    )""",
)


class UnsupportedLedgerError(Exception):
    """The file is not a ledger this version of xwalk can open. It was not modified."""


class LedgerClosedError(RuntimeError):
    """A write was attempted on a closed ledger: the write would have been lost."""


@dataclass(frozen=True)
class CurrentEntry:
    """One source in the current view. `result is None` means pending."""

    source_id: str
    source_hash: str
    result: MatchResult | None
    revision: int | None


@dataclass(frozen=True)
class HistoryEntry:
    result: MatchResult
    revision: int
    current: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not str(row[0]).startswith("sqlite_")
    }


def _free_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.v1-backup")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.v1-backup.{counter}")
        counter += 1
    return candidate


class Ledger:
    """SQLite in WAL mode, with a lock serialising writers.

    Reads go straight through — WAL allows concurrent readers. Every write is one
    explicit transaction executed without an `await` inside it, so a cancelled task can
    never leave half a transaction behind, and a failed one is rolled back.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        migrated_from: int | None = None,
        backup_path: Path | None = None,
    ) -> None:
        self._conn = connection
        self._write_lock = asyncio.Lock()
        self._closed = False
        self.migrated_from = migrated_from
        self.backup_path = backup_path

    @classmethod
    def open(cls, path: str | Path) -> Ledger:
        """Open or create a ledger.

        A 0.1.1 ledger is copied to ``<name>.v1-backup`` before its additive upgrade.
        A ledger from a newer xwalk, or a file that is not a ledger, raises
        `UnsupportedLedgerError` and is left exactly as it was.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            version = cls._detect_version(conn, path)
            backup: Path | None = None
            if version == 1:
                backup = _free_backup_path(path)
                shutil.copy2(path, backup)
                wal = path.with_name(path.name + "-wal")
                if wal.exists() and wal.stat().st_size:
                    shutil.copy2(wal, backup.with_name(backup.name + "-wal"))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            if version == 0:
                cls._create(conn)
            elif version == 1:
                cls._upgrade_from_v1(conn)
        except BaseException:
            conn.close()
            raise
        return cls(conn, migrated_from=1 if version == 1 else None, backup_path=backup)

    @staticmethod
    def _detect_version(conn: sqlite3.Connection, path: Path) -> int:
        """0 = empty file, 1 = 0.1.1 layout, 2 = current. Anything else raises."""
        try:
            tables = _tables(conn)
        except sqlite3.DatabaseError as exc:
            raise UnsupportedLedgerError(f"{path} is not an SQLite database: {exc}") from exc
        if not tables:
            return 0
        if "ledger_meta" in tables:
            row = conn.execute(
                "SELECT value FROM ledger_meta WHERE key = 'schema_version'"
            ).fetchone()
            raw = None if row is None else str(row[0])
            if raw is None or not raw.isdigit():
                raise UnsupportedLedgerError(f"{path} has no readable schema_version ({raw!r})")
            version = int(raw)
            if version != LEDGER_SCHEMA_VERSION:
                raise UnsupportedLedgerError(
                    f"{path} is ledger schema version {version}; this xwalk reads versions "
                    f"1-{LEDGER_SCHEMA_VERSION}. Use the xwalk version that wrote it; the "
                    f"file was not modified"
                )
            return version
        if tables >= _V1_TABLES:
            return 1
        raise UnsupportedLedgerError(
            f"{path} is not an xwalk ledger (tables: {sorted(tables)}); it was not modified"
        )

    @staticmethod
    def _create(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _V1_SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            Ledger._apply_v2(conn, migrated_from=None)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _upgrade_from_v1(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            Ledger._apply_v2(conn, migrated_from=1)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _apply_v2(conn: sqlite3.Connection, *, migrated_from: int | None) -> None:
        for statement in _V2_ADDITIONS:
            conn.execute(statement)
        meta = {"schema_version": str(LEDGER_SCHEMA_VERSION), "created_at": _now()}
        if migrated_from is not None:
            meta = {**meta, "migrated_from": str(migrated_from), "migrated_at": meta["created_at"]}
            del meta["created_at"]
        conn.executemany("INSERT INTO ledger_meta (key, value) VALUES (?, ?)", sorted(meta.items()))

    # --- lifecycle ---------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM ledger_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row[0])

    @property
    def journal_mode(self) -> str:
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    def _check_open(self) -> None:
        if self._closed:
            raise LedgerClosedError("the ledger is closed; this write would be lost")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._check_open()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            # A failed COMMIT leaves the transaction open; without this ROLLBACK the next
            # write's COMMIT would silently commit the half-done work.
            with suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._conn.close()

    # --- results -------------------------------------------------------------------

    async def put_result(self, result: MatchResult) -> None:
        """Commit a result. Rewriting an existing key keeps the old revision in history."""
        self._check_open()
        blob = json.dumps(result_to_dict(result), ensure_ascii=False)
        async with self._write_lock:
            self._write_result(result, blob)

    def _write_result(self, result: MatchResult, blob: str) -> None:
        with self._transaction():
            existing = self._conn.execute(
                "SELECT revision FROM results WHERE result_key = ?", (result.result_key,)
            ).fetchone()
            seq = int(
                self._conn.execute(
                    "SELECT COALESCE(MAX(commit_seq), 0) + 1 FROM results"
                ).fetchone()[0]
            )
            revision = 1
            if existing is not None:
                revision = int(existing["revision"]) + 1
                self._conn.execute(
                    """INSERT INTO result_history (result_key, run_fingerprint, source_id,
                        source_hash, status, reason, matched_id, confidence, blob, revision,
                        commit_seq, superseded_at)
                    SELECT result_key, run_fingerprint, source_id, source_hash, status, reason,
                        matched_id, confidence, blob, revision, commit_seq, ?
                    FROM results WHERE result_key = ?""",
                    (_now(), result.result_key),
                )
            self._conn.execute(
                """INSERT INTO results (result_key, run_fingerprint, source_id, source_hash,
                    status, reason, matched_id, confidence, blob, revision, commit_seq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(result_key) DO UPDATE SET
                    run_fingerprint=excluded.run_fingerprint,
                    source_id=excluded.source_id,
                    source_hash=excluded.source_hash,
                    status=excluded.status,
                    reason=excluded.reason,
                    matched_id=excluded.matched_id,
                    confidence=excluded.confidence,
                    blob=excluded.blob,
                    revision=excluded.revision,
                    commit_seq=excluded.commit_seq""",
                (
                    result.result_key,
                    result.run_fingerprint,
                    result.source_id,
                    result.source_hash,
                    result.status.value,
                    result.reason.value,
                    result.matched_id,
                    result.confidence,
                    blob,
                    revision,
                    seq,
                ),
            )

    def get_result(self, result_key: str) -> MatchResult | None:
        """The latest revision stored under a key, current or not."""
        row = self._conn.execute(
            "SELECT blob FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return None if row is None else result_from_dict(json.loads(row["blob"]))

    def has_result(self, result_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return row is not None

    def is_settled(self, result_key: str) -> bool:
        """A committed result that resume should keep. `failed` results are retried."""
        row = self._conn.execute(
            "SELECT status FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return row is not None and row["status"] != MatchStatus.FAILED.value

    def result_revision(self, result_key: str) -> int | None:
        row = self._conn.execute(
            "SELECT revision FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return None if row is None else int(row["revision"])

    # --- current view ------------------------------------------------------------------

    def current_snapshot_id(self, run_fingerprint: str) -> int | None:
        row = self._conn.execute(
            "SELECT snapshot_id FROM snapshots WHERE run_fingerprint = ? "
            "ORDER BY snapshot_id DESC LIMIT 1",
            (run_fingerprint,),
        ).fetchone()
        return None if row is None else int(row["snapshot_id"])

    def snapshot_size(self, run_fingerprint: str) -> int | None:
        row = self._conn.execute(
            "SELECT size FROM snapshots WHERE run_fingerprint = ? "
            "ORDER BY snapshot_id DESC LIMIT 1",
            (run_fingerprint,),
        ).fetchone()
        return None if row is None else int(row["size"])

    def put_snapshot(self, run_fingerprint: str, entries: Sequence[tuple[str, str]]) -> int:
        """Record the full ordered `(source_id, source_hash)` list of one invocation.

        An identical consecutive snapshot is reused, so an unchanged rerun adds nothing.
        """
        self._check_open()
        digest = hash_value([list(entry) for entry in entries])
        latest = self._conn.execute(
            "SELECT snapshot_id, digest FROM snapshots WHERE run_fingerprint = ? "
            "ORDER BY snapshot_id DESC LIMIT 1",
            (run_fingerprint,),
        ).fetchone()
        if latest is not None and latest["digest"] == digest:
            return int(latest["snapshot_id"])
        with self._transaction():
            cursor = self._conn.execute(
                "INSERT INTO snapshots (run_fingerprint, digest, size, created_at) "
                "VALUES (?, ?, ?, ?)",
                (run_fingerprint, digest, len(entries), _now()),
            )
            snapshot_id = int(cursor.lastrowid or 0)
            self._conn.executemany(
                "INSERT INTO snapshot_entries (snapshot_id, position, source_id, source_hash) "
                "VALUES (?, ?, ?, ?)",
                [(snapshot_id, i, sid, shash) for i, (sid, shash) in enumerate(entries)],
            )
        return snapshot_id

    def _current_rows(self, run_fingerprint: str, columns: str) -> Iterator[sqlite3.Row]:
        """Rows of the current view, ordered by source id.

        With a snapshot: one row per snapshot entry, result columns NULL when pending.
        Without one (a 0.1.1 ledger, or results written outside `run_batch`): the latest
        committed row per source id.
        """
        snapshot_id = self.current_snapshot_id(run_fingerprint)
        if snapshot_id is not None:
            yield from self._conn.execute(
                f"""SELECT e.source_id AS entry_source_id, e.source_hash AS entry_source_hash,
                    {columns}
                FROM snapshot_entries e
                LEFT JOIN results r ON r.run_fingerprint = ? AND r.source_id = e.source_id
                    AND r.source_hash = e.source_hash
                WHERE e.snapshot_id = ?
                ORDER BY e.source_id""",
                (run_fingerprint, snapshot_id),
            )
            return
        yield from self._conn.execute(
            f"""SELECT r.source_id AS entry_source_id, r.source_hash AS entry_source_hash,
                {columns}
            FROM results r
            WHERE r.run_fingerprint = ? AND r.commit_seq = (
                SELECT MAX(r2.commit_seq) FROM results r2
                WHERE r2.run_fingerprint = r.run_fingerprint AND r2.source_id = r.source_id)
            ORDER BY r.source_id""",
            (run_fingerprint,),
        )

    def iter_current(self, run_fingerprint: str) -> Iterator[CurrentEntry]:
        for row in self._current_rows(run_fingerprint, "r.blob, r.revision"):
            blob = row["blob"]
            yield CurrentEntry(
                source_id=str(row["entry_source_id"]),
                source_hash=str(row["entry_source_hash"]),
                result=None if blob is None else result_from_dict(json.loads(blob)),
                revision=None if row["revision"] is None else int(row["revision"]),
            )

    def iter_results(self, run_fingerprint: str) -> Iterator[MatchResult]:
        """Current results, one per source, ordered by source id. Pending ones are absent."""
        for entry in self.iter_current(run_fingerprint):
            if entry.result is not None:
                yield entry.result

    def current_keys(self, run_fingerprint: str) -> dict[str, int]:
        """result_key -> revision for every current (non-pending) result."""
        return {
            str(row["result_key"]): int(row["revision"])
            for row in self._current_rows(run_fingerprint, "r.result_key, r.revision")
            if row["result_key"] is not None
        }

    def count(self, run_fingerprint: str) -> int:
        """Current results (pending entries excluded)."""
        return len(self.current_keys(run_fingerprint))

    def iter_current_summaries(self, run_fingerprint: str) -> Iterator[dict[str, Any]]:
        """The current view as light rows, ordered by source id, without decoding any
        stored result: `source_id`, `status` (`pending` when unprocessed), `reason`,
        `matched_id`, `confidence`, `revision`, `result_key`."""
        columns = "r.result_key, r.status, r.reason, r.matched_id, r.confidence, r.revision"
        for row in self._current_rows(run_fingerprint, columns):
            pending = row["result_key"] is None
            yield {
                "source_id": str(row["entry_source_id"]),
                "status": "pending" if pending else str(row["status"]),
                "reason": "pending" if pending else str(row["reason"]),
                "matched_id": row["matched_id"],
                "confidence": row["confidence"],
                "revision": None if row["revision"] is None else int(row["revision"]),
                "result_key": row["result_key"],
            }

    def pending_count(self, run_fingerprint: str) -> int:
        return sum(
            1
            for row in self._current_rows(run_fingerprint, "r.result_key")
            if row["result_key"] is None
        )

    def count_by_status(self, run_fingerprint: str) -> dict[MatchStatus, int]:
        counts: dict[MatchStatus, int] = {}
        for row in self._current_rows(run_fingerprint, "r.status"):
            if row["status"] is not None:
                status = MatchStatus(row["status"])
                counts[status] = counts.get(status, 0) + 1
        return counts

    def duplicate_targets(self, run_fingerprint: str) -> dict[str, list[str]]:
        """Several current source records selected the same target. Reported, never
        resolved."""
        by_target: dict[str, list[str]] = {}
        for row in self._current_rows(run_fingerprint, "r.matched_id"):
            if row["matched_id"] is not None:
                by_target.setdefault(str(row["matched_id"]), []).append(str(row["entry_source_id"]))
        return {target: sorted(ids) for target, ids in by_target.items() if len(ids) > 1}

    def removed_sources(self, run_fingerprint: str) -> list[str]:
        """Source ids with results in this run that the current snapshot no longer has."""
        snapshot_id = self.current_snapshot_id(run_fingerprint)
        if snapshot_id is None:
            return []
        cursor = self._conn.execute(
            """SELECT DISTINCT source_id FROM results WHERE run_fingerprint = ?
                AND source_id NOT IN (
                    SELECT source_id FROM snapshot_entries WHERE snapshot_id = ?)
            ORDER BY source_id""",
            (run_fingerprint, snapshot_id),
        )
        return [str(row["source_id"]) for row in cursor]

    # --- history -----------------------------------------------------------------------

    def iter_history(self, run_fingerprint: str) -> Iterator[HistoryEntry]:
        """Every result ever committed for the run, by source id then commit order."""
        current = self.current_keys(run_fingerprint)
        cursor = self._conn.execute(
            """SELECT blob, result_key, revision, commit_seq, source_id, 1 AS live
                FROM results WHERE run_fingerprint = ?
            UNION ALL
            SELECT blob, result_key, revision, commit_seq, source_id, 0 AS live
                FROM result_history WHERE run_fingerprint = ?
            ORDER BY source_id, commit_seq""",
            (run_fingerprint, run_fingerprint),
        )
        for row in cursor:
            key, revision = str(row["result_key"]), int(row["revision"])
            yield HistoryEntry(
                result=result_from_dict(json.loads(row["blob"])),
                revision=revision,
                current=bool(row["live"]) and current.get(key) == revision,
            )

    def history_count(self, run_fingerprint: str) -> int:
        row = self._conn.execute(
            """SELECT (SELECT COUNT(*) FROM results WHERE run_fingerprint = ?)
                + (SELECT COUNT(*) FROM result_history WHERE run_fingerprint = ?) AS n""",
            (run_fingerprint, run_fingerprint),
        ).fetchone()
        return int(row["n"])

    # --- invocations -------------------------------------------------------------------

    def begin_invocation(
        self, run_fingerprint: str, *, library_version: str, resume: bool, limit: int | None
    ) -> int:
        with self._transaction():
            # Single writer per run directory: an earlier invocation still marked running
            # was killed before it could record how it ended.
            self._conn.execute(
                "UPDATE invocations SET run_state = 'interrupted' "
                "WHERE run_fingerprint = ? AND run_state = 'running'",
                (run_fingerprint,),
            )
            cursor = self._conn.execute(
                """INSERT INTO invocations (run_fingerprint, library_version, started_at,
                    run_state, resume, record_limit) VALUES (?, ?, ?, 'running', ?, ?)""",
                (run_fingerprint, library_version, _now(), int(resume), limit),
            )
        return int(cursor.lastrowid or 0)

    def finish_invocation(
        self,
        invocation_id: int,
        *,
        run_state: str,
        snapshot_id: int | None,
        usage: Mapping[str, Any],
        errors: Sequence[Mapping[str, Any]],
    ) -> None:
        with self._transaction():
            self._conn.execute(
                """UPDATE invocations SET finished_at = ?, run_state = ?, snapshot_id = ?,
                    usage = ?, errors = ? WHERE invocation_id = ?""",
                (
                    _now(),
                    run_state,
                    snapshot_id,
                    json.dumps(dict(usage), sort_keys=True),
                    json.dumps([dict(e) for e in errors], sort_keys=True, ensure_ascii=False),
                    invocation_id,
                ),
            )

    def last_invocation(self, run_fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM invocations WHERE run_fingerprint = ? "
            "ORDER BY invocation_id DESC LIMIT 1",
            (run_fingerprint,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["usage"] = json.loads(data["usage"]) if data["usage"] else None
        data["errors"] = json.loads(data["errors"]) if data["errors"] else []
        return data

    # --- llm cache -----------------------------------------------------------------------

    def get_cached(self, cache_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return None if row is None else str(row["response"])

    async def put_cached(self, cache_key: str, response: str) -> None:
        self._check_open()
        async with self._write_lock:
            with self._transaction():
                self._conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (cache_key, response) VALUES (?, ?)",
                    (cache_key, response),
                )

    # --- manifest ------------------------------------------------------------------------

    def put_manifest(self, run_fingerprint: str, manifest: Mapping[str, Any]) -> None:
        with self._transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO manifests (run_fingerprint, blob) VALUES (?, ?)",
                (run_fingerprint, json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
            )

    def get_manifest(self, run_fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT blob FROM manifests WHERE run_fingerprint = ?", (run_fingerprint,)
        ).fetchone()
        return None if row is None else dict(json.loads(row["blob"]))

    # --- reviews -------------------------------------------------------------------------

    def put_review(self, row: Mapping[str, Any]) -> None:
        """Append a decision, bound to the result revision it was made against."""
        self._check_open()
        values = dict(row)
        if values.get("result_revision") is None:
            values["result_revision"] = self.result_revision(str(values["result_key"])) or 1
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO reviews
                    (result_key, run_fingerprint, source_id, source_hash, proposed_target_id,
                     decision, corrected_target_id, reviewer, review_note, reviewed_at,
                     result_revision)
                VALUES (:result_key, :run_fingerprint, :source_id, :source_hash,
                        :proposed_target_id, :decision, :corrected_target_id, :reviewer,
                        :review_note, :reviewed_at, :result_revision)
                """,
                values,
            )

    def iter_reviews(self, run_fingerprint: str) -> Iterator[dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM reviews WHERE run_fingerprint = ? ORDER BY review_id",
            (run_fingerprint,),
        )
        for row in cursor:
            yield dict(row)

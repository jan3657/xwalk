"""The run ledger: transactional, resumable execution state.

Every completed result is committed as it finishes, so a batch run never loses work.
JSONL/CSV/manifest files are exports regenerated from here, never the source of truth.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from xwalk.records import MatchResult, MatchStatus
from xwalk.serde import result_from_dict, result_to_dict

_SCHEMA = """
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


class Ledger:
    """SQLite in WAL mode, with a lock serialising writers.

    Reads go straight through — WAL allows concurrent readers. Writes take an asyncio
    lock so a batch run with concurrency 32 never interleaves two transactions.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._write_lock = asyncio.Lock()

    @classmethod
    def open(cls, path: str | Path) -> Ledger:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        return cls(conn)

    @property
    def journal_mode(self) -> str:
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    # --- results ---------------------------------------------------------------

    async def put_result(self, result: MatchResult) -> None:
        blob = json.dumps(result_to_dict(result), ensure_ascii=False)
        async with self._write_lock:
            self._conn.execute(
                """
                INSERT INTO results
                    (result_key, run_fingerprint, source_id, source_hash,
                     status, reason, matched_id, confidence, blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(result_key) DO UPDATE SET
                    run_fingerprint=excluded.run_fingerprint,
                    source_id=excluded.source_id,
                    source_hash=excluded.source_hash,
                    status=excluded.status,
                    reason=excluded.reason,
                    matched_id=excluded.matched_id,
                    confidence=excluded.confidence,
                    blob=excluded.blob
                """,
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
                ),
            )

    def get_result(self, result_key: str) -> MatchResult | None:
        row = self._conn.execute(
            "SELECT blob FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return None if row is None else result_from_dict(json.loads(row["blob"]))

    def has_result(self, result_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return row is not None

    def iter_results(self, run_fingerprint: str) -> Iterator[MatchResult]:
        cursor = self._conn.execute(
            "SELECT blob FROM results WHERE run_fingerprint = ? ORDER BY source_id",
            (run_fingerprint,),
        )
        for row in cursor:
            yield result_from_dict(json.loads(row["blob"]))

    def count(self, run_fingerprint: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM results WHERE run_fingerprint = ?", (run_fingerprint,)
        ).fetchone()
        return int(row["n"])

    def count_by_status(self, run_fingerprint: str) -> dict[MatchStatus, int]:
        cursor = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM results WHERE run_fingerprint = ? GROUP BY status",
            (run_fingerprint,),
        )
        return {MatchStatus(row["status"]): int(row["n"]) for row in cursor}

    def duplicate_targets(self, run_fingerprint: str) -> dict[str, list[str]]:
        """Several source records selected the same target. Reported, never resolved."""
        cursor = self._conn.execute(
            """
            SELECT matched_id, GROUP_CONCAT(source_id) AS sources
            FROM results
            WHERE run_fingerprint = ? AND matched_id IS NOT NULL
            GROUP BY matched_id HAVING COUNT(*) > 1
            """,
            (run_fingerprint,),
        )
        return {row["matched_id"]: sorted(row["sources"].split(",")) for row in cursor}

    # --- llm cache -------------------------------------------------------------

    def get_cached(self, cache_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return None if row is None else str(row["response"])

    async def put_cached(self, cache_key: str, response: str) -> None:
        async with self._write_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, response) VALUES (?, ?)",
                (cache_key, response),
            )

    # --- manifest --------------------------------------------------------------

    def put_manifest(self, run_fingerprint: str, manifest: Mapping[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO manifests (run_fingerprint, blob) VALUES (?, ?)",
            (run_fingerprint, json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
        )

    def get_manifest(self, run_fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT blob FROM manifests WHERE run_fingerprint = ?", (run_fingerprint,)
        ).fetchone()
        return None if row is None else dict(json.loads(row["blob"]))

    # --- reviews ---------------------------------------------------------------

    def put_review(self, row: Mapping[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO reviews
                (result_key, run_fingerprint, source_id, source_hash, proposed_target_id,
                 decision, corrected_target_id, reviewer, review_note, reviewed_at)
            VALUES (:result_key, :run_fingerprint, :source_id, :source_hash,
                    :proposed_target_id, :decision, :corrected_target_id, :reviewer,
                    :review_note, :reviewed_at)
            """,
            dict(row),
        )

    def iter_reviews(self, run_fingerprint: str) -> Iterator[dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM reviews WHERE run_fingerprint = ? ORDER BY review_id",
            (run_fingerprint,),
        )
        for row in cursor:
            yield dict(row)

    def close(self) -> None:
        self._conn.close()

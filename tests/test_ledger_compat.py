"""A ledger written by xwalk 0.1.1 stays readable and resumable.

`tests/fixtures/ledger_v0_1_1/rundir` was produced by the released 0.1.1 code (commit
f737099) with `make_fixture.py` beside it. Tests always work on a copy: the fixture is
the evidence, and nothing here may change it.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import pytest

from tests.test_batch import make_matcher
from tests.test_matcher import ScriptedRetriever, score_reply, select_reply
from xwalk.batch import RunState, export_mapping_csv, run_batch
from xwalk.ledger import LEDGER_SCHEMA_VERSION, Ledger
from xwalk.llm.fake import FakeLLM
from xwalk.policy import MatchPolicy
from xwalk.records import MatchStatus, Record
from xwalk.review import adjudicated, review_history

FIXTURE = Path(__file__).parent / "fixtures" / "ledger_v0_1_1" / "rundir"
OLD_S1_HASH = "9830dddca39deb0d"
NEW_S1_HASH = "fa2af010a87d529d"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run_dir(tmp_path):
    before = digest(FIXTURE / "ledger.sqlite")
    target = tmp_path / "run"
    shutil.copytree(FIXTURE, target, ignore=shutil.ignore_patterns("*-wal", "*-shm"))
    yield target
    assert digest(FIXTURE / "ledger.sqlite") == before, "the preserved fixture was modified"


def test_the_fixture_really_is_a_0_1_1_ledger():
    import sqlite3

    # immutable: no locking files are created next to the preserved fixture.
    conn = sqlite3.connect(f"file:{FIXTURE / 'ledger.sqlite'}?immutable=1", uri=True)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "results" in tables and "ledger_meta" not in tables
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 4
    finally:
        conn.close()


def test_opening_upgrades_a_copy_and_keeps_the_original_as_a_backup(run_dir):
    original = digest(run_dir / "ledger.sqlite")
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        assert ledger.schema_version == LEDGER_SCHEMA_VERSION
        assert ledger.migrated_from == 1
        assert ledger.backup_path is not None
        assert digest(ledger.backup_path) == original
    finally:
        ledger.close()

    reopened = Ledger.open(run_dir / "ledger.sqlite")
    try:
        assert reopened.migrated_from is None  # upgraded once, not on every open
    finally:
        reopened.close()
    assert len(list(run_dir.glob("ledger.sqlite.v1-backup*"))) == 1


def test_the_current_view_has_one_row_per_source_and_all_rows_stay_in_history(run_dir):
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        current = list(ledger.iter_results("v011fp"))
        assert [(r.source_id, r.status) for r in current] == [
            ("s1", MatchStatus.MATCHED),
            ("s2", MatchStatus.NEEDS_REVIEW),
            ("s3", MatchStatus.FAILED),
        ]
        assert current[0].source_hash == NEW_S1_HASH
        history = list(ledger.iter_history("v011fp"))
        assert len(history) == 4
        old = [e for e in history if e.result.source_hash == OLD_S1_HASH]
        assert len(old) == 1 and not old[0].current
        assert ledger.duplicate_targets("v011fp") == {}
    finally:
        ledger.close()


def test_old_results_without_new_usage_fields_read_as_zero(run_dir):
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        for result in ledger.iter_results("v011fp"):
            assert result.usage.unknown_calls == 0 and result.usage.cache_hits == 0
    finally:
        ledger.close()


def test_review_overlays_survive_the_upgrade(run_dir, tmp_path):
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        rows = {row.source_id: row for row in adjudicated(ledger, "v011fp")}
        assert rows["s2"].reviewer == "jan"
        assert rows["s2"].final_status is MatchStatus.MATCHED
        assert rows["s2"].review_note == "sugar here means glucose-free syrup"
        export_mapping_csv(ledger, "v011fp", tmp_path / "mapping.csv")
    finally:
        ledger.close()
    exported = list(csv.DictReader((tmp_path / "mapping.csv").open(encoding="utf-8")))
    assert [r["source_id"] for r in exported] == ["s1", "s2", "s3"]


def fixture_handler(request):
    if "## Candidates" in request.user:
        return select_reply("C01")
    return score_reply(0.5 if "sugar" in request.user else 0.95)


SOURCES = [
    Record(id="s1", fields={"mention": "glucose", "note": "edited"}),
    Record(id="s2", fields={"mention": "sugar"}),
    Record(id="s3", fields={"mention": "unobtainium"}),
]


async def test_resuming_an_old_run_retries_only_the_failed_record_and_keeps_the_review(run_dir):
    retriever = ScriptedRetriever({"glucose": ["T1"], "sugar": ["T2"], "unobtainium": ["T3"]})
    matcher = make_matcher(
        FakeLLM(handler=fixture_handler),
        retriever,
        run_fp="v011fp",
        policy=MatchPolicy(max_attempts=1, verify_band=None),
    )
    report = await run_batch(matcher, SOURCES, out=run_dir)

    assert retriever.queries == ["unobtainium"]
    assert report.run_state is RunState.COMPLETE
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        rows = {row.source_id: row for row in adjudicated(ledger, "v011fp")}
        assert rows["s2"].reviewer == "jan"
        assert rows["s3"].model_status is MatchStatus.MATCHED
        assert [h.state for h in review_history(ledger, "v011fp")] == ["applied"]
    finally:
        ledger.close()


def test_inspecting_an_old_run_with_a_failed_row_exits_3(run_dir):
    """A 0.1.1 run has no recorded invocation, so its run state is unknown; its failed
    row must still make `inspect` report a failed run (CONTRACTS.md section 8)."""
    from xwalk import ops

    result = ops.inspect(run_dir)
    assert result.run is not None and result.run["run_state"] == "unknown"
    assert result.counts["failed"] == 1
    assert result.exit_code == ops.EXIT_RUNTIME

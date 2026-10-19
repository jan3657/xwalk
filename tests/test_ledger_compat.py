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


# --- read-only commands never migrate (review dx P4) ---------------------------------


def tree(path: Path) -> dict[str, str]:
    """Every file under `path` (names relative to it) with its digest."""
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def test_a_read_only_open_reads_an_old_ledger_without_touching_the_directory(run_dir):
    before = tree(run_dir)
    ledger = Ledger.open(run_dir / "ledger.sqlite", read_only=True)
    try:
        assert ledger.read_only
        assert ledger.migrated_from == 1 and ledger.backup_path is None
        assert {r.source_id for r in ledger.iter_results("v011fp")} == {"s1", "s2", "s3"}
        assert [h.state for h in review_history(ledger, "v011fp")] == ["applied"]
    finally:
        ledger.close()
    assert tree(run_dir) == before  # no upgrade, no backup, no -wal/-shm


def test_a_read_only_ledger_refuses_writes(run_dir, tmp_path):
    from xwalk.ledger import ReadOnlyLedgerError

    ledger = Ledger.open(run_dir / "ledger.sqlite", read_only=True)
    try:
        with pytest.raises(ReadOnlyLedgerError):
            ledger.put_manifest("v011fp", {})
    finally:
        ledger.close()

    current = tmp_path / "current" / "ledger.sqlite"
    Ledger.open(current).close()
    ledger = Ledger.open(current, read_only=True)
    try:
        assert ledger.schema_version == LEDGER_SCHEMA_VERSION
        with pytest.raises(ReadOnlyLedgerError):
            ledger.put_manifest("fp", {})
    finally:
        ledger.close()


def test_a_read_only_open_of_a_missing_ledger_creates_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Ledger.open(tmp_path / "absent" / "ledger.sqlite", read_only=True)
    assert not (tmp_path / "absent").exists()


def test_read_commands_leave_an_old_run_byte_identical(run_dir, tmp_path, capsys):
    from xwalk import ops
    from xwalk.cli.main import main

    before = tree(run_dir)
    ops.inspect(run_dir)
    ops.explain(run_dir, "s1")
    ops.list_results(run_dir)
    for view in ops.EXPORT_VIEWS:
        ops.export(run_dir, view, tmp_path / f"out-{view}")
    ops.review_export(run_dir, tmp_path / "review.csv")
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\ns2,T2\n", encoding="utf-8")
    assert main(["eval", "--run", str(run_dir), "--gold", str(gold)]) in (0, 1, 3)
    capsys.readouterr()
    assert tree(run_dir) == before


def test_upgrade_migrates_explicitly_and_keeps_a_backup(run_dir):
    from xwalk import ops

    original = digest(run_dir / "ledger.sqlite")
    result = ops.upgrade(run_dir)
    assert result.exit_code == 0
    backup = Path(result.artifacts["backup"])
    assert digest(backup) == original
    ledger = Ledger.open(run_dir / "ledger.sqlite", read_only=True)
    try:
        assert ledger.migrated_from is None and ledger.schema_version == LEDGER_SCHEMA_VERSION
    finally:
        ledger.close()
    again = ops.upgrade(run_dir)
    assert again.exit_code == 0 and "backup" not in again.artifacts
    assert len(list(run_dir.glob("ledger.sqlite.v1-backup*"))) == 1


def _current_ledger(directory: Path) -> Path:
    path = directory / "ledger.sqlite"
    writer = Ledger.open(path)
    writer.put_manifest("fp", {"job": "x"})
    writer.close()
    return path


def test_a_read_only_open_through_a_symlink_sees_an_active_writer(tmp_path):
    """The write-ahead log sits next to the real file, not next to the link."""
    real = tmp_path / "real" / "ledger.sqlite"
    writer = Ledger.open(real)
    try:
        writer.put_manifest("fp", {"job": "x"})
        assert real.with_name("ledger.sqlite-wal").stat().st_size > 0  # still in the log
        link_dir = tmp_path / "link"
        link_dir.mkdir()
        (link_dir / "ledger.sqlite").symlink_to(real)
        reader = Ledger.open(link_dir / "ledger.sqlite", read_only=True)
        try:
            assert reader.get_manifest("fp") == {"job": "x"}
        finally:
            reader.close()
    finally:
        writer.close()


def test_a_read_only_open_of_a_current_ledger_leaves_the_directory_alone(tmp_path):
    path = _current_ledger(tmp_path / "run")
    before = tree(path.parent)
    ledger = Ledger.open(path, read_only=True)
    try:
        assert ledger.get_manifest("fp") == {"job": "x"}
        assert ledger.schema_version == LEDGER_SCHEMA_VERSION
    finally:
        ledger.close()
    assert tree(path.parent) == before  # no -wal/-shm left behind


def _spy_connections(monkeypatch) -> list[str]:
    import sqlite3

    import xwalk.ledger as ledger_module

    opened: list[str] = []
    real = sqlite3.connect

    def connect(database, *args, **kwargs):
        opened.append(str(database))
        return real(database, *args, **kwargs)

    monkeypatch.setattr(ledger_module.sqlite3, "connect", connect)
    return opened


def test_read_only_opens_never_copy_a_ledger_into_memory(tmp_path, run_dir, monkeypatch):
    """Memory must not grow with the ledger: a 240 MiB run cost 312 MiB of RSS per
    reader when every read-only open copied it into RAM (review dx2 #1)."""
    current = _current_ledger(tmp_path / "current")
    writer = Ledger.open(tmp_path / "live" / "ledger.sqlite")
    writer.put_manifest("fp", {"job": "live"})
    opened = _spy_connections(monkeypatch)
    try:
        for path in (current, tmp_path / "live" / "ledger.sqlite", run_dir / "ledger.sqlite"):
            ledger = Ledger.open(path, read_only=True)
            try:
                assert ledger.get_manifest("fp") or ledger.get_manifest("v011fp")
            finally:
                ledger.close()
    finally:
        writer.close()
    assert opened and not any(":memory:" in target for target in opened), opened


def test_an_old_ledger_is_read_through_a_temporary_file_removed_on_close(run_dir):
    ledger = Ledger.open(run_dir / "ledger.sqlite", read_only=True)
    snapshot = ledger.snapshot_path
    try:
        assert snapshot is not None and snapshot.is_file()
        assert not snapshot.is_relative_to(run_dir)
    finally:
        ledger.close()
    assert not snapshot.exists()


def test_a_read_only_open_works_on_read_only_files_and_directories(tmp_path):
    import os
    import stat

    path = _current_ledger(tmp_path / "run")
    path.chmod(0o444)
    path.parent.chmod(0o555)
    try:
        if os.access(path.parent, os.W_OK):  # pragma: no cover - running as root
            pytest.skip("permissions are not enforced for this user")
        ledger = Ledger.open(path, read_only=True)
        try:
            assert ledger.get_manifest("fp") == {"job": "x"}
            assert ledger.count("fp") == 0
        finally:
            ledger.close()
    finally:
        path.parent.chmod(stat.S_IRWXU)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_the_upgrade_command_reports_the_backup(run_dir, capsys):
    import json

    from xwalk.cli.main import main

    assert main(["upgrade", "--run", str(run_dir), "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["operation"] == "upgrade"
    assert envelope["data"]["migrated_from"] == 1
    assert Path(envelope["artifacts"]["backup"]).is_file()

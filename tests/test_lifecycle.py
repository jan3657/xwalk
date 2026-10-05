"""Run lifecycle: source snapshots, the current view, resume, abort and interruption.

The contract lives in docs/claude-upgrade/CONTRACTS.md sections 2-4. Every test here
drives the real `run_batch` and `Ledger` with FakeLLM and a scripted retriever.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest

from tests.test_batch import RETRIEVER, SOURCES, make_matcher, two_good_matches
from tests.test_matcher import ScriptedRetriever, score_reply, select_reply
from xwalk.batch import DuplicateSourceIdError, RunState, run_batch
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMFatalError, LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.policy import MatchPolicy
from xwalk.records import MatchStatus, Record

THREE = [
    Record(id="s1", fields={"mention": "glucose"}),
    Record(id="s2", fields={"mention": "fructose"}),
    Record(id="s3", fields={"mention": "sucrose"}),
]
RETRIEVER3 = {**RETRIEVER, "sucrose": ["T3"]}


def mapping_rows(out: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))


def manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def comparable(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Everything but wall-clock time, which no two runs share."""
    return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in rows]


def history(out: Path, run_fp: str = "fp1"):
    ledger = Ledger.open(out / "ledger.sqlite")
    try:
        return list(ledger.iter_history(run_fp))
    finally:
        ledger.close()


# --- snapshot and current view ----------------------------------------------------


async def test_an_edited_source_has_exactly_one_current_row_after_resume(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    edited = [Record(id="s1", fields={"mention": "glucose", "note": "new"}), SOURCES[1]]
    report = await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), edited, out=out
    )

    assert [r["source_id"] for r in mapping_rows(out)] == ["s1", "s2"]
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["source_id"] for line in lines] == ["s1", "s2"]
    assert report.total == 2
    assert report.by_status() == {MatchStatus.MATCHED: 2}
    assert manifest(out)["counts"] == {"matched": 2}


async def test_the_superseded_source_version_stays_in_history(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    edited = [Record(id="s1", fields={"mention": "glucose", "note": "new"}), SOURCES[1]]
    await run_batch(make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), edited, out=out)

    entries = [e for e in history(out) if e.result.source_id == "s1"]
    assert len(entries) == 2
    assert sorted(e.current for e in entries) == [False, True]


async def test_duplicate_targets_ignore_superseded_versions(tmp_path):
    """Both s1 versions chose T1; only one of them is current, so there is no conflict."""
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    edited = [Record(id="s1", fields={"mention": "glucose", "note": "new"}), SOURCES[1]]
    report = await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), edited, out=out
    )
    assert report.duplicate_targets() == {}
    assert manifest(out)["duplicate_targets"] == {}


async def test_a_removed_source_leaves_the_current_view_but_not_history(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    retriever = ScriptedRetriever(RETRIEVER)
    report = await run_batch(make_matcher(two_good_matches(), retriever), [SOURCES[1]], out=out)

    assert retriever.queries == []  # s2 was already done; s1 is gone, not re-run
    assert [r["source_id"] for r in mapping_rows(out)] == ["s2"]
    assert report.total == 1
    assert manifest(out)["removed_sources"] == 1
    assert {e.result.source_id for e in history(out)} == {"s1", "s2"}


async def test_limit_processes_the_first_unfinished_records_and_leaves_the_run_partial(
    tmp_path,
):
    out = tmp_path / "run"
    retriever = ScriptedRetriever(RETRIEVER3)
    report = await run_batch(make_matcher(two_good_matches(), retriever), THREE, out=out, limit=1)

    assert retriever.queries == ["glucose"]
    assert report.run_state is RunState.PARTIAL
    assert report.total == 3 and report.pending == 2
    rows = mapping_rows(out)
    assert [(r["source_id"], r["status"]) for r in rows] == [
        ("s1", "matched"),
        ("s2", "pending"),
        ("s3", "pending"),
    ]
    assert manifest(out)["run_state"] == "partial"
    assert manifest(out)["counts"] == {"matched": 1, "pending": 2}


async def test_a_second_limited_run_continues_where_the_first_stopped(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER3)), THREE, out=out, limit=1
    )
    retriever = ScriptedRetriever(RETRIEVER3)
    report = await run_batch(make_matcher(two_good_matches(), retriever), THREE, out=out, limit=1)
    assert retriever.queries == ["fructose"]
    assert report.pending == 1

    retriever = ScriptedRetriever(RETRIEVER3)
    report = await run_batch(make_matcher(two_good_matches(), retriever), THREE, out=out)
    assert retriever.queries == ["sucrose"]
    assert report.run_state is RunState.COMPLETE


async def test_duplicate_source_ids_in_one_snapshot_are_rejected(tmp_path):
    twice = [SOURCES[0], Record(id="s1", fields={"mention": "other"})]
    with pytest.raises(DuplicateSourceIdError, match="s1"):
        await run_batch(
            make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), twice, out=tmp_path
        )


def failing_for(mention: str, error: BaseException):
    def handler(request):
        if mention in request.user:
            raise error
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.95)

    return FakeLLM(handler=handler)


async def test_failed_records_make_the_run_failed_not_complete(tmp_path):
    out = tmp_path / "run"
    matcher = make_matcher(
        failing_for("sucrose", LLMRetryableError("busy")),
        ScriptedRetriever(RETRIEVER3),
        policy=MatchPolicy(max_attempts=1),
    )
    report = await run_batch(matcher, THREE, out=out)
    assert report.by_status()[MatchStatus.FAILED] == 1
    assert report.run_state is RunState.FAILED
    assert manifest(out)["run_state"] == "failed"


async def test_a_failed_record_is_retried_on_resume_and_the_failure_kept_in_history(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(
            failing_for("sucrose", LLMRetryableError("busy")),
            ScriptedRetriever(RETRIEVER3),
            policy=MatchPolicy(max_attempts=1),
        ),
        THREE,
        out=out,
    )
    retriever = ScriptedRetriever(RETRIEVER3)
    report = await run_batch(make_matcher(two_good_matches(), retriever), THREE, out=out)

    assert retriever.queries == ["sucrose"]
    assert report.run_state is RunState.COMPLETE
    assert [r["status"] for r in mapping_rows(out)] == ["matched"] * 3
    s3 = [e for e in history(out) if e.result.source_id == "s3"]
    assert [(e.result.status, e.current) for e in s3] == [
        (MatchStatus.FAILED, False),
        (MatchStatus.MATCHED, True),
    ]


async def test_no_resume_recomputes_and_keeps_the_earlier_results_in_history(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    retriever = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), retriever), SOURCES, out=out, resume=False)

    assert len(retriever.queries) == 2
    assert len(mapping_rows(out)) == 2
    entries = history(out)
    assert len(entries) == 4
    assert sum(e.current for e in entries) == 2


async def test_an_unchanged_complete_run_repeats_with_zero_calls_and_identical_exports(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    first = (out / "mapping.csv").read_text(encoding="utf-8")
    llm = two_good_matches()
    report = await run_batch(make_matcher(llm, ScriptedRetriever(RETRIEVER)), SOURCES, out=out)
    assert llm.requests == []
    assert report.usage.calls == 0
    assert (out / "mapping.csv").read_text(encoding="utf-8") == first


# --- usage in exports ---------------------------------------------------------------


async def test_unknown_usage_is_visible_in_the_mapping_and_the_manifest(tmp_path):
    out = tmp_path / "run"
    matcher = make_matcher(
        failing_for("sucrose", LLMRetryableError("busy")),
        ScriptedRetriever(RETRIEVER3),
        policy=MatchPolicy(max_attempts=1),
    )
    await run_batch(matcher, THREE, out=out)
    rows = {r["source_id"]: r for r in mapping_rows(out)}
    assert rows["s3"]["unknown_calls"] == "1"
    assert rows["s1"]["unknown_calls"] == "0"
    assert rows["s1"]["cache_hits"] == "0"
    usage = manifest(out)["usage"]
    assert usage["unknown_calls"] == 1
    assert "unknown usage" in usage["tokens"]


# --- fatal provider failure (abort) -------------------------------------------------


async def test_a_fatal_provider_failure_is_not_committed_and_aborts_the_run(tmp_path):
    out = tmp_path / "run"
    matcher = make_matcher(
        failing_for("fructose", LLMFatalError("invalid api key")),
        ScriptedRetriever(RETRIEVER3),
        policy=MatchPolicy(concurrency=1),
    )
    report = await run_batch(matcher, THREE, out=out)

    assert report.run_state is RunState.ABORTED
    assert [(e.code, e.source_id) for e in report.errors] == [("fatal_provider_failure", "s2")]
    assert "invalid api key" in report.errors[0].message
    assert {e.result.source_id for e in history(out)} == {"s1"}  # s2 never committed
    assert manifest(out)["run_state"] == "aborted"
    assert manifest(out)["errors"][0]["source_id"] == "s2"


async def test_a_fatal_failure_stops_scheduling_further_records(tmp_path):
    many = [Record(id=f"s{i}", fields={"mention": "fructose"}) for i in range(10)]
    retriever = ScriptedRetriever(RETRIEVER)
    matcher = make_matcher(
        failing_for("fructose", LLMFatalError("model not found")),
        retriever,
        policy=MatchPolicy(concurrency=2),
    )
    report = await run_batch(matcher, many, out=tmp_path / "run")
    assert report.run_state is RunState.ABORTED
    assert len(retriever.queries) <= 2


async def test_resume_after_an_abort_retries_the_fatal_record(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(
            failing_for("fructose", LLMFatalError("invalid api key")),
            ScriptedRetriever(RETRIEVER3),
            policy=MatchPolicy(concurrency=1),
        ),
        THREE,
        out=out,
    )
    retriever = ScriptedRetriever(RETRIEVER3)
    report = await run_batch(make_matcher(two_good_matches(), retriever), THREE, out=out)
    assert retriever.queries == ["fructose", "sucrose"]
    assert report.run_state is RunState.COMPLETE


class GatedRetriever(ScriptedRetriever):
    """Holds one query open until released, recording whether it was cancelled."""

    def __init__(self, by_query, slow: str):
        super().__init__(by_query)
        self.slow = slow
        self.started = asyncio.Event()
        self.cancelled = False

    async def search(self, request):
        if request.text == self.slow:
            self.started.set()
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().search(request)


class SpyLedger(Ledger):
    """Records every result write and whether the ledger was already closed."""

    events: list[str] = []

    async def put_result(self, result):
        SpyLedger.events.append("write-after-close" if self.closed else "write")
        await super().put_result(result)

    def close(self):
        SpyLedger.events.append("close")
        super().close()


@pytest.fixture
def spy_ledger(monkeypatch):
    SpyLedger.events = []
    monkeypatch.setattr("xwalk.batch.Ledger", SpyLedger)
    return SpyLedger.events


async def test_one_task_failing_cancels_its_sibling_and_nothing_writes_after_close(
    tmp_path, spy_ledger
):
    retriever = GatedRetriever(RETRIEVER, slow="glucose")

    def handler(request):
        if "fructose" in request.user:
            raise RuntimeError("programming error")
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.95)

    matcher = make_matcher(FakeLLM(handler=handler), retriever, policy=MatchPolicy(concurrency=2))
    with pytest.raises(RuntimeError, match="programming error"):
        await run_batch(matcher, SOURCES, out=tmp_path / "run")

    await asyncio.sleep(0.3)  # long enough for an orphaned sibling to finish and write
    assert retriever.cancelled
    assert "write-after-close" not in spy_ledger
    assert spy_ledger[-1] == "close"
    assert manifest(tmp_path / "run")["run_state"] == "aborted"
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert others == []


async def test_a_fatal_failure_cancels_in_flight_siblings_before_closing(tmp_path, spy_ledger):
    retriever = GatedRetriever(RETRIEVER, slow="glucose")
    matcher = make_matcher(
        failing_for("fructose", LLMFatalError("invalid api key")),
        retriever,
        policy=MatchPolicy(concurrency=2),
    )
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    await asyncio.sleep(0.3)
    assert report.run_state is RunState.ABORTED
    assert retriever.cancelled
    assert spy_ledger == ["close"]  # neither record was committed


# --- interruption -------------------------------------------------------------------


async def clean_run(tmp_path: Path) -> list[dict[str, str]]:
    out = tmp_path / "clean"
    await run_batch(
        make_matcher(
            two_good_matches(), ScriptedRetriever(RETRIEVER3), policy=MatchPolicy(concurrency=1)
        ),
        THREE,
        out=out,
    )
    return comparable(mapping_rows(out))


async def resume_to_completion(out: Path) -> list[dict[str, str]]:
    report = await run_batch(
        make_matcher(
            two_good_matches(), ScriptedRetriever(RETRIEVER3), policy=MatchPolicy(concurrency=1)
        ),
        THREE,
        out=out,
    )
    assert report.run_state is RunState.COMPLETE
    return comparable(mapping_rows(out))


async def test_cancelling_a_run_records_it_as_interrupted(tmp_path, spy_ledger):
    out = tmp_path / "run"
    holder: dict[str, asyncio.Task] = {}

    def progress(result):
        holder["task"].cancel()  # Ctrl-C arrives right after the first commit

    matcher = make_matcher(
        two_good_matches(), ScriptedRetriever(RETRIEVER3), policy=MatchPolicy(concurrency=1)
    )
    holder["task"] = asyncio.create_task(run_batch(matcher, THREE, out=out, progress=progress))
    with pytest.raises(asyncio.CancelledError):
        await holder["task"]

    assert manifest(out)["run_state"] == "interrupted"
    assert spy_ledger[-1] == "close" and "write-after-close" not in spy_ledger
    # The source was never read to the end, so there is no snapshot to report pending
    # rows against: the export shows what was committed.
    assert [r["source_id"] for r in mapping_rows(out)] == ["s1"]


@pytest.mark.parametrize("boundary", ["after_first_commit", "before_snapshot", "inside_commit"])
async def test_interrupt_then_resume_equals_a_clean_run(tmp_path, monkeypatch, boundary):
    expected = await clean_run(tmp_path)
    out = tmp_path / "run"
    holder: dict[str, asyncio.Task] = {}
    progress = None

    if boundary == "after_first_commit":

        def progress(result):
            holder["task"].cancel()

    elif boundary == "before_snapshot":

        def put_snapshot(self, *args, **kwargs):
            raise KeyboardInterruptLike("interrupted before the snapshot commit")

        monkeypatch.setattr(Ledger, "put_snapshot", put_snapshot)

    else:  # inside_commit: the second result transaction dies after it has begun
        monkeypatch.setattr("xwalk.batch.Ledger", ExplodingLedger)
        ExplodingLedger.inserts = 0

    matcher = make_matcher(
        two_good_matches(), ScriptedRetriever(RETRIEVER3), policy=MatchPolicy(concurrency=1)
    )
    holder["task"] = asyncio.create_task(run_batch(matcher, THREE, out=out, progress=progress))
    with pytest.raises((asyncio.CancelledError, KeyboardInterruptLike)):
        await holder["task"]
    monkeypatch.undo()

    # The cleanup after the failure could still write: the ledger was left usable.
    expected_state = "interrupted" if boundary == "after_first_commit" else "aborted"
    assert manifest(out)["run_state"] == expected_state
    assert await resume_to_completion(out) == expected
    # No half-written transaction: one current row and one history entry per source.
    assert len(history(out)) == 3


class KeyboardInterruptLike(Exception):
    """An abrupt failure at a chosen point. Not a real KeyboardInterrupt, which pytest
    would treat as the user stopping the test session."""


class _ExplodingConnection:
    """Forwards to a real connection, but fails the COMMIT of the second result write,
    after its INSERT has already run inside the open transaction."""

    def __init__(self, conn, owner):
        self._conn = conn
        self._owner = owner
        self._armed = False

    def execute(self, sql, *args):
        if "INSERT INTO results" in sql:
            type(self._owner).inserts += 1
            self._armed = type(self._owner).inserts == 2
        elif self._armed and sql.strip().upper().startswith("COMMIT"):
            self._armed = False
            raise KeyboardInterruptLike("disk vanished mid-transaction")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class ExplodingLedger(Ledger):
    inserts = 0

    def __init__(self, connection, **kwargs):
        super().__init__(_ExplodingConnection(connection, self), **kwargs)

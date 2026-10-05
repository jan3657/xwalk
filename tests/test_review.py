import csv

import pytest

from tests.test_serde import sample_result
from xwalk.ledger import Ledger
from xwalk.records import DecisionReason, MatchResult, MatchStatus
from xwalk.review import (
    SnapshotMismatch,
    adjudicated,
    apply_review,
    export_review,
    read_review,
    review_history,
)


def needs_review(**kwargs) -> MatchResult:
    base = sample_result()
    return MatchResult(
        **{
            **base.__dict__,
            "status": MatchStatus.NEEDS_REVIEW,
            "reason": DecisionReason.BELOW_ACCEPT_THRESHOLD,
            **kwargs,
        }
    )


@pytest.fixture
async def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    led.put_manifest("fp1", {"target_fingerprint": "tf1"})
    await led.put_result(needs_review())
    yield led
    led.close()


# --- export ---------------------------------------------------------------------


async def test_export_writes_one_row_per_reviewable_result(ledger, tmp_path):
    out = tmp_path / "review.csv"
    assert export_review(ledger, "fp1", out) == 1
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1


async def test_export_carries_the_identity_columns(ledger, tmp_path):
    out = tmp_path / "review.csv"
    export_review(ledger, "fp1", out)
    row = next(iter(csv.DictReader(out.open(encoding="utf-8"))))
    for column in (
        "result_key",
        "run_fingerprint",
        "source_id",
        "source_hash",
        "proposed_target_id",
        "decision",
        "corrected_target_id",
        "reviewer",
        "review_note",
        "reviewed_at",
    ):
        assert column in row


async def test_export_leaves_the_decision_columns_blank_for_the_reviewer(ledger, tmp_path):
    out = tmp_path / "review.csv"
    export_review(ledger, "fp1", out)
    row = next(iter(csv.DictReader(out.open(encoding="utf-8"))))
    assert row["decision"] == "" and row["corrected_target_id"] == ""


async def test_export_only_includes_the_requested_statuses(ledger, tmp_path):
    await ledger.put_result(
        needs_review(
            result_key="rk2",
            source_id="s2",
            status=MatchStatus.MATCHED,
            reason=DecisionReason.ACCEPT_THRESHOLD,
        )
    )
    out = tmp_path / "review.csv"
    assert export_review(ledger, "fp1", out) == 1


async def test_export_can_include_matched_rows_for_spot_checking(ledger, tmp_path):
    await ledger.put_result(
        needs_review(
            result_key="rk2",
            source_id="s2",
            status=MatchStatus.MATCHED,
            reason=DecisionReason.ACCEPT_THRESHOLD,
        )
    )
    out = tmp_path / "review.csv"
    count = export_review(
        ledger, "fp1", out, statuses=(MatchStatus.NEEDS_REVIEW, MatchStatus.MATCHED)
    )
    assert count == 2


# --- apply ----------------------------------------------------------------------


def write_review(path, **overrides) -> None:
    row = {
        "result_key": "rk1",
        "run_fingerprint": "fp1",
        "source_id": "s1",
        "source_hash": "sh1",
        "proposed_target_id": "T1",
        "decision": "accept",
        "corrected_target_id": "",
        "reviewer": "jan",
        "review_note": "",
        "reviewed_at": "2026-07-26T10:00:00Z",
    }
    row.update(overrides)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


async def test_apply_accepts_a_decision(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path)
    report = apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    assert report.applied == 1 and report.rejected == []


async def test_apply_never_mutates_the_original_result(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="reject")
    before = ledger.get_result("rk1")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    assert ledger.get_result("rk1") == before


async def test_apply_records_the_decision_in_the_overlay(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="T9")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    stored = list(ledger.iter_reviews("fp1"))
    assert stored[0]["decision"] == "replace" and stored[0]["corrected_target_id"] == "T9"


async def test_apply_rejects_a_stale_source_hash(ledger, tmp_path):
    """A decision made against different data is not a decision about this data."""
    path = tmp_path / "r.csv"
    write_review(path, source_hash="DIFFERENT")
    with pytest.raises(SnapshotMismatch, match="source"):
        apply_review(ledger, read_review(path), target_store_fingerprint="tf1")


async def test_apply_rejects_a_stale_target_snapshot(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path)
    with pytest.raises(SnapshotMismatch, match="target"):
        apply_review(ledger, read_review(path), target_store_fingerprint="DIFFERENT")


async def test_apply_rejects_an_unknown_result_key(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, result_key="nope")
    with pytest.raises(SnapshotMismatch, match="result_key"):
        apply_review(ledger, read_review(path), target_store_fingerprint="tf1")


async def test_apply_rejects_an_unknown_decision(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="maybe")
    with pytest.raises(ValueError, match="decision"):
        read_review(path)


async def test_apply_requires_a_corrected_id_for_replace(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="")
    with pytest.raises(ValueError, match="corrected_target_id"):
        read_review(path)


async def test_apply_skips_blank_decisions_without_failing(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="")
    rows = read_review(path)
    assert rows == []


async def test_apply_requires_a_reviewer(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, reviewer="")
    with pytest.raises(ValueError, match="reviewer"):
        read_review(path)


# --- adjudicated view -----------------------------------------------------------


async def test_adjudicated_accept_keeps_the_model_target(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="accept")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_target_id == "T1" and row.final_status is MatchStatus.MATCHED


async def test_adjudicated_reject_clears_the_target(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="reject")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_target_id is None and row.final_status is MatchStatus.UNMATCHED


async def test_adjudicated_replace_uses_the_corrected_target(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="T9")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_target_id == "T9" and row.final_status is MatchStatus.MATCHED


async def test_adjudicated_defer_leaves_the_row_in_review(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="defer")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_status is MatchStatus.NEEDS_REVIEW


async def test_adjudicated_preserves_the_model_answer_alongside_the_final_one(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="T9")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.model_target_id == "T1" and row.final_target_id == "T9"


async def test_the_latest_review_wins_when_a_row_is_reviewed_twice(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="accept")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    write_review(path, decision="reject")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_status is MatchStatus.UNMATCHED
    assert len(list(ledger.iter_reviews("fp1"))) == 2  # both decisions retained


async def test_unreviewed_rows_appear_with_their_model_status(ledger, tmp_path):
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.reviewer is None and row.final_status is MatchStatus.NEEDS_REVIEW


# --- reviews across resume ----------------------------------------------------------


def _review_everything(ledger, run_fp, decision="accept"):
    from xwalk.review import ReviewDecision, ReviewRow

    rows = [
        ReviewRow(
            result_key=r.result_key,
            run_fingerprint=run_fp,
            source_id=r.source_id,
            source_hash=r.source_hash,
            proposed_target_id=r.matched_id,
            decision=ReviewDecision(decision),
            corrected_target_id=None,
            reviewer="jan",
            review_note="checked",
            reviewed_at="2026-10-05T10:00:00Z",
        )
        for r in ledger.iter_results(run_fp)
    ]
    from tests.test_matcher import STORE

    apply_review(ledger, rows, target_store_fingerprint=STORE.fingerprint)


def _uncertain():
    from tests.test_matcher import score_reply, select_reply
    from xwalk.llm.fake import FakeLLM

    def handler(request):
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.5)

    return FakeLLM(handler=handler)


async def _run(out, sources, **kwargs):
    from tests.test_batch import RETRIEVER, make_matcher
    from tests.test_matcher import ScriptedRetriever
    from xwalk.batch import run_batch

    return await run_batch(
        make_matcher(_uncertain(), ScriptedRetriever(RETRIEVER)), sources, out=out, **kwargs
    )


async def _reviewed_run(tmp_path):
    from tests.test_batch import SOURCES

    out = tmp_path / "run"
    await _run(out, SOURCES)
    led = Ledger.open(out / "ledger.sqlite")
    _review_everything(led, "fp1")
    led.close()
    return out


def _adjudicated(out):
    led = Ledger.open(out / "ledger.sqlite")
    try:
        rows = {row.source_id: row for row in adjudicated(led, "fp1")}
        history = list(review_history(led, "fp1"))
        return rows, history
    finally:
        led.close()


async def test_a_review_still_applies_after_an_unchanged_resume(tmp_path):
    from tests.test_batch import SOURCES

    out = await _reviewed_run(tmp_path)
    await _run(out, SOURCES)
    rows, history = _adjudicated(out)
    assert rows["s1"].reviewer == "jan" and rows["s1"].final_status is MatchStatus.MATCHED
    assert [h.state for h in history] == ["applied", "applied"]


async def test_a_review_of_an_edited_source_is_stale_not_applied(tmp_path):
    from tests.test_batch import SOURCES
    from xwalk.records import Record

    out = await _reviewed_run(tmp_path)
    await _run(out, [Record(id="s1", fields={"mention": "glucose", "x": 1}), SOURCES[1]])
    rows, history = _adjudicated(out)
    assert rows["s1"].reviewer is None
    assert rows["s1"].final_status is MatchStatus.NEEDS_REVIEW
    assert rows["s2"].reviewer == "jan"
    states = {h.review["source_id"]: h.state for h in history}
    assert states == {"s1": "stale", "s2": "applied"}


async def test_a_review_of_a_recomputed_result_is_stale_not_applied(tmp_path):
    from tests.test_batch import SOURCES

    out = await _reviewed_run(tmp_path)
    await _run(out, SOURCES, resume=False)
    rows, history = _adjudicated(out)
    assert rows["s1"].reviewer is None
    assert [h.state for h in history] == ["stale", "stale"]


async def test_reviewing_a_result_that_is_no_longer_current_is_refused(tmp_path):
    from tests.test_batch import SOURCES
    from xwalk.records import Record

    out = tmp_path / "run"
    await _run(out, SOURCES)
    led = Ledger.open(out / "ledger.sqlite")
    stale_csv = tmp_path / "review.csv"
    export_review(led, "fp1", stale_csv)
    led.close()

    await _run(out, [Record(id="s1", fields={"mention": "glucose", "x": 1}), SOURCES[1]])
    rows = [
        {**row, "decision": "accept", "reviewer": "jan"}
        for row in csv.DictReader(stale_csv.open(encoding="utf-8"))
    ]
    with stale_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from tests.test_matcher import STORE

    led = Ledger.open(out / "ledger.sqlite")
    try:
        with pytest.raises(SnapshotMismatch, match="no longer current"):
            apply_review(led, read_review(stale_csv), target_store_fingerprint=STORE.fingerprint)
        assert list(led.iter_reviews("fp1")) == []  # all-or-nothing
    finally:
        led.close()

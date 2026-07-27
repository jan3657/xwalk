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

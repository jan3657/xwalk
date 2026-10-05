"""Human review as an immutable overlay.

Three layers are preserved and never collapsed: the model result (in `results`), the
reviewer decision (in `reviews`), and the adjudicated view derived from both.

A review is bound to the result it was made against: its `result_key` and the revision
of that result. It applies to the adjudicated view only while that exact result is
current. When the source changes, the source is removed, or the result is recomputed,
the review stays in the ledger and `review_history` reports it as stale.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from xwalk.ledger import Ledger
from xwalk.records import MatchStatus

COLUMNS = (
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
)


class SnapshotMismatch(Exception):
    """A review was made against data that no longer matches this run."""


class ReviewDecision(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REPLACE = "replace"
    NO_MATCH = "no_match"
    DEFER = "defer"


@dataclass(frozen=True)
class ReviewRow:
    result_key: str
    run_fingerprint: str
    source_id: str
    source_hash: str
    proposed_target_id: str | None
    decision: ReviewDecision
    corrected_target_id: str | None
    reviewer: str
    review_note: str
    reviewed_at: str


@dataclass(frozen=True)
class ApplyReport:
    applied: int
    rejected: list[tuple[str, str]]


@dataclass(frozen=True)
class ReviewHistoryEntry:
    """A stored decision and whether it shapes the current adjudicated view.

    `state` is ``applied`` (the latest decision on a current result), ``superseded``
    (a later decision on the same result replaced it) or ``stale`` (its result is no
    longer current: the source changed or was removed, or the result was recomputed).
    """

    review: dict[str, object]
    state: str


@dataclass(frozen=True)
class AdjudicatedResult:
    result_key: str
    source_id: str
    model_target_id: str | None
    model_status: MatchStatus
    confidence: float | None
    final_target_id: str | None
    final_status: MatchStatus
    reviewer: str | None
    review_note: str
    reviewed_at: str | None


def export_review(
    ledger: Ledger,
    run_fingerprint: str,
    out_path: str | Path,
    *,
    statuses: Sequence[MatchStatus] = (MatchStatus.NEEDS_REVIEW,),
) -> int:
    """Write a CSV with identity columns filled and decision columns blank."""
    wanted = set(statuses)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for result in ledger.iter_results(run_fingerprint):
            if result.status not in wanted:
                continue
            writer.writerow(
                {
                    "result_key": result.result_key,
                    "run_fingerprint": result.run_fingerprint,
                    "source_id": result.source_id,
                    "source_hash": result.source_hash,
                    "proposed_target_id": result.matched_id or "",
                    "decision": "",
                    "corrected_target_id": "",
                    "reviewer": "",
                    "review_note": "",
                    "reviewed_at": "",
                }
            )
            written += 1
    return written


def read_review(path: str | Path) -> list[ReviewRow]:
    """Parse a reviewed CSV. Blank decisions are skipped; bad ones raise."""
    rows: list[ReviewRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_no, raw in enumerate(csv.DictReader(handle), start=2):
            decision_text = (raw.get("decision") or "").strip().lower()
            if not decision_text:
                continue
            try:
                decision = ReviewDecision(decision_text)
            except ValueError as exc:
                raise ValueError(
                    f"row {line_no}: unknown decision {decision_text!r}; "
                    f"expected one of {[d.value for d in ReviewDecision]}"
                ) from exc

            corrected = (raw.get("corrected_target_id") or "").strip() or None
            if decision is ReviewDecision.REPLACE and not corrected:
                raise ValueError(f"row {line_no}: decision 'replace' needs corrected_target_id")

            reviewer = (raw.get("reviewer") or "").strip()
            if not reviewer:
                raise ValueError(f"row {line_no}: reviewer must be set for an applied decision")

            rows.append(
                ReviewRow(
                    result_key=(raw.get("result_key") or "").strip(),
                    run_fingerprint=(raw.get("run_fingerprint") or "").strip(),
                    source_id=(raw.get("source_id") or "").strip(),
                    source_hash=(raw.get("source_hash") or "").strip(),
                    proposed_target_id=(raw.get("proposed_target_id") or "").strip() or None,
                    decision=decision,
                    corrected_target_id=corrected,
                    reviewer=reviewer,
                    review_note=(raw.get("review_note") or "").strip(),
                    reviewed_at=(raw.get("reviewed_at") or "").strip(),
                )
            )
    return rows


def apply_review(
    ledger: Ledger,
    rows: Sequence[ReviewRow],
    *,
    target_store_fingerprint: str,
) -> ApplyReport:
    """Validate every row against the originating run, then append to the overlay.

    Validation is all-or-nothing: one stale row fails the whole application, because a
    partially-applied review file is worse than an unapplied one.
    """
    keys_by_run: dict[str, dict[str, int]] = {}

    def current_keys(run_fingerprint: str) -> dict[str, int]:
        if run_fingerprint not in keys_by_run:
            keys_by_run[run_fingerprint] = ledger.current_keys(run_fingerprint)
        return keys_by_run[run_fingerprint]

    for row in rows:
        result = ledger.get_result(row.result_key)
        if result is None:
            raise SnapshotMismatch(f"unknown result_key {row.result_key!r}")
        if result.source_hash != row.source_hash:
            raise SnapshotMismatch(
                f"{row.result_key}: source record changed since the run "
                f"({row.source_hash!r} -> {result.source_hash!r}); re-run before applying"
            )
        if row.result_key not in current_keys(row.run_fingerprint):
            raise SnapshotMismatch(
                f"{row.result_key}: the result for source {row.source_id!r} is no longer "
                f"current (the source changed or was removed); export the review again"
            )
        if row.proposed_target_id != result.matched_id:
            raise SnapshotMismatch(
                f"{row.result_key}: the result is no longer current: it was recomputed and "
                f"now proposes {result.matched_id!r}, not {row.proposed_target_id!r}; "
                f"export the review again"
            )
        manifest = ledger.get_manifest(row.run_fingerprint) or {}
        recorded_target = manifest.get("target_fingerprint")
        if recorded_target is not None and recorded_target != target_store_fingerprint:
            raise SnapshotMismatch(
                f"{row.result_key}: target snapshot changed since the run "
                f"({recorded_target!r} -> {target_store_fingerprint!r}); re-run before applying"
            )

    for row in rows:
        ledger.put_review(
            {
                "result_key": row.result_key,
                "run_fingerprint": row.run_fingerprint,
                "source_id": row.source_id,
                "source_hash": row.source_hash,
                "proposed_target_id": row.proposed_target_id,
                "decision": row.decision.value,
                "corrected_target_id": row.corrected_target_id,
                "reviewer": row.reviewer,
                "review_note": row.review_note,
                "reviewed_at": row.reviewed_at,
                "result_revision": current_keys(row.run_fingerprint)[row.result_key],
            }
        )
    return ApplyReport(applied=len(rows), rejected=[])


def _applicable_reviews(ledger: Ledger, run_fingerprint: str) -> dict[str, dict[str, object]]:
    """result_key -> the latest review made against the current revision of that result."""
    current = ledger.current_keys(run_fingerprint)
    latest: dict[str, dict[str, object]] = {}
    for review in ledger.iter_reviews(run_fingerprint):
        key = str(review["result_key"])
        if current.get(key) == review["result_revision"]:
            latest[key] = review  # later rows overwrite earlier ones
    return latest


def review_history(ledger: Ledger, run_fingerprint: str) -> Iterator[ReviewHistoryEntry]:
    """Every stored decision, in the order applied, with its effect on the current view."""
    winners = {
        int(str(review["review_id"]))
        for review in _applicable_reviews(ledger, run_fingerprint).values()
    }
    current = ledger.current_keys(run_fingerprint)
    for review in ledger.iter_reviews(run_fingerprint):
        if int(review["review_id"]) in winners:
            state = "applied"
        elif current.get(str(review["result_key"])) == review["result_revision"]:
            state = "superseded"
        else:
            state = "stale"
        yield ReviewHistoryEntry(review=review, state=state)


def adjudicated(ledger: Ledger, run_fingerprint: str) -> Iterator[AdjudicatedResult]:
    """The derived view over current results: model answer plus the latest reviewer
    decision made against that exact result, if any."""
    latest = _applicable_reviews(ledger, run_fingerprint)

    for result in ledger.iter_results(run_fingerprint):
        decision_row = latest.get(result.result_key)
        if decision_row is None:
            yield AdjudicatedResult(
                result_key=result.result_key,
                source_id=result.source_id,
                model_target_id=result.matched_id,
                model_status=result.status,
                confidence=result.confidence,
                final_target_id=result.matched_id,
                final_status=result.status,
                reviewer=None,
                review_note="",
                reviewed_at=None,
            )
            continue

        decision = ReviewDecision(str(decision_row["decision"]))
        if decision is ReviewDecision.ACCEPT:
            final_id, final_status = result.matched_id, MatchStatus.MATCHED
        elif decision is ReviewDecision.REPLACE:
            final_id = str(decision_row["corrected_target_id"])
            final_status = MatchStatus.MATCHED
        elif decision in (ReviewDecision.REJECT, ReviewDecision.NO_MATCH):
            final_id, final_status = None, MatchStatus.UNMATCHED
        else:  # DEFER
            final_id, final_status = result.matched_id, MatchStatus.NEEDS_REVIEW

        yield AdjudicatedResult(
            result_key=result.result_key,
            source_id=result.source_id,
            model_target_id=result.matched_id,
            model_status=result.status,
            confidence=result.confidence,
            final_target_id=final_id,
            final_status=final_status,
            reviewer=str(decision_row["reviewer"]),
            review_note=str(decision_row["review_note"]),
            reviewed_at=str(decision_row["reviewed_at"]),
        )

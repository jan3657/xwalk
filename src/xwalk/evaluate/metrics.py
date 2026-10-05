"""Operational metrics.

Accuracy alone hides the decisions that matter: how much was matched *automatically*,
what precision that automation bought, and how much human time the review bucket costs.
A run at 85% accuracy with a 60% review rate is a worse product than one at 80% with a
5% review rate, and one accuracy number cannot tell you which you have.

Definitions, all computed over **labelled** results only (a source id absent from the
gold file is excluded entirely):

- accepted_precision  : of MATCHED results, the fraction whose matched_id is in the gold set
- automatic_coverage  : MATCHED / labelled
- review_rate         : NEEDS_REVIEW / labelled
- unmatched_rate      : UNMATCHED / labelled
- error_rate          : FAILED / labelled
- unresolved_rate     : reason == UNRESOLVED_OUTPUT / labelled
- recall_at_any_status: of labelled results with a non-empty gold set, the fraction whose
                        matched_id is in the gold set at *any* status -- the ceiling that
                        perfect threshold tuning could reach
- no_match_precision  : of results predicted no-match (matched_id is None, not FAILED),
                        the fraction whose gold set is empty
- no_match_recall     : of results whose gold set is empty, the fraction predicted no-match
- duplicate_target_conflicts : target ids selected by more than one source record
- mean_llm_calls / mean_tokens / mean_cost_usd / mean_seconds : per completed (non-FAILED)
  result
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from xwalk.evaluate.gold import GoldSet
from xwalk.ledger import Ledger
from xwalk.records import DecisionReason, MatchResult, MatchStatus

_MIN_CALIBRATION_SAMPLES = 3
_MIN_SEPARATION = 0.10


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    coverage: float
    precision: float | None
    accepted: int
    correct: int


@dataclass(frozen=True)
class EvalReport:
    total: int
    labelled: int
    accepted_precision: float | None
    automatic_coverage: float | None
    review_rate: float | None
    unmatched_rate: float | None
    error_rate: float | None
    unresolved_rate: float | None
    recall_at_any_status: float | None
    no_match_precision: float | None
    no_match_recall: float | None
    duplicate_target_conflicts: int
    mean_llm_calls: float | None
    mean_tokens: float | None
    mean_cost_usd: float | None
    mean_seconds: float | None
    status_counts: dict[str, int]
    reason_counts: dict[str, int]
    calibration_warning: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "labelled": self.labelled,
            "accepted_precision": self.accepted_precision,
            "automatic_coverage": self.automatic_coverage,
            "review_rate": self.review_rate,
            "unmatched_rate": self.unmatched_rate,
            "error_rate": self.error_rate,
            "unresolved_rate": self.unresolved_rate,
            "recall_at_any_status": self.recall_at_any_status,
            "no_match_precision": self.no_match_precision,
            "no_match_recall": self.no_match_recall,
            "duplicate_target_conflicts": self.duplicate_target_conflicts,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_tokens": self.mean_tokens,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_seconds": self.mean_seconds,
            "status_counts": self.status_counts,
            "reason_counts": self.reason_counts,
            "calibration_warning": self.calibration_warning,
        }


def evaluate_results(results: Sequence[MatchResult], gold: GoldSet) -> EvalReport:
    labelled = [r for r in results if r.source_id in gold]

    matched = [r for r in labelled if r.status is MatchStatus.MATCHED]
    matched_correct = sum(1 for r in matched if gold.is_correct(r.source_id, r.matched_id))

    with_gold = [r for r in labelled if gold.get(r.source_id)]
    recalled = sum(
        1
        for r in with_gold
        if r.matched_id is not None and gold.is_correct(r.source_id, r.matched_id)
    )

    predicted_no_match = [
        r for r in labelled if r.matched_id is None and r.status is not MatchStatus.FAILED
    ]
    no_match_correct = sum(1 for r in predicted_no_match if not gold.get(r.source_id))
    gold_no_match = [r for r in labelled if gold.get(r.source_id) == frozenset()]
    no_match_found = sum(
        1 for r in gold_no_match if r.matched_id is None and r.status is not MatchStatus.FAILED
    )

    target_counts = Counter(r.matched_id for r in results if r.matched_id is not None)
    duplicates = sum(1 for count in target_counts.values() if count > 1)

    completed = [r for r in results if r.status is not MatchStatus.FAILED]

    n = len(labelled)
    return EvalReport(
        total=len(results),
        labelled=n,
        accepted_precision=_ratio(matched_correct, len(matched)),
        automatic_coverage=_ratio(len(matched), n),
        review_rate=_ratio(sum(1 for r in labelled if r.status is MatchStatus.NEEDS_REVIEW), n),
        unmatched_rate=_ratio(sum(1 for r in labelled if r.status is MatchStatus.UNMATCHED), n),
        error_rate=_ratio(sum(1 for r in labelled if r.status is MatchStatus.FAILED), n),
        unresolved_rate=_ratio(
            sum(1 for r in labelled if r.reason is DecisionReason.UNRESOLVED_OUTPUT), n
        ),
        recall_at_any_status=_ratio(recalled, len(with_gold)),
        no_match_precision=_ratio(no_match_correct, len(predicted_no_match)),
        no_match_recall=_ratio(no_match_found, len(gold_no_match)),
        duplicate_target_conflicts=duplicates,
        mean_llm_calls=mean(r.usage.calls for r in completed) if completed else None,
        mean_tokens=mean(r.usage.total_tokens for r in completed) if completed else None,
        mean_cost_usd=mean(r.usage.cost_usd for r in completed) if completed else None,
        mean_seconds=mean(r.elapsed_seconds for r in completed) if completed else None,
        status_counts=dict(Counter(r.status.value for r in results)),
        reason_counts=dict(Counter(r.reason.value for r in results)),
        calibration_warning=calibration_warning(results, gold),
    )


def evaluate_run(ledger: Ledger, run_fingerprint: str, gold: GoldSet) -> EvalReport:
    return evaluate_results(list(ledger.iter_results(run_fingerprint)), gold)


def threshold_curve(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    steps: int = 21,
) -> list[ThresholdPoint]:
    """What precision and coverage would be if `accept_at` were set differently.

    Re-derived from recorded confidences, so it costs nothing and needs no re-run.
    """
    results = list(results)
    scored = [
        (r.confidence, gold.is_correct(r.source_id, r.matched_id))
        for r in results
        if r.source_id in gold and r.confidence is not None and r.matched_id is not None
    ]
    labelled_total = sum(1 for r in results if r.source_id in gold)

    points: list[ThresholdPoint] = []
    for i in range(steps):
        threshold = i / (steps - 1) if steps > 1 else 0.0
        accepted = [correct for confidence, correct in scored if confidence >= threshold]
        correct = sum(1 for c in accepted if c)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                coverage=(len(accepted) / labelled_total) if labelled_total else 0.0,
                precision=_ratio(correct, len(accepted)),
                accepted=len(accepted),
                correct=correct,
            )
        )
    return points


def calibration_warning(results: Iterable[MatchResult], gold: GoldSet) -> str | None:
    """Flag confidences that do not separate correct from incorrect.

    When they overlap, every threshold recommendation derived from them is noise, and a
    user tuning `accept_at` is tuning nothing. Needs at least
    `_MIN_CALIBRATION_SAMPLES` of *each* class: two right and one wrong is not evidence.
    """
    correct: list[float] = []
    wrong: list[float] = []
    for result in results:
        if result.confidence is None or result.matched_id is None:
            continue
        verdict = gold.is_correct(result.source_id, result.matched_id)
        if verdict is None:
            continue
        (correct if verdict else wrong).append(result.confidence)

    if len(correct) < _MIN_CALIBRATION_SAMPLES or len(wrong) < _MIN_CALIBRATION_SAMPLES:
        return None
    separation = mean(correct) - mean(wrong)
    if separation >= _MIN_SEPARATION:
        return None
    return (
        f"confidence scores barely separate correct from incorrect matches "
        f"(mean correct {mean(correct):.2f} vs mean incorrect {mean(wrong):.2f}, "
        f"separation {separation:.2f} < {_MIN_SEPARATION}); threshold tuning will not help "
        f"until the scoring prompt discriminates better"
    )

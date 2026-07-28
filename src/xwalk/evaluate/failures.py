"""Choosing which failures to show the optimising model.

"Only selection failures feed the optimiser" is correct for the selector and wrong for
everything else. A selector-optimisation prompt shown a case where the gold record was
never retrieved teaches nothing -- the model never saw the right answer.

That argument does *not* extend to the scorer. Rejecting a slate that contains nothing
correct is precisely the scorer's job, so a confident accept in that situation is its
canonical failure and must reach the optimiser.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from xwalk.evaluate.ceiling import CeilingBucket, classify_ceiling
from xwalk.evaluate.gold import GoldSet
from xwalk.records import DecisionReason, MatchResult, MatchStatus

_NOT_A_FAILURE_QUESTION = (CeilingBucket.UNLABELLED, CeilingBucket.NO_GOLD)
_ABSTENTION_REASONS = (DecisionReason.SELECTOR_ABSTAINED, DecisionReason.BELOW_REVIEW_FLOOR)


class PromptRole(Enum):
    SELECTOR = "selector"
    SCORER = "scorer"
    REWRITER = "rewriter"
    DOC_TEMPLATE = "doc_template"


@dataclass(frozen=True)
class FailureCase:
    source_id: str
    source_fields: dict[str, str]
    gold_ids: tuple[str, ...]
    chosen_id: str | None
    confidence: float | None
    status: str
    reason: str
    bucket: CeilingBucket
    presented: tuple[str, ...]
    queries: tuple[str, ...]
    explanation: str


def _to_case(result: MatchResult, gold: GoldSet, bucket: CeilingBucket) -> FailureCase:
    presented: list[str] = []
    for attempt in result.attempts:
        for record_id in attempt.issued_keys.values():
            if record_id not in presented:
                presented.append(record_id)
    return FailureCase(
        source_id=result.source_id,
        source_fields={},  # populated by the optimizer, which has the source records
        gold_ids=tuple(sorted(gold.get(result.source_id) or ())),
        chosen_id=result.matched_id,
        confidence=result.confidence,
        status=result.status.value,
        reason=result.reason.value,
        bucket=bucket,
        presented=tuple(presented),
        queries=tuple(a.query for a in result.attempts),
        explanation=result.explanation,
    )


def select_failures(
    results: Sequence[MatchResult],
    gold: GoldSet,
    role: PromptRole,
    *,
    limit: int | None = None,
) -> list[FailureCase]:
    """Cases worth showing the model for this role, in a deterministic order."""
    cases: list[FailureCase] = []

    for result in results:
        bucket = classify_ceiling(result, gold)
        if bucket in _NOT_A_FAILURE_QUESTION:
            continue
        expected = gold.get(result.source_id) or frozenset()
        correct = result.matched_id is not None and result.matched_id in expected

        keep = False
        if role is PromptRole.SELECTOR:
            # gold was presented and something else was chosen
            keep = bucket is CeilingBucket.MISJUDGED

        elif role is PromptRole.SCORER:
            gold_presented = bool(
                expected & {rid for a in result.attempts for rid in a.issued_keys.values()}
            )
            wrong_accept = result.status is MatchStatus.MATCHED and not correct
            needless_abstain = (
                gold_presented
                and result.matched_id is None
                and result.reason in _ABSTENTION_REASONS
            )
            suppressed = correct and result.status in (
                MatchStatus.UNMATCHED,
                MatchStatus.NEEDS_REVIEW,
            )
            keep = wrong_accept or needless_abstain or suppressed

        elif role is PromptRole.REWRITER:
            first = result.attempts[0] if result.attempts else None
            found_first = bool(first and expected & {c.id for c in first.candidates})
            keep = not found_first  # either recovered later, or never -- both are its job

        elif role is PromptRole.DOC_TEMPLATE:
            keep = bucket is CeilingBucket.NEVER_RETRIEVED

        if keep:
            cases.append(_to_case(result, gold, bucket))

    cases.sort(key=lambda c: c.source_id)  # deterministic under a limit
    return cases if limit is None else cases[:limit]


def render_failure(case: FailureCase) -> str:
    """A compact, prose-safe rendering for inclusion in an optimisation prompt."""
    lines = [f"Source record {case.source_id}"]
    for key, value in case.source_fields.items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  correct answer: {', '.join(case.gold_ids) or 'no match'}")
    lines.append(f"  model chose:    {case.chosen_id or 'no match'}")
    if case.confidence is not None:
        lines.append(f"  confidence:     {case.confidence:.2f}")
    lines.append(f"  outcome:        {case.status} / {case.reason}")
    if case.presented:
        lines.append(f"  candidates shown: {', '.join(case.presented)}")
    if case.queries:
        lines.append(f"  queries tried:    {', '.join(case.queries)}")
    if case.explanation:
        lines.append(f"  model said:     {case.explanation}")
    return "\n".join(lines)

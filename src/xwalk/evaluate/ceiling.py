"""Where the failures actually are.

Three buckets, never two. The selector budget creates a genuinely distinct failure: if
the gold record was retrieved but cut before the model saw it, the fix is a larger
budget -- not a better retriever and not a better prompt. Collapsing that into "retrieval
failure" sends users to fix the wrong thing.

Nearly free: the loop already records `evidence` on every candidate and `issued_keys`
on every attempt, so this reads a completed run and calls nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from xwalk.evaluate.gold import GoldSet
from xwalk.records import DecisionReason, MatchResult


class CeilingBucket(Enum):
    UNAVAILABLE = "unavailable"
    GOLD_UNREACHABLE = "gold_unreachable"
    FOUND = "found"
    NEVER_RETRIEVED = "never_retrieved"
    TRUNCATED = "truncated"
    MISJUDGED = "misjudged"
    NO_GOLD = "no_gold"  # the correct answer is no match; not a ceiling question
    UNLABELLED = "unlabelled"  # excluded from every total


_NOT_A_CEILING_QUESTION = (
    CeilingBucket.UNLABELLED,
    CeilingBucket.NO_GOLD,
    CeilingBucket.UNAVAILABLE,
    CeilingBucket.GOLD_UNREACHABLE,
)


def _retrieved_ids(result: MatchResult) -> set[str]:
    return {c.id for attempt in result.attempts for c in attempt.candidates}


def _issued_ids(result: MatchResult) -> set[str]:
    return {rid for attempt in result.attempts for rid in attempt.issued_keys.values()}


def classify_ceiling(
    result: MatchResult,
    gold: GoldSet,
    *,
    target_ids: Collection[str] | None = None,
    path: Literal["llm", "decider"] | None = None,
) -> CeilingBucket:
    """Precedence matters: presented beats retrieved beats absent.

    A record that reached the model is never reported as a budget miss, and a record the
    budget cut is never reported as a retrieval miss.
    Only the LLM path retains candidates cut by the selector budget. A clipped
    decider or unknown path cannot prove absence from the original retrieved slate.
    """
    expected = gold.get(result.source_id)
    if expected is None:
        return CeilingBucket.UNLABELLED
    if not expected:
        return CeilingBucket.NO_GOLD
    if target_ids is not None and not expected.intersection(target_ids):
        return CeilingBucket.GOLD_UNREACHABLE
    if result.matched_id is not None and result.matched_id in expected:
        return CeilingBucket.FOUND
    if expected & _issued_ids(result):
        return CeilingBucket.MISJUDGED
    if expected & _retrieved_ids(result):
        # A missing key alone is not clipping evidence: selection may have failed
        # before returning its presentation map. Decider traces retain only the
        # post-budget slate, so their retained gold cannot prove a budget cut.
        if path != "decider" and any(
            expected.intersection(c.id for c in a.candidates)
            and a.candidates_truncated > 0
            and len(a.candidates) == a.candidate_count
            and len(a.candidates) == len(a.issued_keys) + a.candidates_truncated
            and (a.issued_keys or a.reason is DecisionReason.CANDIDATE_BUDGET_EXCEEDED)
            for a in result.attempts
        ):
            return CeilingBucket.TRUNCATED
        return CeilingBucket.UNAVAILABLE
    # Positive evidence survives missing traces; absence requires a complete slate.
    if not result.attempts or any(
        len(a.candidates) < a.candidate_count
        or (
            a.candidates_truncated > 0
            and (path != "llm" or len(a.candidates) != len(a.issued_keys) + a.candidates_truncated)
        )
        or a.retrieval_failures
        for a in result.attempts
    ):
        return CeilingBucket.UNAVAILABLE
    return CeilingBucket.NEVER_RETRIEVED


@dataclass(frozen=True)
class CeilingReport:
    evaluable: int
    buckets: dict[CeilingBucket, int]
    by_retriever: dict[str, int]
    by_attempt: dict[int, int]
    recommendation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluable": self.evaluable,
            "buckets": {b.value: n for b, n in self.buckets.items()},
            "by_retriever": self.by_retriever,
            "by_attempt": {str(k): v for k, v in self.by_attempt.items()},
            "recommendation": self.recommendation,
        }


def ceiling_report(
    results: Sequence[MatchResult],
    gold: GoldSet,
    *,
    target_ids: Collection[str] | None = None,
    path: Literal["llm", "decider"] | None = None,
) -> CeilingReport:
    buckets: Counter[CeilingBucket] = Counter()
    by_retriever: Counter[str] = Counter()
    by_attempt: Counter[int] = Counter()

    for result in results:
        bucket = classify_ceiling(result, gold, target_ids=target_ids, path=path)
        buckets[bucket] += 1
        if bucket in _NOT_A_CEILING_QUESTION:
            continue

        expected = gold.get(result.source_id) or frozenset()
        seen_retrievers: set[str] = set()
        first_attempt: int | None = None
        for attempt in result.attempts:
            for candidate in attempt.candidates:
                if candidate.id not in expected:
                    continue
                if first_attempt is None:
                    first_attempt = attempt.index
                seen_retrievers.update(hit.retriever for hit in candidate.evidence)
        for name in seen_retrievers:
            by_retriever[name] += 1
        if first_attempt is not None:
            by_attempt[first_attempt] += 1

    evaluable = sum(
        count for bucket, count in buckets.items() if bucket not in _NOT_A_CEILING_QUESTION
    )
    failures = {
        CeilingBucket.NEVER_RETRIEVED: buckets[CeilingBucket.NEVER_RETRIEVED],
        CeilingBucket.TRUNCATED: buckets[CeilingBucket.TRUNCATED],
        CeilingBucket.MISJUDGED: buckets[CeilingBucket.MISJUDGED],
    }
    total_failures = sum(failures.values())

    if total_failures == 0:
        recommendation = "no ceiling failures on labelled records"
    else:
        worst = max(failures, key=lambda b: failures[b])
        share = failures[worst] / total_failures
        recommendation = {
            CeilingBucket.NEVER_RETRIEVED: (
                f"{share:.0%} of failures never retrieved the gold record -- spend effort on "
                f"retrieval: the doc template, the retriever set, or retrieval depth"
            ),
            CeilingBucket.TRUNCATED: (
                f"{share:.0%} of failures retrieved the gold record but the selector budget "
                f"cut it before the model saw it -- raise SelectorPolicy.max_candidates"
            ),
            CeilingBucket.MISJUDGED: (
                f"{share:.0%} of failures presented the gold record and the model chose "
                f"otherwise -- spend effort on prompts, not retrieval"
            ),
        }[worst]

    if buckets[CeilingBucket.UNAVAILABLE]:
        recommendation += "; retrieval ceiling unavailable for incomplete traces or searches"
    if buckets[CeilingBucket.GOLD_UNREACHABLE]:
        recommendation += "; gold_unreachable excluded (no gold id in target catalogue)"

    return CeilingReport(
        evaluable=evaluable,
        buckets=dict(buckets),
        by_retriever=dict(by_retriever),
        by_attempt=dict(by_attempt),
        recommendation=recommendation,
    )

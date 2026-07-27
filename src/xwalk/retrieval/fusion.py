"""Reciprocal rank fusion.

Combines rankings from retrievers whose raw scores are not comparable — a BM25 score
and a cosine similarity have no shared scale — by using rank position alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from xwalk.records import Candidate, RetrievalHit
from xwalk.stores.base import TargetStore


def reciprocal_rank_fusion(
    hit_groups: Sequence[Sequence[RetrievalHit]],
    store: TargetStore,
    *,
    k: int = 60,
) -> list[Candidate]:
    """Fuse per-retriever rankings into scored candidates carrying their evidence.

    Score is the sum of `1 / (k + rank)` over the retrievers that surfaced the record.
    Records absent from the store are dropped: a stale index may name a deleted record.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    evidence: dict[str, dict[str, RetrievalHit]] = {}

    for group in hit_groups:
        for hit in group:
            per_retriever = evidence.setdefault(hit.record_id, {})
            if hit.retriever in per_retriever:
                continue  # one vote per retriever, best rank wins by arrival order
            per_retriever[hit.retriever] = hit
            scores[hit.record_id] = scores.get(hit.record_id, 0.0) + 1.0 / (k + hit.rank)

    if not scores:
        return []

    records = {r.id: r for r in store.get_many(sorted(scores))}

    candidates = [
        Candidate(
            record=records[record_id],
            fused_score=score,
            # sorted so the trace is byte-identical regardless of retriever completion order
            evidence=tuple(
                sorted(evidence[record_id].values(), key=lambda h: (h.retriever, h.rank))
            ),
        )
        for record_id, score in scores.items()
        if record_id in records
    ]
    # descending score, then ascending id — a stable, reproducible order
    candidates.sort(key=lambda c: (-c.fused_score, c.id))
    return candidates

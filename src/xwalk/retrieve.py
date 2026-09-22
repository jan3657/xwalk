"""Retrieval shared by both matchers: every retriever, every query, one fused list."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.retrieval.base import Retriever, SearchRequest
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.stores.base import TargetStore


async def retrieve(
    queries: Sequence[str],
    source: Record,
    retrievers: Sequence[Retriever],
    store: TargetStore,
    *,
    timeout: float,
    rrf_k: int = 60,
    fallback_limit: int = 20,
) -> tuple[list[Candidate], list[str], bool]:
    """Search every retriever with every query concurrently and fuse by reciprocal rank.

    Returns (candidates, degradation notes, all_failed). A record surfaced by several
    queries accumulates several votes, which is the point of asking more than one.
    Depth comes from each retriever's own `default_limit`; `fallback_limit` covers a
    backend that does not declare one.
    """
    jobs = [(retriever, query) for query in queries for retriever in retrievers]

    async def one(retriever: Retriever, query: str) -> Sequence[RetrievalHit]:
        limit = getattr(retriever, "default_limit", None) or fallback_limit
        request = SearchRequest(text=query, limit=limit, source_record=source)
        return await asyncio.wait_for(retriever.search(request), timeout=timeout)

    outcomes = await asyncio.gather(*(one(r, q) for r, q in jobs), return_exceptions=True)

    groups: list[Sequence[RetrievalHit]] = []
    notes: list[str] = []
    for (retriever, _query), outcome in zip(jobs, outcomes, strict=True):
        # TimeoutError is an Exception, so it must be tested first.
        if isinstance(outcome, asyncio.TimeoutError):
            notes.append(f"{retriever.name}: timed out after {timeout}s")
        elif isinstance(outcome, BaseException):
            notes.append(f"{retriever.name}: {outcome}")
        else:
            # One vote per (retriever, query): tag the hits so fusion counts each query.
            groups.append(
                [
                    RetrievalHit(
                        record_id=h.record_id,
                        retriever=f"{h.retriever}#{_query}" if len(queries) > 1 else h.retriever,
                        raw_score=h.raw_score,
                        rank=h.rank,
                    )
                    for h in outcome
                ]
            )

    all_failed = bool(jobs) and len(notes) == len(jobs)
    if not groups:
        return [], notes, all_failed
    return reciprocal_rank_fusion(groups, store, k=rrf_k), notes, all_failed

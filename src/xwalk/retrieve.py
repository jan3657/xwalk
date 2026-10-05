"""Retrieval shared by both matchers: every retriever, every query, one fused list."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

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
    # One vote per (retriever, query) needs a distinct fusion key per query, but only when
    # there is more than one query: the single-query path must stay byte-identical.
    tagged = len(queries) > 1

    async def one(retriever: Retriever, query: str) -> Sequence[RetrievalHit]:
        limit = getattr(retriever, "default_limit", None) or fallback_limit
        request = SearchRequest(text=query, limit=limit, source_record=source)
        return await asyncio.wait_for(retriever.search(request), timeout=timeout)

    outcomes = await asyncio.gather(*(one(r, q) for r, q in jobs), return_exceptions=True)

    groups: list[Sequence[RetrievalHit]] = []
    notes: list[str] = []
    # Tagged fusion key -> the real retriever name, so the tag is never parsed back out of
    # a string a query may itself contain "#" in.
    real_name: dict[str, str] = {}
    for (retriever, query), outcome in zip(jobs, outcomes, strict=True):
        # TimeoutError is an Exception, so it must be tested first.
        if isinstance(outcome, asyncio.TimeoutError):
            notes.append(f"{retriever.name}: timed out after {timeout}s")
        elif isinstance(outcome, BaseException):
            notes.append(f"{retriever.name}: {outcome}")
        elif not tagged:
            groups.append(outcome)
        else:
            group: list[RetrievalHit] = []
            for hit in outcome:
                tag = f"{hit.retriever}#{query}"
                real_name[tag] = hit.retriever
                group.append(
                    RetrievalHit(
                        record_id=hit.record_id,
                        retriever=tag,
                        raw_score=hit.raw_score,
                        rank=hit.rank,
                    )
                )
            groups.append(group)

    all_failed = bool(jobs) and len(notes) == len(jobs)
    if not groups:
        return [], notes, all_failed
    candidates = reciprocal_rank_fusion(groups, store, k=rrf_k)
    if not tagged:
        return candidates, notes, all_failed
    return [_untag(c, real_name) for c in candidates], notes, all_failed


def _untag(candidate: Candidate, real_name: Mapping[str, str]) -> Candidate:
    """Strip the per-query fusion tag from a candidate's evidence, keeping the score.

    The tag exists only to buy a vote per query inside fusion. Persisted traces and the
    retrieval-ceiling report count evidence by retriever name, so one retriever must not
    show up as one entry per query once fusion is done.
    """
    best: dict[str, RetrievalHit] = {}
    for hit in candidate.evidence:
        name = real_name.get(hit.retriever, hit.retriever)
        previous = best.get(name)
        if previous is None or hit.rank < previous.rank:
            best[name] = RetrievalHit(
                record_id=hit.record_id,
                retriever=name,
                raw_score=hit.raw_score,
                rank=hit.rank,
            )
    return Candidate(
        record=candidate.record,
        fused_score=candidate.fused_score,
        evidence=tuple(sorted(best.values(), key=lambda h: (h.retriever, h.rank))),
    )

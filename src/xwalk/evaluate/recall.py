"""Retrieval recall@k of a job's retrievers against gold, offline and without a decider.

This is the one module in `xwalk.evaluate` that runs retrievers rather than reading a
ledger: it builds the job's indexes, asks every retriever every rendered query exactly
as the matchers do, and checks whether a gold id is in the top k of the fused list. It
never calls an LLM or a decider, so it needs no credentials.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from xwalk.config import JobSpec
from xwalk.evaluate.gold import GoldSet
from xwalk.retrieve import retrieve


def retrieval_recall(
    job: JobSpec,
    gold: GoldSet,
    *,
    index_dir: Path,
    ks: Sequence[int] = (10, 50, 200),
) -> dict[int, float]:
    """Fraction of labelled source rows whose gold set meets the top-k fused candidates.

    Rows labelled "no match" (an empty gold set) and unlabelled rows are excluded from
    numerator and denominator alike: neither has an id retrieval could find.
    """
    if any(k < 1 for k in ks):
        raise ValueError(f"every k must be at least 1, got {list(ks)}")
    labelled = [
        (source, wanted) for source in job.build_source_records() if (wanted := gold.get(source.id))
    ]
    if not labelled:
        raise ValueError("no source record has a gold id to retrieve")
    templates = job.build_templates()
    retrievers = job.build_retrievers(list(job.build_target_records()), templates, index_dir)
    store = job.build_store()
    timeout = (
        job.decision_policy.retriever_timeout
        if job.decider is not None
        else job.policy.retriever_timeout
    )

    async def ranked_ids() -> list[list[str]]:
        # Sequential on purpose: one record's retrievers already run concurrently.
        out: list[list[str]] = []
        for source, _ in labelled:
            fused, _notes, _failed = await retrieve(
                templates.render_queries(source), source, retrievers, store, timeout=timeout
            )
            out.append([candidate.id for candidate in fused])
        return out

    hits = {k: 0 for k in ks}
    for (_, wanted), ids in zip(labelled, asyncio.run(ranked_ids()), strict=True):
        for k in ks:
            hits[k] += not wanted.isdisjoint(ids[:k])
    return {k: hits[k] / len(labelled) for k in ks}

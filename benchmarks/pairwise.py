"""Pairwise track: classify preblocked (source, target) pairs as same / not same.

The pair table is built once from BM25 top-`k` per source, so every classifier judges
the same pairs. A pair table measures the classifier only: a gold target that blocking
never proposed is not in the table, so it cannot be missed here. That count is reported
as ``blocking_misses`` -- it is what the source-to-catalog track measures and this one
does not.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from benchmarks.data import Variant
from benchmarks.methods import FUZZY_ACCEPT, _bm25_hits, _names, _query, build_bm25
from benchmarks.metrics import pairwise_metrics
from benchmarks.synthetic_llm import normalise
from xwalk.batch import usage_to_dict
from xwalk.llm.base import LLMClient
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.records import Candidate, Record
from xwalk.stages.gate import Scorer
from xwalk.stages.keying import assign_keys


@dataclass(frozen=True)
class Pair:
    source: Record
    target: Record
    label: bool


def build_pairs(variant: Variant, work: Path, k: int) -> tuple[list[Pair], int, Any]:
    spec, retriever, store = build_bm25(variant, work / "pairwise_index")
    hits = asyncio.run(_bm25_hits(spec, retriever, variant.sources))
    pairs: list[Pair] = []
    misses = 0
    for source in variant.sources:
        gold = variant.gold.get(source.id) or frozenset()
        block = hits[source.id][:k]
        if gold and not gold & set(block):
            misses += 1
        for rid in block:
            if rid in store:
                pairs.append(Pair(source, store.get(rid), rid in gold))
    return pairs, misses, spec


def classify_exact(variant: Variant, pairs: list[Pair]) -> list[bool]:
    return [
        normalise(_query(variant, p.source)) in {normalise(n) for n in _names(p.target)}
        for p in pairs
    ]


def classify_fuzzy(variant: Variant, pairs: list[Pair]) -> list[bool]:
    out = []
    for p in pairs:
        query = normalise(_query(variant, p.source))
        best = max(
            (SequenceMatcher(None, query, normalise(n)).ratio() for n in _names(p.target)),
            default=0.0,
        )
        out.append(best >= FUZZY_ACCEPT)
    return out


def classify_scorer(
    variant: Variant, pairs: list[Pair], spec: Any, llm: LLMClient, max_calls: int | None = None
) -> tuple[list[bool], dict[str, Any]]:
    """xwalk's scorer stage, one call per pair, accepted at the job's `accept_at`."""
    budgeted = BudgetedLLM(llm, CallBudget(max_calls))
    templates = spec.build_templates()
    policy = spec.build_policy()
    scorer = Scorer(budgeted, spec.build_prompts(), templates, review_floor=policy.review_floor)

    async def run_all() -> list[bool]:
        semaphore = asyncio.Semaphore(policy.concurrency)

        async def one(pair: Pair) -> bool:
            async with semaphore:
                keyed = assign_keys(
                    [Candidate(record=pair.target, fused_score=1.0, evidence=())], templates
                )
                outcome = await scorer.score(
                    pair.source, templates.render_context(pair.source), keyed, keyed.order[0]
                )
                return outcome.score is not None and outcome.score >= policy.accept_at

        return list(await asyncio.gather(*(one(p) for p in pairs)))

    decisions = asyncio.run(run_all())
    return decisions, usage_to_dict(budgeted.usage)


def run_pairwise(
    variant: Variant, work: Path, llm: LLMClient | None, k: int, max_calls: int | None = None
) -> dict[str, Any]:
    pairs, misses, spec = build_pairs(variant, work, k)
    labels = [p.label for p in pairs]
    out: dict[str, Any] = {
        "block_k": k,
        "blocking": "bm25 top-k with the job's index settings",
        "blocking_misses": misses,
        "methods": {
            "exact": pairwise_metrics(labels, classify_exact(variant, pairs)),
            "fuzzy": pairwise_metrics(labels, classify_fuzzy(variant, pairs)),
        },
    }
    if llm is not None:
        decided, usage = classify_scorer(variant, pairs, spec, llm, max_calls)
        out["methods"]["xwalk_scorer"] = {**pairwise_metrics(labels, decided), "usage": usage}
    return out

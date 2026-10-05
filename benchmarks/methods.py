"""The compared methods. Each returns one `Prediction` per source record plus its cost.

All methods read the same materialised variant (same catalog, same fields: label and
synonyms) and, where they retrieve, the same BM25 index settings and candidate budget as
the job. Intentional differences are listed in docs/benchmarks.md.

Thresholds are fixed here, before any result was seen, and are never tuned on the data.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from benchmarks.data import Variant
from benchmarks.metrics import Prediction
from benchmarks.synthetic_llm import normalise
from xwalk import ops
from xwalk.batch import usage_to_dict
from xwalk.config import JobSpec, load_job
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMClient
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.records import MatchResult, Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.stages.keying import Resolution
from xwalk.stages.select import Selector
from xwalk.stores.memory import MemoryStore

FUZZY_ACCEPT = 0.90
FUZZY_REVIEW = 0.75


@dataclass
class MethodOutput:
    predictions: list[Prediction]
    usage: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _names(record: Record) -> list[str]:
    names = [str(record.fields.get("label") or "")]
    synonyms = record.fields.get("synonyms") or []
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    names.extend(str(s) for s in synonyms)
    return [n for n in names if n.strip()]


def _query(variant: Variant, source: Record) -> str:
    return str(source.fields.get(variant.query_field) or "")


# --- string baselines -----------------------------------------------------------------


def exact(variant: Variant) -> MethodOutput:
    """Normalised mention == normalised label or synonym. One hit accepts; several are
    ambiguous (review); none is no-match."""
    index: dict[str, set[str]] = defaultdict(set)
    for target in variant.targets:
        for name in _names(target):
            index[normalise(name)].add(target.id)
    predictions = []
    for source in variant.sources:
        hits = sorted(index.get(normalise(_query(variant, source)), ()))
        if len(hits) == 1:
            predictions.append(Prediction(source.id, "matched", hits[0], 1.0))
        elif hits:
            predictions.append(Prediction(source.id, "needs_review", hits[0], 1.0))
        else:
            predictions.append(Prediction(source.id, "unmatched", None))
    return MethodOutput(predictions, config={"normalise": "lowercase alphanumeric tokens"})


def fuzzy(variant: Variant) -> MethodOutput:
    """Best stdlib difflib ratio over every label and synonym (brute force)."""
    names = [(normalise(n), t.id) for t in variant.targets for n in _names(t)]
    predictions = []
    for source in variant.sources:
        query = normalise(_query(variant, source))
        best_score, best_id = 0.0, None
        matcher = SequenceMatcher(None, b=query)
        for name, tid in names:
            matcher.set_seq1(name)
            # The quick ratios are upper bounds of ratio(); skipping on them cannot change
            # the best score, only avoid computing ratios that cannot win.
            if matcher.real_quick_ratio() < best_score or matcher.quick_ratio() < best_score:
                continue
            score = matcher.ratio()
            if score > best_score or (score == best_score and best_id and tid < best_id):
                best_score, best_id = score, tid
        if best_id is not None and best_score >= FUZZY_ACCEPT:
            status = "matched"
        elif best_id is not None and best_score >= FUZZY_REVIEW:
            status = "needs_review"
        else:
            status, best_id = "unmatched", None
        predictions.append(Prediction(source.id, status, best_id, round(best_score, 4)))
    return MethodOutput(
        predictions,
        config={
            "scorer": "difflib.SequenceMatcher.ratio",
            "accept": FUZZY_ACCEPT,
            "review": FUZZY_REVIEW,
        },
    )


# --- retrieval ------------------------------------------------------------------------


def build_bm25(variant: Variant, index_dir: Path) -> tuple[JobSpec, BM25Retriever, MemoryStore]:
    spec = load_job(variant.job_path)
    retriever_spec = spec.retrievers[0]
    if retriever_spec.kind != "bm25":
        raise ValueError("the pilot jobs use one bm25 retriever")
    targets = list(spec.build_target_records())
    retriever = BM25Retriever.build(
        targets,
        spec.build_templates(),
        index_dir,
        name=retriever_spec.name or "bm25",
        exact_fields=retriever_spec.effective_exact_fields,
        default_limit=retriever_spec.limit,
    )
    return spec, retriever, MemoryStore.from_source(targets)


async def _bm25_hits(
    spec: JobSpec, retriever: BM25Retriever, sources: Sequence[Record]
) -> dict[str, list[str]]:
    templates = spec.build_templates()
    out: dict[str, list[str]] = {}
    for source in sources:
        hits = await retriever.search(
            SearchRequest(
                text=templates.render_query(source),
                limit=retriever.default_limit,
                source_record=source,
            )
        )
        out[source.id] = [h.record_id for h in hits]
    return out


def bm25_only(variant: Variant, work: Path) -> MethodOutput:
    """Retrieval only: BM25 top-1 is the answer. It cannot abstain."""
    spec, retriever, _ = build_bm25(variant, work / "bm25_only_index")
    hits = asyncio.run(_bm25_hits(spec, retriever, variant.sources))
    predictions = [
        Prediction(
            s.id,
            "matched" if hits[s.id] else "unmatched",
            hits[s.id][0] if hits[s.id] else None,
            retrieved_ids=tuple(hits[s.id]),
        )
        for s in variant.sources
    ]
    return MethodOutput(predictions, config={"k": retriever.default_limit, "accept": "top-1"})


def selector_one_pass(
    variant: Variant, work: Path, llm: LLMClient, *, max_calls: int | None = None
) -> MethodOutput:
    """Retrieval + one LLM pass: the job's selector prompt over the job's candidate
    budget, accepted on its own confidence. No scorer, verifier or retry."""
    spec, retriever, store = build_bm25(variant, work / "one_pass_index")
    budgeted = BudgetedLLM(llm, CallBudget(max_calls))
    policy = spec.build_policy()
    templates = spec.build_templates()
    selector = Selector(
        budgeted,
        spec.build_prompts(),
        templates,
        policy=spec.build_selector_policy(),
    )

    async def run_all() -> list[Prediction]:
        hits = await _bm25_hits(spec, retriever, variant.sources)
        semaphore = asyncio.Semaphore(policy.concurrency)

        async def one(source: Record) -> Prediction:
            async with semaphore:
                group = [
                    RetrievalHit(record_id=rid, retriever=retriever.name, raw_score=None, rank=i)
                    for i, rid in enumerate(hits[source.id], start=1)
                ]
                candidates = reciprocal_rank_fusion([group], store)
                outcome = await selector.select(
                    source, templates.render_context(source), candidates
                )
                chosen = outcome.choice.record_id
                shown = tuple(outcome.keyed.issued.values())
                conf = outcome.confidence
                if chosen is None or conf is None:
                    # Abstention (or no candidates) is a no-match answer; an unparseable
                    # or unresolvable answer is a failure, never a match.
                    abstained = outcome.choice.resolution is Resolution.ABSTAIN
                    status = "unmatched" if abstained else "failed"
                    return Prediction(source.id, status, None, conf, tuple(hits[source.id]), shown)
                if conf >= policy.accept_at:
                    status = "matched"
                elif conf >= policy.review_floor:
                    status = "needs_review"
                else:
                    status = "unmatched"
                return Prediction(source.id, status, chosen, conf, tuple(hits[source.id]), shown)

        return list(await asyncio.gather(*(one(s) for s in variant.sources)))

    predictions = asyncio.run(run_all())
    return MethodOutput(
        predictions,
        usage=usage_to_dict(budgeted.usage),
        config={
            "k": retriever.default_limit,
            "max_candidates": spec.selector.max_candidates,
            "accept_at": policy.accept_at,
            "review_floor": policy.review_floor,
            "model": llm.model,
        },
    )


# --- xwalk ----------------------------------------------------------------------------


def result_to_prediction(result: MatchResult) -> Prediction:
    retrieved = tuple(dict.fromkeys(c.id for a in result.attempts for c in a.candidates))
    shown = tuple(dict.fromkeys(rid for a in result.attempts for rid in a.issued_keys.values()))
    first = result.attempts[0].chosen_id if result.attempts else None
    return Prediction(
        source_id=result.source_id,
        status=result.status.value,
        predicted_id=result.matched_id,
        confidence=result.confidence,
        retrieved_ids=retrieved,
        shown_ids=shown,
        first_choice_id=first,
        has_first_choice=bool(result.attempts),
        attempts=len(result.attempts),
    )


def xwalk_run(
    variant: Variant,
    out: Path,
    llm: LLMClient,
    *,
    policy_overrides: dict[str, Any] | None = None,
    max_calls: int | None = None,
) -> MethodOutput:
    """The full pipeline through `xwalk.ops.run` (what `xwalk match` runs)."""
    job_path = variant.job_path
    if policy_overrides:
        import yaml

        data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
        data["policy"].update(policy_overrides)
        # Paths in a job resolve against its directory; keep the variant's directory.
        job_path = variant.directory / f"job.{out.name}.yaml"
        job_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = ops.run(job_path, out, llm=llm, max_calls=max_calls)
    if result.run is None:
        raise RuntimeError(f"xwalk run produced no run: {result.envelope()}")
    ledger = Ledger.open(out / "ledger.sqlite")
    try:
        results = list(ledger.iter_results(str(result.run["run_fingerprint"])))
    finally:
        ledger.close()
    spec = load_job(job_path)
    return MethodOutput(
        [result_to_prediction(r) for r in results],
        usage=result.usage,
        config={
            "policy": spec.policy.model_dump(),
            "selector": spec.selector.model_dump(),
            "retrievers": [r.model_dump(exclude_none=True) for r in spec.retrievers],
            "model": llm.model,
            "max_calls": max_calls,
        },
        extra={
            "exit_code": result.exit_code,
            "run_state": result.run.get("run_state"),
            "counts": result.counts,
            "errors": [e.to_dict() for e in result.errors],
        },
    )

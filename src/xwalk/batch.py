"""Batch matching: the mapping platform.

Point at a source, get a resumable run directory containing a ledger and three exports.
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from xwalk import __version__
from xwalk.decide.base import DecisionClient
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.fingerprint import hash_record, hash_value, result_key
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMClient
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet
from xwalk.records import MatchResult, MatchStatus, Record, Usage
from xwalk.retrieval.base import Retriever
from xwalk.review import adjudicated
from xwalk.serde import result_to_dict
from xwalk.stages.select import SelectorPolicy
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet

MAPPING_COLUMNS = (
    "source_id",
    "matched_id",
    "confidence",
    "status",
    "reason",
    "explanation",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "llm_calls",
    "elapsed_seconds",
    "cost_usd",
)


class MatcherLike(Protocol):
    @property
    def run_fingerprint(self) -> str: ...

    @property
    def policy(self) -> Any: ...  # anything with a `.concurrency: int`

    @property
    def store_fingerprint(self) -> str: ...

    async def match(self, source: Record) -> MatchResult: ...


def build_run_fingerprint(
    *,
    templates: TemplateSet,
    prompts: PromptSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    llm: LLMClient,
    policy: MatchPolicy,
    selector_policy: SelectorPolicy,
    retriever_limit: int = 20,
    rrf_k: int = 60,
) -> str:
    """Everything whose change should invalidate prior results.

    Deliberately excludes credentials, output paths, and concurrency — none of them
    change what a result means. `retriever_limit` and `rrf_k` must match the values
    handed to `Matcher`, or a resumed run will reuse results produced at another depth.
    """
    return hash_value(
        {
            "library_version": __version__,
            "templates": templates.fingerprint,
            "prompts": prompts.fingerprint,
            "target": store.fingerprint,
            "retrievers": sorted(f"{r.name}:{r.fingerprint}" for r in retrievers),
            # Depth and fusion constant change which candidates exist at all, so they
            # belong here: raising k from 20 to 100 must not silently reuse old results.
            "retrieval": {
                "depths": sorted(
                    f"{r.name}:{getattr(r, 'default_limit', retriever_limit)}" for r in retrievers
                ),
                "retriever_limit": retriever_limit,
                "rrf_k": rrf_k,
                "retriever_timeout": policy.retriever_timeout,
            },
            "llm": llm.fingerprint,
            "policy": {
                "max_attempts": policy.max_attempts,
                "accept_at": policy.accept_at,
                "review_floor": policy.review_floor,
                "verify_band": list(policy.verify_band) if policy.verify_band else None,
                "audit_rate": policy.audit_rate,
                "legacy_id_resolution": policy.legacy_id_resolution,
            },
            "selector_policy": {
                "max_candidates": selector_policy.max_candidates,
                "max_candidate_tokens": selector_policy.max_candidate_tokens,
            },
        }
    )


def build_decision_run_fingerprint(
    *,
    templates: TemplateSet,
    questions: QuestionSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    decider: DecisionClient,
    policy: DecisionPolicy,
    rrf_k: int = 60,
    rewrite: str | None = None,
) -> str:
    """The decider path's counterpart of `build_run_fingerprint`.

    Concurrency, timeouts, and credentials are excluded for the same reason as on the
    LLM path: they change how fast an answer arrives, never what it means. `rewrite` is
    the fingerprint of the optional miss-only query rewrite; it is hashed only when set,
    so a job without one keeps the fingerprint (and the cached ledger) it had before.
    """
    extra: dict[str, str] = {} if rewrite is None else {"rewrite": rewrite}
    return hash_value(
        {
            **extra,
            "library_version": __version__,
            "path": "decider",
            "templates": templates.fingerprint,
            "questions": questions.fingerprint,
            "target": store.fingerprint,
            "retrievers": sorted(f"{r.name}:{r.fingerprint}" for r in retrievers),
            "retrieval": {
                "depths": sorted(f"{r.name}:{getattr(r, 'default_limit', 20)}" for r in retrievers),
                "rrf_k": rrf_k,
                "max_candidates": policy.max_candidates,
            },
            "decider": decider.fingerprint,
            "policy": {
                "screen_floor": policy.screen_floor,
                "shortlist_size": policy.shortlist_size,
                "shortlist_floor": policy.shortlist_floor,
                "none_at": policy.none_at,
                "choose_at": policy.choose_at,
                "accept_at": policy.accept_at,
                "rubric_floor": policy.rubric_floor,
                "property_floor": policy.property_floor,
                "chunk_size": policy.chunk_size,
            },
        }
    )


@dataclass(frozen=True)
class BatchReport:
    run_fingerprint: str
    out_dir: Path
    total: int
    usage: Usage
    _ledger_path: Path

    def _ledger(self) -> Ledger:
        return Ledger.open(self._ledger_path)

    def by_status(self) -> dict[MatchStatus, int]:
        ledger = self._ledger()
        try:
            return ledger.count_by_status(self.run_fingerprint)
        finally:
            ledger.close()

    def needs_review(self) -> list[MatchResult]:
        ledger = self._ledger()
        try:
            return [
                r
                for r in ledger.iter_results(self.run_fingerprint)
                if r.status is MatchStatus.NEEDS_REVIEW
            ]
        finally:
            ledger.close()

    def duplicate_targets(self) -> dict[str, list[str]]:
        ledger = self._ledger()
        try:
            return ledger.duplicate_targets(self.run_fingerprint)
        finally:
            ledger.close()


async def run_batch(
    matcher: MatcherLike,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    manifest_extra: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open(out_dir / "ledger.sqlite")
    run_fp = matcher.run_fingerprint
    policy = matcher.policy

    ledger.put_manifest(
        run_fp,
        {
            "run_fingerprint": run_fp,
            "library_version": __version__,
            "target_fingerprint": matcher.store_fingerprint,
            **dict(manifest_extra or {}),
        },
    )

    semaphore = asyncio.Semaphore(policy.concurrency)
    total_usage = Usage.zero()
    completed = 0

    async def one(record: Record) -> None:
        nonlocal total_usage, completed
        key = result_key(run_fp, record.id, hash_record(record))
        if resume and ledger.has_result(key):
            completed += 1
            return
        async with semaphore:
            result = await matcher.match(record)
            # Commit inside the slot. If the write happened after release, a crash in
            # the next record could interleave ahead of this one's commit, and the
            # resume guarantee this whole module exists for would be probabilistic.
            await ledger.put_result(result)
        total_usage = total_usage + result.usage
        completed += 1
        if progress is not None:
            progress(result)

    try:
        # Chunked so an unbounded source does not materialise every task at once.
        pending: list[asyncio.Task[None]] = []
        for record in source:
            pending.append(asyncio.create_task(one(record)))
            if len(pending) >= policy.concurrency * 4:
                await asyncio.gather(*pending)
                pending = []
        if pending:
            await asyncio.gather(*pending)

        export_results_jsonl(ledger, run_fp, out_dir / "results.jsonl")
        export_mapping_csv(ledger, run_fp, out_dir / "mapping.csv")
        export_manifest(ledger, run_fp, out_dir / "manifest.json")

        return BatchReport(
            run_fingerprint=run_fp,
            out_dir=out_dir,
            total=ledger.count(run_fp),
            usage=total_usage,
            _ledger_path=out_dir / "ledger.sqlite",
        )
    finally:
        ledger.close()


def run_batch_sync(
    matcher: MatcherLike,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    manifest_extra: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport:
    return asyncio.run(
        run_batch(
            matcher,
            source,
            out=out,
            resume=resume,
            manifest_extra=manifest_extra,
            progress=progress,
        )
    )


# --- exports --------------------------------------------------------------------


def export_results_jsonl(ledger: Ledger, run_fingerprint: str, path: str | Path) -> int:
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for result in ledger.iter_results(run_fingerprint):
            handle.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")
            written += 1
    return written


def export_mapping_csv(
    ledger: Ledger,
    run_fingerprint: str,
    path: str | Path,
    *,
    use_review: bool = False,
) -> int:
    """The deliverable. `use_review=True` writes the adjudicated view instead."""
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MAPPING_COLUMNS))
        writer.writeheader()

        if use_review:
            for row in adjudicated(ledger, run_fingerprint):
                writer.writerow(
                    {
                        "source_id": row.source_id,
                        "matched_id": row.final_target_id or "",
                        "confidence": "" if row.confidence is None else row.confidence,
                        "status": row.final_status.value,
                        "reason": "reviewed" if row.reviewer else row.model_status.value,
                        "explanation": row.review_note,
                        "attempts": "",
                        "prompt_tokens": "",
                        "completion_tokens": "",
                        "llm_calls": "",
                        "elapsed_seconds": "",
                        "cost_usd": "",
                    }
                )
                written += 1
            return written

        for result in ledger.iter_results(run_fingerprint):
            writer.writerow(
                {
                    "source_id": result.source_id,
                    "matched_id": result.matched_id or "",
                    "confidence": "" if result.confidence is None else result.confidence,
                    "status": result.status.value,
                    "reason": result.reason.value,
                    "explanation": result.explanation,
                    "attempts": len(result.attempts),
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "llm_calls": result.usage.calls,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "cost_usd": result.usage.cost_usd,
                }
            )
            written += 1
    return written


def export_manifest(ledger: Ledger, run_fingerprint: str, path: str | Path) -> None:
    manifest = dict(ledger.get_manifest(run_fingerprint) or {})
    manifest["counts"] = {
        status.value: count for status, count in ledger.count_by_status(run_fingerprint).items()
    }
    manifest["duplicate_targets"] = ledger.duplicate_targets(run_fingerprint)
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

"""The decider loop: retrieve wide, screen everything, choose among survivors, gate the choice.

No retries: the LLM path retries because 25 candidates were the wrong 25. This path
retrieves 300 instead. One attempt per record, always index 0.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence

from xwalk.decide.base import DecisionError, DecisionFatalError
from xwalk.decide.policy import DecisionPolicy, Signals, derive_status, render_explanation
from xwalk.fingerprint import hash_record, result_key
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    Usage,
)
from xwalk.retrieval.base import Retriever
from xwalk.retrieve import retrieve
from xwalk.stages.choose import ChooseOutcome, Chooser
from xwalk.stages.keying import Resolution
from xwalk.stages.property_gate import GateOutcome, PropertyGate
from xwalk.stages.screen import Screener, ScreenOutcome
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet


class DecisionMatcher:
    def __init__(
        self,
        *,
        templates: TemplateSet,
        retrievers: Sequence[Retriever],
        store: TargetStore,
        screener: Screener,
        chooser: Chooser,
        gate: PropertyGate,
        policy: DecisionPolicy | None = None,
        run_fingerprint: str = "",
        rrf_k: int = 60,
        keep_candidates_in_trace: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self._templates = templates
        self._retrievers = list(retrievers)
        self._store = store
        self._screener = screener
        self._chooser = chooser
        self._gate = gate
        self._policy = policy or DecisionPolicy()
        self._run_fingerprint = run_fingerprint
        self._rrf_k = rrf_k
        self._keep_candidates = keep_candidates_in_trace

    @property
    def run_fingerprint(self) -> str:
        return self._run_fingerprint

    @property
    def policy(self) -> DecisionPolicy:
        return self._policy

    @property
    def store_fingerprint(self) -> str:
        return self._store.fingerprint

    async def match(self, source: Record) -> MatchResult:
        started = time.perf_counter()
        context = self._templates.render_context(source)
        queries = self._templates.render_queries(source)

        fused, notes, all_failed = await retrieve(
            queries,
            source,
            self._retrievers,
            self._store,
            timeout=self._policy.retriever_timeout,
            rrf_k=self._rrf_k,
        )
        candidates = fused[: self._policy.max_candidates]
        truncated = len(fused) - len(candidates)

        screen: ScreenOutcome | None = None
        chosen: ChooseOutcome | None = None
        gated: GateOutcome | None = None
        usage = Usage.zero()
        error: str | None = None
        provider_failed = False

        try:
            if candidates and not all_failed:
                screen = await self._screener.screen(source, context, candidates)
                usage = usage + screen.usage
                best = screen.best
                if best is not None and best >= self._policy.screen_floor and screen.shortlist:
                    by_id: dict[str, Candidate] = {c.id: c for c in candidates}
                    shortlist = [by_id[rid] for rid in screen.shortlist]
                    chosen = await self._chooser.choose(source, context, shortlist)
                    usage = usage + chosen.usage
                    if chosen.record_id is not None:
                        gated = await self._gate.gate(source, context, by_id[chosen.record_id])
                        usage = usage + gated.usage
        except DecisionFatalError:
            raise
        except DecisionError as exc:
            provider_failed = True
            error = f"decider: {exc}"

        chosen_id = None if chosen is None else chosen.record_id
        chosen_key = None
        if screen is not None and chosen_id is not None:
            chosen_key = next((k for k, rid in screen.issued.items() if rid == chosen_id), None)

        signals = Signals(
            candidate_count=len(candidates),
            retrieval_failed=all_failed,
            screen_best=None if screen is None else screen.best,
            screen_chosen=(
                None if screen is None or chosen_key is None else screen.probabilities[chosen_key]
            ),
            resolution=None if chosen is None else chosen.resolution.value,
            p_choice=None if chosen is None else chosen.p_choice,
            p_none=None if chosen is None else chosen.p_none,
            choice_confidence=None if chosen is None else chosen.confidence,
            rubric=None if gated is None else gated.rubric_score,
            rubric_levels=None if gated is None else gated.rubric_levels,
            rubric_confidence=None if gated is None else gated.rubric_confidence,
            properties={} if gated is None else dict(gated.properties),
        )

        if provider_failed:
            status, reason = MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE
        else:
            status, reason = derive_status(signals, self._policy)

        unmatched = status in (MatchStatus.UNMATCHED, MatchStatus.FAILED)
        matched_id = None if unmatched else chosen_id

        raw = json.dumps(
            {
                "served_model": None if screen is None else screen.model,
                "choose": None if chosen is None else json.loads(chosen.raw or "null"),
                "gate": None if gated is None else json.loads(gated.raw),
            },
            ensure_ascii=False,
        )
        all_notes = list(notes) + ([] if screen is None else list(screen.notes))
        joined = "; ".join(all_notes) if all_notes else None
        flat = signals.flat()
        if screen is not None:
            # Keep the shortlist's per-candidate probabilities so screen-stage recall can
            # be measured from the ledger. Only the shortlist, to keep the blob small.
            wanted = set(screen.shortlist)
            for key, rid in screen.issued.items():
                if rid in wanted:
                    flat[f"screen_{key}"] = screen.probabilities[key]
        explanation = render_explanation(signals)
        resolution = Resolution.ABSTAIN.value if chosen is None else chosen.resolution.value

        attempt = Attempt(
            index=0,
            query=queries[0] if queries else "",
            proposal=None,
            candidates=tuple(candidates) if self._keep_candidates else (),
            candidate_count=len(candidates),
            candidates_truncated=truncated,
            issued_keys={} if screen is None else dict(screen.issued),
            raw_selection=raw,
            chosen_id=chosen_id,
            resolution=resolution,
            primary_score=signals.screen_chosen,
            explanation=explanation,
            verifier_decision=None,
            verifier_score=None,
            verifier_preferred_id=None,
            audited=False,
            dropped_proposals=(),
            reason=reason,
            error=error if error else joined,
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
            finish_reason=None,
            signals=flat,
        )

        record = None
        if matched_id is not None:
            try:
                record = self._store.get(matched_id)
            except KeyError:
                record = None

        source_hash = hash_record(source)
        return MatchResult(
            result_key=result_key(self._run_fingerprint, source.id, source_hash),
            source_id=source.id,
            source_hash=source_hash,
            matched_id=matched_id,
            matched_record=record,
            confidence=signals.screen_chosen,
            status=status,
            reason=reason,
            explanation=explanation,
            candidates=tuple(candidates) if self._keep_candidates else (),
            attempts=(attempt,),
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
            run_fingerprint=self._run_fingerprint,
            signals=flat,
        )

    def match_sync(self, source: Record) -> MatchResult:
        return asyncio.run(self.match(source))


__all__ = ["DecisionMatcher"]

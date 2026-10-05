"""The decider loop: retrieve wide, screen everything, choose among survivors, gate the choice.

No retries: the LLM path retries because 25 candidates were the wrong 25. This path
retrieves 300 instead. One attempt per record, always index 0.

The one exception is opt-in: with a `QueryRewriter`, a first screen that found nothing
worth choosing from (no candidates, or a best probability below `screen_floor`) earns one
second retrieve-and-screen with queries an LLM proposes. The LLM only writes query
strings; it never sees the screened candidates and never decides anything.

Provider errors follow CONTRACTS.md section 3, as on the LLM path: `match` never raises
for them. A recoverable one (retries exhausted, a malformed response) makes the record
`failed` with `provider_failure`; a fatal one (auth, unknown model, a spent call limit),
from the decider or the rewrite LLM, makes it `failed` with `fatal_provider_failure`,
which `run_batch` does not commit and treats as a run abort.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence

from xwalk.decide.base import (
    DecisionError,
    DecisionFatalError,
    DecisionResponseError,
    failure_usage,
)
from xwalk.decide.policy import DecisionPolicy, Signals, derive_status, render_explanation
from xwalk.fingerprint import hash_record, result_key
from xwalk.llm.base import LLMFatalError
from xwalk.llm.base import failure_usage as llm_failure_usage
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
from xwalk.stages.keying import KeyedCandidates, Resolution
from xwalk.stages.property_gate import GateOutcome, PropertyGate
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.screen import Screener, ScreenFailed, ScreenOutcome
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
        rewriter: QueryRewriter | None = None,
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
        self._rewriter = rewriter

    @property
    def run_fingerprint(self) -> str:
        return self._run_fingerprint

    @property
    def rewriter(self) -> QueryRewriter | None:
        return self._rewriter

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
        fatal = False
        tried = list(queries)  # grows only if a second pass runs
        rewrite_notes: list[str] = []

        try:
            if candidates and not all_failed:
                screen = await self._screener.screen(source, context, candidates)
                usage = usage + screen.usage
            if self._rewriter is not None and not all_failed and self._missed(screen):
                extra, rewrite_usage, note = await self._rewrite(source, context, queries)
                usage = usage + rewrite_usage
                if extra:
                    tried.extend(extra)
                    fused2, notes2, failed2 = await retrieve(
                        extra,
                        source,
                        self._retrievers,
                        self._store,
                        timeout=self._policy.retriever_timeout,
                        rrf_k=self._rrf_k,
                    )
                    notes = list(notes) + list(notes2)
                    second = fused2[: self._policy.max_candidates]
                    truncated += len(fused2) - len(second)
                    seen = {c.id for c in candidates}
                    new = [c for c in second if c.id not in seen]
                    note = f"rewrite: {len(extra)} queries proposed, {len(new)} new candidates"
                    if second and not failed2:
                        screen2 = await self._screener.screen(source, context, second)
                        usage = usage + screen2.usage
                        if screen is None:
                            screen, candidates = screen2, list(second)
                        else:
                            candidates = list(candidates) + new
                            screen = self._screener.merge(screen, screen2, candidates)
                rewrite_notes.append(note)
            if screen is not None:
                best = screen.best
                if best is not None and best >= self._policy.screen_floor and screen.shortlist:
                    by_id: dict[str, Candidate] = {c.id: c for c in candidates}
                    shortlist = [by_id[rid] for rid in screen.shortlist]
                    chosen = await self._chooser.choose(source, context, shortlist)
                    usage = usage + chosen.usage
                    if chosen.record_id is not None:
                        gated = await self._gate.gate(source, context, by_id[chosen.record_id])
                        usage = usage + gated.usage
        except ScreenFailed as exc:
            # Sibling chunks were already paid for before the failing one; keep their
            # usage on the attempt so the run's cost is not understated.
            usage = usage + exc.usage
            provider_failed = True
            error = f"decider: {exc}"
        except DecisionResponseError as exc:
            usage = usage + failure_usage(exc)
            provider_failed = True
            error = f"decider: {exc}"
        except DecisionFatalError as exc:
            usage = usage + failure_usage(exc)
            provider_failed = fatal = True
            error = f"decider: {exc}"
        except LLMFatalError as exc:  # only the rewrite LLM gets here
            usage = usage + llm_failure_usage(exc)
            provider_failed = fatal = True
            error = f"rewrite: {exc}"
        except DecisionError as exc:
            usage = usage + failure_usage(exc)
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

        if fatal:
            status, reason = MatchStatus.FAILED, DecisionReason.FATAL_PROVIDER_FAILURE
        elif provider_failed:
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
        all_notes = list(notes) + ([] if screen is None else list(screen.notes)) + rewrite_notes
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
            # One query unless a rewrite ran; then every query tried, first pass first.
            query=" | ".join(tried)
            if len(tried) > len(queries)
            else (queries[0] if queries else ""),
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

    def _missed(self, screen: ScreenOutcome | None) -> bool:
        """Nothing to choose from: no candidates, or none cleared the screen floor."""
        best = None if screen is None else screen.best
        return best is None or best < self._policy.screen_floor

    async def _rewrite(
        self, source: Record, context: str, queries: Sequence[str]
    ) -> tuple[list[str], Usage, str]:
        """The rewriter's new queries, what the call cost, and the note for the trace.

        Any failure is a note: the first screen's result stands. The rewriter is handed
        the queries already tried and nothing about the candidates they returned.
        """
        assert self._rewriter is not None
        try:
            outcome = await self._rewriter.rewrite(source, context, queries, _NO_CANDIDATES)
        except LLMFatalError:
            raise  # auth, unknown model, a spent call limit: the run stops (section 3)
        except Exception as exc:  # noqa: BLE001 -- a rewrite is an extra, never a failure
            return [], llm_failure_usage(exc), f"rewrite: failed ({type(exc).__name__})"
        extra = [p.value for p in outcome.proposals]
        if outcome.error is not None:
            return [], outcome.usage, "rewrite: skipped (unparseable reply)"
        if not extra:
            return [], outcome.usage, "rewrite: skipped (no new queries)"
        return extra, outcome.usage, ""

    def match_sync(self, source: Record) -> MatchResult:
        return asyncio.run(self.match(source))


# The rewriter renders a "best candidates" section only when this is non-empty, so the
# prompt carries the source and the queries tried, and nothing the screen saw.
_NO_CANDIDATES = KeyedCandidates(order=(), by_key={}, issued={}, blocks={})

__all__ = ["DecisionMatcher"]

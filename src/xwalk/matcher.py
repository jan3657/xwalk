"""The matching loop. Orchestration only — every decision lives in a stage or in policy.

Stage-failure policy (CONTRACTS.md section 3). Every stage call goes through
`_call_stage`, which books the call's usage and classifies a provider error:

- recoverable (`LLMError` other than fatal, or a timeout) in the selector, scorer or
  gating verifier: the attempt gets `reason=provider_failure` and the same query is
  retried within `max_attempts`. A gating-verifier failure keeps the score with
  `verifier_decision="error"`, so that score can never become `matched`.
- recoverable in the audit verifier: the decision is unchanged; the error is noted.
- recoverable in the rewriter: the loop ends; status comes from the attempts so far.
- fatal (`LLMFatalError`) anywhere: the result is `failed` with
  `reason=fatal_provider_failure`. `match` never raises for a provider error.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TypeVar

from xwalk.fingerprint import hash_record, result_key
from xwalk.llm.base import LLMError, LLMFatalError, failure_usage
from xwalk.policy import MatchPolicy, derive_status, is_acceptable, should_audit, should_verify
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetryProposal,
    Usage,
)
from xwalk.retrieval.base import Retriever
from xwalk.retrieve import retrieve
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.keying import KeyedCandidates, Resolution
from xwalk.stages.proposals import RoutedProposals, normalise_query, route_proposals
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet

_T = TypeVar("_T")
_NO_PROPOSALS = RoutedProposals(candidate_keys=(), queries=(), dropped=())
_RETRY_SAME_QUERY = frozenset({DecisionReason.RETRIEVER_FAILURE, DecisionReason.PROVIDER_FAILURE})
# asyncio.TimeoutError is only an alias of TimeoutError from Python 3.11.
_RECOVERABLE_ERRORS = (LLMError, TimeoutError, asyncio.TimeoutError)


@dataclass(frozen=True)
class _StageFailure:
    """A stage call that raised a provider error, with the usage it consumed."""

    stage: str
    error: BaseException
    usage: Usage
    fatal: bool

    @property
    def reason(self) -> DecisionReason:
        return (
            DecisionReason.FATAL_PROVIDER_FAILURE if self.fatal else DecisionReason.PROVIDER_FAILURE
        )

    @property
    def note(self) -> str:
        return f"{self.stage}: {self.error}"


async def _call_stage(stage: str, call: Awaitable[_T]) -> _T | _StageFailure:
    """Await one stage call, turning a provider error into a booked `_StageFailure`.

    Anything that is not a provider error (a programming error, cancellation) is raised.
    """
    try:
        return await call
    except LLMFatalError as exc:
        return _StageFailure(stage, exc, failure_usage(exc), fatal=True)
    except _RECOVERABLE_ERRORS as exc:
        return _StageFailure(stage, exc, failure_usage(exc), fatal=False)


@dataclass(frozen=True)
class _AttemptOutcome:
    """What one attempt hands back to the loop.

    `keyed` and `candidates` are the live runtime state of the attempt. The loop reuses
    them for a candidate proposal and never reads them back from the `Attempt`, whose
    `candidates` field is trace data that `keep_candidates_in_trace` may leave empty.
    """

    attempt: Attempt
    routed: RoutedProposals
    keyed: KeyedCandidates | None
    candidates: tuple[Candidate, ...]


def _join_errors(*parts: str | None) -> str | None:
    kept = [p for p in parts if p]
    return "; ".join(kept) if kept else None


class Matcher:
    def __init__(
        self,
        *,
        templates: TemplateSet,
        retrievers: Sequence[Retriever],
        store: TargetStore,
        selector: Selector,
        scorer: Scorer,
        verifier: Verifier,
        rewriter: QueryRewriter,
        policy: MatchPolicy | None = None,
        run_fingerprint: str = "",
        retriever_limit: int = 20,
        rrf_k: int = 60,
        keep_candidates_in_trace: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self._templates = templates
        self._retrievers = list(retrievers)
        self._store = store
        self._selector = selector
        self._scorer = scorer
        self._verifier = verifier
        self._rewriter = rewriter
        self._policy = policy or MatchPolicy()
        self._run_fingerprint = run_fingerprint
        self._retriever_limit = retriever_limit
        self._rrf_k = rrf_k
        self._keep_candidates = keep_candidates_in_trace

    @property
    def run_fingerprint(self) -> str:
        return self._run_fingerprint

    @property
    def policy(self) -> MatchPolicy:
        return self._policy

    @property
    def store_fingerprint(self) -> str:
        return self._store.fingerprint

    # --- retrieval -------------------------------------------------------------

    async def _retrieve(self, query: str, source: Record) -> tuple[list[Candidate], list[str]]:
        """One query per attempt on this path; the rewriter supplies the next one."""
        candidates, notes, _ = await retrieve(
            [query],
            source,
            self._retrievers,
            self._store,
            timeout=self._policy.retriever_timeout,
            rrf_k=self._rrf_k,
            fallback_limit=self._retriever_limit,
        )
        return candidates, notes

    # --- one attempt -----------------------------------------------------------

    async def _attempt(
        self,
        source: Record,
        context: str,
        query: str,
        proposal: RetryProposal | None,
        index: int,
        reuse: tuple[KeyedCandidates, tuple[Candidate, ...]] | None,
        forced_key: str | None,
        seen_queries: AbstractSet[str],
    ) -> _AttemptOutcome:
        """Run one retrieve -> select -> score -> verify pass.

        Returns the attempt, its *already routed* proposals, and the `KeyedCandidates`
        and candidates actually used, so the caller can re-examine a candidate proposal
        without re-running retrieval. They are never reconstructed from the `Attempt`:
        `blocks` and `by_key` cannot be recovered from `issued_keys`, and the trace may
        omit the candidates.
        """
        started = time.perf_counter()
        usage = Usage.zero()
        notes: list[str] = []
        keyed: KeyedCandidates | None = None
        truncated = 0
        # Overwritten by each stage that talks to the provider; the last one wins,
        # which is the call that ended the attempt.
        finish_reason: str | None = None

        if reuse is not None:
            keyed, candidates = reuse
        else:
            fused, notes = await self._retrieve(query, source)
            candidates = tuple(fused)

        def build(
            *,
            chosen_id: str | None = None,
            resolution: str = Resolution.ABSTAIN.value,
            raw: str | None = None,
            score: float | None = None,
            explanation: str = "",
            verifier_decision: str | None = None,
            verifier_score: float | None = None,
            verifier_preferred_id: str | None = None,
            audited: bool = False,
            dropped: tuple[tuple[str, str], ...] = (),
            reason: DecisionReason | None = None,
            error: str | None = None,
        ) -> Attempt:
            return Attempt(
                index=index,
                query=query,
                proposal=proposal,
                candidates=candidates if self._keep_candidates else (),
                candidate_count=len(candidates),
                candidates_truncated=truncated,
                issued_keys=dict(keyed.issued) if keyed is not None else {},
                raw_selection=raw,
                chosen_id=chosen_id,
                resolution=resolution,
                primary_score=score,
                explanation=explanation,
                verifier_decision=verifier_decision,
                verifier_score=verifier_score,
                verifier_preferred_id=verifier_preferred_id,
                audited=audited,
                dropped_proposals=dropped,
                reason=reason,
                error=_join_errors(*notes, error),
                usage=usage,
                elapsed_seconds=time.perf_counter() - started,
                finish_reason=finish_reason,
            )

        def done(attempt: Attempt, routed: RoutedProposals = _NO_PROPOSALS) -> _AttemptOutcome:
            return _AttemptOutcome(attempt, routed, keyed, candidates)

        # Every retriever failed. A dead index is not evidence of a non-match.
        if not candidates and notes and len(notes) == len(self._retrievers):
            return done(build(reason=DecisionReason.RETRIEVER_FAILURE))

        if not candidates:
            return done(build(reason=DecisionReason.NO_CANDIDATES))

        # --- selection ---
        if forced_key is not None and keyed is not None:
            # A candidate proposal: the record is already in hand, so no new retrieval
            # and no second selector call. Go straight to scoring it.
            chosen_key: str = forced_key
            # Annotated because the selector branch below assigns `str | None` here.
            chosen_id: str | None = keyed.issued[forced_key]
            resolution = Resolution.EXACT_KEY.value
            selection_raw = f"(reused candidate proposal {forced_key})"
            selection_explanation = ""
        else:
            selection = await _call_stage(
                "selector", self._selector.select(source, context, candidates)
            )
            if isinstance(selection, _StageFailure):
                usage = usage + selection.usage
                return done(build(reason=selection.reason, error=selection.note))
            usage = usage + selection.usage
            finish_reason = selection.finish_reason
            keyed = selection.keyed
            truncated = selection.truncated
            chosen_id = selection.choice.record_id
            resolution = selection.choice.resolution.value
            selection_raw = selection.raw
            selection_explanation = selection.explanation
            resolved_key = next((k for k, rid in keyed.issued.items() if rid == chosen_id), None)

            if chosen_id is None or resolved_key is None:
                # Abstention or unresolvable output. Both are terminal for this attempt;
                # neither is worth a scorer call.
                return done(
                    build(
                        resolution=resolution,
                        raw=selection_raw,
                        explanation=selection_explanation,
                        error=selection.error,
                    )
                )
            # Bound to a narrowed local so `chosen_key` is `str`, not `str | None`, in
            # both branches — the scorer and verifier both require `str`.
            chosen_key = resolved_key

        assert keyed is not None

        # --- scoring, against the full source record, never the retrieval query ---
        scored = await _call_stage("scorer", self._scorer.score(source, context, keyed, chosen_key))
        if isinstance(scored, _StageFailure):
            usage = usage + scored.usage
            return done(
                build(
                    chosen_id=chosen_id,
                    resolution=resolution,
                    raw=selection_raw,
                    explanation=selection_explanation,
                    reason=scored.reason,
                    error=scored.note,
                )
            )
        usage = usage + scored.usage
        finish_reason = scored.finish_reason

        if scored.score is None:
            # The gate produced no usable number (missing, malformed, non-finite or out
            # of range). Falling back to the selector's own self-reported confidence
            # would let a malformed answer become an automatic MATCH on a score the gate
            # never gave; the spec routes malformed output to review instead.
            return done(
                build(
                    chosen_id=chosen_id,
                    resolution=Resolution.UNRESOLVED.value,
                    raw=selection_raw,
                    explanation=selection_explanation,
                    error=f"scorer: {scored.error}",
                )
            )

        score = scored.score
        routed = route_proposals(scored.proposals, keyed.order, seen_queries)
        dropped = tuple((p.value, why) for p, why in routed.dropped)
        explanation = scored.explanation or selection_explanation

        def build_scored(
            *,
            verifier_decision: str | None = None,
            verifier_score: float | None = None,
            verifier_preferred_id: str | None = None,
            audited: bool = False,
            reason: DecisionReason | None = None,
            error: str | None = None,
        ) -> Attempt:
            return build(
                chosen_id=chosen_id,
                resolution=resolution,
                raw=selection_raw,
                score=score,
                explanation=explanation,
                dropped=dropped,
                verifier_decision=verifier_decision,
                verifier_score=verifier_score,
                verifier_preferred_id=verifier_preferred_id,
                audited=audited,
                reason=reason,
                error=error,
            )

        # --- verification, as a cost control; audit, as a sample of the invisible ---
        gating = should_verify(score, self._policy)
        auditing = not gating and should_audit(
            score, self._policy, self._run_fingerprint, source.id
        )
        if not (gating or auditing):
            return done(build_scored(), routed)

        stage = "verifier" if gating else "audit verifier"
        verdict = await _call_stage(
            stage, self._verifier.verify(source, context, keyed, chosen_key)
        )
        if isinstance(verdict, _StageFailure):
            usage = usage + verdict.usage
            if verdict.fatal:
                attempt = build_scored(reason=verdict.reason, error=verdict.note)
            elif gating:
                # The score stands in the trace, but it is unverified: retried like any
                # provider failure, and never matched on its own.
                attempt = build_scored(
                    verifier_decision="error", reason=verdict.reason, error=verdict.note
                )
            else:
                # A failed audit changes nothing about the decision; it is only noted.
                attempt = build_scored(audited=True, error=verdict.note)
            return done(attempt, routed)

        usage = usage + verdict.usage
        finish_reason = verdict.finish_reason
        attempt = build_scored(
            verifier_decision=verdict.decision,
            verifier_score=verdict.confidence,
            verifier_preferred_id=(
                None if verdict.preferred_key is None else keyed.issued.get(verdict.preferred_key)
            ),
            audited=auditing,
            error=f"{stage}: {verdict.error}" if verdict.error else None,
        )
        return done(attempt, routed)

    # --- the loop --------------------------------------------------------------

    def _is_acceptable(self, attempt: Attempt) -> bool:
        return is_acceptable(attempt, self._policy)

    async def match(self, source: Record) -> MatchResult:
        match_started = time.perf_counter()
        context = self._templates.render_context(source)
        first_query = self._templates.render_query(source)

        queue: list[tuple[str, RetryProposal | None]] = [(first_query, None)]
        seen_queries = {normalise_query(first_query)}
        tried_queries: list[str] = []  # original casing, for the rewriter's prompt
        candidate_queue: list[str] = []
        attempts: list[Attempt] = []
        all_candidates: tuple[Candidate, ...] = ()
        last_keyed: KeyedCandidates | None = None
        last_candidates: tuple[Candidate, ...] = ()
        last_query = first_query

        for index in range(self._policy.max_attempts):
            reuse: tuple[KeyedCandidates, tuple[Candidate, ...]] | None = None
            forced_key: str | None = None
            proposal: RetryProposal | None = None

            if candidate_queue and last_keyed is not None:
                forced_key = candidate_queue.pop(0)
                proposal = RetryProposal(kind="candidate", value=forced_key, source="scorer")
                reuse = (last_keyed, last_candidates)
                query = last_query
            elif queue:
                query, proposal = queue.pop(0)
                last_query = query
                if query not in tried_queries:
                    tried_queries.append(query)
            else:
                break

            outcome = await self._attempt(
                source, context, query, proposal, index, reuse, forced_key, seen_queries
            )
            attempt = outcome.attempt
            attempts.append(attempt)
            if outcome.candidates:
                all_candidates = outcome.candidates
            if outcome.keyed is not None and outcome.keyed.order:
                last_keyed = outcome.keyed
                last_candidates = outcome.candidates

            if attempt.reason is DecisionReason.FATAL_PROVIDER_FAILURE:
                break  # a bad key or unknown model will not fix itself
            if self._is_acceptable(attempt):
                break
            if index + 1 >= self._policy.max_attempts:
                break

            if attempt.reason in _RETRY_SAME_QUERY:
                # Not evidence about this record. Retry the same query rather than
                # asking the rewriter to invent a new one for a provider outage.
                queue.insert(0, (query, proposal))
                continue

            candidate_queue.extend(outcome.routed.candidate_keys)
            for text in outcome.routed.queries:
                seen_queries.add(normalise_query(text))
                queue.append((text, RetryProposal(kind="query", value=text, source="scorer")))

            if not queue and not candidate_queue:
                rewritten = await _call_stage(
                    "rewriter",
                    self._rewriter.rewrite(
                        source, context, tried_queries, last_keyed or _empty_keyed()
                    ),
                )
                # The rewrite ran after this attempt, so this attempt carries its cost.
                if isinstance(rewritten, _StageFailure):
                    attempts[-1] = dataclasses.replace(
                        attempt,
                        usage=attempt.usage + rewritten.usage,
                        error=_join_errors(attempt.error, rewritten.note),
                        reason=rewritten.reason if rewritten.fatal else attempt.reason,
                    )
                    break
                attempts[-1] = dataclasses.replace(
                    attempt,
                    usage=attempt.usage + rewritten.usage,
                    error=_join_errors(
                        attempt.error,
                        f"rewriter: {rewritten.error}" if rewritten.error else None,
                    ),
                )
                for out in rewritten.proposals:
                    normalised = normalise_query(out.value)
                    if normalised in seen_queries:
                        continue
                    seen_queries.add(normalised)
                    queue.append((out.value, out))
                if not queue:
                    break

        status, reason, best = derive_status(attempts, self._policy)

        matched_id = None if best is None else best.chosen_id
        if status in (MatchStatus.UNMATCHED, MatchStatus.FAILED):
            matched_id = None

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
            confidence=None if best is None else best.primary_score,
            status=status,
            reason=reason,
            explanation="" if best is None else best.explanation,
            candidates=all_candidates if self._keep_candidates else (),
            attempts=tuple(attempts),
            usage=sum((a.usage for a in attempts), Usage.zero()),
            elapsed_seconds=time.perf_counter() - match_started,
            run_fingerprint=self._run_fingerprint,
        )

    def match_sync(self, source: Record) -> MatchResult:
        return asyncio.run(self.match(source))


def _empty_keyed() -> KeyedCandidates:
    return KeyedCandidates(order=(), by_key={}, issued={}, blocks={})

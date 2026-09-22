"""The matching loop. Orchestration only — every decision lives in a stage or in policy."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from xwalk.fingerprint import hash_record, result_key
from xwalk.llm.base import LLMError, LLMFatalError
from xwalk.policy import MatchPolicy, derive_status, should_audit, should_verify
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

_NO_PROPOSALS = RoutedProposals(candidate_keys=(), queries=(), dropped=())
_INFRASTRUCTURE_FAILURES = frozenset(
    {DecisionReason.RETRIEVER_FAILURE, DecisionReason.PROVIDER_FAILURE}
)


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
        reuse: tuple[KeyedCandidates, list[Candidate]] | None,
        forced_key: str | None,
        seen_queries: AbstractSet[str],
    ) -> tuple[Attempt, RoutedProposals, KeyedCandidates | None]:
        """Run one retrieve -> select -> score -> verify pass.

        Returns the attempt, its *already routed* proposals, and the `KeyedCandidates`
        actually issued — the caller keeps that object so a candidate proposal can be
        re-examined without re-running retrieval. It is never reconstructed from
        `issued_keys`: `rendered` and `by_key` cannot be recovered from a key->id map,
        and the scorer needs both.
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
            candidates, notes = await self._retrieve(query, source)

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
            joined = "; ".join(notes) if notes else None
            return Attempt(
                index=index,
                query=query,
                proposal=proposal,
                candidates=tuple(candidates) if self._keep_candidates else (),
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
                error=error if error else joined,
                usage=usage,
                elapsed_seconds=time.perf_counter() - started,
                finish_reason=finish_reason,
            )

        # Every retriever failed. A dead index is not evidence of a non-match.
        if not candidates and notes and len(notes) == len(self._retrievers):
            return build(reason=DecisionReason.RETRIEVER_FAILURE), _NO_PROPOSALS, None

        if not candidates:
            return build(reason=DecisionReason.NO_CANDIDATES), _NO_PROPOSALS, None

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
            try:
                selection = await self._selector.select(source, context, candidates)
            except LLMFatalError:
                raise  # a bad key or unknown model will not fix itself; let match() stop
            except LLMError as exc:
                return (
                    build(reason=DecisionReason.PROVIDER_FAILURE, error=f"selector: {exc}"),
                    _NO_PROPOSALS,
                    None,
                )
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
                return (
                    build(
                        resolution=resolution,
                        raw=selection_raw,
                        explanation=selection_explanation,
                        error=selection.error,
                    ),
                    _NO_PROPOSALS,
                    keyed,
                )
            # Bound to a narrowed local so `chosen_key` is `str`, not `str | None`, in
            # both branches — the scorer and verifier both require `str`.
            chosen_key = resolved_key

        assert keyed is not None

        # --- scoring, against the full source record, never the retrieval query ---
        try:
            scored = await self._scorer.score(source, context, keyed, chosen_key)
        except LLMFatalError:
            raise
        except LLMError as exc:
            return (
                build(
                    chosen_id=chosen_id,
                    resolution=resolution,
                    raw=selection_raw,
                    explanation=selection_explanation,
                    reason=DecisionReason.PROVIDER_FAILURE,
                    error=f"scorer: {exc}",
                ),
                _NO_PROPOSALS,
                keyed,
            )
        usage = usage + scored.usage
        finish_reason = scored.finish_reason

        if scored.score is None:
            # The gate produced no usable number. Falling back to the selector's own
            # self-reported confidence would let a malformed answer become an automatic
            # MATCH on a score the gate never gave; the spec routes malformed output to
            # review instead.
            return (
                build(
                    chosen_id=chosen_id,
                    resolution=Resolution.UNRESOLVED.value,
                    raw=selection_raw,
                    explanation=selection_explanation,
                    error=f"scorer: {scored.error}",
                ),
                _NO_PROPOSALS,
                keyed,
            )

        score = scored.score

        # --- verification, as a cost control; audit, as a sample of the invisible ---
        verdict = None
        audited = False
        if should_verify(score, self._policy):
            verdict = await self._verifier.verify(source, context, keyed, chosen_key)
        elif should_audit(score, self._policy, self._run_fingerprint, source.id):
            verdict = await self._verifier.verify(source, context, keyed, chosen_key)
            audited = True
        if verdict is not None:
            usage = usage + verdict.usage
            finish_reason = verdict.finish_reason

        routed = route_proposals(scored.proposals, keyed.order, seen_queries)

        attempt = build(
            chosen_id=chosen_id,
            resolution=resolution,
            raw=selection_raw,
            score=score,
            explanation=scored.explanation or selection_explanation,
            verifier_decision=None if verdict is None else verdict.decision,
            verifier_score=None if verdict is None else verdict.confidence,
            verifier_preferred_id=(
                None
                if verdict is None or verdict.preferred_key is None
                else keyed.issued.get(verdict.preferred_key)
            ),
            audited=audited,
            dropped=tuple((p.value, why) for p, why in routed.dropped),
        )
        return attempt, routed, keyed

    # --- the loop --------------------------------------------------------------

    def _is_acceptable(self, attempt: Attempt) -> bool:
        """Good enough to stop early: a scored, exactly-resolved, unchallenged match."""
        return (
            attempt.primary_score is not None
            and attempt.primary_score >= self._policy.accept_at
            and attempt.resolution == Resolution.EXACT_KEY.value
            and attempt.verifier_decision not in ("disagree", "no_match")
        )

    def _fatal_attempt(
        self, index: int, query: str, proposal: RetryProposal | None, exc: Exception
    ) -> Attempt:
        return Attempt(
            index=index,
            query=query,
            proposal=proposal,
            candidates=(),
            candidate_count=0,
            candidates_truncated=0,
            issued_keys={},
            raw_selection=None,
            chosen_id=None,
            resolution=Resolution.ABSTAIN.value,
            primary_score=None,
            explanation="",
            verifier_decision=None,
            verifier_score=None,
            verifier_preferred_id=None,
            audited=False,
            dropped_proposals=(),
            reason=DecisionReason.PROVIDER_FAILURE,
            error=str(exc),
            usage=Usage.zero(),
            elapsed_seconds=0.0,
            finish_reason=None,
        )

    async def match(self, source: Record) -> MatchResult:
        match_started = time.perf_counter()
        context = self._templates.render_context(source)
        first_query = self._templates.render_query(source)

        queue: list[tuple[str, RetryProposal | None]] = [(first_query, None)]
        seen_queries = {normalise_query(first_query)}
        tried_queries: list[str] = []  # original casing, for the rewriter's prompt
        candidate_queue: list[str] = []
        attempts: list[Attempt] = []
        all_candidates: list[Candidate] = []
        last_keyed: KeyedCandidates | None = None
        last_candidates: list[Candidate] = []
        last_query = first_query

        for index in range(self._policy.max_attempts):
            reuse: tuple[KeyedCandidates, list[Candidate]] | None = None
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

            try:
                attempt, routed, keyed = await self._attempt(
                    source, context, query, proposal, index, reuse, forced_key, seen_queries
                )
            except LLMFatalError as exc:
                attempts.append(self._fatal_attempt(index, query, proposal, exc))
                break

            attempts.append(attempt)
            if attempt.candidates:
                all_candidates = list(attempt.candidates)
            if keyed is not None and keyed.order:
                last_keyed = keyed
                last_candidates = list(attempt.candidates) or last_candidates

            if self._is_acceptable(attempt):
                break
            if index + 1 >= self._policy.max_attempts:
                break

            if attempt.reason in _INFRASTRUCTURE_FAILURES:
                # Not evidence about this record. Retry the same query rather than
                # asking the rewriter to invent a new one for a provider outage.
                queue.insert(0, (query, proposal))
                continue

            candidate_queue.extend(routed.candidate_keys)
            for text in routed.queries:
                seen_queries.add(normalise_query(text))
                queue.append((text, RetryProposal(kind="query", value=text, source="scorer")))

            if not queue and not candidate_queue:
                rewritten = await self._rewriter.rewrite(
                    source, context, tried_queries, last_keyed or _empty_keyed()
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
            candidates=tuple(all_candidates) if self._keep_candidates else (),
            attempts=tuple(attempts),
            usage=sum((a.usage for a in attempts), Usage.zero()),
            elapsed_seconds=time.perf_counter() - match_started,
            run_fingerprint=self._run_fingerprint,
        )

    def match_sync(self, source: Record) -> MatchResult:
        return asyncio.run(self.match(source))


def _empty_keyed() -> KeyedCandidates:
    return KeyedCandidates(order=(), by_key={}, issued={}, rendered="")

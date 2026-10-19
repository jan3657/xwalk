"""Pure policy reduction of recorded evidence, conditional on the recorded attempts.

This is not a simulation of a new run: a different acceptance threshold could have
stopped earlier or bought more attempts. No evidence is acquired here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from xwalk.decide.policy import STATUS_SEMANTICS_VERSION as DECISION_STATUS_SEMANTICS_VERSION
from xwalk.decide.policy import DecisionPolicy, Signals
from xwalk.decide.policy import derive_status as derive_decision_status
from xwalk.evaluate.gold import GoldSet
from xwalk.policy import STATUS_SEMANTICS_VERSION, MatchPolicy, derive_status, is_valid_score
from xwalk.records import Attempt, DecisionReason, MatchResult, MatchStatus

Policy = MatchPolicy | DecisionPolicy
CONDITIONAL = (
    "Conditional on recorded attempts; changed thresholds may change stopping in a new run."
)
HISTORICAL_REDUCER = (
    "historical or unknown reducer version: status changes may reflect current "
    "reduction semantics as well as changed thresholds"
)
RETRY_DEFINITION = (
    "descriptive, not causal: recovered = first attempt's chosen id is not gold and final "
    "status is matched with a gold id; broken_by_retry = first choice is gold, more than "
    "one attempt, and final answer is not an accepted gold id. First choice ignores its "
    "score/status. Only labelled rows with stored attempts count."
)
_CLASSIFICATION = {
    MatchPolicy: frozenset({"accept_at", "review_floor"}),
    DecisionPolicy: frozenset(
        {"accept_at", "choose_at", "none_at", "rubric_floor", "property_floor"}
    ),
}


def status_semantics_version(policy: Policy) -> int:
    return (
        STATUS_SEMANTICS_VERSION
        if isinstance(policy, MatchPolicy)
        else DECISION_STATUS_SEMANTICS_VERSION
    )


def policy_from_components(components: Mapping[str, Any]) -> Policy | None:
    """Read the stored policy identity without instantiating any run machinery."""
    raw = components.get("policy")
    if not isinstance(raw, dict):
        return None
    policy_type = DecisionPolicy if components.get("path") == "decider" else MatchPolicy
    names = {f.name for f in fields(policy_type)}
    unknown = set(raw) - names - {"status_semantics"}
    if unknown:
        raise ValueError(f"unsupported recorded policy fields: {', '.join(sorted(unknown))}")
    values = {k: v for k, v in raw.items() if k in names}
    retrieval = components.get("retrieval", {})
    for name in ("retriever_timeout", "max_candidates"):
        if name in names and name in retrieval:
            values[name] = retrieval[name]
    if values.get("verify_band") is not None:
        values["verify_band"] = tuple(values["verify_band"])
    return policy_type(**values)


def validate_replay_policy(recorded: Policy, proposed: Policy) -> None:
    """Refuse changes to evidence acquisition, routing or attempt bounds."""
    if type(recorded) is not type(proposed):
        raise ValueError("changing matcher path requires unrecorded calls")
    changed = [
        f.name
        for f in fields(recorded)
        if f.name not in _CLASSIFICATION[type(recorded)] | {"concurrency"}
        and getattr(recorded, f.name) != getattr(proposed, f.name)
    ]
    if changed:
        raise ValueError(f"policy changes require unrecorded calls: {', '.join(changed)}")


def replay_result(result: MatchResult, policy: Policy) -> MatchResult:
    """Reclassify one result without mutating it or writing back to its ledger.

    The policy type selects the path, including decider failures with empty signals.
    Caller supplies acquisition-compatible policy; replay_results validates changes.
    """
    if not result.attempts:
        raise ValueError(f"{result.source_id}: replay unavailable without recorded attempts")
    if isinstance(policy, MatchPolicy):
        status, reason, best = derive_status(result.attempts, policy)
    else:
        if len(result.attempts) != 1:
            raise ValueError("DecisionMatcher replay requires its single aggregate attempt")
        best = result.attempts[0]
        # DecisionMatcher catches provider failures outside its numerical reducer.
        if best.reason in (DecisionReason.PROVIDER_FAILURE, DecisionReason.FATAL_PROVIDER_FAILURE):
            status, reason = MatchStatus.FAILED, best.reason
        else:
            signals = Signals.from_flat(
                best.signals,
                candidate_count=best.candidate_count,
                retrieval_failed=(
                    best.candidate_count == 0
                    and (
                        bool(best.retrieval_failures)
                        or best.reason is DecisionReason.RETRIEVER_FAILURE
                    )
                ),
                # Attempts store "abstain" even when no chooser ran. A chooser
                # outcome always records p_choice, including zero probability.
                resolution=best.resolution if "p_choice" in best.signals else None,
            )
            signals = replace(signals, retrieval_degraded=bool(best.retrieval_failures))
            status, reason = derive_decision_status(signals, policy)
    matched_id = None if best is None else best.chosen_id
    if status in (MatchStatus.FAILED, MatchStatus.UNMATCHED):
        matched_id = None
    record = result.matched_record if matched_id == result.matched_id else None
    if record is None and matched_id is not None:
        record = next(
            (c.record for a in result.attempts for c in a.candidates if c.id == matched_id), None
        )
    return replace(
        result,
        status=status,
        reason=reason,
        matched_id=matched_id,
        matched_record=record,
        confidence=None if best is None else best.primary_score,
        explanation="" if best is None else best.explanation,
    )


def _gold_outcome(row: MatchResult, gold: GoldSet) -> str:
    expected = gold.get(row.source_id)
    if expected is None:
        return "unlabelled"
    if row.status is MatchStatus.MATCHED:
        return "accepted_correct" if row.matched_id in expected else "accepted_wrong"
    if row.status is MatchStatus.UNMATCHED:
        return "correct_no_match" if not expected else "incorrect_no_match"
    return "review" if row.status is MatchStatus.NEEDS_REVIEW else "failed"


def _selector_ids(attempt: Attempt, policy: Policy) -> set[str]:
    if isinstance(policy, MatchPolicy):
        return set(attempt.issued_keys.values())
    # Decider keys name the whole screened slate, but only shortlisted candidates
    # reached the chooser. Their individual probabilities are retained in signals.
    return {rid for key, rid in attempt.issued_keys.items() if f"screen_{key}" in attempt.signals}


def decision_diagnostics(
    results: Sequence[MatchResult], gold: GoldSet, policy: Policy
) -> dict[str, int]:
    """Overlapping row counts on non-accepted gold positives, separate from ceiling.

    A reviewed correct target is FOUND for retrieval but may be suppressed here.
    selector_wrong means another *shown* candidate was chosen, not abstention.
    lost_by_retry means a previously chosen gold id is absent from the final proposal.
    """
    counts = dict.fromkeys(
        ("selector_wrong", "scorer_below", "verifier_disagreed", "lost_by_retry"), 0
    )
    for row in results:
        expected = gold.get(row.source_id)
        if not expected or (row.status is MatchStatus.MATCHED and row.matched_id in expected):
            continue
        attempts = row.attempts
        if any(
            expected.intersection(shown := _selector_ids(a, policy))
            and a.chosen_id in shown
            and a.chosen_id not in expected
            for a in attempts
        ):
            counts["selector_wrong"] += 1
        if any(
            a.chosen_id in expected
            and is_valid_score(a.primary_score)
            and (a.primary_score or 0.0) < policy.accept_at
            for a in attempts
        ):
            counts["scorer_below"] += 1
        if any(
            a.chosen_id in expected and a.verifier_decision in ("disagree", "no_match")
            for a in attempts
        ):
            counts["verifier_disagreed"] += 1
        if (
            len(attempts) > 1
            and row.matched_id not in expected
            and any(a.chosen_id in expected for a in attempts[:-1])
        ):
            counts["lost_by_retry"] += 1
    return counts


def retry_metrics(results: Sequence[MatchResult], gold: GoldSet) -> dict[str, int | None]:
    rows = [r for r in results if r.source_id in gold and r.attempts]
    if not rows:
        return {"recovered": None, "broken_by_retry": None}
    recovered = broken = 0
    for row in rows:
        expected = gold.get(row.source_id) or frozenset()
        first_ok = row.attempts[0].chosen_id in expected
        final_ok = row.status is MatchStatus.MATCHED and row.matched_id in expected
        recovered += int(not first_ok and final_ok)
        broken += int(first_ok and not final_ok and len(row.attempts) > 1)
    return {"recovered": recovered, "broken_by_retry": broken}


@dataclass(frozen=True)
class ReplayReport:
    results: tuple[MatchResult, ...]
    status_transitions: dict[str, dict[str, int]]
    gold_transitions: dict[str, dict[str, int]]
    misjudged: dict[str, int]
    retry_metrics: dict[str, int | None]
    warnings: tuple[str, ...]
    status_semantics_version: int
    recorded_semantics_version: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "conditional_on_recorded_attempts": True,
            "definition": CONDITIONAL,
            "status_semantics_version": self.status_semantics_version,
            "recorded_semantics_version": self.recorded_semantics_version,
            "status_transitions": self.status_transitions,
            "gold_transitions": self.gold_transitions,
            "misjudged": self.misjudged,
            "misjudged_definition": (
                "overlapping row counts on non-accepted gold positives; "
                "separate from retrieval ceiling"
            ),
            "retry_metrics": self.retry_metrics,
            "retry_definition": RETRY_DEFINITION,
            "warnings": list(self.warnings),
            "results": [
                {
                    "source_id": r.source_id,
                    "status": r.status.value,
                    "reason": r.reason.value,
                    "matched_id": r.matched_id,
                    "confidence": r.confidence,
                }
                for r in self.results
            ],
        }


def replay_results(
    results: Sequence[MatchResult],
    gold: GoldSet,
    *,
    recorded_policy: Policy,
    policy: Policy | None = None,
    recorded_semantics_version: int | None = None,
) -> ReplayReport:
    policy = policy or recorded_policy
    validate_replay_policy(recorded_policy, policy)
    replayed = tuple(replay_result(r, policy) for r in results)
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for before, after in zip(results, replayed, strict=True):
        statuses[before.status.value][after.status.value] += 1
        outcomes[_gold_outcome(before, gold)][_gold_outcome(after, gold)] += 1
    version = status_semantics_version(policy)
    warnings = [CONDITIONAL]
    if recorded_semantics_version != version:
        warnings.append(HISTORICAL_REDUCER)
    if (
        isinstance(policy, MatchPolicy)
        and isinstance(recorded_policy, MatchPolicy)
        and policy.review_floor != recorded_policy.review_floor
    ):
        warnings.append("changed review_floor may alter scorer proposals/routing in a new run")
    return ReplayReport(
        results=replayed,
        status_transitions={k: dict(v) for k, v in statuses.items()},
        gold_transitions={k: dict(v) for k, v in outcomes.items()},
        misjudged=decision_diagnostics(replayed, gold, policy),
        retry_metrics=retry_metrics(replayed, gold),
        warnings=tuple(warnings),
        status_semantics_version=version,
        recorded_semantics_version=recorded_semantics_version,
    )

import pytest

from xwalk.policy import MatchPolicy, derive_status, should_audit, should_verify
from xwalk.records import Attempt, DecisionReason, MatchStatus, Usage
from xwalk.stages.keying import Resolution

POLICY = MatchPolicy()


def attempt(**kwargs) -> Attempt:
    base = dict(
        index=0,
        query="q",
        proposal=None,
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
        reason=None,
        error=None,
        usage=Usage.zero(),
        elapsed_seconds=0.0,
        finish_reason="stop",
    )
    base.update(kwargs)
    return Attempt(**base)


def matched(score: float, **kwargs) -> Attempt:
    # Defaults merged rather than passed through, so a caller may override `resolution`
    # (which `test_a_legacy_resolution_forces_review_even_at_a_high_score` does).
    base = dict(
        chosen_id="T1",
        resolution=Resolution.EXACT_KEY.value,
        primary_score=score,
        candidate_count=3,
    )
    base.update(kwargs)
    return attempt(**base)


# --- classification --------------------------------------------------------------


def test_a_high_score_is_matched():
    status, reason, best = derive_status([matched(0.9)], POLICY)
    assert status is MatchStatus.MATCHED
    assert reason is DecisionReason.ACCEPT_THRESHOLD
    assert best is not None and best.chosen_id == "T1"


def test_a_score_in_the_review_band_needs_review():
    status, reason, _ = derive_status([matched(0.5)], POLICY)
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.BELOW_ACCEPT_THRESHOLD


def test_a_score_below_the_floor_is_unmatched():
    status, reason, _ = derive_status([matched(0.2)], POLICY)
    assert status is MatchStatus.UNMATCHED
    assert reason is DecisionReason.BELOW_REVIEW_FLOOR


def test_the_accept_threshold_is_inclusive():
    status, _, _ = derive_status([matched(0.6)], POLICY)
    assert status is MatchStatus.MATCHED


def test_the_review_floor_is_inclusive():
    status, reason, _ = derive_status([matched(0.4)], POLICY)
    assert reason is DecisionReason.BELOW_ACCEPT_THRESHOLD


# --- abstention, no candidates, unresolved ---------------------------------------


def test_an_explicit_abstention_is_unmatched():
    status, reason, _ = derive_status([attempt(candidate_count=3)], POLICY)
    assert status is MatchStatus.UNMATCHED
    assert reason is DecisionReason.SELECTOR_ABSTAINED


def test_no_candidates_is_distinguishable_from_abstention():
    status, reason, _ = derive_status([attempt(candidate_count=0)], POLICY)
    assert reason is DecisionReason.NO_CANDIDATES


def test_unresolved_output_routes_to_review_not_to_silence():
    status, reason, _ = derive_status(
        [attempt(candidate_count=3, resolution=Resolution.UNRESOLVED.value, raw_selection="???")],
        POLICY,
    )
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.UNRESOLVED_OUTPUT


# --- precedence ------------------------------------------------------------------


def test_a_weak_pick_outranks_an_earlier_abstention():
    """The model found something worth surfacing; that beats an earlier shrug."""
    status, reason, best = derive_status([attempt(candidate_count=3), matched(0.45)], POLICY)
    assert reason is DecisionReason.BELOW_ACCEPT_THRESHOLD
    assert best is not None and best.primary_score == 0.45


def test_the_highest_scoring_attempt_wins():
    _, _, best = derive_status([matched(0.5), matched(0.8), matched(0.3)], POLICY)
    assert best is not None and best.primary_score == 0.8


def test_ties_resolve_to_the_earlier_attempt():
    _, _, best = derive_status([matched(0.8, index=0), matched(0.8, index=1)], POLICY)
    assert best is not None and best.index == 0


def test_verifier_disagreement_overrides_a_passing_score():
    status, reason, _ = derive_status([matched(0.95, verifier_decision="disagree")], POLICY)
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.VERIFIER_DISAGREEMENT


def test_verifier_no_match_also_forces_review():
    status, reason, _ = derive_status([matched(0.95, verifier_decision="no_match")], POLICY)
    assert reason is DecisionReason.VERIFIER_DISAGREEMENT


def test_verifier_support_leaves_the_classification_alone():
    status, reason, _ = derive_status([matched(0.95, verifier_decision="support")], POLICY)
    assert status is MatchStatus.MATCHED


def test_a_legacy_resolution_forces_review_even_at_a_high_score():
    status, reason, _ = derive_status(
        [matched(0.95, resolution=Resolution.LEGACY_RANK.value)], POLICY
    )
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.UNRESOLVED_OUTPUT


# --- failure ---------------------------------------------------------------------


def test_all_attempts_erroring_is_failed_not_unmatched():
    """A provider 500 is not evidence of a non-match."""
    status, reason, _ = derive_status(
        [attempt(error="HTTP 500", reason=DecisionReason.PROVIDER_FAILURE)] * 2, POLICY
    )
    assert status is MatchStatus.FAILED
    assert reason is DecisionReason.PROVIDER_FAILURE


def test_retriever_failure_is_distinguishable_from_provider_failure():
    status, reason, _ = derive_status(
        [attempt(error="timeout", reason=DecisionReason.RETRIEVER_FAILURE)], POLICY
    )
    assert reason is DecisionReason.RETRIEVER_FAILURE


def test_one_good_attempt_beats_one_failed_attempt():
    status, _, _ = derive_status(
        [attempt(error="HTTP 500", reason=DecisionReason.PROVIDER_FAILURE), matched(0.9)], POLICY
    )
    assert status is MatchStatus.MATCHED


def test_no_attempts_at_all_is_failed():
    status, reason, best = derive_status([], POLICY)
    assert status is MatchStatus.FAILED and best is None


# --- verify band and audit -------------------------------------------------------


def test_verify_band_is_inclusive_at_both_ends():
    policy = MatchPolicy(verify_band=(0.6, 0.8))
    assert should_verify(0.6, policy) and should_verify(0.8, policy)
    assert not should_verify(0.59, policy) and not should_verify(0.81, policy)


def test_verify_band_of_none_disables_verification():
    assert not should_verify(0.7, MatchPolicy(verify_band=None))


def test_a_none_score_is_never_verified():
    assert not should_verify(None, MatchPolicy())


def test_audit_is_off_by_default():
    assert not should_audit(0.95, MatchPolicy(), "fp", "s1")


def test_audit_only_applies_above_the_verify_band():
    policy = MatchPolicy(audit_rate=1.0, verify_band=(0.6, 0.8))
    assert should_audit(0.95, policy, "fp", "s1")
    assert not should_audit(0.7, policy, "fp", "s1")


def test_audit_sampling_is_deterministic_for_the_same_run_and_record():
    policy = MatchPolicy(audit_rate=0.5)
    first = should_audit(0.95, policy, "fp", "s1")
    second = should_audit(0.95, policy, "fp", "s1")
    assert first == second


def test_audit_sampling_differs_across_records():
    policy = MatchPolicy(audit_rate=0.5)
    picks = [should_audit(0.95, policy, "fp", f"s{i}") for i in range(200)]
    assert 0 < sum(picks) < 200


def test_audit_rate_is_approximately_honoured():
    policy = MatchPolicy(audit_rate=0.2)
    picks = [should_audit(0.95, policy, "fp", f"s{i}") for i in range(2000)]
    assert 0.15 < sum(picks) / 2000 < 0.25


# --- policy validation -----------------------------------------------------------


def test_review_floor_above_accept_at_is_rejected():
    with pytest.raises(ValueError, match="review_floor"):
        MatchPolicy(accept_at=0.5, review_floor=0.7)


def test_a_negative_audit_rate_is_rejected():
    with pytest.raises(ValueError, match="audit_rate"):
        MatchPolicy(audit_rate=-0.1)


def test_an_inverted_verify_band_is_rejected():
    with pytest.raises(ValueError, match="verify_band"):
        MatchPolicy(verify_band=(0.8, 0.6))


def test_max_attempts_below_one_is_rejected():
    with pytest.raises(ValueError, match="max_attempts"):
        MatchPolicy(max_attempts=0)

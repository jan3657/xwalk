from xwalk.decide.fit import fit_thresholds, render_fit
from xwalk.decide.policy import DecisionPolicy
from xwalk.evaluate.gold import GoldSet
from xwalk.records import Attempt, DecisionReason, MatchResult, MatchStatus, Usage


def _result(sid, matched, screen, prop=0.9):
    signals = {
        "screen_best": screen,
        "screen_chosen": screen,
        "p_choice": 0.8,
        "p_none": 0.05,
        "choice_confidence": 0.7,
        "rubric": 2.5,
        "rubric_levels": 3.0,
        "rubric_confidence": 0.8,
        "prop_form": prop,
    }
    attempt = Attempt(
        index=0,
        query="q",
        proposal=None,
        candidates=(),
        candidate_count=5,
        candidates_truncated=0,
        issued_keys={},
        raw_selection=None,
        chosen_id=matched,
        resolution="exact_key",
        primary_score=screen,
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
        finish_reason=None,
        signals=signals,
    )
    return MatchResult(
        result_key=sid,
        source_id=sid,
        source_hash="h",
        matched_id=matched,
        matched_record=None,
        confidence=screen,
        status=MatchStatus.NEEDS_REVIEW,
        reason=DecisionReason.BELOW_ACCEPT_THRESHOLD,
        explanation="",
        candidates=(),
        attempts=(attempt,),
        usage=Usage.zero(),
        elapsed_seconds=0.0,
        run_fingerprint="fp",
        signals=signals,
    )


GOLD = GoldSet(
    {
        "s1": frozenset({"T1"}),
        "s2": frozenset({"T2"}),
        "s3": frozenset({"T3"}),
        "s4": frozenset({"T4"}),
    }
)
RESULTS = [
    _result("s1", "T1", 0.95),  # right, confident
    _result("s2", "T2", 0.90),  # right
    _result("s3", "T9", 0.70),  # wrong, mid
    _result("s4", "T4", 0.60, prop=0.2),  # right, but a property disagrees
]


def test_recommends_the_widest_threshold_that_meets_precision():
    report = fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=1.0)
    assert report.recommended is not None
    assert report.recommended.accept_at > 0.70
    assert report.recommended.correct == report.recommended.accepted


def test_a_lower_target_admits_the_wrong_one():
    report = fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=0.6)
    assert report.recommended is not None
    assert report.recommended.accept_at <= 0.70
    assert report.recommended.accepted >= 3


def test_near_threshold_counts_rows_inside_the_margin():
    report = fit_thresholds(
        RESULTS, GOLD, base=DecisionPolicy(), accept_grid=(0.9,), property_grid=(0.0,), margin=0.06
    )
    (point,) = report.points
    assert point.near_threshold == 2  # 0.95 and 0.90


def test_render_mentions_the_recommendation_or_its_absence():
    assert "recommended" in render_fit(
        fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=1.0)
    )
    assert "no grid point" in render_fit(
        fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=1.01)
    )

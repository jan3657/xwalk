from xwalk.decide.fit import (
    FitPoint,
    fit_holdout,
    fit_thresholds,
    holdout_split,
    render_fit,
    render_holdout,
)
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
        RESULTS,
        GOLD,
        base=DecisionPolicy(),
        accept_grid=(0.9,),
        property_grid=(0.0,),
        choose_grid=(0.5,),
        margin=0.06,
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


def test_the_sweep_varies_choose_at():
    report = fit_thresholds(
        RESULTS, GOLD, base=DecisionPolicy(), accept_grid=(0.9,), property_grid=(0.0,)
    )
    assert sorted(p.choose_at for p in report.points) == [0.3, 0.5, 0.7]
    # p_choice is 0.8 on every row, so a choose_at above it accepts nothing.
    strict = fit_thresholds(
        RESULTS,
        GOLD,
        base=DecisionPolicy(),
        accept_grid=(0.5,),
        property_grid=(0.0,),
        choose_grid=(0.9,),
    )
    assert [p.accepted for p in strict.points if p.choose_at == 0.9] == [0]


def test_the_runs_own_choose_at_is_always_swept():
    report = fit_thresholds(
        RESULTS,
        GOLD,
        base=DecisionPolicy(choose_at=0.6),
        accept_grid=(0.9,),
        property_grid=(0.0,),
    )
    assert sorted(p.choose_at for p in report.points) == [0.3, 0.5, 0.6, 0.7]


def test_holdout_split_is_deterministic_and_disjoint():
    ids = [f"id{i}" for i in range(17)]
    dev, holdout = holdout_split(ids, "fp-abc")
    again = holdout_split(list(reversed(ids)), "fp-abc")
    assert (dev, holdout) == again
    assert not set(dev) & set(holdout)
    assert set(dev) | set(holdout) == set(ids)
    assert abs(len(dev) - len(holdout)) <= 1
    assert holdout_split(ids, "fp-other") != (dev, holdout)


def _sixteen():
    """The four synthetic rows above, each pattern repeated four times under fresh ids."""
    patterns = [("T1", 0.95, 0.9, True), ("T2", 0.90, 0.9, True), ("T9", 0.70, 0.9, False)]
    patterns.append(("T4", 0.60, 0.2, True))
    results, gold, kind = [], {}, {}
    for i in range(16):
        matched, screen, prop, right = patterns[i % 4]
        sid = f"r{i}"
        results.append(_result(sid, matched, screen, prop=prop))
        gold[sid] = frozenset({matched if right else "T3"})
        kind[sid] = i % 4
    return results, GoldSet(gold), kind


def test_holdout_reports_the_recommended_point_on_unseen_rows():
    results, gold, kind = _sixteen()
    report, point = fit_holdout(
        results, gold, base=DecisionPolicy(), seed_fingerprint="fp", target_precision=1.0
    )
    _, holdout = holdout_split([r.source_id for r in results], "fp")
    assert report.labelled == 16 - len(holdout)  # the report is fitted on dev only
    r = report.recommended
    assert r is not None
    # At precision 1.0 the wrong row (0.70) must be refused, so accept_at > 0.70 and only
    # the 0.95 and 0.90 patterns are accepted. Count them by hand over holdout ids only.
    assert 0.70 < r.accept_at <= 0.90
    expected = sum(1 for sid in holdout if kind[sid] in (0, 1))
    assert expected < 8  # all 16 rows would give 8: the count really is holdout-only
    assert point is not None
    assert point.accepted == expected
    assert point.correct == expected
    assert point.coverage == expected / len(holdout)
    assert point.precision == (1.0 if expected else None)
    assert (point.accept_at, point.property_floor, point.choose_at) == (
        r.accept_at,
        r.property_floor,
        r.choose_at,
    )


def test_holdout_has_no_point_without_a_recommendation():
    results, gold, _ = _sixteen()
    report, point = fit_holdout(
        results, gold, base=DecisionPolicy(), seed_fingerprint="fp", target_precision=1.01
    )
    assert report.recommended is None
    assert point is None


def _point(accept_at, property_floor, choose_at):
    return FitPoint(
        accept_at=accept_at,
        property_floor=property_floor,
        choose_at=choose_at,
        accepted=3,
        correct=2,
        precision=2 / 3,
        coverage=0.5,
        near_threshold=0,
    )


def test_holdout_line_names_the_point_it_scored():
    line = render_holdout(_point(0.9, 0.5, 0.7), _point(0.9, 0.5, 0.7))
    assert line == (
        "holdout: accept_at=0.90 property_floor=0.50 choose_at=0.70 "
        "accepted=3 correct=2 precision=0.67 coverage=0.50"
    )


def test_holdout_line_says_when_the_dev_point_is_not_the_written_one():
    line = render_holdout(_point(0.85, 0.5, 0.7), _point(0.9, 0.5, 0.7))
    assert line.startswith("holdout: accept_at=0.85 property_floor=0.50 choose_at=0.70 ")
    assert line.endswith(
        "(holdout of the dev-half recommendation, which differs from the full-data recommendation)"
    )


def test_holdout_line_without_a_point():
    assert render_holdout(None, _point(0.9, 0.5, 0.7)).startswith("holdout: no holdout point")

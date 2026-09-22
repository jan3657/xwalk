import pytest

from xwalk.decide.policy import (
    DecisionPolicy,
    Signals,
    derive_status,
    effective_rubric_floor,
    render_explanation,
)
from xwalk.records import DecisionReason, MatchStatus

POLICY = DecisionPolicy()


def _signals(**overrides):
    base = dict(
        candidate_count=10,
        retrieval_failed=False,
        screen_best=0.95,
        screen_chosen=0.95,
        resolution="exact_key",
        p_choice=0.8,
        p_none=0.02,
        choice_confidence=0.7,
        rubric=2.7,
        rubric_levels=4,
        rubric_confidence=0.8,
        properties={"processing_state": 0.9, "species": 0.95},
    )
    base.update(overrides)
    return Signals(**base)


def test_policy_validates_ranges():
    with pytest.raises(ValueError):
        DecisionPolicy(accept_at=1.5)
    with pytest.raises(ValueError):
        DecisionPolicy(shortlist_floor=0.9, accept_at=0.8)
    with pytest.raises(ValueError):
        DecisionPolicy(chunk_size=0)


def test_rubric_floor_defaults_to_the_second_best_level():
    assert effective_rubric_floor(POLICY, 4) == 2
    assert effective_rubric_floor(POLICY, 2) == 0
    assert effective_rubric_floor(DecisionPolicy(rubric_floor=1.5), 4) == 1.5


@pytest.mark.parametrize(
    "overrides, status, reason",
    [
        (
            {"retrieval_failed": True, "candidate_count": 0},
            MatchStatus.FAILED,
            DecisionReason.RETRIEVER_FAILURE,
        ),
        (
            {"candidate_count": 0, "screen_best": None},
            MatchStatus.UNMATCHED,
            DecisionReason.NO_CANDIDATES,
        ),
        (
            {"screen_best": 0.1, "screen_chosen": None, "resolution": None},
            MatchStatus.UNMATCHED,
            DecisionReason.BELOW_REVIEW_FLOOR,
        ),
        (
            {"resolution": "abstain", "screen_chosen": None, "p_none": 0.9, "p_choice": 0.9},
            MatchStatus.UNMATCHED,
            DecisionReason.SELECTOR_ABSTAINED,
        ),
        (
            {"resolution": "abstain", "screen_chosen": None, "p_none": 0.5, "p_choice": 0.5},
            MatchStatus.NEEDS_REVIEW,
            DecisionReason.UNRESOLVED_OUTPUT,
        ),
        (
            {"resolution": "unresolved", "screen_chosen": None},
            MatchStatus.NEEDS_REVIEW,
            DecisionReason.UNRESOLVED_OUTPUT,
        ),
        ({}, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD),
        ({"screen_chosen": 0.7}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        ({"p_choice": 0.3}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        ({"rubric": 1.2}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        (
            {"properties": {"species": 0.2}},
            MatchStatus.NEEDS_REVIEW,
            DecisionReason.BELOW_ACCEPT_THRESHOLD,
        ),
        ({"properties": {}}, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD),
        (
            {"rubric": None, "rubric_levels": None},
            MatchStatus.NEEDS_REVIEW,
            DecisionReason.BELOW_ACCEPT_THRESHOLD,
        ),
    ],
)
def test_derive_status_table(overrides, status, reason):
    assert derive_status(_signals(**overrides), POLICY) == (status, reason)


def test_flat_round_trips():
    s = _signals()
    back = Signals.from_flat(
        s.flat(), candidate_count=s.candidate_count, retrieval_failed=False, resolution=s.resolution
    )
    assert back == s
    assert "prop_species" in s.flat()
    assert "rubric_levels" in s.flat()


def test_flat_omits_unset_fields():
    assert "p_choice" not in _signals(p_choice=None).flat()


def test_render_explanation_is_deterministic_and_readable():
    text = render_explanation(_signals())
    assert text == render_explanation(_signals())
    assert "same 0.95" in text
    assert "p_choice 0.80" in text
    assert "rubric 2.7/3" in text
    assert "processing_state 0.90" in text
    assert render_explanation(_signals(screen_best=None, candidate_count=0)) == "no candidates"


def test_from_flat_ignores_keys_it_does_not_own():
    flat = dict(_signals().flat())
    flat["screen_C001"] = 0.42
    back = Signals.from_flat(
        flat, candidate_count=10, retrieval_failed=False, resolution="exact_key"
    )
    assert back == _signals()

import pytest

from xwalk.decide.base import (
    Choice,
    ChoiceAnswer,
    DecisionFatalError,
    DecisionResponseError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    parse_response,
    question_to_dict,
    response_from_dict,
    response_to_dict,
)
from xwalk.records import Usage

QUESTIONS = {
    "same": Noul(instructions="Is it the same?"),
    "best": Choice(instructions="Which?", criteria={"A": "apple", "B": None, "NONE": "none"}),
    "grade": Score(instructions="How good?", criteria=["bad", "ok", "good"]),
}

PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "same": {"type": "noul", "noul": 0.9},
        "best": {
            "type": "choice",
            "choice": "A",
            "probabilities": {"A": 0.8, "B": 0.15, "NONE": 0.05},
            "confidence": 0.7,
        },
        "grade": {
            "type": "score",
            "score": 1.6,
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            "confidence": 0.6,
            "legend": {"0": "bad", "1": "ok", "2": "good"},
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 9, "cost": 0.000005},
}


def test_questions_serialise_to_the_wire_shape():
    assert question_to_dict(QUESTIONS["same"]) == {
        "type": "noul",
        "instructions": "Is it the same?",
    }
    assert question_to_dict(QUESTIONS["best"]) == {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"A": "apple", "B": None, "NONE": "none"},
    }
    assert question_to_dict(QUESTIONS["grade"]) == {
        "type": "score",
        "instructions": "How good?",
        "criteria": ["bad", "ok", "good"],
    }


def test_parse_response_types_every_answer():
    response = parse_response(PAYLOAD, QUESTIONS, configured_model="jev-latest")
    assert response.answers["same"] == NoulAnswer(noul=0.9)
    assert isinstance(response.answers["best"], ChoiceAnswer)
    assert response.answers["best"].choice == "A"
    assert isinstance(response.answers["grade"], ScoreAnswer)
    assert response.answers["grade"].legend["2"] == "good"
    assert response.model == "typesafe/jev-1.13-20260917"
    assert response.usage.prompt_tokens == 120
    assert response.usage.completion_tokens == 0
    assert response.usage.calls == 1
    assert response.usage.cost_usd == pytest.approx(0.000005)


def test_parse_response_falls_back_to_the_configured_model():
    payload = {**PAYLOAD, "model": None}
    assert parse_response(payload, QUESTIONS, configured_model="jev-latest").model == "jev-latest"


def test_a_missing_answer_is_fatal():
    payload = {**PAYLOAD, "answers": {k: v for k, v in PAYLOAD["answers"].items() if k != "grade"}}
    with pytest.raises(DecisionFatalError, match="grade"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_a_choice_outside_the_criteria_is_fatal():
    bad = {**PAYLOAD["answers"]["best"], "choice": "Z"}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "best": bad}}
    with pytest.raises(DecisionFatalError, match="Z"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_a_probability_outside_the_unit_interval_is_fatal():
    bad = {"type": "noul", "noul": 1.7}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "same": bad}}
    with pytest.raises(DecisionFatalError, match="same"):
        parse_response(payload, QUESTIONS, configured_model="m")


@pytest.mark.parametrize(
    "score",
    [-0.1, 2.01, 3, 99.0, float("inf"), float("-inf"), float("nan"), True, "2"]
    + [pytest.param(10**400, id="huge-int")],
)
def test_a_rubric_score_outside_the_offered_levels_is_rejected_not_clamped(score):
    # Three criteria offer levels 0, 1 and 2. Anything else -- including a number Python
    # cannot even represent as a float -- is a broken response, never data.
    bad = {**PAYLOAD["answers"]["grade"], "score": score}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "grade": bad}}
    with pytest.raises(DecisionResponseError, match="grade"):
        parse_response(payload, QUESTIONS, configured_model="m")


@pytest.mark.parametrize("score", [0, 0.0, 1.6, 2, 2.0])
def test_a_rubric_score_on_or_inside_the_bounds_is_kept(score):
    good = {**PAYLOAD["answers"]["grade"], "score": score}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "grade": good}}
    answer = parse_response(payload, QUESTIONS, configured_model="m").answers["grade"]
    assert isinstance(answer, ScoreAnswer) and answer.score == float(score)


@pytest.mark.parametrize(
    "value", [pytest.param(10**400, id="huge-int"), float("nan"), float("inf")]
)
def test_a_probability_that_is_not_a_finite_unit_number_is_rejected(value):
    bad = {"type": "noul", "noul": value}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "same": bad}}
    with pytest.raises(DecisionResponseError, match="same"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_reported_usage_is_read():
    usage = parse_response(PAYLOAD, QUESTIONS, configured_model="m").usage
    assert (usage.prompt_tokens, usage.calls, usage.unknown_calls) == (120, 1, 0)
    assert usage.cost_usd == 0.000005


@pytest.mark.parametrize(
    "usage",
    [
        None,
        "lots",
        {},
        {"cost": 0.1},
        {"input_tokens": None},
        {"input_tokens": "120"},
        {"input_tokens": -3},
        {"input_tokens": True},
        {"input_tokens": float("nan")},
        {"input_tokens": float("inf")},
        {"input_tokens": 1.5},
    ],
)
def test_missing_or_invalid_token_usage_is_unknown_not_zero(usage):
    payload = {k: v for k, v in PAYLOAD.items() if k != "usage"}
    if usage is not None:
        payload["usage"] = usage
    parsed = parse_response(payload, QUESTIONS, configured_model="m").usage
    assert (parsed.calls, parsed.unknown_calls, parsed.total_tokens) == (1, 1, 0)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.0, "0.1"])
def test_an_invalid_cost_makes_the_whole_usage_block_unknown(cost):
    # A malformed accounting block is not trusted in part: the call is one with unknown
    # usage, and no NaN or negative cost can poison a run's totals.
    payload = {**PAYLOAD, "usage": {"input_tokens": 120, "cost": cost}}
    parsed = parse_response(payload, QUESTIONS, configured_model="m").usage
    assert parsed == Usage.unreported(1)


def test_a_type_mismatch_is_fatal():
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "same": PAYLOAD["answers"]["best"]}}
    with pytest.raises(DecisionFatalError, match="same"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_response_round_trips_through_dicts():
    response = parse_response(PAYLOAD, QUESTIONS, configured_model="m")
    assert response_from_dict(response_to_dict(response), QUESTIONS) == response

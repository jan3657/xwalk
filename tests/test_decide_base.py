import pytest

from xwalk.decide.base import (
    Choice,
    ChoiceAnswer,
    DecisionFatalError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    parse_response,
    question_to_dict,
    response_from_dict,
    response_to_dict,
)

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


def test_a_type_mismatch_is_fatal():
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "same": PAYLOAD["answers"]["best"]}}
    with pytest.raises(DecisionFatalError, match="same"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_response_round_trips_through_dicts():
    response = parse_response(PAYLOAD, QUESTIONS, configured_model="m")
    assert response_from_dict(response_to_dict(response), QUESTIONS) == response

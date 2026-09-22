import pytest
from pydantic import ValidationError

from xwalk.decide.base import Choice, Noul, Score
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.prompts.contract import PromptSlots

SLOTS = PromptSlots(
    entity_noun="food record",
    target_noun="FoodOn term",
    domain_brief="food composition data",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "same food, every property compatible"},
        {"score": 0.85, "name": "High", "when": "same product, incidental detail differs"},
        {"score": 0.55, "name": "Plausible", "when": "right food, processing differs"},
        {"score": 0.2, "name": "Speculative", "when": "only a broad category is shared"},
    ],
    hard_rules=["A dish is not one of its ingredients.", "Processing state is identity."],
    disambiguation_steps="Identify the core food first.",
    properties=[
        {"name": "processing_state", "question": "Do they agree on processing state?"},
        {"name": "species", "question": "Do they agree on species?"},
    ],
)


def test_screen_question_names_the_candidate_and_carries_every_hard_rule():
    q = QuestionSet.from_slots(SLOTS).screen_question("C007")
    assert isinstance(q, Noul)
    text = str(q.instructions)
    assert "`candidates.C007`" in text
    assert "`source`" in text
    for rule in SLOTS.hard_rules:
        assert rule in text
    assert q.criteria is not None and set(q.criteria) == {"true", "false"}


def test_choose_question_offers_every_key_plus_none():
    q = QuestionSet.from_slots(SLOTS).choose_question({"C001": "apple", "C002": "pear"})
    assert isinstance(q, Choice)
    assert list(q.criteria) == ["C001", "C002", NONE_KEY]
    assert "most specific" in str(q.instructions)
    assert SLOTS.disambiguation_steps in str(q.instructions)


def test_rubric_question_is_ascending_and_names_each_level():
    q = QuestionSet.from_slots(SLOTS).rubric_question()
    assert isinstance(q, Score)
    levels = [str(level) for level in q.criteria]
    assert levels[0].startswith("Speculative")
    assert levels[-1].startswith("Certain")
    assert QuestionSet.from_slots(SLOTS).rubric_levels == 4


def test_property_questions_are_prefixed_and_reference_both_sides():
    qs = QuestionSet.from_slots(SLOTS).property_questions()
    assert list(qs) == ["prop_processing_state", "prop_species"]
    assert "`candidate`" in str(qs["prop_species"].instructions)
    assert "`source`" in str(qs["prop_species"].instructions)


def test_a_rubric_with_more_than_ten_rows_is_rejected():
    rows = [{"score": 1 - i * 0.05, "name": f"L{i}", "when": "x"} for i in range(11)]
    slots = SLOTS.model_copy(
        update={"rubric": [type(SLOTS.rubric[0]).model_validate(r) for r in rows]}
    )
    with pytest.raises(ValueError, match="10"):
        QuestionSet.from_slots(slots).rubric_question()


def test_property_names_are_identifiers():
    with pytest.raises(ValidationError):
        PromptSlots.model_validate(
            {**SLOTS.model_dump(), "properties": [{"name": "Bad Name", "question": "?"}]}
        )


def test_fingerprint_changes_with_slots():
    a = QuestionSet.from_slots(SLOTS)
    b = QuestionSet.from_slots(SLOTS.model_copy(update={"hard_rules": ["other"]}))
    assert a.fingerprint != b.fingerprint

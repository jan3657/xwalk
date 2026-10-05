from pathlib import Path

import pytest
from pydantic import ValidationError

from xwalk.decide.base import Choice, Noul, Score
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.fingerprint import hash_value
from xwalk.prompts.contract import PromptSlots, load_slots

REF_ZIVILA_SLOTS = Path(__file__).parent.parent / "examples/ref_zivila/jobs/foodon/slots.yaml"

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


def test_screen_question_refers_to_the_shared_rules_instead_of_repeating_them():
    q = QuestionSet.from_slots(SLOTS).screen_question("C007")
    assert isinstance(q, Noul)
    text = str(q.instructions)
    for reference in (
        "`candidates.C007`",
        "`source`",
        "`rules.entity`",
        "`rules.hard_rules`",
        "`rules.same`",
        "`rules.different`",
    ):
        assert reference in text
    assert SLOTS.domain_brief not in text
    for rule in SLOTS.hard_rules:
        assert rule not in text
    assert q.criteria is None


def test_rules_state_carries_the_nouns_the_brief_and_every_hard_rule_verbatim():
    rules = QuestionSet.from_slots(SLOTS).rules_state()
    assert set(rules) == {"entity", "target", "domain", "hard_rules", "same", "different"}
    assert rules["entity"] == SLOTS.entity_noun
    assert rules["target"] == SLOTS.target_noun
    assert rules["domain"] == SLOTS.domain_brief
    assert rules["hard_rules"] == SLOTS.hard_rules
    assert "same entity" in rules["same"]
    assert "different entity" in rules["different"]


def test_rules_state_keeps_the_real_slots_hard_rules_verbatim():
    slots = load_slots(REF_ZIVILA_SLOTS)
    rules = QuestionSet.from_slots(slots).rules_state()
    assert rules["hard_rules"] == list(slots.hard_rules)
    assert len(rules["hard_rules"]) == 5


def test_fingerprint_moves_with_the_questions_version():
    """Version 2 hoisted the screen preamble; a cached version-1 answer must not be reused."""
    questions = QuestionSet.from_slots(SLOTS)
    version_one = hash_value({"questions_version": 1, "slots": SLOTS.model_dump(mode="json")})
    version_two = hash_value({"questions_version": 2, "slots": SLOTS.model_dump(mode="json")})
    assert questions.fingerprint != version_one
    assert questions.fingerprint == version_two


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

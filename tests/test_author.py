import json

import pytest
import yaml

from xwalk.llm.fake import FakeLLM
from xwalk.prompts.author import draft_slots, slots_diff, write_slots
from xwalk.prompts.contract import PromptSlots, load_slots
from xwalk.records import Record

SOURCES = [
    Record(id="s1", fields={"mention": "aspirin"}),
    Record(id="s2", fields={"mention": "paracetamol"}),
]
TARGETS = [
    Record(id="T1", fields={"label": "acetylsalicylic acid", "synonyms": ["aspirin"]}),
    Record(id="T2", fields={"label": "paracetamol", "synonyms": ["acetaminophen"]}),
]

GOOD = {
    "entity_noun": "drug mention",
    "target_noun": "chemical compound record",
    "domain_brief": "pharmaceutical nomenclature",
    "rubric": [
        {
            "score": 1.0,
            "name": "Certain",
            "when": "exact label or synonym match",
            "example": "aspirin -> acetylsalicylic acid",
        },
        {
            "score": 0.6,
            "name": "Plausible",
            "when": "same active ingredient, different salt",
            "example": "...",
        },
        {"score": 0.3, "name": "Speculative", "when": "same drug class only", "example": "..."},
    ],
    "hard_rules": ["A salt or ester is a distinct entity from its parent compound"],
    "disambiguation_steps": "",
}


async def test_returns_validated_slots():
    llm = FakeLLM([json.dumps(GOOD)])
    draft = await draft_slots(
        llm,
        description="matching drugs to compounds",
        source_samples=SOURCES,
        target_samples=TARGETS,
    )
    assert isinstance(draft.slots, PromptSlots)
    assert draft.slots.entity_noun == "drug mention"


async def test_the_description_reaches_the_model():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(
        llm,
        description="matching legal citations to court decisions",
        source_samples=SOURCES,
        target_samples=TARGETS,
    )
    assert "legal citations" in llm.requests[0].user


async def test_sample_records_from_both_sides_reach_the_model():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)
    prompt = llm.requests[0].user
    assert "aspirin" in prompt and "acetylsalicylic acid" in prompt


async def test_samples_are_capped():
    llm = FakeLLM([json.dumps(GOOD)])
    many = [Record(id=f"s{i}", fields={"mention": f"drug{i}"}) for i in range(50)]
    await draft_slots(
        llm, description="d", source_samples=many, target_samples=TARGETS, max_samples=3
    )
    prompt = llm.requests[0].user
    assert "drug0" in prompt and "drug40" not in prompt


def test_the_schema_offers_the_decider_paths_properties():
    """Without it the drafting model cannot return `properties` at all."""
    from xwalk.prompts.author import SLOTS_SCHEMA

    properties = SLOTS_SCHEMA["properties"]["properties"]
    assert properties["type"] == "array"
    assert set(properties["items"]["required"]) == {"name", "question"}
    assert properties["items"]["properties"]["name"]["pattern"] == "^[a-z][a-z0-9_]*$"
    assert "properties" not in SLOTS_SCHEMA["required"]


async def test_the_drafting_model_is_told_about_properties():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(
        llm,
        description="matching drugs to compounds",
        source_samples=SOURCES,
        target_samples=TARGETS,
    )
    assert "properties" in llm.requests[0].user


async def test_drafted_properties_are_kept():
    payload = {**GOOD, "properties": [{"name": "salt_form", "question": "Same salt form?"}]}
    draft = await draft_slots(
        FakeLLM([json.dumps(payload)]),
        description="matching drugs to compounds",
        source_samples=SOURCES,
        target_samples=TARGETS,
    )
    assert [p.name for p in draft.slots.properties] == ["salt_form"]


async def test_the_model_is_never_asked_for_prompt_text():
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)
    prompt = llm.requests[0].user.lower()
    assert "chosen_key" not in prompt  # the output contract is not the model's business


async def test_an_out_of_order_rubric_is_repaired_and_warned_about():
    payload = {**GOOD, "rubric": list(reversed(GOOD["rubric"]))}
    llm = FakeLLM([json.dumps(payload)])
    draft = await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)
    assert [r.score for r in draft.slots.rubric] == [1.0, 0.6, 0.3]
    assert any("rubric" in w for w in draft.warnings)


async def test_an_out_of_range_score_is_clamped_and_warned_about():
    payload = {**GOOD, "rubric": [{**GOOD["rubric"][0], "score": 1.4}, *GOOD["rubric"][1:]]}
    llm = FakeLLM([json.dumps(payload)])
    draft = await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)
    assert draft.slots.rubric[0].score == 1.0
    assert any("clamp" in w.lower() for w in draft.warnings)


async def test_a_missing_rubric_score_is_a_clear_error_not_a_type_error():
    """A model that omits `score` must not crash the sort with a None comparison."""
    payload = {**GOOD, "rubric": [{"name": "C", "when": "x"}, *GOOD["rubric"][1:]]}
    llm = FakeLLM([json.dumps(payload)])
    with pytest.raises(ValueError, match="score"):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_a_single_row_rubric_is_rejected_with_a_clear_error():
    payload = {**GOOD, "rubric": GOOD["rubric"][:1]}
    llm = FakeLLM([json.dumps(payload)])
    with pytest.raises(ValueError, match="rubric"):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_malformed_output_is_a_clear_error_not_a_broken_prompt():
    llm = FakeLLM(["I'm not sure what you want."])
    with pytest.raises(ValueError, match="could not"):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_a_draft_that_fails_contract_validation_is_rejected():
    """Belt and braces: even valid slots must render a contract-satisfying prompt."""
    payload = {**GOOD, "entity_noun": "   "}
    llm = FakeLLM([json.dumps(payload)])
    with pytest.raises(ValueError):
        await draft_slots(llm, description="d", source_samples=SOURCES, target_samples=TARGETS)


async def test_existing_slots_are_shown_when_refining():
    existing = PromptSlots(**GOOD)
    llm = FakeLLM([json.dumps(GOOD)])
    await draft_slots(
        llm,
        description="d",
        source_samples=SOURCES,
        target_samples=TARGETS,
        existing=existing,
    )
    assert "pharmaceutical nomenclature" in llm.requests[0].user


def test_slots_diff_shows_changed_fields():
    before = PromptSlots(**GOOD)
    after = before.model_copy(update={"domain_brief": "veterinary pharmacology"})
    diff = slots_diff(before, after)
    assert "domain_brief" in diff and "veterinary pharmacology" in diff


def test_slots_diff_is_empty_for_identical_slots():
    before = PromptSlots(**GOOD)
    assert slots_diff(before, before) == ""


def test_write_slots_produces_a_loadable_file(tmp_path):
    path = tmp_path / "slots.yaml"
    write_slots(PromptSlots(**GOOD), path)
    assert load_slots(path).entity_noun == "drug mention"


def test_write_slots_produces_readable_yaml(tmp_path):
    path = tmp_path / "slots.yaml"
    write_slots(PromptSlots(**GOOD), path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data["rubric"], list)
    assert "!!python" not in path.read_text(encoding="utf-8")

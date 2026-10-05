import json

import pytest
import yaml

from xwalk.prompts.contract import (
    BASE_DIR,
    ContractError,
    PromptSet,
    PromptSlots,
    load_slots,
    validate_contract,
)


def skeleton_dir(tmp_path, **overrides: str):
    """Copy the shipped skeletons into a temp dir, replacing the named ones."""
    base = tmp_path / "base"
    base.mkdir()
    for name in ("select", "score", "verify", "rewrite"):
        text = overrides.get(name, (BASE_DIR / f"{name}.j2").read_text(encoding="utf-8"))
        (base / f"{name}.j2").write_text(text, encoding="utf-8")
    return base


SLOTS = PromptSlots(
    entity_noun="chemical entity mention",
    target_noun="ChEBI ontology term",
    domain_brief="biomedical chemistry nomenclature",
    rubric=[
        {
            "score": 1.0,
            "name": "Certain",
            "when": "exact match to label or synonym",
            "example": "garlic -> Garlic",
        },
        {
            "score": 0.9,
            "name": "High",
            "when": "normalized form or common abbreviation",
            "example": "ASA -> aspirin",
        },
        {
            "score": 0.6,
            "name": "Plausible",
            "when": "a specific instance of the candidate",
            "example": "Fuji apple -> apple",
        },
        {
            "score": 0.4,
            "name": "Speculative",
            "when": "related by broad category only",
            "example": "sugar -> carbohydrate",
        },
    ],
    hard_rules=["Any difference in a regulated substance's name or number means a distinct entity"],
)


def test_slots_reject_an_empty_rubric():
    with pytest.raises(ValueError, match="rubric"):
        PromptSlots(entity_noun="a", target_noun="b", domain_brief="c", rubric=[])


def test_slots_reject_a_score_outside_zero_to_one():
    with pytest.raises(ValueError, match="between 0 and 1"):
        PromptSlots(
            entity_noun="a",
            target_noun="b",
            domain_brief="c",
            rubric=[
                {"score": 1.5, "name": "x", "when": "y"},
                {"score": 0.5, "name": "z", "when": "w"},
            ],
        )


def test_slots_require_strictly_decreasing_scores():
    with pytest.raises(ValueError, match="decreasing"):
        PromptSlots(
            entity_noun="a",
            target_noun="b",
            domain_brief="c",
            rubric=[
                {"score": 0.5, "name": "x", "when": "y"},
                {"score": 0.9, "name": "z", "when": "w"},
            ],
        )


def test_slots_require_at_least_two_rubric_rows():
    with pytest.raises(ValueError, match="at least 2"):
        PromptSlots(
            entity_noun="a",
            target_noun="b",
            domain_brief="c",
            rubric=[{"score": 0.9, "name": "x", "when": "y"}],
        )


def test_load_slots_reads_yaml(tmp_path):
    path = tmp_path / "slots.yaml"
    path.write_text(yaml.safe_dump(json.loads(SLOTS.model_dump_json())), encoding="utf-8")
    assert load_slots(path).entity_noun == "chemical entity mention"


def test_load_slots_refuses_a_duplicate_key(tmp_path):
    path = tmp_path / "slots.yaml"
    text = yaml.safe_dump(json.loads(SLOTS.model_dump_json()), sort_keys=False)
    path.write_text(text + "entity_noun: SECRETVALUE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key 'entity_noun'") as info:
        load_slots(path)
    assert "SECRETVALUE" not in str(info.value)  # the position, not the content


def test_select_prompt_contains_the_candidate_block():
    prompts = PromptSet.from_slots(SLOTS)
    rendered = prompts.render_select(
        source_fields={"mention": "glucose"},
        context="blood [glucose] levels",
        candidate_block="[C01] ID: CHEBI:17234 Label: glucose",
    )
    assert "[C01] ID: CHEBI:17234" in rendered


def test_select_prompt_forbids_answering_with_an_identifier():
    rendered = PromptSet.from_slots(SLOTS).render_select(
        source_fields={"mention": "x"}, context="", candidate_block="[C01] ID: a"
    )
    assert "Never answer with an identifier" in rendered


def test_select_prompt_omits_the_context_heading_when_there_is_no_context():
    rendered = PromptSet.from_slots(SLOTS).render_select(
        source_fields={"mention": "x"}, context="", candidate_block="[C01] ID: a"
    )
    assert "## Context" not in rendered


def test_score_prompt_shows_the_whole_source_record_not_a_query():
    rendered = PromptSet.from_slots(SLOTS).render_score(
        source_fields={"symbol": "TP53", "organism": "Homo sapiens"},
        context="",
        chosen_block="[C01] ID: NCBIGene:7157",
        other_candidates="[C02] ID: NCBIGene:22059",
        review_floor=0.4,
    )
    assert "symbol: TP53" in rendered and "organism: Homo sapiens" in rendered


def test_score_prompt_renders_every_rubric_row():
    rendered = PromptSet.from_slots(SLOTS).render_score(
        source_fields={"a": "b"},
        context="",
        chosen_block="x",
        other_candidates="",
        review_floor=0.4,
    )
    for row in SLOTS.rubric:
        assert row.name in rendered


def test_verify_prompt_offers_all_three_decisions():
    rendered = PromptSet.from_slots(SLOTS).render_verify(
        source_fields={"a": "b"}, context="", chosen_block="x", other_candidates=""
    )
    for decision in ("support", "disagree", "no_match"):
        assert decision in rendered


def test_rewrite_prompt_lists_previous_queries():
    rendered = PromptSet.from_slots(SLOTS).render_rewrite(
        source_fields={"a": "b"},
        context="",
        previous_queries=["glucose", "dextrose"],
        best_candidates="",
        max_queries=2,
    )
    assert "- glucose" in rendered and "- dextrose" in rendered


def test_hard_rules_appear_in_select_score_and_verify():
    prompts = PromptSet.from_slots(SLOTS)
    rule = SLOTS.hard_rules[0]
    assert rule in prompts.render_select(source_fields={}, context="", candidate_block="x")
    assert rule in prompts.render_score(
        source_fields={}, context="", chosen_block="x", other_candidates="", review_floor=0.4
    )
    assert rule in prompts.render_verify(
        source_fields={}, context="", chosen_block="x", other_candidates=""
    )


def test_the_declared_output_examples_parse_as_their_schemas():
    validate_contract(PromptSet.from_slots(SLOTS))  # raises on failure


def test_contract_validation_rejects_a_skeleton_missing_the_candidate_block(tmp_path):
    base = skeleton_dir(tmp_path, select="no blocks here\n")
    with pytest.raises(ContractError, match="candidate_block"):
        validate_contract(PromptSet.from_slots(SLOTS, base_dir=base))


def test_contract_validation_rejects_a_duplicated_input_block(tmp_path):
    """The spec requires every input block to appear *exactly once*. Rendering the
    candidate list twice would show the model two lists and is not merely untidy."""
    original = (BASE_DIR / "select.j2").read_text(encoding="utf-8")
    base = skeleton_dir(tmp_path, select=original + "\n## Candidates\n{{ candidate_block }}\n")
    with pytest.raises(ContractError, match="exactly once"):
        validate_contract(PromptSet.from_slots(SLOTS, base_dir=base))


def test_contract_validation_rejects_a_skeleton_that_lost_the_key_instruction(tmp_path):
    """Without 'Never answer with an identifier' the model may answer with a target ID,
    which resolves to UNRESOLVED_OUTPUT every time. An optimizer must not be able to
    delete it while leaving a prompt that still looks valid."""
    original = (BASE_DIR / "select.j2").read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in original.splitlines() if "Never answer with an identifier" not in line
    )
    base = skeleton_dir(tmp_path, select=stripped + "\n")
    with pytest.raises(ContractError, match="key instruction"):
        validate_contract(PromptSet.from_slots(SLOTS, base_dir=base))


def test_fingerprint_changes_when_a_slot_changes():
    a = PromptSet.from_slots(SLOTS)
    b = PromptSet.from_slots(SLOTS.model_copy(update={"domain_brief": "legal citations"}))
    assert a.fingerprint != b.fingerprint


def test_fingerprint_covers_the_skeleton_text_too(tmp_path):
    from xwalk.prompts.contract import BASE_DIR

    base = tmp_path / "base"
    base.mkdir()
    for name in ("select.j2", "score.j2", "verify.j2", "rewrite.j2"):
        (base / name).write_text((BASE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    (base / "select.j2").write_text(
        (base / "select.j2").read_text(encoding="utf-8") + "\nextra line\n", encoding="utf-8"
    )
    assert PromptSet.from_slots(SLOTS).fingerprint != (
        PromptSet.from_slots(SLOTS, base_dir=base).fingerprint
    )


def test_the_shipped_chemistry_example_validates():
    from pathlib import Path

    # Anchored to this file, not to the working directory the suite happens to run in.
    path = Path(__file__).resolve().parent.parent / "examples" / "chemistry" / "slots.yaml"
    validate_contract(PromptSet.from_slots(load_slots(path)))


def test_slots_without_properties_still_load():
    from xwalk.prompts.contract import PromptSlots

    slots = PromptSlots(
        entity_noun="a",
        target_noun="b",
        domain_brief="c",
        rubric=[{"score": 1.0, "name": "x", "when": "y"}, {"score": 0.5, "name": "z", "when": "w"}],
    )
    assert slots.properties == []

"""The NER step of the demo site (docs/ner-site/PLAN.md section 4): offline, FakeLLM only."""

from __future__ import annotations

import json
from typing import Any

import pytest
from ner_site.extract import (
    DEFAULT_PROMPT,
    ENTITY_SCHEMA,
    SYSTEM_PROMPT,
    ExtractionError,
    build_request,
    extract,
    locate,
    parse_entities,
)
from ner_site.schema import ENTITY_TYPES, Entity, Limits, ServiceError

from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import strict_schema
from xwalk.records import Usage

TEXT = "Aspirin reduced fever in rheumatoid arthritis."


def answer(*items: dict[str, Any]) -> str:
    return json.dumps({"entities": list(items)})


def item(text: str, type_: str, start: Any = None, end: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"text": text, "type": type_}
    if start is not None:
        out["start"] = start
    if end is not None:
        out["end"] = end
    return out


def spans(entities: list[Entity], text: str) -> list[tuple[str, int, int, str]]:
    # The invariant every entity must keep: its text is exactly the visitor's substring.
    for e in entities:
        assert text[e.start : e.end] == e.text
    return [(e.text, e.start, e.end, e.type) for e in entities]


# --- prompts, schema, request -------------------------------------------------------


def test_system_prompt_fixes_the_format_and_cannot_be_overridden() -> None:
    lowered = SYSTEM_PROMPT.lower()
    for type_ in ENTITY_TYPES:
        assert f'"{type_}"' in SYSTEM_PROMPT
    assert "json" in lowered
    assert "0-based" in lowered
    assert "instructions" in lowered and "text" in lowered
    assert "cannot change" in lowered


def test_default_prompt_covers_the_types_and_the_rules() -> None:
    for type_ in ENTITY_TYPES:
        assert f"{type_}:" in DEFAULT_PROMPT
    assert "exactly as written" in DEFAULT_PROMPT
    assert "overlap" in DEFAULT_PROMPT
    assert f"at most {Limits().max_entities}" in DEFAULT_PROMPT


def test_entity_schema_requires_the_four_fields_and_the_enum() -> None:
    assert ENTITY_SCHEMA["required"] == ["entities"]
    items = ENTITY_SCHEMA["properties"]["entities"]["items"]
    assert sorted(items["required"]) == ["end", "start", "text", "type"]
    assert items["properties"]["type"]["enum"] == list(ENTITY_TYPES)
    assert items["properties"]["start"]["type"] == "integer"
    # OpenAI's strict structured output accepts it unchanged.
    assert strict_schema(ENTITY_SCHEMA) == dict(ENTITY_SCHEMA)


def test_build_request() -> None:
    request = build_request("Some text.", "Find chemicals.")
    assert request.system == SYSTEM_PROMPT
    assert request.user == "INSTRUCTIONS:\nFind chemicals.\n\nTEXT:\nSome text."
    assert request.schema == ENTITY_SCHEMA
    assert request.schema_name == "entities"
    assert request.temperature == 0.0
    assert request.max_tokens == 2048


# --- locate --------------------------------------------------------------------------


def test_locate_exact_takes_the_occurrence_nearest_the_hint() -> None:
    text = "aspirin, then aspirin, then aspirin"
    assert locate(text, "aspirin", None) == (0, 7)
    assert locate(text, "aspirin", 15) == (14, 21)
    assert locate(text, "aspirin", 100) == (28, 35)


def test_locate_skips_claimed_occurrences() -> None:
    text = "aspirin, then aspirin"
    assert locate(text, "aspirin", 0, claimed=[(0, 7)]) == (14, 21)
    assert locate(text, "aspirin", 0, claimed=[(0, 7), (14, 21)]) is None
    # Only an identical span is claimed: overlaps are settled later by parse_entities.
    assert locate(text, "aspirin", 0, claimed=[(3, 5)]) == (0, 7)


def test_locate_casefold_maps_back_to_the_original_span() -> None:
    assert locate("Take ASPIRIN daily", "aspirin", None) == (5, 12)


def test_locate_offsets_are_code_points_after_non_ascii_text() -> None:
    # 'ß' casefolds to 'ss': matching against text.casefold() would shift every later
    # offset by one. The emoji is one code point (two UTF-16 units, four UTF-8 bytes).
    text = "Die Straße 🙂: ASPIRIN wirkt"
    start, end = 14, 21
    assert text[start:end] == "ASPIRIN"
    assert locate(text, "aspirin", None) == (start, end)
    assert locate(text, "ASPIRIN", 0) == (start, end)


def test_locate_does_not_accept_a_casefold_match_that_changes_length() -> None:
    # 'STRASSE'.casefold() == 'straße'.casefold(), but no span of 'Straße' has seven
    # characters; such a match would have no correct span, so it is not a match.
    assert locate("Die Straße ist lang", "STRASSE", None) is None
    assert locate("DIE STRASSE IST LANG", "straße", None) is None


def test_locate_collapses_whitespace_on_both_sides() -> None:
    text = "patients with rheumatoid\n   arthritis improved"
    assert locate(text, "rheumatoid arthritis", None) == (14, 37)
    assert locate("rheumatoid arthritis", "rheumatoid \t arthritis", None) == (0, 20)
    assert locate("RHEUMATOID\nARTHRITIS", "rheumatoid arthritis", None) == (0, 20)


def test_locate_missing_or_blank_mention_is_none() -> None:
    assert locate(TEXT, "feverr", 16) is None
    assert locate(TEXT, "", 0) is None
    assert locate(TEXT, "   ", 0) is None


# --- parse_entities ------------------------------------------------------------------


def test_parse_keeps_correct_spans_sorts_and_numbers() -> None:
    raw = answer(
        item("rheumatoid arthritis", "disease", 25, 45),
        item("Aspirin", "chemical", 0, 7),
        item("fever", "other", 16, 21),
    )
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [
        ("Aspirin", 0, 7, "chemical"),
        ("fever", 16, 21, "other"),
        ("rheumatoid arthritis", 25, 45, "disease"),
    ]
    assert [e.id for e in entities] == ["e1", "e2", "e3"]
    assert warnings == []


def test_parse_accepts_fenced_json_with_prose() -> None:
    raw = "Here you go:\n```json\n" + answer(item("Aspirin", "chemical", 0, 7)) + "\n```\nDone."
    entities, _ = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("Aspirin", 0, 7, "chemical")]


def test_parse_relocates_wrong_or_missing_offsets() -> None:
    raw = answer(
        item("fever", "other", 13, 18),  # off by three
        item("rheumatoid arthritis", "disease"),  # no offsets
        item("Aspirin", "chemical", "0", True),  # not integers
    )
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [
        ("Aspirin", 0, 7, "chemical"),
        ("fever", 16, 21, "other"),
        ("rheumatoid arthritis", 25, 45, "disease"),
    ]
    assert warnings == []


def test_parse_relocates_repeated_mentions_onto_distinct_occurrences() -> None:
    text = "aspirin or aspirin"
    raw = answer(item("aspirin", "chemical", 0, 7), item("aspirin", "chemical", 0, 9))
    entities, _ = parse_entities(raw, text, max_entities=25)
    assert spans(entities, text) == [("aspirin", 0, 7, "chemical"), ("aspirin", 11, 18, "chemical")]


def test_parse_drops_a_mention_not_in_the_text_with_a_warning() -> None:
    raw = answer(item("Aspirin", "chemical", 0, 7), item("feverr", "other", 16, 22))
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("Aspirin", 0, 7, "chemical")]
    assert warnings == ["entity 'feverr' dropped: not found in text"]


def test_parse_unknown_type_becomes_other_with_a_warning() -> None:
    raw = answer(item("Aspirin", "drug", 0, 7), item("fever", " Disease ", 16, 21))
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("Aspirin", 0, 7, "other"), ("fever", 16, 21, "disease")]
    assert warnings == ["entity 'Aspirin': unknown type 'drug', using 'other'"]


def test_parse_overlaps_keep_the_earlier_longer_span() -> None:
    raw = answer(
        item("arthritis", "disease", 36, 45),
        item("rheumatoid arthritis", "disease", 25, 45),
        item("rheumatoid", "other", 25, 35),
        item("in rheumatoid", "other", 22, 35),
    )
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    # 'in rheumatoid' starts first, so it is kept and the two spans overlapping it go;
    # 'arthritis' no longer overlaps anything kept.
    assert spans(entities, TEXT) == [
        ("in rheumatoid", 22, 35, "other"),
        ("arthritis", 36, 45, "disease"),
    ]
    assert warnings == [
        "entity 'rheumatoid arthritis' dropped: overlaps 'in rheumatoid'",
        "entity 'rheumatoid' dropped: overlaps 'in rheumatoid'",
    ]

    raw = answer(item("arthritis", "disease", 36, 45), item("rheumatoid arthritis", "disease"))
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("rheumatoid arthritis", 25, 45, "disease")]
    assert warnings == ["entity 'arthritis' dropped: overlaps 'rheumatoid arthritis'"]


def test_parse_deduplicates_identical_spans_silently() -> None:
    raw = answer(item("Aspirin", "chemical", 0, 7), item("Aspirin", "chemical", 0, 7))
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("Aspirin", 0, 7, "chemical")]
    assert warnings == []


def test_parse_repeated_mention_with_one_occurrence_is_a_duplicate_not_missing() -> None:
    raw = answer(item("fever", "other"), item("fever", "other", 3, 8))
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("fever", 16, 21, "other")]
    assert warnings == []


def test_parse_truncates_to_max_entities_in_text_order() -> None:
    raw = answer(
        item("rheumatoid arthritis", "disease", 25, 45),
        item("fever", "other", 16, 21),
        item("Aspirin", "chemical", 0, 7),
    )
    entities, warnings = parse_entities(raw, TEXT, max_entities=2)
    assert spans(entities, TEXT) == [("Aspirin", 0, 7, "chemical"), ("fever", 16, 21, "other")]
    assert [e.id for e in entities] == ["e1", "e2"]
    assert warnings == ["3 entities found; only the first 2 are kept"]


def test_parse_validates_items() -> None:
    long_mention = "x" * 201
    raw = answer(
        "Aspirin",  # type: ignore[arg-type]
        {"type": "chemical"},
        item("   ", "chemical"),
        item(long_mention, "chemical"),
        item("  fever ", "other", 16, 21),  # stripped, then located
    )
    entities, warnings = parse_entities(raw, TEXT, max_entities=25)
    assert spans(entities, TEXT) == [("fever", 16, 21, "other")]
    assert warnings == [
        "entity #1 dropped: not an object",
        "entity #2 dropped: no mention text",
        "entity #3 dropped: no mention text",
        "entity #4 dropped: mention longer than 200 characters",
    ]


@pytest.mark.parametrize("raw", ["I cannot help with that.", "[1, 2]", '"entities"', ""])
def test_parse_non_object_is_unparseable(raw: str) -> None:
    with pytest.raises(ServiceError) as info:
        parse_entities(raw, TEXT, max_entities=25)
    assert info.value.code == "extraction_unparseable"
    assert info.value.status == 422


@pytest.mark.parametrize(
    "raw", ['{"mentions": []}', '{"entities": "Aspirin"}', '{"entities": null}']
)
def test_parse_object_without_an_entities_list_is_a_schema_error(raw: str) -> None:
    with pytest.raises(ServiceError) as info:
        parse_entities(raw, TEXT, max_entities=25)
    assert info.value.code == "extraction_schema"
    assert info.value.status == 422
    assert TEXT not in info.value.message


def test_parse_empty_entities_is_fine() -> None:
    assert parse_entities('{"entities": []}', TEXT, max_entities=25) == ([], [])


# --- extract -------------------------------------------------------------------------


async def test_extract_with_fake_llm() -> None:
    llm = FakeLLM([answer(item("Aspirin", "chemical", 0, 7), item("feverr", "other"))])
    entities, warnings, usage = await extract(llm, TEXT, DEFAULT_PROMPT, max_entities=25)
    assert spans(entities, TEXT) == [("Aspirin", 0, 7, "chemical")]
    assert warnings == ["entity 'feverr' dropped: not found in text"]
    assert usage == Usage(prompt_tokens=100, completion_tokens=20, calls=1)
    assert llm.requests == [build_request(TEXT, DEFAULT_PROMPT)]


async def test_extract_unparseable_answer_carries_the_call_usage() -> None:
    llm = FakeLLM(["Sorry, no JSON today."])
    with pytest.raises(ExtractionError) as info:
        await extract(llm, TEXT, DEFAULT_PROMPT, max_entities=25)
    assert isinstance(info.value, ServiceError)
    assert info.value.code == "extraction_unparseable"
    assert info.value.status == 422
    assert info.value.usage == Usage(prompt_tokens=100, completion_tokens=20, calls=1)


async def test_extract_does_not_catch_provider_errors() -> None:
    error = LLMRetryableError("upstream 503")
    llm = FakeLLM([error])
    with pytest.raises(LLMRetryableError) as info:
        await extract(llm, TEXT, DEFAULT_PROMPT, max_entities=25)
    assert info.value is error


async def test_extract_says_when_the_answer_was_cut_off() -> None:
    llm = FakeLLM(['{"entities": [{"text": "Asp'], finish_reason="length")
    with pytest.raises(ExtractionError) as info:
        await extract(llm, TEXT, DEFAULT_PROMPT, max_entities=25)
    assert info.value.code == "extraction_unparseable"
    assert info.value.message.endswith("(the answer was cut off at the token limit)")
    assert str(info.value) == info.value.message

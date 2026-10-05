import pytest

from xwalk.llm.base import ParseError
from xwalk.llm.parsing import (
    extract_json_object,
    parse_confidence,
    parse_json_object,
    repair_json,
    strip_thinking,
)


def test_strips_a_complete_think_block():
    assert strip_thinking("<think>hmm</think>ANSWER") == "ANSWER"


def test_strips_thinking_and_reasoning_tags():
    assert strip_thinking("<thinking>a</thinking>X<reasoning>b</reasoning>Y") == "XY"


def test_strips_a_multiline_think_block():
    assert strip_thinking('<think>\nline1\nline2\n</think>\n{"a": 1}') == '{"a": 1}'


def test_handles_a_dangling_closing_tag():
    """Some models emit reasoning with no opening tag. Everything before </think> is reasoning."""
    assert strip_thinking('I should pick C01.\n</think>\n{"chosen_key": "C01"}') == (
        '{"chosen_key": "C01"}'
    )


def test_handles_an_unclosed_opening_tag():
    """Truncated output: an opening tag with no close means nothing usable follows."""
    assert strip_thinking("prefix<think>reasoning that never ends") == "prefix"


def test_leaves_ordinary_text_alone():
    assert strip_thinking('{"a": 1}') == '{"a": 1}'


def test_is_case_insensitive_about_tags():
    assert strip_thinking("<THINK>x</THINK>Y") == "Y"


def test_extracts_json_from_a_fenced_block():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json_object(text) == '{"a": 1}'


def test_extracts_json_from_an_unlabelled_fence():
    assert extract_json_object('```\n{"a": 1}\n```') == '{"a": 1}'


def test_extracts_a_bare_object_surrounded_by_prose():
    assert extract_json_object('Sure. {"a": 1} Done.') == '{"a": 1}'


def test_extracts_a_balanced_object_containing_nested_braces():
    assert extract_json_object('x {"a": {"b": 2}} y') == '{"a": {"b": 2}}'


def test_ignores_braces_inside_strings():
    assert extract_json_object('{"a": "} not the end"}') == '{"a": "} not the end"}'


def test_returns_none_when_there_is_no_object():
    assert extract_json_object("no json here") is None


def test_repair_removes_a_trailing_comma():
    assert repair_json('{"a": 1,}') == '{"a": 1}'


def test_repair_removes_a_trailing_comma_in_a_list():
    assert repair_json('{"a": [1, 2,]}') == '{"a": [1, 2]}'


def test_parse_handles_a_clean_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_handles_thinking_plus_fence_plus_trailing_comma():
    raw = '<think>deciding</think>\n```json\n{"chosen_key": "C01", "confidence": 0.9,}\n```'
    assert parse_json_object(raw) == {"chosen_key": "C01", "confidence": 0.9}


def test_parse_raises_on_unrecoverable_output():
    with pytest.raises(ParseError):
        parse_json_object("I refuse to answer.")


def test_parse_raises_when_the_value_is_not_an_object():
    with pytest.raises(ParseError, match="object"):
        parse_json_object("[1, 2, 3]")


def test_parse_error_carries_the_raw_text_for_the_trace():
    with pytest.raises(ParseError) as exc:
        parse_json_object("garbage")
    assert exc.value.raw == "garbage"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (1, 1.0), (0.42, 0.42)],
)
def test_a_finite_number_in_range_is_a_confidence(value, expected):
    assert parse_confidence(value) == (expected, None)


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (None, "missing"),
        ("0.9", "must be a number"),
        (True, "must be a number"),
        ([0.9], "must be a number"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
        (10**400, "finite"),
        (1.0000001, "outside"),
        (-0.5, "outside"),
    ],
)
def test_anything_else_is_rejected_with_a_reason(value, why):
    score, error = parse_confidence(value)
    assert score is None
    assert why in (error or "")


def test_nan_survives_json_parsing_so_the_validator_must_catch_it():
    payload = parse_json_object('{"confidence_score": NaN}')
    assert parse_confidence(payload["confidence_score"])[0] is None

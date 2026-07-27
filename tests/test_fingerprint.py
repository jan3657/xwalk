from datetime import date

import pytest

from xwalk.fingerprint import canonical_json, hash_record, hash_value, result_key, run_fingerprint
from xwalk.records import Record


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_is_insertion_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_normalises_sets_deterministically():
    assert canonical_json({"tags": {"b", "a"}}) == '{"tags":["a","b"]}'


def test_canonical_json_renders_tuples_as_lists():
    assert canonical_json(("a", "b")) == '["a","b"]'


def test_canonical_json_handles_dates():
    assert canonical_json(date(2026, 7, 26)) == '"2026-07-26"'


def test_canonical_json_rejects_unserialisable_values():
    with pytest.raises(TypeError, match="not serialisable"):
        canonical_json({"f": object()})


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"score": float("nan")})


def test_hash_value_is_sixteen_hex_chars():
    digest = hash_value({"a": 1})
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_value_is_stable_across_calls():
    assert hash_value({"a": [1, 2]}) == hash_value({"a": [1, 2]})


def test_hash_record_changes_when_a_field_value_changes():
    before = hash_record(Record(id="s1", fields={"name": "glucose"}))
    after = hash_record(Record(id="s1", fields={"name": "fructose"}))
    assert before != after


def test_hash_record_ignores_field_insertion_order():
    a = hash_record(Record(id="s1", fields={"name": "x", "city": "y"}))
    b = hash_record(Record(id="s1", fields={"city": "y", "name": "x"}))
    assert a == b


def test_run_fingerprint_changes_when_any_component_changes():
    base = dict(model="gpt-4o", target="t1", policy={"accept_at": 0.6})
    assert run_fingerprint(**base) != run_fingerprint(**{**base, "model": "gpt-4o-mini"})
    assert run_fingerprint(**base) != run_fingerprint(**{**base, "policy": {"accept_at": 0.7}})


def test_run_fingerprint_ignores_keyword_order():
    assert run_fingerprint(a=1, b=2) == run_fingerprint(b=2, a=1)


def test_result_key_is_not_vulnerable_to_boundary_collisions():
    """('a', 'bc') and ('ab', 'c') must not collide, which naive concatenation allows."""
    assert result_key("fp", "a", "bc") != result_key("fp", "ab", "c")


def test_result_key_changes_with_run_fingerprint():
    assert result_key("fp1", "s1", "h1") != result_key("fp2", "s1", "h1")


def test_result_key_changes_with_source_hash():
    assert result_key("fp1", "s1", "h1") != result_key("fp1", "s1", "h2")

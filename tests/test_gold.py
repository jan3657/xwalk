import json

import pytest

from tests.conftest import FIXTURES
from xwalk.evaluate.gold import GoldSet, load_gold_csv, load_gold_jsonl

GOLD_TINY = FIXTURES / "gold_tiny.csv"


def test_loads_one_entry_per_row():
    assert len(load_gold_csv(GOLD_TINY)) == 4


def test_a_populated_cell_becomes_a_one_element_set():
    assert load_gold_csv(GOLD_TINY).get("s1") == frozenset({"CHEBI:17234"})


def test_an_empty_cell_means_the_answer_is_no_match():
    gold = load_gold_csv(GOLD_TINY)
    assert gold.get("s4") == frozenset()
    assert "s4" in gold


def test_an_absent_source_id_is_unlabelled_not_no_match():
    gold = load_gold_csv(GOLD_TINY)
    assert gold.get("s99") is None
    assert "s99" not in gold


def test_no_match_ids_lists_exactly_the_empty_labels():
    assert load_gold_csv(GOLD_TINY).no_match_ids == frozenset({"s4"})


def test_multi_gold_splits_on_the_separator(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A|B|C\n", encoding="utf-8")
    assert load_gold_csv(path).get("s1") == frozenset({"A", "B", "C"})


def test_blank_parts_are_dropped(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A||B|\n", encoding="utf-8")
    assert load_gold_csv(path).get("s1") == frozenset({"A", "B"})


def test_whitespace_around_ids_is_stripped(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1, A | B \n", encoding="utf-8")
    assert load_gold_csv(path).get("s1") == frozenset({"A", "B"})


def test_a_duplicate_source_id_is_rejected(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A\ns1,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_gold_csv(path)


def test_a_missing_column_is_reported_clearly(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("id,gold\ns1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_id"):
        load_gold_csv(path)


def test_a_user_supplied_normalizer_runs_on_every_id(tmp_path):
    """This replaces the paper repo's normalize_ncbi_gene_prediction hook."""
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,3|7\n", encoding="utf-8")
    gold = load_gold_csv(path, normalize=lambda i: f"NCBIGene:{i}" if i.isdigit() else i)
    assert gold.get("s1") == frozenset({"NCBIGene:3", "NCBIGene:7"})


def test_a_normalizer_returning_none_drops_the_id(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,A|SKIP\n", encoding="utf-8")
    gold = load_gold_csv(path, normalize=lambda i: None if i == "SKIP" else i)
    assert gold.get("s1") == frozenset({"A"})


def test_a_user_supplied_expander_adds_aliases(tmp_path):
    """This replaces the paper repo's expand_ctd_eval_ids hook."""
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,MESH:D001\n", encoding="utf-8")
    aliases = {"MESH:D001": {"MESH:D001", "OMIM:100"}}
    gold = load_gold_csv(
        path,
        expand=lambda ids: frozenset().union(*(aliases.get(i, {i}) for i in ids)),
    )
    assert gold.get("s1") == frozenset({"MESH:D001", "OMIM:100"})


def test_the_expander_is_not_called_for_a_no_match_label(tmp_path):
    calls = []
    path = tmp_path / "g.csv"
    path.write_text("source_id,gold_ids\ns1,\n", encoding="utf-8")

    def expand(ids):
        calls.append(ids)
        return ids

    gold = load_gold_csv(path, expand=expand)
    assert gold.get("s1") == frozenset()
    assert calls == []


def test_jsonl_gold_accepts_a_list(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps({"source_id": "s1", "gold_ids": ["A", "B"]}) + "\n", encoding="utf-8"
    )
    assert load_gold_jsonl(path).get("s1") == frozenset({"A", "B"})


def test_jsonl_gold_accepts_a_bare_string(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"source_id": "s1", "gold_ids": "A"}) + "\n", encoding="utf-8")
    assert load_gold_jsonl(path).get("s1") == frozenset({"A"})


def test_jsonl_gold_accepts_an_empty_list_as_no_match(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"source_id": "s1", "gold_ids": []}) + "\n", encoding="utf-8")
    gold = load_gold_jsonl(path)
    assert gold.get("s1") == frozenset() and "s1" in gold


def test_gold_set_is_constructible_directly():
    gold = GoldSet({"s1": frozenset({"A"})})
    assert gold.get("s1") == frozenset({"A"})


def test_is_correct_is_none_for_an_unlabelled_source():
    """None means 'no opinion', which every metric must exclude rather than count."""
    assert GoldSet({"s1": frozenset({"A"})}).is_correct("s99", "A") is None


def test_is_correct_treats_a_no_match_prediction_against_a_no_match_label_as_right():
    assert GoldSet({"s1": frozenset()}).is_correct("s1", None) is True
    assert GoldSet({"s1": frozenset({"A"})}).is_correct("s1", None) is False

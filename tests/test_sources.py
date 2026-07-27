import json

import pytest

from xwalk.sources.tabular import csv_source, jsonl_source


def test_csv_source_yields_one_record_per_row(targets_csv):
    records = list(csv_source(targets_csv, id_column="id"))
    assert len(records) == 5
    assert records[0].id == "CHEBI:17234"


def test_csv_source_drops_the_id_column_from_fields(targets_csv):
    record = next(iter(csv_source(targets_csv, id_column="id")))
    assert "id" not in record.fields
    assert record.fields["label"] == "glucose"


def test_csv_source_splits_multivalue_columns(targets_csv):
    record = next(iter(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"])))
    assert record.fields["synonyms"] == ["dextrose", "grape sugar"]


def test_csv_source_yields_empty_list_for_a_blank_multivalue_cell(targets_csv):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    assert records[4].fields["synonyms"] == []


def test_csv_source_rejects_a_missing_id_column(targets_csv):
    with pytest.raises(ValueError, match="id_column"):
        list(csv_source(targets_csv, id_column="nope"))


def test_csv_source_rejects_a_blank_id(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,label\n,glucose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        list(csv_source(path, id_column="id"))


def test_csv_source_is_lazy(targets_csv):
    """A 10M-row CSV must not be materialised to read the first record."""
    import types

    assert isinstance(csv_source(targets_csv, id_column="id"), types.GeneratorType)


def test_jsonl_source_reads_nested_values(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"pk": "a", "name": "x", "tags": ["t1", "t2"]}) + "\n", encoding="utf-8"
    )
    record = next(iter(jsonl_source(path, id_field="pk")))
    assert record.id == "a"
    assert record.fields["tags"] == ["t1", "t2"]


def test_jsonl_source_skips_blank_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"pk":"a"}\n\n{"pk":"b"}\n', encoding="utf-8")
    assert [r.id for r in jsonl_source(path, id_field="pk")] == ["a", "b"]


def test_jsonl_source_reports_the_line_number_on_bad_json(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"pk":"a"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(jsonl_source(path, id_field="pk"))


def test_jsonl_source_coerces_a_numeric_id_to_string(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"pk": 3, "name": "x"}\n', encoding="utf-8")
    assert next(iter(jsonl_source(path, id_field="pk"))).id == "3"

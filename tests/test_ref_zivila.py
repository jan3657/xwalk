from __future__ import annotations

import csv
from pathlib import Path

import pytest

from examples.ref_zivila.merge_results import merge
from examples.ref_zivila.prepare_foodon import flatten
from examples.ref_zivila.prepare_source import prepare_row, read_and_validate, select_pilot
from xwalk.records import Record


def test_prepare_row_keeps_raw_warning_but_cleans_retrieval_alias() -> None:
    row = prepare_row(
        {
            "ID": "source-1",
            "SHORT_NAME_SLO": "Stročji fižol, sušeno",
            "NAME_SLO": "NE POTRDI  Stročji fižol, sušeno",
            "TAG_NAME": "",
            "NAME_ENG": "",
            "FGNM": "Sveža zelenjava",
        }
    )

    assert row["aliases"] == "Stročji fižol, sušeno"
    assert "NE POTRDI" in row["curation_note"]
    assert "NAME_ENG missing" in row["curation_note"]


def test_read_and_validate_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "SHORT_NAME_SLO", "NAME_SLO", "TAG_NAME", "NAME_ENG", "FGNM"],
        )
        writer.writeheader()
        base = {
            "ID": "same",
            "SHORT_NAME_SLO": "Ananas",
            "NAME_SLO": "Ananas",
            "TAG_NAME": "",
            "NAME_ENG": "Pineapple",
            "FGNM": "Sadje",
        }
        writer.writerow(base)
        writer.writerow(base)

    with pytest.raises(ValueError, match="duplicate ID"):
        read_and_validate(path)


def test_select_pilot_requires_every_requested_id() -> None:
    rows = [{"ID": "a"}, {"ID": "b"}]
    assert [row["ID"] for row in select_pilot(rows, ("b",))] == ["b"]
    with pytest.raises(ValueError, match="absent"):
        select_pilot(rows, ("missing",))


def test_foodon_flatten_resolves_parent_labels() -> None:
    records = [
        Record(
            id="FOODON:1",
            fields={
                "label": "fruit food product",
                "synonyms": [],
                "definition": "",
                "parents": [],
            },
        ),
        Record(
            id="FOODON:2",
            fields={
                "label": "pineapple food product",
                "synonyms": ["pineapple"],
                "definition": "A food product.",
                "parents": ["FOODON:1", "EXTERNAL:9"],
            },
        ),
    ]

    rows = flatten(records)
    assert rows[1]["parent_labels"] == "fruit food product"
    assert rows[1]["synonyms"] == "pineapple"


def test_merge_populates_accepted_and_preserves_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    mapping_path = tmp_path / "mapping.csv"
    out_path = tmp_path / "out.csv"
    long_out_path = tmp_path / "long.csv"

    fieldnames = [
        "ID",
        "SHORT_NAME_SLO",
        "NAME_SLO",
        "TAG_NAME",
        "NAME_ENG",
        "FGNM",
        "FOODON",
        "SNOMEDCT",
        "FOODEX",
        "HANSARD",
        "Chembl (Maybe)",
        "UMLS (maybe)",
    ]
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"ID": "id-1", "NAME_ENG": "Pineapple", "FGNM": "Fruit", "FOODON": ""})
        writer.writerow({"ID": "id-2", "NAME_ENG": "Jam", "FGNM": "Sweets", "FOODON": ""})
        writer.writerow({"ID": "id-3", "NAME_ENG": "Unknown", "FGNM": "Other", "FOODON": ""})

    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "matched_id", "confidence", "status", "reason", "explanation"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "id-1",
                "matched_id": "FOODON:00001234",
                "confidence": "0.95",
                "status": "accepted",
                "reason": "exact_match",
                "explanation": "Exact match",
            }
        )
        writer.writerow(
            {
                "source_id": "id-2",
                "matched_id": "FOODON:00005678",
                "confidence": "0.40",
                "status": "needs_review",
                "reason": "broader",
                "explanation": "Plausible broader",
            }
        )
        writer.writerow(
            {
                "source_id": "id-3",
                "matched_id": "",
                "confidence": "0.10",
                "status": "no_match",
                "reason": "abstain",
                "explanation": "No candidate",
            }
        )

    total, matched = merge(
        input_csv=source_path,
        mapping_csv=mapping_path,
        out_csv=out_path,
        vocabulary="foodon",
        long_out=long_out_path,
    )

    assert total == 3
    assert matched == 1

    with out_path.open("r", encoding="utf-8", newline="") as handle:
        merged_rows = list(csv.DictReader(handle))

    assert len(merged_rows) == 3
    assert merged_rows[0]["ID"] == "id-1"
    assert merged_rows[0]["FOODON"] == "FOODON:00001234"
    assert merged_rows[1]["ID"] == "id-2"
    assert merged_rows[1]["FOODON"] == ""
    assert merged_rows[2]["ID"] == "id-3"
    assert merged_rows[2]["FOODON"] == ""

    with long_out_path.open("r", encoding="utf-8", newline="") as handle:
        long_rows = list(csv.DictReader(handle))
    assert len(long_rows) == 3
    assert long_rows[0]["status"] == "accepted"
    assert long_rows[1]["status"] == "needs_review"
    assert long_rows[2]["status"] == "no_match"

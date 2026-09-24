"""The Ref_zivila gold file built from the adjudicated Qwen/Jev disagreements."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from xwalk.evaluate.gold import load_gold_csv

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "examples" / "ref_zivila" / "gold_adjudicated.csv"
ADJUDICATED = ROOT / "docs" / "superpowers" / "specs" / "2026-09-22-jev-adjudicated-sample.csv"
SCRIPT = ROOT / "scripts" / "adjudication_to_gold.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("adjudication_to_gold", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_gold_file_is_labelled_foodon_and_keeps_both_acceptable_ids() -> None:
    gold = load_gold_csv(GOLD)
    assert len(gold) >= 35
    assert all(i.startswith("FOODON:") for ids in gold.labels.values() for i in ids)
    with ADJUDICATED.open(encoding="utf-8", newline="") as handle:
        both = [r for r in csv.DictReader(handle) if r["verdict"] == "both_acceptable"]
    assert both
    # Two ids wherever both deciders named one; a blank side ("no match") is dropped.
    for row in both:
        named = {i for i in (row["qwen_id"], row["jev_id"]) if i}
        assert gold.labels[row["source_id"]] == named
    assert any(len(gold.labels[row["source_id"]]) == 2 for row in both)


def test_conversion_maps_each_verdict(tmp_path: Path) -> None:
    source = tmp_path / "adjudicated.csv"
    fields = ["source_id", "qwen_id", "jev_id", "verdict"]
    rows = [
        ("d", "FOODON:q4", "FOODON:j4", "unsure"),
        ("a", "FOODON:q1", "FOODON:j1", "jev"),
        ("c", "FOODON:q3", "FOODON:j3", "both_acceptable"),
        ("b", "FOODON:q2", "FOODON:j2", "qwen"),
        ("e", "FOODON:q5", "FOODON:j5", "both_wrong"),
        ("f", "FOODON:q6", "", "both_acceptable"),
        ("g", "FOODON:q7", "", "jev"),
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    out = tmp_path / "gold.csv"

    _script().convert(source, out)

    assert out.read_text(encoding="utf-8").splitlines() == [
        "source_id,gold_ids",
        "a,FOODON:j1",
        "b,FOODON:q2",
        "c,FOODON:q3|FOODON:j3",
        "f,FOODON:q6",
        "g,",
    ]


def test_conversion_rejects_an_unknown_verdict(tmp_path: Path) -> None:
    source = tmp_path / "adjudicated.csv"
    source.write_text("source_id,qwen_id,jev_id,verdict\na,FOODON:q,FOODON:j,maybe\n")
    with pytest.raises(ValueError, match="maybe"):
        _script().convert(source, tmp_path / "gold.csv")


def test_the_committed_gold_file_matches_a_fresh_conversion(tmp_path: Path) -> None:
    """Drift guard: the committed gold is exactly what the script makes of the adjudication."""
    out = tmp_path / "gold.csv"
    _script().convert(ADJUDICATED, out)
    assert out.read_bytes() == GOLD.read_bytes()

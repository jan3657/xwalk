"""Dense retrieval on Ref_zivila rows that carry only a Slovenian name (A2).

`examples/ref_zivila/gold_slo_mini.csv` holds ten source rows whose `mention_en` is
empty. Their FoodOn ids were chosen by a person, by looking each Slovenian name up
against the labels in `data/ref_zivila/targets/foodon.csv`, not taken from any run.
Where several records are equally right (raw and unspecified, pea and pea pod for
"grah z luščinami"), all are listed.

BM25 cannot reach these rows: the queries share no token with any English label. The
multilingual dense retriever is what should recover them.

`_env.sh` runs HuggingFace offline, so the model must already be in HF_HOME; pull it once with
`HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from xwalk.retrieval.dense import
SentenceTransformerEncoder as E; E('intfloat/multilingual-e5-small')"`.
"""

from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import E5_MODEL, e5_retriever, load_encoder_or_skip
from xwalk.config import JobSpec
from xwalk.evaluate.gold import GoldSet, load_gold_csv
from xwalk.evaluate.recall import retrieval_recall

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "examples" / "ref_zivila" / "jobs" / "foodon" / "job_jev.yaml"
MINI_GOLD = ROOT / "examples" / "ref_zivila" / "gold_slo_mini.csv"
# Git-ignored run data: present on the development machine, absent from a fresh clone.
SOURCE = ROOT / "data" / "ref_zivila" / "source.csv"
TARGET = ROOT / "data" / "ref_zivila" / "targets" / "foodon.csv"

# The block commented out in the job file (Ref_zivila screens up to 300 candidates anyway).
E5 = e5_retriever(limit=150)

needs_data = pytest.mark.skipif(
    not (SOURCE.exists() and TARGET.exists()), reason="data/ref_zivila is not present"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_mini_gold_has_ten_rows_of_foodon_ids() -> None:
    gold = load_gold_csv(MINI_GOLD)
    assert len(gold) == 10
    for ids in gold.labels.values():
        assert ids
        assert all(i.startswith("FOODON:") for i in ids)


@needs_data
def test_the_mini_gold_is_slovenian_only_and_names_real_targets() -> None:
    gold = load_gold_csv(MINI_GOLD)
    sources = {row["ID"]: row for row in _read_csv(SOURCE)}
    targets = {row["id"] for row in _read_csv(TARGET)}
    for source_id, ids in gold.labels.items():
        assert not sources[source_id]["mention_en"].strip(), source_id
        assert ids <= targets, ids - targets


def _recall_at_50(data: dict[str, Any], gold: GoldSet, kinds: set[str], index_dir: Path) -> float:
    data = copy.deepcopy(data)
    data["retrievers"] = [r for r in data["retrievers"] if r["kind"] in kinds]
    job = JobSpec.model_validate(data).model_copy(update={"base_dir": JOB.parent.resolve()})
    return retrieval_recall(job, gold, index_dir=index_dir, ks=(50,))[50]


@pytest.mark.dense
@needs_data
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "A2 Slovenian gate: multilingual-e5-small measured fused 0.20, dense-only 0.20, "
        "BM25-only 0.00 at @50 (2026-09-23; e5-large 0.30); see spec Results"
    ),
)
def test_dense_recovers_slovenian_only_rows(tmp_path: Path) -> None:
    """A2 gate: the shipped BM25+dense fusion finds a gold id in the top 50 for at least
    seven of the ten rows; BM25 alone for at most two. The whole FoodOn target is indexed;
    only the source is cut down to the ten rows, since recall renders every row it gets."""
    load_encoder_or_skip(E5_MODEL)
    gold = load_gold_csv(MINI_GOLD)
    rows = [row for row in _read_csv(SOURCE) if row["ID"] in gold.labels]
    assert len(rows) == len(gold)
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    # The job's commented-out dense block, uncommented (a no-op once the job ships it).
    if not any(r["kind"] == "dense" for r in data["retrievers"]):
        data["retrievers"].append(dict(E5))
    data["source"]["path"] = str(source)

    started = time.perf_counter()
    fused = _recall_at_50(data, gold, {"bm25", "dense"}, tmp_path / "fused")
    dense = _recall_at_50(data, gold, {"dense"}, tmp_path / "dense")
    bm25 = _recall_at_50(data, gold, {"bm25"}, tmp_path / "bm25")
    print(
        f"Slovenian-only recall@50: fused {fused:.2f}, dense-only {dense:.2f}, "
        f"BM25-only {bm25:.2f} ({time.perf_counter() - started:.0f}s)"
    )
    assert fused >= 0.7, (fused, dense, bm25)
    assert bm25 <= 0.2, (fused, dense, bm25)

"""The offline retrieval-recall harness, pinned to the baselines the evaluation reported.

Every later retrieval change on this branch is judged by `retrieval_recall`, so it must
first reproduce the recall the live decider runs measured with the shipped BM25 setup.
If it does not, the harness is wrong (query rendering, fusion, gold handling), not the
retriever.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest
import yaml

from xwalk.config import JobSpec, load_job
from xwalk.evaluate.gold import GoldSet, load_gold_csv
from xwalk.evaluate.recall import retrieval_recall

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

SAMPLES = {"cafeteria_fcd": 0.94, "chebi": 0.60, "ncbi_disease": 0.58, "nlm_gene": 0.70}

# The dense retriever the sample decider jobs ship beside BM25 (A2). E5 wants its prefixes.
E5: dict[str, Any] = {
    "kind": "dense",
    "name": "e5",
    "model": "intfloat/multilingual-e5-small",
    "limit": 150,
    "query_prefix": "query: ",
    "doc_prefix": "passage: ",
}
# Only the English samples: the Ref_zivila job keeps its dense block commented out until
# the Slovenian gate (tests/test_ref_zivila_dense.py) passes.
DECIDER_JOBS = [EXAMPLES / ex / "job_jev.yaml" for ex in sorted(SAMPLES)]


def bm25_only(data: dict[str, Any]) -> None:
    """The baselines were measured with BM25 alone; they must not need the dense extra."""
    data["retrievers"] = [r for r in data["retrievers"] if r["kind"] == "bm25"]


def with_dense(data: dict[str, Any]) -> None:
    """BM25 plus E5, whether or not the job file already ships the dense block."""
    if not any(r["kind"] == "dense" for r in data["retrievers"]):
        data["retrievers"].append(dict(E5))


class RecallFor(Protocol):
    def __call__(
        self, example: str, mutate: Callable[[dict[str, Any]], None] | None = None
    ) -> float: ...


@pytest.fixture
def recall_for(tmp_path: Path) -> RecallFor:
    calls = itertools.count()

    def _recall_for(example: str, mutate: Callable[[dict[str, Any]], None] | None = None) -> float:
        job_path = EXAMPLES / example / "job_jev.yaml"
        data: dict[str, Any] = yaml.safe_load(job_path.read_text(encoding="utf-8"))
        if mutate is not None:
            mutate(data)
        # Same base-directory handling as `load_job`: paths stay relative to the job file.
        job = JobSpec.model_validate(data).model_copy(
            update={"base_dir": job_path.parent.resolve()}
        )
        gold = load_gold_csv(EXAMPLES / example / "sample" / "gold.csv")
        # A fresh directory per call, so a mutated build never reopens an earlier index.
        index_dir = tmp_path / f"{example}-{next(calls)}"
        return retrieval_recall(job, gold, index_dir=index_dir, ks=(200,))[200]

    return _recall_for


@pytest.mark.parametrize("example", sorted(SAMPLES))
def test_shipped_bm25_reproduces_the_baselines(recall_for: RecallFor, example: str) -> None:
    assert abs(recall_for(example, bm25_only) - SAMPLES[example]) <= 0.02


def test_recall_for_can_compare_a_mutated_job_against_the_baseline(
    recall_for: RecallFor, tmp_path: Path
) -> None:
    """Later tasks write `recall_for(ex, mutate) >= recall_for(ex)` in one test. Each call
    needs its own index directory: a mutation that changes the BM25 schema (an analyzer,
    say) would otherwise reopen the previous index and tantivy refuses a schema mismatch.
    Today's schema is fixed, so the directory count is what pins that down."""

    def plain_bm25(data: dict[str, Any]) -> None:
        bm25_only(data)
        data["retrievers"][0]["exact_fields"] = []

    baseline = recall_for("cafeteria_fcd", bm25_only)
    assert 0.0 <= recall_for("cafeteria_fcd", plain_bm25) <= 1.0
    assert recall_for("cafeteria_fcd", bm25_only) == baseline
    assert len(list(tmp_path.iterdir())) == 3


def test_no_match_and_unlabelled_rows_are_outside_the_denominator(tmp_path: Path) -> None:
    """The shipped samples hold no "no match" rows, so pin that path here: T1 is a hit,
    T3 cannot be, T2 is a deliberate no-match, and every other source is unlabelled.
    Relies on BM25 ranking a gold id for cafeteria_fcd-T1 within the top 10."""
    job = load_job(EXAMPLES / "cafeteria_fcd" / "job_jev.yaml")
    job = job.model_copy(update={"retrievers": [r for r in job.retrievers if r.kind == "bm25"]})
    shipped = load_gold_csv(EXAMPLES / "cafeteria_fcd" / "sample" / "gold.csv")
    gold = GoldSet(
        {
            "cafeteria_fcd-T1": shipped.labels["cafeteria_fcd-T1"],
            "cafeteria_fcd-T2": frozenset(),
            "cafeteria_fcd-T3": frozenset({"FOODON:not-a-target"}),
        }
    )
    assert retrieval_recall(job, gold, index_dir=tmp_path, ks=(10, 200)) == {10: 0.5, 200: 0.5}


def test_a_gold_file_with_nothing_to_retrieve_is_an_error(tmp_path: Path) -> None:
    job = load_job(EXAMPLES / "cafeteria_fcd" / "job_jev.yaml")
    with pytest.raises(ValueError, match="no source record has a gold id"):
        retrieval_recall(job, GoldSet({"cafeteria_fcd-T1": frozenset()}), index_dir=tmp_path)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A1 gate: en_stem + fuzzy_distance 1 measured 0.94/0.70/0.58/0.70, "
        "mean 0.730 < 0.735 (2026-09-23); see spec Results"
    ),
)
def test_stem_and_fuzzy_lift_recall(recall_for: RecallFor) -> None:
    def stem_and_fuzzy(data: dict[str, Any]) -> None:
        bm25_only(data)
        data["retrievers"][0]["analyzer"] = "en_stem"
        data["retrievers"][0]["fuzzy_distance"] = 1

    recalls = {ex: recall_for(ex, stem_and_fuzzy) for ex in sorted(SAMPLES)}
    print(f"recall@200 with en_stem + fuzzy 1: {recalls}")
    for ex, recall in recalls.items():
        assert recall >= SAMPLES[ex] - 0.0, f"{ex}: {recall:.3f} < {SAMPLES[ex]}"
    assert sum(recalls.values()) / len(recalls) >= 0.705 + 0.03, recalls


@pytest.mark.parametrize("job_path", DECIDER_JOBS, ids=lambda p: p.parent.name)
def test_every_sample_decider_job_ships_e5_beside_bm25(job_path: Path) -> None:
    data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    assert [r["kind"] for r in data["retrievers"]] == ["bm25", "dense"]
    assert data["retrievers"][1] == E5


@pytest.mark.dense
def test_dense_lifts_english_recall(recall_for: RecallFor) -> None:
    """A2 gate, against the shipped BM25 baselines (A1 was not adopted, so BM25 is as
    shipped): mean recall@200 at least 0.08 above 0.705, no sample more than 0.02 below."""
    pytest.importorskip("sentence_transformers")
    recalls = {ex: recall_for(ex, with_dense) for ex in sorted(SAMPLES)}
    print(f"recall@200 with BM25 + e5: {recalls}")
    for ex, recall in recalls.items():
        assert recall >= SAMPLES[ex] - 0.02, f"{ex}: {recall:.3f} < {SAMPLES[ex]} - 0.02"
    assert sum(recalls.values()) / len(recalls) >= 0.705 + 0.08, recalls

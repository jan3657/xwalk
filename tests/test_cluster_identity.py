"""Clustering run identity: the pool's retrieval heads are part of the run fingerprint.

A lexical run and a dense run can cluster the same records differently, so a run
directory made with one must refuse the other instead of resuming it with zero calls.
Every refusal happens before anything is written or any model is called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cluster_helpers import CHOCOLATE, Oracle, cluster, concepts, partition, records, settings
from tests.conftest import FIXTURES
from tests.test_cluster_pool import ToyEncoder
from xwalk import ops
from xwalk.cluster import ClusterRunError
from xwalk.cluster.job import ClusterJobSpec

TABLE = {"a": ("heart attack", "mi"), "b": ("myocardial infarction", "mi")}
# No scan: only retrieval can bring the two synonyms together.
CONFIG = settings(order="input", scan_below=0)


class RenamedEncoder(ToyEncoder):
    @property
    def name(self) -> str:
        return "toy-2"


class PrefixedEncoder(ToyEncoder):
    settings = {"query_prefix": "query: "}


def _files(run: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run.iterdir()) if p.is_file()}


async def _run(run: Path, encoder: object | None) -> tuple[object, Oracle]:
    oracle = Oracle(concepts(TABLE))
    report = await cluster(records(TABLE), run, oracle, settings=CONFIG, encoder=encoder)
    return report, oracle


async def _refused(run: Path, encoder: object | None) -> ClusterRunError:
    before = _files(run)
    oracle = Oracle(concepts(TABLE))
    with pytest.raises(ClusterRunError) as caught:
        await cluster(records(TABLE), run, oracle, settings=CONFIG, encoder=encoder)
    assert oracle.llm.requests == []  # refused before any call
    assert _files(run) == before  # and before any write
    return caught.value


async def test_lexical_and_dense_runs_have_different_fingerprints(tmp_path):
    lexical, _ = await _run(tmp_path / "lexical", None)
    dense, _ = await _run(tmp_path / "dense", ToyEncoder())
    assert lexical.run_fingerprint != dense.run_fingerprint
    # the probe that found it: dense retrieval joins the synonyms, lexical cannot
    assert partition(tmp_path / "lexical") == {frozenset({"a"}), frozenset({"b"})}
    assert partition(tmp_path / "dense") == {frozenset({"a", "b"})}
    assert lexical.components["retrieval"]["dense"] is None
    assert dense.components["retrieval"]["dense"]["encoder"]["name"] == "toy"
    # pool settings keep their own component
    assert lexical.components["pool"]["scan_below"] == 0


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (None, ToyEncoder()),
        (ToyEncoder(), None),
        (ToyEncoder(), RenamedEncoder()),
        (ToyEncoder(), PrefixedEncoder()),
    ],
    ids=["lexical-then-dense", "dense-then-lexical", "encoder-name", "encoder-settings"],
)
async def test_a_changed_retrieval_head_is_refused(tmp_path, first, second):
    await _run(tmp_path / "run", first)
    error = await _refused(tmp_path / "run", second)
    assert error.code == "run_fingerprint_mismatch"
    assert [k.split(".")[0] for k in error.data["differences"]] == ["retrieval"]


@pytest.mark.parametrize("encoder", [None, ToyEncoder()], ids=["lexical", "dense"])
async def test_an_unchanged_retrieval_head_resumes_with_zero_calls(tmp_path, encoder):
    first, _ = await _run(tmp_path / "run", encoder)
    again, oracle = await _run(tmp_path / "run", encoder)
    assert again.run_fingerprint == first.run_fingerprint
    assert oracle.llm.requests == []


def _make_format_1(run: Path) -> None:
    """Rewrite a manifest the way format 1 recorded it: the pool settings had replaced
    the retrieval identity, so the stored fingerprint cannot say which heads ran."""
    path = run / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    components = manifest["fingerprint_components"]
    components["format_version"] = 1
    del components["retrieval"]
    manifest["run_fingerprint"] = "f" * 16
    path.write_text(json.dumps(manifest), encoding="utf-8")


async def test_a_format_1_run_directory_is_refused_with_a_clear_message(tmp_path):
    await _run(tmp_path / "run", None)
    _make_format_1(tmp_path / "run")
    error = await _refused(tmp_path / "run", None)
    assert error.code == "run_format_outdated"
    assert "retrieval" in str(error) and "new --out" in str(error)


def test_ops_reports_a_format_1_run_directory_as_exit_3(tmp_path, monkeypatch):
    judge = Oracle(concepts(CHOCOLATE))
    monkeypatch.setattr(ClusterJobSpec, "build_llm", lambda self: judge.llm)
    job = FIXTURES / "cluster_tiny" / "job.yaml"
    assert ops.cluster(job, tmp_path / "run").exit_code in (0, 1)
    _make_format_1(tmp_path / "run")
    calls = len(judge.llm.requests)
    with pytest.raises(ops.OpError) as caught:
        ops.cluster(job, tmp_path / "run")
    assert caught.value.result.exit_code == ops.EXIT_RUNTIME
    assert caught.value.code == "run_format_outdated"
    assert len(judge.llm.requests) == calls

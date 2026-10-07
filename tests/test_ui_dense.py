"""Dense explorer configuration and shared-index reuse, without downloading weights."""

import json

from tests.test_dense import ToyEncoder
from xwalk import ops
from xwalk.config import load_job
from xwalk.records import Record
from xwalk.retrieval.dense import DenseRetriever
from xwalk.ui.api import Workspace, dispatch
from xwalk.ui.library import Library
from xwalk.ui.projects import TargetChoice, library_lookup_key, lookup_job
from xwalk.ui.retrieval import heads


def test_dense_settings_reach_generated_projects_and_do_not_reuse_bm25_only_jobs(tmp_path):
    lib = Library(tmp_path / "library")
    lib.add("Toy", [Record("T1", {"label": "glucose"})], slug="toy", source="test")
    choice = TargetChoice(libraries=["toy"])
    lexical = lookup_job(lib, choice, library_lookup_key(["toy"]), tmp_path / "cache")
    hybrid = lookup_job(
        lib, choice, library_lookup_key(["toy"]), tmp_path / "cache", retrievers=heads(dense=True)
    )
    assert lexical != hybrid
    assert [r.name for r in load_job(hybrid).retrievers] == [
        "bm25",
        "dense-general",
        "dense-biomedical",
    ]


def test_warmed_dense_indexes_are_shared_by_lookup_and_mapping_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "_default_encoder_factory", lambda spec: ToyEncoder())
    out = tmp_path / "library"
    Library(out).add(
        "Toy",
        [Record("T1", {"label": "glucose"}), Record("T2", {"label": "fructose"})],
        slug="toy",
        source="test",
    )
    root = tmp_path / "visitor"
    root.mkdir()
    ws = Workspace(
        root,
        public=True,
        shared_library=[out],
        lookup_cache=tmp_path / "cache",
        retrieval_heads=heads(dense=True),
    )
    found = dispatch(
        ws, "POST", "/api/library/search", {"libraries": ["toy"], "query": "glucose", "top_k": 2}
    )
    assert set(found["indexes"]) == {"bm25", "dense-general", "dense-biomedical"}
    assert set(found["candidates"][0]["retrievers"]) == set(found["indexes"])

    def forbid_build(*args, **kwargs):
        raise AssertionError("the warmed dense index must be opened, not encoded again")

    monkeypatch.setattr(DenseRetriever, "build", forbid_build)
    started = dispatch(
        ws,
        "POST",
        "/api/quickmap",
        {"terms": "glucose", "target": {"libraries": ["toy"]}, "mode": "offline"},
    )
    task = ws.tasks.wait(started["task"]["id"])
    assert task.result["status"] == "ok", task.result
    assert task.result["counts"]["matched"] == 1
    job = root / started["project"]["job_path"]
    assert [r.name for r in load_job(job).retrievers] == [
        "bm25",
        "dense-general",
        "dense-biomedical",
    ]
    assert not (root / started["project"]["suggested_out"] / "index").exists()

    # A project edited to have other targets cannot reuse the original library's index.
    (job.parent / "targets.jsonl").write_text(json.dumps({"id": "T3", "label": "other"}) + "\n")
    assert ws._prepare_shared_index(job) is None

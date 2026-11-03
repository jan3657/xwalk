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
    # The injected encoder needs no optional packages; bypass only the extras preflight.
    monkeypatch.setattr(ops, "_extra_errors", lambda job, **kwargs: [])
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


# --- API dense retrieval: `xwalk ui --dense api` --------------------------------------

import httpx  # noqa: E402
import pytest  # noqa: E402

from xwalk.cli.main import _build_parser as build_parser  # noqa: E402
from xwalk.retrieval import embeddings_api  # noqa: E402
from xwalk.ui import make_server  # noqa: E402
from xwalk.ui.workbench import credentials  # noqa: E402


def test_the_api_mode_adds_the_default_encoder_with_name_vectors():
    found = heads(dense="api")
    assert [h["name"] for h in found] == ["bm25", "dense-qwen3"]
    api = found[1]
    assert api["encoder"] == "openai_embeddings" and "model" not in api
    assert api["name_fields"] == ["label", "synonyms"]


def test_true_and_local_still_mean_the_local_models():
    assert heads(dense=True) == heads(dense="local")
    assert [h["name"] for h in heads(dense="local")][1:] == ["dense-general", "dense-biomedical"]
    assert heads(dense=False) == heads(dense="off") == heads()


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="dense"):
        heads(dense="remote")


def test_generated_projects_get_the_default_qwen3_retriever(tmp_path):
    lib = Library(tmp_path / "library")
    lib.add("Toy", [Record("T1", {"label": "glucose"})], slug="toy", source="test")
    job = lookup_job(
        lib,
        TargetChoice(libraries=["toy"]),
        library_lookup_key(["toy"]),
        tmp_path / "cache",
        retrievers=heads(dense="api"),
    )
    spec = load_job(job).retrievers[1]
    assert spec.model == embeddings_api.DEFAULT_MODEL
    assert spec.api_key_env == "OPENROUTER_API_KEY"
    assert spec.name_fields == ["label", "synonyms"]
    assert spec.required_extra is None


def test_the_cli_reads_dense_with_and_without_a_mode():
    parser = build_parser()
    base = ["ui", "--no-browser"]
    assert parser.parse_args([*base, "--dense"]).dense == "local"
    assert parser.parse_args([*base, "--dense", "api"]).dense == "api"
    assert parser.parse_args(base).dense == "off"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--dense", "remote"])


def _no_sentence_transformers(monkeypatch):
    real = __import__("importlib.util").util.find_spec

    def find_spec(name, *args):
        if name == "sentence_transformers":
            return None
        return real(name, *args)

    monkeypatch.setattr("importlib.util.find_spec", find_spec)


def test_the_api_mode_needs_no_local_model_packages(tmp_path, monkeypatch):
    _no_sentence_transformers(monkeypatch)
    server = make_server(str(tmp_path), port=0, token="t", dense="api")
    try:
        names = [h["name"] for h in server.workspace.retrieval_heads]
        assert "dense-qwen3" in names
    finally:
        server.server_close()


def test_the_api_mode_is_refused_offline(tmp_path):
    with pytest.raises(ValueError, match="offline"):
        make_server(str(tmp_path), port=0, token="t", dense="api", offline_only=True)


def test_a_public_site_needs_the_operators_embedding_key(tmp_path, monkeypatch):
    """Library indexes are shared by every visitor, so they are embedded with the
    operator's key from the environment, never one visitor's session key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_server(str(tmp_path), port=0, token="t", dense="api", public=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-operator")
    server = make_server(str(tmp_path), port=0, token="t", dense="api", public=True)
    server.server_close()


def test_the_settings_page_offers_the_embedding_key_locally(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ws = Workspace(tmp_path, retrieval_heads=heads(dense="api"))
    listed = {c["env"]: c for c in credentials(ws, {})["credentials"]}
    assert listed["OPENROUTER_API_KEY"]["set"] is False
    plain = Workspace(tmp_path, retrieval_heads=heads())
    assert "OPENROUTER_API_KEY" not in {c["env"] for c in credentials(plain, {})["credentials"]}


def test_a_public_visitor_is_not_shown_the_operators_embedding_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-operator")
    ws = Workspace(tmp_path, public=True, retrieval_heads=heads(dense="api"))
    assert "OPENROUTER_API_KEY" not in {c["env"] for c in credentials(ws, {})["credentials"]}


def test_library_search_runs_through_the_api_head(tmp_path, monkeypatch):
    bodies: list[dict] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        data = []
        for i, text in enumerate(body["input"]):
            vector = [0.0] * 1024
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            data.append({"index": i, "embedding": vector})
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 1}})

    real = httpx.Client

    def client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(gateway)
        return real(**kwargs)

    monkeypatch.setattr("xwalk.retrieval.embeddings_api.httpx.Client", client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    out = tmp_path / "library"
    Library(out).add(
        "Toy",
        [
            Record("T1", {"label": "glucose", "synonyms": ["dextrose"]}),
            Record("T2", {"label": "fructose", "synonyms": []}),
        ],
        slug="toy",
        source="test",
    )
    root = tmp_path / "visitor"
    root.mkdir()
    ws = Workspace(
        root,
        shared_library=[out],
        lookup_cache=tmp_path / "cache",
        retrieval_heads=heads(dense="api"),
    )
    found = dispatch(
        ws, "POST", "/api/library/search", {"libraries": ["toy"], "query": "dextrose", "top_k": 2}
    )
    assert set(found["indexes"]) == {"bm25", "dense-qwen3"}
    assert found["candidates"][0]["id"] == "T1"
    assert {b["model"] for b in bodies} == {embeddings_api.DEFAULT_MODEL}
    sent = [t for b in bodies for t in b["input"]]
    assert "dextrose" in sent  # the synonym was embedded as its own name vector


def test_a_missing_embedding_key_is_reported_as_a_credential_with_a_settings_hint(
    tmp_path, monkeypatch
):
    from xwalk.ui.api import ApiError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "library"
    Library(out).add("Toy", [Record("T1", {"label": "glucose"})], slug="toy", source="test")
    ws = Workspace(
        tmp_path,
        shared_library=[out],
        lookup_cache=tmp_path / "cache",
        retrieval_heads=heads(dense="api"),
    )
    with pytest.raises(ApiError) as info:
        dispatch(ws, "POST", "/api/library/search", {"libraries": ["toy"], "query": "glucose"})
    assert info.value.code == "credential_missing"
    assert "OPENROUTER_API_KEY" in str(info.value) and "Settings page" in str(info.value)

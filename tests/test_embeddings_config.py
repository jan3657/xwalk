"""`encoder: openai_embeddings` in a job file: validation, credentials, extras, index.

No network: the one end-to-end index test routes the encoder's httpx client through
`httpx.MockTransport`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from tests.conftest import FIXTURES
from xwalk import ops
from xwalk.config import (
    CredentialMissingError,
    JobSpec,
    JobValidationError,
    RetrieverSpec,
    load_job,
)
from xwalk.retrieval.embeddings_api import OpenAIEmbeddingsEncoder

JOB = FIXTURES / "job_tiny.yaml"
BASE = yaml.safe_load(JOB.read_text(encoding="utf-8"))
CLUSTER = FIXTURES / "cluster_tiny" / "job.yaml"

API = {
    "kind": "dense",
    "name": "api",
    "encoder": "openai_embeddings",
    "model": "vendor/embed-1",
    "base_url": "https://gateway.example/api/v1",
    "api_key_env": "XWALK_TEST_EMBED_KEY",
    "dimensions": 26,
    "send_dimensions": False,
}


def _job_file(tmp_path: Path, retriever: dict[str, Any] | None = None) -> Path:
    data = copy.deepcopy(BASE)
    data["prompts"]["slots"] = str((FIXTURES / data["prompts"]["slots"]).resolve())
    for role in ("target", "source"):
        source = FIXTURES / data[role]["path"]
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    data["retrievers"] = [retriever if retriever is not None else dict(API)]
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _spec(**changes: Any) -> RetrieverSpec:
    return RetrieverSpec.model_validate({**API, **changes})


# --- the spec ------------------------------------------------------------------------


def test_an_api_dense_retriever_loads(tmp_path):
    job = load_job(_job_file(tmp_path))
    spec = job.retrievers[0]
    assert spec.encoder == "openai_embeddings" and spec.dimensions == 26


def test_the_encoder_defaults_to_sentence_transformers():
    assert RetrieverSpec(kind="dense", model="m").encoder == "sentence_transformers"


@pytest.mark.parametrize("missing", ["base_url", "dimensions"])
def test_an_api_encoder_needs_a_base_url_and_dimensions(missing):
    data = {k: v for k, v in API.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        RetrieverSpec.model_validate(data)


def test_an_inline_key_is_refused():
    with pytest.raises(ValueError, match="api_key_env"):
        _spec(api_key="sk-live")


@pytest.mark.parametrize("field, value", [("device", "cuda"), ("revision", "abc")])
def test_local_model_settings_are_refused_for_an_api_encoder(field, value):
    with pytest.raises(ValueError, match=f"{field} does not apply to encoder 'openai_embeddings'"):
        _spec(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("base_url", "https://x.example/v1"),
        ("api_key_env", "K"),
        ("dimensions", 8),
        ("send_dimensions", True),
        ("truncate", True),
        ("query_input_type", "q"),
        ("doc_input_type", "d"),
        ("provider", {"order": ["x"]}),
        ("batch_size", 8),
        ("timeout", 5.0),
        ("max_retries", 1),
    ],
)
def test_api_settings_are_refused_for_a_local_encoder(field, value):
    with pytest.raises(
        ValueError, match=f"{field} does not apply to encoder 'sentence_transformers'"
    ):
        RetrieverSpec.model_validate({"kind": "dense", "model": "m", field: value})


@pytest.mark.parametrize("field, value", [("encoder", "openai_embeddings"), ("dimensions", 8)])
def test_dense_settings_are_refused_for_bm25(field, value):
    with pytest.raises(ValueError, match=f"{field} does not apply to kind 'bm25'"):
        RetrieverSpec.model_validate({"kind": "bm25", field: value})


@pytest.mark.parametrize(
    "field, value", [("dimensions", 0), ("batch_size", 0), ("timeout", 0), ("max_retries", -1)]
)
def test_api_settings_are_range_checked(field, value):
    with pytest.raises(ValueError, match=field):
        _spec(**{field: value})


def test_a_misspelled_key_names_the_nearest_field(tmp_path):
    path = _job_file(tmp_path, {**API, "dimension": 26})
    with pytest.raises(JobValidationError, match="dimensions"):
        load_job(path)


def test_an_api_encoder_needs_no_optional_extra():
    assert _spec().required_extra is None
    assert RetrieverSpec(kind="dense", model="m").required_extra == (
        "dense",
        "sentence_transformers",
    )


# --- building the encoder ------------------------------------------------------------


def test_the_spec_builds_an_embeddings_encoder_with_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("XWALK_TEST_EMBED_KEY", "sk-from-env")
    encoder = JobSpec.build_encoder(
        _spec(
            query_prefix="query: ",
            query_input_type="search_query",
            provider={"order": ["nebius"]},
            truncate=True,
        )
    )
    assert isinstance(encoder, OpenAIEmbeddingsEncoder)
    assert encoder.name == "vendor/embed-1" and encoder.dimension == 26
    assert encoder.settings["query_prefix"] == "query: "
    assert encoder.settings["query_input_type"] == "search_query"
    assert encoder.settings["provider"] == {"order": ["nebius"]}
    assert encoder.settings["truncate"] is True
    assert "sk-from-env" not in json.dumps(encoder.settings)


def test_an_unset_key_variable_is_a_credential_error(monkeypatch):
    monkeypatch.delenv("XWALK_TEST_EMBED_KEY", raising=False)
    with pytest.raises(CredentialMissingError) as info:
        JobSpec.build_encoder(_spec(name="api"))
    assert info.value.variable == "XWALK_TEST_EMBED_KEY"
    assert "api_key_env" in str(info.value)


def test_no_key_variable_builds_a_keyless_encoder_for_a_local_server():
    data = {k: v for k, v in API.items() if k != "api_key_env"}
    encoder = JobSpec.build_encoder(RetrieverSpec.model_validate(data))
    assert isinstance(encoder, OpenAIEmbeddingsEncoder)


# --- validate ------------------------------------------------------------------------


def test_validate_does_not_ask_for_sentence_transformers(tmp_path, monkeypatch):
    real = __import__("importlib.util").util.find_spec

    def find_spec(name, *args):
        return None if name == "sentence_transformers" else real(name, *args)

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    result = ops.validate(_job_file(tmp_path), check_credentials=False)
    assert result.exit_code == 0, [e.message for e in result.errors]


def test_validate_reports_the_embeddings_credential_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("XWALK_TEST_EMBED_KEY", raising=False)
    result = ops.validate(_job_file(tmp_path), scan_records=False)
    assert [e.code for e in result.errors] == ["credential_missing"]
    assert "retrievers.0.api_key_env" in result.errors[0].message
    assert result.data["credentials"]["XWALK_TEST_EMBED_KEY"] == "missing"

    monkeypatch.setenv("XWALK_TEST_EMBED_KEY", "sk-embed")
    result = ops.validate(_job_file(tmp_path), scan_records=False)
    assert result.exit_code == 0
    assert result.data["credentials"]["XWALK_TEST_EMBED_KEY"] == "set"


def test_index_without_the_credential_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.delenv("XWALK_TEST_EMBED_KEY", raising=False)
    with pytest.raises(CredentialMissingError):
        ops.index(_job_file(tmp_path), tmp_path / "idx")
    result = ops.failure_result("index", CredentialMissingError("XWALK_TEST_EMBED_KEY"))
    assert result.exit_code == 2


# --- index, end to end over a mocked gateway -----------------------------------------


class _Gateway:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        data = []
        for i, text in enumerate(body["input"]):
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            data.append({"index": i, "embedding": vector})
        usage = {"prompt_tokens": len(body["input"]), "total_tokens": 0, "cost": 0.5}
        return httpx.Response(200, json={"data": data, "usage": usage})


@pytest.fixture
def gateway(monkeypatch):
    gateway = _Gateway()
    real = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(gateway)
        return real(**kwargs)

    monkeypatch.setattr("xwalk.retrieval.embeddings_api.httpx.Client", client)
    monkeypatch.setenv("XWALK_TEST_EMBED_KEY", "sk-embed")
    return gateway


def test_index_builds_then_reopens_without_embedding(tmp_path, gateway):
    job = _job_file(tmp_path)
    first = ops.index(job, tmp_path / "idx")
    entry = first.data["retrievers"][0]
    assert entry["action"] == "built"
    assert gateway.bodies and all(b["model"] == "vendor/embed-1" for b in gateway.bodies)
    assert entry["embedding_usage"]["texts"] == 5
    assert entry["embedding_usage"]["prompt_tokens"] == 5
    assert math.isclose(entry["embedding_usage"]["cost"], 0.5 * len(gateway.bodies))
    assert any("embedded 5 texts" in line for line in first.lines)

    sent = len(gateway.bodies)
    second = ops.index(job, tmp_path / "idx")
    assert second.data["retrievers"][0]["action"] == "opened"
    assert len(gateway.bodies) == sent
    assert second.data["retrievers"][0]["embedding_usage"]["texts"] == 0


def test_changing_the_provider_preferences_makes_the_index_incompatible(tmp_path, gateway):
    ops.index(_job_file(tmp_path), tmp_path / "idx")
    changed = _job_file(tmp_path, {**API, "provider": {"order": ["other"]}})
    with pytest.raises(ops.OpError) as info:
        ops.index(changed, tmp_path / "idx")
    assert "encoder.provider" in info.value.result.data["differences"]["api"]


def test_search_embeds_the_query_with_the_query_settings(tmp_path, gateway):
    job = _job_file(tmp_path, {**API, "query_input_type": "search_query"})
    ops.index(job, tmp_path / "idx")
    before = len(gateway.bodies)
    result = ops.search(job, "glucose", index_dir=tmp_path / "idx")
    assert result.exit_code == 0
    queries = gateway.bodies[before:]
    assert [b.get("input_type") for b in queries] == ["search_query"]


# --- clustering ----------------------------------------------------------------------


def _cluster_file(tmp_path: Path) -> Path:
    data = yaml.safe_load(CLUSTER.read_text(encoding="utf-8"))
    labels = CLUSTER.parent / data["source"]["path"]
    (tmp_path / labels.name).write_text(labels.read_text(encoding="utf-8"), encoding="utf-8")
    data["dense"] = {k: v for k, v in API.items() if k != "name"}
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_cluster_job_with_an_api_encoder_needs_no_extra(tmp_path, monkeypatch):
    real = __import__("importlib.util").util.find_spec

    def find_spec(name, *args):
        return None if name == "sentence_transformers" else real(name, *args)

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    result = ops.validate(_cluster_file(tmp_path), check_credentials=False)
    assert result.exit_code == 0, [e.message for e in result.errors]


def test_a_cluster_job_reports_the_embeddings_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("XWALK_TEST_CLUSTER_KEY", "sk-llm")
    monkeypatch.delenv("XWALK_TEST_EMBED_KEY", raising=False)
    result = ops.validate(_cluster_file(tmp_path))
    assert [e.code for e in result.errors] == ["credential_missing"]
    assert "dense.api_key_env" in result.errors[0].message

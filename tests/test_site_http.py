"""ner_site.http.handle and the api/*.py Vercel shims (docs/ner-site/PLAN.md 3.1, 7).

Offline: FakeLLM, the offline stand-in and httpx.MockTransport only. BM25 is built in
tmp_path; dense always gets an explicit directory (tmp or a missing one), never the
committed ``ner-site/indexes``.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from ner_site import http as http_mod
from ner_site import indexes as indexes_mod
from ner_site.schema import HealthResponse, NerResponse
from ner_site.service import Settings, make_llm

from xwalk.llm.base import LLMFatalError, LLMRetryableError
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.llm.fake import FakeLLM
from xwalk.retrieval import embeddings_api

SENTINEL_KEY = "sk-or-v1-SENTINEL-c0ffee-4242"
PASSCODE = "correct horse battery"
SECRET_TEXT = "Quokkaline therapy for haloperidol"
SECRET_PROMPT = "PROMPTSENTINEL extract chemicals only"
PROJECT = Path(__file__).resolve().parents[1] / "ner-site"
EXAMPLE_TEXT = (
    "Clinic note 🩺: the patient has Alkaptonuria and was screened for "
    "Hydroxykynureninuria. She took haloperidol, later ketamine, and aspirin for a fever. "
    "Genotyping covered NAT2 and ADA. Her diet: hummus, garlic and cumin."
)
POST_HEADERS = {"content-type": "application/json", "x-requested-with": "xwalk-ner"}

Handled = tuple[int, dict[str, str], bytes]


@pytest.fixture(autouse=True)
def _fresh_memo() -> Iterator[None]:
    indexes_mod.reset()
    http_mod.reset()
    yield
    indexes_mod.reset()
    http_mod.reset()


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One BM25 cache for the module: built once, reused warm (as on Vercel)."""
    return tmp_path_factory.mktemp("bm25-cache")


def _settings(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env(
        {
            "NER_SITE_CACHE_DIR": str(cache_dir),
            "NER_SITE_DENSE_DIR": str(tmp_path / "no-dense"),
            **env,
        }
    )


def _answer(*items: tuple[str, str, int, int]) -> str:
    return json.dumps(
        {"entities": [{"text": t, "type": ty, "start": s, "end": e} for t, ty, s, e in items]}
    )


def _post(
    settings: Settings,
    payload: object,
    *,
    headers: dict[str, str] | None = None,
    llm_factory: Callable[[Settings, str], Any] | None = None,
    raw: bytes | None = None,
) -> Handled:
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return http_mod.handle(
        "POST",
        "/api/ner",
        dict(POST_HEADERS if headers is None else headers),
        body,
        settings=settings,
        llm_factory=llm_factory,
    )


def _json(result: Handled) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result[2])
    return data


def _error(result: Handled) -> tuple[int, str]:
    data = _json(result)
    assert data["ok"] is False and set(data) == {"ok", "error"}
    assert set(data["error"]) == {"code", "message"} and data["error"]["message"]
    return result[0], data["error"]["code"]


def _fake(*script: Any) -> Callable[[Settings, str], FakeLLM]:
    llm = FakeLLM(list(script))
    return lambda settings, model: llm


def _never(settings: Settings, model: str) -> Any:
    pytest.fail("the LLM was created for a request that should have been refused")


def _assert_common_headers(headers: dict[str, str]) -> None:
    lowered = {k.lower(): v for k, v in headers.items()}
    assert lowered["cache-control"] == "no-store"
    assert lowered["content-type"].startswith("application/json")
    assert not any(k.startswith("access-control-") for k in lowered)


# --- health -------------------------------------------------------------------------


def test_health_shape(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    status, headers, body = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert status == 200
    _assert_common_headers(headers)
    health = HealthResponse.model_validate_json(body)
    assert health.mode == "live" and health.default_model == "openai/gpt-4o-mini"
    assert health.models == list(s.models)
    assert health.limits == {
        "max_chars": 5000,
        "max_prompt_chars": 4000,
        "max_entities": 25,
        "top_k_max": 10,
    }
    assert health.passcode_required is False
    assert "chemical" in health.default_prompt
    assert [o.slug for o in health.ontologies] == [
        "chebi-sample",
        "ctd-disease-sample",
        "ncbi-gene-sample",
        "foodon-sample",
    ]
    assert all(o.count == 200 for o in health.ontologies)
    assert health.retrieval["bm25"].status == "ok"
    dense = health.retrieval["dense"]
    assert dense.status == "unavailable" and dense.reason and "no index at" in dense.reason
    assert _json((status, headers, body))["llm"] == {"status": "ok"}
    assert SENTINEL_KEY.encode() not in body


def test_health_without_a_key_still_answers_and_says_what_is_missing(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _settings(cache_dir, tmp_path)
    result = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert result[0] == 200
    data = _json(result)
    assert data["mode"] == "live"
    assert data["llm"]["status"] == "unavailable"
    assert "OPENROUTER_API_KEY" in data["llm"]["reason"]


def test_health_offline_reports_the_offline_model(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, NER_SITE_OFFLINE="1")
    health = HealthResponse.model_validate_json(
        http_mod.handle("GET", "/api/health", {}, b"", settings=s)[2]
    )
    assert health.mode == "offline"
    assert health.default_model == "offline" and health.models == ["offline"]


def test_health_offline_with_committed_indexes_and_no_key_is_dense_unavailable(
    cache_dir: Path, tmp_path: Path
) -> None:
    """A dense index exists (a fake one here), but no key: never `ok`, and the reason
    names the key."""
    index = tmp_path / "dense" / "chebi-sample" / "dense"
    index.mkdir(parents=True)
    (index / "xwalk_meta.json").write_text("{}")
    s = _settings(
        cache_dir, tmp_path, NER_SITE_OFFLINE="1", NER_SITE_DENSE_DIR=str(tmp_path / "dense")
    )
    dense = _json(http_mod.handle("GET", "/api/health", {}, b"", settings=s))["retrieval"]["dense"]
    assert dense["status"] == "unavailable" and "OPENROUTER_API_KEY is not set" in dense["reason"]


def test_health_reports_the_passcode_requirement_without_needing_it(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    result = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert result[0] == 200 and _json(result)["passcode_required"] is True
    assert PASSCODE.encode() not in result[2]


def test_head_on_health_is_allowed(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path)
    assert http_mod.handle("HEAD", "/api/health", {}, b"", settings=s)[0] == 200


# --- routing, methods, headers ------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/nope", "/api", "/", "/api/ner/extra"])
def test_unknown_path_is_404(cache_dir: Path, tmp_path: Path, path: str) -> None:
    s = _settings(cache_dir, tmp_path)
    result = http_mod.handle("GET", path, {}, b"", settings=s)
    assert _error(result) == (404, "not_found")
    _assert_common_headers(result[1])


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/ner"),
        ("PUT", "/api/ner"),
        ("DELETE", "/api/ner"),
        ("POST", "/api/health"),
        ("OPTIONS", "/api/ner"),
        ("OPTIONS", "/api/health"),
    ],
)
def test_wrong_method_is_405(cache_dir: Path, tmp_path: Path, method: str, path: str) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = http_mod.handle(
        method,
        path,
        {"origin": "https://evil.example", "access-control-request-method": "POST"},
        b"",
        settings=s,
        llm_factory=_never,
    )
    assert _error(result) == (405, "method_not_allowed")
    _assert_common_headers(result[1])
    assert result[1]["Allow"] in {"POST", "GET, HEAD"}


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json"},
        {"content-type": "application/json", "x-requested-with": "XMLHttpRequest"},
        {"content-type": "application/json", "x-requested-with": ""},
    ],
)
def test_the_requested_with_header_is_required(
    cache_dir: Path, tmp_path: Path, headers: dict[str, str]
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, headers=headers, llm_factory=_never)
    assert _error(result) == (400, "bad_header")


def test_header_names_are_case_insensitive(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(
        s,
        {"text": "nothing"},
        headers={"Content-Type": "application/json", "X-Requested-With": "xwalk-ner"},
        llm_factory=_fake('{"entities": []}'),
    )
    assert result[0] == 200


# --- passcode -----------------------------------------------------------------------


def test_passcode_required_and_invalid(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    assert _error(_post(s, {"text": "a"}, llm_factory=_never)) == (401, "passcode_required")
    empty = {**POST_HEADERS, "x-passcode": ""}
    assert _error(_post(s, {"text": "a"}, headers=empty, llm_factory=_never)) == (
        401,
        "passcode_required",
    )
    wrong = {**POST_HEADERS, "x-passcode": "correct horse"}
    result = _post(s, {"text": "a"}, headers=wrong, llm_factory=_never)
    assert _error(result) == (401, "passcode_invalid")
    assert PASSCODE.encode() not in result[2]
    right = {**POST_HEADERS, "x-passcode": PASSCODE}
    ok = _post(s, {"text": "nothing"}, headers=right, llm_factory=_fake('{"entities": []}'))
    assert ok[0] == 200


def test_passcode_is_compared_in_constant_time(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = http_mod.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(http_mod.hmac, "compare_digest", spy)
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE="pässe")
    headers = {**POST_HEADERS, "x-passcode": "pässe"}
    assert _post(s, {"text": "x"}, headers=headers, llm_factory=_fake('{"entities": []}'))[0] == 200
    assert seen == [("pässe".encode(), "pässe".encode())]


# --- request validation codes -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"{not json", "bad_json"),
        (b"", "bad_json"),
        (b"\xff\xfe{}", "bad_json"),
        (b"[1, 2]", "bad_json"),
        (b'"text"', "bad_json"),
        (b'{"text": "a", "extra": 1}', "bad_json"),
        (b'{"text": "a", "prompt": 5}', "bad_json"),
        (b'{"text": "a", "model": 5}', "bad_json"),
        (b"{}", "missing_text"),
        (b'{"text": null}', "missing_text"),
        (b'{"text": 42}', "missing_text"),
        (b'{"text": ""}', "missing_text"),
        (b'{"text": "  \\n "}', "missing_text"),
        (b'{"text": "a", "top_k": 0}', "bad_top_k"),
        (b'{"text": "a", "top_k": 11}', "bad_top_k"),
        (b'{"text": "a", "top_k": "5"}', "bad_top_k"),
        (b'{"text": "a", "top_k": true}', "bad_top_k"),
        (b'{"text": "a", "top_k": 2.5}', "bad_top_k"),
        (b'{"text": "a", "model": "anthropic/claude-opus"}', "unknown_model"),
    ],
)
def test_bad_requests_are_400_with_their_code(
    cache_dir: Path, tmp_path: Path, raw: bytes, code: str
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, None, raw=raw, llm_factory=_never)
    assert _error(result) == (400, code)
    _assert_common_headers(result[1])


def test_text_and_prompt_limits(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    long_text = _post(s, {"text": "a" * 5001}, llm_factory=_never)
    assert _error(long_text) == (400, "text_too_long")
    assert _json(long_text)["error"]["message"] == "text is 5001 characters; the limit is 5000"
    long_prompt = _post(s, {"text": "a", "prompt": "p" * 4001}, llm_factory=_never)
    assert _error(long_prompt) == (400, "prompt_too_long")


def test_body_over_64_kb_is_413(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    raw = json.dumps({"text": "a" * 70000}).encode()
    assert _error(_post(s, None, raw=raw, llm_factory=_never)) == (413, "body_too_large")


def test_no_key_and_not_offline_is_503(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path)
    result = _post(s, {"text": "aspirin"}, llm_factory=_never)
    assert _error(result) == (503, "not_configured")
    assert "OPENROUTER_API_KEY" in _json(result)["error"]["message"]


# --- errors from the extraction and the provider ------------------------------------


def test_unparseable_answer_is_422(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, llm_factory=_fake("Sorry, here is prose."))
    assert _error(result) == (422, "extraction_unparseable")


def test_answer_without_entities_is_422(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, llm_factory=_fake('{"items": []}'))
    assert _error(result) == (422, "extraction_schema")


def test_exhausted_call_budget_is_429(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    budgeted = BudgetedLLM(FakeLLM(['{"entities": []}']), CallBudget(limit=0))
    result = _post(s, {"text": "abc"}, llm_factory=lambda settings, model: budgeted)
    assert _error(result) == (429, "call_budget_exhausted")


@pytest.mark.parametrize(
    "failure",
    [
        LLMFatalError(f"HTTP 401: No auth credentials for {SENTINEL_KEY}"),
        LLMRetryableError("exhausted 2 retries: ConnectError at https://u:hunter2@gw.example/v1"),
    ],
)
def test_provider_failure_is_502_without_secrets(
    cache_dir: Path, tmp_path: Path, failure: Exception
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, llm_factory=_fake(failure))
    assert _error(result) == (502, "provider_error")
    assert SENTINEL_KEY.encode() not in result[2] and b"hunter2" not in result[2]


def test_a_real_client_401_through_mock_transport_is_502_without_the_key(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def chat(request: httpx.Request) -> httpx.Response:
        echoed = request.headers["authorization"]
        return httpx.Response(401, json={"error": {"message": f"bad credentials: {echoed}"}})

    s = _settings(
        cache_dir,
        tmp_path,
        OPENROUTER_API_KEY=SENTINEL_KEY,
        NER_SITE_BASE_URL="https://user:hunter2@gw.example/v1",
    )
    result = _post(
        s,
        {"text": "abc"},
        llm_factory=lambda st, m: make_llm(st, m, transport=httpx.MockTransport(chat)),
    )
    assert _error(result) == (502, "provider_error")
    assert SENTINEL_KEY.encode() not in result[2] and b"hunter2" not in result[2]
    assert SENTINEL_KEY not in capsys.readouterr().err


def test_an_unexpected_exception_is_a_generic_500(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(settings: Settings, model: str) -> Any:
        raise RuntimeError(f"kaboom {SENTINEL_KEY} {SECRET_TEXT}")

    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": SECRET_TEXT}, llm_factory=boom)
    assert _error(result) == (500, "internal")
    assert b"kaboom" not in result[2] and SENTINEL_KEY.encode() not in result[2]
    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert SENTINEL_KEY not in err and SECRET_TEXT not in err and "kaboom" not in err


# --- full POSTs ---------------------------------------------------------------------


def test_full_post_offline_through_real_bm25(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, NER_SITE_OFFLINE="1")
    result = _post(s, {"text": EXAMPLE_TEXT, "prompt": None, "model": "offline", "top_k": 3})
    assert result[0] == 200
    _assert_common_headers(result[1])
    out = NerResponse.model_validate_json(result[2])
    assert out.mode == "offline" and out.model == "offline"
    by_text = {e.text: e for e in out.entities}
    for e in out.entities:
        assert EXAMPLE_TEXT[e.start : e.end] == e.text  # code-point offsets
    assert by_text["haloperidol"].matches[0].id == "CHEBI:5613"
    assert by_text["Alkaptonuria"].matches[0].id == "MESH:D000474"
    assert by_text["NAT2"].ontology == "ncbi-gene-sample"
    assert by_text["fever"].ontology is None and by_text["fever"].matches == []
    assert all(len(e.matches) <= 3 for e in out.entities)
    assert out.retrieval["bm25"].status == "ok"
    dense = out.retrieval["dense"]
    assert dense.status == "unavailable" and dense.reason
    assert any("dense retrieval unavailable" in w for w in out.warnings)
    assert out.usage.llm_calls == 0 and out.usage.prompt_tokens is None
    assert out.usage.embedding_requests == 0


def test_full_post_live_through_mock_transport(cache_dir: Path, tmp_path: Path) -> None:
    text = "She took haloperidol and ketamine."
    bodies: list[dict[str, Any]] = []

    def chat(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        answer = _answer(("haloperidol", "chemical", 9, 20), ("ketamine", "chemical", 0, 8))
        return httpx.Response(
            200,
            json={
                "model": "google/gemini-2.5-flash-lite",
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 412, "completion_tokens": 88},
            },
        )

    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(
        s,
        {"text": text, "prompt": SECRET_PROMPT, "model": "google/gemini-2.5-flash-lite"},
        llm_factory=lambda st, m: make_llm(st, m, transport=httpx.MockTransport(chat)),
    )
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    assert out.model == "google/gemini-2.5-flash-lite" and out.mode == "live"
    assert [(e.text, e.start, e.end) for e in out.entities] == [
        ("haloperidol", 9, 20),
        ("ketamine", 25, 33),
    ]
    assert out.usage.model_dump() == {
        "llm_calls": 1,
        "prompt_tokens": 412,
        "completion_tokens": 88,
        "embedding_requests": 0,
    }
    assert bodies[0]["model"] == "google/gemini-2.5-flash-lite"
    assert SECRET_PROMPT in bodies[0]["messages"][-1]["content"]
    assert SENTINEL_KEY.encode() not in result[2]


def _histogram_gateway(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        data = []
        for i, text in enumerate(body["input"]):
            vector = [0.0] * embeddings_api.DEFAULT_DIMENSIONS
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            vector[26] = 0.01  # never an all-zero vector
            data.append({"index": i, "embedding": vector})
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 1}})

    return gateway


def _build_tool() -> ModuleType:
    tool = PROJECT / "tools" / "build_indexes.py"
    spec = importlib.util.spec_from_file_location("ner_site_build_indexes_http", tool)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_post_with_dense_through_the_default_encoder(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dense index is built with the site's own default encoder (Qwen3 settings),
    its HTTP calls answered by a 26-letter histogram gateway, then opened by
    `get_indexes` exactly as on Vercel."""
    seen: list[httpx.Request] = []
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(_histogram_gateway(seen))
        return real_client(**kwargs)

    monkeypatch.setattr("xwalk.retrieval.embeddings_api.httpx.Client", client)
    dense_dir = tmp_path / "indexes"
    encoder = embeddings_api.default_embeddings_encoder(api_key=SENTINEL_KEY)
    _build_tool().build_all(dense_dir, encoder)
    built = len(seen)
    assert built > 0

    s = _settings(
        cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_DENSE_DIR=str(dense_dir)
    )
    health = _json(http_mod.handle("GET", "/api/health", {}, b"", settings=s))
    assert health["retrieval"] == {"bm25": {"status": "ok"}, "dense": {"status": "ok"}}

    text = "haloperidol and Alkaptonuria, and fever"
    answer = _answer(
        ("haloperidol", "chemical", 0, 11),
        ("Alkaptonuria", "disease", 16, 28),
        ("fever", "other", 34, 39),
    )
    result = _post(s, {"text": text, "top_k": 5}, llm_factory=_fake(answer))
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    assert out.retrieval["dense"].status == "ok"
    halo = out.entities[0]
    # The histogram also embeds the Qwen3 query instruction, so dense ranks are noise;
    # what matters is that both retrievers ran and were fused.
    by_id = {m.id: m for m in halo.matches}
    assert by_id["CHEBI:5613"].retrievers["bm25"] == 1
    assert any("dense-qwen3" in m.retrievers for m in halo.matches)
    assert all(m.id.startswith("CHEBI:") for m in halo.matches)
    assert out.usage.embedding_requests == 2  # one per routed entity, none for "other"
    assert not any("dense" in w for w in out.warnings)
    query_calls = seen[built:]
    assert len(query_calls) == 2
    assert all(r.headers["authorization"] == f"Bearer {SENTINEL_KEY}" for r in query_calls)
    assert SENTINEL_KEY.encode() not in result[2]
    assert SENTINEL_KEY not in capsys.readouterr().err


# --- logging ------------------------------------------------------------------------


def test_one_log_line_per_request_without_text_prompt_or_key(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    capsys.readouterr()
    headers = {**POST_HEADERS, "x-passcode": PASSCODE}
    answer = _answer(("haloperidol", "chemical", 23, 34))
    ok = _post(
        s,
        {"text": SECRET_TEXT, "prompt": SECRET_PROMPT, "top_k": 2},
        headers=headers,
        llm_factory=_fake(answer),
    )
    assert ok[0] == 200
    bad = _post(s, {"text": SECRET_TEXT}, headers={**headers, "x-passcode": "nope"})
    assert bad[0] == 401
    http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    first = records[0]
    assert first["status"] == 200 and first["path"] == "/api/ner"
    assert first["model"] == "openai/gpt-4o-mini" and first["mode"] == "live"
    assert first["text_chars"] == len(SECRET_TEXT) and first["custom_prompt"] is True
    assert first["entities"] == 1 and first["llm_calls"] == 1
    assert {"extract", "map", "total"} <= set(first["timing_ms"])
    assert records[1]["status"] == 401 and records[1]["code"] == "passcode_invalid"
    assert records[2]["path"] == "/api/health" and records[2]["status"] == 200
    for line in lines:
        for secret in (SENTINEL_KEY, PASSCODE, SECRET_TEXT, SECRET_PROMPT, "Quokka", "PROMPTSENT"):
            assert secret not in line


# --- api/*.py shims through a real BaseHTTPRequestHandler on a socket --------------


def _shim(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ner_site_api_{name}", PROJECT / "api" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["ner.py", "health.py"])
def shim_server(
    request: pytest.FixtureRequest,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[int]:
    for name in ("OPENROUTER_API_KEY", "NER_SITE_PASSCODE", "NER_SITE_MODELS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NER_SITE_OFFLINE", "1")
    monkeypatch.setenv("NER_SITE_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("NER_SITE_DENSE_DIR", str(tmp_path / "no-dense"))
    module = _shim(request.param)
    handler = module.handler
    assert handler.__name__ == "handler"  # Vercel looks up exactly this name
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _request(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        conn.close()


def test_shim_serves_health_and_ner(shim_server: int, capsys: pytest.CaptureFixture[str]) -> None:
    status, headers, body = _request(shim_server, "GET", "/api/health?x=1")
    assert status == 200 and headers["cache-control"] == "no-store"
    assert not any(k.startswith("access-control-") for k in headers)
    assert HealthResponse.model_validate_json(body).mode == "offline"
    payload = json.dumps({"text": EXAMPLE_TEXT, "model": "offline"}).encode()
    status, headers, body = _request(shim_server, "POST", "/api/ner", payload, {**POST_HEADERS})
    assert status == 200, body
    assert int(headers["content-length"]) == len(body)
    assert any(e["text"] == "haloperidol" for e in json.loads(body)["entities"])
    status, _, body = _request(shim_server, "HEAD", "/api/health")
    assert status == 200 and body == b""
    status, headers, _ = _request(shim_server, "OPTIONS", "/api/ner")
    assert status == 405 and not any(k.startswith("access-control-") for k in headers)
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 4  # one line per request, no access-log lines on top
    assert all(json.loads(line)["status"] in {200, 405} for line in err)


def _raw_exchange(port: int, request: bytes) -> tuple[bytes, bytes]:
    """Send `request` as is and read until the server closes; (status line, body)."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    data = b""
    try:
        sock.sendall(request)
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()
    head, _, body = data.partition(b"\r\n\r\n")
    return head.split(b"\r\n")[0], body


def test_shim_refuses_a_large_body_before_reading_it(shim_server: int) -> None:
    """Content-Length says 10 MB and no byte of it is sent: the answer comes anyway, so
    the shim did not try to read the body."""
    status_line, body = _raw_exchange(
        shim_server,
        b"POST /api/ner HTTP/1.1\r\nHost: x\r\nX-Requested-With: xwalk-ner\r\n"
        b"Content-Type: application/json\r\nContent-Length: 10000000\r\n\r\n",
    )
    assert b" 413 " in status_line
    assert json.loads(body)["error"]["code"] == "body_too_large"


@pytest.mark.parametrize(
    ("length_header", "status", "code"),
    [(b"", b" 411 ", "length_required"), (b"Content-Length: -3\r\n", b" 400 ", "bad_header")],
)
def test_shim_needs_a_valid_content_length_on_post(
    shim_server: int, length_header: bytes, status: bytes, code: str
) -> None:
    status_line, body = _raw_exchange(
        shim_server,
        b"POST /api/ner HTTP/1.1\r\nHost: x\r\nX-Requested-With: xwalk-ner\r\n"
        + length_header
        + b"Connection: close\r\n\r\n",
    )
    assert status in status_line
    assert json.loads(body)["error"]["code"] == code


def test_shim_with_a_bad_setting_answers_503_not_a_traceback(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NER_SITE_MAX_CHARS", "lots")
    monkeypatch.setenv("NER_SITE_CACHE_DIR", str(cache_dir))
    module = _shim("health.py")
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _request(server.server_address[1], "GET", "/api/health")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 503
    error = json.loads(body)["error"]
    assert error["code"] == "not_configured" and "NER_SITE_MAX_CHARS" in error["message"]
    assert "Traceback" not in error["message"]


def test_a_short_passcode_does_not_corrupt_responses(cache_dir: Path, tmp_path: Path) -> None:
    """The secret scrubber must not rewrite ordinary JSON ("ok", "true") in a body."""
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE="ok")
    health = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert _json(health)["ok"] is True
    result = _post(
        s,
        {"text": "nothing"},
        headers={**POST_HEADERS, "x-passcode": "ok"},
        llm_factory=_fake('{"entities": []}'),
    )
    assert result[0] == 200 and _json(result)["ok"] is True

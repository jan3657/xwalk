"""ner_site.service and ner_site.fake (docs/ner-site/PLAN.md sections 3.2, 3.3, 7).

Offline: FakeLLM, an offline stand-in, and httpx.MockTransport; BM25 is built in
tmp_path and dense is always given an explicit directory (never the committed one).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from ner_site import fake, service
from ner_site import indexes as indexes_mod
from ner_site.extract import DEFAULT_PROMPT, ExtractionError, extract
from ner_site.schema import Limits, NerRequest, NerResponse, ServiceError
from ner_site.service import Settings, SettingsError, make_llm, run, validate_request

from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
)
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.records import Usage

SENTINEL_KEY = "sk-or-v1-SENTINEL-c0ffee-4242"
PROJECT = Path(__file__).resolve().parents[1] / "ner-site"
# public/app.js EXAMPLE_TEXT, verbatim.
EXAMPLE_TEXT = (
    "Clinic note 🩺: the patient has Alkaptonuria and was screened for "
    "Hydroxykynureninuria. She took haloperidol, later ketamine, and aspirin for a fever. "
    "Genotyping covered NAT2 and ADA. Her diet: hummus, garlic and cumin."
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    env = {
        "NER_SITE_CACHE_DIR": str(tmp_path / "cache"),
        "NER_SITE_DENSE_DIR": str(tmp_path / "no-dense"),
        **overrides,
    }
    return Settings.from_env(env)


@pytest.fixture(autouse=True)
def _fresh_memo() -> Any:
    indexes_mod.reset()
    yield
    indexes_mod.reset()


@pytest.fixture(scope="module")
def bm25_only(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return indexes_mod.open_indexes(
        cache_dir=tmp_path_factory.mktemp("bm25"), dense_dir=None, api_key=None
    )


def _answer(*items: tuple[str, str, int, int]) -> str:
    return json.dumps(
        {"entities": [{"text": t, "type": ty, "start": s, "end": e} for t, ty, s, e in items]}
    )


# --- Settings.from_env ------------------------------------------------------------


def test_from_env_defaults() -> None:
    s = Settings.from_env({})
    assert s.api_key is None and s.passcode is None and s.offline is False
    assert s.base_url == "https://openrouter.ai/api/v1"
    assert s.models == (
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash-lite",
        "qwen/qwen3-next-80b-a3b-instruct",
    )
    assert s.default_model == "openai/gpt-4o-mini" and s.offered_models == s.models
    assert s.limits == Limits()
    assert s.dense_dir == PROJECT / "indexes"
    assert s.cache_dir == Path("/tmp/xwalk-ner-site")
    assert s.mode == "live"


def test_from_env_reads_every_variable(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "OPENROUTER_API_KEY": f"  {SENTINEL_KEY} ",
            "NER_SITE_BASE_URL": "https://gateway.example/v1/",
            "NER_SITE_MODELS": " a/one , b/two,,a/one ",
            "NER_SITE_PASSCODE": "open sesame",
            "NER_SITE_MAX_CHARS": "1200",
            "NER_SITE_MAX_ENTITIES": "7",
            "NER_SITE_MAX_PROMPT_CHARS": "300",
            "NER_SITE_DENSE_DIR": str(tmp_path / "d"),
            "NER_SITE_CACHE_DIR": str(tmp_path / "c"),
        }
    )
    assert s.api_key == SENTINEL_KEY
    assert s.base_url == "https://gateway.example/v1"
    assert s.models == ("a/one", "b/two") and s.default_model == "a/one"
    assert s.passcode == "open sesame"
    assert (s.limits.max_chars, s.limits.max_entities, s.limits.max_prompt_chars) == (1200, 7, 300)
    assert s.dense_dir == tmp_path / "d" and s.cache_dir == tmp_path / "c"


def test_empty_values_mean_unset() -> None:
    s = Settings.from_env(
        {"OPENROUTER_API_KEY": " ", "NER_SITE_PASSCODE": "", "NER_SITE_MAX_CHARS": ""}
    )
    assert s.api_key is None and s.passcode is None and s.limits.max_chars == 5000


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_offline_mode(value: str) -> None:
    s = Settings.from_env({"NER_SITE_OFFLINE": value})
    assert s.offline and s.mode == "offline"
    assert s.default_model == "offline" and s.offered_models == ("offline",)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NER_SITE_MAX_CHARS", "lots"),
        ("NER_SITE_MAX_CHARS", "0"),
        ("NER_SITE_MAX_CHARS", "-5"),
        ("NER_SITE_MAX_CHARS", "1000000"),
        ("NER_SITE_MAX_CHARS", "5.5"),
        ("NER_SITE_MAX_ENTITIES", "0"),
        ("NER_SITE_MAX_ENTITIES", "101"),
        ("NER_SITE_MAX_PROMPT_CHARS", "99999"),
        ("NER_SITE_OFFLINE", "maybe"),
        ("NER_SITE_MODELS", " , "),
        ("NER_SITE_BASE_URL", "ftp://example/v1"),
        ("NER_SITE_BASE_URL", "not a url"),
    ],
)
def test_bad_values_are_a_settings_error_naming_the_variable(name: str, value: str) -> None:
    with pytest.raises(SettingsError) as exc:
        Settings.from_env({name: value})
    assert name in str(exc.value)


def test_secrets_stay_out_of_repr() -> None:
    s = Settings.from_env({"OPENROUTER_API_KEY": SENTINEL_KEY, "NER_SITE_PASSCODE": "pass-XYZ"})
    assert SENTINEL_KEY not in repr(s) and "pass-XYZ" not in repr(s)


def test_offline_index_view_never_carries_the_key(tmp_path: Path) -> None:
    live = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    offline = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_OFFLINE="1")
    assert live.index_settings().api_key == SENTINEL_KEY
    assert offline.index_settings().api_key is None
    assert offline.index_settings().cache_dir == offline.cache_dir


# --- make_llm -----------------------------------------------------------------------


def test_make_llm_wraps_a_budgeted_openai_client(tmp_path: Path) -> None:
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    llm = make_llm(s, "google/gemini-2.5-flash-lite")
    assert isinstance(llm, BudgetedLLM)
    assert llm.budget.limit == Limits().llm_call_cap == 3
    inner = llm.inner
    assert isinstance(inner, OpenAICompatClient)
    assert inner.model == "google/gemini-2.5-flash-lite"
    assert inner.capabilities.json_schema and inner.capabilities.strict_schema  # "openai"
    asyncio.run(llm.aclose())


def test_make_llm_sends_the_key_headers_and_generation_settings(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def chat(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-4o-mini",
                "choices": [{"message": {"content": '{"entities": []}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    llm = make_llm(s, "openai/gpt-4o-mini", transport=httpx.MockTransport(chat))

    async def go() -> tuple[Any, ...]:
        try:
            return await extract(llm, "nothing here", DEFAULT_PROMPT, max_entities=25)
        finally:
            await llm.aclose()

    entities, _, usage = asyncio.run(go())
    assert entities == [] and usage.prompt_tokens == 50
    (request,) = seen
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SENTINEL_KEY}"
    assert request.headers["x-title"] == "xwalk NER demo"
    assert request.headers["http-referer"].startswith("https://")
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["temperature"] == 0 and body["max_tokens"] == 2048
    assert body["response_format"]["type"] == "json_schema"


def test_make_llm_retries_at_most_twice(tmp_path: Path) -> None:
    calls: list[int] = []

    def down(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    llm = make_llm(s, s.default_model, transport=httpx.MockTransport(down), backoff_base=0.0)
    with pytest.raises(Exception, match="exhausted 2 retries"):
        asyncio.run(llm.complete(LLMRequest(system="s", user="u")))
    assert len(calls) == 3


# --- validate_request ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_", "code"),
    [
        (NerRequest(text=""), "missing_text"),
        (NerRequest(text="   \n"), "missing_text"),
        (NerRequest(text="x" * 5001), "text_too_long"),
        (NerRequest(text="ok", prompt="p" * 4001), "prompt_too_long"),
        (NerRequest(text="ok", top_k=0), "bad_top_k"),
        (NerRequest(text="ok", top_k=11), "bad_top_k"),
        (NerRequest(text="ok", model="evil/model"), "unknown_model"),
        (NerRequest(text="ok", model="offline"), "unknown_model"),
    ],
)
def test_validate_request_codes(tmp_path: Path, request_: NerRequest, code: str) -> None:
    with pytest.raises(ServiceError) as exc:
        validate_request(request_, _settings(tmp_path))
    assert exc.value.code == code and exc.value.status == 400


def test_validate_request_resolves_defaults(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    got = validate_request(NerRequest(text="x" * 5000, prompt="  ", top_k=10), s)
    assert got.prompt == DEFAULT_PROMPT and got.model == s.default_model and got.top_k == 10
    custom = validate_request(NerRequest(text="ok", prompt="only genes", model=s.models[1]), s)
    assert custom.prompt == "only genes" and custom.model == s.models[1]


def test_text_length_counts_code_points(tmp_path: Path) -> None:
    s = _settings(tmp_path, NER_SITE_MAX_CHARS="3")
    validate_request(NerRequest(text="🩺🩺🩺"), s)  # 3 code points, 6 UTF-16 units
    with pytest.raises(ServiceError, match="text is 4 characters; the limit is 3"):
        validate_request(NerRequest(text="🩺🩺🩺a"), s)


def test_offline_accepts_only_the_offline_model(tmp_path: Path) -> None:
    s = _settings(tmp_path, NER_SITE_OFFLINE="1")
    assert validate_request(NerRequest(text="ok", model="offline"), s).model == "offline"
    with pytest.raises(ServiceError) as exc:
        validate_request(NerRequest(text="ok", model="openai/gpt-4o-mini"), s)
    assert exc.value.code == "unknown_model"


# --- run --------------------------------------------------------------------------


def test_run_end_to_end_with_fake_llm(tmp_path: Path, bm25_only: Any) -> None:
    text = "She took haloperidol for a fever."
    llm = FakeLLM(
        [
            _answer(
                ("haloperidol", "chemical", 0, 3), ("fever", "other", 27, 32), ("x", "gene", 0, 1)
            )
        ]
    )
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    out = asyncio.run(run(NerRequest(text=text, top_k=2), settings=s, llm=llm, indexes=bm25_only))
    assert isinstance(out, NerResponse)
    assert out.model == "openai/gpt-4o-mini" and out.mode == "live"
    first, second = out.entities
    assert (first.text, first.start, first.end) == ("haloperidol", 9, 20)  # relocated
    assert text[first.start : first.end] == first.text
    assert first.ontology == "chebi-sample" and first.matches[0].id == "CHEBI:5613"
    assert len(first.matches) <= 2 and first.matches[0].retrievers == {"bm25": 1}
    assert second.ontology is None and second.notes == ["no ontology for type 'other'"]
    assert "entity 'x' dropped: not found in text" in out.warnings
    assert any("dense retrieval unavailable" in w for w in out.warnings)
    assert out.retrieval["dense"].status == "unavailable"
    assert out.usage.model_dump() == {
        "llm_calls": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "embedding_requests": 0,
    }
    assert set(out.timing_ms) == {"extract", "map", "total"}
    assert llm.requests[0].user.endswith(f"TEXT:\n{text}")
    assert DEFAULT_PROMPT in llm.requests[0].user


class _UnreportedLLM:
    """Answers once and reports no token counts (as a provider omitting `usage`)."""

    model = "unreported"
    fingerprint = "unreported"
    capabilities = LLMCapabilities()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text='{"entities": []}', usage=Usage.unreported(), model=self.model)


def test_unknown_tokens_are_null_not_zero(tmp_path: Path, bm25_only: Any) -> None:
    llm: LLMClient = _UnreportedLLM()
    out = asyncio.run(
        run(NerRequest(text="nothing"), settings=_settings(tmp_path), llm=llm, indexes=bm25_only)
    )
    assert out.usage.llm_calls == 1
    assert out.usage.prompt_tokens is None and out.usage.completion_tokens is None


def test_run_passes_extraction_errors_through(tmp_path: Path, bm25_only: Any) -> None:
    llm = FakeLLM(["I could not find anything, sorry."])
    with pytest.raises(ExtractionError) as exc:
        asyncio.run(
            run(NerRequest(text="abc"), settings=_settings(tmp_path), llm=llm, indexes=bm25_only)
        )
    assert exc.value.code == "extraction_unparseable" and exc.value.status == 422


def test_run_maps_a_provider_failure_to_502_without_secrets(tmp_path: Path, bm25_only: Any) -> None:
    llm = FakeLLM(
        [
            LLMFatalError(
                f"HTTP 401: invalid key {SENTINEL_KEY} at https://user:hunter2@gw.example/v1?key=abc"
            )
        ]
    )
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    with pytest.raises(ServiceError) as exc:
        asyncio.run(run(NerRequest(text="abc"), settings=s, llm=llm, indexes=bm25_only))
    assert exc.value.status == 502 and exc.value.code == "provider_error"
    message = exc.value.message
    assert SENTINEL_KEY not in message and "hunter2" not in message and "abc" not in message
    assert "HTTP 401" in message and "gw.example" in message


def test_run_maps_an_exhausted_budget_to_429(tmp_path: Path, bm25_only: Any) -> None:
    llm = BudgetedLLM(FakeLLM(['{"entities": []}']), CallBudget(limit=0))
    with pytest.raises(ServiceError) as exc:
        asyncio.run(
            run(NerRequest(text="abc"), settings=_settings(tmp_path), llm=llm, indexes=bm25_only)
        )
    assert exc.value.status == 429 and exc.value.code == "call_budget_exhausted"
    assert "--max-calls" not in exc.value.message  # the library's CLI advice is not ours


def test_run_bounds_the_extraction_time(
    tmp_path: Path, bm25_only: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Slow:
        model = "slow"
        fingerprint = "slow"
        capabilities = LLMCapabilities()

        async def complete(self, request: LLMRequest) -> LLMResponse:
            await asyncio.sleep(5)
            raise AssertionError("not reached")

    monkeypatch.setattr(service, "EXTRACT_TIMEOUT_S", 0.05)
    with pytest.raises(ServiceError) as exc:
        asyncio.run(
            run(NerRequest(text="abc"), settings=_settings(tmp_path), llm=Slow(), indexes=bm25_only)
        )
    assert exc.value.status == 502 and "did not answer" in exc.value.message


# --- fake.offline_llm --------------------------------------------------------------


def test_offline_llm_is_an_llm_client() -> None:
    llm = fake.offline_llm()
    assert isinstance(llm, LLMClient)
    assert llm.model == "offline"


def test_offline_llm_answers_like_a_model_and_types_by_ontology() -> None:
    llm = fake.offline_llm()
    entities, warnings, usage = asyncio.run(
        extract(llm, EXAMPLE_TEXT, DEFAULT_PROMPT, max_entities=25)
    )
    found = {(e.text, e.type) for e in entities}
    assert {
        ("Alkaptonuria", "disease"),
        ("Hydroxykynureninuria", "disease"),
        ("haloperidol", "chemical"),
        ("ketamine", "chemical"),
        ("aspirin", "chemical"),
        ("fever", "other"),
        ("NAT2", "gene"),
        ("ADA", "gene"),
        ("hummus", "food"),
        ("garlic", "food"),
        ("cumin", "food"),
    } <= found
    for e in entities:
        assert EXAMPLE_TEXT[e.start : e.end] == e.text
    assert not any(e.text.lower() in {"the", "and", "her", "she", "was"} for e in entities)
    assert warnings == []
    assert usage.calls == 0  # a stand-in, not a provider call


def test_offline_llm_is_deterministic_and_returns_model_shaped_json() -> None:
    llm = fake.offline_llm()
    request = LLMRequest(system="s", user=f"INSTRUCTIONS:\nanything\n\nTEXT:\n{EXAMPLE_TEXT}")
    a = asyncio.run(llm.complete(request)).text
    b = asyncio.run(fake.offline_llm().complete(request)).text
    assert a == b
    payload = json.loads(a)
    assert set(payload) == {"entities"}
    for item in payload["entities"]:
        assert set(item) == {"text", "type", "start", "end"}
        assert EXAMPLE_TEXT[item["start"] : item["end"]] == item["text"]


def test_offline_llm_finds_nothing_in_plain_prose() -> None:
    llm = fake.offline_llm()
    request = LLMRequest(system="s", user="INSTRUCTIONS:\nx\n\nTEXT:\nWe met on a rainy day.")
    assert json.loads(asyncio.run(llm.complete(request)).text) == {"entities": []}


def test_one_reason_shared_by_every_ontology_is_reported_once(tmp_path: Path) -> None:
    from ner_site.service import retrieval_status

    index = tmp_path / "dense" / "chebi-sample" / "dense"
    index.mkdir(parents=True)
    (index / "xwalk_meta.json").write_text("{}")
    all_missing_key = indexes_mod.open_indexes(
        cache_dir=tmp_path / "c", dense_dir=tmp_path / "dense", api_key=None
    )
    reason = retrieval_status(all_missing_key, _settings(tmp_path))["dense"].reason
    # chebi has an index but no key; the other three have no index: two distinct reasons.
    assert reason is not None and reason.count("|") == 3
    same = indexes_mod.open_indexes(cache_dir=tmp_path / "c", dense_dir=None, api_key=None)
    once = retrieval_status(same, _settings(tmp_path))["dense"].reason
    assert once == "no dense index directory configured (NER_SITE_DENSE_DIR) (all 4 ontologies)"


# --- review fixes -------------------------------------------------------------------


def test_a_non_ascii_passcode_is_a_settings_error() -> None:
    """http.server reads header values as Latin-1, so a non-ASCII passcode could never
    match what a visitor sends; it is refused at settings load, naming the variable."""
    with pytest.raises(SettingsError) as exc:
        Settings.from_env({"NER_SITE_PASSCODE": "pässwort-geheim"})
    assert "NER_SITE_PASSCODE" in str(exc.value) and "ASCII" in str(exc.value)
    assert "pässwort" not in str(exc.value)
    assert Settings.from_env({"NER_SITE_PASSCODE": "plain-ascii-ok"}).passcode


def test_the_default_prompt_follows_the_configured_entity_limit(tmp_path: Path) -> None:
    s = _settings(tmp_path, NER_SITE_MAX_ENTITIES="7")
    assert "at most 7 entities" in s.default_prompt
    got = validate_request(NerRequest(text="abc"), s)
    assert got.prompt == s.default_prompt
    assert _settings(tmp_path).default_prompt == DEFAULT_PROMPT


class _LeakyRetriever:
    """Fails with the key, a URL with userinfo and a very long message."""

    name = "dense-qwen3"
    fingerprint = "leaky"
    default_limit = 5

    async def search(self, request: Any) -> Any:
        raise RuntimeError(
            f"HTTP 401: No auth credentials: Bearer {SENTINEL_KEY} "
            f"at https://user:hunter2@emb.example/v1/embeddings " + "x" * 2000
        )


def test_run_scrubs_and_caps_entity_notes_and_warnings(tmp_path: Path, bm25_only: Any) -> None:
    chebi = bm25_only["chebi-sample"]
    leaky = {
        "chebi-sample": indexes_mod.OntologyIndex(
            "chebi-sample", chebi.store, [*chebi.retrievers, _LeakyRetriever()], chebi.status
        )
    }
    llm = FakeLLM([_answer(("haloperidol", "chemical", 0, 11))])
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    out = asyncio.run(run(NerRequest(text="haloperidol"), settings=s, llm=llm, indexes=leaky))
    (entity,) = out.entities
    assert entity.matches and entity.matches[0].id == "CHEBI:5613"  # BM25 still serves
    leaked = [n for n in entity.notes if "No auth credentials" in n]
    assert leaked, entity.notes
    for text in [*entity.notes, *out.warnings]:
        assert SENTINEL_KEY not in text and "hunter2" not in text
        assert len(text) <= service.MAX_ERROR_DETAIL + 40
    assert "emb.example" in leaked[0] and "[redacted]" in leaked[0]

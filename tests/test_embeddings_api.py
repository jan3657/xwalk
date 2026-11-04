"""`OpenAIEmbeddingsEncoder`: an OpenAI-compatible `/embeddings` client as a dense encoder.

Every test runs against `httpx.MockTransport`; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from xwalk.records import Record
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.retrieval.dense import DenseRetriever, Encoder, index_components
from xwalk.retrieval.embeddings_api import EmbeddingAPIError, OpenAIEmbeddingsEncoder
from xwalk.templates import TemplateSet

BASE = "https://gateway.example/api/v1"
TEMPLATES = TemplateSet(
    query="{{ mention }}", context="", doc="{{ label }}", candidate="ID: {{ id }}"
)
RECORDS = [
    Record(id="T1", fields={"label": "glucose"}),
    Record(id="T2", fields={"label": "fructose"}),
    Record(id="T3", fields={"label": "automobile"}),
]


def histogram(text: str, dimension: int = 26) -> list[float]:
    """Unnormalised letter counts, padded: a stand-in model with real structure."""
    vector = [0.0] * dimension
    for ch in text.lower():
        if "a" <= ch <= "z":
            vector[ord(ch) - 97] += 1.0
    return vector


class Gateway:
    """A scripted `/embeddings` endpoint. Records every request body."""

    def __init__(
        self,
        *,
        dimension: int = 26,
        usage: dict[str, Any] | None = None,
        respond: Callable[[dict[str, Any], int], httpx.Response | None] | None = None,
        reverse: bool = False,
    ) -> None:
        self.dimension = dimension
        self.usage = usage if usage is not None else {"prompt_tokens": 3, "total_tokens": 3}
        self.respond = respond
        self.reverse = reverse
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        if self.respond is not None:
            scripted = self.respond(body, len(self.bodies))
            if scripted is not None:
                return scripted
        texts = body["input"]
        data = [
            {"object": "embedding", "index": i, "embedding": histogram(t, self.dimension)}
            for i, t in enumerate(texts)
        ]
        if self.reverse:
            data.reverse()
        payload: dict[str, Any] = {"object": "list", "data": data, "model": body["model"]}
        if self.usage:
            payload["usage"] = self.usage
        return httpx.Response(200, json=payload)


def encoder(gateway: Gateway, **kwargs: Any) -> OpenAIEmbeddingsEncoder:
    settings: dict[str, Any] = {"dimensions": gateway.dimension, "send_dimensions": False}
    settings.update(kwargs)
    return OpenAIEmbeddingsEncoder(
        BASE,
        "vendor/embed-1",
        api_key="sk-test",
        transport=httpx.MockTransport(gateway),
        sleep=lambda seconds: None,
        **settings,
    )


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


# --- the request ---------------------------------------------------------------------


def test_it_satisfies_the_encoder_protocol():
    assert isinstance(encoder(Gateway()), Encoder)


def test_the_request_is_an_openai_embeddings_call():
    gateway = Gateway()
    encoder(gateway).encode(["glucose", "fructose"])
    assert gateway.bodies == [
        {"model": "vendor/embed-1", "input": ["glucose", "fructose"], "encoding_format": "float"}
    ]
    assert gateway.headers[0]["authorization"] == "Bearer sk-test"


def test_no_key_sends_no_authorization_header():
    gateway = Gateway()
    OpenAIEmbeddingsEncoder(
        BASE,
        "m",
        dimensions=26,
        send_dimensions=False,
        transport=httpx.MockTransport(gateway),
    ).encode(["glucose"])
    assert "authorization" not in gateway.headers[0]


def test_dimensions_are_sent_by_default():
    gateway = Gateway()
    encoder(gateway, send_dimensions=True).encode(["glucose"])
    assert gateway.bodies[0]["dimensions"] == 26


def test_prefixes_and_input_types_apply_to_their_own_side():
    gateway = Gateway()
    enc = encoder(
        gateway,
        query_prefix="query: ",
        doc_prefix="passage: ",
        query_input_type="search_query",
        doc_input_type="search_document",
    )
    enc.encode(["glucose"], is_query=False)
    enc.encode(["sugar"], is_query=True)
    assert gateway.bodies[0]["input"] == ["passage: glucose"]
    assert gateway.bodies[0]["input_type"] == "search_document"
    assert gateway.bodies[1]["input"] == ["query: sugar"]
    assert gateway.bodies[1]["input_type"] == "search_query"


def test_provider_preferences_are_passed_through():
    gateway = Gateway()
    encoder(gateway, provider={"order": ["nebius"], "allow_fallbacks": False}).encode(["x"])
    assert gateway.bodies[0]["provider"] == {"order": ["nebius"], "allow_fallbacks": False}


def test_texts_are_sent_in_batches():
    gateway = Gateway()
    vectors = encoder(gateway, batch_size=2).encode(["a", "b", "c", "d", "e"])
    assert [b["input"] for b in gateway.bodies] == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == 5


def test_a_blank_text_is_a_zero_vector_and_is_never_sent():
    """Gateways reject an empty string, which would fail the whole batch."""
    gateway = Gateway()
    vectors = encoder(gateway).encode(["glucose", "", "   "])
    assert gateway.bodies[0]["input"] == ["glucose"]
    assert vectors[1] == [0.0] * 26 and vectors[2] == [0.0] * 26
    assert norm(vectors[0]) == pytest.approx(1.0)


def test_only_blank_texts_make_no_request():
    gateway = Gateway()
    assert encoder(gateway).encode(["", ""]) == [[0.0] * 26, [0.0] * 26]
    assert gateway.bodies == []


# --- the response --------------------------------------------------------------------


def test_vectors_are_returned_in_input_order_whatever_the_response_order():
    gateway = Gateway(reverse=True)
    vectors = encoder(gateway, normalize=False).encode(["glucose", "car"])
    assert vectors[0] == histogram("glucose")
    assert vectors[1] == histogram("car")


def test_vectors_are_unit_normalised_by_default():
    vectors = encoder(Gateway()).encode(["glucose"])
    assert norm(vectors[0]) == pytest.approx(1.0)


def test_normalize_false_returns_the_vectors_as_sent():
    vectors = encoder(Gateway(), normalize=False).encode(["glucose"])
    assert vectors[0] == histogram("glucose")


def test_a_vector_of_the_wrong_length_is_refused():
    gateway = Gateway(dimension=32)
    with pytest.raises(EmbeddingAPIError, match="32-dimensional") as info:
        encoder(gateway, dimensions=26).encode(["glucose"])
    assert not info.value.retryable


def test_truncate_cuts_a_longer_vector_and_renormalises():
    gateway = Gateway(dimension=32)
    vectors = encoder(gateway, dimensions=4, truncate=True).encode(["abcd"])
    assert len(vectors[0]) == 4
    assert norm(vectors[0]) == pytest.approx(1.0)


def test_truncate_does_not_pad_a_shorter_vector():
    gateway = Gateway(dimension=8)
    with pytest.raises(EmbeddingAPIError, match="8-dimensional"):
        encoder(gateway, dimensions=16, truncate=True).encode(["abcd"])


def test_a_response_with_the_wrong_number_of_vectors_is_refused():
    def short(body: dict[str, Any], n: int) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": histogram("a")}], "model": "m"}
        )

    with pytest.raises(EmbeddingAPIError, match="2 texts"):
        encoder(Gateway(respond=short), max_retries=0).encode(["a", "b"])


def test_a_non_numeric_embedding_is_refused():
    def junk(body: dict[str, Any], n: int) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": "AAAA"}]})

    with pytest.raises(EmbeddingAPIError, match="not a list of numbers"):
        encoder(Gateway(respond=junk), max_retries=0).encode(["a"])


def test_a_non_finite_value_is_refused():
    def nan(body: dict[str, Any], n: int) -> httpx.Response:
        content = '{"data": [{"index": 0, "embedding": [NaN, 1.0]}]}'
        return httpx.Response(200, content=content.encode())

    with pytest.raises(EmbeddingAPIError, match="non-finite"):
        encoder(Gateway(respond=nan), dimensions=2, max_retries=0).encode(["a"])


# --- failures ------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_a_retryable_status_is_retried(status):
    def flaky(body: dict[str, Any], n: int) -> httpx.Response | None:
        return httpx.Response(status, json={"error": {"message": "busy"}}) if n == 1 else None

    gateway = Gateway(respond=flaky)
    vectors = encoder(gateway).encode(["glucose"])
    assert len(gateway.bodies) == 2 and len(vectors) == 1


def test_retry_after_is_honoured():
    waits: list[float] = []

    def limited(body: dict[str, Any], n: int) -> httpx.Response | None:
        if n == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {}})
        return None

    gateway = Gateway(respond=limited)
    OpenAIEmbeddingsEncoder(
        BASE,
        "m",
        dimensions=26,
        send_dimensions=False,
        transport=httpx.MockTransport(gateway),
        sleep=waits.append,
    ).encode(["glucose"])
    assert waits == [7.0]


def test_retries_are_bounded_and_the_error_says_so():
    def down(body: dict[str, Any], n: int) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    gateway = Gateway(respond=down)
    with pytest.raises(EmbeddingAPIError, match="overloaded") as info:
        encoder(gateway, max_retries=2).encode(["glucose"])
    assert len(gateway.bodies) == 3
    assert info.value.retryable and info.value.status == 503


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_a_fatal_status_is_not_retried(status):
    def refuse(body: dict[str, Any], n: int) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "no"}})

    gateway = Gateway(respond=refuse)
    with pytest.raises(EmbeddingAPIError) as info:
        encoder(gateway).encode(["glucose"])
    assert len(gateway.bodies) == 1
    assert not info.value.retryable and info.value.status == status


def test_the_error_is_a_retriever_error_so_the_matcher_degrades():
    assert issubclass(EmbeddingAPIError, RetrieverError)


def test_an_error_in_a_200_body_is_detected():
    """OpenRouter reports some upstream failures as HTTP 200 with an `error` object."""

    def body_error(body: dict[str, Any], n: int) -> httpx.Response | None:
        if n == 1:
            return httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}})
        return None

    gateway = Gateway(respond=body_error)
    assert len(encoder(gateway).encode(["glucose"])) == 1
    assert len(gateway.bodies) == 2


def test_a_fatal_code_in_a_200_body_is_not_retried():
    def body_error(body: dict[str, Any], n: int) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": 401, "message": "bad key"}})

    gateway = Gateway(respond=body_error)
    with pytest.raises(EmbeddingAPIError, match="bad key"):
        encoder(gateway).encode(["glucose"])
    assert len(gateway.bodies) == 1


def test_the_upstream_cause_is_unwrapped_from_gateway_metadata():
    raw = json.dumps({"error": {"message": "input_type must be query or document"}})

    def refuse(body: dict[str, Any], n: int) -> httpx.Response:
        error = {"message": "Provider returned error", "metadata": {"raw": raw}}
        return httpx.Response(400, json={"error": error})

    with pytest.raises(EmbeddingAPIError, match="input_type must be query or document"):
        encoder(Gateway(respond=refuse)).encode(["glucose"])


def test_a_non_json_200_is_retried():
    def html(body: dict[str, Any], n: int) -> httpx.Response | None:
        return httpx.Response(200, content=b"<html>oops</html>") if n == 1 else None

    gateway = Gateway(respond=html)
    assert len(encoder(gateway).encode(["glucose"])) == 1


def test_a_transport_error_is_retried():
    calls = {"n": 0}
    gateway = Gateway()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset", request=request)
        return gateway(request)

    enc = OpenAIEmbeddingsEncoder(
        BASE,
        "m",
        dimensions=26,
        send_dimensions=False,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )
    assert len(enc.encode(["glucose"])) == 1 and calls["n"] == 2


def test_payload_too_large_halves_the_batch():
    def size_limited(body: dict[str, Any], n: int) -> httpx.Response | None:
        if len(body["input"]) > 2:
            return httpx.Response(413, json={"error": {"message": "too large"}})
        return None

    gateway = Gateway(respond=size_limited)
    vectors = encoder(gateway, batch_size=8, normalize=False).encode(list("abcde"))
    assert [histogram(c) for c in "abcde"] == vectors
    sent = [b["input"] for b in gateway.bodies if len(b["input"]) <= 2]
    assert [t for batch in sent for t in batch] == list("abcde")


def test_a_single_text_that_is_too_large_is_fatal():
    def size_limited(body: dict[str, Any], n: int) -> httpx.Response:
        return httpx.Response(413, json={"error": {"message": "too large"}})

    with pytest.raises(EmbeddingAPIError, match="too large") as info:
        encoder(Gateway(respond=size_limited)).encode(["a"])
    assert info.value.status == 413


# --- usage ---------------------------------------------------------------------------


def test_usage_counts_reported_tokens_and_cost():
    gateway = Gateway(usage={"prompt_tokens": 5, "total_tokens": 5, "cost": 0.0001})
    enc = encoder(gateway, batch_size=1)
    enc.encode(["a", "b"])
    usage = enc.usage
    assert usage.requests == 2 and usage.texts == 2
    assert usage.prompt_tokens == 10
    assert usage.cost == pytest.approx(0.0002)


def test_unreported_usage_is_unknown_not_zero():
    gateway = Gateway(usage={})
    enc = encoder(gateway)
    enc.encode(["a"])
    assert enc.usage.prompt_tokens is None
    assert enc.usage.cost is None


def test_one_unreported_response_makes_the_total_unknown():
    def mixed(body: dict[str, Any], n: int) -> httpx.Response | None:
        if n == 2:
            data = [{"index": 0, "embedding": histogram(body["input"][0])}]
            return httpx.Response(200, json={"data": data})
        return None

    enc = encoder(Gateway(respond=mixed), batch_size=1)
    enc.encode(["a", "b"])
    assert enc.usage.prompt_tokens is None


def test_cost_is_unknown_when_only_tokens_are_reported():
    enc = encoder(Gateway(usage={"prompt_tokens": 2, "total_tokens": 2}))
    enc.encode(["a"])
    assert enc.usage.prompt_tokens == 2 and enc.usage.cost is None


# --- identity ------------------------------------------------------------------------


def _components(**kwargs: Any) -> dict[str, Any]:
    return index_components(RECORDS, TEMPLATES, encoder(Gateway(), **kwargs))


@pytest.mark.parametrize(
    "change",
    [
        {"dimensions": 13},
        {"send_dimensions": True},
        {"truncate": True},
        {"normalize": False},
        {"query_prefix": "query: "},
        {"doc_prefix": "passage: "},
        {"query_input_type": "search_query"},
        {"doc_input_type": "search_document"},
        {"provider": {"order": ["nebius"]}},
    ],
)
def test_a_setting_that_changes_vectors_changes_the_index_identity(change):
    assert _components(**change) != _components()


@pytest.mark.parametrize(
    "change", [{"batch_size": 3}, {"timeout": 5.0}, {"max_retries": 1}, {"backoff_base": 0.1}]
)
def test_a_setting_that_only_changes_speed_keeps_the_identity(change):
    assert _components(**change) == _components()


def test_the_key_is_not_identity_and_never_appears_in_it():
    one = index_components(RECORDS, TEMPLATES, encoder(Gateway()))
    other = OpenAIEmbeddingsEncoder(
        BASE, "vendor/embed-1", api_key="sk-other", dimensions=26, send_dimensions=False
    )
    assert one == index_components(RECORDS, TEMPLATES, other)
    assert "sk-test" not in json.dumps(one)


def test_the_base_url_is_identity_and_is_redacted():
    leaky = OpenAIEmbeddingsEncoder(
        "https://user:secret@gateway.example/api/v1/",
        "vendor/embed-1",
        dimensions=26,
        send_dimensions=False,
    )
    components = index_components(RECORDS, TEMPLATES, leaky)
    assert "secret" not in json.dumps(components)
    other = OpenAIEmbeddingsEncoder(
        "https://elsewhere.example/v1", "vendor/embed-1", dimensions=26, send_dimensions=False
    )
    assert index_components(RECORDS, TEMPLATES, other) != components


def test_the_identity_says_which_adapter_made_the_vectors():
    """A local model and an API model can share a name; their vectors need not agree."""
    settings = encoder(Gateway()).settings
    assert settings["adapter"] == "openai_embeddings"
    assert settings["revision"] == "unknown"


def test_computing_the_identity_makes_no_request():
    gateway = Gateway()
    index_components(RECORDS, TEMPLATES, encoder(gateway))
    assert gateway.bodies == []


# --- with DenseRetriever -------------------------------------------------------------


def test_a_dense_retriever_over_the_api_ranks_the_closest_record_first(tmp_path):
    retriever = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", encoder(Gateway()))
    hits = asyncio.run(retriever.search(SearchRequest(text="glucose", limit=3)))
    assert hits[0].record_id == "T1"
    assert retriever.name == "dense:vendor/embed-1"


def test_reopening_an_api_index_embeds_nothing(tmp_path):
    gateway = Gateway()
    built = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", encoder(gateway))
    sent = len(gateway.bodies)
    expected = index_components(RECORDS, TEMPLATES, encoder(gateway))
    reopened = DenseRetriever.open(tmp_path / "idx", encoder(gateway), expected=expected)
    assert reopened.fingerprint == built.fingerprint
    assert len(gateway.bodies) == sent


def test_a_failed_query_embedding_surfaces_as_a_search_error(tmp_path):
    def fail_queries(body: dict[str, Any], n: int) -> httpx.Response | None:
        if body.get("input_type") == "q":
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        return None

    enc = encoder(Gateway(respond=fail_queries), query_input_type="q")
    retriever = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", enc)
    with pytest.raises(EmbeddingAPIError):
        asyncio.run(retriever.search(SearchRequest(text="glucose", limit=3)))


def test_probe_dimension_returns_the_native_length_in_one_call():
    gateway = Gateway(dimension=40)
    enc = encoder(gateway, dimensions=1, doc_prefix="passage: ")
    assert enc.probe_dimension() == 40
    assert len(gateway.bodies) == 1
    assert gateway.bodies[0]["input"] == ["passage: dimension probe"]


def test_usage_is_booked_even_when_the_vector_shape_is_refused():
    """The endpoint billed the call; refusing its answer does not unspend it."""
    gateway = Gateway(dimension=32, usage={"prompt_tokens": 4, "total_tokens": 4})
    enc = encoder(gateway, dimensions=26)
    with pytest.raises(EmbeddingAPIError):
        enc.encode(["glucose"])
    assert enc.usage.prompt_tokens == 4


# --- construction --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dimensions": 0}, "dimensions"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_invalid_settings_are_refused(kwargs, message):
    settings: dict[str, Any] = {"dimensions": 8, **kwargs}
    with pytest.raises(ValueError, match=message):
        OpenAIEmbeddingsEncoder(BASE, "m", **settings)


def test_a_blank_model_or_base_url_is_refused():
    with pytest.raises(ValueError, match="model"):
        OpenAIEmbeddingsEncoder(BASE, " ", dimensions=8)
    with pytest.raises(ValueError, match="base_url"):
        OpenAIEmbeddingsEncoder("", "m", dimensions=8)


def test_the_endpoint_is_base_url_plus_embeddings():
    seen: list[str] = []
    gateway = Gateway()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return gateway(request)

    OpenAIEmbeddingsEncoder(
        BASE + "/",
        "m",
        dimensions=26,
        send_dimensions=False,
        transport=httpx.MockTransport(handler),
    ).encode(["a"])
    assert seen == [f"{BASE}/embeddings"]

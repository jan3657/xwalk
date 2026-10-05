import json
import os

import httpx
import pytest

from xwalk.llm.base import LLMCapabilities, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.openai_compat import CAPABILITY_PROFILES, OpenAICompatClient

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def ok_body(content: str = '{"a": "x"}') -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        "model": "test-model",
    }


def client(handler, **kwargs) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="k",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_returns_the_message_content():
    llm = client(lambda r: httpx.Response(200, json=ok_body()))
    assert (await llm.complete(LLMRequest(system="s", user="u"))).text == '{"a": "x"}'


async def test_sends_system_and_user_messages():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    await client(handler).complete(LLMRequest(system="SYS", user="USR"))
    assert seen["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


async def test_sends_the_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_body())

    await client(handler).complete(LLMRequest(system="", user="u"))
    assert seen["auth"] == "Bearer k"


async def test_records_usage():
    llm = client(lambda r: httpx.Response(200, json=ok_body()))
    usage = (await llm.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (11, 3, 1)


async def test_missing_usage_block_is_not_an_error():
    body = ok_body()
    del body["usage"]
    llm = client(lambda r: httpx.Response(200, json=body))
    assert (await llm.complete(LLMRequest(system="", user="u"))).usage.calls == 1


async def test_requests_structured_output_only_when_declared():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    without = client(handler, capabilities=LLMCapabilities(json_schema=False))
    await without.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert "response_format" not in seen[-1]

    with_ = client(handler, capabilities=LLMCapabilities(json_schema=True))
    await with_.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert seen[-1]["response_format"]["type"] == "json_schema"


async def test_strict_flag_follows_the_declared_capability():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    llm = client(handler, capabilities=LLMCapabilities(json_schema=True, strict_schema=True))
    await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert seen[-1]["response_format"]["json_schema"]["strict"] is True


async def test_falls_back_to_prompt_only_json_when_the_provider_rejects_the_format():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(
                400, json={"error": {"message": "response_format is not supported"}}
            )
        return httpx.Response(200, json=ok_body())

    llm = client(handler, capabilities=LLMCapabilities(json_schema=True))
    response = await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert response.text == '{"a": "x"}'
    assert response.structured is False
    assert len(calls) == 2


async def test_the_fallback_is_remembered_for_later_calls():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return httpx.Response(200, json=ok_body())

    llm = client(handler, capabilities=LLMCapabilities(json_schema=True))
    await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert sum("response_format" in c for c in calls) == 1


async def test_retries_on_429_then_succeeds():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=ok_body())

    llm = client(lambda r: handler(r), backoff_base=0.0)
    assert (await llm.complete(LLMRequest(system="", user="u"))).text == '{"a": "x"}'
    assert state["n"] == 2


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
async def test_retryable_statuses_eventually_raise_retryable(status):
    llm = client(lambda r: httpx.Response(status, json={}), max_retries=2, backoff_base=0.0)
    with pytest.raises(LLMRetryableError):
        await llm.complete(LLMRequest(system="", user="u"))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_non_retryable_statuses_raise_fatal_immediately(status):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(LLMFatalError):
        await client(handler, backoff_base=0.0).complete(LLMRequest(system="", user="u"))
    assert calls["n"] == 1


def retry_after_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, headers={"retry-after": "7"}, json={})


async def test_honours_retry_after_when_declared():
    """`max_retries=0` so the header is read but never slept on — the test must not
    actually wait 7 seconds to assert that it parsed 7 seconds."""
    declared = client(
        retry_after_handler,
        capabilities=LLMCapabilities(native_retry_after=True),
        max_retries=0,
        backoff_base=0.0,
    )
    with pytest.raises(LLMRetryableError) as exc:
        await declared.complete(LLMRequest(system="", user="u"))
    assert exc.value.retry_after == 7.0


async def test_ignores_retry_after_when_not_declared():
    undeclared = client(
        retry_after_handler,
        capabilities=LLMCapabilities(native_retry_after=False),
        max_retries=0,
        backoff_base=0.0,
    )
    with pytest.raises(LLMRetryableError) as exc:
        await undeclared.complete(LLMRequest(system="", user="u"))
    assert exc.value.retry_after is None


async def test_a_connection_error_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMRetryableError):
        await client(handler, max_retries=1, backoff_base=0.0).complete(
            LLMRequest(system="", user="u")
        )


async def test_an_empty_choices_array_is_fatal():
    llm = client(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMFatalError, match="no choices"):
        await llm.complete(LLMRequest(system="", user="u"))


def test_profiles_exist_for_the_endpoints_we_document():
    for name in ("openai", "anthropic-compat", "google-compat", "vllm", "unknown"):
        assert name in CAPABILITY_PROFILES


def test_the_unknown_profile_claims_nothing():
    caps = CAPABILITY_PROFILES["unknown"]
    assert not any(
        [
            caps.json_schema,
            caps.strict_schema,
            caps.usage_reporting,
            caps.seed,
            caps.native_retry_after,
        ]
    )


def test_fingerprint_changes_with_the_model():
    a = client(lambda r: httpx.Response(200, json=ok_body()))
    b = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="other-model",
        api_key="k",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    assert a.fingerprint != b.fingerprint


def test_fingerprint_ignores_the_api_key():
    """Rotating a key must not invalidate a resumable run — and must not leak into it."""
    a = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        api_key="key-one",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    b = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        api_key="key-two",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    assert a.fingerprint == b.fingerprint


@pytest.mark.integration
async def test_against_a_real_provider():
    key = os.environ.get("XWALK_TEST_API_KEY")
    base = os.environ.get("XWALK_TEST_BASE_URL")
    model = os.environ.get("XWALK_TEST_MODEL")
    if not (key and base and model):
        pytest.skip("set XWALK_TEST_API_KEY, XWALK_TEST_BASE_URL, XWALK_TEST_MODEL")
    llm = OpenAICompatClient(base_url=base, model=model, api_key=key, profile="openai")
    response = await llm.complete(
        LLMRequest(
            system="Reply with JSON only.",
            user='Return exactly {"a": "x"}',
            schema=SCHEMA,
            max_tokens=64,
        )
    )
    from xwalk.llm.parsing import parse_json_object

    assert parse_json_object(response.text)["a"] == "x"


# --- usage accounting and generation precedence (CONTRACTS.md sections 5 and 6) ---


async def test_missing_usage_block_is_counted_as_unknown_not_zero():
    body = ok_body()
    del body["usage"]
    llm = client(lambda r: httpx.Response(200, json=body))
    usage = (await llm.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.calls, usage.unknown_calls, usage.total_tokens) == (1, 1, 0)


async def test_internal_retries_are_counted_as_dispatched_calls():
    responses = iter(
        [httpx.Response(429, json={"error": "slow"}), httpx.Response(200, json=ok_body())]
    )
    llm = client(lambda r: next(responses), backoff_base=0.0)
    usage = (await llm.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.calls, usage.unknown_calls) == (2, 1)
    assert (usage.prompt_tokens, usage.completion_tokens) == (11, 3)


async def test_exhausted_retries_carry_their_call_count():
    llm = client(
        lambda r: httpx.Response(503, json={"error": "down"}), max_retries=2, backoff_base=0.0
    )
    with pytest.raises(LLMRetryableError) as caught:
        await llm.complete(LLMRequest(system="", user="u"))
    assert caught.value.usage is not None
    assert (caught.value.usage.calls, caught.value.usage.unknown_calls) == (3, 3)


async def test_a_fatal_response_carries_its_call_count():
    llm = client(lambda r: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(LLMFatalError) as caught:
        await llm.complete(LLMRequest(system="", user="u"))
    assert caught.value.usage is not None and caught.value.usage.unknown_calls == 1


async def test_the_request_value_overrides_the_client_value():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    llm = client(handler, temperature=0.5, max_tokens=900)
    await llm.complete(LLMRequest(system="", user="u"))
    await llm.complete(LLMRequest(system="", user="u", temperature=0.0, max_tokens=64))
    assert (seen[0]["temperature"], seen[0]["max_tokens"]) == (0.5, 900)
    assert (seen[1]["temperature"], seen[1]["max_tokens"]) == (0.0, 64)


async def test_a_client_seed_is_sent_when_the_provider_supports_it():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    await client(handler, seed=7, profile="openai").complete(LLMRequest(system="", user="u"))
    await client(handler, seed=7, profile="unknown").complete(LLMRequest(system="", user="u"))
    assert seen[0]["seed"] == 7
    assert "seed" not in seen[1]


def test_the_seed_enters_the_fingerprint_only_when_set():
    def h(r):
        return httpx.Response(200, json=ok_body())

    assert client(h).fingerprint == client(h, seed=None).fingerprint
    assert client(h).fingerprint != client(h, seed=1).fingerprint

import pytest

from xwalk.llm.base import LLMClient, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.litellm import LiteLLMClient
from xwalk.records import Usage


async def _async(value):
    return value


class FakeCompletion:
    def __init__(self, content, *, prompt=10, completion=3):
        self.choices = [
            type(
                "C",
                (),
                {"message": type("M", (), {"content": content})(), "finish_reason": "stop"},
            )()
        ]
        self.usage = type("U", (), {"prompt_tokens": prompt, "completion_tokens": completion})()
        self.model = "fake/model"


async def test_returns_the_content(monkeypatch):
    client = LiteLLMClient("fake/model")
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(FakeCompletion('{"a":1}')))
    response = await client.complete(LLMRequest(system="s", user="u"))
    assert response.text == '{"a":1}'


async def test_reports_usage(monkeypatch):
    client = LiteLLMClient("fake/model")
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(FakeCompletion("x")))
    usage = (await client.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (10, 3, 1)


async def test_carries_the_finish_reason(monkeypatch):
    """A truncated answer is otherwise indistinguishable from a badly-answered one."""
    client = LiteLLMClient("fake/model")
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(FakeCompletion("x")))
    assert (await client.complete(LLMRequest(system="", user="u"))).finish_reason == "stop"


async def test_sends_system_and_user_messages(monkeypatch):
    seen = {}
    client = LiteLLMClient("fake/model")

    def capture(**kwargs):
        seen.update(kwargs)
        return _async(FakeCompletion("x"))

    monkeypatch.setattr(client, "_acompletion", capture)
    await client.complete(LLMRequest(system="SYS", user="USR"))
    assert seen["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


async def test_a_rate_limit_becomes_retryable(monkeypatch):
    client = LiteLLMClient("fake/model")

    def boom(**kwargs):
        raise RuntimeError("RateLimitError: too many requests")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMRetryableError):
        await client.complete(LLMRequest(system="", user="u"))


async def test_an_auth_error_becomes_fatal(monkeypatch):
    client = LiteLLMClient("fake/model")

    def boom(**kwargs):
        raise RuntimeError("AuthenticationError: bad key")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMFatalError):
        await client.complete(LLMRequest(system="", user="u"))


async def _raise_named(monkeypatch, class_name: str, message: str):
    """Raise an exception whose class has LiteLLM's public name (litellm not needed)."""
    client = LiteLLMClient("fake/model")
    error_type = type(class_name, (Exception,), {})

    def boom(**kwargs):
        raise error_type(message)

    monkeypatch.setattr(client, "_acompletion", boom)
    await client.complete(LLMRequest(system="", user="u"))


@pytest.mark.parametrize(
    "class_name",
    [
        # LiteLLM's own "the provider's response was malformed" error: recoverable (R1).
        "APIResponseValidationError",
        # LiteLLM's catch-all for provider statuses it does not map one by one (502...).
        "APIError",
        "RateLimitError",
        "Timeout",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIConnectionError",
    ],
)
async def test_litellm_transient_and_malformed_response_errors_are_retryable(
    monkeypatch, class_name
):
    with pytest.raises(LLMRetryableError):
        await _raise_named(monkeypatch, class_name, "upstream said no")


@pytest.mark.parametrize(
    "class_name",
    [
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
        "UnprocessableEntityError",
        "ContextWindowExceededError",
        "ContentPolicyViolationError",
        "RejectedRequestError",
        "BudgetExceededError",
    ],
)
async def test_litellm_auth_model_and_request_errors_are_fatal(monkeypatch, class_name):
    # Classified by name, so a message that happens to mention a transient word cannot
    # turn a request that will never succeed into a retry.
    with pytest.raises(LLMFatalError):
        await _raise_named(monkeypatch, class_name, "rejected; not a timeout")


async def test_a_transient_class_name_wins_over_a_fatal_sounding_message(monkeypatch):
    with pytest.raises(LLMRetryableError):
        await _raise_named(monkeypatch, "RateLimitError", "model not found; invalid request")


async def test_an_unrecognised_error_is_fatal_not_retried_forever(monkeypatch):
    """Retrying an error we do not understand burns budget on a permanent failure."""
    client = LiteLLMClient("fake/model")

    def boom(**kwargs):
        raise RuntimeError("something entirely novel")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMFatalError):
        await client.complete(LLMRequest(system="", user="u"))


async def test_no_choices_is_a_clear_retryable_error_not_an_index_error(monkeypatch):
    # CONTRACTS.md section 3: malformed output is recoverable, never a run abort.
    client = LiteLLMClient("fake/model")

    class Empty:
        choices: list[object] = []

    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(Empty()))
    with pytest.raises(LLMRetryableError, match="choices") as caught:
        await client.complete(LLMRequest(system="", user="u"))
    assert caught.value.usage == Usage.unreported(1)


def _malformed(kind):
    completion = FakeCompletion("x")
    if kind == "no choices attribute":
        del completion.choices
    elif kind == "choices is a string":
        completion.choices = "x"
    elif kind == "choice without a message":
        completion.choices = [type("C", (), {"finish_reason": "stop"})()]
    elif kind == "message is a string":
        completion.choices = [type("C", (), {"message": "x"})()]
    elif kind == "content is a list":
        completion.choices[0].message.content = [{"type": "text", "text": "x"}]
    elif kind == "response is None":
        return None
    return completion


@pytest.mark.parametrize(
    "kind",
    [
        "no choices attribute",
        "choices is a string",
        "choice without a message",
        "message is a string",
        "content is a list",
        "response is None",
    ],
)
async def test_a_malformed_response_is_retryable_with_unknown_usage(monkeypatch, kind):
    client = LiteLLMClient("fake/model")
    completion = _malformed(kind)
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(completion))
    with pytest.raises(LLMRetryableError) as caught:
        await client.complete(LLMRequest(system="", user="u"))
    assert caught.value.usage == Usage.unreported(1)


@pytest.mark.parametrize("prompt", ["ten", -1, float("nan"), float("inf"), True, [10]])
async def test_unusable_token_counts_are_unknown_and_keep_the_answer(monkeypatch, prompt):
    client = LiteLLMClient("fake/model")
    completion = FakeCompletion("x", prompt=prompt)
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(completion))
    response = await client.complete(LLMRequest(system="", user="u"))
    assert response.text == "x"
    assert response.usage == Usage.unreported(1)


@pytest.mark.parametrize("prompt, completion", [(10, None), (None, 3)])
async def test_a_half_reported_usage_is_unknown_not_a_known_zero(monkeypatch, prompt, completion):
    client = LiteLLMClient("fake/model")
    completion_obj = FakeCompletion("x", prompt=prompt, completion=completion)
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(completion_obj))
    response = await client.complete(LLMRequest(system="", user="u"))
    assert response.text == "x"
    assert response.usage == Usage.unreported(1)


async def test_a_dict_shaped_response_is_read(monkeypatch):
    client = LiteLLMClient("fake/model")
    completion = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "model": "fake/model",
    }
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(completion))
    response = await client.complete(LLMRequest(system="", user="u"))
    assert (response.text, response.finish_reason) == ("x", "stop")
    assert response.usage == Usage(prompt_tokens=4, completion_tokens=2, calls=1)


async def test_a_schema_is_only_sent_when_the_capability_is_declared(monkeypatch):
    """Declared, not trusted: a schema sent to a provider that cannot honour it is an
    error, not a hint."""
    seen = {}
    client = LiteLLMClient("fake/model")

    def capture(**kwargs):
        seen.update(kwargs)
        return _async(FakeCompletion("x"))

    monkeypatch.setattr(client, "_acompletion", capture)
    await client.complete(LLMRequest(system="", user="u", schema={"type": "object"}))
    assert "response_format" not in seen


def test_default_capabilities_claim_nothing():
    caps = LiteLLMClient("fake/model").capabilities
    assert not caps.json_schema and not caps.strict_schema


def test_fingerprint_changes_with_the_model():
    assert LiteLLMClient("a/b").fingerprint != LiteLLMClient("c/d").fingerprint


def test_satisfies_the_llm_client_protocol():
    assert isinstance(LiteLLMClient("fake/model"), LLMClient)


async def test_a_missing_usage_block_is_unknown_not_zero(monkeypatch):
    client = LiteLLMClient("fake/model")
    completion = FakeCompletion("x")
    completion.usage = None
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(completion))
    usage = (await client.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.calls, usage.unknown_calls, usage.total_tokens) == (1, 1, 0)


async def test_a_raised_provider_error_counts_one_unknown_call(monkeypatch):
    client = LiteLLMClient("fake/model")

    async def boom(**kw):
        raise RuntimeError("RateLimitError: slow down")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMRetryableError) as caught:
        await client.complete(LLMRequest(system="", user="u"))
    assert caught.value.usage is not None and caught.value.usage.unknown_calls == 1


async def test_client_generation_values_apply_unless_the_request_overrides(monkeypatch):
    seen = []

    def capture(**kw):
        seen.append(kw)
        return _async(FakeCompletion("x"))

    client = LiteLLMClient("fake/model", temperature=0.4, max_tokens=700)
    monkeypatch.setattr(client, "_acompletion", capture)
    await client.complete(LLMRequest(system="", user="u"))
    await client.complete(LLMRequest(system="", user="u", max_tokens=50))
    assert (seen[0]["temperature"], seen[0]["max_tokens"]) == (0.4, 700)
    assert (seen[1]["temperature"], seen[1]["max_tokens"]) == (0.4, 50)

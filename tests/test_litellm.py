import pytest

from xwalk.llm.base import LLMClient, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.litellm import LiteLLMClient


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


async def test_an_unrecognised_error_is_fatal_not_retried_forever(monkeypatch):
    """Retrying an error we do not understand burns budget on a permanent failure."""
    client = LiteLLMClient("fake/model")

    def boom(**kwargs):
        raise RuntimeError("something entirely novel")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMFatalError):
        await client.complete(LLMRequest(system="", user="u"))


async def test_no_choices_is_a_clear_error_not_an_index_error(monkeypatch):
    client = LiteLLMClient("fake/model")

    class Empty:
        choices: list[object] = []

    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(Empty()))
    with pytest.raises(LLMFatalError, match="choices"):
        await client.complete(LLMRequest(system="", user="u"))


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

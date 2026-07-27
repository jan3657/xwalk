import pytest

from xwalk.llm.base import LLMClient, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.fake import FakeLLM

REQ = LLMRequest(system="s", user="u")


async def test_returns_scripted_responses_in_order():
    llm = FakeLLM(["first", "second"])
    assert (await llm.complete(REQ)).text == "first"
    assert (await llm.complete(REQ)).text == "second"


async def test_records_every_request():
    llm = FakeLLM(["ok"])
    await llm.complete(LLMRequest(system="sys", user="hello"))
    assert llm.requests[0].user == "hello"


async def test_raises_a_scripted_exception():
    llm = FakeLLM([LLMRetryableError("429")])
    with pytest.raises(LLMRetryableError):
        await llm.complete(REQ)


async def test_running_out_of_script_is_a_loud_failure():
    llm = FakeLLM(["only one"])
    await llm.complete(REQ)
    with pytest.raises(AssertionError, match="exhausted"):
        await llm.complete(REQ)


async def test_a_handler_can_respond_based_on_the_request():
    def handler(request: LLMRequest) -> str:
        return "picked" if "C01" in request.user else "none"

    llm = FakeLLM(handler=handler)
    assert (await llm.complete(LLMRequest(system="", user="[C01] ID: x"))).text == "picked"
    assert (await llm.complete(LLMRequest(system="", user="nothing"))).text == "none"


async def test_usage_is_reported_and_call_count_accumulates():
    llm = FakeLLM(["a", "b"])
    first = await llm.complete(REQ)
    second = await llm.complete(REQ)
    assert first.usage.calls == 1 and second.usage.calls == 1
    assert llm.total_usage.calls == 2


async def test_capabilities_are_configurable():
    from xwalk.llm.base import LLMCapabilities

    llm = FakeLLM(["x"], capabilities=LLMCapabilities(json_schema=True))
    assert llm.capabilities.json_schema is True


async def test_default_capabilities_claim_nothing():
    llm = FakeLLM(["x"])
    caps = llm.capabilities
    assert not any(
        [
            caps.json_schema,
            caps.strict_schema,
            caps.usage_reporting,
            caps.seed,
            caps.native_retry_after,
        ]
    )


def test_fingerprint_changes_with_the_script():
    assert FakeLLM(["a"]).fingerprint != FakeLLM(["b"]).fingerprint


def test_fake_satisfies_the_llm_client_protocol():
    assert isinstance(FakeLLM(["x"]), LLMClient)


async def test_fatal_errors_pass_through_unchanged():
    llm = FakeLLM([LLMFatalError("bad key")])
    with pytest.raises(LLMFatalError, match="bad key"):
        await llm.complete(REQ)

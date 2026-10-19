"""Every dispatched request settles once, after transport and parsing finish."""

import json

import httpx
import pytest

from xwalk.decide.base import DecisionError, Noul
from xwalk.decide.budget import DecisionCallLimitExceeded
from xwalk.decide.jev import JevClient
from xwalk.llm.base import LLMError, LLMRequest
from xwalk.llm.budget import CallBudget, CallLimitExceeded
from xwalk.llm.chatgpt import ChatGPTClient


class RecordingBudget(CallBudget):
    def __init__(self, limit):
        super().__init__(limit)
        self.settlements = []

    def settle(self, index, error):
        self.settlements.append((index, error))
        super().settle(index, error)


def response(provider, outcome):
    if outcome == "transport":
        raise httpx.ConnectError("connection failed")
    if outcome in ("http", "fatal_http"):
        return httpx.Response(503 if outcome == "http" else 401, json={"error": "failed"})
    if provider == "jev":
        if outcome == "json":
            return httpx.Response(200, text="not JSON")
        payload = {
            "answers": {"n": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if outcome == "shape":
            payload = []
        elif outcome == "answer":
            payload["answers"] = {}
        return httpx.Response(200, json=payload)
    if outcome == "json":
        return httpx.Response(200, text="data: not JSON\n\n")
    if outcome == "shape":
        return httpx.Response(200, text='data: {"type":"response.completed"}\n\n')
    event = {
        "type": "response.completed",
        "response": {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    if outcome == "answer":
        event["response"]["output"] = []
    return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")


def client_and_call(provider, transport):
    if provider == "jev":
        client = JevClient("https://jev.invalid/decide", "m", transport=transport, max_retries=0)

        async def call():
            return await client.decide("state", {"n": Noul(instructions="q")})
    else:
        client = ChatGPTClient("m", lambda: "test-token", transport=transport)

        async def call():
            return await client.complete(LLMRequest(system="", user=""))

    return client, call


@pytest.mark.parametrize("provider", ["jev", "chatgpt"])
@pytest.mark.parametrize(
    "outcome", ["http", "fatal_http", "transport", "json", "shape", "answer", "ok"]
)
async def test_dispatch_settles_once_after_interpreting_response(provider, outcome):
    client, call = client_and_call(
        provider, httpx.MockTransport(lambda r: response(provider, outcome))
    )
    budget = RecordingBudget(1)
    client.attach_call_budget(budget)
    try:
        if outcome == "ok":
            assert (await call()).usage.calls == 1
        else:
            with pytest.raises((DecisionError, LLMError)) as failed:
                await call()
            assert failed.value.usage.calls == 1
        assert len(budget.settlements) == 1
        assert budget.settlements[0][0] == 1
        assert (budget.provider_error is None) == (outcome == "ok")
        with pytest.raises((DecisionCallLimitExceeded, CallLimitExceeded)) as refused:
            await call()
        assert refused.value.usage.calls == 0
        assert ("provider was failing" in str(refused.value)) == (outcome != "ok")
        assert len(budget.settlements) == budget.dispatched == 1
    finally:
        if provider == "jev":
            await client.aclose()


@pytest.mark.parametrize("provider", ["jev", "chatgpt"])
async def test_later_success_clears_the_settled_provider_error(provider):
    outcomes = iter(["http", "ok"])
    client, call = client_and_call(
        provider, httpx.MockTransport(lambda r: response(provider, next(outcomes)))
    )
    budget = RecordingBudget(2)
    client.attach_call_budget(budget)
    try:
        with pytest.raises((DecisionError, LLMError)):
            await call()
        assert budget.provider_error
        await call()
        assert budget.provider_error is None
        assert [index for index, _ in budget.settlements] == [1, 2]
    finally:
        if provider == "jev":
            await client.aclose()

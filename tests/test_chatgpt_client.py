"""Provider contract regression tests retained with the core after UI extraction."""

import json

import httpx
import pytest

from xwalk.llm.base import LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.budget import CallBudget, CallLimitExceeded
from xwalk.llm.chatgpt import ChatGPTClient

pytestmark = pytest.mark.chatgpt


def sse(*events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events)


def completed(text='{"key":"C01"}'):
    return {
        "type": "response.completed",
        "response": {
            "model": "allowed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        },
    }


async def test_responses_contract_completion_and_usage():
    def handle(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer secret"
        assert json.loads(request.content) == {
            "model": "allowed",
            "instructions": "system instructions",
            "input": [{"role": "user", "content": "user text"}],
            "store": False,
            "stream": True,
        }
        return httpx.Response(200, text=sse(completed()))

    client = ChatGPTClient("allowed", lambda: "secret", transport=httpx.MockTransport(handle))
    response = await client.complete(
        LLMRequest(
            system="system instructions",
            user="user text",
            temperature=1,
            max_tokens=5,
            seed=9,
            extra={"store": True},
        )
    )
    assert response.text == '{"key":"C01"}'
    assert (
        response.usage.calls,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    ) == (1, 11, 3)
    assert client.fingerprint == ChatGPTClient("allowed", lambda: "rotated").fingerprint


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "response.output_text.delta", "delta": "partial"}],
        [
            {
                "type": "response.failed",
                "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
            }
        ],
        [{"type": "response.incomplete"}],
        [{"type": "error"}],
        [],
    ],
)
async def test_partial_and_failed_streams_are_never_successful(events):
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse(*events))),
    )
    with pytest.raises(LLMFatalError) as exc:
        await client.complete(LLMRequest(system="", user=""))
    assert exc.value.usage.calls == 1
    if events and events[0]["type"] == "response.failed":
        assert "subscription_sharing_usage_limit_exceeded" in str(exc.value)


@pytest.mark.parametrize(
    "status,error",
    [
        (401, LLMFatalError),
        (403, LLMFatalError),
        (429, LLMRetryableError),
        (503, LLMRetryableError),
    ],
)
async def test_http_admission_errors(status, error):
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(status, json={"detail": "admission refused"})
        ),
    )
    with pytest.raises(error) as exc:
        await client.complete(LLMRequest(system="", user=""))
    assert str(status) in str(exc.value)
    assert exc.value.usage.calls == 1


async def test_call_budget_stops_requests_before_dispatch():
    seen = []

    def handle(r):
        seen.append(r)
        return httpx.Response(200, text=sse(completed()))

    client = ChatGPTClient("allowed", lambda: "secret", transport=httpx.MockTransport(handle))
    client.attach_call_budget(CallBudget(1))
    await client.complete(LLMRequest(system="", user=""))
    with pytest.raises(CallLimitExceeded):
        await client.complete(LLMRequest(system="", user=""))
    assert len(seen) == 1


@pytest.mark.parametrize("use_done", [False, True])
async def test_streamed_text_is_retained_when_completed_envelope_has_no_output(use_done):
    events = [
        {"type": "response.output_text.delta", "delta": '{"chosen_key":'},
        {"type": "response.output_text.delta", "delta": '"C01"}'},
    ]
    if use_done:
        events.append({"type": "response.output_text.done", "text": '{"chosen_key":"C01"}'})
    terminal = completed()
    terminal["response"]["output"] = []
    events.append(terminal)
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse(*events))),
    )
    response = await client.complete(LLMRequest(system="", user=""))
    assert response.text == '{"chosen_key":"C01"}'
    assert response.usage.completion_tokens == 3


async def test_empty_completed_response_is_a_provider_error_with_usage():
    terminal = completed("")
    client = ChatGPTClient(
        "allowed",
        lambda: "secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse(terminal))),
    )
    with pytest.raises(LLMFatalError, match="without any answer text") as exc:
        await client.complete(LLMRequest(system="", user=""))
    assert exc.value.usage.completion_tokens == 3

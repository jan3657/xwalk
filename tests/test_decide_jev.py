import json

import httpx
import pytest

from xwalk.decide.base import DecisionFatalError, DecisionRetryableError, Noul
from xwalk.decide.jev import JevClient

URL = "https://openrouter.ai/api/alpha/decisions"
OK = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {"n": {"type": "noul", "noul": 0.42}},
    "usage": {"input_tokens": 50, "output_tokens": 3, "cost": 0.000002},
}


def _client(responses, **kwargs):
    """`responses` is a list of (status, json_body) consumed in order; records requests."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    client = JevClient(
        URL,
        "~typesafe/jev-latest",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
        backoff_base=0.0,
        **kwargs,
    )
    return client, seen


async def test_posts_the_wire_shape_to_the_exact_url():
    client, seen = _client([(200, OK)])
    response = await client.decide({"source": "x"}, {"n": Noul(instructions="q?")})
    assert response.answers["n"].noul == 0.42  # type: ignore[union-attr]
    assert response.model == "typesafe/jev-1.13-20260917"
    request = seen[0]
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {
        "model": "~typesafe/jev-latest",
        "state": {"source": "x"},
        "questions": {"n": {"type": "noul", "instructions": "q?"}},
    }


async def test_retries_a_429_then_succeeds():
    client, seen = _client([(429, {"error": "slow down"}), (200, OK)], max_retries=2)
    await client.decide("s", {"n": Noul(instructions="q?")})
    assert len(seen) == 2


async def test_exhausted_retries_raise_retryable():
    client, seen = _client([(503, {}), (503, {}), (503, {})], max_retries=2)
    with pytest.raises(DecisionRetryableError):
        await client.decide("s", {"n": Noul(instructions="q?")})
    assert len(seen) == 3


async def test_a_400_is_fatal_and_carries_the_message():
    client, _ = _client([(400, {"error": {"message": "Model x does not exist"}})])
    with pytest.raises(DecisionFatalError, match="does not exist"):
        await client.decide("s", {"n": Noul(instructions="q?")})


async def test_a_401_is_fatal():
    client, _ = _client([(401, {"error": "bad key"})])
    with pytest.raises(DecisionFatalError):
        await client.decide("s", {"n": Noul(instructions="q?")})


async def test_a_200_missing_an_answer_is_fatal():
    client, _ = _client([(200, {"model": "m", "answers": {}, "usage": {}})])
    with pytest.raises(DecisionFatalError, match="'n'"):
        await client.decide("s", {"n": Noul(instructions="q?")})


def test_fingerprint_excludes_the_key_and_covers_host_and_model():
    a = JevClient(URL, "m", api_key="one")
    b = JevClient(URL, "m", api_key="two")
    c = JevClient(URL, "other", api_key="one")
    d = JevClient("https://api.typesafe.ai/v1/systemone", "m", api_key="one")
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.fingerprint != d.fingerprint
    assert "one" not in a.fingerprint

import pytest

from xwalk.ledger import Ledger
from xwalk.llm.base import LLMCapabilities, LLMClient, LLMRequest
from xwalk.llm.cache import CachingLLM, llm_cache_key
from xwalk.llm.fake import FakeLLM

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    yield led
    led.close()


def wrap(ledger, script):
    inner = FakeLLM(script)
    return inner, CachingLLM(inner, ledger)


async def test_a_repeated_request_is_served_from_the_cache(ledger):
    inner, cached = wrap(ledger, ["first"])
    request = LLMRequest(system="s", user="u")
    assert (await cached.complete(request)).text == "first"
    assert (await cached.complete(request)).text == "first"
    assert len(inner.requests) == 1  # the script would have raised on a second call


async def test_a_different_user_prompt_misses(ledger):
    inner, cached = wrap(ledger, ["a", "b"])
    assert (await cached.complete(LLMRequest(system="s", user="one"))).text == "a"
    assert (await cached.complete(LLMRequest(system="s", user="two"))).text == "b"


async def test_a_different_system_prompt_misses(ledger):
    inner, cached = wrap(ledger, ["a", "b"])
    await cached.complete(LLMRequest(system="one", user="u"))
    await cached.complete(LLMRequest(system="two", user="u"))
    assert len(inner.requests) == 2


def test_the_key_is_not_vulnerable_to_prompt_boundary_collisions(ledger):
    """system='ab', user='c' and system='a', user='bc' are different requests."""
    llm = FakeLLM(["x"])
    assert llm_cache_key(llm, LLMRequest(system="ab", user="c")) != llm_cache_key(
        llm, LLMRequest(system="a", user="bc")
    )


def test_the_key_covers_the_schema(ledger):
    llm = FakeLLM(["x"])
    assert llm_cache_key(llm, LLMRequest(system="", user="u")) != llm_cache_key(
        llm, LLMRequest(system="", user="u", schema=SCHEMA)
    )


def test_the_key_covers_generation_parameters(ledger):
    llm = FakeLLM(["x"])
    assert llm_cache_key(llm, LLMRequest(system="", user="u", temperature=0.0)) != llm_cache_key(
        llm, LLMRequest(system="", user="u", temperature=0.7)
    )


def test_the_key_covers_provider_identity(ledger):
    """Same prompt, different client — never the same cache entry."""
    request = LLMRequest(system="", user="u")
    assert llm_cache_key(FakeLLM(["x"]), request) != llm_cache_key(FakeLLM(["y"]), request)


async def test_a_cache_hit_reports_no_token_spend(ledger):
    inner, cached = wrap(ledger, ["first"])
    request = LLMRequest(system="s", user="u")
    await cached.complete(request)
    second = await cached.complete(request)
    assert second.usage.calls == 0
    assert second.usage.total_tokens == 0


async def test_hit_and_miss_counts_are_tracked(ledger):
    inner, cached = wrap(ledger, ["first"])
    request = LLMRequest(system="s", user="u")
    await cached.complete(request)
    await cached.complete(request)
    assert (cached.hits, cached.misses) == (1, 1)


async def test_write_can_be_disabled(ledger):
    inner = FakeLLM(["a", "b"])
    cached = CachingLLM(inner, ledger, write=False)
    request = LLMRequest(system="s", user="u")
    await cached.complete(request)
    await cached.complete(request)
    assert len(inner.requests) == 2


async def test_the_cache_survives_reopening_the_ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    await CachingLLM(FakeLLM(["first"]), led).complete(LLMRequest(system="s", user="u"))
    led.close()

    reopened = Ledger.open(tmp_path / "run.sqlite")
    # The same script gives the same client fingerprint, hence the same cache key. A
    # client scripted differently is a *different* provider identity and must miss —
    # that is the point of putting `client.fingerprint` in the key.
    fresh = FakeLLM(["first"])
    response = await CachingLLM(fresh, reopened).complete(LLMRequest(system="s", user="u"))
    reopened.close()
    assert response.text == "first"
    assert fresh.requests == []  # answered from disk, never delegated


def test_identity_delegates_to_the_inner_client(ledger):
    inner = FakeLLM(["x"], capabilities=LLMCapabilities(json_schema=True), model="m1")
    cached = CachingLLM(inner, ledger)
    assert cached.model == "m1"
    assert cached.capabilities.json_schema is True
    # The wrapper must be invisible to run fingerprinting: caching changes how an
    # answer was obtained, never what the answer means.
    assert cached.fingerprint == inner.fingerprint


def test_caching_llm_satisfies_the_llm_client_protocol(ledger):
    assert isinstance(CachingLLM(FakeLLM(["x"]), ledger), LLMClient)

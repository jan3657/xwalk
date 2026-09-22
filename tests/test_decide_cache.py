import pytest

from xwalk.decide.base import Noul
from xwalk.decide.cache import CachingDecider, decision_cache_key
from xwalk.decide.fake import FakeDecider
from xwalk.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    yield led
    led.close()


STATE = {"source": {"fields": {"mention": "glucose"}}, "candidates": {"C001": "glucose"}}
QUESTIONS = {"n_C001": Noul(instructions="`candidates.C001`?")}


async def test_a_repeated_request_is_served_from_the_cache(ledger):
    inner = FakeDecider()
    cached = CachingDecider(inner, ledger)
    first = await cached.decide(STATE, QUESTIONS)
    second = await cached.decide(STATE, QUESTIONS)
    assert first.answers == second.answers
    assert len(inner.calls) == 1
    assert second.usage.calls == 0
    assert (cached.hits, cached.misses) == (1, 1)


async def test_a_different_state_misses(ledger):
    inner = FakeDecider()
    cached = CachingDecider(inner, ledger)
    await cached.decide(STATE, QUESTIONS)
    await cached.decide({**STATE, "candidates": {"C001": "fructose"}}, QUESTIONS)
    assert len(inner.calls) == 2


def test_the_key_covers_questions_and_model():
    a = FakeDecider(model="a")
    b = FakeDecider(model="b")
    other = {"n_C001": Noul(instructions="different?")}
    assert decision_cache_key(a, STATE, QUESTIONS) != decision_cache_key(b, STATE, QUESTIONS)
    assert decision_cache_key(a, STATE, QUESTIONS) != decision_cache_key(a, STATE, other)


def test_the_key_does_not_collide_on_state_boundaries():
    assert decision_cache_key(FakeDecider(), {"a": "bc"}, QUESTIONS) != decision_cache_key(
        FakeDecider(), {"ab": "c"}, QUESTIONS
    )


def test_wrapping_does_not_change_identity(ledger):
    inner = FakeDecider()
    cached = CachingDecider(inner, ledger)
    assert cached.fingerprint == inner.fingerprint
    assert cached.model == inner.model

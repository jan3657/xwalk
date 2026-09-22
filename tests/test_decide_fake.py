import pytest

from xwalk.decide.base import Choice, DecisionFatalError, Noul, Score
from xwalk.decide.fake import FakeDecider, overlap_handler

STATE = {
    "source": {"fields": {"mention": "glucose"}},
    "candidates": {"C001": "ID: T1 Label: glucose", "C002": "ID: T2 Label: fructose"},
}


async def test_overlap_handler_scores_by_shared_tokens():
    fake = FakeDecider()
    response = await fake.decide(
        STATE,
        {
            "n_C001": Noul(
                instructions="Does `candidates.C001` denote the same entity as `source`?"
            ),
            "n_C002": Noul(
                instructions="Does `candidates.C002` denote the same entity as `source`?"
            ),
        },
    )
    assert response.answers["n_C001"].noul > response.answers["n_C002"].noul  # type: ignore[union-attr]
    assert response.usage.calls == 1
    assert len(fake.calls) == 1


async def test_choice_picks_the_best_overlap_or_none():
    fake = FakeDecider()
    best = Choice(
        instructions="Which?", criteria={"C001": "glucose", "C002": "fructose", "NONE": "none"}
    )
    response = await fake.decide(STATE, {"best": best})
    assert response.answers["best"].choice == "C001"  # type: ignore[union-attr]

    nothing = {"source": {"fields": {"mention": "unobtainium"}}, "candidates": STATE["candidates"]}
    response = await fake.decide(nothing, {"best": best})
    assert response.answers["best"].choice == "NONE"  # type: ignore[union-attr]


async def test_score_returns_the_middle_level():
    fake = FakeDecider()
    response = await fake.decide(STATE, {"g": Score(instructions="?", criteria=["a", "b", "c"])})
    assert response.answers["g"].score == pytest.approx(1.0)  # type: ignore[union-attr]


async def test_a_handler_can_raise():
    fake = FakeDecider(handler=lambda state, questions: DecisionFatalError("boom"))
    with pytest.raises(DecisionFatalError):
        await fake.decide(STATE, {"n": Noul(instructions="?")})


def test_overlap_handler_is_deterministic():
    questions = {"n_C001": Noul(instructions="`candidates.C001`")}
    assert overlap_handler(STATE, questions) == overlap_handler(STATE, questions)

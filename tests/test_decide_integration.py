"""Hits the real decision endpoint. Skipped without a key; costs well under a cent."""

import os

import pytest

from xwalk.decide.base import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from xwalk.decide.jev import JevClient

pytestmark = pytest.mark.integration

KEY = os.environ.get("XWALK_TEST_API_KEY")
URL = os.environ.get("XWALK_TEST_DECIDER_URL", "https://openrouter.ai/api/alpha/decisions")
MODEL = os.environ.get("XWALK_TEST_DECIDER_MODEL", "~typesafe/jev-latest")


@pytest.mark.skipif(not KEY, reason="XWALK_TEST_API_KEY unset")
async def test_one_call_with_all_three_question_types():
    client = JevClient(URL, MODEL, api_key=KEY, max_retries=2)
    try:
        response = await client.decide(
            {
                "source": {"fields": {"english_name": "pineapple, canned in syrup"}},
                "candidates": {
                    "C1": "pineapple (whole, raw)",
                    "C2": "pineapple (canned, in syrup)",
                },
            },
            {
                "n_C1": Noul(instructions="Does `candidates.C1` denote the same food as `source`?"),
                "n_C2": Noul(instructions="Does `candidates.C2` denote the same food as `source`?"),
                "best": Choice(
                    instructions="Which candidate is the same food as `source`? NONE if none.",
                    criteria={"C1": "whole raw", "C2": "canned in syrup", "NONE": "none"},
                ),
                "grade": Score(
                    instructions="How well does `candidates.C2` match `source`?",
                    criteria=["poor", "fair", "exact"],
                ),
            },
        )
    finally:
        await client.aclose()
    n1, n2, best, grade = (response.answers[k] for k in ("n_C1", "n_C2", "best", "grade"))
    assert isinstance(n1, NoulAnswer) and isinstance(n2, NoulAnswer)
    assert isinstance(best, ChoiceAnswer) and isinstance(grade, ScoreAnswer)
    assert n2.noul > n1.noul
    assert best.choice == "C2"
    assert abs(sum(best.probabilities.values()) - 1.0) < 0.02
    assert response.model.startswith("typesafe/jev")
    assert response.usage.prompt_tokens > 0

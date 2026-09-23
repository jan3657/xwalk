"""Hits the real decision endpoint. Skipped without a key; costs well under a cent."""

import os
from pathlib import Path

import pytest

from xwalk.decide.base import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from xwalk.decide.jev import JevClient
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import load_slots
from xwalk.records import Candidate, Record
from xwalk.stages.screen import Screener
from xwalk.templates import TemplateSet

pytestmark = pytest.mark.integration

KEY = os.environ.get("XWALK_TEST_API_KEY")
URL = os.environ.get("XWALK_TEST_DECIDER_URL", "https://openrouter.ai/api/alpha/decisions")
MODEL = os.environ.get("XWALK_TEST_DECIDER_MODEL", "~typesafe/jev-latest")
REF_ZIVILA_SLOTS = Path(__file__).parent.parent / "examples/ref_zivila/jobs/foodon/slots.yaml"


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


@pytest.mark.skipif(not KEY, reason="XWALK_TEST_API_KEY unset")
async def test_the_short_screen_question_reads_the_rules_from_the_state():
    """Version 2 moves the preamble into `rules`; Jev must still follow it by reference."""
    questions = QuestionSet.from_slots(load_slots(REF_ZIVILA_SLOTS))
    templates = TemplateSet(
        query="{{ mention_en }}", context="", doc="{{ label }}", candidate="{{ label }}"
    )
    candidates = [
        Candidate(record=Record(id=rid, fields={"label": label}), fused_score=1.0, evidence=())
        for rid, label in (
            ("T1", "pineapple (whole, raw)"),
            ("T2", "pineapple (canned, in syrup)"),
            ("T3", "canning process"),
        )
    ]
    source = Record(id="s1", fields={"mention_en": "Pineapple, canned in syrup"})
    client = JevClient(URL, MODEL, api_key=KEY, max_retries=2)
    try:
        outcome = await Screener(client, questions, templates).screen(source, "", candidates)
    finally:
        await client.aclose()
    by_id = {outcome.issued[key]: p for key, p in outcome.probabilities.items()}
    assert by_id["T2"] > by_id["T1"]
    assert by_id["T2"] > by_id["T3"]
    assert outcome.shortlist[0] == "T2"
    assert outcome.usage.prompt_tokens > 0

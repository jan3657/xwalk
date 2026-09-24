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


REWRITE_URL = os.environ.get("XWALK_TEST_REWRITE_URL", "https://openrouter.ai/api/v1")
REWRITE_MODEL = os.environ.get("XWALK_TEST_REWRITE_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
NCBI_SLOTS = Path(__file__).parent.parent / "examples/ncbi_disease/slots.yaml"


@pytest.mark.skipif(not KEY, reason="XWALK_TEST_API_KEY unset")
async def test_a_screen_miss_is_rewritten_by_a_real_llm_and_decided_by_jev():
    """The first query only retrieves an unrelated disease; the rewrite must find the gold."""
    from xwalk.decide.matcher import DecisionMatcher
    from xwalk.llm.openai_compat import OpenAICompatClient
    from xwalk.prompts.contract import PromptSet
    from xwalk.records import RetrievalHit
    from xwalk.retrieval.base import SearchRequest
    from xwalk.stages.choose import Chooser
    from xwalk.stages.property_gate import PropertyGate
    from xwalk.stages.rewrite import QueryRewriter
    from xwalk.stores.memory import MemoryStore

    labels = {"D1": "myocardial infarction", "D2": "asthma", "D3": "psoriasis"}
    store = MemoryStore.from_source([Record(id=k, fields={"label": v}) for k, v in labels.items()])

    class Keyword:
        name, fingerprint, default_limit = "kw", "kw", 10

        async def search(self, request: SearchRequest):
            text = request.text.lower()
            ids = ["D2"] if text == "heart attack" else []
            ids += [k for k, v in labels.items() if any(w in text for w in v.split())]
            return [
                RetrievalHit(record_id=rid, retriever="kw", raw_score=1.0, rank=i)
                for i, rid in enumerate(dict.fromkeys(ids), start=1)
            ]

    slots = load_slots(NCBI_SLOTS)
    questions = QuestionSet.from_slots(slots)
    templates = TemplateSet(
        query="{{ mention }}", context="", doc="{{ label }}", candidate="{{ label }}"
    )
    decider = JevClient(URL, MODEL, api_key=KEY, max_retries=2)
    llm = OpenAICompatClient(REWRITE_URL, REWRITE_MODEL, api_key=KEY, max_retries=2)
    matcher = DecisionMatcher(
        templates=templates,
        retrievers=[Keyword()],
        store=store,
        screener=Screener(decider, questions, templates),
        chooser=Chooser(decider, questions, templates),
        gate=PropertyGate(decider, questions, templates),
        rewriter=QueryRewriter(llm, PromptSet.from_slots(slots), templates, max_queries=3),
    )
    try:
        result = await matcher.match(Record(id="s1", fields={"mention": "heart attack"}))
    finally:
        await decider.aclose()
        await llm.aclose()
    attempt = result.attempts[0]
    assert "rewrite:" in (attempt.error or "") and "failed" not in (attempt.error or "")
    assert attempt.query.startswith("heart attack | ")
    assert "D1" in {c.id for c in result.candidates}
    assert result.matched_id == "D1"

import json

import pytest

from xwalk.llm.fake import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import assign_keys
from xwalk.stages.rewrite import QueryRewriter
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="", doc="", candidate="ID: {{ id }} Label: {{ label }}"
)
PROMPTS = PromptSet.from_slots(
    PromptSlots(
        entity_noun="mention",
        target_noun="term",
        domain_brief="test",
        rubric=[
            {"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.4, "name": "Weak", "when": "vague"},
        ],
    )
)
SOURCE = Record(id="s1", fields={"mention": "ASA"})
KEYED = assign_keys(
    [
        Candidate(
            record=Record(id="T1", fields={"label": "aspirin"}),
            fused_score=0.5,
            evidence=(RetrievalHit(record_id="T1", retriever="bm25", raw_score=1.0, rank=1),),
        )
    ],
    TEMPLATES,
)


async def test_returns_query_proposals():
    llm = FakeLLM([json.dumps({"queries": ["acetylsalicylic acid"], "explanation": "expanded"})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert [(p.kind, p.value, p.source) for p in outcome.proposals] == [
        ("query", "acetylsalicylic acid", "rewriter")
    ]


async def test_all_proposals_from_the_rewriter_are_queries():
    llm = FakeLLM([json.dumps({"queries": ["a", "b"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert all(p.kind == "query" for p in outcome.proposals)


async def test_respects_max_queries():
    llm = FakeLLM([json.dumps({"queries": ["a", "b", "c", "d"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES, max_queries=2).rewrite(
        SOURCE, "", ["ASA"], KEYED
    )
    assert len(outcome.proposals) == 2


async def test_repeats_of_previous_queries_are_dropped():
    llm = FakeLLM([json.dumps({"queries": ["ASA", "acetylsalicylic acid"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert [p.value for p in outcome.proposals] == ["acetylsalicylic acid"]


async def test_the_prompt_lists_previous_queries():
    llm = FakeLLM([json.dumps({"queries": ["x"], "explanation": ""})])
    await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA", "aspirin"], KEYED)
    prompt = llm.requests[0].user
    assert "- ASA" in prompt and "- aspirin" in prompt


async def test_malformed_output_yields_no_proposals_and_an_error():
    llm = FakeLLM(["nope"])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert outcome.proposals == () and outcome.error is not None


@pytest.mark.parametrize(
    "queries",
    [123, "glucose", {"q": "glucose"}, None, [1, 2], ["ok", 3], ["ok", None]],
)
async def test_queries_that_are_not_a_list_of_strings_yield_an_error_and_no_proposals(queries):
    # A string must never be iterated character by character into queries "g", "l".
    llm = FakeLLM([json.dumps({"queries": queries, "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert outcome.proposals == ()
    assert outcome.error is not None and "queries" in outcome.error


async def test_missing_queries_yield_an_error_and_no_proposals():
    llm = FakeLLM([json.dumps({"explanation": "nothing"})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert outcome.proposals == ()
    assert outcome.error is not None and "queries" in outcome.error


async def test_an_empty_query_list_is_a_valid_answer_not_an_error():
    llm = FakeLLM([json.dumps({"queries": [], "explanation": "no better idea"})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert outcome.proposals == () and outcome.error is None


async def test_works_when_there_were_no_candidates():
    from xwalk.stages.keying import assign_keys as ak

    llm = FakeLLM([json.dumps({"queries": ["acetylsalicylic acid"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(
        SOURCE, "", ["ASA"], ak([], TEMPLATES)
    )
    assert len(outcome.proposals) == 1

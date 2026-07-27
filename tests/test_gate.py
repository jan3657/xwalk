import json

import pytest

from xwalk.llm.fake import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.keying import assign_keys
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ symbol }}", context="", doc="", candidate="ID: {{ id }} Label: {{ label }}"
)
SLOTS = PromptSlots(
    entity_noun="gene mention",
    target_noun="NCBI gene",
    domain_brief="gene nomenclature",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact symbol and organism match"},
        {"score": 0.4, "name": "Weak", "when": "symbol matches, organism unknown"},
    ],
)
PROMPTS = PromptSet.from_slots(SLOTS)
SOURCE = Record(id="s1", fields={"symbol": "TP53", "organism": "Homo sapiens"})


def cand(record_id: str, label: str) -> Candidate:
    return Candidate(
        record=Record(id=record_id, fields={"label": label}),
        fused_score=0.5,
        evidence=(RetrievalHit(record_id=record_id, retriever="bm25", raw_score=1.0, rank=1),),
    )


KEYED = assign_keys([cand("NCBIGene:7157", "TP53"), cand("NCBIGene:22059", "Trp53")], TEMPLATES)


def score_reply(**kwargs) -> str:
    body = {"confidence_score": 0.5, "explanation": ""}
    body.update(kwargs)
    return json.dumps(body)


def verify_reply(**kwargs) -> str:
    body = {"decision": "support", "explanation": ""}
    body.update(kwargs)
    return json.dumps(body)


# --- scorer ----------------------------------------------------------------------


async def test_scorer_returns_the_confidence():
    llm = FakeLLM([score_reply(confidence_score=0.83, explanation="symbol and organism agree")])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert outcome.score == 0.83


async def test_scorer_sees_every_source_field_not_just_the_query():
    llm = FakeLLM([score_reply(confidence_score=0.8)])
    await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    prompt = llm.requests[0].user
    assert "TP53" in prompt and "Homo sapiens" in prompt


async def test_scorer_shows_the_chosen_candidate_and_the_others_separately():
    llm = FakeLLM([score_reply(confidence_score=0.8)])
    await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    prompt = llm.requests[0].user
    assert "## Proposed match" in prompt
    assert "[C01] ID: NCBIGene:7157" in prompt
    assert "[C02] ID: NCBIGene:22059" in prompt


async def test_scorer_emits_candidate_proposals_for_issued_keys():
    llm = FakeLLM([score_reply(confidence_score=0.3, better_candidate_keys=["C02"])])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert [(p.kind, p.value, p.source) for p in outcome.proposals] == [
        ("candidate", "C02", "scorer")
    ]


async def test_scorer_emits_query_proposals_separately():
    llm = FakeLLM([score_reply(confidence_score=0.3, better_queries=["tumour protein p53"])])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert [(p.kind, p.value) for p in outcome.proposals] == [("query", "tumour protein p53")]


async def test_scorer_keeps_candidate_and_query_proposals_distinct():
    """The paper repo funnels both into one query queue and re-runs full retrieval on
    records it already has in hand. They are different objects."""
    llm = FakeLLM(
        [score_reply(confidence_score=0.3, better_candidate_keys=["C02"], better_queries=["p53"])]
    )
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    kinds = {p.kind for p in outcome.proposals}
    assert kinds == {"candidate", "query"}


async def test_scorer_drops_blank_and_duplicate_proposals():
    llm = FakeLLM([score_reply(confidence_score=0.3, better_queries=["p53", "  ", "p53", ""])])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert [p.value for p in outcome.proposals] == ["p53"]


async def test_scorer_malformed_output_yields_a_zero_score_and_an_error():
    llm = FakeLLM(["not json"])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert outcome.score is None and outcome.error is not None


async def test_scorer_clamps_an_out_of_range_score():
    llm = FakeLLM([score_reply(confidence_score=-2)])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert outcome.score == 0.0


async def test_scorer_rejects_an_unissued_chosen_key():
    llm = FakeLLM([])
    with pytest.raises(KeyError):
        await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C99")


# --- verifier --------------------------------------------------------------------


async def test_verifier_supports():
    llm = FakeLLM([verify_reply(decision="support", confidence_score=0.9)])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "support"


async def test_verifier_disagrees_and_names_a_preferred_key():
    llm = FakeLLM([verify_reply(decision="disagree", preferred_key="C02")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree" and verdict.preferred_key == "C02"


async def test_verifier_returns_no_match():
    llm = FakeLLM([verify_reply(decision="no_match")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "no_match"


async def test_verifier_drops_a_preferred_key_that_was_never_issued():
    """A preferred key we cannot resolve is noise, not a lead."""
    llm = FakeLLM([verify_reply(decision="disagree", preferred_key="C99")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree" and verdict.preferred_key is None


async def test_verifier_treats_an_unknown_decision_as_disagreement():
    """Fail towards review, never towards silent acceptance."""
    llm = FakeLLM([verify_reply(decision="maybe")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree"


async def test_verifier_malformed_output_is_disagreement_with_an_error():
    llm = FakeLLM(["???"])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree" and verdict.error is not None


async def test_verifier_sees_the_full_source_record():
    llm = FakeLLM([verify_reply()])
    await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert "Homo sapiens" in llm.requests[0].user


async def test_verifier_prompt_is_not_the_scorer_prompt():
    """An independent verdict requires an independent framing."""
    scorer_llm = FakeLLM([score_reply(confidence_score=0.7)])
    verifier_llm = FakeLLM([verify_reply()])
    await Scorer(scorer_llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    await Verifier(verifier_llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert scorer_llm.requests[0].user != verifier_llm.requests[0].user

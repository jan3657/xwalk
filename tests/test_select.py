import json

import pytest

from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import Resolution
from xwalk.stages.select import Selector, SelectorPolicy, apply_budget
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="", doc="", candidate="ID: {{ id }} Label: {{ label }}"
)
SLOTS = PromptSlots(
    entity_noun="mention",
    target_noun="term",
    domain_brief="test domain",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact"},
        {"score": 0.4, "name": "Weak", "when": "vague"},
    ],
)
PROMPTS = PromptSet.from_slots(SLOTS)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


def cand(record_id: str, label: str, score: float) -> Candidate:
    return Candidate(
        record=Record(id=record_id, fields={"label": label}),
        fused_score=score,
        evidence=(RetrievalHit(record_id=record_id, retriever="bm25", raw_score=1.0, rank=1),),
    )


CANDIDATES = [cand("T1", "glucose", 0.9), cand("T2", "fructose", 0.5)]


def reply(**kwargs) -> str:
    body = {"chosen_key": None, "confidence_score": 0.0, "explanation": ""}
    body.update(kwargs)
    return json.dumps(body)


# --- budget ---------------------------------------------------------------------


def test_budget_keeps_everything_when_under_the_cap():
    kept, dropped = apply_budget(CANDIDATES, SelectorPolicy())
    assert len(kept) == 2 and dropped == 0


def test_budget_truncates_to_max_candidates_and_reports_the_count():
    many = [cand(f"T{i}", f"L{i}", 1.0 - i / 100) for i in range(50)]
    kept, dropped = apply_budget(many, SelectorPolicy(max_candidates=10))
    assert len(kept) == 10 and dropped == 40


def test_budget_keeps_the_highest_scoring_candidates():
    many = [cand(f"T{i}", f"L{i}", i / 100) for i in range(10)]
    kept, _ = apply_budget(
        sorted(many, key=lambda c: -c.fused_score), SelectorPolicy(max_candidates=3)
    )
    assert [c.id for c in kept] == ["T9", "T8", "T7"]


def test_budget_truncation_is_deterministic_across_runs():
    many = [cand(f"T{i}", f"L{i}", 0.5) for i in range(20)]  # all tied
    a, _ = apply_budget(many, SelectorPolicy(max_candidates=5))
    b, _ = apply_budget(list(reversed(many)), SelectorPolicy(max_candidates=5))
    assert [c.id for c in a] == [c.id for c in b]


def test_token_budget_trims_further_than_the_count_budget():
    long_label = "x" * 4000
    many = [cand(f"T{i}", long_label, 1.0 - i / 100) for i in range(20)]
    kept, dropped = apply_budget(many, SelectorPolicy(max_candidates=20, max_candidate_tokens=2000))
    assert len(kept) < 20 and dropped == 20 - len(kept)


def test_token_budget_always_keeps_at_least_one_candidate():
    huge = [cand("T1", "y" * 100_000, 1.0)]
    kept, dropped = apply_budget(huge, SelectorPolicy(max_candidate_tokens=10))
    assert len(kept) == 1 and dropped == 0


# --- selection ------------------------------------------------------------------


async def test_resolves_a_chosen_key_to_a_record_id():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.95, explanation="exact")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id == "T1"
    assert outcome.choice.resolution is Resolution.EXACT_KEY
    assert outcome.confidence == 0.95


async def test_abstention_yields_no_record_id():
    llm = FakeLLM([reply(chosen_key=None, confidence_score=0.0, explanation="none fit")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id is None
    assert outcome.choice.resolution is Resolution.ABSTAIN


async def test_a_hallucinated_id_does_not_resolve_to_a_record():
    llm = FakeLLM([reply(chosen_key="T1", confidence_score=0.99, explanation="sure")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id is None
    assert outcome.choice.resolution is Resolution.UNRESOLVED


async def test_a_bare_integer_does_not_resolve_to_a_record():
    llm = FakeLLM([reply(chosen_key="1", confidence_score=0.99, explanation="the first")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id is None


async def test_the_prompt_carries_the_keyed_candidate_block():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert "[C01] ID: T1" in llm.requests[0].user


async def test_the_prompt_carries_the_context_when_present():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "blood [glucose] levels", CANDIDATES)
    assert "blood [glucose] levels" in llm.requests[0].user


async def test_the_request_carries_the_select_schema():
    from xwalk.prompts.contract import SELECT_SCHEMA

    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert llm.requests[0].schema == SELECT_SCHEMA


async def test_malformed_output_is_unresolved_not_an_exception():
    llm = FakeLLM(["I decline to answer."])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.resolution is Resolution.UNRESOLVED
    assert outcome.error is not None


async def test_output_with_a_wrong_confidence_type_is_unresolved():
    llm = FakeLLM(['{"chosen_key": "C01", "confidence_score": "high", "explanation": "x"}'])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.resolution is Resolution.UNRESOLVED


async def test_a_confidence_outside_zero_to_one_is_clamped():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=1.7, explanation="x")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.confidence == 1.0


async def test_provider_errors_propagate_for_the_matcher_to_classify():
    llm = FakeLLM([LLMRetryableError("429")])
    with pytest.raises(LLMRetryableError):
        await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)


async def test_no_candidates_short_circuits_without_calling_the_model():
    llm = FakeLLM([])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", [])
    assert outcome.choice.resolution is Resolution.ABSTAIN
    assert llm.requests == []
    assert outcome.usage.calls == 0


async def test_truncation_count_is_reported_on_the_outcome():
    many = [cand(f"T{i}", f"L{i}", 1.0 - i / 100) for i in range(40)]
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    outcome = await Selector(
        llm, PROMPTS, TEMPLATES, policy=SelectorPolicy(max_candidates=10)
    ).select(SOURCE, "", many)
    assert outcome.truncated == 30
    assert len(outcome.keyed) == 10


async def test_legacy_mode_accepts_a_raw_id_and_flags_it():
    llm = FakeLLM([reply(chosen_key="T1", confidence_score=0.9)])
    outcome = await Selector(llm, PROMPTS, TEMPLATES, legacy_id_resolution=True).select(
        SOURCE, "", CANDIDATES
    )
    assert outcome.choice.record_id == "T1"
    assert outcome.choice.resolution is Resolution.LEGACY_EXACT_ID


async def test_usage_is_recorded():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.usage.calls == 1

import pytest

from xwalk.decide.base import DecisionRetryableError, Noul
from xwalk.decide.fake import FakeDecider
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.screen import Screener, build_source_state
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ note }}",
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)
QUESTIONS = QuestionSet.from_slots(
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
SOURCE = Record(id="s1", fields={"mention": "glucose", "note": "in blood"})


def _cands(*labels):
    return [
        Candidate(
            record=Record(id=f"T{i}", fields={"label": label}),
            fused_score=1.0 / (i + 1),
            evidence=(
                RetrievalHit(record_id=f"T{i}", retriever="bm25", raw_score=1.0, rank=i + 1),
            ),
        )
        for i, label in enumerate(labels)
    ]


def test_source_state_omits_an_empty_context():
    assert build_source_state(SOURCE, "") == {"fields": dict(SOURCE.fields)}
    assert build_source_state(SOURCE, "ctx") == {"fields": dict(SOURCE.fields), "context": "ctx"}


async def test_one_noul_per_candidate_with_keys_numbered_across_chunks():
    fake = FakeDecider()
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=2)
    outcome = await screener.screen(SOURCE, "in blood", _cands("glucose", "fructose", "sucrose"))
    assert outcome.chunks == 2
    assert len(fake.calls) == 2
    first_state, first_questions = fake.calls[0]
    second_state, second_questions = fake.calls[1]
    assert list(first_state["candidates"]) == ["C001", "C002"]
    assert list(second_state["candidates"]) == ["C003"]
    assert set(first_questions) == {"n_C001", "n_C002"} and set(second_questions) == {"n_C003"}
    assert all(isinstance(q, Noul) for q in first_questions.values())
    assert first_state["source"] == {"fields": dict(SOURCE.fields), "context": "in blood"}
    assert first_state["candidates"]["C001"] == "ID: T0 Label: glucose"
    assert outcome.issued == {"C001": "T0", "C002": "T1", "C003": "T2"}
    assert outcome.usage.calls == 2


async def test_shortlist_is_best_first_and_floored():
    screener = Screener(FakeDecider(), QUESTIONS, TEMPLATES, shortlist_size=5, shortlist_floor=0.1)
    outcome = await screener.screen(SOURCE, "", _cands("fructose", "glucose", "sucrose"))
    assert outcome.shortlist[0] == "T1"
    assert all(
        outcome.probabilities[k] >= 0.1
        for k, rid in outcome.issued.items()
        if rid in outcome.shortlist
    )


async def test_shortlist_is_capped():
    screener = Screener(FakeDecider(), QUESTIONS, TEMPLATES, shortlist_size=1, shortlist_floor=0.0)
    outcome = await screener.screen(SOURCE, "", _cands("glucose", "glucose syrup", "sucrose"))
    assert len(outcome.shortlist) == 1


async def test_no_candidates_means_no_calls():
    fake = FakeDecider()
    outcome = await Screener(fake, QUESTIONS, TEMPLATES).screen(SOURCE, "", [])
    assert outcome.shortlist == () and outcome.chunks == 0 and fake.calls == []


async def test_provider_errors_propagate():
    fake = FakeDecider(handler=lambda s, q: DecisionRetryableError("busy"))
    with pytest.raises(DecisionRetryableError):
        await Screener(fake, QUESTIONS, TEMPLATES).screen(SOURCE, "", _cands("glucose"))


async def test_an_oversized_chunk_is_split_with_a_note():
    fake = FakeDecider()
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=3, max_state_chars=80)
    outcome = await screener.screen(SOURCE, "", _cands("glucose", "fructose", "sucrose"))
    assert outcome.chunks >= 2
    assert any("split" in note for note in outcome.notes)

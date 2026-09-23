import asyncio
import json
from pathlib import Path

import pytest

from xwalk.decide.base import (
    DecisionFatalError,
    DecisionRetryableError,
    Noul,
    question_to_dict,
)
from xwalk.decide.fake import FakeDecider, overlap_handler
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots, load_slots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.screen import Screener, ScreenFailed, build_source_state
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
    assert first_state["rules"] == QUESTIONS.rules_state() == second_state["rules"]
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
    """Wrapped, so the caller can bill for whatever the surviving chunks already cost."""
    fake = FakeDecider(handler=lambda s, q: DecisionRetryableError("busy"))
    with pytest.raises(ScreenFailed) as caught:
        await Screener(fake, QUESTIONS, TEMPLATES).screen(SOURCE, "", _cands("glucose"))
    assert isinstance(caught.value.cause, DecisionRetryableError)
    assert caught.value.usage.calls == 0


async def test_an_oversized_chunk_is_split_with_a_note():
    fake = FakeDecider()
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=3, max_state_chars=80)
    outcome = await screener.screen(SOURCE, "", _cands("glucose", "fructose", "sucrose"))
    assert outcome.chunks >= 2
    assert any("split" in note for note in outcome.notes)


async def test_a_failing_chunk_reports_the_usage_its_siblings_already_cost():
    """The sibling chunk was billed before this one failed; the caller must see it."""

    def second_chunk_fails(state, questions):
        if "n_C003" in questions:
            return DecisionRetryableError("busy")
        return overlap_handler(state, questions)

    fake = FakeDecider(handler=second_chunk_fails)
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=2)
    with pytest.raises(ScreenFailed) as caught:
        await screener.screen(SOURCE, "", _cands("glucose", "fructose", "sucrose"))
    assert caught.value.usage.calls == 1
    assert isinstance(caught.value.cause, DecisionRetryableError)
    assert "busy" in str(caught.value)


async def test_a_fatal_chunk_error_still_propagates_as_itself():
    def second_chunk_is_fatal(state, questions):
        if "n_C003" in questions:
            return DecisionFatalError("bad key")
        return overlap_handler(state, questions)

    screener = Screener(
        FakeDecider(handler=second_chunk_is_fatal), QUESTIONS, TEMPLATES, chunk_size=2
    )
    with pytest.raises(DecisionFatalError):
        await screener.screen(SOURCE, "", _cands("glucose", "fructose", "sucrose"))


async def test_a_failing_chunk_cancels_the_siblings_still_in_flight():
    """Once one chunk has failed the rest are wasted spend, so they are cancelled."""
    cancelled: list[str] = []

    class SlowDecider(FakeDecider):
        async def decide(self, state, questions):
            if "n_C001" in questions:
                return await super().decide(state, questions)
            if "n_C002" in questions:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    cancelled.append("C002")
                    raise
            raise DecisionRetryableError("busy")

    screener = Screener(SlowDecider(), QUESTIONS, TEMPLATES, chunk_size=1)
    with pytest.raises(ScreenFailed):
        await screener.screen(SOURCE, "", _cands("glucose", "fructose", "sucrose"))
    assert cancelled == ["C002"]


async def test_a_single_candidate_over_the_cap_is_sent_with_a_note():
    fake = FakeDecider()
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=1, max_state_chars=10)
    outcome = await screener.screen(SOURCE, "", _cands("glucose"))
    assert outcome.chunks == 1 and len(fake.calls) == 1
    assert any("exceeds" in note for note in outcome.notes)


async def test_the_state_cap_counts_the_rules_block():
    """The rules travel in every chunk's state, so the split must budget for them."""
    fake = FakeDecider()
    with_rules = len(json.dumps(QUESTIONS.rules_state()))
    source = len(json.dumps(build_source_state(SOURCE, "")))
    # Room for the source and both candidates, but not once the rules are counted too.
    cap = source + 2 * 40 + with_rules // 2
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=2, max_state_chars=cap)
    outcome = await screener.screen(SOURCE, "", _cands("glucose", "fructose"))
    assert outcome.chunks == 2
    assert any("split" in note for note in outcome.notes)


async def test_cancelling_the_screen_cancels_the_chunks_in_flight():
    """A cancelled screen() must not leave its chunk tasks running and billing."""
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled: list[str] = []

    class BlockingDecider(FakeDecider):
        async def decide(self, state, questions):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.extend(state["candidates"])
                raise
            return await super().decide(state, questions)

    screener = Screener(BlockingDecider(), QUESTIONS, TEMPLATES, chunk_size=1)
    outer = asyncio.create_task(screener.screen(SOURCE, "", _cands("glucose", "fructose")))
    await started.wait()
    await asyncio.sleep(0)  # let the second chunk start as well
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert sorted(cancelled) == ["C001", "C002"]
    leftovers = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert leftovers == []


REF_ZIVILA_SLOTS = Path(__file__).parent.parent / "examples/ref_zivila/jobs/foodon/slots.yaml"


def legacy_screen_question(questions: QuestionSet, key: str) -> Noul:
    """The version-1 screen question, which repeated the whole preamble per candidate."""
    s = questions.slots
    rules = " ".join(rule.strip() for rule in s.hard_rules)
    return Noul(
        instructions=(
            f"Does `candidates.{key}` denote the same entity as the {s.entity_noun} "
            f"described in `source`? The candidate is a {s.target_noun}. "
            f"Domain: {s.domain_brief.strip()} {rules}"
        ).strip(),
        criteria={
            "true": (
                f"`candidates.{key}` denotes the same entity as `source`, with every "
                "identity-bearing property that either side states compatible"
            ),
            "false": (
                f"`candidates.{key}` denotes a different entity, a broader or narrower "
                "one, or a related concept that is not the same entity"
            ),
        },
    )


def _request(state, questions) -> str:
    """The body the Jev client posts, minus the model name both versions share."""
    return json.dumps(
        {
            "state": state,
            "questions": {name: question_to_dict(q) for name, q in questions.items()},
        },
        ensure_ascii=False,
    )


async def test_a_fifty_candidate_request_is_less_than_half_the_version_one_size():
    questions = QuestionSet.from_slots(load_slots(REF_ZIVILA_SLOTS))
    source = Record(
        id="1042",
        fields={
            "mention_en": "Pineapple, canned in light syrup",
            "name_slo": "Ananas, konzerviran v lahkem sirupu",
            "fgnm": "Fruit and fruit products",
        },
    )
    context = "English name: Pineapple, canned in light syrup\nFood group: Fruit"
    candidates = [
        Candidate(
            record=Record(
                id=f"FOODON_0330{i:04d}",
                fields={
                    "label": f"pineapple food product {i} (canned, in syrup)",
                    "synonyms": f"ananas {i}; canned pineapple {i}",
                    "definition": "A food product made from the fruit of Ananas comosus.",
                },
            ),
            fused_score=1.0 / i,
            evidence=(),
        )
        for i in range(1, 51)
    ]
    templates = TemplateSet(
        query="{{ mention_en }}",
        context="{{ mention_en }}",
        doc="{{ label }}",
        candidate=(
            "ID: {{ id }}\nLabel: {{ label }}\nSynonyms: {{ synonyms }}\n"
            "Definition: {{ definition }}"
        ),
    )
    fake = FakeDecider()
    await Screener(fake, questions, templates, chunk_size=50).screen(source, context, candidates)
    assert len(fake.calls) == 1
    state, new_questions = fake.calls[0]
    new = _request(state, new_questions)

    old_state = {"source": state["source"], "candidates": state["candidates"]}
    old_questions = {
        f"n_{key}": legacy_screen_question(questions, key) for key in old_state["candidates"]
    }
    old = _request(old_state, old_questions)

    assert len(new) <= 0.45 * len(old), f"new/old = {len(new) / len(old):.3f}"

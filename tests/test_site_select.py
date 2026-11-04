"""The NER site's selection step: Jev or an LLM picks one retrieved candidate per entity.

Offline: FakeDecider, FakeLLM and in-memory stores. Plan: docs/ner-site/PLAN.md section 13.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest
from ner_site.indexes import OntologyIndex
from ner_site.schema import EntityResult
from ner_site.select import JEV, RETRIEVAL_ONLY, SelectStats, select_entities

from xwalk.decide.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)
from xwalk.decide.fake import FakeDecider
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stores.memory import MemoryStore

TEXT = "Patients took aspirin daily."
ASPIRIN = Record(
    "CHEBI:15365",
    {"label": "aspirin", "synonyms": ["acetylsalicylic acid"], "definition": "a drug"},
)
PARACETAMOL = Record("CHEBI:46195", {"label": "paracetamol", "synonyms": [], "definition": ""})


def _entity(**overrides: Any) -> EntityResult:
    fields: dict[str, Any] = {
        "id": "e1",
        "text": "aspirin",
        "start": 14,
        "end": 21,
        "type": "chemical",
        "ontology": "chebi",
        "matches": [],
        "notes": [],
    }
    return EntityResult(**{**fields, **overrides})


def _setup() -> tuple[dict[str, OntologyIndex], dict[str, list[Candidate]]]:
    store = MemoryStore.from_source([ASPIRIN, PARACETAMOL])
    index = OntologyIndex("chebi", store, [], {})
    # Retrieval ranked the wrong term first: the selector is what fixes that.
    ranked = [PARACETAMOL, ASPIRIN]
    candidates = [
        Candidate(r, 1.0 / (i + 1), (RetrievalHit(r.id, "bm25", None, i + 1),))
        for i, r in enumerate(ranked)
    ]
    return {"chebi": index}, {"e1": candidates}


def _run(selector: str, **kw: Any) -> tuple[list[EntityResult], SelectStats]:
    indexes, candidates = _setup()
    stats = SelectStats()
    results = asyncio.run(
        select_entities(
            kw.pop("results", [_entity()]),
            kw.pop("candidates", candidates),
            indexes,
            selector=selector,
            text=TEXT,
            stats=stats,
            **kw,
        )
    )
    return results, stats


def _knows_aspirin(state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
    """A scripted Jev: yes to the aspirin record, no to anything else, confident."""
    candidates: Mapping[str, Any] = state.get("candidates", {})
    right = {key for key, value in candidates.items() if "CHEBI:15365" in json.dumps(value)}
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if isinstance(question, Noul):
            text = json.dumps(question.instructions)
            refers = [key for key in candidates if key in text]
            yes = not refers or any(key in right for key in refers)
            answers[name] = NoulAnswer(noul=0.95 if yes else 0.05)
        elif isinstance(question, Choice):
            options = [k for k in question.criteria if k != "NONE"]
            best = next((k for k in options if k in right), "NONE")
            probabilities = {k: (0.95 if k == best else 0.0) for k in question.criteria}
            probabilities["NONE"] = 0.05 if best != "NONE" else 1.0
            answers[name] = ChoiceAnswer(best, probabilities, probabilities[best])
        elif isinstance(question, Score):
            top = len(question.criteria) - 1
            answers[name] = ScoreAnswer(
                score=top,
                probabilities={str(i): float(i == top) for i in range(top + 1)},
                confidence=0.95,
                legend={str(i): str(c) for i, c in enumerate(question.criteria)},
            )
    return answers


def _key_of(request: LLMRequest, record_id: str) -> str:
    found = re.search(r"\[(C\d+)\] ID: " + re.escape(record_id), request.user)
    assert found, request.user
    return found.group(1)


def test_jev_picks_the_right_candidate_over_the_retrieval_ranking() -> None:
    decider = FakeDecider(handler=_knows_aspirin)
    (result,), stats = _run(JEV, decider=decider)
    assert result.pick is not None
    assert result.pick.selector == JEV
    assert result.pick.id == "CHEBI:15365" and result.pick.label == "aspirin"
    assert result.pick.status in {"match", "review"}
    assert result.pick.url and "CHEBI:15365" in result.pick.url
    assert stats.calls == len(decider.calls) >= 2
    # Jev saw the mention in its sentence, not the bare word.
    assert any("took" in json.dumps(state) for state, _ in decider.calls)


def test_an_llm_selector_resolves_only_an_issued_key() -> None:
    llm = FakeLLM(
        handler=lambda r: json.dumps(
            {"chosen_key": _key_of(r, "CHEBI:15365"), "confidence_score": 0.9, "explanation": "x"}
        )
    )
    (result,), stats = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None and result.pick.selector == "openai/gpt-6-luna"
    assert result.pick.status == "match" and result.pick.id == "CHEBI:15365"
    assert result.pick.confidence == pytest.approx(0.9)
    assert stats.calls == 1 and stats.prompt_tokens == 100 and stats.completion_tokens == 20


def test_an_unissued_key_never_becomes_an_id() -> None:
    llm = FakeLLM(['{"chosen_key": "CHEBI:15365", "confidence_score": 0.99, "explanation": ""}'])
    (result,), _ = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None
    assert result.pick.id is None and result.pick.status == "failed"


def test_an_llm_abstention_is_no_match() -> None:
    llm = FakeLLM(['{"chosen_key": null, "confidence_score": 0.8, "explanation": "none fit"}'])
    (result,), _ = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None
    assert result.pick.status == "no_match" and result.pick.id is None
    assert result.pick.explanation == "none fit"


def test_retrieval_only_and_unrouted_entities_call_nothing() -> None:
    decider = FakeDecider()
    (result,), stats = _run(RETRIEVAL_ONLY, decider=decider)
    assert result.pick is None and stats.calls == 0
    other = _entity(id="e2", type="other", ontology=None)
    (result,), stats = _run(JEV, decider=decider, results=[other], candidates={})
    assert result.pick is None and stats.calls == 0 and not decider.calls


class _SlowDecider:
    model = "slow"
    fingerprint = "slow"

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        await asyncio.sleep(5)
        raise AssertionError("not reached")


def test_the_deadline_cuts_a_slow_selector_off() -> None:
    (result,), _ = _run(JEV, decider=_SlowDecider(), deadline=0.05)
    assert result.pick is not None and result.pick.status == "failed"
    assert result.pick.id is None and "deadline" in result.pick.explanation


def test_a_failing_decider_is_a_failed_pick_not_an_exception() -> None:
    decider = FakeDecider(handler=lambda state, questions: RuntimeError("boom"))
    (result,), _ = _run(JEV, decider=decider)
    assert result.pick is not None and result.pick.status == "failed" and result.pick.id is None

"""A scripted decision client. The reason the whole decider loop is testable offline."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

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
from xwalk.fingerprint import hash_value
from xwalk.records import Usage

Handler = Callable[[Any, Mapping[str, Question]], "Mapping[str, Answer] | BaseException"]

_KEY_REF = re.compile(r"candidates\.([A-Za-z0-9_]+)")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return set(_TOKEN.findall(text.lower()))


def _overlap(source: Any, candidate: Any) -> float:
    a, b = _tokens(source), _tokens(candidate)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_handler(state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
    """Token-overlap scoring: enough to drive the loop, never a model of the domain."""
    source = state.get("source") if isinstance(state, Mapping) else state
    candidates: Mapping[str, Any] = (
        state.get("candidates", {}) if isinstance(state, Mapping) else {}
    )
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if isinstance(question, Noul):
            instructions = question.instructions
            text = instructions if isinstance(instructions, str) else json.dumps(instructions)
            found = _KEY_REF.search(text)
            target = candidates.get(found.group(1)) if found else state
            answers[name] = NoulAnswer(noul=round(_overlap(source, target), 4))
        elif isinstance(question, Choice):
            scores = {
                key: _overlap(source, candidates.get(key, description))
                for key, description in question.criteria.items()
                if key != "NONE"
            }
            best = max(scores, key=lambda k: (scores[k], k)) if scores else "NONE"
            if scores and scores[best] == 0.0 and "NONE" in question.criteria:
                best = "NONE"
            total = sum(scores.values())
            probabilities = (
                {k: (v / total if total else 0.0) for k, v in scores.items()}
                if best != "NONE"
                else {k: 0.0 for k in scores}
            )
            if "NONE" in question.criteria:
                probabilities["NONE"] = 1.0 - sum(probabilities.values())
            answers[name] = ChoiceAnswer(
                choice=best,
                probabilities=probabilities,
                confidence=probabilities.get(best, 0.0),
            )
        elif isinstance(question, Score):
            levels = len(question.criteria)
            middle = (levels - 1) / 2
            answers[name] = ScoreAnswer(
                score=middle,
                probabilities={str(i): (1.0 if i == int(middle) else 0.0) for i in range(levels)},
                confidence=0.5,
                legend={str(i): str(level) for i, level in enumerate(question.criteria)},
            )
    return answers


class FakeDecider:
    def __init__(
        self,
        handler: Handler | None = None,
        *,
        model: str = "fake-decider",
        prompt_tokens: int = 100,
    ) -> None:
        self._handler: Handler = handler or overlap_handler
        self._model = model
        self._prompt_tokens = prompt_tokens
        self.calls: list[tuple[Any, Mapping[str, Question]]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def fingerprint(self) -> str:
        return hash_value({"adapter": "fake-decider", "model": self._model})

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        self.calls.append((state, dict(questions)))
        outcome = self._handler(state, questions)
        if isinstance(outcome, BaseException):
            raise outcome
        return DecisionResponse(
            answers=dict(outcome),
            model=self._model,
            usage=Usage(prompt_tokens=self._prompt_tokens, completion_tokens=0, calls=1),
        )

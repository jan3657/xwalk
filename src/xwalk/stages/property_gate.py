"""Gate: the absolute question about the one chosen record, with a clean state.

Returns the rubric level and one agreement probability per declared property. Those
per-property numbers are what replaces a generated explanation for a reviewer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from xwalk.decide.base import DecisionClient, NoulAnswer, Question, ScoreAnswer
from xwalk.decide.questions import QuestionSet
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.screen import build_source_state
from xwalk.templates import TemplateSet

_PREFIX = "prop_"


@dataclass(frozen=True)
class GateOutcome:
    rubric_score: float
    rubric_confidence: float
    rubric_levels: int
    properties: Mapping[str, float]
    raw: str
    usage: Usage
    model: str


class PropertyGate:
    def __init__(
        self, decider: DecisionClient, questions: QuestionSet, templates: TemplateSet
    ) -> None:
        self._decider = decider
        self._questions = questions
        self._templates = templates

    async def gate(self, source: Record, context: str, chosen: Candidate) -> GateOutcome:
        state = {
            "source": build_source_state(source, context),
            "candidate": self._templates.render_candidate(chosen.record),
        }
        questions: dict[str, Question] = {"rubric": self._questions.rubric_question()}
        questions.update(self._questions.property_questions())
        response = await self._decider.decide(state, questions)

        rubric = response.answers["rubric"]
        assert isinstance(rubric, ScoreAnswer)
        properties: dict[str, float] = {}
        for name, answer in response.answers.items():
            if name.startswith(_PREFIX):
                assert isinstance(answer, NoulAnswer)
                properties[name[len(_PREFIX) :]] = answer.noul

        raw = json.dumps(
            {
                "model": response.model,
                "rubric": {
                    "score": rubric.score,
                    "confidence": rubric.confidence,
                    "probabilities": dict(rubric.probabilities),
                    "legend": dict(rubric.legend),
                },
                "properties": properties,
            },
            ensure_ascii=False,
        )
        return GateOutcome(
            rubric_score=rubric.score,
            rubric_confidence=rubric.confidence,
            rubric_levels=self._questions.rubric_levels,
            properties=properties,
            raw=raw,
            usage=response.usage,
            model=response.model,
        )

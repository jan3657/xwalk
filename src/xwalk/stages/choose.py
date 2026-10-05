"""Choose: the comparative question over the screen's survivors.

Near-duplicate ontology labels split a Choice's probability mass, which is why this
runs only over the shortlist and why the screen probability, not `p_choice`, is the
calibrated number the policy accepts on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.decide.base import ChoiceAnswer, DecisionClient
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.keying import Resolution
from xwalk.stages.screen import build_source_state
from xwalk.templates import TemplateSet


@dataclass(frozen=True)
class ChooseOutcome:
    record_id: str | None
    resolution: Resolution
    p_choice: float
    p_none: float
    confidence: float
    raw: str
    usage: Usage
    model: str


class Chooser:
    def __init__(
        self, decider: DecisionClient, questions: QuestionSet, templates: TemplateSet
    ) -> None:
        self._decider = decider
        self._questions = questions
        self._templates = templates

    async def choose(
        self, source: Record, context: str, shortlist: Sequence[Candidate]
    ) -> ChooseOutcome:
        if not shortlist:
            return ChooseOutcome(
                None, Resolution.ABSTAIN, 1.0, 1.0, 1.0, "", Usage.zero(), self._decider.model
            )
        width = max(2, len(str(len(shortlist))))
        keyed = {
            f"C{i:0{width}d}": self._templates.render_candidate(c.record)
            for i, c in enumerate(shortlist, start=1)
        }
        issued = {key: c.id for key, c in zip(keyed, shortlist, strict=True)}
        state = {"source": build_source_state(source, context), "candidates": keyed}
        question = self._questions.choose_question(keyed)
        response = await self._decider.decide(state, {"best": question})
        answer = response.answers["best"]
        assert isinstance(answer, ChoiceAnswer)
        raw = json.dumps(
            {
                "model": response.model,
                "choice": answer.choice,
                "probabilities": dict(answer.probabilities),
                "confidence": answer.confidence,
            },
            ensure_ascii=False,
        )
        p_none = answer.probabilities.get(NONE_KEY, 0.0)
        if answer.choice == NONE_KEY:
            return ChooseOutcome(
                None,
                Resolution.ABSTAIN,
                p_none,
                p_none,
                answer.confidence,
                raw,
                response.usage,
                response.model,
            )
        if answer.choice not in issued:
            # Unreachable through parse_response, kept because the invariant lives here.
            return ChooseOutcome(
                None,
                Resolution.UNRESOLVED,
                0.0,
                p_none,
                answer.confidence,
                raw,
                response.usage,
                response.model,
            )
        return ChooseOutcome(
            issued[answer.choice],
            Resolution.EXACT_KEY,
            answer.probabilities.get(answer.choice, 0.0),
            p_none,
            answer.confidence,
            raw,
            response.usage,
            response.model,
        )

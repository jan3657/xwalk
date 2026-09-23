"""Questions for a decision model, composed from the same slots the LLM prompts use.

Jev is literal. Every instruction states the exact condition, names the state fields it
refers to in backticks, and spells out what true and false mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from xwalk.decide.base import Choice, Noul, Score
from xwalk.fingerprint import hash_value
from xwalk.prompts.contract import PromptSlots

NONE_KEY = "NONE"
_MAX_LEVELS = 10
_MIN_LEVELS = 2
_SCREEN_SAME = (
    "the candidate denotes the same entity as `source`, with every identity-bearing property "
    "that either side states compatible"
)
_SCREEN_DIFFERENT = (
    "the candidate denotes a different entity, a broader or narrower one, or a related "
    "concept that is not the same entity"
)


@dataclass(frozen=True)
class QuestionSet:
    slots: PromptSlots

    @classmethod
    def from_slots(cls, slots: PromptSlots) -> QuestionSet:
        return cls(slots=slots)

    @property
    def rubric_levels(self) -> int:
        return len(self.slots.rubric)

    def _rules(self) -> str:
        return " ".join(rule.strip() for rule in self.slots.hard_rules)

    def rules_state(self) -> dict[str, Any]:
        """The screen's preamble, sent once per chunk in the state rather than per candidate."""
        s = self.slots
        return {
            "entity": s.entity_noun,
            "target": s.target_noun,
            "domain": s.domain_brief,
            "hard_rules": list(s.hard_rules),
            "same": _SCREEN_SAME,
            "different": _SCREEN_DIFFERENT,
        }

    def screen_question(self, key: str) -> Noul:
        return Noul(
            instructions=(
                f"Does `candidates.{key}` denote the same entity as the `rules.entity` "
                "described in `source`? The candidate is a `rules.target` from the domain in "
                "`rules.domain`. Apply every rule in `rules.hard_rules`. True means "
                "`rules.same`; false means `rules.different`."
            )
        )

    def choose_question(self, criteria: Mapping[str, str]) -> Choice:
        s = self.slots
        options: dict[str, str | None] = dict(criteria)
        options[NONE_KEY] = (
            f"no candidate denotes the same entity as the {s.entity_noun} in `source`"
        )
        steps = s.disambiguation_steps.strip()
        return Choice(
            instructions=(
                f"Among `candidates`, select the most specific {s.target_noun} that denotes the "
                f"same entity as the {s.entity_noun} in `source` and whose stated properties the "
                f"source supports. Do not infer a property the source does not state. "
                f"{steps} {self._rules()} If no candidate denotes the same entity, choose "
                f"{NONE_KEY}."
            ).strip(),
            criteria=options,
        )

    def rubric_question(self) -> Score:
        rows = sorted(self.slots.rubric, key=lambda r: r.score)
        if not _MIN_LEVELS <= len(rows) <= _MAX_LEVELS:
            raise ValueError(
                f"a decider rubric needs {_MIN_LEVELS} to {_MAX_LEVELS} rows, got {len(rows)}"
            )
        s = self.slots
        return Score(
            instructions=(
                f"Rate how well `candidate` denotes the same entity as the {s.entity_noun} in "
                f"`source`, using the levels in order from worst to best. {self._rules()}"
            ).strip(),
            criteria=[f"{row.name}: {row.when.strip()}" for row in rows],
        )

    def property_questions(self) -> dict[str, Noul]:
        return {
            f"prop_{p.name}": Noul(
                instructions=(
                    f"Considering `source` and `candidate`: {p.question.strip()} Answer about "
                    "the two records as given; if neither states the property, answer yes."
                ),
                criteria={
                    "true": "the two records agree on this property, or neither states it",
                    "false": "the two records disagree on this property",
                },
            )
            for p in self.slots.properties
        }

    @property
    def fingerprint(self) -> str:
        return hash_value({"questions_version": 2, "slots": self.slots.model_dump(mode="json")})

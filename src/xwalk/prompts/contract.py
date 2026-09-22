"""Prompt skeletons plus the contract that keeps them safe to modify.

The skeleton owns the fixed structure. Slots own domain content. Everything that could
break the machine-readable contract lives in the skeleton, so an LLM-authored or
optimizer-mutated slots file can produce a poor rubric but never a broken prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field, field_validator, model_validator

from xwalk.fingerprint import hash_value
from xwalk.llm.parsing import parse_json_object

BASE_DIR = Path(__file__).parent / "base"

SELECT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "chosen_key": {"type": ["string", "null"]},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
    },
    "required": ["chosen_key", "confidence_score", "explanation"],
    "additionalProperties": False,
}

SCORE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
        "better_candidate_keys": {"type": "array", "items": {"type": "string"}},
        "better_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confidence_score", "explanation"],
    "additionalProperties": False,
}

VERIFY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["support", "disagree", "no_match"]},
        "preferred_key": {"type": ["string", "null"]},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
    },
    "required": ["decision", "explanation"],
    "additionalProperties": False,
}

REWRITE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["queries"],
    "additionalProperties": False,
}


class ContractError(Exception):
    """A skeleton no longer satisfies the machine-readable contract."""


class RubricRow(BaseModel):
    score: float
    name: str
    when: str
    example: str = ""


class PropertyQuestion(BaseModel):
    """One identity-bearing property the decider path checks on the chosen candidate."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str = Field(min_length=1)


class PromptSlots(BaseModel):
    entity_noun: str
    target_noun: str
    domain_brief: str
    rubric: list[RubricRow] = Field(min_length=1)
    hard_rules: list[str] = Field(default_factory=list)
    disambiguation_steps: str = ""
    # Used only by the decider path; the LLM skeletons ignore it.
    properties: list[PropertyQuestion] = Field(default_factory=list)

    @field_validator("rubric")
    @classmethod
    def _check_rubric(cls, rows: list[RubricRow]) -> list[RubricRow]:
        if len(rows) < 2:
            raise ValueError("rubric needs at least 2 rows to be usable")
        for row in rows:
            if not 0.0 <= row.score <= 1.0:
                raise ValueError(f"rubric score {row.score} must be between 0 and 1")
        scores = [row.score for row in rows]
        # Lengths differ by one on purpose: each row is compared with its successor.
        if any(b >= a for a, b in zip(scores, scores[1:], strict=False)):
            raise ValueError("rubric scores must be strictly decreasing as declared")
        return rows

    @model_validator(mode="after")
    def _check_nouns(self) -> PromptSlots:
        for name in ("entity_noun", "target_noun", "domain_brief"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        return self


def load_slots(path: str | Path) -> PromptSlots:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PromptSlots.model_validate(data)


@dataclass(frozen=True)
class PromptSet:
    slots: PromptSlots
    skeletons: Mapping[str, str]

    @classmethod
    def from_slots(cls, slots: PromptSlots, *, base_dir: Path | None = None) -> PromptSet:
        directory = base_dir or BASE_DIR
        skeletons = {
            name: (directory / f"{name}.j2").read_text(encoding="utf-8")
            for name in ("select", "score", "verify", "rewrite")
        }
        return cls(slots=slots, skeletons=skeletons)

    def _render(self, name: str, **variables: Any) -> str:
        env = Environment(
            loader=FileSystemLoader(str(BASE_DIR)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )
        template = env.from_string(self.skeletons[name])
        return template.render(slots=self.slots, **variables).strip()

    def render_select(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        candidate_block: str,
    ) -> str:
        return self._render(
            "select",
            source_fields=source_fields,
            context=context,
            candidate_block=candidate_block,
        )

    def render_score(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        chosen_block: str,
        other_candidates: str,
        review_floor: float,
    ) -> str:
        return self._render(
            "score",
            source_fields=source_fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
            review_floor=review_floor,
        )

    def render_verify(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        chosen_block: str,
        other_candidates: str,
    ) -> str:
        return self._render(
            "verify",
            source_fields=source_fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
        )

    def render_rewrite(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        previous_queries: Sequence[str],
        best_candidates: str,
        max_queries: int,
    ) -> str:
        return self._render(
            "rewrite",
            source_fields=source_fields,
            context=context,
            previous_queries=list(previous_queries),
            best_candidates=best_candidates,
            max_queries=max_queries,
        )

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "slots": json.loads(self.slots.model_dump_json()),
                "skeletons": dict(sorted(self.skeletons.items())),
            }
        )


# Section headings that must be present, and present exactly once. A skeleton that
# renders "## Candidates" twice would show the model two candidate lists.
_REQUIRED_SECTIONS: Mapping[str, tuple[str, ...]] = {
    "select": ("## Source record", "## Candidates", "## Output"),
    "score": ("## Source record", "## Rubric", "## Output"),
    "verify": ("## Source record", "## Proposed match", "## Output"),
    "rewrite": ("## Source record", "## Output"),
}

# The instruction that makes opaque keys safe. If an optimizer deletes it, the model is
# free to answer with an identifier and every answer becomes UNRESOLVED_OUTPUT.
_KEY_INSTRUCTION = "Never answer with an identifier"

# Typed individually rather than as one heterogeneous dict: `dict(a="s", b=0.4, c=2)`
# infers `dict[str, object]`, and every use site then fails `mypy --strict` on arg-type.
_SOURCE_FIELDS: Mapping[str, Any] = {"mention": "glucose", "organism": "Homo sapiens"}
_CONTEXT = "blood [glucose] levels were elevated"
_CANDIDATE_BLOCK = "[C01] ID: T1 Label: glucose\n\n[C02] ID: T2 Label: fructose"
_CHOSEN_BLOCK = "[C01] ID: T1 Label: glucose"
_OTHER_CANDIDATES = "[C02] ID: T2 Label: fructose"
_REVIEW_FLOOR = 0.4
_PREVIOUS_QUERIES: Sequence[str] = ("glucose",)
_BEST_CANDIDATES = "[C01] ID: T1"
_MAX_QUERIES = 2

# Rendered inputs that must survive into each prompt, exactly once.
_REQUIRED_INPUTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "select": (("candidate_block", _CANDIDATE_BLOCK), ("context", _CONTEXT)),
    "score": (("chosen_block", _CHOSEN_BLOCK), ("context", _CONTEXT)),
    "verify": (("chosen_block", _CHOSEN_BLOCK), ("context", _CONTEXT)),
    "rewrite": (("context", _CONTEXT),),
}

_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "select": SELECT_SCHEMA,
    "score": SCORE_SCHEMA,
    "verify": VERIFY_SCHEMA,
    "rewrite": REWRITE_SCHEMA,
}


def validate_contract(prompts: PromptSet) -> None:
    """Render every skeleton and assert the machine-readable contract survived.

    Run this before anything is saved — a drafting model or an optimizer must not be
    able to ship a prompt whose output cannot be parsed, whose input blocks went
    missing or got duplicated, or whose key instruction was edited away.

    Order matters: the rendered-input check runs first so that a skeleton which dropped
    a block reports *that*, rather than a downstream missing-heading error.
    """
    rendered = {
        "select": prompts.render_select(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            candidate_block=_CANDIDATE_BLOCK,
        ),
        "score": prompts.render_score(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            chosen_block=_CHOSEN_BLOCK,
            other_candidates=_OTHER_CANDIDATES,
            review_floor=_REVIEW_FLOOR,
        ),
        "verify": prompts.render_verify(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            chosen_block=_CHOSEN_BLOCK,
            other_candidates=_OTHER_CANDIDATES,
        ),
        "rewrite": prompts.render_rewrite(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            previous_queries=_PREVIOUS_QUERIES,
            best_candidates=_BEST_CANDIDATES,
            max_queries=_MAX_QUERIES,
        ),
    }

    for name, text in rendered.items():
        for label, block in _REQUIRED_INPUTS[name]:
            count = text.count(block)
            if count != 1:
                raise ContractError(
                    f"{name} prompt rendered {label} {count} times; it must appear exactly once"
                )

        for section in _REQUIRED_SECTIONS[name]:
            count = text.count(section)
            if count != 1:
                raise ContractError(
                    f"{name} prompt contains section {section!r} {count} times; "
                    f"it must appear exactly once"
                )

        if name == "select" and _KEY_INSTRUCTION not in text:
            raise ContractError(
                "select prompt lost the key instruction "
                f"({_KEY_INSTRUCTION!r}); without it the model may answer with an identifier"
            )

        tail = text.split("## Output", 1)[1]
        try:
            example = parse_json_object(tail)
        except Exception as exc:  # noqa: BLE001
            raise ContractError(f"{name} output block is not a parseable example: {exc}") from exc

        required = _SCHEMAS[name].get("required", [])
        missing = [key for key in required if key not in example]
        if missing:
            raise ContractError(f"{name} output example is missing keys: {missing}")

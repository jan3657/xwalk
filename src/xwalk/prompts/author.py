"""Drafting domain slots with an LLM.

The model returns **slots only** -- never prompt text. Phase 1's skeleton owns every part
of the machine-readable contract, so a bad draft can produce a poor rubric but never a
broken prompt. `validate_contract` runs before anything reaches disk.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.prompts.contract import PromptSet, PromptSlots, validate_contract
from xwalk.records import Record, Usage

SLOTS_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "entity_noun": {"type": "string"},
        "target_noun": {"type": "string"},
        "domain_brief": {"type": "string"},
        "rubric": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "name": {"type": "string"},
                    "when": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["score", "name", "when"],
            },
        },
        "hard_rules": {"type": "array", "items": {"type": "string"}},
        "disambiguation_steps": {"type": "string"},
        "properties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                    "question": {"type": "string"},
                },
                "required": ["name", "question"],
            },
        },
    },
    "required": ["entity_noun", "target_noun", "domain_brief", "rubric"],
}

_SYSTEM = (
    "You design the domain-specific vocabulary for a record-matching system. "
    "You return JSON only. No prose, no code fences."
)

_INSTRUCTIONS = """\
Describe the matching task below as a set of slots.

- entity_noun: what one source record is, as a noun phrase ("chemical entity mention")
- target_noun: what one target record is ("ChEBI ontology term")
- domain_brief: one clause naming the domain and its naming conventions
- rubric: 3 to 5 confidence bands, scores strictly decreasing, each with a short `when`
  condition and a concrete `example` drawn from the samples where possible
- hard_rules: absolute rules a careful domain expert would insist on; omit if none apply
- disambiguation_steps: free text, only if this domain has a specific procedure
  (for example, using surrounding context to decide the organism for a gene symbol)
- properties: optional, used only by the decider path -- the identity-bearing properties
  a match must agree on, each a snake_case `name` and a yes/no `question` about the pair
  (for example, `preparation`: "Do both describe the same preparation state?")

Write for a careful annotator who knows nothing about this domain."""


def _sample_block(records: Sequence[Record], limit: int) -> str:
    lines = []
    for record in list(records)[:limit]:
        rendered = "; ".join(f"{k}={v}" for k, v in record.fields.items())
        lines.append(f"- {record.id}: {rendered}")
    return "\n".join(lines)


@dataclass(frozen=True)
class DraftResult:
    slots: PromptSlots
    warnings: tuple[str, ...]
    usage: Usage
    raw: str


def _repair(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fix what is mechanically fixable; everything else raises through PromptSlots."""
    warnings: list[str] = []
    rows = payload.get("rubric") or []
    if not isinstance(rows, list):
        return payload, warnings

    clamped: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            clamped.append(row)
            continue
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            # A missing or non-numeric score is not mechanically fixable; leave it for
            # PromptSlots to reject by name rather than crashing the sort below.
            raise ValueError(f"rubric row {row.get('name', '?')!r} has no usable numeric score")
        if not 0.0 <= float(score) <= 1.0:
            warnings.append(f"clamped rubric score {score} into [0, 1]")
            row = {**row, "score": max(0.0, min(1.0, float(score)))}
        clamped.append(row)

    scores = [float(r["score"]) for r in clamped if isinstance(r, dict)]
    if len(scores) == len(clamped) and scores != sorted(scores, reverse=True):
        warnings.append("rubric was not in strictly decreasing score order; reordered")
        clamped = sorted(clamped, key=lambda r: -float(r["score"]))

    return {**payload, "rubric": clamped}, warnings


async def draft_slots(
    llm: LLMClient,
    *,
    description: str,
    source_samples: Sequence[Record],
    target_samples: Sequence[Record],
    existing: PromptSlots | None = None,
    max_samples: int = 8,
    max_tokens: int = 2048,
) -> DraftResult:
    parts = [
        _INSTRUCTIONS,
        f"\n## The task\n{description}",
        f"\n## Sample source records\n{_sample_block(source_samples, max_samples)}",
        f"\n## Sample target records\n{_sample_block(target_samples, max_samples)}",
    ]
    if existing is not None:
        parts.append(
            "\n## Current slots, to refine rather than replace\n"
            + json.dumps(json.loads(existing.model_dump_json()), indent=2)
        )

    response = await llm.complete(
        LLMRequest(
            system=_SYSTEM,
            user="\n".join(parts),
            schema=SLOTS_SCHEMA,
            schema_name="slots",
            max_tokens=max_tokens,
        )
    )

    try:
        payload = parse_json_object(response.text)
    except ParseError as exc:
        raise ValueError(
            f"could not read slots from the drafting model: {exc}\n"
            f"raw response:\n{response.text[:1000]}"
        ) from exc

    repaired, warnings = _repair(payload)
    slots = PromptSlots.model_validate(repaired)  # raises with a clear message on bad input
    validate_contract(PromptSet.from_slots(slots))  # never write a prompt that cannot parse

    return DraftResult(
        slots=slots, warnings=tuple(warnings), usage=response.usage, raw=response.text
    )


def slots_diff(before: PromptSlots, after: PromptSlots) -> str:
    """A field-level diff, shown before anything is written."""
    old = json.loads(before.model_dump_json())
    new = json.loads(after.model_dump_json())
    lines: list[str] = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) == new.get(key):
            continue
        lines.append(f"- {key}:")
        lines.append(f"    before: {json.dumps(old.get(key), ensure_ascii=False)}")
        lines.append(f"    after:  {json.dumps(new.get(key), ensure_ascii=False)}")
    return "\n".join(lines)


def write_slots(slots: PromptSlots, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(json.loads(slots.model_dump_json()), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

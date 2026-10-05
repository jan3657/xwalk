"""Selection: show the model a keyed candidate list, get back one key or null."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_confidence, parse_json_object
from xwalk.prompts.contract import SELECT_SCHEMA, PromptSet
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.keying import (
    KeyedCandidates,
    Resolution,
    ResolvedChoice,
    assign_keys,
    resolve_key,
)
from xwalk.templates import TemplateSet

_CHARS_PER_TOKEN = 4  # a deliberately crude estimate; the budget is a guard rail, not a meter


@dataclass(frozen=True)
class SelectorPolicy:
    """How much of the fused candidate list reaches the model.

    This is not retrieval depth — each retriever owns its own `k`. With N retrievers the
    fused list grows without bound, and this is the separate constraint that governs it.
    """

    max_candidates: int = 30
    max_candidate_tokens: int = 8_000


@dataclass(frozen=True)
class SelectionOutcome:
    choice: ResolvedChoice
    confidence: float | None
    explanation: str
    raw: str
    keyed: KeyedCandidates
    truncated: int
    usage: Usage
    error: str | None = None
    finish_reason: str | None = None


def apply_budget(
    candidates: Sequence[Candidate], policy: SelectorPolicy
) -> tuple[list[Candidate], int]:
    """Deterministically trim the fused list. Returns (kept, dropped_count)."""
    ordered = sorted(candidates, key=lambda c: (-c.fused_score, c.id))
    kept = ordered[: policy.max_candidates]

    budget_chars = policy.max_candidate_tokens * _CHARS_PER_TOKEN
    total = 0
    trimmed: list[Candidate] = []
    for candidate in kept:
        # a cheap proxy for the rendered size; the exact render happens after keying
        size = sum(len(str(v)) for v in candidate.record.fields.values()) + len(candidate.id) + 16
        if trimmed and total + size > budget_chars:
            break
        trimmed.append(candidate)
        total += size

    return trimmed, len(candidates) - len(trimmed)


class Selector:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        policy: SelectorPolicy | None = None,
        legacy_id_resolution: bool = False,
        system: str = "You return JSON only. No prose, no code fences.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """`temperature`/`max_tokens` override the client's values for this stage only;
        `None` (the default) sends the client's configured values."""
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._policy = policy or SelectorPolicy()
        self._legacy = legacy_id_resolution
        self._system = system
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def select(
        self,
        source: Record,
        context: str,
        candidates: Sequence[Candidate],
    ) -> SelectionOutcome:
        kept, dropped = apply_budget(candidates, self._policy)
        keyed = assign_keys(kept, self._templates)

        if not keyed.order:
            return SelectionOutcome(
                choice=ResolvedChoice(None, Resolution.ABSTAIN, None),
                confidence=None,
                explanation="no candidates were retrieved",
                raw="",
                keyed=keyed,
                truncated=dropped,
                usage=Usage.zero(),
            )

        prompt = self._prompts.render_select(
            source_fields=source.fields,
            context=context,
            candidate_block=keyed.rendered,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=self._system,
                user=prompt,
                schema=SELECT_SCHEMA,
                schema_name="selection",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            return SelectionOutcome(
                choice=ResolvedChoice(None, Resolution.UNRESOLVED, response.text),
                confidence=None,
                explanation="",
                raw=response.text,
                keyed=keyed,
                truncated=dropped,
                usage=response.usage,
                error=str(exc),
                finish_reason=response.finish_reason,
            )

        raw_key = payload.get("chosen_key")
        if raw_key is not None and not isinstance(raw_key, str):
            raw_key = str(raw_key)
        confidence, invalid = parse_confidence(payload.get("confidence_score"))
        explanation = str(payload.get("explanation", ""))

        choice = resolve_key(raw_key, keyed, legacy=self._legacy)

        # A choice we cannot score is a choice we cannot classify.
        error: str | None = None
        if choice.resolution is Resolution.EXACT_KEY and confidence is None:
            choice = ResolvedChoice(None, Resolution.UNRESOLVED, raw_key)
            error = invalid
        elif choice.resolution is Resolution.UNRESOLVED:
            error = f"model returned {raw_key!r}, which is not a key issued this attempt"

        return SelectionOutcome(
            choice=choice,
            confidence=confidence,
            explanation=explanation,
            raw=response.text,
            keyed=keyed,
            truncated=dropped,
            usage=response.usage,
            error=error,
            finish_reason=response.finish_reason,
        )

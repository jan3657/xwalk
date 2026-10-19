"""Selection: show the model a keyed candidate list, get back one key or null."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.llm.base import LLMClient, LLMRequest, ParseError, effective_generation
from xwalk.llm.parsing import parse_confidence, parse_json_object
from xwalk.prompts.contract import SELECT_SCHEMA, PromptSet
from xwalk.records import Candidate, DecisionReason, Record, Usage
from xwalk.stages.keying import (
    BLOCK_SEPARATOR,
    KeyedCandidates,
    Resolution,
    ResolvedChoice,
    resolve_key,
)
from xwalk.templates import TemplateSet

# 2: budget the exact rendered blocks, refusing an oversized first block.
CANDIDATE_BUDGET_VERSION = 2

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
    reason: DecisionReason | None = None


def apply_budget(
    candidates: Sequence[Candidate], policy: SelectorPolicy, templates: TemplateSet
) -> tuple[KeyedCandidates, int]:
    """Render once and keep a ranked prefix within the count and character budgets.

    Count the actual key prefixes and inter-block separators at four characters per
    estimated token. Whole blocks are retained, never sliced. Keys retain the width
    assigned to the count-capped slate, even if the text budget later trims it.
    """
    ordered = sorted(candidates, key=lambda c: (-c.fused_score, c.id))
    capped = ordered[: policy.max_candidates]
    width = max(2, len(str(len(capped))))
    budget_chars = policy.max_candidate_tokens * _CHARS_PER_TOKEN
    total = 0
    kept: list[str] = []
    by_key: dict[str, Candidate] = {}
    issued: dict[str, str] = {}
    blocks: dict[str, str] = {}
    for index, candidate in enumerate(capped, start=1):
        key = f"C{index:0{width}d}"
        block = f"[{key}] {templates.render_candidate(candidate.record)}"
        size = len(block) + (len(BLOCK_SEPARATOR) if kept else 0)
        if total + size > budget_chars:
            break
        kept.append(key)
        by_key[key] = candidate
        issued[key] = candidate.id
        blocks[key] = block
        total += size

    return KeyedCandidates(order=tuple(kept), by_key=by_key, issued=issued, blocks=blocks), len(
        candidates
    ) - len(kept)


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

    @property
    def identity(self) -> dict[str, Any]:
        """Everything about this stage that changes what it sends or how it reads the
        answer, for the run fingerprint. Never a credential: the client contributes its
        fingerprint."""
        return {
            "llm": self._llm.fingerprint,
            "prompts": self._prompts.fingerprint,
            "templates": self._templates.fingerprint,
            **effective_generation(
                self._llm, temperature=self._temperature, max_tokens=self._max_tokens
            ),
            "system": self._system,
            "max_candidates": self._policy.max_candidates,
            "max_candidate_tokens": self._policy.max_candidate_tokens,
            "budget_version": CANDIDATE_BUDGET_VERSION,
            "legacy_id_resolution": self._legacy,
        }

    async def select(
        self,
        source: Record,
        context: str,
        candidates: Sequence[Candidate],
    ) -> SelectionOutcome:
        keyed, dropped = apply_budget(candidates, self._policy, self._templates)

        if not keyed.order:
            over_budget = bool(candidates)
            return SelectionOutcome(
                choice=ResolvedChoice(
                    None, Resolution.UNRESOLVED if over_budget else Resolution.ABSTAIN, None
                ),
                confidence=None,
                explanation=(
                    "first candidate block exceeds the candidate budget"
                    if over_budget
                    else "no candidates were retrieved"
                ),
                raw="",
                keyed=keyed,
                truncated=dropped,
                usage=Usage.zero(),
                reason=DecisionReason.CANDIDATE_BUDGET_EXCEEDED if over_budget else None,
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

        # Only an explicit `"chosen_key": null` abstains. An absent field is malformed
        # output: reading it as null would turn a broken answer into a confident no-match.
        if "chosen_key" not in payload:
            return SelectionOutcome(
                choice=ResolvedChoice(None, Resolution.UNRESOLVED, None),
                confidence=confidence,
                explanation=explanation,
                raw=response.text,
                keyed=keyed,
                truncated=dropped,
                usage=response.usage,
                error="chosen_key missing from the selector output",
                finish_reason=response.finish_reason,
            )

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

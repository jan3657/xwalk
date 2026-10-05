"""The gate: an independent score, and — when it is uncertain — an independent verdict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_confidence, parse_json_object
from xwalk.prompts.contract import SCORE_SCHEMA, VERIFY_SCHEMA, PromptSet
from xwalk.records import Record, RetryProposal, Usage
from xwalk.stages.keying import KeyedCandidates
from xwalk.templates import TemplateSet

Decision = Literal["support", "disagree", "no_match"]
_SYSTEM = "You return JSON only. No prose, no code fences."


@dataclass(frozen=True)
class ScoreOutcome:
    score: float | None
    explanation: str
    proposals: tuple[RetryProposal, ...]
    raw: str
    usage: Usage
    error: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class VerifierVerdict:
    decision: Decision
    preferred_key: str | None
    confidence: float | None
    explanation: str
    raw: str
    usage: Usage
    error: str | None = None
    finish_reason: str | None = None


def _blocks(keyed: KeyedCandidates, chosen_key: str) -> tuple[str, str]:
    """(chosen block, every other block), selected by key from the structured list."""
    if chosen_key not in keyed.blocks:
        raise KeyError(f"{chosen_key!r} was not issued for this attempt")
    return keyed.blocks[chosen_key], keyed.render_except(chosen_key)


def _clean(values: object, kind: Literal["candidate", "query"], issued: Sequence[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        if kind == "candidate" and text.upper() not in issued:
            continue  # naming a key we never issued is a hallucination, not a lead
        seen.append(text.upper() if kind == "candidate" else text)
    return seen


class Scorer:
    """Judges the selected match against the whole source record."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        review_floor: float = 0.4,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """`temperature`/`max_tokens` override the client's values for this stage only;
        `None` (the default) sends the client's configured values."""
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._review_floor = review_floor
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def score(
        self,
        source: Record,
        context: str,
        keyed: KeyedCandidates,
        chosen_key: str,
    ) -> ScoreOutcome:
        chosen_block, other_candidates = _blocks(keyed, chosen_key)
        prompt = self._prompts.render_score(
            source_fields=source.fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
            review_floor=self._review_floor,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=prompt,
                schema=SCORE_SCHEMA,
                schema_name="score",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            return ScoreOutcome(
                score=None,
                explanation="",
                proposals=(),
                raw=response.text,
                usage=response.usage,
                error=str(exc),
                finish_reason=response.finish_reason,
            )

        score, invalid = parse_confidence(payload.get("confidence_score"))
        issued = list(keyed.order)
        proposals = tuple(
            RetryProposal(kind="candidate", value=key, source="scorer")
            for key in _clean(payload.get("better_candidate_keys"), "candidate", issued)
        ) + tuple(
            RetryProposal(kind="query", value=query, source="scorer")
            for query in _clean(payload.get("better_queries"), "query", issued)
        )

        return ScoreOutcome(
            score=score,
            explanation=str(payload.get("explanation", "")),
            proposals=proposals,
            raw=response.text,
            usage=response.usage,
            error=invalid,
            finish_reason=response.finish_reason,
        )


class Verifier:
    """A second, independent opinion. Returns a verdict, not a number to average."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def verify(
        self,
        source: Record,
        context: str,
        keyed: KeyedCandidates,
        chosen_key: str,
    ) -> VerifierVerdict:
        chosen_block, other_candidates = _blocks(keyed, chosen_key)
        prompt = self._prompts.render_verify(
            source_fields=source.fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=prompt,
                schema=VERIFY_SCHEMA,
                schema_name="verification",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            # An unreadable second opinion is not a supporting one.
            return VerifierVerdict(
                decision="disagree",
                preferred_key=None,
                confidence=None,
                explanation="",
                raw=response.text,
                usage=response.usage,
                error=str(exc),
                finish_reason=response.finish_reason,
            )

        raw_decision = str(payload.get("decision", "")).strip().lower()
        decision: Decision = (
            raw_decision if raw_decision in ("support", "disagree", "no_match") else "disagree"  # type: ignore[assignment]
        )
        problems: list[str] = []
        if raw_decision != decision:
            problems.append(f"unrecognised decision {raw_decision!r}")
        # The verdict's own confidence is advisory; an invalid one is dropped, not stored.
        confidence, invalid = parse_confidence(payload.get("confidence_score"))
        if invalid is not None and "confidence_score" in payload:
            problems.append(invalid)

        raw_preferred = payload.get("preferred_key")
        preferred: str | None = None
        if isinstance(raw_preferred, str) and raw_preferred.strip().upper() in keyed.by_key:
            preferred = raw_preferred.strip().upper()

        return VerifierVerdict(
            decision=decision,
            preferred_key=preferred,
            confidence=confidence,
            explanation=str(payload.get("explanation", "")),
            raw=response.text,
            usage=response.usage,
            error="; ".join(problems) or None,
            finish_reason=response.finish_reason,
        )

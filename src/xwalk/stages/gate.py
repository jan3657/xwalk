"""The gate: an independent score, and — when it is uncertain — an independent verdict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
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


@dataclass(frozen=True)
class VerifierVerdict:
    decision: Decision
    preferred_key: str | None
    confidence: float | None
    explanation: str
    raw: str
    usage: Usage
    error: str | None = None


def _clamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _blocks(keyed: KeyedCandidates, chosen_key: str) -> tuple[str, str]:
    """Split the rendered candidate list into (chosen, others)."""
    if chosen_key not in keyed.by_key:
        raise KeyError(f"{chosen_key!r} was not issued for this attempt")
    entries = keyed.rendered.split("\n\n")
    chosen = next(e for e in entries if e.startswith(f"[{chosen_key}]"))
    others = [e for e in entries if not e.startswith(f"[{chosen_key}]")]
    return chosen, "\n\n".join(others)


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
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._review_floor = review_floor
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
            )

        score = _clamp(payload.get("confidence_score"))
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
            error=None if score is not None else "confidence_score was missing or not a number",
        )


class Verifier:
    """A second, independent opinion. Returns a verdict, not a number to average."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
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
            )

        raw_decision = str(payload.get("decision", "")).strip().lower()
        decision: Decision = (
            raw_decision if raw_decision in ("support", "disagree", "no_match") else "disagree"  # type: ignore[assignment]
        )
        error = None if raw_decision == decision else f"unrecognised decision {raw_decision!r}"

        raw_preferred = payload.get("preferred_key")
        preferred: str | None = None
        if isinstance(raw_preferred, str) and raw_preferred.strip().upper() in keyed.by_key:
            preferred = raw_preferred.strip().upper()

        return VerifierVerdict(
            decision=decision,
            preferred_key=preferred,
            confidence=_clamp(payload.get("confidence_score")),
            explanation=str(payload.get("explanation", "")),
            raw=response.text,
            usage=response.usage,
            error=error,
        )

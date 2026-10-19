"""Query reformulation. Produces query proposals only — never candidate proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.prompts.contract import REWRITE_SCHEMA, PromptSet
from xwalk.records import Record, RetryProposal, Usage
from xwalk.stages.keying import KeyedCandidates
from xwalk.stages.proposals import normalise_query
from xwalk.templates import TemplateSet

_SYSTEM = "You return JSON only. No prose, no code fences."


@dataclass(frozen=True)
class RewriteOutcome:
    proposals: tuple[RetryProposal, ...]
    explanation: str
    raw: str
    usage: Usage
    error: str | None = None


class QueryRewriter:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        max_queries: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """`temperature`/`max_tokens` override the client's values for this stage only;
        `None` (the default) sends the client's configured values."""
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._max_queries = max_queries
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def rewrite(
        self,
        source: Record,
        context: str,
        previous_queries: Sequence[str],
        keyed: KeyedCandidates,
    ) -> RewriteOutcome:
        prompt = self._prompts.render_rewrite(
            source_fields=source.fields,
            context=context,
            previous_queries=list(previous_queries),
            best_candidates=keyed.rendered,
            max_queries=self._max_queries,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=prompt,
                schema=REWRITE_SCHEMA,
                schema_name="rewrite",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            return RewriteOutcome(
                proposals=(),
                explanation="",
                raw=response.text,
                usage=response.usage,
                error=str(exc),
            )

        explanation = str(payload.get("explanation", ""))
        queries = payload.get("queries")
        # `queries` must be a list of strings. Anything else is malformed output: no
        # proposals, error noted. A bare string is never iterated into one-letter queries.
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            return RewriteOutcome(
                proposals=(),
                explanation=explanation,
                raw=response.text,
                usage=response.usage,
                error=f"queries must be a list of strings, got {queries!r}"[:200],
            )

        seen = {normalise_query(q) for q in previous_queries}
        proposals: list[RetryProposal] = []
        for value in queries:
            text = value.strip()
            normalised = normalise_query(text)
            if not text or normalised in seen:
                continue
            seen.add(normalised)
            proposals.append(RetryProposal(kind="query", value=text, source="rewriter"))
            if len(proposals) >= self._max_queries:
                break

        return RewriteOutcome(
            proposals=tuple(proposals),
            explanation=explanation,
            raw=response.text,
            usage=response.usage,
        )

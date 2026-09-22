"""Screen: one yes/no per candidate, in chunks, all against the same source.

This is where the decider path gets its recall. The LLM path can afford to show the
model 25 candidates; this stage shows it hundreds, fifty at a time, because a
per-candidate probability from a small clean state beat a single large call in the
gold-sample spike that shaped the design.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.decide.base import DecisionClient, Noul, NoulAnswer
from xwalk.decide.questions import QuestionSet
from xwalk.records import Candidate, Record, Usage
from xwalk.templates import TemplateSet


def build_source_state(source: Record, context: str) -> dict[str, Any]:
    state: dict[str, Any] = {"fields": dict(source.fields)}
    if context:
        state["context"] = context
    return state


@dataclass(frozen=True)
class ScreenOutcome:
    probabilities: Mapping[str, float]  # opaque key -> p(same entity)
    shortlist: tuple[str, ...]  # record ids, best first
    issued: Mapping[str, str]  # opaque key -> record id
    usage: Usage
    chunks: int
    model: str
    notes: tuple[str, ...] = ()

    @property
    def best(self) -> float | None:
        return max(self.probabilities.values()) if self.probabilities else None


class Screener:
    def __init__(
        self,
        decider: DecisionClient,
        questions: QuestionSet,
        templates: TemplateSet,
        *,
        chunk_size: int = 50,
        shortlist_size: int = 15,
        shortlist_floor: float = 0.2,
        max_state_chars: int = 120_000,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
        self._decider = decider
        self._questions = questions
        self._templates = templates
        self._chunk_size = chunk_size
        self._shortlist_size = shortlist_size
        self._shortlist_floor = shortlist_floor
        self._max_state_chars = max_state_chars

    def _chunks(
        self, keyed: list[tuple[str, str]], source_state: Mapping[str, Any]
    ) -> tuple[list[list[tuple[str, str]]], list[str]]:
        """Fixed-size chunks, halved while a chunk's state would exceed the size cap."""
        base = len(json.dumps(source_state, ensure_ascii=False))
        chunks: list[list[tuple[str, str]]] = []
        notes: list[str] = []
        pending = [keyed[i : i + self._chunk_size] for i in range(0, len(keyed), self._chunk_size)]
        while pending:
            chunk = pending.pop(0)
            size = base + sum(len(key) + len(text) + 8 for key, text in chunk)
            if size > self._max_state_chars and len(chunk) > 1:
                half = len(chunk) // 2
                pending[:0] = [chunk[:half], chunk[half:]]
                notes.append(f"screen: split a chunk of {len(chunk)} to stay under the state cap")
                continue
            chunks.append(chunk)
        return chunks, notes

    async def screen(
        self, source: Record, context: str, candidates: Sequence[Candidate]
    ) -> ScreenOutcome:
        if not candidates:
            return ScreenOutcome({}, (), {}, Usage.zero(), 0, self._decider.model)

        width = max(3, len(str(len(candidates))))
        keyed = [
            (f"C{i:0{width}d}", self._templates.render_candidate(c.record))
            for i, c in enumerate(candidates, start=1)
        ]
        issued = {key: c.id for (key, _), c in zip(keyed, candidates, strict=True)}
        source_state = build_source_state(source, context)
        chunks, notes = self._chunks(keyed, source_state)

        async def one(chunk: list[tuple[str, str]]) -> tuple[dict[str, float], Usage, str]:
            state = {"source": source_state, "candidates": dict(chunk)}
            questions: dict[str, Noul] = {
                f"n_{key}": self._questions.screen_question(key) for key, _ in chunk
            }
            response = await self._decider.decide(state, questions)
            probabilities = {}
            for key, _ in chunk:
                answer = response.answers[f"n_{key}"]
                assert isinstance(answer, NoulAnswer)  # parse_response guarantees the type
                probabilities[key] = answer.noul
            return probabilities, response.usage, response.model

        results = await asyncio.gather(*(one(chunk) for chunk in chunks))

        probabilities: dict[str, float] = {}
        usage = Usage.zero()
        model = self._decider.model
        for chunk_probabilities, chunk_usage, served in results:
            probabilities.update(chunk_probabilities)
            usage = usage + chunk_usage
            model = served

        ranked = sorted(probabilities, key=lambda k: (-probabilities[k], k))
        shortlist = tuple(
            issued[key]
            for key in ranked[: self._shortlist_size]
            if probabilities[key] >= self._shortlist_floor
        )
        return ScreenOutcome(
            probabilities=probabilities,
            shortlist=shortlist,
            issued=issued,
            usage=usage,
            chunks=len(chunks),
            model=model,
            notes=tuple(notes),
        )

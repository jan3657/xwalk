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

from xwalk.decide.base import (
    DecisionClient,
    DecisionError,
    DecisionFatalError,
    DecisionResponseError,
    Noul,
    NoulAnswer,
    failure_usage,
)
from xwalk.decide.questions import QuestionSet
from xwalk.records import Candidate, Record, Usage
from xwalk.templates import TemplateSet


class ScreenFailed(DecisionError):
    """A screen chunk failed. `usage` is what the screen cost: the sibling chunks that
    answered, plus the failing request itself (dispatched, its tokens unknown)."""

    def __init__(self, cause: DecisionError, usage: Usage) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.usage = usage


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


async def _cancel(tasks: Sequence[asyncio.Task[Any]]) -> None:
    """Cancel whatever is still running and await it, so no task is left dangling."""
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


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
        self, keyed: list[tuple[str, str]], shared_state: Mapping[str, Any]
    ) -> tuple[list[list[tuple[str, str]]], list[str]]:
        """Fixed-size chunks, halved while a chunk's state would exceed the size cap.

        `shared_state` is what every chunk carries besides its candidates: the source and
        the rules.
        """
        base = len(json.dumps(shared_state, ensure_ascii=False))
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
            if size > self._max_state_chars:
                # One candidate on its own is over the cap. There is nothing left to
                # split, so it is sent as it is and the note is the only warning.
                notes.append(
                    f"screen: candidate {chunk[0][0]} alone exceeds the state cap "
                    f"({size} > {self._max_state_chars} chars); sent anyway"
                )
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
        # The rules go in the state once per chunk; each candidate's question only refers
        # to them, which is most of what a screen call used to cost.
        shared = {
            "source": build_source_state(source, context),
            "rules": self._questions.rules_state(),
        }
        chunks, notes = self._chunks(keyed, shared)

        async def one(chunk: list[tuple[str, str]]) -> tuple[dict[str, float], Usage, str]:
            state = {**shared, "candidates": dict(chunk)}
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

        tasks = [asyncio.create_task(one(chunk)) for chunk in chunks]
        # FIRST_EXCEPTION, not a plain gather: the moment one chunk fails the rest are
        # wasted spend, and a batch that is about to stop should not keep paying for them.
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except BaseException:
            # The screen itself was cancelled (or interrupted): take the chunks down with it
            # rather than leave them running and billing.
            await _cancel(tasks)
            raise

        results: list[tuple[dict[str, float], Usage, str]] = []
        first_error: BaseException | None = None
        for task in tasks:
            if not task.done() or task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                if first_error is None:
                    first_error = error
            else:
                results.append(task.result())

        if first_error is not None:
            await _cancel(tasks)
            if isinstance(first_error, DecisionError):
                # The sibling chunks were billed before this one failed, and the failing
                # request was dispatched too. Hand the caller the bill with the error.
                paid = sum((chunk_usage for _, chunk_usage, _ in results), Usage.zero())
                paid = paid + failure_usage(first_error)
                if isinstance(first_error, DecisionFatalError) and not isinstance(
                    first_error, DecisionResponseError
                ):
                    # The batch stops (fatal_provider_failure); the matcher books this.
                    first_error.usage = paid
                    raise first_error
                raise ScreenFailed(first_error, paid)
            raise first_error

        probabilities: dict[str, float] = {}
        usage = Usage.zero()
        model = self._decider.model
        for chunk_probabilities, chunk_usage, served in results:
            probabilities.update(chunk_probabilities)
            usage = usage + chunk_usage
            model = served

        return ScreenOutcome(
            probabilities=probabilities,
            shortlist=self._shortlist(probabilities, issued),
            issued=issued,
            usage=usage,
            chunks=len(chunks),
            model=model,
            notes=tuple(notes),
        )

    def _shortlist(
        self, probabilities: Mapping[str, float], issued: Mapping[str, str]
    ) -> tuple[str, ...]:
        ranked = sorted(probabilities, key=lambda k: (-probabilities[k], k))
        return tuple(
            issued[key]
            for key in ranked[: self._shortlist_size]
            if probabilities[key] >= self._shortlist_floor
        )

    def merge(
        self, first: ScreenOutcome, second: ScreenOutcome, candidates: Sequence[Candidate]
    ) -> ScreenOutcome:
        """Two screens of overlapping candidate sets, as one outcome over `candidates`.

        `candidates` is the deduplicated union, the first screen's order first. Keys are
        re-issued over it (each screen issued its own C001...), and a record screened twice
        keeps the higher of its two probabilities.
        """
        by_id: dict[str, float] = {}
        for outcome in (first, second):
            for key, probability in outcome.probabilities.items():
                record_id = outcome.issued[key]
                by_id[record_id] = max(probability, by_id.get(record_id, probability))
        width = max(3, len(str(len(candidates))))
        issued = {
            f"C{i:0{width}d}": c.id for i, c in enumerate(candidates, start=1) if c.id in by_id
        }
        probabilities = {key: by_id[record_id] for key, record_id in issued.items()}
        return ScreenOutcome(
            probabilities=probabilities,
            shortlist=self._shortlist(probabilities, issued),
            issued=issued,
            usage=first.usage + second.usage,
            chunks=first.chunks + second.chunks,
            model=second.model,
            notes=first.notes + second.notes,
        )

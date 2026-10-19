"""The decision-model contract.

A System One model answers typed questions about a state with probabilities, not text.
Nothing here parses prose; the only parsing is of a JSON payload whose shape the vendor
documents, and every answer is validated against the question that was asked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from xwalk.llm.base import token_count
from xwalk.records import Usage

Structured = str | Mapping[str, Any] | Sequence[Any]


class DecisionError(Exception):
    """Base class for decision-provider failures.

    `usage` is what the client observed before failing (requests dispatched, their
    tokens unknown); `None` means it could not say, and callers count one call with
    unknown usage.
    """

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class DecisionRetryableError(DecisionError):
    """Transient: 408, 429, 5xx, 529, timeouts, connection resets."""

    def __init__(
        self, message: str, *, retry_after: float | None = None, usage: Usage | None = None
    ) -> None:
        super().__init__(message, usage=usage)
        self.retry_after = retry_after


class DecisionFatalError(DecisionError):
    """Non-retryable: auth failure, unknown model, malformed request, broken contract.

    A batch run treats it as `fatal_provider_failure` and aborts (CONTRACTS.md section
    3), except for its subclass `DecisionResponseError`.
    """


class DecisionResponseError(DecisionFatalError):
    """A response that does not fit the questions asked (not JSON, a missing or
    mistyped answer, a probability outside [0, 1]).

    Not retried by the client, but -- like malformed LLM output under CONTRACTS.md
    section 3 -- a recoverable per-record outcome for the matcher: the record is
    `failed` with `provider_failure`, and the run goes on.
    """


def failure_usage(exc: BaseException) -> Usage:
    """The usage to book for a decision call that raised `exc`: what the client attached,
    else one dispatched call with unknown usage, never a free call."""
    usage = getattr(exc, "usage", None)
    return usage if isinstance(usage, Usage) else Usage.unreported()


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is the probability that the answer is yes."""

    instructions: Structured
    criteria: Mapping[str, Structured] | None = None


@dataclass(frozen=True)
class Choice:
    """Pick one option from `criteria`. At most 255 options."""

    instructions: Structured
    criteria: Mapping[str, Structured | None]

    def __post_init__(self) -> None:
        if not 1 <= len(self.criteria) <= 255:
            raise ValueError(f"a Choice needs 1 to 255 options, got {len(self.criteria)}")


@dataclass(frozen=True)
class Score:
    """Rate the state against ordered levels. Between 2 and 10 levels."""

    instructions: Structured
    criteria: Sequence[Structured]

    def __post_init__(self) -> None:
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError(f"a Score needs 2 to 10 levels, got {len(self.criteria)}")


Question = Noul | Choice | Score


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: Mapping[str, float]
    confidence: float
    legend: Mapping[str, str]


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DecisionResponse:
    answers: Mapping[str, Answer]
    model: str
    usage: Usage


@runtime_checkable
class DecisionClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """Digest of endpoint host, model id, and adapter version. Never a secret."""

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse: ...


# --- wire format ---------------------------------------------------------------


def _plain(value: Structured | None) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return [_plain(v) for v in value]


def question_to_dict(question: Question) -> dict[str, Any]:
    if isinstance(question, Noul):
        body: dict[str, Any] = {"type": "noul", "instructions": _plain(question.instructions)}
        if question.criteria is not None:
            body["criteria"] = _plain(question.criteria)
        return body
    if isinstance(question, Choice):
        return {
            "type": "choice",
            "instructions": _plain(question.instructions),
            "criteria": {k: _plain(v) for k, v in question.criteria.items()},
        }
    return {
        "type": "score",
        "instructions": _plain(question.instructions),
        "criteria": [_plain(v) for v in question.criteria],
    }


def _bounded(value: object, low: float, high: float, where: str) -> float:
    """A number in [low, high], never clamped. NaN, infinities, booleans and integers
    too large for a float are rejected before any conversion can raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionResponseError(f"{where}: expected a number, got {value!r}")
    # Compared before float(): a JSON integer can exceed the float range. NaN fails both
    # comparisons, an infinity one of them.
    if not low <= value <= high:
        raise DecisionResponseError(f"{where}: {value!r} is outside [{low:g}, {high:g}]")
    return float(value)


def _unit(value: object, where: str) -> float:
    return _bounded(value, 0.0, 1.0, where)


def _distribution(value: object, where: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DecisionResponseError(f"{where}: probabilities must be a mapping")
    return {str(k): _unit(v, f"{where}[{k!r}]") for k, v in value.items()}


def _answer(question: Question, raw: object, name: str) -> Answer:
    if not isinstance(raw, Mapping):
        raise DecisionResponseError(f"answer {name!r} is not an object")
    kind = raw.get("type")
    if isinstance(question, Noul):
        if kind != "noul":
            raise DecisionResponseError(f"answer {name!r}: expected noul, got {kind!r}")
        return NoulAnswer(noul=_unit(raw.get("noul"), f"answer {name!r}.noul"))
    if isinstance(question, Choice):
        if kind != "choice":
            raise DecisionResponseError(f"answer {name!r}: expected choice, got {kind!r}")
        choice = raw.get("choice")
        if not isinstance(choice, str) or choice not in question.criteria:
            raise DecisionResponseError(
                f"answer {name!r}: choice {choice!r} is not one of the offered options"
            )
        return ChoiceAnswer(
            choice=choice,
            probabilities=_distribution(raw.get("probabilities"), f"answer {name!r}"),
            confidence=_unit(raw.get("confidence"), f"answer {name!r}.confidence"),
        )
    if kind != "score":
        raise DecisionResponseError(f"answer {name!r}: expected score, got {kind!r}")
    # The rubric offers levels 0..len(criteria)-1; a score outside them is not a grade.
    score = _bounded(
        raw.get("score"), 0.0, float(len(question.criteria) - 1), f"answer {name!r}.score"
    )
    legend_raw = raw.get("legend") or {}
    if not isinstance(legend_raw, Mapping):
        raise DecisionResponseError(f"answer {name!r}.legend must be a mapping")
    return ScoreAnswer(
        score=score,
        probabilities=_distribution(raw.get("probabilities"), f"answer {name!r}"),
        confidence=_unit(raw.get("confidence"), f"answer {name!r}.confidence"),
        legend={str(k): str(v) for k, v in legend_raw.items()},
    )


def parse_response(
    payload: Mapping[str, Any],
    questions: Mapping[str, Question],
    *,
    configured_model: str,
) -> DecisionResponse:
    """Type every answer against its question. A broken contract is fatal, not data."""
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise DecisionResponseError("response has no answers object")
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if name not in raw_answers:
            raise DecisionResponseError(f"response is missing an answer for {name!r}")
        answers[name] = _answer(question, raw_answers[name], name)

    model = payload.get("model")
    return DecisionResponse(
        answers=answers,
        model=model if isinstance(model, str) and model else configured_model,
        usage=_reported_usage(payload.get("usage")),
    )


def _reported_usage(raw: object) -> Usage:
    """One answered call. Tokens and cost count only when reported and valid.

    A missing usage block, a missing or invalid `input_tokens` (NaN, infinite, negative,
    fractional, a string, a boolean) or an invalid `cost` makes the call's usage unknown:
    never a known zero, never a partial trust of a malformed block, and never an
    exception. An absent `cost` is 0.0 (not every deployment reports one).
    """
    if not isinstance(raw, Mapping):
        return Usage.unreported()
    raw_tokens = raw.get("input_tokens")
    tokens = None if raw_tokens is None else token_count(raw_tokens)
    raw_cost = raw.get("cost", 0.0)
    cost_ok = (
        isinstance(raw_cost, (int, float))
        and not isinstance(raw_cost, bool)
        and math.isfinite(raw_cost)
        and raw_cost >= 0
    )
    if tokens is None or not cost_ok:
        return Usage.unreported()
    return Usage(prompt_tokens=tokens, completion_tokens=0, calls=1, cost_usd=float(raw_cost))


def response_to_dict(response: DecisionResponse) -> dict[str, Any]:
    """The cache representation. Mirrors the wire shape so `parse_response` reads it back."""
    answers: dict[str, Any] = {}
    for name, answer in response.answers.items():
        if isinstance(answer, NoulAnswer):
            answers[name] = {"type": "noul", "noul": answer.noul}
        elif isinstance(answer, ChoiceAnswer):
            answers[name] = {
                "type": "choice",
                "choice": answer.choice,
                "probabilities": dict(answer.probabilities),
                "confidence": answer.confidence,
            }
        else:
            answers[name] = {
                "type": "score",
                "score": answer.score,
                "probabilities": dict(answer.probabilities),
                "confidence": answer.confidence,
                "legend": dict(answer.legend),
            }
    return {
        "model": response.model,
        "answers": answers,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": 0,
            "cost": response.usage.cost_usd,
        },
    }


def response_from_dict(
    data: Mapping[str, Any], questions: Mapping[str, Question]
) -> DecisionResponse:
    return parse_response(data, questions, configured_model=str(data.get("model", "")))


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionClient",
    "DecisionError",
    "DecisionFatalError",
    "DecisionResponse",
    "DecisionResponseError",
    "DecisionRetryableError",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "Structured",
    "failure_usage",
    "parse_response",
    "question_to_dict",
    "response_from_dict",
    "response_to_dict",
]

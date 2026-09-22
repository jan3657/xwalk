# Jev Decider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second decision path in xwalk where TypeSafe's Jev decision model screens hundreds of candidates, chooses among the survivors, and gates the choice with calibrated probabilities, writing the same `MatchResult` the LLM path writes.

**Architecture:** A new `xwalk.decide` package holds the `DecisionClient` protocol, the Jev HTTP adapter, a fake, a ledger cache, the question set composed from `slots.yaml`, the decision policy, and a `DecisionMatcher`. Three new stages (screen, choose, property gate) do the work. Retrieval is extracted from `Matcher` into a shared function that also handles several queries per source. Configuration adds a `decider:` block that is mutually exclusive with `llm:`.

**Tech Stack:** Python 3.10+, httpx, pydantic 2, pytest with `asyncio_mode = "auto"`, ruff, mypy `--strict`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-jev-decider-design.md`

## Global Constraints

- Python floor is 3.10 (`requires-python = ">=3.10"`); no `match` statements on 3.10-incompatible patterns, no `typing.Self`.
- Every task must pass `ruff check src tests`, `ruff format --check src tests`, `mypy` (strict, configured in `pyproject.toml`), and `pytest`.
- Line length 100.
- No API key ever appears in a job file, a fingerprint, a manifest, or a log line.
- A malformed model answer must never resolve to a real target record.
- `failed` is for infrastructure; `unmatched` is evidence about the data. Never conflate.
- Additive changes only to `Attempt`, `MatchResult`, `Usage`: new fields get defaults so every existing ledger blob still deserialises.
- The LLM path's behaviour and tests stay unchanged. Run the full suite after every task.
- Commit after every task with the attribution line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work on branch `jev-decider`.

## File map

| File | Responsibility |
|---|---|
| `src/xwalk/records.py` | add `Usage.cost_usd`, `Attempt.signals`, `MatchResult.signals` |
| `src/xwalk/serde.py` | serialise the three new fields, tolerate their absence |
| `src/xwalk/decide/__init__.py` | public re-exports |
| `src/xwalk/decide/base.py` | question and answer types, `DecisionResponse`, `DecisionClient`, errors, response parsing |
| `src/xwalk/decide/fake.py` | `FakeDecider` for offline tests |
| `src/xwalk/decide/jev.py` | `JevClient` over httpx with retries |
| `src/xwalk/decide/cache.py` | `CachingDecider` over the ledger's `llm_cache` table |
| `src/xwalk/decide/questions.py` | `QuestionSet.from_slots`: screen, choose, rubric, property questions |
| `src/xwalk/decide/policy.py` | `DecisionPolicy`, `Signals`, `derive_status`, `render_explanation` |
| `src/xwalk/decide/matcher.py` | `DecisionMatcher` |
| `src/xwalk/decide/fit.py` | threshold fitting against gold |
| `src/xwalk/retrieve.py` | `retrieve()` shared by both matchers, multi-query aware |
| `src/xwalk/templates.py` | `TemplateSet.queries` and `render_queries` |
| `src/xwalk/matcher.py` | delegate `_retrieve` to `xwalk.retrieve` |
| `src/xwalk/stages/screen.py` | `Screener` |
| `src/xwalk/stages/choose.py` | `Chooser` |
| `src/xwalk/stages/property_gate.py` | `PropertyGate` |
| `src/xwalk/prompts/contract.py` | `PromptSlots.properties` |
| `src/xwalk/config.py` | `DeciderSpec`, `DecisionPolicySpec`, `TemplateSpec.queries`, builders |
| `src/xwalk/batch.py` | `MatcherLike` protocol, `build_decision_run_fingerprint` |
| `src/xwalk/cli/main.py` | `match` dispatches on `decider:`; new `fit` command |
| `tests/test_decide_*.py`, `tests/test_screen.py`, `tests/test_choose.py`, `tests/test_property_gate.py`, `tests/test_retrieve.py` | tests |
| `docs/reference/decide.md`, `docs/README.md`, `docs/components.md`, `docs/reference/job-file.md`, `docs/reference/cli.md` | documentation |
| `examples/cafeteria_fcd/job_jev.yaml`, `examples/ref_zivila/jobs/foodon/job_jev.yaml`, both `slots.yaml` | runnable jobs |

---

### Task 1: Additive record fields

**Files:**
- Modify: `src/xwalk/records.py` (`Usage`, `Attempt`, `MatchResult`)
- Modify: `src/xwalk/serde.py` (`_usage_to_dict`, `_attempt_to_dict`, `_attempt_from_dict`, `result_to_dict`, `result_from_dict`)
- Test: `tests/test_serde.py`, `tests/test_records.py`

**Interfaces:**
- Produces: `Usage(prompt_tokens, completion_tokens, calls, cost_usd: float = 0.0)`; `Attempt.signals: Mapping[str, float]` (default `{}`); `MatchResult.signals: Mapping[str, float]` (default `{}`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_records.py`:

```python
from xwalk.records import Usage


def test_usage_adds_cost():
    total = Usage(prompt_tokens=1, completion_tokens=0, calls=1, cost_usd=0.001) + Usage(
        prompt_tokens=2, completion_tokens=0, calls=1, cost_usd=0.002
    )
    assert total.cost_usd == 0.003
    assert total.calls == 2


def test_usage_cost_defaults_to_zero():
    assert Usage().cost_usd == 0.0
```

Append to `tests/test_serde.py` (the file already builds a full `MatchResult`; reuse its helper if one exists, otherwise this self-contained test):

```python
from xwalk.records import Attempt, DecisionReason, MatchResult, MatchStatus, Usage
from xwalk.serde import result_from_dict, result_to_dict


def _attempt(**overrides):
    base = dict(
        index=0, query="q", proposal=None, candidates=(), candidate_count=0,
        candidates_truncated=0, issued_keys={}, raw_selection=None, chosen_id=None,
        resolution="abstain", primary_score=None, explanation="", verifier_decision=None,
        verifier_score=None, verifier_preferred_id=None, audited=False,
        dropped_proposals=(), reason=None, error=None, usage=Usage.zero(),
        elapsed_seconds=0.0, finish_reason=None,
    )
    base.update(overrides)
    return Attempt(**base)


def _result(**overrides):
    base = dict(
        result_key="k", source_id="s1", source_hash="h", matched_id=None,
        matched_record=None, confidence=None, status=MatchStatus.UNMATCHED,
        reason=DecisionReason.NO_CANDIDATES, explanation="", candidates=(),
        attempts=(_attempt(),), usage=Usage.zero(), elapsed_seconds=0.0,
        run_fingerprint="fp",
    )
    base.update(overrides)
    return MatchResult(**base)


def test_signals_and_cost_round_trip():
    result = _result(
        signals={"screen_chosen": 0.93, "p_choice": 0.8},
        attempts=(_attempt(signals={"screen_chosen": 0.93}),),
        usage=Usage(prompt_tokens=10, completion_tokens=0, calls=2, cost_usd=0.0004),
    )
    back = result_from_dict(result_to_dict(result))
    assert back.signals == {"screen_chosen": 0.93, "p_choice": 0.8}
    assert back.attempts[0].signals == {"screen_chosen": 0.93}
    assert back.usage.cost_usd == 0.0004


def test_old_blobs_without_the_new_fields_still_load():
    data = result_to_dict(_result())
    del data["signals"]
    del data["attempts"][0]["signals"]
    del data["usage"]["cost_usd"]
    back = result_from_dict(data)
    assert back.signals == {}
    assert back.attempts[0].signals == {}
    assert back.usage.cost_usd == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_records.py tests/test_serde.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'cost_usd'` / `'signals'`.

- [ ] **Step 3: Add the fields**

In `src/xwalk/records.py`, change `Usage`:

```python
@dataclass(frozen=True)
class Usage:
    """Token, call, and cost accounting. Additive so attempts can be summed into a result."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    @classmethod
    def zero(cls) -> Usage:
        return cls()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Any) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
            cost_usd=self.cost_usd + other.cost_usd,
        )
```

Add as the **last** field of `Attempt` (after `finish_reason`), and the last field of `MatchResult` (after `run_fingerprint`). Both need `field` imported from `dataclasses`:

```python
    # Calibrated signals from a decision model. Empty on the LLM path.
    signals: Mapping[str, float] = field(default_factory=dict)
```

- [ ] **Step 4: Serialise them**

In `src/xwalk/serde.py`:

```python
def _usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "calls": usage.calls,
        "cost_usd": usage.cost_usd,
    }
```

In `_attempt_to_dict` add `"signals": dict(attempt.signals),`. In `_attempt_from_dict` add `signals=dict(data.get("signals") or {}),`. In `result_to_dict` add `"signals": dict(result.signals),`. In `result_from_dict` add `signals=dict(data.get("signals") or {}),`. `Usage(**data["usage"])` already tolerates a missing `cost_usd` because of the default.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/mypy`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/xwalk/records.py src/xwalk/serde.py tests/test_records.py tests/test_serde.py
git commit -m "feat: signals and cost on records, additively

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Decision types and the client protocol

**Files:**
- Create: `src/xwalk/decide/__init__.py`, `src/xwalk/decide/base.py`
- Test: `tests/test_decide_base.py`

**Interfaces:**
- Produces:
  - `Structured = str | Mapping[str, Any] | Sequence[Any]`
  - `Noul(instructions: Structured, criteria: Mapping[str, Structured] | None = None)`
  - `Choice(instructions: Structured, criteria: Mapping[str, Structured | None])`
  - `Score(instructions: Structured, criteria: Sequence[Structured])`
  - `Question = Noul | Choice | Score`
  - `NoulAnswer(noul: float)`, `ChoiceAnswer(choice: str, probabilities: Mapping[str, float], confidence: float)`, `ScoreAnswer(score: float, probabilities: Mapping[str, float], confidence: float, legend: Mapping[str, str])`
  - `Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer`
  - `DecisionResponse(answers: Mapping[str, Answer], model: str, usage: Usage)`
  - `question_to_dict(q: Question) -> dict[str, Any]`
  - `parse_response(payload: Mapping[str, Any], questions: Mapping[str, Question], *, configured_model: str) -> DecisionResponse`
  - `response_to_dict(r) -> dict`, `response_from_dict(d, questions) -> DecisionResponse`
  - `DecisionError`, `DecisionRetryableError(message, *, retry_after=None)`, `DecisionFatalError`
  - `DecisionClient` protocol: `model: str`, `fingerprint: str`, `async decide(state: Any, questions: Mapping[str, Question]) -> DecisionResponse`

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_base.py`:

```python
import pytest

from xwalk.decide.base import (
    Choice,
    ChoiceAnswer,
    DecisionFatalError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    parse_response,
    question_to_dict,
    response_from_dict,
    response_to_dict,
)

QUESTIONS = {
    "same": Noul(instructions="Is it the same?"),
    "best": Choice(instructions="Which?", criteria={"A": "apple", "B": None, "NONE": "none"}),
    "grade": Score(instructions="How good?", criteria=["bad", "ok", "good"]),
}

PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "same": {"type": "noul", "noul": 0.9},
        "best": {
            "type": "choice",
            "choice": "A",
            "probabilities": {"A": 0.8, "B": 0.15, "NONE": 0.05},
            "confidence": 0.7,
        },
        "grade": {
            "type": "score",
            "score": 1.6,
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            "confidence": 0.6,
            "legend": {"0": "bad", "1": "ok", "2": "good"},
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 9, "cost": 0.000005},
}


def test_questions_serialise_to_the_wire_shape():
    assert question_to_dict(QUESTIONS["same"]) == {"type": "noul", "instructions": "Is it the same?"}
    assert question_to_dict(QUESTIONS["best"]) == {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"A": "apple", "B": None, "NONE": "none"},
    }
    assert question_to_dict(QUESTIONS["grade"]) == {
        "type": "score",
        "instructions": "How good?",
        "criteria": ["bad", "ok", "good"],
    }


def test_parse_response_types_every_answer():
    response = parse_response(PAYLOAD, QUESTIONS, configured_model="jev-latest")
    assert response.answers["same"] == NoulAnswer(noul=0.9)
    assert isinstance(response.answers["best"], ChoiceAnswer)
    assert response.answers["best"].choice == "A"
    assert isinstance(response.answers["grade"], ScoreAnswer)
    assert response.answers["grade"].legend["2"] == "good"
    assert response.model == "typesafe/jev-1.13-20260917"
    assert response.usage.prompt_tokens == 120
    assert response.usage.completion_tokens == 0
    assert response.usage.calls == 1
    assert response.usage.cost_usd == pytest.approx(0.000005)


def test_parse_response_falls_back_to_the_configured_model():
    payload = {**PAYLOAD, "model": None}
    assert parse_response(payload, QUESTIONS, configured_model="jev-latest").model == "jev-latest"


def test_a_missing_answer_is_fatal():
    payload = {**PAYLOAD, "answers": {k: v for k, v in PAYLOAD["answers"].items() if k != "grade"}}
    with pytest.raises(DecisionFatalError, match="grade"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_a_choice_outside_the_criteria_is_fatal():
    bad = {**PAYLOAD["answers"]["best"], "choice": "Z"}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "best": bad}}
    with pytest.raises(DecisionFatalError, match="Z"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_a_probability_outside_the_unit_interval_is_fatal():
    bad = {"type": "noul", "noul": 1.7}
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "same": bad}}
    with pytest.raises(DecisionFatalError, match="same"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_a_type_mismatch_is_fatal():
    payload = {**PAYLOAD, "answers": {**PAYLOAD["answers"], "same": PAYLOAD["answers"]["best"]}}
    with pytest.raises(DecisionFatalError, match="same"):
        parse_response(payload, QUESTIONS, configured_model="m")


def test_response_round_trips_through_dicts():
    response = parse_response(PAYLOAD, QUESTIONS, configured_model="m")
    assert response_from_dict(response_to_dict(response), QUESTIONS) == response
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_base.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'xwalk.decide'`.

- [ ] **Step 3: Write `src/xwalk/decide/base.py`**

```python
"""The decision-model contract.

A System One model answers typed questions about a state with probabilities, not text.
Nothing here parses prose; the only parsing is of a JSON payload whose shape the vendor
documents, and every answer is validated against the question that was asked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from xwalk.records import Usage

Structured = str | Mapping[str, Any] | Sequence[Any]


class DecisionError(Exception):
    """Base class for decision-provider failures."""


class DecisionRetryableError(DecisionError):
    """Transient: 408, 429, 5xx, 529, timeouts, connection resets."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class DecisionFatalError(DecisionError):
    """Non-retryable: auth failure, unknown model, malformed request, broken contract."""


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


def _unit(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionFatalError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise DecisionFatalError(f"{where}: {number} is outside [0, 1]")
    return number


def _distribution(value: object, where: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DecisionFatalError(f"{where}: probabilities must be a mapping")
    return {str(k): _unit(v, f"{where}[{k!r}]") for k, v in value.items()}


def _answer(question: Question, raw: object, name: str) -> Answer:
    if not isinstance(raw, Mapping):
        raise DecisionFatalError(f"answer {name!r} is not an object")
    kind = raw.get("type")
    if isinstance(question, Noul):
        if kind != "noul":
            raise DecisionFatalError(f"answer {name!r}: expected noul, got {kind!r}")
        return NoulAnswer(noul=_unit(raw.get("noul"), f"answer {name!r}.noul"))
    if isinstance(question, Choice):
        if kind != "choice":
            raise DecisionFatalError(f"answer {name!r}: expected choice, got {kind!r}")
        choice = raw.get("choice")
        if not isinstance(choice, str) or choice not in question.criteria:
            raise DecisionFatalError(
                f"answer {name!r}: choice {choice!r} is not one of the offered options"
            )
        return ChoiceAnswer(
            choice=choice,
            probabilities=_distribution(raw.get("probabilities"), f"answer {name!r}"),
            confidence=_unit(raw.get("confidence"), f"answer {name!r}.confidence"),
        )
    if kind != "score":
        raise DecisionFatalError(f"answer {name!r}: expected score, got {kind!r}")
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise DecisionFatalError(f"answer {name!r}.score: expected a number, got {score!r}")
    legend_raw = raw.get("legend") or {}
    if not isinstance(legend_raw, Mapping):
        raise DecisionFatalError(f"answer {name!r}.legend must be a mapping")
    return ScoreAnswer(
        score=float(score),
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
        raise DecisionFatalError("response has no answers object")
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if name not in raw_answers:
            raise DecisionFatalError(f"response is missing an answer for {name!r}")
        answers[name] = _answer(question, raw_answers[name], name)

    usage_raw = payload.get("usage") or {}
    cost = usage_raw.get("cost") if isinstance(usage_raw, Mapping) else None
    usage = Usage(
        prompt_tokens=int(usage_raw.get("input_tokens") or 0) if isinstance(usage_raw, Mapping) else 0,
        completion_tokens=0,
        calls=1,
        cost_usd=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0,
    )
    model = payload.get("model")
    return DecisionResponse(
        answers=answers,
        model=model if isinstance(model, str) and model else configured_model,
        usage=usage,
    )


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
    "DecisionRetryableError",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "Structured",
    "parse_response",
    "question_to_dict",
    "response_from_dict",
    "response_to_dict",
]
```

`src/xwalk/decide/__init__.py` for now:

```python
"""Decision models: typed questions in, calibrated probabilities out."""

from xwalk.decide.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionClient,
    DecisionError,
    DecisionFatalError,
    DecisionResponse,
    DecisionRetryableError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionClient",
    "DecisionError",
    "DecisionFatalError",
    "DecisionResponse",
    "DecisionRetryableError",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
]
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_decide_base.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass. If mypy complains about the `usage_raw` conditional expressions, restructure into an `if isinstance(usage_raw, Mapping):` block that assigns `input_tokens` and `cost` locals first.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/decide tests/test_decide_base.py
git commit -m "feat(decide): question and answer types, response parsing, client protocol

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: FakeDecider

**Files:**
- Create: `src/xwalk/decide/fake.py`
- Test: `tests/test_decide_fake.py`

**Interfaces:**
- Produces: `FakeDecider(handler: Callable[[Any, Mapping[str, Question]], Mapping[str, Answer] | BaseException] | None = None, *, model: str = "fake-decider", prompt_tokens: int = 100)`. With no handler it uses `overlap_handler`, exported from the same module, which scores each candidate by token overlap with the source. Records every call in `.calls: list[tuple[Any, Mapping[str, Question]]]`.
- `overlap_handler(state, questions) -> dict[str, Answer]`: expects `state["source"]` (any JSON) and `state["candidates"]` (mapping key -> text). A `Noul` whose instructions mention `candidates.<key>` gets the overlap of that key. A `Choice` picks the criteria key with the highest overlap, or `"NONE"` when every overlap is 0 and `"NONE"` is offered. A `Score` returns the middle level.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_fake.py`:

```python
import pytest

from xwalk.decide.base import Choice, DecisionFatalError, Noul, Score
from xwalk.decide.fake import FakeDecider, overlap_handler

STATE = {
    "source": {"fields": {"mention": "glucose"}},
    "candidates": {"C001": "ID: T1 Label: glucose", "C002": "ID: T2 Label: fructose"},
}


async def test_overlap_handler_scores_by_shared_tokens():
    fake = FakeDecider()
    response = await fake.decide(
        STATE,
        {
            "n_C001": Noul(instructions="Does `candidates.C001` denote the same entity as `source`?"),
            "n_C002": Noul(instructions="Does `candidates.C002` denote the same entity as `source`?"),
        },
    )
    assert response.answers["n_C001"].noul > response.answers["n_C002"].noul  # type: ignore[union-attr]
    assert response.usage.calls == 1
    assert len(fake.calls) == 1


async def test_choice_picks_the_best_overlap_or_none():
    fake = FakeDecider()
    best = Choice(instructions="Which?", criteria={"C001": "glucose", "C002": "fructose", "NONE": "none"})
    response = await fake.decide(STATE, {"best": best})
    assert response.answers["best"].choice == "C001"  # type: ignore[union-attr]

    nothing = {"source": {"fields": {"mention": "unobtainium"}}, "candidates": STATE["candidates"]}
    response = await fake.decide(nothing, {"best": best})
    assert response.answers["best"].choice == "NONE"  # type: ignore[union-attr]


async def test_score_returns_the_middle_level():
    fake = FakeDecider()
    response = await fake.decide(STATE, {"g": Score(instructions="?", criteria=["a", "b", "c"])})
    assert response.answers["g"].score == pytest.approx(1.0)  # type: ignore[union-attr]


async def test_a_handler_can_raise():
    fake = FakeDecider(handler=lambda state, questions: DecisionFatalError("boom"))
    with pytest.raises(DecisionFatalError):
        await fake.decide(STATE, {"n": Noul(instructions="?")})


def test_overlap_handler_is_deterministic():
    questions = {"n_C001": Noul(instructions="`candidates.C001`")}
    assert overlap_handler(STATE, questions) == overlap_handler(STATE, questions)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_fake.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'xwalk.decide.fake'`.

- [ ] **Step 3: Write `src/xwalk/decide/fake.py`**

```python
"""A scripted decision client. The reason the whole decider loop is testable offline."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from xwalk.decide.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)
from xwalk.fingerprint import hash_value
from xwalk.records import Usage

Handler = Callable[[Any, Mapping[str, Question]], "Mapping[str, Answer] | BaseException"]

_KEY_REF = re.compile(r"candidates\.([A-Za-z0-9_]+)")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return set(_TOKEN.findall(text.lower()))


def _overlap(source: Any, candidate: Any) -> float:
    a, b = _tokens(source), _tokens(candidate)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_handler(state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
    """Token-overlap scoring: enough to drive the loop, never a model of the domain."""
    source = state.get("source") if isinstance(state, Mapping) else state
    candidates: Mapping[str, Any] = (
        state.get("candidates", {}) if isinstance(state, Mapping) else {}
    )
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if isinstance(question, Noul):
            text = json.dumps(question.instructions) if not isinstance(question.instructions, str) else question.instructions
            found = _KEY_REF.search(text)
            target = candidates.get(found.group(1)) if found else state
            answers[name] = NoulAnswer(noul=round(_overlap(source, target), 4))
        elif isinstance(question, Choice):
            scores = {
                key: _overlap(source, candidates.get(key, description))
                for key, description in question.criteria.items()
                if key != "NONE"
            }
            best = max(scores, key=lambda k: (scores[k], k)) if scores else "NONE"
            if scores and scores[best] == 0.0 and "NONE" in question.criteria:
                best = "NONE"
            total = sum(scores.values())
            probabilities = (
                {k: (v / total if total else 0.0) for k, v in scores.items()}
                if best != "NONE"
                else {k: 0.0 for k in scores}
            )
            if "NONE" in question.criteria:
                probabilities["NONE"] = 1.0 - sum(probabilities.values())
            answers[name] = ChoiceAnswer(
                choice=best,
                probabilities=probabilities,
                confidence=probabilities.get(best, 0.0),
            )
        elif isinstance(question, Score):
            levels = len(question.criteria)
            middle = (levels - 1) / 2
            answers[name] = ScoreAnswer(
                score=middle,
                probabilities={str(i): (1.0 if i == int(middle) else 0.0) for i in range(levels)},
                confidence=0.5,
                legend={str(i): str(level) for i, level in enumerate(question.criteria)},
            )
    return answers


class FakeDecider:
    def __init__(
        self,
        handler: Handler | None = None,
        *,
        model: str = "fake-decider",
        prompt_tokens: int = 100,
    ) -> None:
        self._handler: Handler = handler or overlap_handler
        self._model = model
        self._prompt_tokens = prompt_tokens
        self.calls: list[tuple[Any, Mapping[str, Question]]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def fingerprint(self) -> str:
        return hash_value({"adapter": "fake-decider", "model": self._model})

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        self.calls.append((state, dict(questions)))
        outcome = self._handler(state, questions)
        if isinstance(outcome, BaseException):
            raise outcome
        return DecisionResponse(
            answers=dict(outcome),
            model=self._model,
            usage=Usage(prompt_tokens=self._prompt_tokens, completion_tokens=0, calls=1),
        )
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_decide_fake.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/decide/fake.py tests/test_decide_fake.py
git commit -m "feat(decide): FakeDecider with overlap scoring for offline tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: JevClient over httpx

**Files:**
- Create: `src/xwalk/decide/jev.py`
- Test: `tests/test_decide_jev.py`

**Interfaces:**
- Consumes: `question_to_dict`, `parse_response`, error classes from Task 2.
- Produces: `JevClient(base_url: str, model: str, *, api_key: str | None = None, timeout: float = 60.0, max_retries: int = 5, backoff_base: float = 1.0, backoff_cap: float = 30.0, transport: httpx.AsyncBaseTransport | None = None, extra_headers: Mapping[str, str] | None = None)`; `await client.aclose()`; `ADAPTER_VERSION = 1`; `RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})`.
- The request is posted to `base_url` exactly as given: OpenRouter's path is `/api/alpha/decisions`, TypeSafe's is `/v1/systemone`, and the adapter must not guess.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_jev.py`:

```python
import json

import httpx
import pytest

from xwalk.decide.base import DecisionFatalError, DecisionRetryableError, Noul
from xwalk.decide.jev import JevClient

URL = "https://openrouter.ai/api/alpha/decisions"
OK = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {"n": {"type": "noul", "noul": 0.42}},
    "usage": {"input_tokens": 50, "output_tokens": 3, "cost": 0.000002},
}


def _client(responses, **kwargs):
    """`responses` is a list of (status, json_body) consumed in order; records requests."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    client = JevClient(
        URL,
        "~typesafe/jev-latest",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
        backoff_base=0.0,
        **kwargs,
    )
    return client, seen


async def test_posts_the_wire_shape_to_the_exact_url():
    client, seen = _client([(200, OK)])
    response = await client.decide({"source": "x"}, {"n": Noul(instructions="q?")})
    assert response.answers["n"].noul == 0.42  # type: ignore[union-attr]
    assert response.model == "typesafe/jev-1.13-20260917"
    request = seen[0]
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {
        "model": "~typesafe/jev-latest",
        "state": {"source": "x"},
        "questions": {"n": {"type": "noul", "instructions": "q?"}},
    }


async def test_retries_a_429_then_succeeds():
    client, seen = _client([(429, {"error": "slow down"}), (200, OK)], max_retries=2)
    await client.decide("s", {"n": Noul(instructions="q?")})
    assert len(seen) == 2


async def test_exhausted_retries_raise_retryable():
    client, seen = _client([(503, {}), (503, {}), (503, {})], max_retries=2)
    with pytest.raises(DecisionRetryableError):
        await client.decide("s", {"n": Noul(instructions="q?")})
    assert len(seen) == 3


async def test_a_400_is_fatal_and_carries_the_message():
    client, _ = _client([(400, {"error": {"message": "Model x does not exist"}})])
    with pytest.raises(DecisionFatalError, match="does not exist"):
        await client.decide("s", {"n": Noul(instructions="q?")})


async def test_a_401_is_fatal():
    client, _ = _client([(401, {"error": "bad key"})])
    with pytest.raises(DecisionFatalError):
        await client.decide("s", {"n": Noul(instructions="q?")})


async def test_a_200_missing_an_answer_is_fatal():
    client, _ = _client([(200, {"model": "m", "answers": {}, "usage": {}})])
    with pytest.raises(DecisionFatalError, match="'n'"):
        await client.decide("s", {"n": Noul(instructions="q?")})


def test_fingerprint_excludes_the_key_and_covers_host_and_model():
    a = JevClient(URL, "m", api_key="one")
    b = JevClient(URL, "m", api_key="two")
    c = JevClient(URL, "other", api_key="one")
    d = JevClient("https://api.typesafe.ai/v1/systemone", "m", api_key="one")
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.fingerprint != d.fingerprint
    assert "one" not in a.fingerprint
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_jev.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'xwalk.decide.jev'`.

- [ ] **Step 3: Write `src/xwalk/decide/jev.py`**

```python
"""TypeSafe's Jev over HTTP, through OpenRouter or the vendor endpoint.

The request shape is the same on both; only the URL differs, so the caller passes the
full URL and this adapter appends nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import httpx

from xwalk.decide.base import (
    DecisionFatalError,
    DecisionResponse,
    DecisionRetryableError,
    Question,
    parse_response,
    question_to_dict,
)
from xwalk.fingerprint import hash_value

ADAPTER_VERSION = 1
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping) and "message" in error:
        return str(error["message"])[:500]
    if isinstance(error, str):
        return error[:500]
    return response.text[:500]


class JevClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._url = base_url
        self._model = model
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        headers = {"content-type": "application/json", **dict(extra_headers or {})}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport, headers=headers)

    @property
    def model(self) -> str:
        return self._model

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "adapter": "jev",
                "adapter_version": ADAPTER_VERSION,
                "host": urlparse(self._url).netloc,
                "path": urlparse(self._url).path,
                "model": self._model,
            }
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        body = {
            "model": self._model,
            "state": state,
            "questions": {name: question_to_dict(q) for name, q in questions.items()},
        }
        last_error = "no attempt made"
        retry_after: float | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post(self._url, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise DecisionFatalError(f"response was not JSON: {exc}") from exc
                    if not isinstance(payload, Mapping):
                        raise DecisionFatalError("response was not a JSON object")
                    return parse_response(payload, questions, configured_model=self._model)
                if response.status_code not in RETRYABLE_STATUSES:
                    raise DecisionFatalError(
                        f"HTTP {response.status_code}: {_error_message(response)}"
                    )
                last_error = f"HTTP {response.status_code}: {_error_message(response)}"
                header = response.headers.get("retry-after")
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
            if attempt < self._max_retries:
                delay = retry_after if retry_after is not None else min(
                    self._backoff_cap, self._backoff_base * 2**attempt
                )
                await asyncio.sleep(delay)
        raise DecisionRetryableError(
            f"exhausted {self._max_retries} retries: {last_error}", retry_after=retry_after
        )
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_decide_jev.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/decide/jev.py tests/test_decide_jev.py
git commit -m "feat(decide): JevClient with backoff, retry-after, and a secret-free fingerprint

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: CachingDecider

**Files:**
- Create: `src/xwalk/decide/cache.py`
- Test: `tests/test_decide_cache.py`

**Interfaces:**
- Consumes: `Ledger.get_cached(key) -> str | None`, `await Ledger.put_cached(key, text)` (existing); `response_to_dict`, `response_from_dict`, `question_to_dict` from Task 2.
- Produces: `CachingDecider(inner: DecisionClient, ledger: Ledger, *, read=True, write=True)` with `.hits`, `.misses`, and `decision_cache_key(client, state, questions) -> str`. A cache hit returns `Usage.zero()` (a replay spends nothing), exactly like `CachingLLM`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_cache.py`:

```python
import pytest

from xwalk.decide.base import Noul
from xwalk.decide.cache import CachingDecider, decision_cache_key
from xwalk.decide.fake import FakeDecider
from xwalk.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    yield led
    led.close()


STATE = {"source": {"fields": {"mention": "glucose"}}, "candidates": {"C001": "glucose"}}
QUESTIONS = {"n_C001": Noul(instructions="`candidates.C001`?")}


async def test_a_repeated_request_is_served_from_the_cache(ledger):
    inner = FakeDecider()
    cached = CachingDecider(inner, ledger)
    first = await cached.decide(STATE, QUESTIONS)
    second = await cached.decide(STATE, QUESTIONS)
    assert first.answers == second.answers
    assert len(inner.calls) == 1
    assert second.usage.calls == 0
    assert (cached.hits, cached.misses) == (1, 1)


async def test_a_different_state_misses(ledger):
    inner = FakeDecider()
    cached = CachingDecider(inner, ledger)
    await cached.decide(STATE, QUESTIONS)
    await cached.decide({**STATE, "candidates": {"C001": "fructose"}}, QUESTIONS)
    assert len(inner.calls) == 2


def test_the_key_covers_questions_and_model():
    a = FakeDecider(model="a")
    b = FakeDecider(model="b")
    other = {"n_C001": Noul(instructions="different?")}
    assert decision_cache_key(a, STATE, QUESTIONS) != decision_cache_key(b, STATE, QUESTIONS)
    assert decision_cache_key(a, STATE, QUESTIONS) != decision_cache_key(a, STATE, other)


def test_the_key_does_not_collide_on_state_boundaries():
    assert decision_cache_key(FakeDecider(), {"a": "bc"}, QUESTIONS) != decision_cache_key(
        FakeDecider(), {"ab": "c"}, QUESTIONS
    )


def test_wrapping_does_not_change_identity(ledger):
    inner = FakeDecider()
    cached = CachingDecider(inner, ledger)
    assert cached.fingerprint == inner.fingerprint
    assert cached.model == inner.model
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_cache.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/xwalk/decide/cache.py`**

```python
"""Ledger-backed decision caching.

Jev's probabilities move by a few hundredths between identical calls. A resumed run must
see the answers the first run saw, so every response is stored and replayed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from xwalk.decide.base import (
    DecisionClient,
    DecisionResponse,
    Question,
    question_to_dict,
    response_from_dict,
    response_to_dict,
)
from xwalk.fingerprint import hash_value
from xwalk.ledger import Ledger
from xwalk.records import Usage

CACHE_VERSION = 1


def decision_cache_key(client: DecisionClient, state: Any, questions: Mapping[str, Question]) -> str:
    return hash_value(
        {
            "cache_version": CACHE_VERSION,
            "kind": "decision",
            "client": client.fingerprint,
            "state": state,
            "questions": {name: question_to_dict(q) for name, q in questions.items()},
        }
    )


class CachingDecider:
    def __init__(
        self, inner: DecisionClient, ledger: Ledger, *, read: bool = True, write: bool = True
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._read = read
        self._write = write
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        key = decision_cache_key(self._inner, state, questions)
        if self._read:
            cached = self._ledger.get_cached(key)
            if cached is not None:
                self.hits += 1
                stored = response_from_dict(json.loads(cached), questions)
                return DecisionResponse(answers=stored.answers, model=stored.model, usage=Usage.zero())
        response = await self._inner.decide(state, questions)
        self.misses += 1
        if self._write:
            await self._ledger.put_cached(key, json.dumps(response_to_dict(response), ensure_ascii=False))
        return response
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_decide_cache.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/decide/cache.py tests/test_decide_cache.py
git commit -m "feat(decide): replay decisions from the ledger cache

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Properties in slots, and the QuestionSet

**Files:**
- Modify: `src/xwalk/prompts/contract.py` (`PromptSlots`)
- Create: `src/xwalk/decide/questions.py`
- Test: `tests/test_decide_questions.py`, `tests/test_prompt_contract.py` (one added test)

**Interfaces:**
- Consumes: `PromptSlots` (existing: `entity_noun`, `target_noun`, `domain_brief`, `rubric: list[RubricRow]`, `hard_rules`, `disambiguation_steps`), `Noul`, `Choice`, `Score` from Task 2.
- Produces:
  - `PropertyQuestion(BaseModel)` with `name: str` (matches `^[a-z][a-z0-9_]*$`) and `question: str`; `PromptSlots.properties: list[PropertyQuestion] = []`.
  - `NONE_KEY = "NONE"`.
  - `QuestionSet(slots: PromptSlots)` with `screen_question(key: str) -> Noul`, `choose_question(criteria: Mapping[str, str]) -> Choice` (appends `NONE`), `rubric_question() -> Score`, `property_questions() -> dict[str, Noul]` (keys are `prop_<name>`), `rubric_levels: int`, `fingerprint: str`, and `QuestionSet.from_slots(slots)`.
  - Rubric rows are ordered ascending by score for the `Score` levels, so level index `n-1` is the best row.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_questions.py`:

```python
import pytest
from pydantic import ValidationError

from xwalk.decide.base import Choice, Noul, Score
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.prompts.contract import PromptSlots

SLOTS = PromptSlots(
    entity_noun="food record",
    target_noun="FoodOn term",
    domain_brief="food composition data",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "same food, every property compatible"},
        {"score": 0.85, "name": "High", "when": "same product, incidental detail differs"},
        {"score": 0.55, "name": "Plausible", "when": "right food, processing differs"},
        {"score": 0.2, "name": "Speculative", "when": "only a broad category is shared"},
    ],
    hard_rules=["A dish is not one of its ingredients.", "Processing state is identity."],
    disambiguation_steps="Identify the core food first.",
    properties=[
        {"name": "processing_state", "question": "Do they agree on processing state?"},
        {"name": "species", "question": "Do they agree on species?"},
    ],
)


def test_screen_question_names_the_candidate_and_carries_every_hard_rule():
    q = QuestionSet.from_slots(SLOTS).screen_question("C007")
    assert isinstance(q, Noul)
    text = str(q.instructions)
    assert "`candidates.C007`" in text
    assert "`source`" in text
    for rule in SLOTS.hard_rules:
        assert rule in text
    assert q.criteria is not None and set(q.criteria) == {"true", "false"}


def test_choose_question_offers_every_key_plus_none():
    q = QuestionSet.from_slots(SLOTS).choose_question({"C001": "apple", "C002": "pear"})
    assert isinstance(q, Choice)
    assert list(q.criteria) == ["C001", "C002", NONE_KEY]
    assert "most specific" in str(q.instructions)
    assert SLOTS.disambiguation_steps in str(q.instructions)


def test_rubric_question_is_ascending_and_names_each_level():
    q = QuestionSet.from_slots(SLOTS).rubric_question()
    assert isinstance(q, Score)
    levels = [str(level) for level in q.criteria]
    assert levels[0].startswith("Speculative")
    assert levels[-1].startswith("Certain")
    assert QuestionSet.from_slots(SLOTS).rubric_levels == 4


def test_property_questions_are_prefixed_and_reference_both_sides():
    qs = QuestionSet.from_slots(SLOTS).property_questions()
    assert list(qs) == ["prop_processing_state", "prop_species"]
    assert "`candidate`" in str(qs["prop_species"].instructions)
    assert "`source`" in str(qs["prop_species"].instructions)


def test_a_rubric_with_more_than_ten_rows_is_rejected():
    rows = [{"score": 1 - i * 0.05, "name": f"L{i}", "when": "x"} for i in range(11)]
    slots = SLOTS.model_copy(update={"rubric": [type(SLOTS.rubric[0]).model_validate(r) for r in rows]})
    with pytest.raises(ValueError, match="10"):
        QuestionSet.from_slots(slots).rubric_question()


def test_property_names_are_identifiers():
    with pytest.raises(ValidationError):
        PromptSlots.model_validate(
            {**SLOTS.model_dump(), "properties": [{"name": "Bad Name", "question": "?"}]}
        )


def test_fingerprint_changes_with_slots():
    a = QuestionSet.from_slots(SLOTS)
    b = QuestionSet.from_slots(SLOTS.model_copy(update={"hard_rules": ["other"]}))
    assert a.fingerprint != b.fingerprint
```

Append to `tests/test_prompt_contract.py`:

```python
def test_slots_without_properties_still_load():
    from xwalk.prompts.contract import PromptSlots

    slots = PromptSlots(
        entity_noun="a", target_noun="b", domain_brief="c",
        rubric=[{"score": 1.0, "name": "x", "when": "y"}, {"score": 0.5, "name": "z", "when": "w"}],
    )
    assert slots.properties == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_questions.py tests/test_prompt_contract.py -q`
Expected: FAIL with `ModuleNotFoundError` and, for the contract test, `AttributeError: 'PromptSlots' object has no attribute 'properties'`.

- [ ] **Step 3: Add `properties` to `PromptSlots`**

In `src/xwalk/prompts/contract.py`, above `PromptSlots`:

```python
class PropertyQuestion(BaseModel):
    """One identity-bearing property the decider path checks on the chosen candidate."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str = Field(min_length=1)
```

and inside `PromptSlots`, after `disambiguation_steps`:

```python
    # Used only by the decider path; the LLM skeletons ignore it.
    properties: list[PropertyQuestion] = Field(default_factory=list)
```

The `PromptSet.fingerprint` already serialises the whole slots model, so slots with properties fingerprint differently, which is correct.

- [ ] **Step 4: Write `src/xwalk/decide/questions.py`**

```python
"""Questions for a decision model, composed from the same slots the LLM prompts use.

Jev is literal. Every instruction states the exact condition, names the state fields it
refers to in backticks, and spells out what true and false mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from xwalk.decide.base import Choice, Noul, Score
from xwalk.fingerprint import hash_value
from xwalk.prompts.contract import PromptSlots

NONE_KEY = "NONE"
_MAX_LEVELS = 10
_MIN_LEVELS = 2


@dataclass(frozen=True)
class QuestionSet:
    slots: PromptSlots

    @classmethod
    def from_slots(cls, slots: PromptSlots) -> QuestionSet:
        return cls(slots=slots)

    @property
    def rubric_levels(self) -> int:
        return len(self.slots.rubric)

    def _rules(self) -> str:
        return " ".join(rule.strip() for rule in self.slots.hard_rules)

    def screen_question(self, key: str) -> Noul:
        s = self.slots
        return Noul(
            instructions=(
                f"Does `candidates.{key}` denote the same entity as the {s.entity_noun} "
                f"described in `source`? The candidate is a {s.target_noun}. "
                f"Domain: {s.domain_brief.strip()} {self._rules()}"
            ).strip(),
            criteria={
                "true": (
                    f"`candidates.{key}` denotes the same entity as `source`, with every "
                    "identity-bearing property that either side states compatible"
                ),
                "false": (
                    f"`candidates.{key}` denotes a different entity, a broader or narrower "
                    "one, or a related concept that is not the same entity"
                ),
            },
        )

    def choose_question(self, criteria: Mapping[str, str]) -> Choice:
        s = self.slots
        options: dict[str, str | None] = dict(criteria)
        options[NONE_KEY] = f"no candidate denotes the same entity as the {s.entity_noun} in `source`"
        steps = s.disambiguation_steps.strip()
        return Choice(
            instructions=(
                f"Among `candidates`, select the most specific {s.target_noun} that denotes the "
                f"same entity as the {s.entity_noun} in `source` and whose stated properties the "
                f"source supports. Do not infer a property the source does not state. "
                f"{steps} {self._rules()} If no candidate denotes the same entity, choose "
                f"{NONE_KEY}."
            ).strip(),
            criteria=options,
        )

    def rubric_question(self) -> Score:
        rows = sorted(self.slots.rubric, key=lambda r: r.score)
        if not _MIN_LEVELS <= len(rows) <= _MAX_LEVELS:
            raise ValueError(
                f"a decider rubric needs {_MIN_LEVELS} to {_MAX_LEVELS} rows, got {len(rows)}"
            )
        s = self.slots
        return Score(
            instructions=(
                f"Rate how well `candidate` denotes the same entity as the {s.entity_noun} in "
                f"`source`, using the levels in order from worst to best. {self._rules()}"
            ).strip(),
            criteria=[f"{row.name}: {row.when.strip()}" for row in rows],
        )

    def property_questions(self) -> dict[str, Noul]:
        return {
            f"prop_{p.name}": Noul(
                instructions=(
                    f"Considering `source` and `candidate`: {p.question.strip()} Answer about "
                    "the two records as given; if neither states the property, answer yes."
                ),
                criteria={
                    "true": "the two records agree on this property, or neither states it",
                    "false": "the two records disagree on this property",
                },
            )
            for p in self.slots.properties
        }

    @property
    def fingerprint(self) -> str:
        return hash_value({"questions_version": 1, "slots": self.slots.model_dump(mode="json")})
```

- [ ] **Step 5: Run tests, lint, types; then the full suite**

Run: `.venv/bin/pytest tests/test_decide_questions.py tests/test_prompt_contract.py -q && .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass. `test_prompt_contract.py` and `test_author.py` must still pass with the new optional field.

- [ ] **Step 6: Commit**

```bash
git add src/xwalk/prompts/contract.py src/xwalk/decide/questions.py tests/test_decide_questions.py tests/test_prompt_contract.py
git commit -m "feat(decide): compose screen, choose, rubric, and property questions from slots

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Multi-query retrieval shared by both matchers

**Files:**
- Modify: `src/xwalk/templates.py` (`TemplateSet`)
- Create: `src/xwalk/retrieve.py`
- Modify: `src/xwalk/matcher.py` (`_retrieve` delegates)
- Modify: `src/xwalk/config.py` (`TemplateSpec`, `build_templates`)
- Test: `tests/test_retrieve.py`, `tests/test_templates.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `TemplateSet(query, context, doc, candidate, queries: tuple[str, ...] = ())`; `render_queries(record) -> list[str]` returns the rendered `queries` in order, empty strings dropped, duplicates dropped, falling back to `[render_query(record)]` when `queries` is empty; `fingerprint` covers `queries`.
  - `xwalk.retrieve.retrieve(queries: Sequence[str], source: Record, retrievers: Sequence[Retriever], store: TargetStore, *, timeout: float, rrf_k: int = 60, fallback_limit: int = 20) -> tuple[list[Candidate], list[str], bool]`. The bool is `all_failed`: every (retriever, query) search raised or timed out.
  - `TemplateSpec.query: str | None = None`, `TemplateSpec.queries: list[str] = []`; at least one must be given; `build_templates` passes `query=self.query or self.queries[0]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieve.py`:

```python
import asyncio

from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.retrieve import retrieve
from xwalk.stores.memory import MemoryStore

STORE = MemoryStore.from_source(
    [Record(id="T1", fields={"label": "glucose"}), Record(id="T2", fields={"label": "fructose"})]
)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


class Scripted:
    def __init__(self, by_query, *, name="r", fail=False, hang=False):
        self._by_query, self._name, self._fail, self._hang = by_query, name, fail, hang
        self.queries: list[str] = []

    @property
    def name(self):
        return self._name

    @property
    def fingerprint(self):
        return "s"

    @property
    def default_limit(self):
        return 10

    async def search(self, request: SearchRequest):
        self.queries.append(request.text)
        if self._fail:
            raise RetrieverError("down")
        if self._hang:
            await asyncio.sleep(3600)
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(self._by_query.get(request.text, []), start=1)
        ]


async def test_every_query_is_searched_and_fused():
    r = Scripted({"glucose": ["T1"], "dextrose": ["T2", "T1"]})
    candidates, notes, all_failed = await retrieve(
        ["glucose", "dextrose"], SOURCE, [r], STORE, timeout=1.0
    )
    assert r.queries == ["glucose", "dextrose"]
    assert [c.id for c in candidates] == ["T1", "T2"]
    assert notes == [] and all_failed is False


async def test_a_record_found_by_two_queries_outranks_one_found_once():
    r = Scripted({"a": ["T2"], "b": ["T2"], "c": ["T1"]})
    candidates, _, _ = await retrieve(["a", "b", "c"], SOURCE, [r], STORE, timeout=1.0)
    assert candidates[0].id == "T2"


async def test_a_failing_retriever_degrades_with_a_note():
    ok = Scripted({"q": ["T1"]}, name="ok")
    bad = Scripted({}, name="bad", fail=True)
    candidates, notes, all_failed = await retrieve(["q"], SOURCE, [ok, bad], STORE, timeout=1.0)
    assert [c.id for c in candidates] == ["T1"]
    assert notes == ["bad: down"] and all_failed is False


async def test_everything_failing_is_reported_as_such():
    bad = Scripted({}, fail=True)
    candidates, notes, all_failed = await retrieve(["q", "r"], SOURCE, [bad], STORE, timeout=1.0)
    assert candidates == [] and len(notes) == 2 and all_failed is True


async def test_a_hanging_retriever_times_out():
    slow = Scripted({}, name="slow", hang=True)
    _, notes, all_failed = await retrieve(["q"], SOURCE, [slow], STORE, timeout=0.05)
    assert notes == ["slow: timed out after 0.05s"] and all_failed is True
```

Append to `tests/test_templates.py`:

```python
def test_render_queries_falls_back_to_query():
    from xwalk.records import Record
    from xwalk.templates import TemplateSet

    t = TemplateSet(query="{{ a }}", context="", doc="{{ label }}", candidate="{{ label }}")
    assert t.render_queries(Record(id="s", fields={"a": "x"})) == ["x"]


def test_render_queries_drops_empty_and_duplicate_renderings():
    from xwalk.records import Record
    from xwalk.templates import TemplateSet

    t = TemplateSet(
        query="{{ a }}", context="", doc="{{ label }}", candidate="{{ label }}",
        queries=("{{ a }}", "{{ b }}", "{{ a }}", "{{ missing }}"),
    )
    assert t.render_queries(Record(id="s", fields={"a": "x", "b": "y"})) == ["x", "y"]


def test_queries_change_the_fingerprint():
    from xwalk.templates import TemplateSet

    base = dict(query="{{ a }}", context="", doc="{{ label }}", candidate="{{ label }}")
    assert TemplateSet(**base).fingerprint != TemplateSet(**base, queries=("{{ b }}",)).fingerprint
```

Append to `tests/test_config.py`:

```python
def test_templates_accept_a_queries_list(tmp_path):
    from xwalk.records import Record

    def mutate(data):
        del data["templates"]["query"]
        data["templates"]["queries"] = ["{{ mention }}", "{{ context_left }}"]

    templates = load_job(_job_copy(tmp_path, mutate)).build_templates()
    record = Record(id="s", fields={"mention": "glucose", "context_left": "blood"})
    assert templates.render_query(record) == "glucose"
    assert templates.render_queries(record) == ["glucose", "blood"]


def test_templates_need_a_query_or_queries(tmp_path):
    def mutate(data):
        del data["templates"]["query"]

    with pytest.raises(ValueError, match="query"):
        load_job(_job_copy(tmp_path, mutate))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_retrieve.py tests/test_templates.py tests/test_config.py -q`
Expected: FAIL (`ModuleNotFoundError: xwalk.retrieve`, unexpected keyword `queries`, and the config tests).

- [ ] **Step 3: Extend `TemplateSet`**

In `src/xwalk/templates.py`, add the field after `candidate` and before `_compiled`:

```python
    # Extra retrieval queries, rendered per source alongside `query`. Empty on the LLM path
    # unless the job declares them.
    queries: tuple[str, ...] = ()
```

In `__post_init__`, after the existing loop:

```python
        for index, source in enumerate(self.queries):
            try:
                self._compiled[f"queries.{index}"] = env.from_string(source)
            except JinjaSyntaxError as exc:
                raise TemplateError(f"queries[{index}] template failed to compile: {exc}") from exc
```

Add the method:

```python
    def render_queries(self, record: Record) -> list[str]:
        """Every declared query for this record, in order, non-empty, de-duplicated.

        Falls back to the single `query` template so the two paths share one call site.
        """
        if not self.queries:
            return [self.render_query(record)]
        rendered: list[str] = []
        for index in range(len(self.queries)):
            text = self._render(f"queries.{index}", record)
            if text and text not in rendered:
                rendered.append(text)
        return rendered
```

And in `fingerprint` add `"queries": list(self.queries),`.

- [ ] **Step 4: Write `src/xwalk/retrieve.py`**

```python
"""Retrieval shared by both matchers: every retriever, every query, one fused list."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.retrieval.base import Retriever, SearchRequest
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.stores.base import TargetStore


async def retrieve(
    queries: Sequence[str],
    source: Record,
    retrievers: Sequence[Retriever],
    store: TargetStore,
    *,
    timeout: float,
    rrf_k: int = 60,
    fallback_limit: int = 20,
) -> tuple[list[Candidate], list[str], bool]:
    """Search every retriever with every query concurrently and fuse by reciprocal rank.

    Returns (candidates, degradation notes, all_failed). A record surfaced by several
    queries accumulates several votes, which is the point of asking more than one.
    Depth comes from each retriever's own `default_limit`; `fallback_limit` covers a
    backend that does not declare one.
    """
    jobs = [(retriever, query) for query in queries for retriever in retrievers]

    async def one(retriever: Retriever, query: str) -> Sequence[RetrievalHit]:
        limit = getattr(retriever, "default_limit", None) or fallback_limit
        request = SearchRequest(text=query, limit=limit, source_record=source)
        return await asyncio.wait_for(retriever.search(request), timeout=timeout)

    outcomes = await asyncio.gather(*(one(r, q) for r, q in jobs), return_exceptions=True)

    groups: list[Sequence[RetrievalHit]] = []
    notes: list[str] = []
    for (retriever, _query), outcome in zip(jobs, outcomes, strict=True):
        # TimeoutError is an Exception, so it must be tested first.
        if isinstance(outcome, asyncio.TimeoutError):
            notes.append(f"{retriever.name}: timed out after {timeout}s")
        elif isinstance(outcome, BaseException):
            notes.append(f"{retriever.name}: {outcome}")
        else:
            # One vote per (retriever, query): tag the hits so fusion counts each query.
            groups.append(
                [
                    RetrievalHit(
                        record_id=h.record_id,
                        retriever=f"{h.retriever}#{_query}" if len(queries) > 1 else h.retriever,
                        raw_score=h.raw_score,
                        rank=h.rank,
                    )
                    for h in outcome
                ]
            )

    all_failed = bool(jobs) and len(notes) == len(jobs)
    if not groups:
        return [], notes, all_failed
    return reciprocal_rank_fusion(groups, store, k=rrf_k), notes, all_failed
```

Note the retriever tag: `reciprocal_rank_fusion` allows one vote per `hit.retriever`, so with several queries the tag carries the query so each query's ranking counts. With a single query the tag is untouched, so the LLM path's traces are byte-identical to before.

- [ ] **Step 5: Delegate `Matcher._retrieve`**

In `src/xwalk/matcher.py`, replace the body of `_retrieve` with:

```python
    async def _retrieve(self, query: str, source: Record) -> tuple[list[Candidate], list[str]]:
        """One query per attempt on this path; the rewriter supplies the next one."""
        candidates, notes, _ = await retrieve(
            [query],
            source,
            self._retrievers,
            self._store,
            timeout=self._policy.retriever_timeout,
            rrf_k=self._rrf_k,
            fallback_limit=self._retriever_limit,
        )
        return candidates, notes
```

Add `from xwalk.retrieve import retrieve` and remove the now-unused imports (`SearchRequest`, `reciprocal_rank_fusion`, `RetrievalHit` if unused). The "every retriever failed" check in `_attempt` (`len(notes) == len(self._retrievers)`) still holds because a single query yields one note per retriever.

- [ ] **Step 6: Extend `TemplateSpec`**

In `src/xwalk/config.py`:

```python
class TemplateSpec(BaseModel):
    query: str | None = None
    queries: list[str] = Field(default_factory=list)
    context: str = ""
    doc: str
    candidate: str

    @model_validator(mode="after")
    def _need_a_query(self) -> TemplateSpec:
        if not self.query and not self.queries:
            raise ValueError("templates need a query or a non-empty queries list")
        return self
```

and:

```python
    def build_templates(self) -> TemplateSet:
        return TemplateSet(
            query=self.templates.query or self.templates.queries[0],
            queries=tuple(self.templates.queries),
            context=self.templates.context,
            doc=self.templates.doc,
            candidate=self.templates.candidate,
        )
```

- [ ] **Step 7: Run the full suite, lint, types**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass, including every existing `tests/test_matcher.py` case.

- [ ] **Step 8: Commit**

```bash
git add src/xwalk/templates.py src/xwalk/retrieve.py src/xwalk/matcher.py src/xwalk/config.py tests/test_retrieve.py tests/test_templates.py tests/test_config.py
git commit -m "feat: retrieval over several queries per source, shared by both matchers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Screener

**Files:**
- Create: `src/xwalk/stages/screen.py`
- Test: `tests/test_screen.py`

**Interfaces:**
- Consumes: `DecisionClient`, `NoulAnswer`, `DecisionError` (Task 2); `QuestionSet.screen_question(key)` (Task 6); `TemplateSet.render_candidate` (existing); `Candidate`, `Record`, `Usage`.
- Produces:
  - `build_source_state(source: Record, context: str) -> dict[str, Any]` returning `{"fields": {...}, "context": context}` with the context key omitted when empty. Shared by all three stages.
  - `ScreenOutcome(probabilities: Mapping[str, float], shortlist: tuple[str, ...], issued: Mapping[str, str], usage: Usage, chunks: int, model: str, notes: tuple[str, ...])`. `probabilities` and `issued` are keyed by opaque key (`C001`…); `issued` maps key to record id; `shortlist` holds record ids best first.
  - `Screener(decider, questions, templates, *, chunk_size=50, shortlist_size=15, shortlist_floor=0.2, max_state_chars=120_000)` with `async screen(source, context, candidates) -> ScreenOutcome`. Keys are numbered across the whole candidate list, not per chunk. Chunks run concurrently. Any `DecisionError` propagates (the matcher decides what it means).

- [ ] **Step 1: Write the failing tests**

`tests/test_screen.py`:

```python
import pytest

from xwalk.decide.base import DecisionRetryableError, Noul, NoulAnswer
from xwalk.decide.fake import FakeDecider
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.screen import Screener, build_source_state
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="{{ note }}", doc="{{ label }}", candidate="ID: {{ id }} Label: {{ label }}"
)
QUESTIONS = QuestionSet.from_slots(
    PromptSlots(
        entity_noun="mention", target_noun="term", domain_brief="test",
        rubric=[{"score": 1.0, "name": "Certain", "when": "exact"}, {"score": 0.4, "name": "Weak", "when": "vague"}],
    )
)
SOURCE = Record(id="s1", fields={"mention": "glucose", "note": "in blood"})


def _cands(*labels):
    return [
        Candidate(
            record=Record(id=f"T{i}", fields={"label": label}),
            fused_score=1.0 / (i + 1),
            evidence=(RetrievalHit(record_id=f"T{i}", retriever="bm25", raw_score=1.0, rank=i + 1),),
        )
        for i, label in enumerate(labels)
    ]


def test_source_state_omits_an_empty_context():
    assert build_source_state(SOURCE, "") == {"fields": dict(SOURCE.fields)}
    assert build_source_state(SOURCE, "ctx") == {"fields": dict(SOURCE.fields), "context": "ctx"}


async def test_one_noul_per_candidate_with_keys_numbered_across_chunks():
    fake = FakeDecider()
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=2)
    outcome = await screener.screen(SOURCE, "in blood", _cands("glucose", "fructose", "sucrose"))
    assert outcome.chunks == 2
    assert len(fake.calls) == 2
    first_state, first_questions = fake.calls[0]
    second_state, second_questions = fake.calls[1]
    assert list(first_state["candidates"]) == ["C001", "C002"]
    assert list(second_state["candidates"]) == ["C003"]
    assert set(first_questions) == {"n_C001", "n_C002"} and set(second_questions) == {"n_C003"}
    assert all(isinstance(q, Noul) for q in first_questions.values())
    assert first_state["source"] == {"fields": dict(SOURCE.fields), "context": "in blood"}
    assert first_state["candidates"]["C001"] == "ID: T0 Label: glucose"
    assert outcome.issued == {"C001": "T0", "C002": "T1", "C003": "T2"}
    assert outcome.usage.calls == 2


async def test_shortlist_is_best_first_and_floored():
    screener = Screener(FakeDecider(), QUESTIONS, TEMPLATES, shortlist_size=5, shortlist_floor=0.1)
    outcome = await screener.screen(SOURCE, "", _cands("fructose", "glucose", "sucrose"))
    assert outcome.shortlist[0] == "T1"
    assert all(outcome.probabilities[k] >= 0.1 for k, rid in outcome.issued.items() if rid in outcome.shortlist)


async def test_shortlist_is_capped():
    screener = Screener(FakeDecider(), QUESTIONS, TEMPLATES, shortlist_size=1, shortlist_floor=0.0)
    outcome = await screener.screen(SOURCE, "", _cands("glucose", "glucose syrup", "sucrose"))
    assert len(outcome.shortlist) == 1


async def test_no_candidates_means_no_calls():
    fake = FakeDecider()
    outcome = await Screener(fake, QUESTIONS, TEMPLATES).screen(SOURCE, "", [])
    assert outcome.shortlist == () and outcome.chunks == 0 and fake.calls == []


async def test_provider_errors_propagate():
    fake = FakeDecider(handler=lambda s, q: DecisionRetryableError("busy"))
    with pytest.raises(DecisionRetryableError):
        await Screener(fake, QUESTIONS, TEMPLATES).screen(SOURCE, "", _cands("glucose"))


async def test_an_oversized_chunk_is_split_with_a_note():
    fake = FakeDecider()
    screener = Screener(fake, QUESTIONS, TEMPLATES, chunk_size=3, max_state_chars=80)
    outcome = await screener.screen(SOURCE, "", _cands("glucose", "fructose", "sucrose"))
    assert outcome.chunks >= 2
    assert any("split" in note for note in outcome.notes)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_screen.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'xwalk.stages.screen'`.

- [ ] **Step 3: Write `src/xwalk/stages/screen.py`**

```python
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
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_screen.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/stages/screen.py tests/test_screen.py
git commit -m "feat(decide): screen stage, one noul per candidate in concurrent chunks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Chooser

**Files:**
- Create: `src/xwalk/stages/choose.py`
- Test: `tests/test_choose.py`

**Interfaces:**
- Consumes: `build_source_state` (Task 8), `QuestionSet.choose_question`, `NONE_KEY` (Task 6), `ChoiceAnswer`, `Resolution` (existing `xwalk.stages.keying`).
- Produces: `ChooseOutcome(record_id: str | None, resolution: Resolution, p_choice: float, p_none: float, confidence: float, raw: str, usage: Usage, model: str)` and `Chooser(decider, questions, templates)` with `async choose(source, context, shortlist: Sequence[Candidate]) -> ChooseOutcome`. `p_choice` is the probability of the chosen key when resolved, or of `NONE` when abstaining. An answer naming an unissued key cannot happen after `parse_response` (it rejects choices outside the criteria), but the stage still guards it and returns `Resolution.UNRESOLVED`, because that invariant belongs here, not only in the adapter.

- [ ] **Step 1: Write the failing tests**

`tests/test_choose.py`:

```python
from xwalk.decide.base import ChoiceAnswer, DecisionResponse
from xwalk.decide.fake import FakeDecider
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit, Usage
from xwalk.stages.choose import Chooser
from xwalk.stages.keying import Resolution
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(query="{{ mention }}", context="", doc="{{ label }}", candidate="ID: {{ id }} Label: {{ label }}")
QUESTIONS = QuestionSet.from_slots(
    PromptSlots(
        entity_noun="mention", target_noun="term", domain_brief="test",
        rubric=[{"score": 1.0, "name": "Certain", "when": "exact"}, {"score": 0.4, "name": "Weak", "when": "vague"}],
    )
)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


def _cands(*labels):
    return [
        Candidate(
            record=Record(id=f"T{i}", fields={"label": label}), fused_score=1.0,
            evidence=(RetrievalHit(record_id=f"T{i}", retriever="bm25", raw_score=1.0, rank=i + 1),),
        )
        for i, label in enumerate(labels)
    ]


async def test_chooses_by_key_and_resolves_to_the_record():
    fake = FakeDecider()
    outcome = await Chooser(fake, QUESTIONS, TEMPLATES).choose(SOURCE, "", _cands("fructose", "glucose"))
    assert outcome.record_id == "T1"
    assert outcome.resolution is Resolution.EXACT_KEY
    assert outcome.p_choice > 0.5 and outcome.p_none < 0.5
    state, questions = fake.calls[0]
    assert list(state["candidates"]) == ["C01", "C02"]
    assert list(questions) == ["best"]
    assert list(questions["best"].criteria) == ["C01", "C02", NONE_KEY]


async def test_none_is_an_abstention():
    fake = FakeDecider()
    outcome = await Chooser(fake, QUESTIONS, TEMPLATES).choose(
        Record(id="s", fields={"mention": "unobtainium"}), "", _cands("glucose")
    )
    assert outcome.record_id is None
    assert outcome.resolution is Resolution.ABSTAIN
    assert outcome.p_none == outcome.p_choice


async def test_an_unissued_key_is_unresolved_not_a_record():
    def handler(state, questions):
        return {"best": ChoiceAnswer(choice="C99", probabilities={"C99": 1.0}, confidence=1.0)}

    outcome = await Chooser(FakeDecider(handler=handler), QUESTIONS, TEMPLATES).choose(
        SOURCE, "", _cands("glucose")
    )
    assert outcome.record_id is None
    assert outcome.resolution is Resolution.UNRESOLVED


async def test_empty_shortlist_abstains_without_a_call():
    fake = FakeDecider()
    outcome = await Chooser(fake, QUESTIONS, TEMPLATES).choose(SOURCE, "", [])
    assert outcome.resolution is Resolution.ABSTAIN and fake.calls == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_choose.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/xwalk/stages/choose.py`**

```python
"""Choose: the comparative question over the screen's survivors.

Near-duplicate ontology labels split a Choice's probability mass, which is why this
runs only over the shortlist and why the screen probability, not `p_choice`, is the
calibrated number the policy accepts on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.decide.base import ChoiceAnswer, DecisionClient
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.keying import Resolution
from xwalk.stages.screen import build_source_state
from xwalk.templates import TemplateSet


@dataclass(frozen=True)
class ChooseOutcome:
    record_id: str | None
    resolution: Resolution
    p_choice: float
    p_none: float
    confidence: float
    raw: str
    usage: Usage
    model: str


class Chooser:
    def __init__(
        self, decider: DecisionClient, questions: QuestionSet, templates: TemplateSet
    ) -> None:
        self._decider = decider
        self._questions = questions
        self._templates = templates

    async def choose(
        self, source: Record, context: str, shortlist: Sequence[Candidate]
    ) -> ChooseOutcome:
        if not shortlist:
            return ChooseOutcome(
                None, Resolution.ABSTAIN, 1.0, 1.0, 1.0, "", Usage.zero(), self._decider.model
            )
        width = max(2, len(str(len(shortlist))))
        keyed = {
            f"C{i:0{width}d}": self._templates.render_candidate(c.record)
            for i, c in enumerate(shortlist, start=1)
        }
        issued = {key: c.id for key, c in zip(keyed, shortlist, strict=True)}
        state = {"source": build_source_state(source, context), "candidates": keyed}
        question = self._questions.choose_question(keyed)
        response = await self._decider.decide(state, {"best": question})
        answer = response.answers["best"]
        assert isinstance(answer, ChoiceAnswer)
        raw = json.dumps(
            {
                "model": response.model,
                "choice": answer.choice,
                "probabilities": dict(answer.probabilities),
                "confidence": answer.confidence,
            },
            ensure_ascii=False,
        )
        p_none = answer.probabilities.get(NONE_KEY, 0.0)
        if answer.choice == NONE_KEY:
            return ChooseOutcome(
                None, Resolution.ABSTAIN, p_none, p_none, answer.confidence, raw, response.usage, response.model
            )
        if answer.choice not in issued:
            # Unreachable through parse_response, kept because the invariant lives here.
            return ChooseOutcome(
                None, Resolution.UNRESOLVED, 0.0, p_none, answer.confidence, raw, response.usage, response.model
            )
        return ChooseOutcome(
            issued[answer.choice],
            Resolution.EXACT_KEY,
            answer.probabilities.get(answer.choice, 0.0),
            p_none,
            answer.confidence,
            raw,
            response.usage,
            response.model,
        )
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_choose.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/stages/choose.py tests/test_choose.py
git commit -m "feat(decide): choose stage over the shortlist with NONE as abstention

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: PropertyGate

**Files:**
- Create: `src/xwalk/stages/property_gate.py`
- Test: `tests/test_property_gate.py`

**Interfaces:**
- Consumes: `build_source_state` (Task 8), `QuestionSet.rubric_question`, `QuestionSet.property_questions` (Task 6), `ScoreAnswer`, `NoulAnswer`.
- Produces: `GateOutcome(rubric_score: float, rubric_confidence: float, rubric_levels: int, properties: Mapping[str, float], raw: str, usage: Usage, model: str)` where `properties` is keyed by the bare property name (`processing_state`, not `prop_processing_state`), and `PropertyGate(decider, questions, templates)` with `async gate(source, context, chosen: Candidate) -> GateOutcome`. The state is `{"source": ..., "candidate": <rendered candidate>}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_property_gate.py`:

```python
from xwalk.decide.fake import FakeDecider
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.property_gate import PropertyGate
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(query="{{ mention }}", context="", doc="{{ label }}", candidate="ID: {{ id }} Label: {{ label }}")
SLOTS = PromptSlots(
    entity_noun="mention", target_noun="term", domain_brief="test",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact"},
        {"score": 0.7, "name": "High", "when": "close"},
        {"score": 0.4, "name": "Weak", "when": "vague"},
    ],
    properties=[{"name": "processing_state", "question": "Same processing state?"}],
)
CHOSEN = Candidate(
    record=Record(id="T1", fields={"label": "glucose"}), fused_score=1.0,
    evidence=(RetrievalHit(record_id="T1", retriever="bm25", raw_score=1.0, rank=1),),
)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


async def test_asks_the_rubric_and_every_property_in_one_call():
    fake = FakeDecider()
    outcome = await PropertyGate(fake, QuestionSet.from_slots(SLOTS), TEMPLATES).gate(SOURCE, "", CHOSEN)
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert state == {"source": {"fields": {"mention": "glucose"}}, "candidate": "ID: T1 Label: glucose"}
    assert set(questions) == {"rubric", "prop_processing_state"}
    assert outcome.rubric_levels == 3
    assert 0.0 <= outcome.rubric_score <= 2.0
    assert list(outcome.properties) == ["processing_state"]
    assert outcome.usage.calls == 1
    assert "rubric" in outcome.raw


async def test_no_properties_means_only_the_rubric():
    fake = FakeDecider()
    slots = SLOTS.model_copy(update={"properties": []})
    outcome = await PropertyGate(fake, QuestionSet.from_slots(slots), TEMPLATES).gate(SOURCE, "", CHOSEN)
    assert set(fake.calls[0][1]) == {"rubric"}
    assert outcome.properties == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_property_gate.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/xwalk/stages/property_gate.py`**

```python
"""Gate: the absolute question about the one chosen record, with a clean state.

Returns the rubric level and one agreement probability per declared property. Those
per-property numbers are what replaces a generated explanation for a reviewer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from xwalk.decide.base import DecisionClient, NoulAnswer, Question, ScoreAnswer
from xwalk.decide.questions import QuestionSet
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.screen import build_source_state
from xwalk.templates import TemplateSet

_PREFIX = "prop_"


@dataclass(frozen=True)
class GateOutcome:
    rubric_score: float
    rubric_confidence: float
    rubric_levels: int
    properties: Mapping[str, float]
    raw: str
    usage: Usage
    model: str


class PropertyGate:
    def __init__(
        self, decider: DecisionClient, questions: QuestionSet, templates: TemplateSet
    ) -> None:
        self._decider = decider
        self._questions = questions
        self._templates = templates

    async def gate(self, source: Record, context: str, chosen: Candidate) -> GateOutcome:
        state = {
            "source": build_source_state(source, context),
            "candidate": self._templates.render_candidate(chosen.record),
        }
        questions: dict[str, Question] = {"rubric": self._questions.rubric_question()}
        questions.update(self._questions.property_questions())
        response = await self._decider.decide(state, questions)

        rubric = response.answers["rubric"]
        assert isinstance(rubric, ScoreAnswer)
        properties: dict[str, float] = {}
        for name, answer in response.answers.items():
            if name.startswith(_PREFIX):
                assert isinstance(answer, NoulAnswer)
                properties[name[len(_PREFIX) :]] = answer.noul

        raw = json.dumps(
            {
                "model": response.model,
                "rubric": {
                    "score": rubric.score,
                    "confidence": rubric.confidence,
                    "probabilities": dict(rubric.probabilities),
                    "legend": dict(rubric.legend),
                },
                "properties": properties,
            },
            ensure_ascii=False,
        )
        return GateOutcome(
            rubric_score=rubric.score,
            rubric_confidence=rubric.confidence,
            rubric_levels=self._questions.rubric_levels,
            properties=properties,
            raw=raw,
            usage=response.usage,
            model=response.model,
        )
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_property_gate.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/stages/property_gate.py tests/test_property_gate.py
git commit -m "feat(decide): property gate with the rubric as a Score and one noul per property

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: DecisionPolicy, Signals, derive_status, render_explanation

**Files:**
- Create: `src/xwalk/decide/policy.py`
- Test: `tests/test_decide_policy.py`

**Interfaces:**
- Consumes: `MatchStatus`, `DecisionReason` (existing), `Resolution` (existing).
- Produces:
  - `DecisionPolicy` frozen dataclass: `screen_floor=0.30, shortlist_size=15, shortlist_floor=0.20, none_at=0.70, choose_at=0.50, accept_at=0.85, rubric_floor: float | None = None, property_floor=0.50, chunk_size=50, max_candidates=300, concurrency=32, retriever_timeout=60.0`. Validation: every probability field in `[0, 1]`; `shortlist_floor <= accept_at`; `shortlist_size`, `chunk_size`, `max_candidates`, `concurrency` at least 1.
  - `Signals` frozen dataclass: `candidate_count: int, retrieval_failed: bool, screen_best: float | None, screen_chosen: float | None, resolution: str | None, p_choice: float | None, p_none: float | None, choice_confidence: float | None, rubric: float | None, rubric_levels: int | None, rubric_confidence: float | None, properties: Mapping[str, float]`; `flat() -> dict[str, float]` (only the numeric fields that are set; properties as `prop_<name>`); `Signals.from_flat(flat: Mapping[str, float], *, candidate_count: int, retrieval_failed: bool, resolution: str | None) -> Signals`.
  - `effective_rubric_floor(policy, levels) -> float`: `policy.rubric_floor` if set, else `max(0, levels - 2)`.
  - `derive_status(signals, policy) -> tuple[MatchStatus, DecisionReason]` implementing the seven rules in the spec.
  - `render_explanation(signals) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_policy.py`:

```python
import pytest

from xwalk.decide.policy import (
    DecisionPolicy,
    Signals,
    derive_status,
    effective_rubric_floor,
    render_explanation,
)
from xwalk.records import DecisionReason, MatchStatus

POLICY = DecisionPolicy()


def _signals(**overrides):
    base = dict(
        candidate_count=10, retrieval_failed=False, screen_best=0.95, screen_chosen=0.95,
        resolution="exact_key", p_choice=0.8, p_none=0.02, choice_confidence=0.7,
        rubric=2.7, rubric_levels=4, rubric_confidence=0.8,
        properties={"processing_state": 0.9, "species": 0.95},
    )
    base.update(overrides)
    return Signals(**base)


def test_policy_validates_ranges():
    with pytest.raises(ValueError):
        DecisionPolicy(accept_at=1.5)
    with pytest.raises(ValueError):
        DecisionPolicy(shortlist_floor=0.9, accept_at=0.8)
    with pytest.raises(ValueError):
        DecisionPolicy(chunk_size=0)


def test_rubric_floor_defaults_to_the_second_best_level():
    assert effective_rubric_floor(POLICY, 4) == 2
    assert effective_rubric_floor(POLICY, 2) == 0
    assert effective_rubric_floor(DecisionPolicy(rubric_floor=1.5), 4) == 1.5


@pytest.mark.parametrize(
    "overrides, status, reason",
    [
        ({"retrieval_failed": True, "candidate_count": 0}, MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE),
        ({"candidate_count": 0, "screen_best": None}, MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES),
        ({"screen_best": 0.1, "screen_chosen": None, "resolution": None}, MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR),
        ({"resolution": "abstain", "screen_chosen": None, "p_none": 0.9, "p_choice": 0.9}, MatchStatus.UNMATCHED, DecisionReason.SELECTOR_ABSTAINED),
        ({"resolution": "abstain", "screen_chosen": None, "p_none": 0.5, "p_choice": 0.5}, MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT),
        ({"resolution": "unresolved", "screen_chosen": None}, MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT),
        ({}, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD),
        ({"screen_chosen": 0.7}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        ({"p_choice": 0.3}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        ({"rubric": 1.2}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        ({"properties": {"species": 0.2}}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
        ({"properties": {}}, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD),
        ({"rubric": None, "rubric_levels": None}, MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD),
    ],
)
def test_derive_status_table(overrides, status, reason):
    assert derive_status(_signals(**overrides), POLICY) == (status, reason)


def test_flat_round_trips():
    s = _signals()
    back = Signals.from_flat(
        s.flat(), candidate_count=s.candidate_count, retrieval_failed=False, resolution=s.resolution
    )
    assert back == s
    assert "prop_species" in s.flat()
    assert "rubric_levels" in s.flat()


def test_flat_omits_unset_fields():
    assert "p_choice" not in _signals(p_choice=None).flat()


def test_render_explanation_is_deterministic_and_readable():
    text = render_explanation(_signals())
    assert text == render_explanation(_signals())
    assert "same 0.95" in text
    assert "p_choice 0.80" in text
    assert "rubric 2.7/3" in text
    assert "processing_state 0.90" in text
    assert render_explanation(_signals(screen_best=None, candidate_count=0)) == "no candidates"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_policy.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/xwalk/decide/policy.py`**

```python
"""Policy for the decider path: thresholds on calibrated probabilities, and nothing else.

The model returns probabilities; this module turns them into one of four outcomes. The
defaults are starting points from the September 2026 probe. `xwalk fit` replaces them
with values fitted on gold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from xwalk.records import DecisionReason, MatchStatus
from xwalk.stages.keying import Resolution

_PROP = "prop_"


@dataclass(frozen=True)
class DecisionPolicy:
    screen_floor: float = 0.30
    shortlist_size: int = 15
    shortlist_floor: float = 0.20
    none_at: float = 0.70
    choose_at: float = 0.50
    accept_at: float = 0.85
    rubric_floor: float | None = None
    property_floor: float = 0.50
    chunk_size: int = 50
    max_candidates: int = 300
    concurrency: int = 32
    retriever_timeout: float = 60.0

    def __post_init__(self) -> None:
        for name in ("screen_floor", "shortlist_floor", "none_at", "choose_at", "accept_at", "property_floor"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.shortlist_floor > self.accept_at:
            raise ValueError(
                f"shortlist_floor ({self.shortlist_floor}) must not exceed accept_at ({self.accept_at})"
            )
        for name in ("shortlist_size", "chunk_size", "max_candidates", "concurrency"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")


@dataclass(frozen=True)
class Signals:
    """Everything the decider path learned about one record, as numbers."""

    candidate_count: int
    retrieval_failed: bool
    screen_best: float | None = None
    screen_chosen: float | None = None
    resolution: str | None = None
    p_choice: float | None = None
    p_none: float | None = None
    choice_confidence: float | None = None
    rubric: float | None = None
    rubric_levels: int | None = None
    rubric_confidence: float | None = None
    properties: Mapping[str, float] = field(default_factory=dict)

    _NUMERIC = (
        "screen_best", "screen_chosen", "p_choice", "p_none", "choice_confidence",
        "rubric", "rubric_levels", "rubric_confidence",
    )

    def flat(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self._NUMERIC:
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        for name, value in sorted(self.properties.items()):
            out[f"{_PROP}{name}"] = float(value)
        return out

    @classmethod
    def from_flat(
        cls,
        flat: Mapping[str, float],
        *,
        candidate_count: int,
        retrieval_failed: bool,
        resolution: str | None,
    ) -> Signals:
        levels = flat.get("rubric_levels")
        return cls(
            candidate_count=candidate_count,
            retrieval_failed=retrieval_failed,
            screen_best=flat.get("screen_best"),
            screen_chosen=flat.get("screen_chosen"),
            resolution=resolution,
            p_choice=flat.get("p_choice"),
            p_none=flat.get("p_none"),
            choice_confidence=flat.get("choice_confidence"),
            rubric=flat.get("rubric"),
            rubric_levels=None if levels is None else int(levels),
            rubric_confidence=flat.get("rubric_confidence"),
            properties={k[len(_PROP) :]: v for k, v in flat.items() if k.startswith(_PROP)},
        )


def effective_rubric_floor(policy: DecisionPolicy, levels: int) -> float:
    if policy.rubric_floor is not None:
        return policy.rubric_floor
    return float(max(0, levels - 2))


def derive_status(signals: Signals, policy: DecisionPolicy) -> tuple[MatchStatus, DecisionReason]:
    if signals.retrieval_failed:
        return MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE
    if signals.candidate_count == 0 or signals.screen_best is None:
        return MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES
    if signals.screen_best < policy.screen_floor:
        return MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR
    if signals.resolution == Resolution.ABSTAIN.value:
        if (signals.p_none or 0.0) >= policy.none_at:
            return MatchStatus.UNMATCHED, DecisionReason.SELECTOR_ABSTAINED
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT
    if signals.resolution != Resolution.EXACT_KEY.value or signals.screen_chosen is None:
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT

    accepted = (
        signals.screen_chosen >= policy.accept_at
        and (signals.p_choice or 0.0) >= policy.choose_at
        and signals.rubric is not None
        and signals.rubric_levels is not None
        and signals.rubric >= effective_rubric_floor(policy, signals.rubric_levels)
        and all(p >= policy.property_floor for p in signals.properties.values())
    )
    if accepted:
        return MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD
    return MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD


def render_explanation(signals: Signals) -> str:
    """A reviewer-facing line built only from numbers, so it is the same every run."""
    if signals.candidate_count == 0 or signals.screen_best is None:
        return "no candidates"
    parts = [f"same {signals.screen_best:.2f}"]
    if signals.screen_chosen is not None and signals.screen_chosen != signals.screen_best:
        parts.append(f"chosen {signals.screen_chosen:.2f}")
    if signals.resolution == Resolution.ABSTAIN.value:
        parts.append(f"NONE {signals.p_none or 0.0:.2f}")
    elif signals.p_choice is not None:
        parts.append(f"p_choice {signals.p_choice:.2f}")
        if signals.p_none is not None:
            parts.append(f"p_none {signals.p_none:.2f}")
    if signals.rubric is not None and signals.rubric_levels:
        parts.append(f"rubric {signals.rubric:.1f}/{signals.rubric_levels - 1}")
    for name, value in sorted(signals.properties.items()):
        parts.append(f"{name} {value:.2f}")
    return "; ".join(parts)
```

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_decide_policy.py -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass. If mypy rejects the class-level `_NUMERIC` tuple inside the frozen dataclass, move it to a module-level `_NUMERIC` constant and reference that.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/decide/policy.py tests/test_decide_policy.py
git commit -m "feat(decide): thresholds on calibrated signals and a numeric explanation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: DecisionMatcher

**Files:**
- Create: `src/xwalk/decide/matcher.py`
- Modify: `src/xwalk/decide/__init__.py` (re-exports)
- Test: `tests/test_decide_matcher.py`

**Interfaces:**
- Consumes: `retrieve` (Task 7), `Screener`/`ScreenOutcome` (Task 8), `Chooser` (Task 9), `PropertyGate` (Task 10), `DecisionPolicy`, `Signals`, `derive_status`, `render_explanation` (Task 11), `DecisionError`, `DecisionFatalError` (Task 2), `TemplateSet.render_queries` and `render_context`.
- Produces: `DecisionMatcher(*, templates, retrievers, store, screener, chooser, gate, policy: DecisionPolicy | None = None, run_fingerprint: str = "", rrf_k: int = 60, keep_candidates_in_trace: bool = True)` with `run_fingerprint`, `policy`, `store_fingerprint`, `async match(source) -> MatchResult`, `match_sync(source)`.
- One `Attempt` per result, `index=0`, `query` = the first rendered query, `issued_keys` from the screen, `raw_selection` = JSON of choose and gate raw answers plus served model, `primary_score` = screen probability of the chosen record, `signals` filled, `explanation` rendered. `MatchResult.confidence` = screen probability of the chosen record; `MatchResult.signals` = the same flat dict.
- Error mapping: `DecisionFatalError` propagates (the batch stops, as with `LLMFatalError`); any other `DecisionError` becomes a `failed` / `provider_failure` result with the error text in `Attempt.error`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_matcher.py`:

```python
import asyncio

import pytest

from xwalk.decide.base import DecisionFatalError, DecisionRetryableError
from xwalk.decide.fake import FakeDecider
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import DecisionReason, MatchStatus, Record, RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.stages.choose import Chooser
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.screen import Screener
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="{{ note }}", doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}", queries=("{{ mention }}", "{{ alias }}"),
)
QUESTIONS = QuestionSet.from_slots(
    PromptSlots(
        entity_noun="mention", target_noun="term", domain_brief="test",
        rubric=[{"score": 1.0, "name": "Certain", "when": "exact"}, {"score": 0.4, "name": "Weak", "when": "vague"}],
        properties=[{"name": "form", "question": "Same form?"}],
    )
)
STORE = MemoryStore.from_source(
    [
        Record(id="T1", fields={"label": "glucose"}),
        Record(id="T2", fields={"label": "fructose"}),
        Record(id="T3", fields={"label": "sucrose"}),
    ]
)
SOURCE = Record(id="s1", fields={"mention": "glucose", "alias": "dextrose", "note": ""})


class Scripted:
    def __init__(self, by_query, *, name="bm25", fail=False):
        self._by_query, self._name, self._fail = by_query, name, fail
        self.queries: list[str] = []

    @property
    def name(self):
        return self._name

    @property
    def fingerprint(self):
        return "s"

    @property
    def default_limit(self):
        return 10

    async def search(self, request: SearchRequest):
        self.queries.append(request.text)
        if self._fail:
            raise RetrieverError("index unavailable")
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(self._by_query.get(request.text, []), start=1)
        ]


def confident(state, questions):
    """Says yes to any candidate whose text contains "glucose", no to the rest."""
    from xwalk.decide.base import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

    answers = {}
    candidates = state.get("candidates", {})
    for name, q in questions.items():
        if isinstance(q, Noul):
            key = name[len("n_"):] if name.startswith("n_") else None
            text = candidates.get(key, state.get("candidate", "")) if key else state.get("candidate", "")
            answers[name] = NoulAnswer(noul=0.95 if "glucose" in str(text) else 0.05)
        elif isinstance(q, Choice):
            hits = [k for k in q.criteria if "glucose" in str(candidates.get(k, ""))]
            choice = hits[0] if hits else "NONE"
            answers[name] = ChoiceAnswer(
                choice=choice,
                probabilities={k: (0.9 if k == choice else 0.1 / max(1, len(q.criteria) - 1)) for k in q.criteria},
                confidence=0.85,
            )
        elif isinstance(q, Score):
            top = len(q.criteria) - 1
            answers[name] = ScoreAnswer(
                score=float(top), probabilities={str(i): (1.0 if i == top else 0.0) for i in range(top + 1)},
                confidence=0.9, legend={str(i): str(c) for i, c in enumerate(q.criteria)},
            )
    return answers


def _matcher(decider=None, retrievers=None, policy=None):
    decider = decider or FakeDecider(handler=confident)
    return DecisionMatcher(
        templates=TEMPLATES,
        retrievers=retrievers or [Scripted({"glucose": ["T1", "T2"], "dextrose": ["T3"]})],
        store=STORE,
        screener=Screener(decider, QUESTIONS, TEMPLATES, chunk_size=2),
        chooser=Chooser(decider, QUESTIONS, TEMPLATES),
        gate=PropertyGate(decider, QUESTIONS, TEMPLATES),
        policy=policy or DecisionPolicy(accept_at=0.5, screen_floor=0.1, shortlist_floor=0.0, property_floor=0.0, rubric_floor=0.0),
        run_fingerprint="fp",
    )


async def test_the_confident_case_matches_with_signals_and_a_numeric_explanation():
    decider = FakeDecider(handler=confident)
    retriever = Scripted({"glucose": ["T1", "T2"], "dextrose": ["T3"]})
    result = await _matcher(decider, [retriever]).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]
    assert result.status is MatchStatus.MATCHED
    assert result.matched_id == "T1"
    assert result.reason is DecisionReason.ACCEPT_THRESHOLD
    assert result.confidence == result.signals["screen_chosen"]
    assert "prop_form" in result.signals
    assert result.explanation.startswith("same ")
    attempt = result.attempts[0]
    assert attempt.chosen_id == "T1" and attempt.resolution == "exact_key"
    assert attempt.issued_keys == {"C001": "T1", "C002": "T2", "C003": "T3"}
    assert attempt.primary_score == result.confidence
    assert '"choose"' in (attempt.raw_selection or "") and '"gate"' in (attempt.raw_selection or "")
    # 2 screen chunks + choose + gate
    assert result.usage.calls == 4 and len(decider.calls) == 4
    assert len(result.candidates) == 3
    # shortlist screen probabilities are kept by key so recall can be measured later
    assert result.signals["screen_C001"] == 0.95


async def test_nothing_similar_is_unmatched_without_choose_or_gate():
    decider = FakeDecider(handler=confident)
    retriever = Scripted({"unobtainium": ["T2", "T3"]})
    result = await _matcher(decider, [retriever]).match(
        Record(id="s", fields={"mention": "unobtainium", "alias": "", "note": ""})
    )
    assert result.status is MatchStatus.UNMATCHED
    assert result.reason is DecisionReason.BELOW_REVIEW_FLOOR
    assert result.matched_id is None
    assert len(decider.calls) == 1  # one screen chunk of two candidates; no choose, no gate


async def test_no_candidates():
    result = await _matcher(retrievers=[Scripted({})]).match(SOURCE)
    assert result.status is MatchStatus.UNMATCHED and result.reason is DecisionReason.NO_CANDIDATES


async def test_every_retriever_failing_is_failed_not_unmatched():
    result = await _matcher(retrievers=[Scripted({}, fail=True)]).match(SOURCE)
    assert result.status is MatchStatus.FAILED and result.reason is DecisionReason.RETRIEVER_FAILURE


async def test_a_retryable_provider_error_is_a_failed_result():
    decider = FakeDecider(handler=lambda s, q: DecisionRetryableError("exhausted"))
    # a handler that raises is still a FakeDecider; the matcher maps the error
    result = await _matcher(decider).match(SOURCE)
    assert result.status is MatchStatus.FAILED and result.reason is DecisionReason.PROVIDER_FAILURE
    assert "exhausted" in (result.attempts[0].error or "")


async def test_a_fatal_provider_error_propagates():
    decider = FakeDecider(handler=lambda s, q: DecisionFatalError("bad key"))
    with pytest.raises(DecisionFatalError):
        await _matcher(decider).match(SOURCE)


async def test_max_candidates_caps_what_the_screen_sees():
    decider = FakeDecider(handler=confident)
    policy = DecisionPolicy(max_candidates=2, accept_at=0.5, screen_floor=0.1, shortlist_floor=0.0, property_floor=0.0, rubric_floor=0.0)
    result = await _matcher(decider, policy=policy).match(SOURCE)
    assert result.attempts[0].candidate_count == 2
    assert result.attempts[0].candidates_truncated == 1


def test_match_sync_runs_the_loop():
    result = _matcher().match_sync(SOURCE)
    assert result.status is MatchStatus.MATCHED
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_matcher.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/xwalk/decide/matcher.py`**

```python
"""The decider loop: retrieve wide, screen everything, choose among survivors, gate the choice.

No retries: the LLM path retries because 25 candidates were the wrong 25. This path
retrieves 300 instead. One attempt per record, always index 0.
"""

from __future__ import annotations

import asyncio
import json
import time

from collections.abc import Sequence

from xwalk.decide.base import DecisionError, DecisionFatalError
from xwalk.decide.policy import DecisionPolicy, Signals, derive_status, render_explanation
from xwalk.fingerprint import hash_record, result_key
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    Usage,
)
from xwalk.retrieval.base import Retriever
from xwalk.retrieve import retrieve
from xwalk.stages.choose import ChooseOutcome, Chooser
from xwalk.stages.keying import Resolution
from xwalk.stages.property_gate import GateOutcome, PropertyGate
from xwalk.stages.screen import Screener, ScreenOutcome
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet


class DecisionMatcher:
    def __init__(
        self,
        *,
        templates: TemplateSet,
        retrievers: Sequence[Retriever],
        store: TargetStore,
        screener: Screener,
        chooser: Chooser,
        gate: PropertyGate,
        policy: DecisionPolicy | None = None,
        run_fingerprint: str = "",
        rrf_k: int = 60,
        keep_candidates_in_trace: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self._templates = templates
        self._retrievers = list(retrievers)
        self._store = store
        self._screener = screener
        self._chooser = chooser
        self._gate = gate
        self._policy = policy or DecisionPolicy()
        self._run_fingerprint = run_fingerprint
        self._rrf_k = rrf_k
        self._keep_candidates = keep_candidates_in_trace

    @property
    def run_fingerprint(self) -> str:
        return self._run_fingerprint

    @property
    def policy(self) -> DecisionPolicy:
        return self._policy

    @property
    def store_fingerprint(self) -> str:
        return self._store.fingerprint

    async def match(self, source: Record) -> MatchResult:
        started = time.perf_counter()
        context = self._templates.render_context(source)
        queries = self._templates.render_queries(source)

        fused, notes, all_failed = await retrieve(
            queries,
            source,
            self._retrievers,
            self._store,
            timeout=self._policy.retriever_timeout,
            rrf_k=self._rrf_k,
        )
        candidates = fused[: self._policy.max_candidates]
        truncated = len(fused) - len(candidates)

        screen: ScreenOutcome | None = None
        chosen: ChooseOutcome | None = None
        gated: GateOutcome | None = None
        usage = Usage.zero()
        error: str | None = None
        provider_failed = False

        try:
            if candidates and not all_failed:
                screen = await self._screener.screen(source, context, candidates)
                usage = usage + screen.usage
                best = screen.best
                if best is not None and best >= self._policy.screen_floor and screen.shortlist:
                    by_id = {c.id: c for c in candidates}
                    shortlist = [by_id[rid] for rid in screen.shortlist]
                    chosen = await self._chooser.choose(source, context, shortlist)
                    usage = usage + chosen.usage
                    if chosen.record_id is not None:
                        gated = await self._gate.gate(source, context, by_id[chosen.record_id])
                        usage = usage + gated.usage
        except DecisionFatalError:
            raise
        except DecisionError as exc:
            provider_failed = True
            error = f"decider: {exc}"

        chosen_id = None if chosen is None else chosen.record_id
        chosen_key = None
        if screen is not None and chosen_id is not None:
            chosen_key = next((k for k, rid in screen.issued.items() if rid == chosen_id), None)

        signals = Signals(
            candidate_count=len(candidates),
            retrieval_failed=all_failed,
            screen_best=None if screen is None else screen.best,
            screen_chosen=(
                None if screen is None or chosen_key is None else screen.probabilities[chosen_key]
            ),
            resolution=None if chosen is None else chosen.resolution.value,
            p_choice=None if chosen is None else chosen.p_choice,
            p_none=None if chosen is None else chosen.p_none,
            choice_confidence=None if chosen is None else chosen.confidence,
            rubric=None if gated is None else gated.rubric_score,
            rubric_levels=None if gated is None else gated.rubric_levels,
            rubric_confidence=None if gated is None else gated.rubric_confidence,
            properties={} if gated is None else dict(gated.properties),
        )

        if provider_failed:
            status, reason = MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE
        else:
            status, reason = derive_status(signals, self._policy)

        if status in (MatchStatus.UNMATCHED, MatchStatus.FAILED):
            matched_id = None
        else:
            matched_id = chosen_id

        raw = json.dumps(
            {
                "served_model": None if screen is None else screen.model,
                "choose": None if chosen is None else json.loads(chosen.raw or "null"),
                "gate": None if gated is None else json.loads(gated.raw),
            },
            ensure_ascii=False,
        )
        all_notes = list(notes) + ([] if screen is None else list(screen.notes))
        joined = "; ".join(all_notes) if all_notes else None
        flat = signals.flat()
        if screen is not None:
            # Keep the shortlist's per-candidate probabilities so screen-stage recall can
            # be measured from the ledger. Only the shortlist, to keep the blob small.
            wanted = set(screen.shortlist)
            for key, rid in screen.issued.items():
                if rid in wanted:
                    flat[f"screen_{key}"] = screen.probabilities[key]
        explanation = render_explanation(signals)
        resolution = Resolution.ABSTAIN.value if chosen is None else chosen.resolution.value

        attempt = Attempt(
            index=0,
            query=queries[0] if queries else "",
            proposal=None,
            candidates=tuple(candidates) if self._keep_candidates else (),
            candidate_count=len(candidates),
            candidates_truncated=truncated,
            issued_keys={} if screen is None else dict(screen.issued),
            raw_selection=raw,
            chosen_id=chosen_id,
            resolution=resolution,
            primary_score=signals.screen_chosen,
            explanation=explanation,
            verifier_decision=None,
            verifier_score=None,
            verifier_preferred_id=None,
            audited=False,
            dropped_proposals=(),
            reason=reason,
            error=error if error else joined,
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
            finish_reason=None,
            signals=flat,
        )

        record = None
        if matched_id is not None:
            try:
                record = self._store.get(matched_id)
            except KeyError:
                record = None

        source_hash = hash_record(source)
        return MatchResult(
            result_key=result_key(self._run_fingerprint, source.id, source_hash),
            source_id=source.id,
            source_hash=source_hash,
            matched_id=matched_id,
            matched_record=record,
            confidence=signals.screen_chosen,
            status=status,
            reason=reason,
            explanation=explanation,
            candidates=tuple(candidates) if self._keep_candidates else (),
            attempts=(attempt,),
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
            run_fingerprint=self._run_fingerprint,
            signals=flat,
        )

    def match_sync(self, source: Record) -> MatchResult:
        return asyncio.run(self.match(source))


__all__ = ["DecisionMatcher"]
```

Also add `DecisionMatcher` and `DecisionPolicy` to `src/xwalk/__init__.py` (import and `__all__`), after the `Matcher` import so the version guard at the top of that file still runs first.

Then extend `src/xwalk/decide/__init__.py` with re-exports of `DecisionMatcher`, `DecisionPolicy`, `Signals`, `derive_status`, `FakeDecider`, `JevClient`, `CachingDecider`, `QuestionSet` and add them to `__all__`. Import order matters: `matcher` imports `stages`, which import `decide.base` and `decide.questions`; nothing in `stages` may import `xwalk.decide` (the package), only its submodules, or the import becomes circular.

- [ ] **Step 4: Run tests, lint, types**

Run: `.venv/bin/pytest tests/test_decide_matcher.py -q && .venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/decide/matcher.py src/xwalk/decide/__init__.py tests/test_decide_matcher.py
git commit -m "feat(decide): DecisionMatcher, one attempt per record, same MatchResult

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Configuration and the run fingerprint

**Files:**
- Modify: `src/xwalk/config.py` (`LLMSpec` optional, `DeciderSpec`, `DecisionPolicySpec`, `JobSpec`)
- Modify: `src/xwalk/batch.py` (`MatcherLike`, `build_decision_run_fingerprint`, `run_batch` signature)
- Test: `tests/test_config.py`, `tests/test_batch.py`

**Interfaces:**
- Produces:
  - `DeciderSpec(kind: Literal["jev"] = "jev", model: str, base_url: str = "https://openrouter.ai/api/alpha/decisions", api_key_env: str | None = None, api_key: str | None = None (rejected), timeout: float = 60.0, max_retries: int = 5)`.
  - `DecisionPolicySpec` mirroring every `DecisionPolicy` field with the same defaults.
  - `JobSpec.llm: LLMSpec | None = None`, `JobSpec.decider: DeciderSpec | None = None`, `JobSpec.decision_policy: DecisionPolicySpec = DecisionPolicySpec()`. A `mode="before"` validator moves a `policy:` block to `decision_policy` when `decider:` is present, so the YAML key is `policy:` on both paths. An `after` validator requires exactly one of `llm` / `decider`.
  - `JobSpec.build_decider() -> DecisionClient`, `build_questions() -> QuestionSet`, `build_decision_policy() -> DecisionPolicy`, `decision_run_fingerprint(*, store, retrievers, decider) -> str`, `build_decision_matcher(*, store, retrievers, decider) -> DecisionMatcher`.
  - `batch.MatcherLike` protocol (`run_fingerprint`, `policy` with `.concurrency`, `store_fingerprint`, `async match`); `run_batch` and `run_batch_sync` accept `MatcherLike`.
  - `batch.build_decision_run_fingerprint(*, templates, questions, store, retrievers, decider, policy: DecisionPolicy, rrf_k=60) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def _jev_job(tmp_path, extra_mutate=None):
    def mutate(data):
        del data["llm"]
        del data["selector"]
        data["decider"] = {
            "kind": "jev",
            "model": "~typesafe/jev-latest",
            "base_url": "https://openrouter.ai/api/alpha/decisions",
            "api_key_env": "XWALK_TEST_API_KEY",
        }
        data["policy"] = {"accept_at": 0.9, "chunk_size": 25, "concurrency": 4}
        if extra_mutate:
            extra_mutate(data)

    return _job_copy(tmp_path, mutate)


def test_a_decider_job_loads_and_routes_policy(tmp_path):
    job = load_job(_jev_job(tmp_path))
    assert job.llm is None and job.decider is not None
    assert job.decider.model == "~typesafe/jev-latest"
    policy = job.build_decision_policy()
    assert (policy.accept_at, policy.chunk_size, policy.concurrency) == (0.9, 25, 4)
    assert policy.screen_floor == 0.30  # untouched default


def test_llm_and_decider_are_mutually_exclusive(tmp_path):
    def keep_llm(data):
        data["llm"] = {
            "kind": "openai_compat", "model": "m", "base_url": "https://x", "api_key_env": "K"
        }

    with pytest.raises(ValueError, match="exactly one"):
        load_job(_jev_job(tmp_path, keep_llm))


def test_neither_llm_nor_decider_is_an_error(tmp_path):
    def drop(data):
        del data["decider"]

    with pytest.raises(ValueError, match="exactly one"):
        load_job(_jev_job(tmp_path, drop))


def test_decider_rejects_inline_keys(tmp_path):
    def inline(data):
        data["decider"]["api_key"] = "sk-live"

    with pytest.raises(ValueError, match="api_key_env"):
        load_job(_jev_job(tmp_path, inline))


def test_build_decider_needs_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("XWALK_TEST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XWALK_TEST_API_KEY"):
        load_job(_jev_job(tmp_path)).build_decider()


def test_build_decision_matcher(tmp_path, monkeypatch):
    from xwalk.decide.fake import FakeDecider
    from xwalk.decide.matcher import DecisionMatcher

    job = load_job(_jev_job(tmp_path))
    store = job.build_store()
    retrievers = job.build_retrievers(list(job.build_target_records()), job.build_templates(), tmp_path / "idx")
    matcher = job.build_decision_matcher(store=store, retrievers=retrievers, decider=FakeDecider())
    assert isinstance(matcher, DecisionMatcher)
    assert matcher.policy.concurrency == 4
    assert len(matcher.run_fingerprint) == 16


def test_decision_fingerprint_moves_with_policy_and_model(tmp_path):
    from xwalk.decide.fake import FakeDecider

    job = load_job(_jev_job(tmp_path))
    store = job.build_store()
    retrievers = job.build_retrievers(list(job.build_target_records()), job.build_templates(), tmp_path / "idx")
    a = job.decision_run_fingerprint(store=store, retrievers=retrievers, decider=FakeDecider(model="a"))
    b = job.decision_run_fingerprint(store=store, retrievers=retrievers, decider=FakeDecider(model="b"))
    c = load_job(_jev_job(tmp_path, lambda d: d["policy"].update({"accept_at": 0.7}))).decision_run_fingerprint(
        store=store, retrievers=retrievers, decider=FakeDecider(model="a")
    )
    assert a != b and a != c
```

Append to `tests/test_batch.py`:

```python
async def test_run_batch_accepts_a_decision_matcher(tmp_path):
    from xwalk.batch import run_batch
    from xwalk.decide.fake import FakeDecider
    from xwalk.decide.matcher import DecisionMatcher
    from xwalk.decide.policy import DecisionPolicy
    from xwalk.decide.questions import QuestionSet
    from xwalk.prompts.contract import PromptSlots
    from xwalk.records import Record, RetrievalHit
    from xwalk.retrieval.base import SearchRequest
    from xwalk.stages.choose import Chooser
    from xwalk.stages.property_gate import PropertyGate
    from xwalk.stages.screen import Screener
    from xwalk.stores.memory import MemoryStore
    from xwalk.templates import TemplateSet

    templates = TemplateSet(query="{{ mention }}", context="", doc="{{ label }}", candidate="{{ label }}")
    questions = QuestionSet.from_slots(
        PromptSlots(
            entity_noun="m", target_noun="t", domain_brief="d",
            rubric=[{"score": 1.0, "name": "A", "when": "a"}, {"score": 0.5, "name": "B", "when": "b"}],
        )
    )
    store = MemoryStore.from_source([Record(id="T1", fields={"label": "glucose"})])

    class One:
        name = "r"
        fingerprint = "r"
        default_limit = 5

        async def search(self, request: SearchRequest):
            return [RetrievalHit(record_id="T1", retriever="r", raw_score=1.0, rank=1)]

    decider = FakeDecider()
    matcher = DecisionMatcher(
        templates=templates, retrievers=[One()], store=store,
        screener=Screener(decider, questions, templates),
        chooser=Chooser(decider, questions, templates),
        gate=PropertyGate(decider, questions, templates),
        policy=DecisionPolicy(concurrency=2, accept_at=0.5, shortlist_floor=0.0, screen_floor=0.1, rubric_floor=0.0),
        run_fingerprint="fp-decide",
    )
    report = await run_batch(matcher, [Record(id="s1", fields={"mention": "glucose"})], out=tmp_path / "run")
    assert report.total == 1
    assert (tmp_path / "run" / "mapping.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py tests/test_batch.py -q`
Expected: FAIL (unknown field `decider`, `llm` required, missing builders).

- [ ] **Step 3: Extend `config.py`**

Add imports:

```python
from xwalk.decide.base import DecisionClient
from xwalk.decide.jev import JevClient
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.stages.choose import Chooser
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.screen import Screener
from xwalk.batch import build_decision_run_fingerprint, build_run_fingerprint
```

Add the two specs after `LLMSpec`:

```python
class DeciderSpec(BaseModel):
    kind: Literal["jev"] = "jev"
    model: str
    base_url: str = "https://openrouter.ai/api/alpha/decisions"
    api_key_env: str | None = None
    api_key: str | None = None  # present only so it can be rejected
    timeout: float = 60.0
    max_retries: int = 5

    @model_validator(mode="after")
    def _no_inline_secrets(self) -> DeciderSpec:
        if self.api_key is not None:
            raise ValueError(
                "api_key must not appear in a job file; use api_key_env with the name of "
                "an environment variable"
            )
        return self


class DecisionPolicySpec(BaseModel):
    screen_floor: float = 0.30
    shortlist_size: int = 15
    shortlist_floor: float = 0.20
    none_at: float = 0.70
    choose_at: float = 0.50
    accept_at: float = 0.85
    rubric_floor: float | None = None
    property_floor: float = 0.50
    chunk_size: int = 50
    max_candidates: int = 300
    concurrency: int = 32
    retriever_timeout: float = 60.0
```

Change `JobSpec` fields and add validators and builders:

```python
class JobSpec(BaseModel):
    name: str
    templates: TemplateSpec
    target: RecordSpec
    source: RecordSpec
    retrievers: list[RetrieverSpec] = Field(min_length=1)
    llm: LLMSpec | None = None
    decider: DeciderSpec | None = None
    prompts: PromptSpec
    policy: PolicySpec = PolicySpec()
    decision_policy: DecisionPolicySpec = DecisionPolicySpec()
    selector: SelectorSpec = SelectorSpec()

    base_dir: Path = Path(".")

    @model_validator(mode="before")
    @classmethod
    def _route_policy(cls, data: Any) -> Any:
        # One YAML key, `policy:`, on both paths. Which model parses it depends on the path.
        if isinstance(data, dict) and data.get("decider") is not None and "policy" in data:
            data = dict(data)
            data["decision_policy"] = data.pop("policy")
        return data

    @model_validator(mode="after")
    def _exactly_one_decision_maker(self) -> JobSpec:
        if (self.llm is None) == (self.decider is None):
            raise ValueError("a job needs exactly one of `llm:` or `decider:`")
        return self
```

`build_llm` must now guard `self.llm is None` with `raise ValueError("this job has no llm: block")`. Then add:

```python
    def build_decider(self) -> DecisionClient:
        if self.decider is None:
            raise ValueError("this job has no decider: block")
        api_key = None
        if self.decider.api_key_env:
            api_key = os.environ.get(self.decider.api_key_env)
            if not api_key:
                raise ValueError(
                    f"environment variable {self.decider.api_key_env} is not set; "
                    f"export it or change decider.api_key_env in the job file"
                )
        return JevClient(
            self.decider.base_url,
            self.decider.model,
            api_key=api_key,
            timeout=self.decider.timeout,
            max_retries=self.decider.max_retries,
        )

    def build_questions(self) -> QuestionSet:
        return QuestionSet.from_slots(load_slots(self.base_dir / self.prompts.slots))

    def build_decision_policy(self) -> DecisionPolicy:
        return DecisionPolicy(**self.decision_policy.model_dump())

    def decision_run_fingerprint(
        self, *, store: TargetStore, retrievers: Sequence[Retriever], decider: DecisionClient
    ) -> str:
        return build_decision_run_fingerprint(
            templates=self.build_templates(),
            questions=self.build_questions(),
            store=store,
            retrievers=retrievers,
            decider=decider,
            policy=self.build_decision_policy(),
        )

    def build_decision_matcher(
        self, *, store: TargetStore, retrievers: Sequence[Retriever], decider: DecisionClient
    ) -> DecisionMatcher:
        templates = self.build_templates()
        questions = self.build_questions()
        policy = self.build_decision_policy()
        return DecisionMatcher(
            templates=templates,
            retrievers=list(retrievers),
            store=store,
            screener=Screener(
                decider,
                questions,
                templates,
                chunk_size=policy.chunk_size,
                shortlist_size=policy.shortlist_size,
                shortlist_floor=policy.shortlist_floor,
            ),
            chooser=Chooser(decider, questions, templates),
            gate=PropertyGate(decider, questions, templates),
            policy=policy,
            run_fingerprint=self.decision_run_fingerprint(
                store=store, retrievers=retrievers, decider=decider
            ),
        )
```

- [ ] **Step 4: Extend `batch.py`**

Add after the imports:

```python
class MatcherLike(Protocol):
    @property
    def run_fingerprint(self) -> str: ...

    @property
    def policy(self) -> Any: ...  # anything with a `.concurrency: int`

    @property
    def store_fingerprint(self) -> str: ...

    async def match(self, source: Record) -> MatchResult: ...
```

(`Protocol` from `typing`.) Change the `matcher: Matcher` annotation on `run_batch` and `run_batch_sync` to `matcher: MatcherLike`. Add:

```python
def build_decision_run_fingerprint(
    *,
    templates: TemplateSet,
    questions: QuestionSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    decider: DecisionClient,
    policy: DecisionPolicy,
    rrf_k: int = 60,
) -> str:
    """The decider path's counterpart of `build_run_fingerprint`.

    Concurrency, timeouts, and credentials are excluded for the same reason as on the
    LLM path: they change how fast an answer arrives, never what it means.
    """
    return hash_value(
        {
            "library_version": __version__,
            "path": "decider",
            "templates": templates.fingerprint,
            "questions": questions.fingerprint,
            "target": store.fingerprint,
            "retrievers": sorted(f"{r.name}:{r.fingerprint}" for r in retrievers),
            "retrieval": {
                "depths": sorted(f"{r.name}:{getattr(r, 'default_limit', 20)}" for r in retrievers),
                "rrf_k": rrf_k,
                "max_candidates": policy.max_candidates,
            },
            "decider": decider.fingerprint,
            "policy": {
                "screen_floor": policy.screen_floor,
                "shortlist_size": policy.shortlist_size,
                "shortlist_floor": policy.shortlist_floor,
                "none_at": policy.none_at,
                "choose_at": policy.choose_at,
                "accept_at": policy.accept_at,
                "rubric_floor": policy.rubric_floor,
                "property_floor": policy.property_floor,
                "chunk_size": policy.chunk_size,
            },
        }
    )
```

with imports `from xwalk.decide.base import DecisionClient`, `from xwalk.decide.policy import DecisionPolicy`, `from xwalk.decide.questions import QuestionSet`. `batch.py` importing `xwalk.decide.*` while `config.py` imports `batch` is fine as long as `xwalk.decide.matcher` does not import `batch` (it does not).

- [ ] **Step 5: Run the full suite, lint, types**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass. `tests/test_examples.py` loads every example job; they all still have `llm:` so they still validate.

- [ ] **Step 6: Commit**

```bash
git add src/xwalk/config.py src/xwalk/batch.py tests/test_config.py tests/test_batch.py
git commit -m "feat(decide): decider job block, decision policy, and a decider run fingerprint

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: CLI dispatch, cache by default, and the offline end-to-end test

**Files:**
- Modify: `src/xwalk/cli/main.py` (`_cmd_match`)
- Create: `tests/fixtures/job_tiny_jev.yaml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `JobSpec.build_decider`, `build_decision_matcher` (Task 13), `CachingDecider` (Task 5), `Ledger`.
- Produces: `xwalk match --job <decider job>` runs the decider path, wrapping the client in `CachingDecider` on the run's ledger, and writes `manifest_extra={"job": name, "model": decider.model, "path": "decider"}`. LLM jobs are unchanged and get `"path": "llm"`.
- A private hook for tests: `_cmd_match` accepts the decider from `_build_decider(job)` module-level function that tests monkeypatch to return a `FakeDecider`.

- [ ] **Step 1: Write the fixture and the failing test**

`tests/fixtures/job_tiny_jev.yaml`:

```yaml
# The decider-path twin of job_tiny.yaml. Same data, same slots, Jev instead of an LLM.
name: tiny-jev
templates:
  queries:
    - "{{ mention }}"
    - "{{ context_left }} {{ mention }}"
  context: "{{ context_left }} [{{ mention }}] {{ context_right }}"
  doc: "{{ label }} {{ synonyms | join(' ') }}"
  candidate: "ID: {{ id }} Label: {{ label }} {% if synonyms %}Synonyms: {{ synonyms | join('; ') }}{% endif %}"

target:
  kind: csv
  path: targets_tiny.csv
  id_column: id
  multivalue_columns: [synonyms]

source:
  kind: csv
  path: sources_tiny.csv
  id_column: mention_id

retrievers:
  - kind: bm25
    name: bm25
    limit: 20
    exact_fields: [label, synonyms]

decider:
  kind: jev
  model: "~typesafe/jev-latest"
  base_url: https://openrouter.ai/api/alpha/decisions
  api_key_env: XWALK_TEST_API_KEY

prompts:
  slots: ../../examples/chemistry/slots.yaml

policy:
  accept_at: 0.5
  screen_floor: 0.1
  shortlist_floor: 0.0
  rubric_floor: 0.0
  property_floor: 0.0
  chunk_size: 2
  concurrency: 4
```

Append to `tests/test_cli.py`:

```python
def test_match_runs_the_decider_path_offline(tmp_path, monkeypatch):
    import json

    from xwalk.cli import main as cli
    from xwalk.decide.fake import FakeDecider

    monkeypatch.setattr(cli, "_build_decider", lambda job: FakeDecider())
    out = tmp_path / "run"
    code = cli.main(["match", "--job", str(FIXTURES / "job_tiny_jev.yaml"), "--out", str(out)])
    assert code in (0, 2)  # 2 = something needs review, still a completed run
    rows = (out / "mapping.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5  # header + 4 sources
    first = json.loads((out / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "signals" in first and first["attempts"][0]["signals"] is not None
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["path"] == "decider" and manifest["model"] == "fake-decider"


def test_a_resumed_decider_run_replays_from_the_cache(tmp_path, monkeypatch):
    from xwalk.cli import main as cli
    from xwalk.decide.fake import FakeDecider

    fakes: list[FakeDecider] = []

    def build(job):
        fakes.append(FakeDecider())
        return fakes[-1]

    monkeypatch.setattr(cli, "_build_decider", build)
    out = tmp_path / "run"
    args = ["match", "--job", str(FIXTURES / "job_tiny_jev.yaml"), "--out", str(out)]
    cli.main(args)
    cli.main(args + ["--no-resume"])
    assert len(fakes[0].calls) > 0
    assert fakes[1].calls == []  # every decision came back from the ledger cache
```

(`FIXTURES` is already imported in `tests/test_cli.py` from `tests.conftest`; if not, add `from tests.conftest import FIXTURES`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cli.py -q -k decider`
Expected: FAIL with `AttributeError: module 'xwalk.cli.main' has no attribute '_build_decider'`.

- [ ] **Step 3: Rewrite `_cmd_match`**

```python
def _build_decider(job: JobSpec) -> DecisionClient:
    """Module-level so a test can swap in a FakeDecider without an API key."""
    return job.build_decider()


def _cmd_match(args: argparse.Namespace) -> int:
    from xwalk.batch import run_batch
    from xwalk.config import load_job
    from xwalk.decide.cache import CachingDecider
    from xwalk.ledger import Ledger

    job = load_job(args.job)
    index_dir = Path(args.index or Path(args.out) / "index")
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, templates, index_dir)

    records = list(job.build_source_records())
    if args.limit is not None:
        records = records[: args.limit]

    cache_ledger: Ledger | None = None
    try:
        if job.decider is not None:
            # The cache lives in the run's own ledger. run_batch opens the same file
            # again; SQLite in WAL mode allows both connections.
            cache_ledger = Ledger.open(Path(args.out) / "ledger.sqlite")
            decider = CachingDecider(_build_decider(job), cache_ledger)
            matcher: MatcherLike = job.build_decision_matcher(
                store=store, retrievers=retrievers, decider=decider
            )
            manifest_extra = {"job": job.name, "model": decider.model, "path": "decider"}
        else:
            llm = job.build_llm()
            matcher = job.build_matcher(store=store, retrievers=retrievers, llm=llm)
            manifest_extra = {"job": job.name, "model": llm.model, "path": "llm"}

        report = asyncio.run(
            run_batch(
                matcher, records, out=args.out, resume=args.resume, manifest_extra=manifest_extra
            )
        )
    finally:
        if cache_ledger is not None:
            cache_ledger.close()

    print(f"matched {report.total} records into {report.out_dir}")
    for status, count in sorted(report.by_status().items(), key=lambda kv: kv[0].value):
        print(f"  {status.value:<14}: {count}")
    duplicates = report.duplicate_targets()
    if duplicates:
        print(f"  duplicate targets: {len(duplicates)} (see manifest.json)")
    review_count = len(report.needs_review())
    if review_count:
        print(f"\n{review_count} rows need review: xwalk review export --run {args.out}")
        return EXIT_ATTENTION
    return EXIT_OK
```

Add the imports at module top under `TYPE_CHECKING` if the CLI keeps its lazy-import style: `from xwalk.batch import MatcherLike`, `from xwalk.config import JobSpec`, `from xwalk.decide.base import DecisionClient`. Check the top of `src/xwalk/cli/main.py` for how it already imports `Path` and `asyncio`, and follow that.

Also in `_cmd_ablate` and `_cmd_prompts` (optimize), add at the top:

```python
    if job.decider is not None:
        print("error: this command works on the LLM path; the job has a decider: block", file=sys.stderr)
        return EXIT_USAGE
```

- [ ] **Step 4: Run the full suite, lint, types**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/xwalk/cli/main.py tests/fixtures/job_tiny_jev.yaml tests/test_cli.py
git commit -m "feat(cli): match runs the decider path with ledger-cached decisions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Threshold fitting against gold

**Files:**
- Create: `src/xwalk/decide/fit.py`
- Modify: `src/xwalk/cli/main.py` (new `fit` subcommand)
- Test: `tests/test_decide_fit.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `GoldSet` (`gold.is_correct(source_id, predicted) -> bool | None`, `source_id in gold`), `MatchResult.signals`, `Signals.from_flat`, `derive_status`, `DecisionPolicy`.
- Produces:
  - `FitPoint(accept_at: float, property_floor: float, accepted: int, correct: int, precision: float | None, coverage: float, near_threshold: int)`.
  - `FitReport(points: list[FitPoint], recommended: FitPoint | None, target_precision: float, labelled: int)`.
  - `fit_thresholds(results: Iterable[MatchResult], gold: GoldSet, *, base: DecisionPolicy, target_precision: float = 0.95, accept_grid: Sequence[float] = (0.5, 0.55, …, 0.95), property_grid: Sequence[float] = (0.0, 0.3, 0.5, 0.7), margin: float = 0.05) -> FitReport`. For each grid point it re-derives every labelled result's status with `dataclasses.replace(base, accept_at=…, property_floor=…)` and counts `matched` rows that are correct. `near_threshold` counts labelled rows whose `screen_chosen` lies within `margin` of `accept_at`, which is the jitter warning the spec asks for. `recommended` is the point with the highest `accepted` among those meeting the target precision; ties go to the higher `accept_at`.
  - `render_fit(report) -> str`.
  - CLI: `xwalk fit --run DIR --gold FILE [--precision 0.95]` prints `render_fit` and returns `EXIT_OK`, or `EXIT_ATTENTION` when no grid point meets the target.

- [ ] **Step 1: Write the failing tests**

`tests/test_decide_fit.py`:

```python
from xwalk.decide.fit import fit_thresholds, render_fit
from xwalk.decide.policy import DecisionPolicy
from xwalk.evaluate.gold import GoldSet
from xwalk.records import Attempt, DecisionReason, MatchResult, MatchStatus, Usage


def _result(sid, matched, screen, prop=0.9):
    signals = {
        "screen_best": screen, "screen_chosen": screen, "p_choice": 0.8, "p_none": 0.05,
        "choice_confidence": 0.7, "rubric": 2.5, "rubric_levels": 3.0, "rubric_confidence": 0.8,
        "prop_form": prop,
    }
    attempt = Attempt(
        index=0, query="q", proposal=None, candidates=(), candidate_count=5,
        candidates_truncated=0, issued_keys={}, raw_selection=None, chosen_id=matched,
        resolution="exact_key", primary_score=screen, explanation="", verifier_decision=None,
        verifier_score=None, verifier_preferred_id=None, audited=False, dropped_proposals=(),
        reason=None, error=None, usage=Usage.zero(), elapsed_seconds=0.0, finish_reason=None,
        signals=signals,
    )
    return MatchResult(
        result_key=sid, source_id=sid, source_hash="h", matched_id=matched, matched_record=None,
        confidence=screen, status=MatchStatus.NEEDS_REVIEW, reason=DecisionReason.BELOW_ACCEPT_THRESHOLD,
        explanation="", candidates=(), attempts=(attempt,), usage=Usage.zero(),
        elapsed_seconds=0.0, run_fingerprint="fp", signals=signals,
    )


GOLD = GoldSet({"s1": frozenset({"T1"}), "s2": frozenset({"T2"}), "s3": frozenset({"T3"}), "s4": frozenset({"T4"})})
RESULTS = [
    _result("s1", "T1", 0.95),           # right, confident
    _result("s2", "T2", 0.90),           # right
    _result("s3", "T9", 0.70),           # wrong, mid
    _result("s4", "T4", 0.60, prop=0.2), # right, but a property disagrees
]


def test_recommends_the_widest_threshold_that_meets_precision():
    report = fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=1.0)
    assert report.recommended is not None
    assert report.recommended.accept_at > 0.70
    assert report.recommended.correct == report.recommended.accepted


def test_a_lower_target_admits_the_wrong_one():
    report = fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=0.6)
    assert report.recommended is not None
    assert report.recommended.accept_at <= 0.70
    assert report.recommended.accepted >= 3


def test_near_threshold_counts_rows_inside_the_margin():
    report = fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), accept_grid=(0.9,), property_grid=(0.0,), margin=0.06)
    (point,) = report.points
    assert point.near_threshold == 2  # 0.95 and 0.90


def test_render_mentions_the_recommendation_or_its_absence():
    assert "recommended" in render_fit(fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=1.0))
    assert "no grid point" in render_fit(fit_thresholds(RESULTS, GOLD, base=DecisionPolicy(), target_precision=1.01))
```

If `GoldSet`'s constructor differs from `GoldSet(mapping)`, build it with `load_gold_csv` on a temporary file instead; check `src/xwalk/evaluate/gold.py` lines 1 to 60 first.

Append to `tests/test_cli.py`:

```python
def test_fit_command_runs_on_a_decider_run(tmp_path, monkeypatch):
    from xwalk.cli import main as cli
    from xwalk.decide.fake import FakeDecider

    monkeypatch.setattr(cli, "_build_decider", lambda job: FakeDecider())
    out = tmp_path / "run"
    cli.main(["match", "--job", str(FIXTURES / "job_tiny_jev.yaml"), "--out", str(out)])
    code = cli.main(["fit", "--run", str(out), "--gold", str(FIXTURES / "gold_tiny.csv"), "--precision", "0.5"])
    assert code in (0, 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_decide_fit.py tests/test_cli.py -q -k "fit"`
Expected: FAIL with `ModuleNotFoundError` and `argparse` rejecting `fit`.

- [ ] **Step 3: Write `src/xwalk/decide/fit.py`**

```python
"""Fit the decider policy's thresholds on a finished run and its gold labels.

Nothing here calls a model. It re-derives statuses from the signals already in the
ledger, so fitting is free and repeatable. The probe showed Jev's probabilities move by
a few hundredths between identical calls, so every point also reports how many labelled
rows sit within `margin` of the threshold: a policy that only looks good because of
where the jitter landed is visible as a large `near_threshold`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from xwalk.decide.policy import DecisionPolicy, Signals, derive_status
from xwalk.evaluate.gold import GoldSet
from xwalk.records import MatchResult, MatchStatus

DEFAULT_ACCEPT_GRID: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
DEFAULT_PROPERTY_GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)


@dataclass(frozen=True)
class FitPoint:
    accept_at: float
    property_floor: float
    accepted: int
    correct: int
    precision: float | None
    coverage: float
    near_threshold: int


@dataclass(frozen=True)
class FitReport:
    points: list[FitPoint]
    recommended: FitPoint | None
    target_precision: float
    labelled: int


def _signals_of(result: MatchResult) -> Signals | None:
    if not result.attempts:
        return None
    attempt = result.attempts[0]
    return Signals.from_flat(
        attempt.signals,
        candidate_count=attempt.candidate_count,
        retrieval_failed=result.status is MatchStatus.FAILED,
        resolution=attempt.resolution,
    )


def fit_thresholds(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    base: DecisionPolicy,
    target_precision: float = 0.95,
    accept_grid: Sequence[float] = DEFAULT_ACCEPT_GRID,
    property_grid: Sequence[float] = DEFAULT_PROPERTY_GRID,
    margin: float = 0.05,
) -> FitReport:
    labelled = [(r, s) for r in results if r.source_id in gold and (s := _signals_of(r)) is not None]
    points: list[FitPoint] = []
    for accept_at in accept_grid:
        for property_floor in property_grid:
            policy = replace(base, accept_at=accept_at, property_floor=property_floor)
            accepted = correct = near = 0
            for result, signals in labelled:
                if signals.screen_chosen is not None and abs(signals.screen_chosen - accept_at) <= margin:
                    near += 1
                status, _ = derive_status(signals, policy)
                if status is MatchStatus.MATCHED:
                    accepted += 1
                    if gold.is_correct(result.source_id, result.matched_id):
                        correct += 1
            points.append(
                FitPoint(
                    accept_at=accept_at,
                    property_floor=property_floor,
                    accepted=accepted,
                    correct=correct,
                    precision=(correct / accepted) if accepted else None,
                    coverage=(accepted / len(labelled)) if labelled else 0.0,
                    near_threshold=near,
                )
            )
    eligible = [p for p in points if p.precision is not None and p.precision >= target_precision]
    recommended = max(eligible, key=lambda p: (p.accepted, p.accept_at, p.property_floor), default=None)
    return FitReport(points=points, recommended=recommended, target_precision=target_precision, labelled=len(labelled))


def render_fit(report: FitReport) -> str:
    lines = [f"labelled rows: {report.labelled}; target precision: {report.target_precision:.2f}", ""]
    lines.append(f"{'accept_at':>9} {'prop_floor':>10} {'accepted':>8} {'correct':>7} {'precision':>9} {'coverage':>8} {'near':>4}")
    for p in report.points:
        precision = "  -" if p.precision is None else f"{p.precision:9.2f}"
        lines.append(
            f"{p.accept_at:9.2f} {p.property_floor:10.2f} {p.accepted:8d} {p.correct:7d} {precision:>9} {p.coverage:8.2f} {p.near_threshold:4d}"
        )
    lines.append("")
    if report.recommended is None:
        lines.append("no grid point meets the target precision; lower the target or improve the questions")
    else:
        r = report.recommended
        lines.append(
            f"recommended: accept_at={r.accept_at:.2f} property_floor={r.property_floor:.2f} "
            f"({r.correct}/{r.accepted} correct, coverage {r.coverage:.2f}, "
            f"{r.near_threshold} rows within the jitter margin of accept_at)"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Add the `fit` subcommand**

In `_build_parser`:

```python
    fit = sub.add_parser("fit", help="fit decider thresholds on a completed run and gold labels")
    fit.add_argument("--run", required=True)
    fit.add_argument("--gold", required=True)
    fit.add_argument("--precision", type=float, default=0.95)
```

Handler, registered in `_DISPATCH` as `"fit": _cmd_fit`:

```python
def _cmd_fit(args: argparse.Namespace) -> int:
    from xwalk.decide.fit import fit_thresholds, render_fit
    from xwalk.decide.policy import DecisionPolicy
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.ledger import Ledger

    run_dir = Path(args.run)
    gold = load_gold_csv(args.gold)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        results = list(ledger.iter_results(_run_fingerprint_of(run_dir)))
    finally:
        ledger.close()
    report = fit_thresholds(results, gold, base=DecisionPolicy(), target_precision=args.precision)
    print(render_fit(report))
    return EXIT_OK if report.recommended is not None else EXIT_ATTENTION
```

- [ ] **Step 5: Run the full suite, lint, types**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/xwalk/decide/fit.py src/xwalk/cli/main.py tests/test_decide_fit.py tests/test_cli.py
git commit -m "feat(decide): fit accept_at and property_floor on gold, with a jitter margin

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: Live integration test

**Files:**
- Create: `tests/test_decide_integration.py`

**Interfaces:**
- Consumes: `JevClient`, `Noul`, `Choice`, `Score`. Skips unless `XWALK_TEST_API_KEY` is set. Uses `XWALK_TEST_DECIDER_URL` if set, else the OpenRouter alpha URL, and `XWALK_TEST_DECIDER_MODEL` if set, else `~typesafe/jev-latest`.

- [ ] **Step 1: Write the test**

```python
"""Hits the real decision endpoint. Skipped without a key; costs well under a cent."""

import os

import pytest

from xwalk.decide.base import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from xwalk.decide.jev import JevClient

pytestmark = pytest.mark.integration

KEY = os.environ.get("XWALK_TEST_API_KEY")
URL = os.environ.get("XWALK_TEST_DECIDER_URL", "https://openrouter.ai/api/alpha/decisions")
MODEL = os.environ.get("XWALK_TEST_DECIDER_MODEL", "~typesafe/jev-latest")


@pytest.mark.skipif(not KEY, reason="XWALK_TEST_API_KEY unset")
async def test_one_call_with_all_three_question_types():
    client = JevClient(URL, MODEL, api_key=KEY, max_retries=2)
    try:
        response = await client.decide(
            {
                "source": {"fields": {"english_name": "pineapple, canned in syrup"}},
                "candidates": {"C1": "pineapple (whole, raw)", "C2": "pineapple (canned, in syrup)"},
            },
            {
                "n_C1": Noul(instructions="Does `candidates.C1` denote the same food as `source`?"),
                "n_C2": Noul(instructions="Does `candidates.C2` denote the same food as `source`?"),
                "best": Choice(
                    instructions="Which candidate is the same food as `source`? NONE if none.",
                    criteria={"C1": "whole raw", "C2": "canned in syrup", "NONE": "none"},
                ),
                "grade": Score(instructions="How well does `candidates.C2` match `source`?", criteria=["poor", "fair", "exact"]),
            },
        )
    finally:
        await client.aclose()
    n1, n2, best, grade = (response.answers[k] for k in ("n_C1", "n_C2", "best", "grade"))
    assert isinstance(n1, NoulAnswer) and isinstance(n2, NoulAnswer)
    assert isinstance(best, ChoiceAnswer) and isinstance(grade, ScoreAnswer)
    assert n2.noul > n1.noul
    assert best.choice == "C2"
    assert abs(sum(best.probabilities.values()) - 1.0) < 0.02
    assert response.model.startswith("typesafe/jev")
    assert response.usage.prompt_tokens > 0
```

- [ ] **Step 2: Run it both ways**

Run: `.venv/bin/pytest tests/test_decide_integration.py -q` (skips) and `set -a; source .env; set +a; .venv/bin/pytest tests/test_decide_integration.py -q -m integration` (passes against the live endpoint).
Expected: skip, then pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_decide_integration.py
git commit -m "test(decide): gated live call against the decisions endpoint

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: Example jobs and documentation

**Files:**
- Create: `examples/cafeteria_fcd/job_jev.yaml`, `examples/ref_zivila/jobs/foodon/job_jev.yaml`, `docs/reference/decide.md`
- Modify: `examples/cafeteria_fcd/slots.yaml`, `examples/ref_zivila/jobs/foodon/slots.yaml` (add `properties`), `docs/README.md`, `docs/components.md`, `docs/reference/job-file.md`, `docs/reference/cli.md`, `docs/concepts.md`
- Test: `tests/test_examples.py` (loads every job), `tests/test_docs.py` (links resolve, Python fences parse, no orphan pages)

- [ ] **Step 1: Add `properties` to both food slots files**

Append to `examples/cafeteria_fcd/slots.yaml` and `examples/ref_zivila/jobs/foodon/slots.yaml`:

```yaml
properties:
  - name: processing_state
    question: >-
      Do the source and the candidate agree on processing state (raw, cooked, dried,
      canned, frozen, smoked, fermented, juice, oil, flour, jam) wherever either states one?
  - name: species_or_ingredient
    question: >-
      Do the source and the candidate agree on the source species or main ingredient
      wherever either states one?
  - name: part_or_form
    question: >-
      Do the source and the candidate agree on the anatomical part or product form
      (whole, pieces, fillet, leg, powder, paste) wherever either states one?
```

- [ ] **Step 2: Write the two example jobs**

`examples/cafeteria_fcd/job_jev.yaml`:

```yaml
# The decider-path twin of job.yaml: same data, same slots, Jev instead of an LLM.
# Run:  set -a; source .env; set +a
#       xwalk index --job examples/cafeteria_fcd/job_jev.yaml --out runs/cfcd_jev/index
#       xwalk match --job examples/cafeteria_fcd/job_jev.yaml --out runs/cfcd_jev
#       xwalk eval  --run runs/cfcd_jev --gold examples/cafeteria_fcd/sample/gold.csv
name: cafeteria_fcd_jev
templates:
  queries:
    - "{{ mention }}"
  context: "{{ context_left }}[{{ mention }}]{{ context_right }}"
  doc: "{{ label }} {{ synonyms | join(' ') }} {{ definition }}"
  candidate: |
    ID: {{ id }}
    Label: {{ label }}
    {% if synonyms %}Synonyms: {{ synonyms | join('; ') }}{% endif %}
    {% if definition %}Definition: {{ definition }}{% endif %}

target:
  kind: owl
  path: sample/targets.owl

source:
  kind: jsonl
  path: sample/mentions.jsonl
  id_field: mention_id

retrievers:
  - kind: bm25
    name: bm25
    limit: 200
    exact_fields: [label, synonyms]

decider:
  kind: jev
  model: "~typesafe/jev-latest"      # pin typesafe/jev-1.13 for a reproducible run
  base_url: https://openrouter.ai/api/alpha/decisions
  api_key_env: XWALK_TEST_API_KEY

prompts:
  slots: slots.yaml

policy:
  accept_at: 0.85
  screen_floor: 0.30
  chunk_size: 50
  max_candidates: 300
  concurrency: 16
```

`examples/ref_zivila/jobs/foodon/job_jev.yaml`:

```yaml
# Full 2,030-row Ref_zivila FoodOn mapping with the Jev decider.
name: ref_zivila_foodon_jev

templates:
  queries:
    - "{{ mention_en }}"
    - "{{ name_slo }}"
    - "{{ tag_name_slo }}"
    - "{{ aliases | join(' ') }}"
  context: |-
    {% if mention_en %}English name: {{ mention_en }}{% endif %}
    {% if name_slo %}Slovenian full name: {{ name_slo }}{% endif %}
    {% if tag_name_slo %}Slovenian tag name: {{ tag_name_slo }}{% endif %}
    {% if short_name_slo %}Slovenian short name: {{ short_name_slo }}{% endif %}
    {% if fgnm %}Food group: {{ fgnm }}{% endif %}
    {% if curation_note %}Curation warning: {{ curation_note }}{% endif %}
  doc: >-
    {{ label }} {{ synonyms | join(' ') }} {{ definition }}
    {{ parent_labels | join(' ') }}
  candidate: |
    ID: {{ id }}
    Label: {{ label }}
    {% if synonyms %}Synonyms: {{ synonyms | join('; ') }}{% endif %}
    {% if definition %}Definition: {{ definition }}{% endif %}
    {% if parent_labels %}Parents: {{ parent_labels | join('; ') }}{% endif %}

target:
  kind: csv
  path: ../../../../data/ref_zivila/targets/foodon.csv
  id_column: id
  multivalue_columns: [synonyms, parents, parent_labels]

source:
  kind: csv
  path: ../../../../data/ref_zivila/source.csv
  id_column: ID
  multivalue_columns: [aliases]

retrievers:
  - kind: bm25
    name: bm25
    limit: 150
    exact_fields: [label, synonyms]
  # Uncomment once xwalk[dense] is installed; this is what recovers Slovenian-only rows.
  # - kind: dense
  #   name: e5
  #   model: intfloat/multilingual-e5-small
  #   limit: 150
  #   query_prefix: "query: "
  #   doc_prefix: "passage: "

decider:
  kind: jev
  model: "~typesafe/jev-latest"
  base_url: https://openrouter.ai/api/alpha/decisions
  api_key_env: XWALK_TEST_API_KEY

prompts:
  slots: slots.yaml

policy:
  accept_at: 0.85
  screen_floor: 0.30
  shortlist_size: 15
  chunk_size: 50
  max_candidates: 300
  concurrency: 16
```

- [ ] **Step 3: Run the example loader test**

Run: `.venv/bin/pytest tests/test_examples.py -q`
Expected: pass. If the test enumerates `job*.yaml` by glob it picks the new files up automatically; if it lists jobs by name, add both.

- [ ] **Step 4: Write `docs/reference/decide.md`**

Sections, each with the signatures copied from the source of Tasks 2 to 15:

1. **What a decision model is** (four sentences; link to the spec is not allowed since `superpowers/` is excluded from the docs tree, so restate: typed questions in, probabilities out, no text).
2. **The loop**: the ASCII diagram from the spec's Architecture section.
3. **`DecisionClient` protocol** and the three question and answer types.
4. **`JevClient`**: constructor signature, both URLs, retry behaviour, what the fingerprint covers.
5. **`FakeDecider`** and `overlap_handler`.
6. **`CachingDecider`** and why it is on by default (the jitter finding).
7. **`QuestionSet`**: how each of the four question kinds is composed from `slots.yaml`; the `properties:` key with the YAML example from Step 1.
8. **Stages**: `Screener`, `Chooser`, `PropertyGate` signatures and outcomes.
9. **`DecisionPolicy`** table (copy the spec's table) and the seven `derive_status` rules.
10. **`DecisionMatcher`** and what lands in `Attempt` and `MatchResult` (`signals`, `raw_selection`, `confidence`).
11. **Fitting thresholds**: `xwalk fit` usage and how to read `near_threshold`.
12. **What the decider path does not do**: no retries, no rewriter, no verifier, no generated explanation, `ablate` and `prompts optimize` unavailable.

Every fenced Python block must parse (`tests/test_docs.py` runs `ast.parse` on them, allowing bare signature displays).

- [ ] **Step 5: Link the new page and describe the new keys**

- `docs/README.md`: add a Reference row `| [Decision models](reference/decide.md) | The decider path: Jev, questions from slots, screen/choose/gate, thresholds, fitting. |` and a Worked-examples note that `cafeteria_fcd` and `ref_zivila` ship a `job_jev.yaml`.
- `docs/components.md`: a section `## Decision models — [reference](reference/decide.md)` of four to six sentences after the LLM clients section.
- `docs/reference/job-file.md`: document `templates.queries`, the `decider:` block with every key and default, the `policy:` keys that apply when `decider:` is present, and `prompts.slots.properties`.
- `docs/reference/cli.md`: document `xwalk fit` (flags, exit codes) and note that `ablate` and `prompts optimize` refuse decider jobs.
- `docs/concepts.md`: one paragraph under "The loop" saying there is a second loop for decision models and linking to the reference.

- [ ] **Step 6: Run the docs test and the full suite**

Run: `.venv/bin/pytest tests/test_docs.py tests/test_examples.py -q && .venv/bin/pytest -q`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add examples docs/reference/decide.md docs/README.md docs/components.md docs/reference/job-file.md docs/reference/cli.md docs/concepts.md
git commit -m "docs: the decider path, and Jev example jobs for cafeteria_fcd and ref_zivila

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: Evaluation runs (no code; results go in the spec)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-22-jev-decider-design.md` (append a "Results" section)
- Create: `scripts/run_jev_eval.sh`

This task produces numbers, not code. Everything runs with the real endpoint and costs well under a dollar in total.

- [ ] **Step 1: Write `scripts/run_jev_eval.sh`**

```bash
#!/usr/bin/env bash
# Run the four gold samples through the decider path and evaluate each.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
for ex in cafeteria_fcd chebi ncbi_disease nlm_gene; do
  job="examples/$ex/job_jev.yaml"
  if [ ! -f "$job" ]; then
    # Only cafeteria_fcd ships one; derive the rest from the LLM job by swapping the block.
    .venv/bin/python - "$ex" <<'EOF'
import sys, yaml, pathlib
ex = sys.argv[1]
src = pathlib.Path(f"examples/{ex}/job.yaml"); data = yaml.safe_load(src.read_text())
data.pop("llm"); data.pop("selector", None)
data["decider"] = {"kind": "jev", "model": "~typesafe/jev-latest",
                   "base_url": "https://openrouter.ai/api/alpha/decisions", "api_key_env": "XWALK_TEST_API_KEY"}
data["policy"] = {"accept_at": 0.85, "screen_floor": 0.30, "chunk_size": 50, "max_candidates": 300, "concurrency": 16}
for r in data["retrievers"]: r["limit"] = 200
data["name"] = f"{ex}_jev"
pathlib.Path(f"examples/{ex}/job_jev.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
EOF
  fi
  .venv/bin/xwalk index --job "$job" --out "runs/jev_eval/$ex/index"
  .venv/bin/xwalk match --job "$job" --out "runs/jev_eval/$ex" --index "runs/jev_eval/$ex/index" || true
  .venv/bin/xwalk eval  --run "runs/jev_eval/$ex" --gold "examples/$ex/sample/gold.csv" --out "runs/jev_eval/$ex/eval"
  .venv/bin/xwalk fit   --run "runs/jev_eval/$ex" --gold "examples/$ex/sample/gold.csv" --precision 0.95 | tail -3
done
```

`chmod +x scripts/run_jev_eval.sh`. The generated `job_jev.yaml` files for chebi, ncbi_disease, and nlm_gene are kept and committed so the runs are reproducible.

- [ ] **Step 2: Run it**

Run: `scripts/run_jev_eval.sh 2>&1 | tee runs/jev_eval/log.txt`
Expected: four run directories with `eval.json`, and four `fit` recommendations. Record per domain: accuracy on labelled rows, matched/needs_review/unmatched counts, precision of `matched`, mean calls and cost per record, and the recommended `accept_at`.

- [ ] **Step 3: Run the LLM baseline on the same samples for a like-for-like comparison**

For each `ex`, with `XWALK_TEST_MODEL` pointed at the model the team uses on OpenRouter (`qwen/qwen3-next-80b-a3b-instruct` today), run the existing `examples/$ex/job.yaml` into `runs/llm_eval/$ex` and `xwalk compare --gold examples/$ex/sample/gold.csv --run llm=runs/llm_eval/$ex --run jev=runs/jev_eval/$ex`. Record the comparison tables.

- [ ] **Step 4: Screen-stage recall**

Write `scripts/screen_recall.py` (about 40 lines). It opens a decider run's ledger and, for every labelled row, reports whether a gold id is (a) among `attempt.candidates` (retrieval recall), (b) among the shortlist, which is the set of record ids `attempt.issued_keys[key]` for every `screen_<key>` entry in `attempt.signals` (Task 12 stores exactly the shortlist's keys), and (c) equal to `matched_id` (final). Print the three recalls per run. This tells us whether the next effort belongs in retrieval or in the questions.

```python
#!/usr/bin/env python
"""Retrieval, shortlist, and final recall of a decider run against gold."""
import sys
from pathlib import Path

from xwalk.evaluate.gold import load_gold_csv
from xwalk.ledger import Ledger

run, gold_path = Path(sys.argv[1]), sys.argv[2]
gold = load_gold_csv(gold_path)
ledger = Ledger.open(run / "ledger.sqlite")
import json
fp = json.loads((run / "manifest.json").read_text())["run_fingerprint"]
retrieved = shortlisted = final = labelled = 0
for r in ledger.iter_results(fp):
    if r.source_id not in gold or not gold.get(r.source_id):
        continue
    labelled += 1
    a = r.attempts[0]
    ids = {c.id for c in a.candidates}
    short = {a.issued_keys[k[len("screen_"):]] for k in a.signals if k.startswith("screen_C")}
    want = gold.get(r.source_id) or frozenset()
    retrieved += bool(want & ids)
    shortlisted += bool(want & short)
    final += r.matched_id in want
ledger.close()
print(f"labelled {labelled}: retrieval {retrieved/labelled:.2f} shortlist {shortlisted/labelled:.2f} final {final/labelled:.2f}")
```

- [ ] **Step 5: Ref_zivila FoodOn**

```bash
set -a; source .env; set +a
.venv/bin/xwalk match --job examples/ref_zivila/jobs/foodon/job_jev.yaml \
  --index runs/ref_zivila/foodon/index --out runs/ref_zivila/foodon_jev
.venv/bin/python scripts/compare_runs.py \
  --run-a runs/ref_zivila/foodon_qwen/mapping.csv --run-b runs/ref_zivila/foodon_jev/mapping.csv \
  --name-a Qwen --name-b Jev --ensemble-out runs/ref_zivila/ensemble_qwen_jev.csv
```

The existing index was built with `limit: 50`; rebuild it with the new job (`xwalk index --job examples/ref_zivila/jobs/foodon/job_jev.yaml --out runs/ref_zivila/foodon_jev/index`) so retrieval depth matches, and pass that index. Sample 40 three-way disagreements (Qwen, Nex, Jev) and adjudicate them by hand into `runs/ref_zivila/adjudicated_sample.csv`.

- [ ] **Step 6: Record results in the spec and commit**

Append a `## Results (2026-09-…)` section to the spec with the tables from Steps 2 to 5 and one paragraph of what to do next (retrieval versus questions, and the fitted thresholds to adopt as the example defaults). Update the two example jobs' `policy:` with the fitted values if they differ materially.

```bash
git add scripts/run_jev_eval.sh scripts/screen_recall.py examples/*/job_jev.yaml docs/superpowers/specs/2026-09-22-jev-decider-design.md
git commit -m "eval: decider path on four gold samples and Ref_zivila FoodOn, thresholds fitted

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Deferred (not in this plan)

- Hierarchy descent over an ontology's parent links as a retrieval strategy.
- Self-consistency (repeat the gate questions and average) as a verifier substitute.
- Multi-label output from screen probabilities.
- An LLM writing explanations for `needs_review` rows only.

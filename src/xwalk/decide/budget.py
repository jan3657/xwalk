"""Call limits and complete usage accounting for decision clients.

The decider-path counterpart of `xwalk.llm.budget`: the same `CallBudget` (one budget can
be shared by a run's decider and its rewrite LLM, so `--max-calls` caps both), reserved
synchronously immediately before every dispatch. A refused reservation raises
`DecisionCallLimitExceeded`, a `DecisionFatalError`: the matcher turns it into
`fatal_provider_failure`, which aborts the batch without committing that record.

`JevClient` takes the budget through `attach_call_budget` and reserves once per HTTP
request, so its internal retries are counted and limited; any other client is charged
one reservation per `decide` call. Give each wrapper its own budget, or a
`CallBudget.share()` view of a run-wide one, so each counts only its own dispatches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xwalk.decide.base import (
    DecisionClient,
    DecisionFatalError,
    DecisionResponse,
    Question,
    failure_usage,
)
from xwalk.llm.budget import CALL_LIMIT_CODE, CallBudget, CallLimitExceeded
from xwalk.records import Usage


class DecisionCallLimitExceeded(DecisionFatalError):
    """The run's call allowance is spent. Nothing was dispatched for this request."""

    code = CALL_LIMIT_CODE


def reserve(budget: CallBudget) -> None:
    """Reserve one dispatch from `budget`, or raise `DecisionCallLimitExceeded`."""
    try:
        budget.acquire()
    except CallLimitExceeded as exc:
        raise DecisionCallLimitExceeded(str(exc), usage=Usage.zero()) from None


class BudgetedDecider:
    """A `DecisionClient` that enforces a `CallBudget` and keeps complete accounting.

    Identity (`model`, `fingerprint`) delegates to the inner client: a limit changes how
    much work is done, never what a result means. `usage` counts every dispatched
    request; one that never reported back (failed or cancelled) is a call with unknown
    usage, never a free one.
    """

    def __init__(self, inner: DecisionClient, budget: CallBudget | None = None) -> None:
        self._inner = inner
        self.budget = budget or CallBudget()
        attach: Any = getattr(inner, "attach_call_budget", None)
        self._gated_inside = callable(attach)
        if self._gated_inside:
            attach(self.budget)
        self._reported = Usage.zero()

    @property
    def inner(self) -> DecisionClient:
        return self._inner

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    @property
    def usage(self) -> Usage:
        missing = self.budget.dispatched - self._reported.calls
        return self._reported + (Usage.unreported(missing) if missing > 0 else Usage.zero())

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        if not self._gated_inside:
            reserve(self.budget)
        try:
            response = await self._inner.decide(state, questions)
        except Exception as exc:
            if not isinstance(exc, DecisionCallLimitExceeded):
                self._reported = self._reported + failure_usage(exc)
            elif isinstance(exc.usage, Usage):
                self._reported = self._reported + exc.usage
            raise
        self._reported = self._reported + response.usage
        return response

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if callable(close):
            await close()


__all__ = ["BudgetedDecider", "DecisionCallLimitExceeded", "reserve"]

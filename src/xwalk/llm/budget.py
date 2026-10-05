"""Per-run call limits, enforced where calls are dispatched.

A `CallBudget` is an allowance of upstream requests. Every dispatch *reserves* one unit
first, synchronously: in asyncio nothing can run between the check and the increment, so
however many record tasks are in flight, the number of dispatched requests can never
exceed the limit. A refused reservation raises `CallLimitExceeded`, a fatal provider
error: the matcher turns it into `fatal_provider_failure`, which aborts the batch without
committing that record (resume picks it up).

`BudgetedLLM` wraps any client. A client that retries internally (the OpenAI-compatible
client) accepts the budget through `attach_call_budget` and reserves once per HTTP
request, so its retries and structured-output fallbacks are counted and limited too;
any other client is charged one reservation per `complete` call. Retries hidden inside a
client that offers no hook (LiteLLM's own `num_retries`) cannot be observed or limited.

`BudgetedLLM.usage` is the invocation's complete accounting: the usage reported by every
call that returned or raised, plus every reserved dispatch that never reported back (a
call cancelled in flight), counted as a call with unknown usage -- never as free.
"""

from __future__ import annotations

from typing import Any

from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    failure_usage,
)
from xwalk.records import Usage

CALL_LIMIT_CODE = "call_limit_reached"


class CallLimitExceeded(LLMFatalError):
    """The run's call allowance is spent. Nothing was dispatched for this request."""

    code = CALL_LIMIT_CODE


class CallBudget:
    """An allowance of upstream dispatches. `limit=None` counts without limiting."""

    def __init__(self, limit: int | None = None) -> None:
        if limit is not None and limit < 0:
            raise ValueError(f"a call limit must be >= 0, got {limit}")
        self.limit = limit
        self.dispatched = 0
        self.refused = 0

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else self.limit - self.dispatched

    @property
    def exhausted(self) -> bool:
        """True once a reservation has been refused."""
        return self.refused > 0

    def acquire(self) -> None:
        """Reserve one dispatch, or raise `CallLimitExceeded`. Call it immediately
        before sending a request; it never awaits, so concurrent callers cannot overshoot."""
        if self.limit is not None and self.dispatched >= self.limit:
            self.refused += 1
            raise CallLimitExceeded(
                f"call limit of {self.limit} reached; nothing more is dispatched in this "
                f"invocation (raise --max-calls and resume to continue)",
                usage=Usage.zero(),
            )
        self.dispatched += 1

    def share(self) -> CallBudget:
        """A view that draws on this budget but counts its own dispatches.

        Two clients of one run (the decider and its rewrite LLM) share one limit this
        way while each wrapper's `usage` still counts only its own requests.
        """
        return _SharedBudget(self)


class _SharedBudget(CallBudget):
    def __init__(self, parent: CallBudget) -> None:
        super().__init__(parent.limit)
        self._parent = parent

    @property
    def remaining(self) -> int | None:
        return self._parent.remaining

    @property
    def exhausted(self) -> bool:
        return self._parent.exhausted

    def acquire(self) -> None:
        try:
            self._parent.acquire()
        except CallLimitExceeded:
            self.refused += 1
            raise
        self.dispatched += 1


class BudgetedLLM:
    """An `LLMClient` that enforces a `CallBudget` and keeps complete usage accounting.

    Identity (`model`, `capabilities`, `fingerprint`) delegates to the inner client:
    a limit changes how much work is done, never what a result means.
    """

    def __init__(self, inner: LLMClient, budget: CallBudget | None = None) -> None:
        self._inner = inner
        self.budget = budget or CallBudget()
        attach: Any = getattr(inner, "attach_call_budget", None)
        self._gated_inside = callable(attach)
        if self._gated_inside:
            attach(self.budget)
        self._reported = Usage.zero()

    @property
    def inner(self) -> LLMClient:
        return self._inner

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._inner.capabilities

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    @property
    def usage(self) -> Usage:
        """Everything dispatched through this client so far.

        Dispatches that never reported (cancelled in flight) count as calls with
        unknown usage.
        """
        missing = self.budget.dispatched - self._reported.calls
        return self._reported + (Usage.unreported(missing) if missing > 0 else Usage.zero())

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._gated_inside:
            self.budget.acquire()
        try:
            response = await self._inner.complete(request)
        except Exception as exc:
            if isinstance(exc, CallLimitExceeded):
                self._reported = self._reported + (exc.usage or Usage.zero())
            else:
                self._reported = self._reported + failure_usage(exc)
            raise
        self._reported = self._reported + response.usage
        return response

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if callable(close):
            await close()


__all__ = ["CALL_LIMIT_CODE", "BudgetedLLM", "CallBudget", "CallLimitExceeded"]

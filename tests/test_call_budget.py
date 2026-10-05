"""Per-run call limits (task 03): reservation before dispatch, retries included."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.conftest import FIXTURES
from tests.test_matcher import score_reply, select_reply
from xwalk.batch import RunState, run_batch
from xwalk.config import load_job
from xwalk.llm.base import LLMRequest
from xwalk.llm.budget import BudgetedLLM, CallBudget, CallLimitExceeded
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.records import DecisionReason, Record, Usage


def _reply(request: LLMRequest) -> str:
    if "## Candidates" not in request.user:
        return score_reply(0.95)
    return select_reply("C01") if "[C01]" in request.user else select_reply(None)


class _InterleavingLLM(FakeLLM):
    """Yields to the event loop inside every call, so concurrent record tasks really
    interleave their dispatches -- the case a check-then-await budget would get wrong."""

    def __init__(self) -> None:
        super().__init__(handler=_reply)
        self.dispatched = 0

    async def complete(self, request):  # type: ignore[override]
        self.dispatched += 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return await super().complete(request)


def _sources(n: int) -> list[Record]:
    words = ["glucose", "dextrose", "table sugar", "fructose", "lactose"]
    return [Record(id=f"s{i}", fields={"mention": words[i % len(words)]}) for i in range(n)]


async def test_a_concurrent_run_never_dispatches_more_than_the_allowance(tmp_path):
    job = load_job(FIXTURES / "job_tiny.yaml")
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, job.build_templates(), tmp_path / "idx")
    inner = _InterleavingLLM()
    llm = BudgetedLLM(inner, CallBudget(limit=5))
    matcher = job.build_matcher(store=store, retrievers=retrievers, llm=llm)
    assert matcher.policy.concurrency == 8

    report = await run_batch(matcher, _sources(30), out=tmp_path / "run")

    assert inner.dispatched <= 5
    assert llm.budget.dispatched == inner.dispatched
    assert llm.budget.exhausted
    assert report.run_state is RunState.ABORTED
    assert report.errors[0].code == DecisionReason.FATAL_PROVIDER_FAILURE.value
    assert "call limit of 5" in report.errors[0].message


async def test_an_unlimited_budget_only_counts():
    llm = BudgetedLLM(FakeLLM(["a", "b"]))
    await llm.complete(LLMRequest(system="s", user="u"))
    await llm.complete(LLMRequest(system="s", user="u"))
    assert llm.budget.dispatched == 2 and not llm.budget.exhausted
    assert llm.usage == Usage(prompt_tokens=200, completion_tokens=40, calls=2)


async def test_a_refused_call_is_not_dispatched_and_costs_nothing():
    inner = FakeLLM(["a"])
    llm = BudgetedLLM(inner, CallBudget(limit=1))
    await llm.complete(LLMRequest(system="s", user="u"))
    with pytest.raises(CallLimitExceeded) as info:
        await llm.complete(LLMRequest(system="s", user="u"))
    assert info.value.usage == Usage.zero()
    assert len(inner.requests) == 1


async def test_client_side_http_retries_spend_the_allowance():
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    inner = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        max_retries=5,
        backoff_base=0.0,
    )
    llm = BudgetedLLM(inner, CallBudget(limit=2))
    with pytest.raises(CallLimitExceeded) as info:
        await llm.complete(LLMRequest(system="s", user="u"))
    assert len(posts) == 2  # not 6: the third POST was refused before dispatch
    assert info.value.usage == Usage.unreported(2)
    assert llm.usage == Usage.unreported(2)


async def test_a_call_cancelled_in_flight_counts_as_unknown_usage():
    started = asyncio.Event()

    class Hanging(FakeLLM):
        async def complete(self, request):  # type: ignore[override]
            started.set()
            await asyncio.Event().wait()

    llm = BudgetedLLM(Hanging(handler=_reply))
    task = asyncio.create_task(llm.complete(LLMRequest(system="s", user="u")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.usage == Usage.unreported(1)


def test_identity_is_the_inner_clients():
    inner = FakeLLM(["a"])
    llm = BudgetedLLM(inner, CallBudget(limit=3))
    assert llm.fingerprint == inner.fingerprint and llm.model == inner.model


def test_a_negative_limit_is_rejected():
    with pytest.raises(ValueError):
        CallBudget(limit=-1)

"""What a failing provider looks like in a run's summary (review dx P1).

A dead endpoint used to be reported only as `call_limit_reached` with the advice to
raise `--max-calls`, and an uncapped run against it ended with bare `failed` counts.
The client's own error has to reach the summary, the envelope and stderr.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.test_ops import _job_file
from xwalk import ops
from xwalk.cli.main import main
from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import OpenAICompatClient

DEAD = "exhausted 5 retries: ConnectError: All connection attempts failed"


def _dead_llm() -> FakeLLM:
    return FakeLLM(handler=lambda request: LLMRetryableError(DEAD))


def test_a_call_limit_spent_on_retries_names_the_provider_error(tmp_path):
    result = ops.run(_job_file(tmp_path), tmp_path / "run", llm=_dead_llm(), max_calls=3)
    assert result.run is not None and result.run["run_state"] == "aborted"
    limits = [e for e in result.errors if e.code == "call_limit_reached"]
    assert len(limits) == 1
    assert "ConnectError: All connection attempts failed" in limits[0].message
    assert "raise --max-calls" not in limits[0].message


def test_a_call_limit_reached_while_healthy_keeps_the_raise_advice(tmp_path):
    from tests.test_ops import _llm

    result = ops.run(_job_file(tmp_path), tmp_path / "run", llm=_llm(), max_calls=3)
    limits = [e for e in result.errors if e.code == "call_limit_reached"]
    assert "raise --max-calls" in limits[0].message


def test_the_openai_client_retries_that_spend_the_limit_are_named(tmp_path):
    """The real client retries inside one `complete`; the budget runs out mid-loop."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed")

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        backoff_base=0.0,
        transport=httpx.MockTransport(refuse),
    )
    result = ops.run(_job_file(tmp_path), tmp_path / "run", llm=client, max_calls=5)
    limits = [e for e in result.errors if e.code == "call_limit_reached"]
    assert len(limits) == 1
    assert "ConnectError: All connection attempts failed" in limits[0].message
    assert "raise --max-calls" not in limits[0].message


def test_failed_rows_are_summarised_by_distinct_error_with_counts(tmp_path):
    result = ops.run(_job_file(tmp_path), tmp_path / "run", llm=_dead_llm())
    assert result.run is not None and result.run["run_state"] == "failed"
    failed = result.counts["failed"]
    assert failed >= 2  # one source has no candidates and makes no call
    failures = [e for e in result.errors if e.code == "provider_failure"]
    assert len(failures) == 1, failures  # deduplicated, not one per record
    assert failures[0].message.startswith(f"{failed} record(s) failed:")
    assert DEAD in failures[0].message
    assert any(DEAD in line for line in result.lines)


def test_an_uncapped_failing_run_warns_about_the_missing_limit(tmp_path):
    uncapped = ops.run(_job_file(tmp_path), tmp_path / "a", llm=_dead_llm())
    assert any(w.code == "uncapped_failures" for w in uncapped.warnings)
    capped = ops.run(_job_file(tmp_path), tmp_path / "b", llm=_dead_llm(), max_calls=100)
    assert not any(w.code == "uncapped_failures" for w in capped.warnings)


def test_inspect_shows_the_failure_summary_too(tmp_path):
    out = tmp_path / "run"
    ops.run(_job_file(tmp_path), out, llm=_dead_llm())
    inspected = ops.inspect(out)
    assert any(e.code == "provider_failure" and DEAD in e.message for e in inspected.errors)


@pytest.fixture
def dead_cli(monkeypatch):
    import xwalk.config as config

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(config.JobSpec, "build_llm", lambda self: _dead_llm())


def test_the_cli_prints_the_cause_on_stderr_and_one_object_with_json(
    tmp_path: Path, dead_cli, capsys
):
    job = _job_file(tmp_path)
    code = main(["match", "--job", str(job), "--out", str(tmp_path / "a"), "--max-calls", "3"])
    assert code == 3
    err = capsys.readouterr().err
    assert "ConnectError: All connection attempts failed" in err
    assert "raise --max-calls" not in err

    code = main(["match", "--job", str(job), "--out", str(tmp_path / "b"), "--json"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)  # exactly one object
    assert any(e["code"] == "provider_failure" for e in envelope["errors"])
    assert any(w["code"] == "uncapped_failures" for w in envelope["warnings"])


def test_the_cli_warns_once_while_an_uncapped_run_keeps_failing(tmp_path, dead_cli, capsys):
    main(["match", "--job", str(_job_file(tmp_path)), "--out", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert err.count("no --max-calls") == 1


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("All connection attempts failed")


def _limit_message(errors) -> str:
    (message,) = [e.message for e in errors if e.code == "call_limit_reached"]
    return message


def test_a_dead_decider_endpoint_names_the_cause(tmp_path):
    """JevClient retries inside one `decide`; the budget runs out mid-loop."""
    from tests.test_decide_ops import JEV_JOB
    from xwalk.decide.jev import JevClient

    client = JevClient(
        "https://example.test/decisions",
        "m",
        transport=httpx.MockTransport(_refused),
        backoff_base=0.0,
    )
    result = ops.run(JEV_JOB, tmp_path / "run", decider=client, max_calls=3)
    message = _limit_message(result.errors)
    assert "ConnectError: All connection attempts failed" in message
    assert "raise --max-calls" not in message


def test_match_inspect_and_the_manifest_report_the_same_stop(tmp_path):
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        backoff_base=0.0,
        transport=httpx.MockTransport(_refused),
    )
    out = tmp_path / "run"
    matched = _limit_message(ops.run(_job_file(tmp_path), out, llm=client, max_calls=5).errors)
    inspected = _limit_message(ops.inspect(out).errors)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    stored = [e["message"] for e in manifest["errors"]]
    assert stored == [matched] and inspected == matched
    assert "ConnectError" in matched and "raise --max-calls" not in matched


def test_a_transient_failure_that_recovered_is_not_blamed_for_the_limit(tmp_path):
    """One failed call, then healthy ones: the limit was spent on real work."""
    from tests.test_ops import _llm

    healthy = _llm()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return LLMRetryableError(DEAD) if calls == 1 else healthy._handler(request)

    result = ops.run(
        _job_file(tmp_path), tmp_path / "run", llm=FakeLLM(handler=handler), max_calls=5
    )
    message = _limit_message(result.errors)
    assert "raise --max-calls" in message
    assert "provider was failing" not in message


# --- the cause travels with the budget (review dx2 #2, #3) ---------------------------


async def test_run_batch_reports_the_provider_error_that_spent_the_limit(tmp_path):
    """The public run_batch API, without ops: the refusal itself names the cause."""
    from tests.test_batch import RETRIEVER, SOURCES, make_matcher
    from tests.test_matcher import ScriptedRetriever
    from xwalk.batch import run_batch
    from xwalk.llm.budget import BudgetedLLM, CallBudget

    llm = BudgetedLLM(_dead_llm(), CallBudget(1))
    report = await run_batch(
        make_matcher(llm, ScriptedRetriever(RETRIEVER)), SOURCES, out=tmp_path / "run"
    )
    assert report.errors, report
    message = report.errors[0].message
    assert DEAD in message and "raise --max-calls" not in message


def test_a_late_failure_of_an_earlier_request_does_not_override_a_later_success():
    from xwalk.llm.budget import CallBudget, CallLimitExceeded

    budget = CallBudget(2)
    first, second = budget.acquire(), budget.acquire()
    budget.settle(second, None)
    budget.settle(first, "ConnectError: refused")  # reported after the later success
    with pytest.raises(CallLimitExceeded) as refused:
        budget.acquire()
    assert "raise --max-calls" in str(refused.value)
    assert "ConnectError" not in str(refused.value)


async def test_a_delayed_retry_after_later_successes_does_not_blame_the_provider():
    """Request A fails and backs off; B and C succeed; A's retry hits the limit."""
    import asyncio

    from tests.test_openai_compat import ok_body
    from xwalk.llm.base import LLMRequest
    from xwalk.llm.budget import BudgetedLLM, CallBudget, CallLimitExceeded

    def handler(request: httpx.Request) -> httpx.Response:
        if b"dead-request" in request.content:
            raise httpx.ConnectError("All connection attempts failed")
        return httpx.Response(200, json=ok_body())

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        backoff_base=0.05,
        transport=httpx.MockTransport(handler),
    )
    llm = BudgetedLLM(client, CallBudget(3))

    async def later(user: str):
        await asyncio.sleep(0.01)
        return await llm.complete(LLMRequest(system="", user=user))

    a, b, c = await asyncio.gather(
        llm.complete(LLMRequest(system="", user="dead-request")),
        later("B"),
        later("C"),
        return_exceptions=True,
    )
    assert not isinstance(b, BaseException) and not isinstance(c, BaseException)
    assert isinstance(a, CallLimitExceeded)
    assert "raise --max-calls" in str(a) and "provider was failing" not in str(a)
    assert "ConnectError" not in str(a)


@pytest.mark.parametrize(
    "body",
    [{"error": {"code": 503, "message": "upstream unavailable"}}, {"choices": []}],
    ids=["error-body", "unparseable"],
)
async def test_a_200_that_is_not_an_answer_counts_as_a_provider_failure(body):
    """The status line said OK; the body did not (review dx3 #3)."""
    from xwalk.llm.base import LLMError, LLMRequest
    from xwalk.llm.budget import BudgetedLLM, CallBudget

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        backoff_base=0.0,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    budget = CallBudget(1)
    with pytest.raises(LLMError):  # a fatal body error, or the spent limit on its retry
        await BudgetedLLM(client, budget).complete(LLMRequest(system="", user="u"))
    assert budget.provider_error is not None
    assert "raise --max-calls" not in budget.limit_message()

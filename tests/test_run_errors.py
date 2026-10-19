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


def test_a_fatal_limit_result_carries_the_earlier_attempt_error():
    """The decider path has no observing wrapper: run_batch names the cause instead."""
    from dataclasses import replace

    from tests.test_decide_fit import _result
    from xwalk.batch import _fatal_error

    base = _result("s1", None, 0.5)
    first = replace(base.attempts[0], error="decider: " + DEAD)
    limit = "decider: call limit of 3 reached; nothing more is dispatched"
    second = replace(base.attempts[0], index=1, error=limit)
    error = _fatal_error(replace(base, attempts=(first, second)))
    assert error.message == f"{limit}; last provider error: decider: {DEAD}"

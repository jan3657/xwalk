"""The decider path under the 0.2 contracts: ops, envelopes, aborts, budgets, caching.

Offline throughout: FakeDecider or an httpx MockTransport, never the Jev endpoint.
"""

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
import yaml

from xwalk import ops
from xwalk.cli.main import main
from xwalk.config import JobValidationError, load_job
from xwalk.decide.base import DecisionFatalError, Noul, NoulAnswer
from xwalk.decide.budget import BudgetedDecider, DecisionCallLimitExceeded
from xwalk.decide.fake import FakeDecider
from xwalk.decide.jev import JevClient
from xwalk.ledger import LEDGER_SCHEMA_VERSION
from xwalk.llm.budget import CallBudget

FIXTURES = Path(__file__).parent / "fixtures"
JEV_JOB = FIXTURES / "job_tiny_jev.yaml"


def _job(tmp_path, mutate):
    data = yaml.safe_load(JEV_JOB.read_text(encoding="utf-8"))
    for block in ("target", "source"):
        data[block]["path"] = str((FIXTURES / data[block]["path"]).resolve())
    data["prompts"]["slots"] = str((FIXTURES / data["prompts"]["slots"]).resolve())
    mutate(data)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _envelope(capsys):
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return json.loads(out[0])


def test_a_decider_run_goes_through_ops_with_an_envelope(tmp_path, monkeypatch, capsys):
    from xwalk.cli import main as cli

    monkeypatch.setenv("XWALK_TEST_API_KEY", "sk-secret-value")
    monkeypatch.setattr(cli, "_build_decider", lambda job: FakeDecider())
    out = tmp_path / "run"
    code = main(["match", "--job", str(JEV_JOB), "--out", str(out), "--json"])
    envelope = _envelope(capsys)
    assert code == envelope["exit_code"] and code in (0, 1)
    assert envelope["run"]["run_state"] == "complete"
    assert envelope["data"]["path"] == "decider" and envelope["data"]["model"] == "fake-decider"
    assert envelope["usage"]["calls"] > 0 and envelope["usage"]["cache_hits"] == 0
    assert envelope["counts"]["total"] == 4
    manifest_text = (out / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["fingerprint_components"]["path"] == "decider"
    assert manifest["ledger_schema_version"] == LEDGER_SCHEMA_VERSION == 2
    assert "sk-secret-value" not in manifest_text


def test_a_rerun_replays_decisions_as_cache_hits(tmp_path):
    out = tmp_path / "run"
    first = FakeDecider()
    ops.run(JEV_JOB, out, decider=first)
    second = FakeDecider()
    result = ops.run(JEV_JOB, out, resume=False, decider=second)
    assert second.calls == []
    assert result.usage is not None
    assert result.usage["calls"] == 0 and result.usage["cache_hits"] == len(first.calls)
    # The decision cache lives in the v2 ledger's existing llm_cache table.
    conn = sqlite3.connect(out / "ledger.sqlite")
    try:
        (rows,) = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        (version,) = conn.execute(
            "SELECT value FROM ledger_meta WHERE key = 'schema_version'"
        ).fetchone()
    finally:
        conn.close()
    assert rows == len(first.calls) and version == "2"


def test_a_fatal_decider_error_aborts_without_committing(tmp_path):
    out = tmp_path / "run"
    broken = FakeDecider(handler=lambda s, q: DecisionFatalError("HTTP 401: bad key"))
    result = ops.run(JEV_JOB, out, decider=broken)
    assert result.exit_code == ops.EXIT_RUNTIME
    assert result.run is not None and result.run["run_state"] == "aborted"
    assert any(e.code == "fatal_provider_failure" for e in result.errors)
    assert "failed" not in result.counts  # the fatal record was not committed
    # Resume retries it.
    again = ops.run(JEV_JOB, out, decider=FakeDecider())
    assert again.run is not None and again.run["run_state"] == "complete"


def test_max_calls_aborts_a_decider_run_and_resume_continues(tmp_path):
    out = tmp_path / "run"
    result = ops.run(JEV_JOB, out, max_calls=1, decider=FakeDecider())
    assert result.exit_code == ops.EXIT_RUNTIME
    assert [e.code for e in result.errors] == ["call_limit_reached"]
    assert result.usage is not None and result.usage["calls"] == 1
    finished = ops.run(JEV_JOB, out, decider=FakeDecider())
    assert finished.run is not None and finished.run["run_state"] == "complete"


def test_another_decider_policy_is_refused_in_the_same_run_directory(tmp_path):
    out = tmp_path / "run"
    ops.run(JEV_JOB, out, decider=FakeDecider())
    other = _job(tmp_path, lambda d: d["policy"].update(accept_at=0.9))
    with pytest.raises(ops.OpError) as info:
        ops.run(other, out, decider=FakeDecider())
    assert info.value.result.exit_code == ops.EXIT_RUNTIME
    assert info.value.result.errors[0].code == "run_fingerprint_mismatch"
    assert "policy" in info.value.result.errors[0].message


def test_decider_policy_typos_get_a_did_you_mean(tmp_path):
    path = _job(tmp_path, lambda d: d["policy"].update(acept_at=0.9))
    with pytest.raises(JobValidationError) as info:
        load_job(path)
    (issue,) = info.value.issues
    assert issue.code == "unknown_field" and issue.loc == "policy.acept_at"
    assert "did you mean 'accept_at'" in issue.message


@pytest.mark.parametrize(
    ("mutate", "where"),
    [
        (lambda d: d["policy"].update(screen_floor=1.5), "policy.screen_floor"),
        (lambda d: d["policy"].update(chunk_size=0), "policy.chunk_size"),
        (lambda d: d["decider"].update(timeout=0), "decider.timeout"),
        (lambda d: d["decider"].update(modle="x"), "decider.modle"),
        (lambda d: d.update(selector={"max_candidates": 3}), ""),
    ],
)
def test_the_decider_blocks_are_validated_strictly(tmp_path, mutate, where):
    with pytest.raises(JobValidationError) as info:
        load_job(_job(tmp_path, mutate))
    assert [issue.loc for issue in info.value.issues] == [where]


def test_validate_checks_the_decider_credential_without_a_call(monkeypatch, capsys):
    monkeypatch.delenv("XWALK_TEST_API_KEY", raising=False)
    code = main(["validate", "--job", str(JEV_JOB), "--json"])
    envelope = _envelope(capsys)
    assert code == 2
    assert envelope["data"]["path"] == "decider" and envelope["data"]["paid_calls"] == 0
    assert envelope["data"]["credentials"] == {"XWALK_TEST_API_KEY": "missing"}
    assert "decider.api_key_env" in envelope["errors"][0]["message"]


def test_a_missing_decider_credential_is_a_usage_error(tmp_path, monkeypatch):
    monkeypatch.delenv("XWALK_TEST_API_KEY", raising=False)
    with pytest.raises(ops.OpError) as info:
        ops.run(JEV_JOB, tmp_path / "run")
    assert info.value.result.exit_code == ops.EXIT_USAGE
    assert info.value.result.errors[0].code == "credential_missing"


# --- the budget and per-request accounting -----------------------------------------


def _jev(statuses):
    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        body = {"answers": {"n": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": 7}}
        return httpx.Response(status, json=body if status == 200 else {"error": "busy"})

    return JevClient(
        "https://example.test/decisions",
        "m",
        transport=httpx.MockTransport(handler),
        backoff_base=0.0,
        max_retries=3,
    )


async def test_every_http_retry_is_a_counted_call():
    budgeted = BudgetedDecider(_jev([429, 200]), CallBudget(5))
    response = await budgeted.decide("s", {"n": Noul(instructions="q?")})
    assert (response.usage.calls, response.usage.unknown_calls) == (2, 1)
    assert budgeted.usage == response.usage
    assert budgeted.budget.dispatched == 2


async def test_the_budget_stops_retries_too():
    budgeted = BudgetedDecider(_jev([429, 429, 200]), CallBudget(2))
    with pytest.raises(DecisionCallLimitExceeded, match="call limit of 2"):
        await budgeted.decide("s", {"n": Noul(instructions="q?")})
    assert budgeted.usage.calls == 2 and budgeted.budget.exhausted


async def test_a_shared_budget_counts_each_client_once():
    budget = CallBudget(3)
    one = BudgetedDecider(FakeDecider(lambda s, q: {"n": NoulAnswer(noul=1.0)}), budget.share())
    two = BudgetedDecider(FakeDecider(lambda s, q: {"n": NoulAnswer(noul=1.0)}), budget.share())
    question = {"n": Noul(instructions="q?")}
    await one.decide("a", question)
    await two.decide("b", question)
    await two.decide("c", question)
    assert (one.usage.calls, two.usage.calls, budget.dispatched) == (1, 2, 3)
    with pytest.raises(DecisionCallLimitExceeded):
        await one.decide("d", question)
    assert budget.exhausted

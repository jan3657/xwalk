"""`--json`: exactly one parseable object on stdout, success or failure (CONTRACTS.md
section 8). Logs and progress stay on stderr."""

from __future__ import annotations

import json
import signal
import sys

import pytest

from tests.conftest import FIXTURES
from tests.test_cli import _scripted_llm
from xwalk.cli.main import main

JOB = str(FIXTURES / "job_tiny.yaml")
KEYS = {
    "schema_version",
    "operation",
    "status",
    "exit_code",
    "run",
    "counts",
    "usage",
    "artifacts",
    "warnings",
    "errors",
    "data",
}


def _envelope(capsys, code: int) -> dict:
    out = capsys.readouterr().out
    envelope = json.loads(out)  # the whole of stdout is one JSON document
    assert set(envelope) == KEYS
    assert envelope["schema_version"] == 1
    assert envelope["exit_code"] == code
    return envelope


@pytest.fixture
def scripted(monkeypatch):
    from xwalk.config import JobSpec

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: _scripted_llm())


def test_match_json_is_clean_and_progress_goes_to_stderr(tmp_path, scripted, capsys):
    code = main(["match", "--job", JOB, "--out", str(tmp_path / "run"), "--json"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert code == envelope["exit_code"] == 0
    assert envelope["status"] == "ok"
    assert envelope["operation"] == "match"
    assert envelope["run"]["run_state"] == "complete"
    assert envelope["counts"] == {"matched": 3, "unmatched": 1, "total": 4}
    assert envelope["usage"]["calls"] > 0
    assert envelope["artifacts"]["mapping.csv"].endswith("mapping.csv")
    assert [w["code"] for w in envelope["warnings"]] == ["duplicate_targets"]
    assert "s1: matched" in captured.err


def test_an_invalid_job_is_a_json_usage_error(tmp_path, capsys):
    path = tmp_path / "job.yaml"
    path.write_text("name: x\npolicy: {acept_at: 1}\n", encoding="utf-8")
    assert main(["validate", "--job", str(path), "--json"]) == 2
    envelope = _envelope(capsys, 2)
    assert envelope["status"] == "error"
    codes = {e["code"] for e in envelope["errors"]}
    assert {"unknown_field", "missing_field"} <= codes


def test_a_missing_job_file_is_a_json_usage_error(tmp_path, capsys):
    assert main(["match", "--job", "nope.yaml", "--out", str(tmp_path), "--json"]) == 2
    envelope = _envelope(capsys, 2)
    assert envelope["errors"][0]["code"] == "job_not_found"


def test_a_bad_flag_still_prints_a_json_envelope(capsys):
    assert main(["inspect", "--json", "--bogus"]) == 2
    assert _envelope(capsys, 2)["errors"][0]["code"] == "usage"


def test_an_unexpected_exception_is_a_json_runtime_error(tmp_path, monkeypatch, capsys):
    from xwalk.config import JobSpec

    def boom(self):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(JobSpec, "build_llm", boom)
    assert main(["match", "--job", JOB, "--out", str(tmp_path / "r"), "--json"]) == 3
    envelope = _envelope(capsys, 3)
    assert envelope["errors"] == [
        {"code": "exception", "message": "RuntimeError: provider exploded"}
    ]


def test_a_fatal_abort_is_exit_three_with_the_error_in_json(tmp_path, monkeypatch, capsys):
    from xwalk.config import JobSpec
    from xwalk.llm.base import LLMFatalError
    from xwalk.llm.fake import FakeLLM

    def handler(request):
        raise LLMFatalError("invalid api key")

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: FakeLLM(handler=handler))
    assert main(["match", "--job", JOB, "--out", str(tmp_path / "r"), "--json"]) == 3
    envelope = _envelope(capsys, 3)
    assert envelope["run"]["run_state"] == "aborted"
    assert envelope["errors"][0]["code"] == "fatal_provider_failure"
    assert envelope["errors"][0]["source_id"]


def test_a_refused_run_directory_reports_the_changed_components(
    tmp_path, scripted, capsys, monkeypatch
):
    out = str(tmp_path / "run")
    main(["match", "--job", JOB, "--out", out])
    capsys.readouterr()
    from xwalk.config import JobSpec

    original = JobSpec.build_policy

    def stricter(self):
        return original(
            self.model_copy(update={"policy": self.policy.model_copy(update={"max_attempts": 2})})
        )

    monkeypatch.setattr(JobSpec, "build_policy", stricter)
    assert main(["match", "--job", JOB, "--out", out, "--json"]) == 3
    envelope = _envelope(capsys, 3)
    assert envelope["errors"][0]["code"] == "run_fingerprint_mismatch"
    assert "policy.max_attempts" in envelope["data"]["differences"]


def test_inspect_explain_export_and_validate_emit_envelopes(tmp_path, scripted, capsys):
    run = str(tmp_path / "run")
    main(["match", "--job", JOB, "--out", run])
    capsys.readouterr()

    assert main(["inspect", "--run", run, "--json"]) == 0
    assert _envelope(capsys, 0)["data"]["job"] == "tiny"

    assert main(["explain", "s3", "--run", run, "--json"]) == 0
    assert _envelope(capsys, 0)["data"]["decision"]["matched_id"] == "CHEBI:17992"

    out = str(tmp_path / "reviewed.csv")
    assert main(["export", "--run", run, "--view", "reviewed", "--out", out, "--json"]) == 0
    assert _envelope(capsys, 0)["artifacts"]["reviewed"] == out

    assert main(["validate", "--job", JOB, "--no-credentials", "--json"]) == 0
    assert _envelope(capsys, 0)["data"]["paid_calls"] == 0

    assert main(["doctor", "--job", JOB, "--no-credentials", "--json"]) == 0
    assert _envelope(capsys, 0)["operation"] == "validate"  # doctor is an alias


def test_init_and_index_emit_envelopes(tmp_path, capsys):
    assert main(["init", str(tmp_path / "qs"), "--json"]) == 0
    assert _envelope(capsys, 0)["artifacts"]["job"].endswith("job.yaml")
    assert main(["index", "--job", JOB, "--out", str(tmp_path / "idx"), "--json"]) == 0
    assert _envelope(capsys, 0)["data"]["retrievers"][0]["action"] == "built"
    assert main(["index", "--job", JOB, "--out", str(tmp_path / "idx"), "--json"]) == 0
    assert _envelope(capsys, 0)["data"]["retrievers"][0]["action"] == "opened"


def test_eval_emits_an_envelope(tmp_path, capsys):
    from tests.test_ceiling import attempt, cand, result
    from tests.test_cli import _seed_run

    run_dir = _seed_run(tmp_path, [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])])
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n", encoding="utf-8")
    assert main(["eval", "--run", str(run_dir), "--gold", str(gold), "--json"]) == 0
    assert _envelope(capsys, 0)["data"]["report"]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.run turns Ctrl-C into cancellation from 3.11"
)
def test_ctrl_c_exits_130_and_the_run_records_interrupted(tmp_path, monkeypatch, capsys):
    from xwalk.config import JobSpec
    from xwalk.llm.fake import FakeLLM

    inner = _scripted_llm()

    def handler(request):
        signal.raise_signal(signal.SIGINT)  # what a user's Ctrl-C delivers
        return inner._handler(request)

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: FakeLLM(handler=handler))
    out = tmp_path / "run"
    assert main(["match", "--job", JOB, "--out", str(out), "--json"]) == 130
    envelope = _envelope(capsys, 130)
    assert envelope["status"] == "interrupted"
    assert envelope["run"]["run_state"] == "interrupted"
    assert envelope["errors"][-1]["code"] == "interrupted"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_state"] == "interrupted"

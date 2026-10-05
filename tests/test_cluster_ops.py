"""`xwalk cluster`, the `kind: cluster` job and `ops.cluster`: envelope, exit codes,
run-directory identity, validation. The model is the scripted `Oracle`."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.cluster_helpers import CHOCOLATE, Oracle, concepts, sections
from tests.conftest import FIXTURES
from xwalk import ops
from xwalk.cli.main import main
from xwalk.cluster.job import ClusterJobSpec, load_cluster_job

JOB = FIXTURES / "cluster_tiny" / "job.yaml"
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


@pytest.fixture
def oracle(monkeypatch):
    judge = Oracle(concepts(CHOCOLATE))
    monkeypatch.setattr(ClusterJobSpec, "build_llm", lambda self: judge.llm)
    return judge


def _envelope(capsys) -> dict:
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert set(envelope) == KEYS and envelope["schema_version"] == 1
    return envelope


def test_cluster_json_envelope_and_exports(tmp_path, oracle, capsys):
    code = main(["cluster", "--job", str(JOB), "--out", str(tmp_path / "run"), "--json"])
    envelope = _envelope(capsys)
    assert code == envelope["exit_code"] == 0
    assert envelope["operation"] == "cluster" and envelope["status"] == "ok"
    assert envelope["run"]["run_state"] == "complete"
    assert envelope["counts"]["total"] == 8 and envelope["counts"]["clusters"] == 5
    assert envelope["usage"]["calls"] == len(oracle.llm.requests) > 0
    assert envelope["data"]["stop_reason"] == "converged"
    assert envelope["data"]["experimental"] is True
    assert [w["code"] for w in envelope["warnings"]] == ["experimental"]
    for name in ("members", "clusters", "unresolved", "decisions", "manifest", "store"):
        assert (tmp_path / "run" / envelope["artifacts"][name].split("/")[-1]).is_file()


def test_progress_goes_to_stderr_and_human_output_to_stdout(tmp_path, oracle, capsys):
    code = main(["cluster", "--job", str(JOB), "--out", str(tmp_path / "run")])
    captured = capsys.readouterr()
    assert code == 0
    assert "clustered 8 records into 5 clusters" in captured.out
    assert "c7: assigned" in captured.err


def test_needs_review_exits_1(tmp_path, monkeypatch, capsys):
    def unsure(name, request):
        if name == "verify-assignment" and "plain chocolate" in request.user:
            return json.dumps({"decision": "equivalent", "confidence_score": 0.6})
        return None

    judge = Oracle(concepts(CHOCOLATE), override=unsure)
    monkeypatch.setattr(ClusterJobSpec, "build_llm", lambda self: judge.llm)
    code = main(["cluster", "--job", str(JOB), "--out", str(tmp_path / "run"), "--json"])
    envelope = _envelope(capsys)
    assert code == 1 and envelope["status"] == "attention"
    assert envelope["counts"]["needs_review"] == 1
    assert "needs_review" in [w["code"] for w in envelope["warnings"]]


def test_failed_sources_exit_3(tmp_path, monkeypatch, capsys):
    from xwalk.llm.base import LLMRetryableError

    def down(name, request):
        if "quinoa" in sections(request.user).get("Record", ""):
            raise LLMRetryableError("503")
        return None

    judge = Oracle(concepts(CHOCOLATE), override=down)
    monkeypatch.setattr(ClusterJobSpec, "build_llm", lambda self: judge.llm)
    code = main(["cluster", "--job", str(JOB), "--out", str(tmp_path / "run"), "--json"])
    envelope = _envelope(capsys)
    assert code == 3 and envelope["run"]["run_state"] == "failed"
    assert envelope["errors"] == [
        {"code": "source_failed", "message": "provider_failure", "source_id": "c8"}
    ]


def test_call_limit_aborts_with_3_and_resume_completes(tmp_path, oracle, capsys):
    out = str(tmp_path / "run")
    code = main(["cluster", "--job", str(JOB), "--out", out, "--max-calls", "4", "--json"])
    envelope = _envelope(capsys)
    assert code == 3 and envelope["run"]["run_state"] == "aborted"
    assert envelope["errors"][0]["code"] == "call_limit_reached"
    assert envelope["usage"]["calls"] == 4 and envelope["usage"]["limit"] == 4
    assert envelope["counts"]["pending"] > 0
    code = main(["cluster", "--job", str(JOB), "--out", out, "--json"])
    envelope = _envelope(capsys)
    assert code == 0 and envelope["run"]["run_state"] == "complete"


def test_a_complete_run_repeats_with_zero_calls(tmp_path, oracle, capsys):
    out = str(tmp_path / "run")
    main(["cluster", "--job", str(JOB), "--out", out, "--json"])
    first = _envelope(capsys)
    members = (tmp_path / "run" / "members.csv").read_text(encoding="utf-8")
    calls = len(oracle.llm.requests)
    main(["cluster", "--job", str(JOB), "--out", out, "--json"])
    second = _envelope(capsys)
    assert len(oracle.llm.requests) == calls and second["usage"]["calls"] == 0
    assert second["counts"] == first["counts"]
    assert (tmp_path / "run" / "members.csv").read_text(encoding="utf-8") == members


def test_a_changed_job_is_refused_and_nothing_is_overwritten(tmp_path, oracle, capsys):
    job_dir = tmp_path / "job"
    shutil.copytree(JOB.parent, job_dir)
    out = str(tmp_path / "run")
    main(["cluster", "--job", str(job_dir / "job.yaml"), "--out", out, "--json"])
    _envelope(capsys)
    manifest = (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    text = (job_dir / "job.yaml").read_text(encoding="utf-8")
    (job_dir / "job.yaml").write_text(text.replace("shown_limit: 8", "shown_limit: 4"))
    code = main(["cluster", "--job", str(job_dir / "job.yaml"), "--out", out, "--json"])
    envelope = _envelope(capsys)
    assert code == 3 and envelope["errors"][0]["code"] == "run_fingerprint_mismatch"
    assert "pool" in envelope["errors"][0]["message"]
    assert (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8") == manifest


def test_an_edited_source_collection_is_another_run(tmp_path, oracle, capsys):
    job_dir = tmp_path / "job"
    shutil.copytree(JOB.parent, job_dir)
    out = str(tmp_path / "run")
    main(["cluster", "--job", str(job_dir / "job.yaml"), "--out", out, "--json"])
    _envelope(capsys)
    with (job_dir / "labels.csv").open("a", encoding="utf-8") as handle:
        handle.write("c9,cocoa\n")
    code = main(["cluster", "--job", str(job_dir / "job.yaml"), "--out", out, "--json"])
    envelope = _envelope(capsys)
    assert code == 3 and "snapshot" in envelope["errors"][0]["message"]


def test_a_foreign_directory_is_refused(tmp_path, oracle, capsys):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "notes.txt").write_text("mine")
    code = main(["cluster", "--job", str(JOB), "--out", str(tmp_path / "run"), "--json"])
    assert code == 3 and _envelope(capsys)["errors"][0]["code"] == "out_not_a_run"


def test_invalid_cluster_job_exits_2_with_a_suggestion(tmp_path, capsys):
    bad = tmp_path / "job.yaml"
    text = JOB.read_text(encoding="utf-8").replace("max_refine_iterations", "max_refine_iteration")
    bad.write_text(text)
    (tmp_path / "labels.csv").write_text((JOB.parent / "labels.csv").read_text())
    code = main(["cluster", "--job", str(bad), "--json"] + ["--out", str(tmp_path / "run")])
    envelope = _envelope(capsys)
    assert code == 2
    assert "did you mean 'max_refine_iterations'" in envelope["errors"][0]["message"]
    assert envelope["errors"][0]["message"].startswith("policy.max_refine_iteration")


def test_out_of_range_threshold_is_rejected(tmp_path):
    from xwalk.cluster.job import parse_cluster_job
    from xwalk.config import JobValidationError

    data = {
        "kind": "cluster",
        "name": "x",
        "source": {"kind": "csv", "path": "a.csv"},
        "templates": {"query": "{{ label }}"},
        "llm": {"base_url": "http://x", "model": "m"},
        "policy": {"merge_accept_at": 0.5, "merge_review_floor": 0.7},
    }
    with pytest.raises(JobValidationError) as info:
        parse_cluster_job(data)
    assert "merge_review_floor" in str(info.value)


def test_validate_understands_cluster_jobs_and_calls_nothing(monkeypatch, capsys):
    monkeypatch.setenv("XWALK_TEST_CLUSTER_KEY", "sk-test")
    code = main(["validate", "--job", str(JOB), "--json"])
    envelope = _envelope(capsys)
    assert code == 0
    assert envelope["data"]["kind"] == "cluster" and envelope["data"]["paid_calls"] == 0
    assert envelope["counts"] == {"sources": 8}


def test_validate_reports_a_missing_credential_and_duplicate_ids(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("XWALK_TEST_CLUSTER_KEY", raising=False)
    shutil.copytree(JOB.parent, tmp_path / "job")
    with (tmp_path / "job" / "labels.csv").open("a", encoding="utf-8") as handle:
        handle.write("c1,duplicate\n")
    code = main(["validate", "--job", str(tmp_path / "job" / "job.yaml"), "--json"])
    codes = {e["code"] for e in _envelope(capsys)["errors"]}
    assert code == 2 and codes == {"credential_missing"}
    monkeypatch.setenv("XWALK_TEST_CLUSTER_KEY", "sk-test")
    code = main(["validate", "--job", str(tmp_path / "job" / "job.yaml"), "--json"])
    assert code == 2 and {e["code"] for e in _envelope(capsys)["errors"]} == {"duplicate_id"}


def test_match_refuses_a_cluster_job_with_a_clear_message(tmp_path, capsys):
    code = main(["match", "--job", str(JOB), "--out", str(tmp_path / "run"), "--json"])
    envelope = _envelope(capsys)
    assert code == 2 and envelope["errors"][0]["code"] == "wrong_job_kind"
    assert "xwalk cluster" in envelope["errors"][0]["message"]


def test_missing_credential_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("XWALK_TEST_CLUSTER_KEY", raising=False)
    code = main(["cluster", "--job", str(JOB), "--out", str(tmp_path / "run"), "--json"])
    assert code == 2 and _envelope(capsys)["errors"][0]["code"] == "credential_missing"
    assert not (tmp_path / "run").exists()


def test_python_route_takes_a_client_and_a_spec(tmp_path):
    spec = load_cluster_job(JOB)
    judge = Oracle(concepts(CHOCOLATE))
    result = ops.cluster(spec, tmp_path / "run", llm=judge.llm, max_calls=100)
    assert result.ok and result.exit_code == 0
    assert result.envelope()["counts"]["assigned"] == 5
    assert result.run is not None and result.run["run_state"] == "complete"

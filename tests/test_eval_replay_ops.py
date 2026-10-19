import json
from dataclasses import asdict

import pytest

from tests.test_ceiling import result
from tests.test_cli import _seed_run
from tests.test_ops import _job_file
from tests.test_replay import scored
from xwalk import ops
from xwalk.cli.main import main
from xwalk.policy import MatchPolicy


def seed(tmp_path, *, metadata=True):
    run = _seed_run(tmp_path, [result("s1", "T1", [scored(score=0.7)])])
    if metadata:
        manifest = {
            "run_fingerprint": "fp1",
            "fingerprint_components": {"policy": {**asdict(MatchPolicy()), "status_semantics": 2}},
        }
        (run / "manifest.json").write_text(json.dumps(manifest))
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n")
    return run, gold


def test_eval_replay_uses_ops_and_one_envelope_without_building_clients(
    tmp_path, capsys, monkeypatch
):
    from xwalk.config import JobSpec

    run, gold = seed(tmp_path)
    job = _job_file(tmp_path, lambda d: d.update(policy={"accept_at": 0.8}))

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation built acquisition machinery")

    for name in (
        "build_llm",
        "build_decider",
        "build_retrievers",
        "build_matcher",
        "build_decision_matcher",
    ):
        monkeypatch.setattr(JobSpec, name, forbidden)
    code = main(
        ["eval", "--run", str(run), "--gold", str(gold), "--replay-policy", str(job), "--json"]
    )
    envelope = json.loads(capsys.readouterr().out)
    assert code == envelope["exit_code"] == 0
    assert envelope["operation"] == "eval"
    replay = envelope["data"]["replay"]
    assert replay["status_transitions"] == {"matched": {"needs_review": 1}}
    assert replay["conditional_on_recorded_attempts"]
    assert envelope["data"]["report"]["threshold_context"]["conditional_on_recorded_attempts"]
    assert any(w["code"] == "acquisition_fixed" for w in envelope["warnings"])
    assert hasattr(ops, "evaluate_run")


def test_eval_refuses_changed_verification_with_a_usage_envelope(tmp_path, capsys):
    run, gold = seed(tmp_path)
    job = _job_file(tmp_path, lambda d: d.update(policy={"verify_band": None}))
    assert (
        main(
            ["eval", "--run", str(run), "--gold", str(gold), "--replay-policy", str(job), "--json"]
        )
        == 2
    )
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["errors"][0]["code"] == "replay_unavailable"
    assert "unrecorded calls" in envelope["errors"][0]["message"]


def test_eval_refuses_replay_without_recorded_policy(tmp_path):
    run, gold = seed(tmp_path, metadata=False)
    job = _job_file(tmp_path)
    with pytest.raises(ops.OpError, match="recorded policy"):
        ops.evaluate_run(run, gold, replay_policy=job)


def test_eval_job_supplies_catalogue_ids_without_loading_a_retriever(tmp_path):
    run, gold = seed(tmp_path)
    gold.write_text("source_id,gold_ids\ns1,not-in-catalogue\n")
    job = _job_file(tmp_path)
    report = ops.evaluate_run(run, gold, job=job).data["report"]
    assert report["ceiling"]["buckets"]["gold_unreachable"] == 1
    assert report["ceiling"]["evaluable"] == 0


def test_eval_refuses_a_changed_retriever_when_original_job_is_supplied(tmp_path):
    run, gold = seed(tmp_path)
    original = _job_file(tmp_path, name="original.yaml")
    proposed = _job_file(
        tmp_path, lambda d: d["retrievers"][0].update(limit=99), name="proposed.yaml"
    )
    with pytest.raises(ops.OpError, match="unrecorded calls"):
        ops.evaluate_run(run, gold, job=original, replay_policy=proposed)


def test_cli_eval_is_only_an_ops_adapter(tmp_path, monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return ops.OpResult(operation="eval")

    monkeypatch.setattr(ops, "evaluate_run", fake)
    assert main(["eval", "--run", "unused", "--gold", "unused"]) == 0
    assert len(calls) == 1


def test_replay_artifacts_include_diagnostics_and_leave_ledger_unchanged(tmp_path):
    run, gold = seed(tmp_path)
    job = _job_file(tmp_path, lambda d: d.update(policy={"accept_at": 0.8}))
    before = (run / "ledger.sqlite").read_bytes()
    out = tmp_path / "analysis"
    envelope = ops.evaluate_run(run, gold, replay_policy=job, out=out)
    saved = json.loads(out.with_suffix(".json").read_text())
    assert saved["replay"] == envelope.data["replay"]
    text = out.with_suffix(".txt").read_text()
    assert "scorer_below" in text and "broken_by_retry" in text
    assert "descriptive" in text and "Conditional" in text
    assert (run / "ledger.sqlite").read_bytes() == before


def test_nondefault_recorded_policy_controls_the_report_curve(tmp_path):
    from dataclasses import replace

    from xwalk.decide.policy import DecisionPolicy
    from xwalk.records import DecisionReason, MatchStatus

    a = replace(
        scored(score=0.9),
        signals={
            "screen_best": 0.9,
            "screen_chosen": 0.9,
            "p_choice": 0.9,
            "rubric": 2.0,
            "rubric_levels": 3.0,
            "prop_form": 0.8,
        },
        reason=DecisionReason.BELOW_ACCEPT_THRESHOLD,
    )
    row = result("s1", "T1", [a], MatchStatus.NEEDS_REVIEW)
    run = _seed_run(tmp_path, [row])
    manifest = {
        "run_fingerprint": "fp1",
        "fingerprint_components": {
            "path": "decider",
            "policy": asdict(DecisionPolicy(property_floor=1.0)),
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n")
    report = ops.evaluate_run(run, gold).data["report"]
    assert report["thresholds"]
    assert all(p["accepted"] == 0 for p in report["thresholds"])


def test_ordinary_eval_warns_about_historical_reduction_in_json_and_text(tmp_path):
    from dataclasses import replace

    from xwalk.records import MatchStatus

    policy = MatchPolicy()
    row = result(
        "s1",
        "T2",
        [scored("T2", 0.9, verifier_decision="disagree"), replace(scored("T1", 0.8), index=1)],
        MatchStatus.NEEDS_REVIEW,
    )
    run = _seed_run(tmp_path, [row])
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_fingerprint": "fp1",
                "fingerprint_components": {"policy": {**asdict(policy), "status_semantics": 1}},
            }
        )
    )
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n")
    out = tmp_path / "analysis"
    envelope = ops.evaluate_run(run, gold, out=out)
    report = envelope.data["report"]
    assert report["metrics"]["automatic_coverage"] == 0.0
    unchanged = next(p for p in report["thresholds"] if p["threshold"] == policy.accept_at)
    assert unchanged["coverage"] == 1.0
    context = report["threshold_context"]
    assert context["recorded_semantics_version"] == 1
    assert context["status_semantics_version"] == 2
    warning = "historical or unknown reducer version"
    assert warning in " ".join(context["warnings"])
    assert any(warning in w.message for w in envelope.warnings)
    assert warning in "\n".join(envelope.lines)
    assert json.loads(out.with_suffix(".json").read_text())["threshold_context"] == context
    assert warning in out.with_suffix(".txt").read_text()

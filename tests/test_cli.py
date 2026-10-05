import asyncio
import csv
import json

import pytest

from tests.conftest import FIXTURES
from xwalk.cli.main import main

JOB = str(FIXTURES / "job_tiny.yaml")


def _scripted_llm(*, score=0.95):
    """The job's LLM, replaced by a script. `xwalk match` is otherwise untestable
    offline, and the subcommands that build a matcher are where the CLI stops being a
    pass-through and starts wiring things together."""
    from tests.test_matcher import score_reply, select_reply
    from xwalk.llm.fake import FakeLLM

    def handler(request):
        if "## Candidates" not in request.user:
            return score_reply(score)
        return select_reply("C01") if "[C01]" in request.user else select_reply(None)

    return FakeLLM(handler=handler)


@pytest.fixture
def scripted_job(monkeypatch):
    """Every `build_llm` in the process answers from the script, so no key is needed."""
    from xwalk.config import JobSpec

    llm = _scripted_llm()
    monkeypatch.setattr(JobSpec, "build_llm", lambda self: llm)
    return llm


def _seed_run(tmp_path, results):
    """A run directory with a ledger and the manifest the CLI reads the fingerprint from."""
    from xwalk.ledger import Ledger

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    for result in results:
        asyncio.run(ledger.put_result(result))
    ledger.put_manifest("fp1", {"run_fingerprint": "fp1"})
    ledger.close()
    (run_dir / "manifest.json").write_text(json.dumps({"run_fingerprint": "fp1"}), encoding="utf-8")
    return run_dir


def test_no_arguments_prints_usage_and_returns_two(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_version_prints_the_version(capsys):
    from xwalk import __version__

    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_an_unknown_subcommand_returns_two_rather_than_raising(capsys):
    """argparse exits the process on an invalid choice. A library entry point that
    raises SystemExit cannot be called from anything but a shell."""
    assert main(["telepathy"]) == 2


def test_help_returns_zero_rather_than_raising(capsys):
    assert main(["--help"]) == 0


def test_index_builds_the_retriever_indexes(tmp_path):
    assert main(["index", "--job", JOB, "--out", str(tmp_path / "idx")]) == 0
    assert (tmp_path / "idx" / "bm25" / "xwalk_meta.json").exists()


def test_index_reports_the_document_count(tmp_path, capsys):
    main(["index", "--job", JOB, "--out", str(tmp_path / "idx")])
    assert "indexed 5 target records" in capsys.readouterr().out


def test_match_without_an_api_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["match", "--job", JOB, "--out", str(tmp_path / "run")]) == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_match_with_a_nonexistent_job_fails_cleanly(tmp_path, capsys):
    assert main(["match", "--job", "nope.yaml", "--out", str(tmp_path)]) == 3
    assert "nope.yaml" in capsys.readouterr().err


def test_eval_prints_a_report(tmp_path, capsys):
    from tests.test_ceiling import attempt, cand, result

    run_dir = _seed_run(tmp_path, [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])])
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n", encoding="utf-8")

    assert main(["eval", "--run", str(run_dir), "--gold", str(gold)]) == 0
    assert "Where to spend effort" in capsys.readouterr().out


def test_eval_writes_a_report_file_when_asked(tmp_path):
    from tests.test_ceiling import attempt, cand, result

    run_dir = _seed_run(tmp_path, [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])])
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n", encoding="utf-8")

    assert (
        main(["eval", "--run", str(run_dir), "--gold", str(gold), "--out", str(tmp_path / "e")])
        == 0
    )
    assert (tmp_path / "e.json").exists() and (tmp_path / "e.txt").exists()


def test_compare_rejects_a_run_without_a_label(tmp_path, capsys):
    from tests.test_ceiling import attempt, cand, result

    run_dir = _seed_run(tmp_path, [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])])
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n", encoding="utf-8")
    assert main(["compare", "--gold", str(gold), "--run", str(run_dir)]) == 2


def test_compare_renders_a_table(tmp_path, capsys):
    from tests.test_ceiling import attempt, cand, result

    run_dir = _seed_run(tmp_path, [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])])
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n", encoding="utf-8")
    assert main(["compare", "--gold", str(gold), "--run", f"mine={run_dir}"]) == 0
    assert "mine" in capsys.readouterr().out


def test_review_export_writes_a_csv(tmp_path):
    from tests.test_serde import sample_result
    from xwalk.records import DecisionReason, MatchResult, MatchStatus

    base = sample_result()
    row = MatchResult(
        **{
            **base.__dict__,
            "status": MatchStatus.NEEDS_REVIEW,
            "reason": DecisionReason.BELOW_ACCEPT_THRESHOLD,
        }
    )
    run_dir = _seed_run(tmp_path, [row])

    out = tmp_path / "review.csv"
    assert main(["review", "export", "--run", str(run_dir), "--out", str(out)]) == 0
    assert "result_key" in out.read_text(encoding="utf-8")


def test_every_subcommand_appears_in_the_help(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for name in ("index", "match", "eval", "compare", "ablate", "prompts", "review"):
        assert name in out


def test_help_for_a_subcommand_lists_its_flags(capsys):
    assert main(["match", "--help"]) == 0
    assert "--resume" in capsys.readouterr().out


def test_prompts_without_a_subcommand_is_a_usage_error(capsys):
    assert main(["prompts"]) == 2


def test_review_without_a_subcommand_is_a_usage_error(capsys):
    assert main(["review"]) == 2


# --- match ------------------------------------------------------------------------


def test_match_writes_a_mapping_for_every_source_record(tmp_path, scripted_job):
    out = tmp_path / "run"
    assert main(["match", "--job", JOB, "--out", str(out)]) == 0
    rows = list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))
    assert [r["source_id"] for r in rows] == ["s1", "s2", "s3", "s4"]


def test_match_writes_the_whole_run_directory(tmp_path, scripted_job):
    out = tmp_path / "run"
    main(["match", "--job", JOB, "--out", str(out)])
    for name in ("mapping.csv", "results.jsonl", "manifest.json", "ledger.sqlite"):
        assert (out / name).exists(), name


def test_match_resolves_the_mention_through_the_real_retriever(tmp_path, scripted_job):
    """Not a smoke test: the CLI must wire templates, index and store together well
    enough that a synonym in the target file actually wins."""
    out = tmp_path / "run"
    main(["match", "--job", JOB, "--out", str(out)])
    rows = {r["source_id"]: r for r in csv.DictReader((out / "mapping.csv").open(encoding="utf-8"))}
    assert rows["s1"]["matched_id"] == "CHEBI:17234"  # glucose, by label
    assert rows["s2"]["matched_id"] == "CHEBI:17234"  # dextrose, by synonym
    assert rows["s3"]["matched_id"] == "CHEBI:17992"  # table sugar, by synonym


def test_match_defaults_the_index_directory_under_the_run(tmp_path, scripted_job):
    out = tmp_path / "run"
    main(["match", "--job", JOB, "--out", str(out)])
    assert (out / "index" / "bm25" / "xwalk_meta.json").exists()


def test_match_honours_an_explicit_index_directory(tmp_path, scripted_job):
    out, index = tmp_path / "run", tmp_path / "shared-index"
    main(["match", "--job", JOB, "--out", str(out), "--index", str(index)])
    assert (index / "bm25" / "xwalk_meta.json").exists()
    assert not (out / "index").exists()


def test_match_limit_processes_only_the_first_n_records(tmp_path, scripted_job):
    """The snapshot is still the whole source: the rest is pending, the run partial."""
    out = tmp_path / "run"
    assert main(["match", "--job", JOB, "--out", str(out), "--limit", "2"]) == 1
    rows = list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))
    assert [(r["source_id"], r["status"] == "pending") for r in rows] == [
        ("s1", False),
        ("s2", False),
        ("s3", True),
        ("s4", True),
    ]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_state"] == (
        "partial"
    )


def test_match_returns_three_when_a_fatal_provider_error_aborts_the_run(
    tmp_path, monkeypatch, capsys
):
    from xwalk.config import JobSpec
    from xwalk.llm.base import LLMFatalError
    from xwalk.llm.fake import FakeLLM

    def handler(request):
        raise LLMFatalError("invalid api key")

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: FakeLLM(handler=handler))
    out = tmp_path / "run"
    assert main(["match", "--job", JOB, "--out", str(out)]) == 3
    assert "fatal_provider_failure" in capsys.readouterr().err
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_state"] == (
        "aborted"
    )


def test_match_prints_a_status_breakdown(tmp_path, scripted_job, capsys):
    main(["match", "--job", JOB, "--out", str(tmp_path / "run")])
    out = capsys.readouterr().out
    assert "matched 4 records" in out
    assert "matched" in out and "unmatched" in out


def test_match_returns_one_and_points_at_review_when_the_bucket_is_not_empty(
    tmp_path, monkeypatch, capsys
):
    from xwalk.config import JobSpec

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: _scripted_llm(score=0.5))
    out = tmp_path / "run"
    assert main(["match", "--job", JOB, "--out", str(out)]) == 1
    assert "need review" in capsys.readouterr().out


def test_match_reports_duplicate_targets(tmp_path, scripted_job, capsys):
    """s1 and s2 both land on glucose. Reporting a many-to-one is not resolving it."""
    main(["match", "--job", JOB, "--out", str(tmp_path / "run")])
    assert "duplicate targets" in capsys.readouterr().out


def test_match_resumes_instead_of_re_running_completed_records(tmp_path, monkeypatch):
    from xwalk.config import JobSpec

    out = tmp_path / "run"
    first = _scripted_llm()
    monkeypatch.setattr(JobSpec, "build_llm", lambda self: first)
    main(["match", "--job", JOB, "--out", str(out)])

    second = _scripted_llm()
    monkeypatch.setattr(JobSpec, "build_llm", lambda self: second)
    assert main(["match", "--job", JOB, "--out", str(out)]) == 0
    assert second.requests == []


def test_match_no_resume_re_runs_everything(tmp_path, monkeypatch):
    from xwalk.config import JobSpec

    out = tmp_path / "run"
    monkeypatch.setattr(JobSpec, "build_llm", lambda self: _scripted_llm())
    main(["match", "--job", JOB, "--out", str(out)])

    second = _scripted_llm()
    monkeypatch.setattr(JobSpec, "build_llm", lambda self: second)
    assert main(["match", "--job", JOB, "--out", str(out), "--no-resume"]) == 0
    assert second.requests != []


# --- ablate -----------------------------------------------------------------------


def _gold(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "source_id,gold_ids\ns1,CHEBI:17234\ns2,CHEBI:17234\ns3,CHEBI:17992\ns4,\n",
        encoding="utf-8",
    )
    return path


def test_ablate_writes_a_report_and_prints_a_delta_per_variant(tmp_path, scripted_job, capsys):
    out = tmp_path / "abl"
    assert main(["ablate", "--job", JOB, "--gold", str(_gold(tmp_path)), "--out", str(out)]) == 0
    report = json.loads((out / "ablation.json").read_text(encoding="utf-8"))
    assert report["rows"]
    assert "delta" in capsys.readouterr().out


def test_ablate_names_the_retrievers_it_actually_built(tmp_path, scripted_job, monkeypatch):
    """A `kind: dense` retriever with no explicit `name` is built as `dense:<model>`,
    but the job spec calls it `dense`. Naming the baseline from the spec means the
    factory's `r.name in config.retriever_names` filter never matches it, so it is
    silently absent from the baseline and from every variant -- the report then
    describes a system nobody configured."""
    from tests.test_matcher import ScriptedRetriever
    from xwalk.config import JobSpec

    built_as = "dense:some-model"
    real = JobSpec.build_retrievers

    def build(self, records, templates, index_dir):
        extra = ScriptedRetriever({"glucose": ["CHEBI:17234"]}, name=built_as)
        return [*real(self, records, templates, index_dir), extra]

    monkeypatch.setattr(JobSpec, "build_retrievers", build)
    out = tmp_path / "abl"
    assert main(["ablate", "--job", JOB, "--gold", str(_gold(tmp_path)), "--out", str(out)]) == 0
    names = {row["name"] for row in json.loads((out / "ablation.json").read_text())["rows"]}
    assert f"no_{built_as}" in names


def test_ablate_completes_on_a_single_retriever_job(tmp_path, scripted_job):
    """The CLI's factory hands `build_matcher` whatever retrievers a variant leaves, and
    `Matcher` rejects an empty list -- so a `no_bm25` variant here would abort the run
    before any variant reported, not merely score badly."""
    out = tmp_path / "abl"
    assert main(["ablate", "--job", JOB, "--gold", str(_gold(tmp_path)), "--out", str(out)]) == 0
    names = {row["name"] for row in json.loads((out / "ablation.json").read_text())["rows"]}
    assert names == {"baseline", "no_verifier", "no_retries", "half_budget"}


# --- prompts ----------------------------------------------------------------------

DRAFTED = {
    "entity_noun": "sugar mention",
    "target_noun": "ChEBI term",
    "domain_brief": "carbohydrate nomenclature",
    "rubric": [
        {"score": 1.0, "name": "Certain", "when": "exact label or synonym", "example": "a -> a"},
        {"score": 0.4, "name": "Weak", "when": "same broad class", "example": "b -> c"},
    ],
    "hard_rules": ["An anomer is distinct from its parent"],
    "disambiguation_steps": "Read the context first.",
}


def test_prompts_draft_writes_a_loadable_slots_file(tmp_path, monkeypatch):
    from xwalk.config import JobSpec
    from xwalk.llm.fake import FakeLLM
    from xwalk.prompts.contract import load_slots

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: FakeLLM([json.dumps(DRAFTED)]))
    out = tmp_path / "slots.yaml"
    assert main(["prompts", "draft", "--job", JOB, "--describe", "sugars", "--out", str(out)]) == 0
    assert load_slots(out).entity_noun == "sugar mention"


def test_prompts_draft_shows_the_diff_against_the_jobs_existing_slots(
    tmp_path, monkeypatch, capsys
):
    from xwalk.config import JobSpec
    from xwalk.llm.fake import FakeLLM

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: FakeLLM([json.dumps(DRAFTED)]))
    main(
        [
            "prompts",
            "draft",
            "--job",
            JOB,
            "--describe",
            "sugars",
            "--out",
            str(tmp_path / "slots.yaml"),
        ]
    )
    assert "entity_noun" in capsys.readouterr().out


def test_prompts_rejects_an_unknown_role(tmp_path, scripted_job, capsys):
    code = main(
        [
            "prompts",
            "optimize",
            "--job",
            JOB,
            "--gold",
            str(_gold(tmp_path)),
            "--out",
            str(tmp_path / "opt"),
            "--role",
            "telepath",
        ]
    )
    assert code == 3
    assert "telepath" in capsys.readouterr().err


def _job_with_enough_sources(tmp_path, n=40):
    """The optimiser splits labelled records three ways. `sources_tiny.csv` has four,
    which hashes to an empty prompt-train partition and stops before round one."""
    import yaml

    mentions = ["glucose", "dextrose", "table sugar", "milk sugar"]
    ids = ["CHEBI:17234", "CHEBI:17234", "CHEBI:17992", "CHEBI:17716"]
    sources = tmp_path / "sources.csv"
    sources.write_text(
        "mention_id,mention,context_left,context_right\n"
        + "".join(f"s{i:03d},{mentions[i % 4]},a sample of ,was tested\n" for i in range(n)),
        encoding="utf-8",
    )
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_id,gold_ids\n" + "".join(f"s{i:03d},{ids[i % 4]}\n" for i in range(n)),
        encoding="utf-8",
    )

    spec = yaml.safe_load((FIXTURES / "job_tiny.yaml").read_text(encoding="utf-8"))
    spec["target"]["path"] = str(FIXTURES / "targets_tiny.csv")
    spec["source"]["path"] = str(sources)
    spec["prompts"]["slots"] = str(FIXTURES.parent.parent / "examples/chemistry/slots.yaml")
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(job), str(gold)


def test_prompts_optimize_runs_a_round_and_reports_why_it_stopped(tmp_path, monkeypatch, capsys):
    """Covers the CLI's matcher factory. A factory that dropped its `prompts` argument
    would re-run the original slots every round and the optimiser would measure
    nothing -- and every assertion about scores would still pass."""
    from tests.test_matcher import score_reply, select_reply
    from xwalk.config import JobSpec
    from xwalk.llm.fake import FakeLLM
    from xwalk.prompts.contract import PromptSet, load_slots

    def handler(request):
        if "## Candidates" in request.user:
            # Deliberately the runner-up, so prompt-train has misjudged cases to mine.
            # With no failures the optimiser stops before it ever calls the model.
            return select_reply("C02") if "[C02]" in request.user else select_reply(None)
        if "## Rubric" in request.user or "## Proposed match" in request.user:
            return score_reply(0.95)
        return json.dumps({**DRAFTED, "domain_brief": "revised brief"})

    seen_briefs: set[str] = set()
    real_build_matcher = JobSpec.build_matcher

    def build_matcher(self, **kwargs):
        prompts = kwargs.get("prompts")
        seen_briefs.add(prompts.slots.domain_brief if prompts else "(job's own slots)")
        return real_build_matcher(self, **kwargs)

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: FakeLLM(handler=handler))
    monkeypatch.setattr(JobSpec, "build_matcher", build_matcher)
    job, gold = _job_with_enough_sources(tmp_path)
    out = tmp_path / "opt"
    code = main(
        [
            "prompts",
            "optimize",
            "--job",
            job,
            "--gold",
            gold,
            "--out",
            str(out),
            "--rounds",
            "1",
        ]
    )
    assert code == 0
    assert "stopped:" in capsys.readouterr().out
    assert PromptSet.from_slots(load_slots(out / "best" / "slots.yaml"))
    assert "revised brief" in seen_briefs, "the candidate slots never reached a matcher"


# --- review apply -----------------------------------------------------------------


def _run_then_export(tmp_path, monkeypatch, *, score):
    from xwalk.config import JobSpec

    monkeypatch.setattr(JobSpec, "build_llm", lambda self: _scripted_llm(score=score))
    out = tmp_path / "run"
    main(["match", "--job", JOB, "--out", str(out)])
    review = tmp_path / "review.csv"
    main(["review", "export", "--run", str(out), "--out", str(review)])
    return out, review


def _decide(review_csv, decision, **overrides):
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    for row in rows:
        row.update(decision=decision, reviewer="jan", reviewed_at="2026-07-28T09:00:00Z")
        row.update(overrides)
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def test_review_apply_records_the_decisions(tmp_path, monkeypatch, capsys):
    out, review = _run_then_export(tmp_path, monkeypatch, score=0.5)
    count = _decide(review, "accept")
    assert (
        main(["review", "apply", "--run", str(out), "--reviewed", str(review), "--job", JOB]) == 0
    )
    assert f"applied {count} decisions" in capsys.readouterr().out


def test_review_apply_changes_the_adjudicated_status(tmp_path, monkeypatch):
    """The overlay is what `--use-review` exports read, so applying must be visible
    there and must leave the model's own answer intact beside it."""
    from xwalk.ledger import Ledger
    from xwalk.review import adjudicated

    out, review = _run_then_export(tmp_path, monkeypatch, score=0.5)
    _decide(review, "accept")
    main(["review", "apply", "--run", str(out), "--reviewed", str(review), "--job", JOB])

    ledger = Ledger.open(out / "ledger.sqlite")
    try:
        fp = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_fingerprint"]
        rows = [r for r in adjudicated(ledger, fp) if r.reviewer == "jan"]
    finally:
        ledger.close()
    assert rows
    assert all(r.final_status.value == "matched" for r in rows)
    assert all(r.model_status.value == "needs_review" for r in rows)


def test_review_apply_refuses_a_decision_made_against_a_stale_snapshot(
    tmp_path, monkeypatch, capsys
):
    """All-or-nothing: one stale row fails the whole file, because a half-applied
    review is worse than an unapplied one."""
    out, review = _run_then_export(tmp_path, monkeypatch, score=0.5)
    _decide(review, "accept", source_hash="not-the-hash-that-was-reviewed")
    code = main(["review", "apply", "--run", str(out), "--reviewed", str(review), "--job", JOB])
    assert code == 3
    assert "source record changed" in capsys.readouterr().err


def test_review_apply_rejects_replace_without_a_correction(tmp_path, monkeypatch, capsys):
    out, review = _run_then_export(tmp_path, monkeypatch, score=0.5)
    _decide(review, "replace")
    assert (
        main(["review", "apply", "--run", str(out), "--reviewed", str(review), "--job", JOB]) == 3
    )
    assert "corrected_target_id" in capsys.readouterr().err


def test_review_apply_replaces_the_target_when_corrected(tmp_path, monkeypatch):
    from xwalk.ledger import Ledger
    from xwalk.review import adjudicated

    out, review = _run_then_export(tmp_path, monkeypatch, score=0.5)
    _decide(review, "replace", corrected_target_id="CHEBI:15903")
    main(["review", "apply", "--run", str(out), "--reviewed", str(review), "--job", JOB])

    ledger = Ledger.open(out / "ledger.sqlite")
    try:
        fp = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["run_fingerprint"]
        rows = [r for r in adjudicated(ledger, fp) if r.reviewer == "jan"]
    finally:
        ledger.close()
    assert rows and all(r.final_target_id == "CHEBI:15903" for r in rows)


# --- error handling ---------------------------------------------------------------


def test_an_unexpected_error_is_reported_without_a_traceback(tmp_path, monkeypatch, capsys):
    from xwalk.config import JobSpec

    def boom(self):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(JobSpec, "build_llm", boom)
    assert main(["match", "--job", JOB, "--out", str(tmp_path / "run")]) == 3
    err = capsys.readouterr().err
    assert "RuntimeError: provider exploded" in err
    assert "Traceback" not in err

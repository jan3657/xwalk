import asyncio
import json

from tests.conftest import FIXTURES
from xwalk.cli.main import main

JOB = str(FIXTURES / "job_tiny.yaml")


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

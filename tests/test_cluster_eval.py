"""`xwalk cluster eval` and `ops.cluster_eval`: scores a finished clustering run against a
gold partition from its exports and store, with no model or retriever call and no write.
Also: matching-run commands pointed at a clustering run say so and name the cluster
commands. The judge is the scripted `Oracle`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.cluster_helpers import (
    CHOCOLATE,
    Oracle,
    cluster,
    concepts,
    member_rows,
    partition,
    records,
    settings,
)
from xwalk import ops
from xwalk.cli.main import main
from xwalk.cluster.pool import PoolIndex
from xwalk.llm.fake import FakeLLM

ENVELOPE_KEYS = {
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


def _gold(path: Path, table: dict[str, tuple[str, str]], **extra: str) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "cluster_id"])
        for rid, (_, concept) in table.items():
            writer.writerow([rid, concept])
        for rid, concept in extra.items():
            writer.writerow([rid, concept])
    return path


def _files(run: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run.iterdir())}


@pytest.fixture
async def chocolate_run(tmp_path) -> Path:
    run = tmp_path / "run"
    await cluster(records(CHOCOLATE), run, Oracle(concepts(CHOCOLATE)))
    return run


def test_cli_scores_a_run_against_gold(tmp_path, chocolate_run, capsys):
    gold = _gold(tmp_path / "gold.csv", CHOCOLATE)
    code = main(["cluster", "eval", "--run", str(chocolate_run), "--gold", str(gold), "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert set(envelope) == ENVELOPE_KEYS and envelope["schema_version"] == 1
    assert code == envelope["exit_code"] == 0
    assert envelope["operation"] == "cluster eval" and envelope["status"] == "ok"
    assert envelope["run"]["run_state"] == "complete"
    data = envelope["data"]
    for convention in ("as_singletons", "accepted_only"):
        scores = data["scores"][convention]
        assert scores["pairwise_precision"] == scores["pairwise_recall"] == 1.0
        assert scores["pairwise_f1"] == scores["bcubed_f1"] == 1.0
    assert data["scores"]["outcome_counts"]["assigned"] == 5
    assert data["scores"]["outcome_counts"]["singleton"] == 3
    assert envelope["counts"]["gold_items"] == 8 and envelope["counts"]["assigned"] == 5
    # dark chocolate (3 pairs) and myocardial infarction (1 pair), each seen in a prompt
    exposure = data["exposure"]
    assert exposure["gold_pairs"] == 4 and exposure["gold_pairs_co_exposed"] == 4
    assert "not an upper bound" in exposure["note"]
    assert envelope["usage"]["calls"] > 0 and envelope["usage"]["paid_calls_made_by_eval"] == 0
    assert data["experimental"] is True


def test_human_output_and_report_file(tmp_path, chocolate_run, capsys):
    gold = _gold(tmp_path / "gold.csv", CHOCOLATE)
    out = tmp_path / "eval.json"
    code = main(
        ["cluster", "eval", "--run", str(chocolate_run), "--gold", str(gold), "--out", str(out)]
    )
    text = capsys.readouterr().out
    assert code == 0
    assert "pairwise" in text and "B-cubed" in text and "co-exposed" in text
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["scores"]["as_singletons"]["bcubed_f1"] == 1.0


def test_evaluation_makes_no_model_or_retriever_call_and_writes_nothing(
    tmp_path, chocolate_run, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("evaluation must not call a model or a retriever")

    monkeypatch.setattr(FakeLLM, "complete", refuse)
    monkeypatch.setattr(PoolIndex, "search", refuse)
    monkeypatch.setattr(PoolIndex, "upsert", refuse)
    gold = _gold(tmp_path / "gold.csv", CHOCOLATE)
    before = _files(chocolate_run)
    result = ops.cluster_eval(chocolate_run, gold)
    assert result.exit_code == 0
    assert _files(chocolate_run) == before  # no file added, none changed


def test_a_store_with_a_pending_log_is_read_without_touching_it(tmp_path, chocolate_run):
    # A writer holding the store open (a run in progress) leaves -wal and -shm files.
    from xwalk.cluster.store import ClusterStore

    writer = ClusterStore.open(chocolate_run / "cluster.sqlite")
    try:
        writer.set_meta("probe", 1)  # a committed write that lives in the log for now
        assert (chocolate_run / "cluster.sqlite-wal").exists()
        before = _files(chocolate_run)
        result = ops.cluster_eval(chocolate_run, _gold(tmp_path / "gold.csv", CHOCOLATE))
        assert result.data["exposure"]["gold_pairs_co_exposed"] == 4
        assert _files(chocolate_run) == before
    finally:
        writer.close()


async def test_merges_count_as_exposure_the_bridge(tmp_path):
    # "alpha" and "beta" share no token and never meet while streaming; "alpha beta"
    # joins "alpha", and a verified merge brings in "beta". Every gold pair is
    # co-clustered, and the merge prompt is where alpha and beta were shown together.
    table = {"a": ("alpha", "g"), "b": ("beta", "g"), "c": ("alpha beta", "g")}
    run = tmp_path / "run"
    await cluster(
        records(table), run, Oracle(concepts(table)), settings=settings(order="input", scan_below=0)
    )
    assert partition(run) == {frozenset({"a", "b", "c"})}
    result = ops.cluster_eval(run, _gold(tmp_path / "gold.csv", table))
    assert result.data["scores"]["as_singletons"]["pairwise_recall"] == 1.0
    assert result.data["exposure"]["gold_pairs_co_exposed"] == 3
    assert result.data["exposure"]["gold_pairs"] == 3


async def test_co_exposure_is_not_an_upper_bound_on_recall(tmp_path):
    # With one member shown per cluster, "x core two" joins after seeing only "x core":
    # it is never in a prompt with "x core one", yet all three are one cluster.
    table = {"a": ("x core", "g"), "b": ("x core one", "g"), "c": ("x core two", "g")}
    run = tmp_path / "run"
    config = settings(order="input", member_evidence=1)
    await cluster(records(table), run, Oracle(concepts(table)), settings=config)
    assert partition(run) == {frozenset({"a", "b", "c"})}
    result = ops.cluster_eval(run, _gold(tmp_path / "gold.csv", table))
    assert result.data["scores"]["as_singletons"]["pairwise_recall"] == 1.0
    exposure = result.data["exposure"]
    assert (exposure["gold_pairs_co_exposed"], exposure["gold_pairs"]) == (2, 3)


async def test_an_aborted_run_is_scored_with_pending_items_and_a_warning(tmp_path):
    run = tmp_path / "run"
    await cluster(records(CHOCOLATE), run, Oracle(concepts(CHOCOLATE)), max_calls=4)
    rows = member_rows(run)
    assert any(r["outcome"] == "pending" for r in rows.values())
    result = ops.cluster_eval(run, _gold(tmp_path / "gold.csv", CHOCOLATE))
    assert result.exit_code == 0
    assert result.run["run_state"] == "aborted"
    assert result.counts["pending"] > 0
    assert "partial_run" in [w.code for w in result.warnings]


def test_gold_items_missing_from_the_run_are_reported(tmp_path, chocolate_run):
    gold = _gold(tmp_path / "gold.csv", CHOCOLATE, zz="other")
    result = ops.cluster_eval(chocolate_run, gold)
    assert result.counts["gold_items"] == 9
    assert result.data["scores"]["outcome_counts"]["missing"] == 1
    assert "gold_not_in_run" in [w.code for w in result.warnings]


def test_usage_errors_exit_2(tmp_path, chocolate_run, capsys):
    gold = _gold(tmp_path / "gold.csv", CHOCOLATE)
    cases = [
        (["--run", str(tmp_path / "nope"), "--gold", str(gold)], "run_not_found"),
        (["--run", str(chocolate_run), "--gold", str(tmp_path / "nope.csv")], "gold_unreadable"),
    ]
    (tmp_path / "matching").mkdir()
    (tmp_path / "matching" / "manifest.json").write_text('{"run_fingerprint": "x"}')
    cases.append((["--run", str(tmp_path / "matching"), "--gold", str(gold)], "not_a_cluster_run"))
    dup = tmp_path / "dup.csv"
    dup.write_text("item_id,cluster_id\nc1,a\nc1,b\n")
    cases.append((["--run", str(chocolate_run), "--gold", str(dup)], "gold_unreadable"))
    for args, expected in cases:
        code = main(["cluster", "eval", *args, "--json"])
        envelope = json.loads(capsys.readouterr().out)
        assert code == envelope["exit_code"] == 2, args
        assert envelope["operation"] == "cluster eval"
        assert envelope["errors"][0]["code"] == expected, args


def test_cluster_without_job_and_out_is_a_usage_error(capsys):
    code = main(["cluster", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert code == envelope["exit_code"] == 2 and envelope["errors"][0]["code"] == "usage"


# --- matching-run commands on a clustering run -------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "--run", "{run}"],
        ["explain", "c1", "--run", "{run}"],
        ["results", "--run", "{run}"],
        ["eval", "--run", "{run}", "--gold", "{gold}"],
    ],
    ids=["inspect", "explain", "results", "eval"],
)
def test_matching_commands_point_a_cluster_run_to_the_cluster_commands(
    tmp_path, chocolate_run, capsys, argv
):
    gold = _gold(tmp_path / "gold.csv", CHOCOLATE)
    before = _files(chocolate_run)
    args = [a.format(run=chocolate_run, gold=gold) for a in argv]
    code = main([*args, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert code == envelope["exit_code"] == 2
    error = envelope["errors"][0]
    assert error["code"] == "cluster_run"
    assert "xwalk cluster eval" in error["message"] and "members.csv" in error["message"]
    assert _files(chocolate_run) == before  # no ledger.sqlite created

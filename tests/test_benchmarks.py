"""The benchmark harness (benchmarks/): data construction, metrics, and runners.

Everything here is offline. The end-to-end checks drive the SYNTHETIC judge; they test
that the runners and accounting work, not how well anything matches.
"""

from __future__ import annotations

import json

import pytest

from benchmarks import cluster_eval, run
from benchmarks.clustering import (
    evaluate_clustering,
    order_sensitivity,
    partition_scores,
    read_assignments,
    write_assignments,
)
from benchmarks.data import (
    DataRevisionError,
    load_manifest,
    materialise,
    verify_digests,
)
from benchmarks.metrics import Prediction, matching_metrics, pairwise_metrics
from benchmarks.synthetic_llm import SyntheticJudgeLLM, parse_candidates
from xwalk.evaluate import GoldSet


def _needs_owl() -> None:
    """The cafeteria catalog is OWL, which needs xwalk[ontology]."""
    pytest.importorskip("rdflib")


# --- data -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cafeteria_fcd", "ncbi_disease"])
def test_manifests_pin_the_frozen_files(name):
    manifest = load_manifest(name)
    digests = verify_digests(manifest)
    assert set(manifest["frozen_files"].values()) <= set(digests)


def test_a_changed_file_is_refused_as_a_new_data_revision(tmp_path):
    manifest = load_manifest("cafeteria_fcd")
    rel = manifest["frozen_files"]["gold"]
    manifest["sha256"][rel] = "0" * 64
    with pytest.raises(DataRevisionError, match="new data revision"):
        verify_digests(manifest)


@pytest.mark.parametrize("name", ["cafeteria_fcd", "ncbi_disease"])
def test_the_no_match_variant_removes_only_gold_targets_and_is_deterministic(tmp_path, name):
    if name == "cafeteria_fcd":
        _needs_owl()
    manifest = load_manifest(name)
    base, nomatch = materialise(manifest, tmp_path / "a")
    _, again = materialise(manifest, tmp_path / "b")
    log = nomatch.construction

    removed = set(log["removed_target_ids"])
    base_gold_ids = {i for ids in base.gold.labels.values() for i in ids}
    assert removed and removed <= base_gold_ids  # only gold targets go
    assert {t.id for t in nomatch.targets} == {t.id for t in base.targets} - removed
    assert not removed & {t.id for t in nomatch.targets}
    empty = [s for s, ids in nomatch.gold.labels.items() if not ids]
    assert log["target_nomatch"] <= len(empty) <= log["max_nomatch"]
    assert sorted(empty) == log["nomatch_mentions"]
    # no surviving gold id points outside the catalog
    catalog = {t.id for t in nomatch.targets}
    assert all(ids <= catalog for ids in nomatch.gold.labels.values())
    assert again.construction == log
    assert (again.directory / "gold.csv").read_text() == (
        nomatch.directory / "gold.csv"
    ).read_text()
    # clustering gold ignores the catalog construction
    assert nomatch.base_gold == base.base_gold


def test_splits_are_hash_based_and_cover_every_source(tmp_path):
    (variant,) = materialise(load_manifest("ncbi_disease"), tmp_path, variants=["ncbi_disease"])
    assert set(variant.split) == {s.id for s in variant.sources}
    assert set(variant.split.values()) <= {"prompt_train", "validation", "test"}
    assert 0 < len(variant.ids_in_split("test")) < len(variant.sources)


# --- the synthetic judge --------------------------------------------------------------

SELECT = """You are matching a food to at most one term.
Domain: food

## Source record
- mention: Onions
- context_left:

## Candidates
[C01] ID: X:1
Label: garlic

[C02] ID: X:2
Label: onion bulb
Synonyms: onion; brown onion

## Rules
- Choose
"""


async def test_the_synthetic_judge_selects_by_similarity_and_abstains_without_one():
    judge = SyntheticJudgeLLM()
    assert [c.key for c in parse_candidates(SELECT)] == ["C01", "C02"]
    role, payload = judge.answer(SELECT)
    assert role == "select" and payload["chosen_key"] == "C02"
    _, payload = judge.answer(SELECT.replace("Onions", "zzzz qqq"))
    assert payload["chosen_key"] is None
    from xwalk.llm.base import LLMRequest

    response = await judge.complete(LLMRequest(system="s", user=SELECT))
    assert response.usage.calls == 1 and response.usage.prompt_tokens > 0
    assert judge.calls_by_role == {"select": 1}
    assert judge.fingerprint == SyntheticJudgeLLM().fingerprint


# --- metrics --------------------------------------------------------------------------


def test_matching_metrics_count_only_accepted_answers_as_output():
    gold = GoldSet(
        {"a": frozenset({"T1"}), "b": frozenset({"T2"}), "c": frozenset(), "d": frozenset({"T4"})}
    )
    preds = [
        Prediction("a", "matched", "T1", retrieved_ids=("T1", "T9"), shown_ids=("T1",)),
        Prediction("b", "needs_review", "T2", retrieved_ids=("T2",), shown_ids=()),
        Prediction("c", "matched", "T3", retrieved_ids=("T3",), shown_ids=("T3",)),
        Prediction("d", "unmatched", None, retrieved_ids=(), shown_ids=()),
    ]
    m = matching_metrics(preds, gold)
    assert m["accepted"] == 2 and m["accepted_precision"] == 0.5
    assert m["final_recall"] == round(1 / 3, 4)
    assert m["review_rate"] == 0.25
    assert m["recall_any_status"] == round(2 / 3, 4)
    assert m["false_accept_on_no_match"] == 1
    assert m["no_match_precision"] == 0.0 and m["no_match_recall"] == 0.0
    assert m["candidate_recall"] == round(2 / 3, 4)
    assert m["truncated"] == 1  # b: retrieved, cut before the model saw it
    assert matching_metrics(preds, gold, ["a"])["n"] == 1


def test_recovery_and_retry_damage_are_counted_separately():
    gold = GoldSet({"a": frozenset({"T1"}), "b": frozenset({"T2"})})
    preds = [
        Prediction("a", "matched", "T1", first_choice_id="T5", has_first_choice=True, attempts=2),
        Prediction(
            "b", "needs_review", "T7", first_choice_id="T2", has_first_choice=True, attempts=2
        ),
    ]
    m = matching_metrics(preds, gold)
    assert m["recovered"] == 1 and m["broken_by_retry"] == 1


def test_pairwise_metrics():
    m = pairwise_metrics([True, True, False, False], [True, False, True, False])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_partition_scores_pairwise_bcubed_merges_and_splits():
    gold = {"a": "1", "b": "1", "c": "2", "d": "3"}
    pred = {"a": "x", "b": "y", "c": "y", "d": "z"}
    s = partition_scores(pred, gold)
    assert s["pairwise_precision"] == 0.0 and s["pairwise_recall"] == 0.0
    # B-cubed precision: a 1, b 1/2, c 1/2, d 1 -> 0.75; recall: a 1/2, b 1/2, c 1, d 1 -> 0.75
    assert s["bcubed_precision"] == 0.75 and s["bcubed_recall"] == 0.75
    assert s["false_merge_clusters"] == 1 and s["false_merge_pairs"] == 1
    assert s["split_gold_clusters"] == 1 and s["missed_pairs"] == 1
    assert s["correct_singletons"] == 1 and s["wrong_singletons"] == 1
    assert partition_scores(gold, gold)["bcubed_f1"] == 1.0


def test_review_items_are_singletons_or_excluded_never_credited():
    gold = {"a": "1", "b": "1", "c": "2"}
    clusters = {"a": "k", "b": "k", "c": "k"}
    outcomes = {"a": "assigned", "b": "assigned", "c": "needs_review"}
    result = evaluate_clustering(clusters, outcomes, gold)
    assert result["outcome_counts"]["needs_review"] == 1
    assert result["as_singletons"]["false_merge_clusters"] == 0
    assert result["accepted_only"]["items"] == 2
    assert result["accepted_only"]["pairwise_f1"] == 1.0


def test_order_sensitivity_flags_items_whose_membership_changes():
    one = {"a": "1", "b": "1", "c": "2"}
    two = {"a": "1", "b": "2", "c": "2"}
    s = order_sensitivity([one, two])
    assert s["items_with_unstable_membership"] == 3
    assert s["min_pairwise_f1_between_runs"] == 0.0
    assert order_sensitivity([one, dict(one)])["items_with_unstable_membership"] == 0
    assert order_sensitivity([one]) is None


def test_assignment_files_round_trip_and_reject_duplicates(tmp_path):
    path = tmp_path / "pred.csv"
    write_assignments(path, {"a": "k", "b": "k", "c": "m"}, {"c": "needs_review"})
    clusters, outcomes = read_assignments(path)
    assert clusters == {"a": "k", "b": "k", "c": "m"}
    assert outcomes == {"a": "assigned", "b": "assigned", "c": "needs_review"}
    jsonl = tmp_path / "pred.jsonl"
    jsonl.write_text('{"item_id": "a", "cluster_id": "k"}\n{"item_id": "a", "cluster_id": "j"}\n')
    with pytest.raises(ValueError, match="repeats item_id"):
        read_assignments(jsonl)


def test_cluster_eval_cli_scores_files_and_carries_meta(tmp_path):
    gold, pred, meta, out = (tmp_path / n for n in ("g.csv", "p.csv", "m.json", "o.json"))
    write_assignments(gold, {"a": "1", "b": "1", "c": "2"})
    write_assignments(pred, {"a": "x", "b": "x", "c": "y"})
    meta.write_text(json.dumps({"calls": 7}))
    assert (
        cluster_eval.main(
            ["--gold", str(gold), "--pred", str(pred), "--meta", str(meta), "--out", str(out)]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["scores"]["as_singletons"]["bcubed_f1"] == 1.0
    assert result["meta"] == {"calls": 7}
    assert cluster_eval.main(["--gold", str(gold), "--pred", str(tmp_path / "nope.csv")]) == 2


# --- end to end -----------------------------------------------------------------------


def _strip_costs(report: dict) -> dict:
    tracks = report["tracks"]
    out = {}
    for variant, methods in tracks["matching"].items():
        for name, row in methods.items():
            out[(variant, name)] = (row["metrics"], row["usage"])
    for variant, row in tracks["pairwise"].items():
        out[(variant, "pairwise")] = row
    for dataset, rows in tracks["clustering"].items():
        for name, row in rows.items():
            if not name.startswith("_"):
                out[(dataset, name)] = row["scores"]
    return out


def test_the_smoke_run_is_labelled_synthetic_complete_and_deterministic(tmp_path):
    first = run.run("smoke", tmp_path / "out1", tmp_path / "w1", datasets=["ncbi_disease"], limit=5)
    second = run.run(
        "smoke", tmp_path / "out2", tmp_path / "w2", datasets=["ncbi_disease"], limit=5
    )

    assert first["kind"] == "SYNTHETIC" and "SYNTHETIC" in first["warning"]
    summary = (tmp_path / "out1" / "summary.md").read_text()
    assert summary.startswith("# Benchmark summary (SYNTHETIC)")
    saved = json.loads((tmp_path / "out1" / "results.json").read_text())
    assert saved["kind"] == "SYNTHETIC"

    matching = first["tracks"]["matching"]
    assert set(matching) == {"ncbi_disease", "ncbi_disease_nomatch"}
    for rows in matching.values():
        assert set(rows) == {
            "exact",
            "fuzzy_difflib",
            "bm25_top1",
            "retrieval_one_llm_pass",
            "xwalk_single_attempt",
            "xwalk_full",
        }
        for name, row in rows.items():
            assert row["metrics"]["all"]["n"] == 5
            assert row["cost"]["wall_seconds"] >= 0
            if name in ("exact", "fuzzy_difflib", "bm25_top1"):
                assert row["usage"] is None
            else:
                assert row["usage"]["calls"] > 0
    # the single-attempt route never verifies or retries, so it makes no more calls
    for rows in matching.values():
        assert (
            rows["xwalk_single_attempt"]["usage"]["calls"] <= rows["xwalk_full"]["usage"]["calls"]
        )
    assert "ncbi_disease" in first["tracks"]["clustering"]
    assert (tmp_path / "out1" / "clustering" / "ncbi_disease" / "gold_clusters.csv").exists()
    assert first["not_run"]["linktransformer"].startswith("pending")
    assert _strip_costs(first) == _strip_costs(second)


def test_real_mode_needs_a_call_cap_and_a_credential(tmp_path, monkeypatch, capsys):
    for name in ("OPENAI_API_KEY", *run.ENV_OVERRIDES):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as exc:
        run.main(["--real", "--out", str(tmp_path / "o")])
    assert exc.value.code == 2
    code = run.main(
        [
            "--real",
            "--max-calls",
            "1",
            "--datasets",
            "ncbi_disease",
            "--limit",
            "1",
            "--out",
            str(tmp_path / "o"),
            "--work",
            str(tmp_path / "w"),
        ]
    )
    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "o" / "results.json").exists()

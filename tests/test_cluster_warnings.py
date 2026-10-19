"""Factual warnings on clustering runs, and regressions that show why they exist.

Three limits of the design are reported on every run and by `cluster inspect`, each
at most once:

- `retrieval_bounded`: the pool grew past `pool.scan_below` clusters with lexical
  retrieval only, so a record met only clusters sharing a word with it.
- `evidence_bounded`: an accepted cluster has more than `pool.member_evidence` members,
  so later members were verified against the first members shown, not every member.
- `resume_determinism`: the run was resumed (or will be), which reproduces an
  uninterrupted run only if the model answers identical prompts identically.

The regressions at the end are executable demonstrations, kept from the advisors'
probes: a cluster whose later members its own judge rejects pairwise (hidden by the
evidence bound), and a membership exported although a later decision rejected it
(revision selection). They pin today's behaviour and its visibility, not a fix.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from tests.cluster_helpers import (
    CHOCOLATE,
    TEMPLATES,
    Oracle,
    cluster,
    concepts,
    member_rows,
    records,
    sections,
    settings,
)
from tests.conftest import FIXTURES
from tests.test_cluster_pool import ToyEncoder
from xwalk import ops
from xwalk.cluster import run_clustering
from xwalk.cluster.prompts import ClusterPrompts, cluster_block
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.records import Record
from xwalk.ui.offline import lexical_cluster_model

FACTUAL = ("retrieval_bounded", "evidence_bounded", "resume_determinism")
# Four concepts that share no word: every record mints its own cluster.
DISTINCT = {f"r{i}": (label, label) for i, label in enumerate(["apple", "brick", "cloud", "drum"])}
# One concept with three members.
TRIO = {"a": ("x core", "g"), "b": ("x core one", "g"), "c": ("x core two", "g")}


def _codes(warnings) -> list[str]:
    return [w["code"] if isinstance(w, dict) else w.code for w in warnings]


def _factual(warnings) -> list[str]:
    return [c for c in _codes(warnings) if c in FACTUAL]


# --- retrieval_bounded --------------------------------------------------------------------


async def test_retrieval_warning_starts_above_scan_below(tmp_path):
    oracle = Oracle(concepts(DISTINCT))
    at = await cluster(records(DISTINCT), tmp_path / "at", oracle, settings=settings(scan_below=4))
    above = await cluster(
        records(DISTINCT), tmp_path / "above", oracle, settings=settings(scan_below=3)
    )
    assert "retrieval_bounded" not in _codes(at.warnings)  # 4 clusters, scan_below 4
    (warning,) = [w for w in above.warnings if w["code"] == "retrieval_bounded"]
    assert "4 clusters" in warning["message"] and "scan_below=3" in warning["message"]
    assert "lexical" in warning["message"] and "synonyms" in warning["message"]


async def test_no_retrieval_warning_with_a_dense_head(tmp_path):
    report = await cluster(
        records(DISTINCT),
        tmp_path / "dense",
        Oracle(concepts(DISTINCT)),
        settings=settings(scan_below=3),
        encoder=ToyEncoder(),
    )
    assert "retrieval_bounded" not in _codes(report.warnings)


# --- evidence_bounded ---------------------------------------------------------------------


async def test_evidence_warning_starts_above_member_evidence(tmp_path):
    oracle = Oracle(concepts(TRIO))
    at = await cluster(records(TRIO), tmp_path / "at", oracle, settings=settings(member_evidence=3))
    above = await cluster(
        records(TRIO), tmp_path / "above", oracle, settings=settings(member_evidence=2)
    )
    assert "evidence_bounded" not in _codes(at.warnings)  # 3 members, 3 shown
    (warning,) = [w for w in above.warnings if w["code"] == "evidence_bounded"]
    assert "1 accepted cluster" in warning["message"] and "member_evidence=2" in warning["message"]
    assert "largest 3" in warning["message"]


async def test_warnings_are_deduplicated(tmp_path):
    table = {**TRIO, "d": ("y base", "h"), "e": ("y base one", "h"), "f": ("y base two", "h")}
    run = tmp_path / "run"
    report = await cluster(
        records(table), run, Oracle(concepts(table)), settings=settings(member_evidence=2)
    )
    (warning,) = [w for w in report.warnings if w["code"] == "evidence_bounded"]
    assert "2 accepted cluster" in warning["message"]
    for warnings in (report.warnings, ops.cluster_inspect(run).warnings):
        codes = _codes(warnings)
        assert len(codes) == len(set(codes)), codes
    assert "evidence_bounded" in _codes(ops.cluster_inspect(run).warnings)


# --- resume_determinism -------------------------------------------------------------------


async def test_resume_warning_only_once_a_run_is_or_will_be_resumed(tmp_path):
    oracle = Oracle(concepts(CHOCOLATE))
    clean = await cluster(records(CHOCOLATE), tmp_path / "clean", oracle)
    assert "resume_determinism" not in _codes(clean.warnings)
    run = tmp_path / "run"
    aborted = await cluster(records(CHOCOLATE), run, oracle, max_calls=4)
    assert aborted.run_state == "aborted"
    assert "resume_determinism" in _codes(aborted.warnings)
    resumed = await cluster(records(CHOCOLATE), run, oracle)
    assert resumed.run_state == "complete"
    (warning,) = [w for w in resumed.warnings if w["code"] == "resume_determinism"]
    assert "2 invocations" in warning["message"] and "identical" in warning["message"]
    assert "resume_determinism" in _codes(ops.cluster_inspect(run).warnings)


# --- the operation's envelope ---------------------------------------------------------------


async def test_cluster_operation_reports_the_warnings_once(tmp_path):
    labels = FIXTURES / "cluster_tiny" / "labels.csv"
    job = tmp_path / "job.yaml"
    job.write_text(
        (FIXTURES / "cluster_tiny" / "job.yaml")
        .read_text()
        .replace("shown_limit: 8", "scan_below: 1")
    )
    (tmp_path / "labels.csv").write_bytes(labels.read_bytes())
    result = await ops.cluster_async(job, tmp_path / "run", llm=Oracle(concepts(CHOCOLATE)).llm)
    codes = _codes(result.warnings)
    assert "retrieval_bounded" in codes and len(codes) == len(set(codes))
    assert "retrieval_bounded" in _codes(ops.cluster_inspect(tmp_path / "run").warnings)


# --- regressions: what the warnings are about --------------------------------------------


async def test_regression_hidden_contradictory_members_behind_the_evidence_bound(tmp_path):
    # A five-word core with four exact-token duplicates, then six variants that each add
    # two private words. Each variant is verified against the five members shown, all
    # duplicates of the core; the variants are never shown to each other, and the same
    # judge rejects every pair of them. (Advisor probe `evidence_probe.py`, edge C.)
    core = "alpha beta gamma delta eps"
    labels = {
        "m0": core,
        "m1": core + " ",
        "m2": "Alpha beta gamma delta eps",
        "m3": core + ".",
        "m4": "alpha, beta gamma delta eps",
    }
    labels.update({f"v{i}": f"{core} x{i} y{i}" for i in range(6)})
    run = tmp_path / "run"
    report = await run_clustering(
        [Record(id=i, fields={"label": label}) for i, label in labels.items()],
        out=run,
        llm=FakeLLM(handler=lexical_cluster_model),
        templates=TEMPLATES,
        settings=settings(order="input"),
    )
    rows = member_rows(run)
    assert {r["cluster_size"] for r in rows.values()} == {"11"}  # one cluster of all 11
    late = [i for i in labels if i.startswith("v")]
    rejected = 0
    for a, b in itertools.combinations(late, 2):
        prompt = ClusterPrompts().verify(labels[a], cluster_block([labels[b]], 1))
        answer = json.loads(lexical_cluster_model(LLMRequest(system="", user=prompt)))
        rejected += answer["decision"] == "not_equivalent"
    assert rejected == 15  # every pair of later members, by the run's own judge

    # The run says so, and explain shows the evidence each variant was judged on.
    (warning,) = [w for w in report.warnings if w["code"] == "evidence_bounded"]
    assert "largest 11" in warning["message"]
    explained = ops.cluster_explain(run, "v5").data
    verify = next(d for d in explained["admission"] if d["kind"] == "verify")
    assert verify["shown"][0]["members_shown"] == ["m0", "m1", "m2", "m3", "m4"]
    assert not any(
        e["subject_id"].startswith("v") for e in ops.cluster_explain(run, "v0").data["evidence_in"]
    )


async def test_regression_an_exported_membership_a_later_decision_rejected(tmp_path):
    # Refinement removes `s` after a failed reverification, but the run stops at
    # max_iterations and exports revision 0, where `s` is still a member. (Advisor probe
    # `evidence_probe.py`, rollback.) This follows CONTRACTS 10.7; the point is that it
    # is visible: the run warns, inspect warns, explain lists the rejection.
    table = {"s": ("aaa core", "X"), "t": ("aaa core variant", "X"), "u": ("core zzz other", "Z")}

    def against_s(name, request):
        if sections(request.user).get("Record", "").strip() != "aaa core":
            return None
        if name == "select-cluster":
            return json.dumps({"chosen_key": None, "confidence_score": 0.9, "explanation": "x"})
        if name == "verify-assignment":
            return json.dumps(
                {"decision": "not_equivalent", "confidence_score": 0.9, "explanation": "x"}
            )
        return None

    run = tmp_path / "run"
    report = await cluster(
        records(table),
        run,
        Oracle(concepts(table), override=against_s),
        settings=settings(max_refine_iterations=1),
    )
    assert member_rows(run)["s"]["outcome"] == "assigned"
    assert "selected_not_last" in _codes(report.warnings)
    assert "selected_not_last" in _codes(ops.cluster_inspect(run).warnings)
    later = ops.cluster_explain(run, "s").data["later"]
    rejection = next(e for e in later if e["type"] == "assignment")
    assert rejection["adverse"] and rejection["reason"] == "reverify_failed"


def test_thresholds_are_documented():
    guide = (Path(__file__).parents[1] / "docs" / "guide" / "clustering.md").read_text()
    for code in (*FACTUAL, "selected_not_last"):
        assert f"`{code}`" in guide, code

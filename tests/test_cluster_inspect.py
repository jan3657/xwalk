"""`xwalk cluster inspect` and `xwalk cluster explain`: read-only views of a clustering run.

Both read the manifest and the store (opened read-only) and nothing else: no model call,
no retriever, no write. Inspect summarises the exported state (outcomes, cluster sizes,
usage over every invocation, selected versus last revision, warnings); explain shows
one record's exported membership, the decisions behind it, the merge ancestry of its
cluster, and later decisions against it that revision selection set aside. The judge is
the scripted `Oracle`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tests.cluster_helpers import (
    CHOCOLATE,
    Oracle,
    cluster,
    concepts,
    member_rows,
    records,
    sections,
    settings,
)
from xwalk import ops
from xwalk.cli.main import main
from xwalk.cluster.pool import PoolIndex
from xwalk.cluster.store import ClusterStore
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


def _files(run: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run.iterdir())}


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main([*argv, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert set(envelope) == ENVELOPE_KEYS and envelope["schema_version"] == 1
    assert code == envelope["exit_code"]
    return code, envelope


def _codes(messages: list[dict]) -> list[str]:
    return [m["code"] for m in messages]


@pytest.fixture
async def chocolate_run(tmp_path) -> Path:
    run = tmp_path / "chocolate"
    await cluster(records(CHOCOLATE), run, Oracle(concepts(CHOCOLATE)))
    return run


@pytest.fixture
async def aborted_run(tmp_path) -> Path:
    run = tmp_path / "aborted"
    await cluster(records(CHOCOLATE), run, Oracle(concepts(CHOCOLATE)), max_calls=4)
    return run


@pytest.fixture
async def failed_run(tmp_path) -> Path:
    def garbage(name, request):
        if name == "select-cluster" and sections(request.user).get("Record") == "quinoa":
            return "this is not JSON"
        return None

    run = tmp_path / "failed"
    await cluster(records(CHOCOLATE), run, Oracle(concepts(CHOCOLATE), override=garbage))
    return run


@pytest.fixture
async def merged_run(tmp_path) -> Path:
    # "alpha" and "beta" share no word; "alpha beta" joins "alpha", and consolidation
    # merges the "beta" cluster into it.
    table = {"a": ("alpha", "g"), "b": ("beta", "g"), "c": ("alpha beta", "g")}
    run = tmp_path / "merged"
    await cluster(
        records(table), run, Oracle(concepts(table)), settings=settings(order="input", scan_below=0)
    )
    return run


ROLLBACK = {"s": ("aaa core", "X"), "t": ("aaa core variant", "X"), "u": ("core zzz other", "Z")}


@pytest.fixture
async def rollback_run(tmp_path) -> Path:
    # Refinement removes `s` after a failed reverification, but the run stops at
    # max_iterations and selects revision 0 (fewer unresolved), which still has `s`.
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

    run = tmp_path / "rollback"
    report = await cluster(
        records(ROLLBACK),
        run,
        Oracle(concepts(ROLLBACK), override=against_s),
        settings=settings(max_refine_iterations=1),
    )
    assert (report.stop_reason, report.selected_revision, report.last_revision) == (
        "max_iterations",
        0,
        1,
    )
    return run


# --- inspect -------------------------------------------------------------------------


def test_inspect_summarises_a_complete_run(chocolate_run, capsys):
    code, envelope = _cli(capsys, "cluster", "inspect", "--run", str(chocolate_run))
    assert code == 0 and envelope["operation"] == "cluster inspect"
    assert envelope["status"] == "ok"
    assert envelope["run"]["run_state"] == "complete"
    counts = envelope["counts"]
    rows = member_rows(chocolate_run)
    for outcome in ("assigned", "singleton", "needs_review", "failed", "pending"):
        assert counts[outcome] == sum(1 for r in rows.values() if r["outcome"] == outcome)
    assert counts["total"] == 8 and counts["clusters"] == 5
    data = envelope["data"]
    # dark chocolate x3, heart attack/myocardial infarction, and three singletons
    assert data["clusters"]["by_size"] == {"1": 3, "2": 1, "3": 1}
    assert data["clusters"]["largest"][0]["size"] == 3
    assert data["selected_revision"] == data["last_revision"] == data["exported_revision"]
    assert data["selected_is_last"] is True
    assert [r["selected"] for r in data["revisions"]].count(True) == 1
    assert sum(data["decisions"]["by_kind"].values()) == data["decisions"]["total"]
    usage = envelope["usage"]
    assert usage["calls"] > 0 and usage["invocations"] == 1
    assert usage["invocations_without_usage"] == 0
    assert usage["calls_made_by_inspect"] == 0
    assert "experimental" in _codes(envelope["warnings"])


def test_inspect_human_output(chocolate_run, capsys):
    code = main(["cluster", "inspect", "--run", str(chocolate_run)])
    text = capsys.readouterr().out
    assert code == 0
    assert "run state" in text and "cluster sizes" in text and "revision" in text


def test_inspect_an_aborted_run_reports_pending_sources(aborted_run, capsys):
    code, envelope = _cli(capsys, "cluster", "inspect", "--run", str(aborted_run))
    assert code == 3  # what the run itself exited with
    assert envelope["run"]["run_state"] == "aborted"
    assert envelope["counts"]["pending"] > 0
    assert "partial_run" in _codes(envelope["warnings"])
    assert "call_limit_reached" in _codes(envelope["errors"])
    assert envelope["data"]["stop_reason"] is None


def test_inspect_a_run_with_failed_sources_exits_3(failed_run, capsys):
    code, envelope = _cli(capsys, "cluster", "inspect", "--run", str(failed_run))
    assert code == 3 and envelope["run"]["run_state"] == "failed"
    assert envelope["counts"]["failed"] == 1
    assert "source_failed" in _codes(envelope["errors"])


def test_inspect_reports_unfinished_invocations_as_a_usage_lower_bound(chocolate_run, capsys):
    conn = sqlite3.connect(chocolate_run / "cluster.sqlite")
    conn.execute(
        "INSERT INTO invocations (started_at, run_state) VALUES ('2026-10-19', 'interrupted')"
    )
    conn.commit()
    conn.close()
    _, envelope = _cli(capsys, "cluster", "inspect", "--run", str(chocolate_run))
    assert envelope["usage"]["invocations"] == 2
    assert envelope["usage"]["invocations_without_usage"] == 1
    assert "usage_incomplete" in _codes(envelope["warnings"])
    assert [i["calls"] for i in envelope["data"]["invocations"]][1] is None


def test_inspect_shows_selected_versus_last_revision(rollback_run, capsys):
    code, envelope = _cli(capsys, "cluster", "inspect", "--run", str(rollback_run))
    data = envelope["data"]
    assert (data["selected_revision"], data["last_revision"]) == (0, 1)
    assert data["exported_revision"] == 0 and data["selected_is_last"] is False
    assert data["stop_reason"] == "max_iterations"
    # `s` is assigned in the export but was removed in the last revision
    assert data["changed_after_export"] == {"sources": 1, "source_ids": ["s"]}
    warning = next(w for w in envelope["warnings"] if w["code"] == "selected_not_last")
    assert "revision 0" in warning["message"] and "xwalk cluster explain" in warning["message"]
    assert code == 0  # the exported state has nothing to review


def test_inspect_counts_merges(merged_run, capsys):
    _, envelope = _cli(capsys, "cluster", "inspect", "--run", str(merged_run))
    assert envelope["data"]["merges"]["applied"] == 1
    assert envelope["data"]["clusters"]["by_size"] == {"3": 1}


# --- explain -------------------------------------------------------------------------


def test_explain_a_verified_member(chocolate_run, capsys):
    rows = member_rows(chocolate_run)
    code, envelope = _cli(capsys, "cluster", "explain", "c3", "--run", str(chocolate_run))
    assert code == 0 and envelope["operation"] == "cluster explain"
    data = envelope["data"]
    assert data["source_id"] == "c3" and data["status"] == rows["c3"]["outcome"] == "assigned"
    membership = data["cluster"]
    assert membership["cluster_id"] == rows["c3"]["cluster_id"]
    assert sorted(m["source_id"] for m in membership["members"]) == ["c2", "c3", "c4"]
    assert membership["size"] == 3 and membership["status"] == "verified"
    kinds = [d["kind"] for d in data["admission"]]
    assert kinds[-1] in ("verify", "mint", "novelty")
    assert data["exported"]["decision_id"] in [d["decision_id"] for d in data["admission"]]
    for decision in data["admission"]:
        assert "prompt" not in decision  # only with --full
        for entry in decision["shown"]:
            assert isinstance(entry["members_shown"], list)
    assert data["later"] == [] and data["superseded_by_selection"] is False
    assert data["merge_ancestry"] == []


def test_explain_full_includes_the_stored_prompts(chocolate_run, capsys):
    _, envelope = _cli(capsys, "cluster", "explain", "c3", "--run", str(chocolate_run), "--full")
    decision = next(d for d in envelope["data"]["admission"] if d["kind"] != "mint")
    assert decision["prompt"].startswith("Task: ") and decision["raw"]


def test_explain_a_merged_member_shows_the_merge_ancestry(merged_run, capsys):
    _, envelope = _cli(capsys, "cluster", "explain", "b", "--run", str(merged_run))
    data = envelope["data"]
    assert data["status"] == "assigned" and data["exported"]["reason"] == "merged"
    assert [d["kind"] for d in data["admission"]] == ["merge_select", "merge"]
    (merge,) = data["merge_ancestry"]
    assert merge["winner_id"] == data["cluster"]["cluster_id"]
    assert merge["absorbed_members"] == ["b"]
    assert merge["decision_id"] == data["exported"]["decision_id"]
    # the record that started the winning cluster sees the same ancestry
    _, other = _cli(capsys, "cluster", "explain", "a", "--run", str(merged_run))
    assert other["data"]["merge_ancestry"] == data["merge_ancestry"]


async def test_explain_merge_ancestry_follows_the_state_restored_on_resume(tmp_path, capsys):
    table = {
        "a": ("alpha", "g"),
        "b": ("beta", "g"),
        "c": ("alpha beta", "g"),
        "q": ("quinoa", "q"),
    }

    def broken(name, request):
        if sections(request.user).get("Record") == "quinoa":
            return "not JSON"
        return None

    run = tmp_path / "restored"
    config = settings(order="input", scan_below=0, max_refine_iterations=1)
    first = await cluster(
        records(table), run, Oracle(concepts(table), override=broken), settings=config
    )
    assert (first.selected_revision, first.last_revision) == (0, 1)
    store = ClusterStore.open_readonly(run / "cluster.sqlite")
    try:
        (merge,) = [m for m in store.merges() if m["outcome"] == "applied"]
    finally:
        store.close()
    before = member_rows(run)
    assert before["q"]["outcome"] == "failed"
    assert before["a"]["cluster_id"] != before["b"]["cluster_id"]
    assert merge["winner_id"] in (before["a"]["cluster_id"], before["b"]["cluster_id"])

    resumed = await cluster(records(table), run, Oracle(concepts(table)), settings=config)
    assert resumed.exported_revision == 2
    after = member_rows(run)
    for sid in ("a", "b", "c"):
        assert after[sid]["cluster_id"] == before[sid]["cluster_id"]
        code, envelope = _cli(capsys, "cluster", "explain", sid, "--run", str(run))
        assert code == 0
        assert envelope["data"]["merge_ancestry"] == []


def test_explain_shows_adverse_decisions_set_aside_by_revision_selection(rollback_run, capsys):
    _, envelope = _cli(capsys, "cluster", "explain", "s", "--run", str(rollback_run))
    data = envelope["data"]
    assert data["status"] == "assigned"  # the exported (selected) revision 0
    assert data["superseded_by_selection"] is True
    adverse = [entry for entry in data["later"] if entry["adverse"]]
    removal = next(e for e in adverse if e["type"] == "assignment")
    assert removal["outcome"] == "deferred" and removal["reason"] == "reverify_failed"
    assert removal["decision"]["kind"] == "reverify"
    assert removal["decision"]["result"]["decision"] == "not_equivalent"
    shrink = next(e for e in adverse if e["type"] == "cluster_revision")
    assert shrink["removed"] == ["s"]
    assert data["current"]["outcome"] == "deferred"
    assert any("not in the exports" in w["message"] for w in envelope["warnings"])
    # `t` keeps its membership, but its cluster lost a member after the export
    _, other = _cli(capsys, "cluster", "explain", "t", "--run", str(rollback_run))
    assert [e["type"] for e in other["data"]["later"] if e["adverse"]] == ["cluster_revision"]


def test_explain_a_pending_source(aborted_run, capsys):
    pending = next(
        sid for sid, row in member_rows(aborted_run).items() if row["outcome"] == "pending"
    )
    code, envelope = _cli(capsys, "cluster", "explain", pending, "--run", str(aborted_run))
    assert code == 0
    data = envelope["data"]
    assert data["status"] == "pending" and data["cluster"] is None
    assert data["admission"] == [] and data["history"] == []
    assert "resume" in data["note"]


def test_explain_a_failed_source(failed_run, capsys):
    rows = member_rows(failed_run)
    failed = next(sid for sid, row in rows.items() if row["outcome"] == "failed")
    _, envelope = _cli(capsys, "cluster", "explain", failed, "--run", str(failed_run))
    data = envelope["data"]
    assert data["status"] == "failed" and data["cluster"] is None
    assert data["admission"][-1]["error"]
    assert "retried" in data["note"]
    # each round retried it: every attempt is in the source's history
    assert len(data["history"]) >= 2


def test_explain_an_unknown_source_exits_2(chocolate_run, capsys):
    code, envelope = _cli(capsys, "cluster", "explain", "nope", "--run", str(chocolate_run))
    assert code == 2 and envelope["errors"][0]["code"] == "source_not_found"


# --- broken or foreign directories, and the read-only guarantee ------------------------


def test_missing_and_foreign_run_directories_exit_2(tmp_path, capsys):
    (tmp_path / "matching").mkdir()
    (tmp_path / "matching" / "manifest.json").write_text('{"run_fingerprint": "x"}')
    for run, expected in (
        (tmp_path / "nope", "run_not_found"),
        (tmp_path / "matching", "not_a_cluster_run"),
    ):
        for argv in (["inspect"], ["explain", "c1"]):
            code, envelope = _cli(capsys, "cluster", *argv, "--run", str(run))
            assert code == 2 and envelope["errors"][0]["code"] == expected, (run, argv)


def test_corrupt_runs_exit_3(tmp_path, chocolate_run, capsys):
    import shutil

    cases = {}
    bad_store = tmp_path / "bad_store"
    shutil.copytree(chocolate_run, bad_store)
    (bad_store / "cluster.sqlite").write_bytes(b"not a database" * 100)
    cases[bad_store] = "store_unreadable"
    no_store = tmp_path / "no_store"
    shutil.copytree(chocolate_run, no_store)
    (no_store / "cluster.sqlite").unlink()
    cases[no_store] = "store_unreadable"
    bad_manifest = tmp_path / "bad_manifest"
    shutil.copytree(chocolate_run, bad_manifest)
    (bad_manifest / "manifest.json").write_text("{not json")
    cases[bad_manifest] = "manifest_unreadable"
    for run, expected in cases.items():
        for argv in (["inspect"], ["explain", "c1"]):
            code, envelope = _cli(capsys, "cluster", *argv, "--run", str(run))
            assert code == 3 and envelope["errors"][0]["code"] == expected, (run, argv)


async def test_a_store_of_another_run_is_refused(tmp_path, chocolate_run, capsys):
    other = tmp_path / "other"
    await cluster(records(ROLLBACK), other, Oracle(concepts(ROLLBACK)))
    (chocolate_run / "cluster.sqlite").write_bytes((other / "cluster.sqlite").read_bytes())
    code, envelope = _cli(capsys, "cluster", "inspect", "--run", str(chocolate_run))
    assert code == 3 and envelope["errors"][0]["code"] == "store_mismatch"


def test_inspect_and_explain_make_no_call_and_write_nothing(
    chocolate_run, rollback_run, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("inspection must not call a model or a retriever, or write")

    monkeypatch.setattr(FakeLLM, "complete", refuse)
    monkeypatch.setattr(PoolIndex, "search", refuse)
    monkeypatch.setattr(PoolIndex, "upsert", refuse)
    monkeypatch.setattr(ClusterStore, "open", refuse)  # the read-write open
    for run in (chocolate_run, rollback_run):
        before = _files(run)
        assert ops.cluster_inspect(run).usage["calls_made_by_inspect"] == 0
        for sid in member_rows(run):
            assert ops.cluster_explain(run, sid, full=True).exit_code == 0
        assert _files(run) == before  # no file added, none changed


@pytest.mark.parametrize("before", [["--out", "x"], ["--job", "job.yaml"], ["--max-calls", "5"]])
@pytest.mark.parametrize("command", [["inspect"], ["explain", "c1"]])
def test_run_flags_before_the_subcommand_are_refused(chocolate_run, capsys, before, command):
    code, envelope = _cli(capsys, "cluster", *before, *command, "--run", str(chocolate_run))
    assert code == 2 and envelope["errors"][0]["code"] == "usage"

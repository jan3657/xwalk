"""`xwalk ui`: the API over `xwalk.ops`, the offline stand-in models and the HTTP layer.

Everything runs offline: matching with the scripted stand-in or a FakeLLM, clustering with
the lexical stand-in. The endpoint route is exercised only up to its refusals (missing
call limit, missing credential), never against a network.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FIXTURES
from xwalk import ops
from xwalk.cli.main import main
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.ui import make_server, offline
from xwalk.ui.api import ApiError, Download, Task, TaskManager, Workspace, dispatch
from xwalk.ui.server import TOKEN_HEADER, UIServer

CLUSTER_JOB = FIXTURES / "cluster_tiny" / "job.yaml"


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root, max_calls_cap=50)


@pytest.fixture
def demo(ws: Workspace) -> str:
    envelope = ws.init({"dest": "demo"})
    assert envelope["status"] == "ok"
    return str(envelope["data"]["job_path"])


def _finish(ws: Workspace, view: dict[str, Any]) -> dict[str, Any]:
    task = ws.tasks.wait(view["id"], timeout=60)
    return task.view()


def _write_gold(ws: Workspace) -> str:
    (ws.root / "gold.csv").write_text(
        "source_id,gold_ids\ns1,CHEBI:17234\ns2,CHEBI:17234\ns3,CHEBI:17992\ns4,\n",
        encoding="utf-8",
    )
    return "gold.csv"


# --- paths ------------------------------------------------------------------------------


def test_paths_resolve_inside_the_workspace(ws: Workspace) -> None:
    assert ws.path("a/b.yaml") == ws.root / "a" / "b.yaml"
    assert ws.path(str(ws.root / "x")) == ws.root / "x"
    assert ws.path(".") == ws.root


@pytest.mark.parametrize("value", ["../outside", "/etc/passwd", "a/../../outside"])
def test_paths_outside_the_workspace_are_refused(ws: Workspace, value: str) -> None:
    with pytest.raises(ApiError) as caught:
        ws.path(value)
    assert caught.value.status == 403 and caught.value.code == "outside_workspace"


def test_a_symlink_out_of_the_workspace_is_refused(ws: Workspace, tmp_path: Path) -> None:
    (tmp_path / "secret.yaml").write_text("x: 1\n", encoding="utf-8")
    (ws.root / "link.yaml").symlink_to(tmp_path / "secret.yaml")
    with pytest.raises(ApiError) as caught:
        ws.job({"path": "link.yaml"})
    assert caught.value.status == 403


def test_every_operation_refuses_an_outside_path(ws: Workspace) -> None:
    for method, path, params in [
        ("GET", "/api/job", {"path": "../x.yaml"}),
        ("POST", "/api/validate", {"job": "../x.yaml"}),
        ("POST", "/api/init", {"dest": "../escape"}),
        ("POST", "/api/tasks", {"job": "../x.yaml", "out": "run"}),
        ("GET", "/api/run", {"dir": "/"}),
        ("GET", "/api/export", {"dir": "..", "view": "raw"}),
    ]:
        with pytest.raises(ApiError) as caught:
            dispatch(ws, method, path, params)
        assert caught.value.status == 403, path
    assert not (ws.root.parent / "escape").exists()


# --- discovery and jobs ---------------------------------------------------------------


def test_workspace_lists_jobs_runs_and_gold_files(ws: Workspace, demo: str) -> None:
    shutil.copytree(CLUSTER_JOB.parent, ws.root / "cluster")
    gold = _write_gold(ws)
    (ws.root / "notes.yaml").write_text("just: a mapping\n", encoding="utf-8")
    hidden = ws.root / ".cache"
    hidden.mkdir()
    shutil.copy(ws.root / demo, hidden / "job.yaml")
    ops.run(ws.root / demo, ws.root / "demo" / "run", llm=offline.match_llm())

    listing = ws.workspace({})
    jobs = {job["path"]: job["kind"] for job in listing["jobs"]}
    assert jobs == {"demo/job.yaml": "match", "cluster/job.yaml": "cluster"}
    assert [run["dir"] for run in listing["runs"]] == ["demo/run"]
    assert listing["runs"][0]["run_state"] == "complete"
    assert listing["gold"] == [gold]


def test_job_returns_the_file_and_a_fresh_run_directory(ws: Workspace, demo: str) -> None:
    job = ws.job({"path": demo})
    assert job["kind"] == "match" and "templates:" in job["text"]
    assert job["suggested_out"] == "demo/runs/ui-1"
    (ws.root / "demo" / "runs" / "ui-1").mkdir(parents=True)
    assert ws.job({"path": demo})["suggested_out"] == "demo/runs/ui-2"


def test_validate_is_offline_and_reports_counts(ws: Workspace, demo: str) -> None:
    envelope = ws.validate({"job": demo})
    assert envelope["status"] == "ok"
    assert envelope["counts"] == {"targets": 5, "sources": 4}
    assert envelope["data"]["paid_calls"] == 0


def test_validate_reports_an_invalid_job_in_the_envelope(ws: Workspace, demo: str) -> None:
    path = ws.root / demo
    path.write_text(path.read_text(encoding="utf-8") + "\nbogus: 1\n", encoding="utf-8")
    envelope = ws.validate({"job": demo})
    assert envelope["status"] == "error" and envelope["exit_code"] == ops.EXIT_USAGE
    assert any("bogus" in e["message"] for e in envelope["errors"])


def test_preview_renders_records_through_the_templates(ws: Workspace, demo: str) -> None:
    sources = ws.preview({"job": demo, "role": "source", "limit": "2"})
    assert [row["id"] for row in sources["rows"]] == ["s1", "s2"]
    assert sources["rows"][0]["rendered"] == {
        "query": "glucose",
        "context": "blood [glucose] levels were elevated",
    }
    assert sources["next_offset"] == 2
    last = ws.preview({"job": demo, "role": "source", "offset": "2", "limit": "2"})
    assert [row["id"] for row in last["rows"]] == ["s3", "s4"]
    assert last["next_offset"] is None
    targets = ws.preview({"job": demo, "role": "target"})
    assert set(targets["rows"][0]["rendered"]) == {"doc", "candidate"}
    assert len(targets["rows"]) == 5


def test_preview_of_a_cluster_job_has_sources_only(ws: Workspace) -> None:
    shutil.copytree(CLUSTER_JOB.parent, ws.root / "cluster")
    rows = ws.preview({"job": "cluster/job.yaml"})["rows"]
    assert rows[0]["rendered"] == {"query": "chocolate"}
    with pytest.raises(ApiError):
        ws.preview({"job": "cluster/job.yaml", "role": "target"})


def test_search_builds_its_index_under_the_ui_directory(ws: Workspace, demo: str) -> None:
    envelope = ws.search({"job": demo, "query": "table sugar", "limit": "3"})
    assert envelope["status"] == "ok"
    candidates = envelope["data"]["candidates"]
    assert candidates[0]["id"] == "CHEBI:17992" and len(candidates) <= 3
    assert envelope["data"]["index_dir"].startswith(".xwalk-ui/index/")
    assert envelope["data"]["indexes"] == {"bm25": "built"}
    again = ws.search({"job": demo, "query": "glucose"})
    assert again["data"]["indexes"] == {"bm25": "opened"}
    rebuilt = ws.search({"job": demo, "query": "glucose", "rebuild": "true"})
    assert rebuilt["status"] == "ok"


def test_search_parameters_are_checked(ws: Workspace, demo: str) -> None:
    with pytest.raises(ApiError):
        ws.search({"job": demo, "query": "  "})
    with pytest.raises(ApiError):
        ws.search({"job": demo, "query": "x", "limit": str(ops.MAX_SEARCH_LIMIT + 1)})


# --- tasks ------------------------------------------------------------------------------


def test_an_offline_match_task_runs_the_pipeline(ws: Workspace, demo: str) -> None:
    view = ws.start_task({"job": demo, "out": "demo/run"})
    assert view["state"] == "running" and view["out"] == "demo/run"
    done = _finish(ws, view)
    assert done["state"] == "done"
    result = done["result"]
    assert result["status"] == "ok" and result["run"]["run_state"] == "complete"
    assert result["counts"] == {"matched": 3, "unmatched": 1, "total": 4}
    assert result["usage"]["calls"] == 7
    assert "offline_model" in [w["code"] for w in result["warnings"]]
    assert sorted(e["source_id"] for e in done["events"]) == ["s1", "s2", "s3", "s4"]

    # The UI remembers which job wrote the run, for review and "continue".
    assert ws.run({"dir": "demo/run"})["job_path"] == demo
    assert ws.workspace({})["runs"][0]["job_path"] == demo
    assert ws.job({"path": demo})["runs"] == ["demo/run"]


def test_a_limited_task_leaves_records_pending_and_resumes(ws: Workspace, demo: str) -> None:
    first = _finish(ws, ws.start_task({"job": demo, "out": "run", "limit": 1}))["result"]
    assert first["run"]["run_state"] == "partial" and first["counts"]["pending"] == 3
    second = _finish(ws, ws.start_task({"job": demo, "out": "run"}))["result"]
    assert second["run"]["run_state"] == "complete" and "pending" not in second["counts"]


def test_the_endpoint_needs_a_call_limit_within_the_cap(ws: Workspace, demo: str) -> None:
    with pytest.raises(ApiError) as caught:
        ws.start_task({"job": demo, "out": "run", "model": "endpoint"})
    assert caught.value.code == "missing_parameter"
    with pytest.raises(ApiError) as caught:
        ws.start_task({"job": demo, "out": "run", "model": "endpoint", "max_calls": 51})
    assert caught.value.code == "invalid_parameter"
    assert not (ws.root / "run").exists()


def test_the_endpoint_without_its_credential_fails_without_a_call(
    ws: Workspace, demo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    view = ws.start_task({"job": demo, "out": "run", "model": "endpoint", "max_calls": 5})
    result = _finish(ws, view)["result"]
    assert result["status"] == "error" and result["exit_code"] == ops.EXIT_USAGE
    assert [e["code"] for e in result["errors"]] == ["credential_missing"]


def test_offline_only_refuses_the_endpoint(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, allow_endpoint=False)
    ws.init({"dest": "demo"})
    with pytest.raises(ApiError) as caught:
        ws.start_task({"job": "demo/job.yaml", "out": "r", "model": "endpoint", "max_calls": 1})
    assert caught.value.status == 403
    assert ws.info({})["allow_endpoint"] is False


def test_task_parameters_are_checked(ws: Workspace, demo: str) -> None:
    for params in [
        {"job": demo, "out": "."},
        {"job": demo, "out": "r", "model": "gpt"},
        {"job": demo, "out": "r", "limit": 0},
        {"job": demo, "out": "r", "max_calls": "many"},
        {"job": "demo/targets.csv", "out": "r"},
    ]:
        with pytest.raises(ApiError):
            ws.start_task(params)


def test_a_second_task_on_the_same_directory_is_refused(ws: Workspace, demo: str) -> None:
    release = threading.Event()
    task = Task(
        id="t1",
        kind="match",
        job=demo,
        out="run",
        out_path=str(ws.root / "run"),
        model="offline",
        max_calls=None,
        limit=None,
    )

    async def blocked() -> ops.OpResult:
        while not release.is_set():
            await asyncio.sleep(0.01)
        return ops.OpResult("match")

    ws.tasks.start(task, "match", blocked)
    try:
        with pytest.raises(ApiError) as caught:
            ws.start_task({"job": demo, "out": "run"})
        assert caught.value.status == 409
    finally:
        release.set()
    assert ws.tasks.wait("t1").state == "done"


def test_cancelling_a_task_reports_it_interrupted(tmp_path: Path) -> None:
    manager = TaskManager()
    started = threading.Event()
    task = Task(
        id="t2",
        kind="match",
        job="job.yaml",
        out="run",
        out_path=str(tmp_path / "run"),
        model="offline",
        max_calls=None,
        limit=None,
    )

    async def forever() -> ops.OpResult:
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("not cancelled")

    manager.start(task, "match", forever)
    assert started.wait(5)
    task.cancel()
    finished = manager.wait("t2", timeout=10)
    assert finished.state == "cancelled"
    assert finished.result is not None
    assert finished.result["status"] == "interrupted"
    assert finished.result["exit_code"] == ops.EXIT_INTERRUPTED


def test_task_polling_returns_events_since_a_cursor(ws: Workspace, demo: str) -> None:
    view = ws.start_task({"job": demo, "out": "run"})
    ws.tasks.wait(view["id"])
    all_events = ws.task(view["id"], {})["events"]
    assert len(all_events) == 4
    assert ws.task(view["id"], {"since": "3"})["events"] == all_events[3:]
    assert [t["id"] for t in ws.list_tasks({})["tasks"]] == [view["id"]]
    with pytest.raises(ApiError):
        ws.task("nope", {})


# --- runs -------------------------------------------------------------------------------


@pytest.fixture
def run_dir(ws: Workspace, demo: str) -> str:
    _finish(ws, ws.start_task({"job": demo, "out": "demo/run"}))
    return "demo/run"


def test_run_results_and_explain(ws: Workspace, run_dir: str) -> None:
    run = ws.run({"dir": run_dir})
    assert run["kind"] == "match" and run["inspect"]["counts"]["total"] == 4

    page = ws.results({"dir": run_dir, "limit": "2"})
    assert [r["source_id"] for r in page["data"]["rows"]] == ["s1", "s2"]
    assert page["data"]["next_offset"] == 2
    unmatched = ws.results({"dir": run_dir, "status": "unmatched"})
    assert [r["source_id"] for r in unmatched["data"]["rows"]] == ["s4"]
    with pytest.raises(ApiError):
        ws.results({"dir": run_dir, "status": "great"})

    why = ws.explain({"dir": run_dir, "source_id": "s1"})
    assert why["data"]["decision"]["matched_id"] == "CHEBI:17234"
    attempt = why["data"]["result"]["attempts"][0]
    assert attempt["issued_keys"]["C01"] == "CHEBI:17234"
    missing = ws.explain({"dir": run_dir, "source_id": "zz"})
    assert missing["status"] == "error" and missing["errors"][0]["code"] == "source_not_found"


def test_a_directory_without_a_manifest_is_not_a_run(ws: Workspace, demo: str) -> None:
    with pytest.raises(ApiError) as caught:
        ws.run({"dir": "demo"})
    assert caught.value.code == "run_not_found"


def test_exports_download_each_view(ws: Workspace, run_dir: str) -> None:
    raw = ws.export({"dir": run_dir, "view": "raw"})
    assert isinstance(raw, Download) and raw.filename == "run-raw.csv"
    assert raw.body.decode("utf-8").startswith("source_id,")
    reviewed = ws.export({"dir": run_dir, "view": "reviewed"})
    assert isinstance(reviewed, Download)
    assert b"final_status" in reviewed.body
    history = ws.export({"dir": run_dir, "view": "history"})
    assert isinstance(history, Download) and history.filename.endswith(".jsonl")
    assert len(history.body.decode("utf-8").splitlines()) == 4
    with pytest.raises(ApiError):
        ws.export({"dir": run_dir, "view": "members"})


def test_evaluation_against_gold_labels(ws: Workspace, run_dir: str) -> None:
    gold = _write_gold(ws)
    envelope = ws.evaluate({"dir": run_dir, "gold": gold})
    assert envelope["status"] == "ok"
    metrics = envelope["data"]["report"]["metrics"]
    assert metrics["labelled"] == 4 and metrics["no_match_recall"] == 1.0
    assert "thresholds" in envelope["data"]["report"] and envelope["data"]["text"]


def _uncertain_model(request: LLMRequest) -> str:
    """First candidate, confidence 0.5: between review_floor and accept_at."""
    return json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.5, "decision": "support", "queries": []}
    )


@pytest.fixture
def review_run(ws: Workspace, demo: str) -> str:
    ops.run(ws.root / demo, ws.root / "rev", llm=FakeLLM(handler=_uncertain_model))
    return "rev"


def test_review_rows_and_apply(ws: Workspace, demo: str, review_run: str) -> None:
    sheet = ws.review_rows({"dir": review_run})
    rows = {row["source_id"]: row for row in sheet["data"]["rows"]}
    assert set(rows) == {"s1", "s2", "s3"}
    assert "accept" in sheet["data"]["decisions"]

    applied = ws.review_apply(
        {
            "dir": review_run,
            "job": demo,
            "reviewer": "jan",
            "decisions": [
                {"result_key": rows["s1"]["result_key"], "decision": "accept"},
                {
                    "result_key": rows["s2"]["result_key"],
                    "decision": "replace",
                    "corrected_target_id": "CHEBI:17992",
                    "review_note": "checked",
                },
            ],
        }
    )
    assert applied["status"] == "ok" and applied["counts"] == {"applied": 2, "rejected": 0}

    after = {row["source_id"] for row in ws.review_rows({"dir": review_run})["data"]["rows"]}
    assert after == {"s1", "s2", "s3"}  # the model's answer is kept; review is an overlay
    why = ws.explain({"dir": review_run, "source_id": "s2"})["data"]["review"]
    assert why["final_matched_id"] == "CHEBI:17992" and why["reviewer"] == "jan"
    reviewed = ws.export({"dir": review_run, "view": "reviewed"})
    assert isinstance(reviewed, Download) and b"replace" in reviewed.body


def test_review_apply_refuses_bad_input(ws: Workspace, demo: str, review_run: str) -> None:
    key = ws.review_rows({"dir": review_run})["data"]["rows"][0]["result_key"]
    base = {"dir": review_run, "job": demo, "reviewer": "jan"}
    for decisions, code in [
        ([], "missing_parameter"),
        ([{"result_key": "nope", "decision": "accept"}], "not_reviewable"),
        ([{"result_key": key, "decision": "maybe"}], "invalid_parameter"),
    ]:
        with pytest.raises(ApiError) as caught:
            ws.review_apply({**base, "decisions": decisions})
        assert caught.value.code == code
    with pytest.raises(ApiError):
        ws.review_apply({"dir": review_run, "job": demo, "decisions": [{"result_key": key}]})
    # A replace without a corrected target is refused by the review file reader.
    result = ws.review_apply({**base, "decisions": [{"result_key": key, "decision": "replace"}]})
    assert result["status"] == "error"


# --- clustering -----------------------------------------------------------------------


@pytest.fixture
def cluster_run(ws: Workspace) -> str:
    shutil.copytree(CLUSTER_JOB.parent, ws.root / "cluster")
    done = _finish(ws, ws.start_task({"job": "cluster/job.yaml", "out": "crun"}))
    assert done["result"]["status"] == "ok", done["result"]
    return "crun"


def test_an_offline_cluster_task_groups_near_identical_labels(
    ws: Workspace, cluster_run: str
) -> None:
    run = ws.run({"dir": cluster_run})
    assert run["kind"] == "cluster" and run["job_path"] == "cluster/job.yaml"
    listing = ws.clusters({"dir": cluster_run, "min_size": "2"})
    assert [c["member_ids"] for c in listing["clusters"]] == [["c2", "c3"]]
    assert listing["clusters"][0]["labels"] == ["chocolate, dark", "dark chocolate"]
    every = ws.clusters({"dir": cluster_run})
    assert every["total"] == 7 and every["unresolved"] == []

    member = ws.cluster_member({"dir": cluster_run, "source_id": "c3"})
    assert member["member"]["outcome"] == "assigned"
    assert {d["kind"] for d in member["decisions"]} >= {"select", "verify"}
    with pytest.raises(ApiError):
        ws.cluster_member({"dir": cluster_run, "source_id": "nope"})

    members = ws.export({"dir": cluster_run, "view": "members"})
    assert isinstance(members, Download) and members.body.startswith(b"order_index,")
    with pytest.raises(ApiError):
        ws.export({"dir": cluster_run, "view": "raw"})
    with pytest.raises(ApiError):
        ws.evaluate({"dir": cluster_run, "gold": _write_gold(ws)})


def test_a_cluster_task_takes_no_limit(ws: Workspace) -> None:
    shutil.copytree(CLUSTER_JOB.parent, ws.root / "cluster")
    with pytest.raises(ApiError):
        ws.start_task({"job": "cluster/job.yaml", "out": "c", "limit": 2})


# --- the offline stand-ins ------------------------------------------------------------


def _ask(prompt: str) -> dict[str, Any]:
    reply = offline.lexical_cluster_model(LLMRequest(system="", user=prompt))
    data: dict[str, Any] = json.loads(reply)
    return data


def test_lexical_select_picks_the_overlapping_cluster_or_none() -> None:
    prompt = (
        "Task: select-cluster\n\nrelation\n\n## Record\ndark chocolate\n\n"
        "## Candidate clusters\n[C01] cluster with 1 member:\n  - quinoa\n"
        "[C02] cluster with 2 members:\n  - chocolate, dark\n  - dark chocolate\n\nChoose."
    )
    assert _ask(prompt)["chosen_key"] == "C02"
    assert _ask(prompt)["confidence_score"] == 1.0
    lonely = prompt.replace("dark chocolate\n\n## Candidate", "heart attack\n\n## Candidate")
    assert _ask(lonely)["chosen_key"] is None


def test_lexical_verify_novelty_merge_and_compare() -> None:
    verify = "Task: verify-assignment\n\n## Record\nmilk chocolate\n\n## Proposed cluster\n"
    assert _ask(verify + "cluster with 1 member:\n  - chocolate, milk\n")["decision"] == (
        "equivalent"
    )
    assert _ask(verify + "cluster with 1 member:\n  - quinoa\n")["decision"] == "not_equivalent"
    novelty = "Task: verify-novelty\n\n## Record\nquinoa\n\n## Nearest existing clusters\n"
    assert _ask(novelty + "[C01] cluster with 1 member:\n  - chocolate\n")["novel"] is True
    assert _ask(novelty + "[C01] cluster with 1 member:\n  - quinoa\n")["novel"] is False
    merge = "Task: verify-merge\n\n## Cluster A\ncluster with 1 member:\n  - dark chocolate\n\n"
    assert _ask(merge + "## Cluster B\ncluster with 1 member:\n  - chocolate dark\n")[
        "decision"
    ] == ("same")
    compare = (
        "Task: compare-clusters\n\n## Record\ndark chocolate\n\n## Current cluster\n"
        "cluster with 1 member:\n  - quinoa\n\n## Alternative cluster\n"
        "cluster with 1 member:\n  - dark chocolate\n"
    )
    assert _ask(compare)["preferred"] == "alternative"
    assert "unknown task" in _ask("Task: something-else\n")["explanation"]


def test_overlap_is_jaccard_on_words() -> None:
    assert offline.overlap("Dark chocolate", "chocolate, dark") == 1.0
    assert offline.overlap("chocolate", "milk chocolate") == 0.5
    assert offline.overlap("", "x") == 0.0


# --- routing ----------------------------------------------------------------------------


def test_dispatch_routes_and_refusals(ws: Workspace, demo: str) -> None:
    info = dispatch(ws, "GET", "/api/info", {})
    assert isinstance(info, dict) and info["max_calls_cap"] == 50
    with pytest.raises(ApiError) as caught:
        dispatch(ws, "GET", "/api/nothing", {})
    assert caught.value.status == 404
    with pytest.raises(ApiError) as caught:
        dispatch(ws, "POST", "/api/info", {})
    assert caught.value.status == 405
    view = dispatch(ws, "POST", "/api/tasks", {"job": demo, "out": "r"})
    assert isinstance(view, dict)
    ws.tasks.wait(view["id"])
    polled = dispatch(ws, "GET", f"/api/tasks/{view['id']}", {})
    assert isinstance(polled, dict) and polled["state"] == "done"
    cancelled = dispatch(ws, "POST", f"/api/tasks/{view['id']}/cancel", {})
    assert isinstance(cancelled, dict) and cancelled["state"] == "done"


# --- HTTP -------------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path) -> Iterator[UIServer]:
    root = tmp_path / "ws"
    root.mkdir()
    instance = make_server(str(root), port=0, token="test-token")
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()


def _request(
    server: UIServer,
    path: str,
    *,
    method: str = "GET",
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(server.url.rstrip("/") + path, data=data, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


AUTH = {TOKEN_HEADER: "test-token"}


def test_the_page_carries_the_token_and_a_strict_policy(server: UIServer) -> None:
    status, headers, body = _request(server, "/")
    assert status == 200
    assert b'content="test-token"' in body and b"__XWALK_TOKEN__" not in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    for name in ("app.js", "app.css", "favicon.svg"):
        assert _request(server, f"/static/{name}")[0] == 200
    assert _request(server, "/static/../api.py")[0] == 404
    assert _request(server, "/static/index.html")[0] == 404


def test_the_api_needs_the_token(server: UIServer) -> None:
    assert _request(server, "/api/info")[0] == 403
    assert _request(server, "/api/info", headers={TOKEN_HEADER: "wrong"})[0] == 403
    status, _, body = _request(server, "/api/info", headers=AUTH)
    assert status == 200 and json.loads(body)["root"] == str(server.workspace.root)


def test_a_foreign_host_header_is_refused(server: UIServer) -> None:
    status, _, body = _request(server, "/", headers={"Host": "attacker.example"})
    assert status == 421 and json.loads(body)["error"]["code"] == "bad_host"
    port = server.server_address[1]
    assert _request(server, "/", headers={"Host": f"localhost:{port}"})[0] == 200


def test_post_bodies_must_be_json_objects(server: UIServer) -> None:
    status, _, body = _request(server, "/api/init", method="POST", body=["x"], headers=AUTH)
    assert status == 400
    raw = urllib.request.Request(
        server.url + "api/init", data=b"dest=x", method="POST", headers=AUTH
    )
    raw.add_header("Content-Type", "application/x-www-form-urlencoded")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(raw, timeout=10)
    assert caught.value.code == 415


def test_a_full_round_trip_over_http(server: UIServer) -> None:
    status, _, body = _request(
        server, "/api/init", method="POST", body={"dest": "demo"}, headers=AUTH
    )
    assert status == 200 and json.loads(body)["status"] == "ok"
    status, _, body = _request(
        server,
        "/api/tasks",
        method="POST",
        body={"job": "demo/job.yaml", "out": "demo/run"},
        headers=AUTH,
    )
    task = json.loads(body)
    server.workspace.tasks.wait(task["id"])
    status, _, body = _request(server, "/api/results?dir=demo/run&status=matched", headers=AUTH)
    assert status == 200 and json.loads(body)["data"]["total"] == 3

    # A download is a plain link: the token rides in the query string.
    status, headers, body = _request(server, "/api/export?dir=demo/run&view=raw")
    assert status == 403
    status, headers, body = _request(server, "/api/export?dir=demo/run&view=raw&token=test-token")
    assert status == 200 and headers["Content-Disposition"].startswith("attachment")
    assert body.startswith(b"source_id,")
    # The query-string token works for downloads only.
    assert _request(server, "/api/info?token=test-token")[0] == 403


def test_errors_are_json_without_a_traceback(server: UIServer) -> None:
    status, _, body = _request(server, "/api/run?dir=../x", headers=AUTH)
    assert status == 403
    error = json.loads(body)["error"]
    assert error["code"] == "outside_workspace" and "Traceback" not in error["message"]


# --- CLI --------------------------------------------------------------------------------


def test_cli_ui_refuses_a_missing_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ui", "--root", str(tmp_path / "missing"), "--no-browser", "--port", "0"])
    assert code == ops.EXIT_USAGE
    assert "not a directory" in capsys.readouterr().err


def test_cli_ui_refuses_a_zero_cap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ui", "--root", str(tmp_path), "--no-browser", "--max-calls-cap", "0"])
    assert code == ops.EXIT_USAGE
    assert "max_calls_cap" in capsys.readouterr().err

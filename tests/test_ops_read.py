"""The bounded read operations (task 06): `ops.search` and `ops.list_results`.

Both are called by `xwalk search`/`xwalk results` and by the MCP tools
`search_candidates`/`list_results`, so their bounds are tested here, without `mcp`.
"""

from __future__ import annotations

import json

import pytest

from tests.test_ops import _job_file, _llm
from xwalk import ops
from xwalk.cli.main import main
from xwalk.config import JobSpec


@pytest.fixture
def partial_run(tmp_path):
    """Three of four sources processed (needs_review), one pending."""
    out = tmp_path / "run"
    ops.run(_job_file(tmp_path), out, llm=_llm(score=0.5), limit=3)
    return out


# --- search -----------------------------------------------------------------------------


def test_search_returns_fused_candidates_and_makes_no_model_call(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("search must not build an LLM client")

    monkeypatch.setattr(JobSpec, "build_llm", forbidden)
    result = ops.search(_job_file(tmp_path), "dextrose", index_dir=tmp_path / "idx", limit=3)
    assert result.exit_code == 0
    first = result.data["candidates"][0]
    assert first["rank"] == 1 and first["id"] == "CHEBI:17234"
    assert first["retrievers"] == {"bm25": 1}
    assert first["text"] == "ID: CHEBI:17234 Label: glucose"  # the job's candidate template
    assert result.data["indexes"] == {"bm25": "built"}
    assert result.counts["candidates"] == len(result.data["candidates"]) <= 3


def test_search_opens_the_index_it_built(tmp_path):
    job = _job_file(tmp_path)
    ops.search(job, "glucose", index_dir=tmp_path / "idx")
    again = ops.search(job, "glucose", index_dir=tmp_path / "idx")
    assert again.data["indexes"] == {"bm25": "opened"}


@pytest.mark.parametrize("limit", [0, ops.MAX_SEARCH_LIMIT + 1])
def test_search_refuses_an_unbounded_limit(tmp_path, limit):
    with pytest.raises(ops.OpError) as info:
        ops.search(_job_file(tmp_path), "glucose", index_dir=tmp_path / "idx", limit=limit)
    assert info.value.result.exit_code == 2


def test_search_refuses_an_empty_query(tmp_path):
    with pytest.raises(ops.OpError) as info:
        ops.search(_job_file(tmp_path), "  ", index_dir=tmp_path / "idx")
    assert info.value.result.errors[0].code == "usage"


# --- list_results -------------------------------------------------------------------


def test_pages_cover_the_current_view_once_in_source_order(partial_run):
    seen = []
    offset: int | None = 0
    while offset is not None:
        page = ops.list_results(partial_run, offset=offset, limit=3)
        assert len(page.data["rows"]) <= 3
        seen += page.data["rows"]
        offset = page.data["next_offset"]
    assert [row["source_id"] for row in seen] == ["s1", "s2", "s3", "s4"]
    assert [row["status"] for row in seen].count("pending") == 1
    pending = next(row for row in seen if row["status"] == "pending")
    assert pending["matched_id"] is None and pending["result_key"] is None
    assert page.data["total"] == 4 and page.run is not None
    assert page.run["run_state"] == "partial"


def test_the_status_filter_counts_only_matching_rows(partial_run):
    page = ops.list_results(partial_run, status="pending")
    assert page.data["total"] == 1 and page.data["next_offset"] is None
    assert page.data["rows"][0]["source_id"] == "s4"


def test_rows_agree_with_explain(partial_run):
    row = ops.list_results(partial_run, limit=1).data["rows"][0]
    decision = ops.explain(partial_run, row["source_id"]).data["decision"]
    assert (row["matched_id"], row["confidence"], row["status"], row["result_key"]) == (
        decision["matched_id"],
        decision["confidence"],
        decision["status"],
        decision["result_key"],
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"limit": ops.MAX_PAGE_SIZE + 1}, {"offset": -1}, {"status": "nope"}],
)
def test_list_results_refuses_bad_paging(partial_run, kwargs):
    with pytest.raises(ops.OpError) as info:
        ops.list_results(partial_run, **kwargs)
    assert info.value.result.exit_code == 2


def test_list_results_on_a_missing_run_is_a_usage_error(tmp_path):
    with pytest.raises(ops.OpError) as info:
        ops.list_results(tmp_path / "nowhere")
    assert info.value.code == "run_not_found"


# --- the CLI formats the same operations --------------------------------------------


def test_cli_results_json_is_the_operation_envelope(partial_run, capsys):
    code = main(["results", "--run", str(partial_run), "--limit", "2", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert code == 0
    assert envelope == ops.list_results(partial_run, limit=2).envelope()
    assert envelope["data"]["next_offset"] == 2


def test_cli_search_json_and_bounds(tmp_path, capsys):
    job = _job_file(tmp_path)
    code = main(["search", "glucose", "--job", str(job), "--index", str(tmp_path / "i"), "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert code == 0 and envelope["data"]["candidates"][0]["id"] == "CHEBI:17234"
    code = main(["search", "x", "--job", str(job), "--index", str(tmp_path / "i"), "--limit", "0"])
    assert code == 2


def test_failure_result_maps_exceptions_like_the_cli(tmp_path):
    from xwalk._extras import MissingExtra

    assert ops.failure_result("x", MissingExtra("no")).errors[0].code == "missing_extra"
    assert ops.failure_result("x", OSError("disk")).exit_code == 3
    assert ops.failure_result("x", RuntimeError("bug")).errors[0].message == "RuntimeError: bug"

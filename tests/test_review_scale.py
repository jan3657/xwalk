"""Reading one record must cost one record: the review loop at scale.

The workbench calls `explain` for every record it shows and records one decision at a
time. These tests pin that those calls read the rows of the record they are about, not
every result in the run, by counting stored results decoded; and that the per-source
reads answer exactly what the full scans answered before (the reference below is the
previous implementation of `ops.explain`, kept verbatim as an oracle).

`scripts/bench_review_step.py` times the same routes on a generated ~2k-row run.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

import xwalk.ledger as ledger_module
from tests.test_ops import _job_file, _llm
from xwalk import ops
from xwalk.ledger import Ledger
from xwalk.review import adjudicated, review_history
from xwalk.ui.api import Workspace, dispatch


def _reference_explain(run_dir: Path, source_id: str) -> tuple[dict[str, Any], list[str]]:
    """`ops.explain(full=True)` as it was: full scans of results, history and reviews."""
    from xwalk.batch import usage_to_dict
    from xwalk.ops import _attempt_summary, _open_run
    from xwalk.serde import result_to_dict

    view, ledger = _open_run(run_dir, "explain")
    try:
        fp = view.run_fingerprint
        entry = next((e for e in ledger.iter_current(fp) if e.source_id == source_id), None)
        history = [h for h in ledger.iter_history(fp) if h.result.source_id == source_id]
        applied = {
            str(h.review["result_key"]): h.review
            for h in review_history(ledger, fp)
            if h.state == "applied"
        }
        overlay = {row.result_key: row for row in adjudicated(ledger, fp)}
    finally:
        ledger.close()
    current = entry.result if entry is not None else None
    data: dict[str, Any] = {
        "source_id": source_id,
        "in_current_snapshot": entry is not None,
        "library_version": view.manifest.get("library_version"),
        "status": "pending" if entry is not None and current is None else None,
        "decision": None,
        "attempts": [],
        "review": None,
        "history": [
            {
                "revision": h.revision,
                "current": h.current,
                "source_hash": h.result.source_hash,
                "status": h.result.status.value,
                "matched_id": h.result.matched_id,
                "reason": h.result.reason.value,
            }
            for h in history
        ],
    }
    warnings: list[str] = []
    if current is not None:
        assert entry is not None
        data["status"] = current.status.value
        data["decision"] = {
            "result_key": current.result_key,
            "revision": entry.revision,
            "matched_id": current.matched_id,
            "confidence": current.confidence,
            "status": current.status.value,
            "reason": current.reason.value,
            "explanation": current.explanation,
            "usage": usage_to_dict(current.usage),
        }
        data["attempts"] = [_attempt_summary(a) for a in current.attempts]
        review = applied.get(current.result_key)
        row = overlay.get(current.result_key)
        if review is not None and row is not None:
            data["review"] = {
                "decision": review["decision"],
                "corrected_target_id": review.get("corrected_target_id"),
                "reviewer": review["reviewer"],
                "review_note": review["review_note"],
                "reviewed_at": review["reviewed_at"],
                "final_matched_id": row.final_target_id,
                "final_status": row.final_status.value,
            }
        data["result"] = result_to_dict(current)
    elif entry is None:
        warnings.append("removed_source")
    return data, warnings


def _decide(run_dir: Path, job: Path, tmp: Path, source_id: str, decision: str, **extra: str):
    sheet = tmp / "review.csv"
    ops.review_export(run_dir, sheet)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    row = next(r for r in rows if r["source_id"] == source_id)
    row.update(decision=decision, reviewer="jan", **extra)
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerow(row)
    assert ops.review_apply(run_dir, sheet, job).counts["applied"] == 1


@pytest.fixture
def busy_run(tmp_path: Path) -> Path:
    """Every state a record can be in: reviewed, re-reviewed, stale, edited, removed."""
    job = _job_file(tmp_path)
    out = tmp_path / "run"
    ops.run(job, out, llm=_llm(score=0.5))
    _decide(out, job, tmp_path, "s2", "accept")
    ops.run(job, out, llm=_llm(score=0.5), resume=False)  # recomputed: the review is stale
    _decide(out, job, tmp_path, "s1", "replace", corrected_target_id="CHEBI:15903")
    _decide(out, job, tmp_path, "s1", "defer")  # supersedes the replace
    _decide(out, job, tmp_path, "s2", "no_match")
    sources = tmp_path / "sources_tiny.csv"
    lines = sources.read_text(encoding="utf-8").splitlines()
    edited = [
        line.replace("table sugar", "cane sugar") for line in lines if not line.startswith("s4,")
    ]
    sources.write_text("\n".join(edited) + "\n", encoding="utf-8")
    ops.run(job, out, llm=_llm(score=0.5))  # s3 edited, s4 removed
    return out


@pytest.mark.parametrize("source_id", ["s1", "s2", "s3", "s4"])
def test_explain_answers_what_the_full_scans_answered(busy_run: Path, source_id: str) -> None:
    expected, warnings = _reference_explain(busy_run, source_id)
    result = ops.explain(busy_run, source_id, full=True)
    assert result.data == expected
    assert [w.code for w in result.warnings] == warnings


def test_the_fixture_covers_every_state(busy_run: Path) -> None:
    by_source = {s: _reference_explain(busy_run, s)[0] for s in ("s1", "s2", "s3", "s4")}
    assert by_source["s1"]["review"]["decision"] == "defer"
    assert by_source["s2"]["review"]["final_status"] == "unmatched"
    assert len(by_source["s3"]["history"]) == 3  # two runs on the old text, one on the new
    assert by_source["s4"]["in_current_snapshot"] is False and by_source["s4"]["history"]
    assert any(not h["current"] for h in by_source["s1"]["history"])


def _fp(run_dir: Path) -> str:
    import json

    return str(json.loads((run_dir / "manifest.json").read_text())["run_fingerprint"])


@pytest.mark.parametrize("source_id", ["s1", "s2", "s3", "s4", "zz"])
def test_per_source_ledger_reads_agree_with_the_full_scans(busy_run: Path, source_id: str) -> None:
    led = Ledger.open(busy_run / "ledger.sqlite")
    try:
        fp = _fp(busy_run)
        scanned = next((e for e in led.iter_current(fp) if e.source_id == source_id), None)
        assert led.current_entry(fp, source_id) == scanned
        assert list(led.iter_source_history(fp, source_id)) == [
            h for h in led.iter_history(fp) if h.result.source_id == source_id
        ]
        keys = {str(r["result_key"]) for r in led.iter_reviews(fp)}
        for key in keys:
            assert list(led.iter_reviews_for(fp, key)) == [
                r for r in led.iter_reviews(fp) if r["result_key"] == key
            ]
    finally:
        led.close()


def test_the_reviewed_overlay_from_summaries_matches_adjudicated(busy_run: Path) -> None:
    from xwalk.review import adjudicated_reviewed

    led = Ledger.open(busy_run / "ledger.sqlite")
    try:
        fp = _fp(busy_run)
        expected = [row for row in adjudicated(led, fp) if row.reviewer is not None]
        assert expected  # the fixture has applied decisions
        assert list(adjudicated_reviewed(led, fp)) == expected
    finally:
        led.close()


class _Decodes:
    """Counts stored results decoded from the ledger."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.n = 0
        original = ledger_module.result_from_dict

        def counting(data: Any) -> Any:
            self.n += 1
            return original(data)

        monkeypatch.setattr(ledger_module, "result_from_dict", counting)


def test_explain_decodes_only_the_rows_of_its_own_source(
    busy_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decodes = _Decodes(monkeypatch)
    data = ops.explain(busy_run, "s1", full=True).data
    # The current result and its history rows; the other sources are never decoded.
    assert decodes.n <= 1 + len(data["history"])


@pytest.fixture
def ws_run(tmp_path: Path) -> tuple[Workspace, str, str]:
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace(root, max_calls_cap=50)
    job = str(ws.init({"dest": "demo"})["data"]["job_path"])
    ops.run(root / job, root / "rev", llm=_llm(score=0.5))
    ws._remember_job(root / "rev", root / job)
    return ws, job, "rev"


def test_recording_one_decision_decodes_only_that_result(
    ws_run: tuple[Workspace, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, job, run = ws_run
    rows = dispatch(ws, "GET", "/api/review", {"dir": run})["data"]["rows"]
    assert len(rows) > 2
    decodes = _Decodes(monkeypatch)
    envelope = dispatch(
        ws,
        "POST",
        "/api/review",
        {
            "dir": run,
            "job": job,
            "reviewer": "jan",
            "decisions": [{"result_key": rows[0]["result_key"], "decision": "accept"}],
        },
    )
    assert envelope["status"] == "ok", envelope["errors"]
    # Look the result up and validate it: a couple of decodes, not one per row.
    assert decodes.n <= 3


def test_a_key_that_is_not_under_review_is_still_refused(
    ws_run: tuple[Workspace, str, str],
) -> None:
    ws, job, run = ws_run
    from xwalk.ui.api import ApiError

    with pytest.raises(ApiError) as info:
        dispatch(
            ws,
            "POST",
            "/api/review",
            {
                "dir": run,
                "job": job,
                "reviewer": "jan",
                "decisions": [{"result_key": "no-such-key", "decision": "accept"}],
            },
        )
    assert info.value.status == 409 and info.value.code == "not_reviewable"


def test_the_review_overlay_route_decodes_no_results(
    ws_run: tuple[Workspace, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, job, run = ws_run
    rows = dispatch(ws, "GET", "/api/review", {"dir": run})["data"]["rows"]
    dispatch(
        ws,
        "POST",
        "/api/review",
        {
            "dir": run,
            "job": job,
            "reviewer": "jan",
            "decisions": [{"result_key": rows[0]["result_key"], "decision": "reject"}],
        },
    )
    led = Ledger.open(Path(ws.root) / run / "ledger.sqlite")
    try:
        fp = _fingerprint(ws, run)
        reference = {
            row.result_key: {
                "source_id": row.source_id,
                "final_matched_id": row.final_target_id,
                "final_status": row.final_status.value,
            }
            for row in adjudicated(led, fp)
            if row.reviewer is not None
        }
    finally:
        led.close()
    decodes = _Decodes(monkeypatch)
    data = dispatch(ws, "GET", "/api/reviews", {"dir": run})["data"]
    assert decodes.n == 0
    assert (
        data["final"] == reference
        and data["final"][rows[0]["result_key"]]["final_status"] == "unmatched"
    )


def _fingerprint(ws: Workspace, run: str) -> str:
    import json

    return str(json.loads((Path(ws.root) / run / "manifest.json").read_text())["run_fingerprint"])

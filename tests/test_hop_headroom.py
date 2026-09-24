"""The offline go/no-go for the graph hop (A4): how many retrieval misses sit one hop away."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from xwalk.evaluate.gold import GoldSet

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hop_headroom.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hop_headroom", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _cand(rid: str, parents: list[str]) -> dict[str, Any]:
    return {"record": {"id": rid, "fields": {"label": rid, "parents": parents}}}


def _row(sid: str, matched: str | None, cands: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_id": sid,
        "matched_id": matched,
        "status": "matched" if matched else "no_match",
        "attempts": [{"candidates": cands, "candidates_truncated": False}],
    }


ROWS = [
    # A hit: the gold id was retrieved (and chosen).
    _row("hit", "G1", [_cand("G1", ["P1"]), _cand("X", [])]),
    # Parent case: gold P2 is the parent of retrieved C2.
    _row("parent", "C2", [_cand("C2", ["P2"])]),
    # Child case: gold K3 has retrieved C3 among its parents (from the target).
    _row("child", None, [_cand("C3", ["ROOT"])]),
    # A miss nowhere near the gold.
    _row("far", "Y", [_cand("Y", ["ROOT"])]),
    # Explicit no-match and unlabelled rows are not counted.
    _row("nomatch", None, [_cand("Z", [])]),
    _row("unlabelled", None, [_cand("Z", [])]),
]
GOLD = GoldSet(
    {
        "hit": frozenset({"G1"}),
        "parent": frozenset({"P2"}),
        "child": frozenset({"K3"}),
        "far": frozenset({"F4"}),
        "nomatch": frozenset(),
    }
)
TARGET_PARENTS = {"K3": ["C3"], "F4": ["ELSEWHERE"], "P2": ["ROOT"], "G1": ["P1"]}


def test_headroom_counts_parent_and_child_misses_only() -> None:
    report = _script().headroom(ROWS, GOLD, TARGET_PARENTS, "parents")
    assert report.labelled == 4
    assert report.retrieval_misses == 3
    assert report.status_misses == 3  # parent, child, far
    assert [(q.source_id, q.gold_id, q.via_id, q.relation) for q in report.qualifying] == [
        ("parent", "P2", "C2", "parent"),
        ("child", "K3", "C3", "child"),
    ]
    assert report.qualifying_rows == 2
    assert report.share == pytest.approx(2 / 3)


def test_the_cli_reads_a_ledger_and_a_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8"
    )
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "source_id,gold_ids\nhit,G1\nparent,P2\nchild,K3\nfar,F4\nnomatch,\n", encoding="utf-8"
    )
    (tmp_path / "targets.csv").write_text(
        "id,label,parents\nK3,k3,C3\nF4,f4,ELSEWHERE\nP2,p2,ROOT\nG1,g1,P1\n", encoding="utf-8"
    )
    (tmp_path / "source.csv").write_text("id,name\nhit,x\n", encoding="utf-8")
    job = tmp_path / "job.yaml"
    job.write_text(
        "name: t\n"
        "templates:\n  queries: ['{{ name }}']\n  context: '{{ name }}'\n"
        "  doc: '{{ label }}'\n  candidate: '{{ label }}'\n"
        "target:\n  kind: csv\n  path: targets.csv\n  id_column: id\n"
        "  multivalue_columns: [parents]\n"
        "source:\n  kind: csv\n  path: source.csv\n  id_column: id\n"
        "retrievers:\n  - kind: bm25\n    name: bm25\n    limit: 5\n"
        "decider:\n  model: m\n"
        "prompts:\n  slots: slots.yaml\n",
        encoding="utf-8",
    )
    assert _script().main([str(run), str(gold), "parents", "--job", str(job)]) == 0
    out = capsys.readouterr().out
    assert "labelled: 4" in out
    assert "retrieval misses: 3" in out
    assert "qualifying: 2" in out
    assert "share: 0.667" in out
    assert "child\tK3\tC3\tchild" in out

#!/usr/bin/env python
"""Offline headroom of a one-hop expansion over the target hierarchy (A4 go/no-go).

Reads a finished decider run's `results.jsonl` and a gold file. A labelled row (gold not
"no match") is a *retrieval miss* when no gold id is among its first attempt's candidates:
the only kind of miss a hop can repair. A miss *qualifies* when a gold id is a parent of a
retrieved candidate (the candidate's FIELD holds it) or a child of one (the gold record's
own FIELD, read from the job's target, holds the candidate id). Prints the counts, the
share (qualifying / retrieval misses) and every qualifying (source, gold, via, relation).
No LLM, no index, no API key.

Usage: scripts/hop_headroom.py RUN_DIR GOLD FIELD --job JOB
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xwalk.config import load_job
from xwalk.evaluate.gold import GoldSet, load_gold_csv


@dataclass(frozen=True)
class Qualifying:
    source_id: str
    gold_id: str
    via_id: str
    relation: str  # "parent" (gold is the candidate's parent) or "child"


@dataclass(frozen=True)
class Headroom:
    labelled: int
    retrieval_misses: int
    status_misses: int
    qualifying: tuple[Qualifying, ...]

    @property
    def qualifying_rows(self) -> int:
        return len({q.source_id for q in self.qualifying})

    @property
    def share(self) -> float:
        return self.qualifying_rows / self.retrieval_misses if self.retrieval_misses else 0.0


def _ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()]


def headroom(
    results: Iterable[Mapping[str, Any]],
    gold: GoldSet,
    target_parents: Mapping[str, Sequence[str]],
    field: str,
) -> Headroom:
    """Count the labelled rows whose missed gold sits one hop from a retrieved candidate."""
    labelled = misses = status_misses = 0
    found: list[Qualifying] = []
    for row in results:
        sid = row["source_id"]
        expected = gold.get(sid)
        if not expected:  # unlabelled (None) or an explicit no-match
            continue
        labelled += 1
        if row.get("matched_id") not in expected:
            status_misses += 1
        attempts = row.get("attempts") or []
        cands = (attempts[0].get("candidates") or []) if attempts else []
        records = [c["record"] for c in cands]
        retrieved = [r["id"] for r in records]
        if expected & set(retrieved):
            continue
        misses += 1
        for gid in sorted(expected):
            gold_parents = set(_ids(target_parents.get(gid)))
            for rec in records:
                if gid in _ids(rec.get("fields", {}).get(field)):
                    found.append(Qualifying(sid, gid, rec["id"], "parent"))
                elif rec["id"] in gold_parents:
                    found.append(Qualifying(sid, gid, rec["id"], "child"))
    return Headroom(labelled, misses, status_misses, tuple(found))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("run", type=Path, help="run directory holding results.jsonl")
    parser.add_argument("gold", type=Path, help="gold CSV (source_id,gold_ids)")
    parser.add_argument("field", help="the target field holding parent ids, e.g. parents")
    parser.add_argument("--job", type=Path, required=True, help="job YAML (loads the target)")
    args = parser.parse_args(argv)

    job = load_job(args.job)
    target_parents = {r.id: _ids(r.fields.get(args.field)) for r in job.build_target_records()}
    with (args.run / "results.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = headroom(rows, load_gold_csv(args.gold), target_parents, args.field)
    print(f"labelled: {report.labelled}")
    print(f"retrieval misses: {report.retrieval_misses}")
    print(f"status misses: {report.status_misses}")
    print(f"qualifying: {report.qualifying_rows}")
    print(f"share: {report.share:.3f}")
    for q in report.qualifying:
        print(f"{q.source_id}\t{q.gold_id}\t{q.via_id}\t{q.relation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

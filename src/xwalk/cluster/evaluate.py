"""Evaluating a finished clustering run against a gold partition (`xwalk cluster eval`).

Reads the run's exports (`members.csv`, the exported state revision) and its store
(opened read-only) and nothing else: no model call, no retriever, no write (the
"evaluation calls nothing" invariant). Scores come from `xwalk.evaluate.clustering`, the
same code the benchmark uses: pairwise and B-cubed precision/recall/F1 under two
conventions (`as_singletons`, `accepted_only`), false merges, splits and outcome counts.

One diagnostic needs the store: **gold pairs co-exposed**, the share of gold
co-cluster pairs whose two records appeared together in at least one prompt the judge
was sent (the subject record, or the shown members of the subject cluster, plus the
member lines of every cluster shown, exactly as rendered: at most `member_evidence` per
cluster, the excluded member left out). It is **not an upper bound** on pairwise recall:
a record joins a cluster on the evidence of its first members only, and a merge links
members neither prompt listed, so pairs never shown together can still be co-clustered.
It answers "how often did the judge get to see both records", which separates search
misses from judge misses when recall is low.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from xwalk.cluster.store import ClusterStore, ClusterStoreError
from xwalk.evaluate.clustering import evaluate_clustering, read_assignments

CO_EXPOSURE_NOTE = (
    "gold pairs whose two records appeared together in at least one judge prompt "
    "(member lines as rendered, at most member_evidence per cluster). This is not an "
    "upper bound on recall: a record joins on the evidence of a cluster's first members, "
    "and merges and consolidation link records no prompt showed together."
)


class ClusterEvalError(ValueError):
    """The run or the gold file cannot be evaluated. `code` is stable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def read_cluster_manifest(run_dir: Path) -> dict[str, Any] | None:
    """The manifest of a clustering run directory, or None if `run_dir` is not one."""
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("operation") != "cluster":
        return None
    return data


def _member_evidence(manifest: Mapping[str, Any]) -> int:
    components = manifest.get("fingerprint_components") or {}
    pool = components.get("pool") if isinstance(components, dict) else None
    value = pool.get("member_evidence") if isinstance(pool, dict) else None
    if not isinstance(value, int) or value < 1:
        raise ClusterEvalError(
            "manifest_invalid", "manifest.json does not record pool.member_evidence"
        )
    return value


def co_exposure(
    store: ClusterStore, gold: Mapping[str, str], member_evidence: int
) -> dict[str, Any]:
    """Gold co-cluster pairs that appeared together in at least one judge prompt."""
    sources = {sid for _, sid, _, _ in store.sources()}
    by_concept: dict[str, list[str]] = defaultdict(list)
    for item, concept in gold.items():
        if item in sources:
            by_concept[concept].append(item)
    gold_pairs = sum(len(g) * (len(g) - 1) // 2 for g in by_concept.values())

    members_at: dict[tuple[str, int], list[str]] = {}
    exposed: set[tuple[str, str]] = set()
    prompts = 0
    for decision in store.decisions():
        if decision["prompt"] is None:  # a model-free mint: nothing was shown
            continue
        prompts += 1
        items: set[str] = set()
        if decision["subject_id"] in sources:
            items.add(decision["subject_id"])
        for entry in decision["shown"]:
            key = (entry["cluster_id"], int(entry["revision"]))
            if key not in members_at:
                members_at[key] = list(store.revision(*key)["members"])
            visible = [m for m in members_at[key] if m != entry.get("exclude")]
            items.update(visible[:member_evidence])
        groups: dict[str, list[str]] = defaultdict(list)
        for item in items:
            if item in gold:
                groups[gold[item]].append(item)
        for group in groups.values():
            exposed.update(combinations(sorted(group), 2))
    shown = len(exposed)
    return {
        "gold_pairs": gold_pairs,
        "gold_pairs_co_exposed": shown,
        "co_exposed_share": None if gold_pairs == 0 else round(shown / gold_pairs, 4),
        "prompts_read": prompts,
        "member_evidence": member_evidence,
        "note": CO_EXPOSURE_NOTE,
    }


def _usage(store: ClusterStore) -> dict[str, Any]:
    """Model usage summed over every invocation of the run, as each recorded it.

    An invocation that never finished (a killed process) recorded no usage; it is
    counted in `invocations_without_usage`, and the totals are then a lower bound."""
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "unknown_calls": 0}
    invocations = store.invocations()
    missing = 0
    for invocation in invocations:
        usage = invocation.get("usage")
        if not usage:
            missing += 1
            continue
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return {**totals, "invocations": len(invocations), "invocations_without_usage": missing}


def evaluate_cluster_run(run_dir: str | Path, gold_path: str | Path) -> dict[str, Any]:
    """Score the exported partition of `run_dir` against the gold file `gold_path`
    (`item_id, cluster_id` rows, CSV with a header or JSONL; `source_id` is accepted
    for `item_id`). Raises `ClusterEvalError` for an unusable run or gold file."""
    run = Path(run_dir)
    if not run.is_dir():
        raise ClusterEvalError("run_not_found", f"{run} is not a directory")
    manifest = read_cluster_manifest(run)
    if manifest is None:
        raise ClusterEvalError(
            "not_a_cluster_run",
            f"{run} is not an xwalk clustering run directory (no manifest.json with "
            "operation: cluster); matching runs are evaluated with `xwalk eval`",
        )
    members = run / "members.csv"
    if not members.is_file():
        raise ClusterEvalError("run_not_found", f"{run} has no members.csv export")
    try:
        gold, _ = read_assignments(gold_path)
    except (OSError, ValueError) as exc:
        raise ClusterEvalError("gold_unreadable", f"--gold: {exc}") from None
    if not gold:
        raise ClusterEvalError("gold_unreadable", f"--gold: {gold_path} has no rows")
    clusters, outcomes = read_assignments(members)
    member_evidence = _member_evidence(manifest)

    store = ClusterStore.open_readonly(run / "cluster.sqlite")
    try:
        exposure = co_exposure(store, gold, member_evidence)
        usage = _usage(store)
    except sqlite3.DatabaseError as exc:  # corruption past the header
        raise ClusterStoreError(f"{store.path}: unreadable clustering store ({exc})") from None
    finally:
        store.close()

    scores = evaluate_clustering(clusters, outcomes, gold)
    return {
        "run": str(run),
        "gold": str(gold_path),
        "run_fingerprint": manifest.get("run_fingerprint"),
        "run_state": manifest.get("run_state"),
        "stop_reason": manifest.get("stop_reason"),
        "exported_revision": manifest.get("exported_revision"),
        "gold_items": len(gold),
        "run_items": len(clusters),
        "scores": scores,
        "exposure": exposure,
        "usage": usage,
        "calls_per_item": None if not clusters else round(usage["calls"] / len(clusters), 3),
        "experimental": True,
    }


__all__ = [
    "CO_EXPOSURE_NOTE",
    "ClusterEvalError",
    "co_exposure",
    "evaluate_cluster_run",
    "read_cluster_manifest",
]

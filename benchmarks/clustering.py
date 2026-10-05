"""Clustering track: metrics over a predicted flat partition.

Independent of any clustering implementation: predictions arrive as a file of
``item_id, cluster_id[, outcome]`` rows (CSV with a header, or JSONL objects with those
keys), so the same evaluation applies to xwalk's clustering (task 04) and to any
baseline. ``outcome`` follows CONTRACTS.md section 10: ``assigned``, ``singleton``,
``needs_review`` or ``failed``. Only ``assigned``/``singleton`` rows form the accepted
partition; the other two are in no accepted cluster.

Two conventions are reported, because both questions are asked:

- ``as_singletons``: every item not in the accepted partition (review, failed, missing)
  is scored as its own singleton cluster. Comparable across systems with different
  review rates; a system cannot score well by sending everything to review.
- ``accepted_only``: only accepted items, against gold restricted to them. The quality of
  what was automated.

Metrics: pairwise precision/recall/F1 (over co-clustered pairs), B-cubed
precision/recall/F1 (per item, averaged), false merges (predicted clusters mixing gold
clusters, and the wrong pairs they create), splits (gold clusters spread over several
predicted clusters, and the missing pairs), singleton behaviour, review coverage, and --
given several predictions for the same items, e.g. different input orders -- order
sensitivity.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

ACCEPTED = frozenset({"assigned", "singleton"})
OUTCOMES = ("assigned", "singleton", "needs_review", "failed")

Assignment = dict[str, str]  # item_id -> cluster_id


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else round(num / den, 4)


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 0.0 if p + r == 0 else round(2 * p * r / (p + r), 4)


def read_assignments(path: str | Path) -> tuple[Assignment, dict[str, str]]:
    """Read ``item_id, cluster_id[, outcome]``. Returns (clusters, outcomes).

    A row without an outcome is ``assigned`` if its cluster has other members and
    ``singleton`` otherwise. Duplicate item ids are an error.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    clusters: Assignment = {}
    outcomes: dict[str, str] = {}
    for n, row in enumerate(rows, start=1):
        item = str(row.get("item_id") or "").strip()
        if not item:
            raise ValueError(f"{path}: row {n} has no item_id")
        if item in clusters:
            raise ValueError(f"{path}: row {n} repeats item_id {item!r}")
        clusters[item] = str(row.get("cluster_id") or "").strip()
        outcome = str(row.get("outcome") or "").strip()
        if outcome and outcome not in OUTCOMES:
            raise ValueError(f"{path}: row {n} has unknown outcome {outcome!r}")
        outcomes[item] = outcome
    sizes = _sizes(clusters)
    for item, outcome in outcomes.items():
        if not outcome:
            outcomes[item] = "assigned" if sizes[clusters[item]] > 1 else "singleton"
    return clusters, outcomes


def write_assignments(
    path: str | Path, clusters: Mapping[str, str], outcomes: Mapping[str, str] | None = None
) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "cluster_id", "outcome"])
        sizes = _sizes(clusters)
        for item, cluster in clusters.items():
            outcome = (outcomes or {}).get(item) or (
                "assigned" if sizes[cluster] > 1 else "singleton"
            )
            writer.writerow([item, cluster, outcome])


def _sizes(clusters: Mapping[str, str]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for cluster in clusters.values():
        sizes[cluster] += 1
    return sizes


def _groups(clusters: Mapping[str, str]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item, cluster in clusters.items():
        groups[cluster].add(item)
    return groups


def _pairs(clusters: Mapping[str, str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for members in _groups(clusters).values():
        out.update(combinations(sorted(members), 2))
    return out


def partition_scores(pred: Mapping[str, str], gold: Mapping[str, str]) -> dict[str, Any]:
    """Pairwise and B-cubed scores of `pred` against `gold` over the same item set."""
    if set(pred) != set(gold):
        raise ValueError("prediction and gold cover different items")
    pred_pairs, gold_pairs = _pairs(pred), _pairs(gold)
    tp = len(pred_pairs & gold_pairs)
    pp = _ratio(tp, len(pred_pairs))
    pr = _ratio(tp, len(gold_pairs))

    pred_groups, gold_groups = _groups(pred), _groups(gold)
    b_p = b_r = 0.0
    for item in pred:
        mine = pred_groups[pred[item]]
        truth = gold_groups[gold[item]]
        overlap = len(mine & truth)
        b_p += overlap / len(mine)
        b_r += overlap / len(truth)
    n = len(pred)
    bp = _ratio(b_p, n)
    br = _ratio(b_r, n)

    merged = [c for c, members in pred_groups.items() if len({gold[i] for i in members}) > 1]
    split = [g for g, members in gold_groups.items() if len({pred[i] for i in members}) > 1]
    pred_single = {i for i in pred if len(pred_groups[pred[i]]) == 1}
    gold_single = {i for i in gold if len(gold_groups[gold[i]]) == 1}
    return {
        "items": n,
        "predicted_clusters": len(pred_groups),
        "gold_clusters": len(gold_groups),
        "pairwise_precision": pp,
        "pairwise_recall": pr,
        "pairwise_f1": _f1(pp, pr),
        "bcubed_precision": bp,
        "bcubed_recall": br,
        "bcubed_f1": _f1(bp, br),
        "false_merge_clusters": len(merged),
        "false_merge_pairs": len(pred_pairs - gold_pairs),
        "split_gold_clusters": len(split),
        "missed_pairs": len(gold_pairs - pred_pairs),
        "predicted_singletons": len(pred_single),
        "gold_singletons": len(gold_single),
        "correct_singletons": len(pred_single & gold_single),
        "wrong_singletons": len(pred_single - gold_single),
    }


def evaluate_clustering(
    clusters: Mapping[str, str],
    outcomes: Mapping[str, str],
    gold: Mapping[str, str],
) -> dict[str, Any]:
    """Score one prediction under both conventions. Items absent from the prediction
    count as ``missing`` (in no accepted cluster); predicted items absent from gold are
    unlabelled and ignored."""
    items = list(gold)
    accepted = [i for i in items if i in clusters and outcomes.get(i) in ACCEPTED]
    as_singletons = {i: (f"c:{clusters[i]}" if i in accepted else f"s:{i}") for i in items}
    counts = {o: sum(1 for i in items if outcomes.get(i) == o) for o in OUTCOMES}
    counts["missing"] = sum(1 for i in items if i not in clusters)
    return {
        "outcome_counts": counts,
        "review_coverage": _ratio(counts["needs_review"], len(items)),
        "accepted_coverage": _ratio(len(accepted), len(items)),
        "unlabelled_predicted_items": sum(1 for i in clusters if i not in gold),
        "as_singletons": partition_scores(as_singletons, dict(gold)),
        "accepted_only": partition_scores(
            {i: clusters[i] for i in accepted}, {i: gold[i] for i in accepted}
        )
        if accepted
        else None,
    }


def order_sensitivity(predictions: Sequence[Mapping[str, str]]) -> dict[str, Any] | None:
    """Agreement between predictions of the same items (e.g. different input orders).

    Reports the pairwise F1 of every prediction against every other (1.0 = identical
    partitions) and the share of items whose co-membership set differs in any run.
    """
    if len(predictions) < 2:
        return None
    items = sorted(set.intersection(*(set(p) for p in predictions)))
    restricted = [{i: p[i] for i in items} for p in predictions]
    agreements = []
    for a, b in combinations(restricted, 2):
        agreements.append(partition_scores(a, b)["pairwise_f1"])
    groups = [_groups(p) for p in restricted]
    unstable = sum(
        1
        for i in items
        if len({frozenset(g[p[i]]) for g, p in zip(groups, restricted, strict=True)}) > 1
    )
    known = [a for a in agreements if a is not None]
    return {
        "runs": len(predictions),
        "items": len(items),
        "min_pairwise_f1_between_runs": min(known) if known else None,
        "mean_pairwise_f1_between_runs": round(sum(known) / len(known), 4) if known else None,
        "items_with_unstable_membership": unstable,
    }

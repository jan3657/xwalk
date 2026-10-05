"""Exports, regenerated from the store.

A finished run exports its *selected* state revision (CLUSTERING_DECISIONS.md: the final
fixpoint when refinement converged, otherwise the revision with the fewest unresolved
sources, ties earliest), never a transient state. A run that stopped early (aborted or
interrupted) exports its latest committed state, and sources it never reached are
`pending`.

- `members.csv`: one row per source in snapshot order -- exactly once.
- `clusters.csv`: one row per accepted cluster, with its members.
- `unresolved.csv`: every source outside the accepted partition, with its best proposal.
- `decisions.jsonl`: every decision ever recorded (history, including decisions whose
  effects a later revision superseded), with retrieved and shown clusters.

Exported outcomes: `assigned`, `singleton` (a seed whose cluster has no other member;
not a verified equivalence), `needs_review` (deferred), `failed`, `pending`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xwalk.cluster.engine import ACCEPTED, DEFERRED, SEED
from xwalk.cluster.store import ClusterStore

MEMBER_COLUMNS = (
    "order_index",
    "source_id",
    "outcome",
    "cluster_id",
    "cluster_size",
    "reason",
    "proposal_cluster_id",
    "confidence",
    "decision_id",
)
CLUSTER_COLUMNS = (
    "cluster_id",
    "revision",
    "size",
    "status",
    "seed_id",
    "mint_provenance",
    "member_ids",
    "representation",
)
UNRESOLVED_COLUMNS = (
    "order_index",
    "source_id",
    "outcome",
    "reason",
    "proposal_cluster_id",
    "confidence",
    "decision_id",
)
EXPORT_OUTCOMES = ("assigned", "singleton", "needs_review", "failed", "pending")


@dataclass(frozen=True)
class StateView:
    """One state: each source's stored outcome and each cluster's revision."""

    revision: int | None
    assignments: dict[str, list[Any]]
    clusters: dict[str, int]


def selected_view(store: ClusterStore) -> StateView:
    """The selected state revision of a finished run; the latest state otherwise."""
    selected = store.get_meta("selected_revision")
    if selected is not None:
        for row in store.state_revisions():
            if row["revision"] == selected:
                return StateView(selected, row["assignments"], row["clusters"])
    current = store.current_assignments()
    return StateView(
        None,
        {
            sid: [
                r["outcome"],
                r["cluster_id"],
                r["reason"],
                r["proposal_cluster_id"],
                r["confidence"],
                r["decision_id"],
            ]
            for sid, r in sorted(current.items())
        },
        {cid: r["revision"] for cid, r in store.latest_revisions().items()},
    )


def _cell(value: Any) -> Any:
    return "" if value is None else value


def write_exports(store: ClusterStore, out: Path) -> dict[str, Any]:
    """Write the four exports into `out`. Returns counts and artifact paths."""
    view = selected_view(store)
    revisions = {cid: store.revision(cid, rev) for cid, rev in view.clusters.items()}
    accepted = {cid: row for cid, row in revisions.items() if row["live"] and row["members"]}
    sizes = {cid: len(row["members"]) for cid, row in accepted.items()}

    counts = dict.fromkeys(EXPORT_OUTCOMES, 0)
    members_path = out / "members.csv"
    unresolved_path = out / "unresolved.csv"
    seen_in_clusters: set[str] = set()
    with (
        members_path.open("w", encoding="utf-8", newline="") as members_handle,
        unresolved_path.open("w", encoding="utf-8", newline="") as unresolved_handle,
    ):
        members_csv = csv.DictWriter(members_handle, fieldnames=list(MEMBER_COLUMNS))
        unresolved_csv = csv.DictWriter(unresolved_handle, fieldnames=list(UNRESOLVED_COLUMNS))
        members_csv.writeheader()
        unresolved_csv.writeheader()
        for order_index, sid, _, _ in store.sources():
            row = view.assignments.get(sid)
            if row is None:
                outcome, cluster_id, reason, proposal, confidence, did = (
                    "pending",
                    None,
                    "not_processed",
                    None,
                    None,
                    None,
                )
            else:
                stored, cluster_id, reason, proposal, confidence, did = row
                if stored in ACCEPTED:
                    if cluster_id not in accepted or sid not in accepted[cluster_id]["members"]:
                        raise RuntimeError(f"{sid} is {stored} but not a member of {cluster_id}")
                    seen_in_clusters.add(sid)
                    outcome = (
                        "singleton" if stored == SEED and sizes[cluster_id] == 1 else "assigned"
                    )
                else:
                    outcome = "needs_review" if stored == DEFERRED else "failed"
                    cluster_id = None
            counts[outcome] += 1
            line = {
                "order_index": order_index,
                "source_id": sid,
                "outcome": outcome,
                "cluster_id": _cell(cluster_id),
                "cluster_size": _cell(sizes.get(cluster_id) if cluster_id else None),
                "reason": reason,
                "proposal_cluster_id": _cell(proposal),
                "confidence": _cell(confidence),
                "decision_id": _cell(did),
            }
            members_csv.writerow(line)
            if outcome in ("needs_review", "failed", "pending"):
                unresolved_csv.writerow({k: line[k] for k in UNRESOLVED_COLUMNS})

    every_member = [m for row in accepted.values() for m in row["members"]]
    if len(every_member) != len(set(every_member)) or set(every_member) != seen_in_clusters:
        raise RuntimeError("accepted clusters are not a partition of the accepted sources")

    clusters_path = out / "clusters.csv"
    with clusters_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLUSTER_COLUMNS))
        writer.writeheader()
        for cid, cluster_row in sorted(accepted.items(), key=lambda item: item[1]["created_seq"]):
            writer.writerow(
                {
                    "cluster_id": cid,
                    "revision": cluster_row["revision"],
                    "size": len(cluster_row["members"]),
                    "status": "singleton" if len(cluster_row["members"]) == 1 else "verified",
                    "seed_id": cluster_row["seed_id"],
                    "mint_provenance": cluster_row["mint_provenance"],
                    "member_ids": "|".join(cluster_row["members"]),
                    "representation": cluster_row["representation"],
                }
            )

    decisions_path = out / "decisions.jsonl"
    written = 0
    with decisions_path.open("w", encoding="utf-8") as handle:
        for decision in store.decisions():
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    counts["total"] = sum(counts[o] for o in EXPORT_OUTCOMES)
    counts["clusters"] = len(accepted)
    counts["decisions"] = written
    return {
        "counts": counts,
        "exported_revision": view.revision,
        "artifacts": {
            "members": str(members_path),
            "clusters": str(clusters_path),
            "unresolved": str(unresolved_path),
            "decisions": str(decisions_path),
        },
    }


__all__ = [
    "CLUSTER_COLUMNS",
    "EXPORT_OUTCOMES",
    "MEMBER_COLUMNS",
    "UNRESOLVED_COLUMNS",
    "StateView",
    "selected_view",
    "write_exports",
]

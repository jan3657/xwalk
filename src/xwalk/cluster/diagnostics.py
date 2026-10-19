"""Factual warnings about the limits a clustering run worked within.

Reported by `xwalk cluster` at the end of every invocation and by `xwalk cluster
inspect`, at most once per code. They state what happened in this run, read from the
store; none of them changes a decision. Thresholds:

- `retrieval_bounded`: the largest number of live clusters at any step boundary is
  greater than `pool.scan_below` and the pool has no dense head. Above `scan_below`,
  expansion pages show only retrieved clusters, so a record meets only clusters whose
  document shares a word with it: synonyms with no word in common may never be
  compared. (A dense head or a larger `scan_below` widens the search; neither makes it
  exhaustive.) Not reported for a format-1 run, whose manifest does not record the
  retrieval heads.
- `evidence_bounded`: an exported accepted cluster has more than `pool.member_evidence`
  members. A record is verified against the first `member_evidence` members listed, so
  members that joined later may never have been compared with each other
  (CONTRACTS.md 10.6 bounds the evidence; it does not make membership pairwise).
- `resume_determinism`: the run took more than one invocation, or its last invocation
  was aborted, interrupted or never finished. Resuming reproduces an uninterrupted run
  only if the model answers identical prompts identically.
- `selected_not_last`: refinement stopped at `max_iterations` or `oscillation` and the
  exported state revision is earlier than the last one, so changes made after it are in
  the store and `decisions.jsonl` but not in the exports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xwalk.cluster.exports import ExportedState
from xwalk.cluster.store import ClusterStore

RESUMABLE_STATES = ("aborted", "interrupted", "running")


def max_live_clusters(store: ClusterStore) -> int:
    """The largest number of live, non-empty clusters at the end of any step."""
    live: dict[str, bool] = {}
    best = 0
    step: int | None = None
    for rev in store.revision_log():
        if step is not None and rev["step_seq"] != step:
            best = max(best, sum(live.values()))
        step = rev["step_seq"]
        live[str(rev["cluster_id"])] = bool(rev["live"]) and bool(rev["members"])
    return max(best, sum(live.values()))


def assignment_differs(exported: Sequence[Any] | None, current: Mapping[str, Any] | None) -> bool:
    """Whether a source's exported (outcome, cluster) differs from its current one."""
    if exported is None or current is None:
        return (exported is None) != (current is None)
    return (exported[0], exported[1]) != (current["outcome"], current["cluster_id"])


def changed_after_export(store: ClusterStore, state: ExportedState) -> list[str]:
    """Sources, in snapshot order, whose last stored state differs from the exported."""
    current = store.current_assignments()
    return [
        row["source_id"]
        for row in state.rows
        if assignment_differs(
            state.view.assignments.get(row["source_id"]), current.get(row["source_id"])
        )
    ]


def run_warnings(
    store: ClusterStore,
    components: Mapping[str, Any] | None,
    state: ExportedState,
    *,
    run_state: str,
    run_dir: str | Path,
) -> list[dict[str, str]]:
    """The factual warnings of a run, one per code (see the module docstring)."""
    warnings: list[dict[str, str]] = []
    components = components or {}
    raw_pool = components.get("pool")
    pool: Mapping[str, Any] = raw_pool if isinstance(raw_pool, dict) else {}
    retrieval = components.get("retrieval")

    scan_below = pool.get("scan_below")
    if (
        isinstance(retrieval, dict)
        and retrieval.get("dense") is None
        and isinstance(scan_below, int)
    ):
        largest_pool = max_live_clusters(store)
        if largest_pool > scan_below:
            warnings.append(
                {
                    "code": "retrieval_bounded",
                    "message": (
                        f"the pool grew to {largest_pool} clusters, more than "
                        f"pool.scan_below={scan_below}, and retrieval is lexical only (no "
                        "dense head): above that size a record is shown only clusters that "
                        f"share a word with it (the best {pool.get('retrieve_limit')}, at most "
                        f"{pool.get('expand_limit')} during expansion), so synonyms with no "
                        "word in common may never have been compared. A dense head "
                        "(`dense:`) or a larger scan_below widens the search; neither makes "
                        "it exhaustive"
                    ),
                }
            )

    evidence = pool.get("member_evidence")
    if isinstance(evidence, int):
        big = sorted(
            (
                len(row["members"])
                for row in state.accepted.values()
                if len(row["members"]) > evidence
            ),
            reverse=True,
        )
        if big:
            warnings.append(
                {
                    "code": "evidence_bounded",
                    "message": (
                        f"{len(big)} accepted cluster(s) have more than "
                        f"pool.member_evidence={evidence} members (largest {big[0]}): a record "
                        f"is verified against the first {evidence} members listed, not every "
                        "member, so members that joined later may never have been compared "
                        "with each other"
                    ),
                }
            )

    invocations = len(store.invocations())
    rule = (
        "reproduces an uninterrupted run only if the model answers identical prompts "
        "identically (deterministic decoding, the same model); with a sampling model the "
        "result can differ from a clean run"
    )
    if invocations > 1:
        warnings.append(
            {
                "code": "resume_determinism",
                "message": f"this run took {invocations} invocations: a resumed run {rule}",
            }
        )
    elif run_state in RESUMABLE_STATES:
        warnings.append(
            {
                "code": "resume_determinism",
                "message": f"the run is {run_state}: resuming it {rule}",
            }
        )

    selected = store.get_meta("selected_revision")
    last = store.get_meta("last_revision")
    if selected is not None and last is not None and selected != last:
        changed = len(changed_after_export(store, state))
        warnings.append(
            {
                "code": "selected_not_last",
                "message": (
                    f"the exports hold state revision {selected}, not the last revision "
                    f"{last} (stop reason {store.get_meta('stop_reason')}): {changed} "
                    f"source(s) changed after revision {selected} and those changes are not "
                    f"in the exports; `xwalk cluster explain --run {run_dir} <source_id>` "
                    "shows them"
                ),
            }
        )
    return warnings


def dedupe(warnings: Sequence[Any]) -> list[Any]:
    """Keep the first warning of each code (objects with `.code` or dicts)."""
    seen: set[str] = set()
    kept = []
    for warning in warnings:
        code = warning["code"] if isinstance(warning, dict) else warning.code
        if code not in seen:
            seen.add(code)
            kept.append(warning)
    return kept


__all__ = [
    "assignment_differs",
    "changed_after_export",
    "dedupe",
    "max_live_clusters",
    "run_warnings",
]

"""Read-only views of a clustering run: `xwalk cluster inspect` and `xwalk cluster explain`.

Both read `manifest.json` and the store, opened read-only (`ClusterStore.open_readonly`),
and nothing else: no model call, no retriever, no write. The exported outcomes come from
`exports.exported_state`, the code that writes `members.csv`, so a view never disagrees
with the files.

Three moments matter when reading a run:

- the **exported** state: the selected state revision of a finished run (the final
  fixpoint when refinement converged, otherwise the revision with the fewest unresolved
  sources), or the latest committed state of a run that stopped early;
- the **last** state: what the store holds after its last step;
- the steps **after the export**: every step committed after the exported revision's
  end step. Their decisions are in the store and in `decisions.jsonl`, but revision
  selection set their effects aside (CONTRACTS.md 10.7). `explain` lists those that went
  against the record or its cluster, so an exported membership that a later decision
  rejected is visible instead of silent.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xwalk.cluster.diagnostics import assignment_differs, changed_after_export, run_warnings
from xwalk.cluster.engine import ACCEPTED
from xwalk.cluster.exports import exported_state
from xwalk.cluster.store import ClusterStore, ClusterStoreError

EXIT_USAGE = 2
EXIT_RUNTIME = 3
LARGEST_SHOWN = 5
CHANGED_IDS_SHOWN = 20


class ClusterInspectError(RuntimeError):
    """The run cannot be read. `code` is stable; `exit_code` follows CONTRACTS.md 8."""

    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


@dataclass
class ClusterRun:
    """An opened clustering run: its manifest and its read-only store."""

    path: Path
    manifest: dict[str, Any]
    store: ClusterStore

    def close(self) -> None:
        self.store.close()

    @property
    def member_evidence(self) -> int | None:
        components = self.manifest.get("fingerprint_components")
        pool = components.get("pool") if isinstance(components, dict) else None
        value = pool.get("member_evidence") if isinstance(pool, dict) else None
        return value if isinstance(value, int) and value > 0 else None


def open_cluster_run(run_dir: str | Path) -> ClusterRun:
    """Open a clustering run for reading. Raises `ClusterInspectError`:
    `run_not_found` / `not_a_cluster_run` (exit 2), `manifest_unreadable`,
    `store_unreadable`, `store_mismatch` (exit 3)."""
    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    if not path.is_dir() or not manifest_path.is_file():
        raise ClusterInspectError(
            "run_not_found",
            f"{path} is not an xwalk run directory (no manifest.json)",
            EXIT_USAGE,
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClusterInspectError(
            "manifest_unreadable", f"{manifest_path}: unreadable manifest ({exc})", EXIT_RUNTIME
        ) from None
    if not isinstance(manifest, dict) or manifest.get("operation") != "cluster":
        raise ClusterInspectError(
            "not_a_cluster_run",
            f"{path} is not an xwalk clustering run directory (manifest.json has no "
            "operation: cluster); matching runs are read with `xwalk inspect` and "
            "`xwalk explain`",
            EXIT_USAGE,
        )
    try:
        store = ClusterStore.open_readonly(path / "cluster.sqlite")
    except ClusterStoreError as exc:
        raise ClusterInspectError("store_unreadable", str(exc), EXIT_RUNTIME) from None
    try:
        stored = store.get_meta("run_fingerprint")
    except sqlite3.DatabaseError as exc:
        store.close()
        raise ClusterInspectError(
            "store_unreadable", f"{store.path}: unreadable clustering store ({exc})", EXIT_RUNTIME
        ) from None
    if stored != manifest.get("run_fingerprint"):
        store.close()
        raise ClusterInspectError(
            "store_mismatch",
            f"{store.path} belongs to run {stored}, but manifest.json names run "
            f"{manifest.get('run_fingerprint')}",
            EXIT_RUNTIME,
        )
    return ClusterRun(path, manifest, store)


# --- shared reads -----------------------------------------------------------------------


def run_usage(store: ClusterStore) -> dict[str, Any]:
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


def run_state_of(store: ClusterStore, manifest: Mapping[str, Any]) -> str:
    """The last invocation's run state (`running` if it never finished), else the
    manifest's."""
    invocations = store.invocations()
    if invocations:
        return str(invocations[-1]["run_state"])
    return str(manifest.get("run_state") or "unknown")


def export_cutoff(store: ClusterStore, revision: int | None) -> int | None:
    """The `seq` of the step that stored the exported state revision, or None when the
    export is the latest state (nothing comes after it)."""
    if revision is None:
        return None
    keys = {f"end/{revision}", f"resume/{revision}/end"}
    for step in store.steps():
        if step["step_key"] in keys:
            return int(step["seq"])
    return None


def _shown_members(
    store: ClusterStore,
    entry: Mapping[str, Any],
    member_evidence: int | None,
    cache: dict[tuple[str, int], list[str]],
) -> list[str]:
    key = (str(entry["cluster_id"]), int(entry["revision"]))
    if key not in cache:
        try:
            cache[key] = list(store.revision(*key)["members"])
        except KeyError:
            raise ClusterInspectError(
                "store_inconsistent",
                f"{store.path}: missing evidence for cluster {key[0]} revision {key[1]}",
                EXIT_RUNTIME,
            ) from None
    visible = [m for m in cache[key] if m != entry.get("exclude")]
    return visible if member_evidence is None else visible[:member_evidence]


def summarise_decision(
    store: ClusterStore,
    decision: Mapping[str, Any],
    member_evidence: int | None,
    cache: dict[tuple[str, int], list[str]],
    *,
    full: bool = False,
) -> dict[str, Any]:
    """One stored decision as `explain` shows it: what was asked about which clusters,
    which members the prompt listed, and the parsed answer. `full` adds the prompts and
    the retrieved candidates."""
    summary: dict[str, Any] = {
        "decision_id": decision["decision_id"],
        "step_key": decision["step_key"],
        "phase": decision["phase"],
        "iteration": decision["iteration"],
        "kind": decision["kind"],
        "subject_id": decision["subject_id"],
        "shown": [
            {
                "key": entry.get("key"),
                "cluster_id": entry["cluster_id"],
                "revision": entry["revision"],
                "excluded": entry.get("exclude"),
                "members_shown": _shown_members(store, entry, member_evidence, cache),
            }
            for entry in decision["shown"]
        ],
        "result": decision["result"],
        "confidence": decision["confidence"],
        "error": decision["error"],
        "raw": decision["raw"],
    }
    if full:
        summary["system"] = decision["system"]
        summary["prompt"] = decision["prompt"]
        summary["retrieved"] = decision["retrieved"]
        summary["usage"] = decision["usage"]
    return summary


def _readable(call: Any) -> Any:
    try:
        return call()
    except sqlite3.DatabaseError as exc:
        raise ClusterInspectError(
            "store_unreadable", f"unreadable clustering store ({exc})", EXIT_RUNTIME
        ) from None


# --- inspect ------------------------------------------------------------------------------


def inspect_cluster_run(run_dir: str | Path) -> dict[str, Any]:
    """Everything `xwalk cluster inspect` reports, as plain data."""
    run = open_cluster_run(run_dir)
    try:
        return dict(_readable(lambda: _inspect(run)))
    finally:
        run.close()


def _inspect(run: ClusterRun) -> dict[str, Any]:
    store, manifest = run.store, run.manifest
    try:
        state = exported_state(store)
    except RuntimeError as exc:
        raise ClusterInspectError(
            "store_inconsistent", f"{store.path}: {exc}", EXIT_RUNTIME
        ) from None
    stop_reason = store.get_meta("stop_reason")
    selected = store.get_meta("selected_revision")
    last = store.get_meta("last_revision")
    exported = state.view.revision

    sizes = Counter(len(row["members"]) for row in state.accepted.values())
    largest = sorted(
        state.accepted.items(), key=lambda item: (-len(item[1]["members"]), item[1]["created_seq"])
    )[:LARGEST_SHOWN]
    changed = changed_after_export(store, state)
    run_state = run_state_of(store, manifest)
    merges = Counter(str(m["outcome"]) for m in store.merges())
    kinds: Counter[str] = Counter()
    errors = 0
    for decision in store.decisions():
        kinds[str(decision["kind"])] += 1
        errors += decision["error"] is not None
    invocations = store.invocations()
    return {
        "manifest": manifest,
        "run_state": run_state,
        "state": state,
        "warnings": run_warnings(
            store,
            manifest.get("fingerprint_components"),
            state,
            run_state=run_state,
            run_dir=run.path,
        ),
        "stop_reason": stop_reason,
        "selected_revision": selected,
        "last_revision": last,
        "exported_revision": exported,
        "selected_is_last": None if selected is None or last is None else selected == last,
        "revisions": [
            {
                "revision": r["revision"],
                "iteration": r["iteration"],
                "unresolved": r["unresolved"],
                "selected": r["revision"] == selected,
            }
            for r in store.state_revisions()
        ],
        "changed_after_export": {
            "sources": len(changed),
            "source_ids": changed[:CHANGED_IDS_SHOWN],
        },
        "clusters": {
            "accepted": len(state.accepted),
            "by_size": {str(size): sizes[size] for size in sorted(sizes)},
            "largest": [
                {
                    "cluster_id": cid,
                    "size": len(row["members"]),
                    "status": "singleton" if len(row["members"]) == 1 else "verified",
                    "seed_id": row["members"][0],
                    "revision": row["revision"],
                    "mint_provenance": row["mint_provenance"],
                }
                for cid, row in largest
            ],
        },
        "merges": {o: merges.get(o, 0) for o in ("applied", "review", "rejected", "error")},
        "decisions": {
            "total": sum(kinds.values()),
            "by_kind": dict(sorted(kinds.items())),
            "with_error": errors,
        },
        "usage": run_usage(store),
        "invocations": [
            {
                "id": i["id"],
                "started_at": i["started_at"],
                "finished_at": i["finished_at"],
                "run_state": i["run_state"],
                "calls": None if not i["usage"] else i["usage"].get("calls"),
                "errors": i["errors"] or [],
            }
            for i in invocations
        ],
    }


# --- explain --------------------------------------------------------------------------------

_NOTES = {
    "pending": "not reached before the run stopped; running `xwalk cluster` again with the "
    "same job and --out resumes it",
    "failed": "a provider error or an unparseable answer; failed records are retried in "
    "every refinement round and when the run is resumed",
    "needs_review": "outside every cluster (see unresolved.csv): no decision was confident "
    "enough to place it",
    "singleton": "alone in its cluster; a singleton is not a verified equivalence",
    "assigned": "a member of a cluster with other members",
}


def explain_cluster_source(
    run_dir: str | Path, source_id: str, *, full: bool = False
) -> dict[str, Any]:
    """Everything `xwalk cluster explain` reports for one source, as plain data.
    Raises `ClusterInspectError("source_not_found", ..., 2)` for an unknown id."""
    run = open_cluster_run(run_dir)
    try:
        return dict(_readable(lambda: _explain(run, source_id, full)))
    finally:
        run.close()


def _explain(run: ClusterRun, source_id: str, full: bool) -> dict[str, Any]:
    store = run.store
    try:
        state = exported_state(store)
    except RuntimeError as exc:
        raise ClusterInspectError(
            "store_inconsistent", f"{store.path}: {exc}", EXIT_RUNTIME
        ) from None
    row = next((r for r in state.rows if r["source_id"] == source_id), None)
    if row is None:
        raise ClusterInspectError(
            "source_not_found",
            f"source {source_id!r} is not in clustering run {run.manifest.get('run_fingerprint')}",
            EXIT_USAGE,
        )
    fields = next(f for _, sid, _, f in store.sources() if sid == source_id)
    evidence = run.member_evidence
    cache: dict[tuple[str, int], list[str]] = {}
    cutoff = export_cutoff(store, state.view.revision)
    seq_of = {str(s["step_key"]): int(s["seq"]) for s in store.steps()}

    def after(step_key: str) -> bool:
        return cutoff is not None and seq_of.get(step_key, -1) > cutoff

    exported_cluster = row["cluster_id"]
    cluster = None
    if exported_cluster is not None:
        crow = state.accepted[exported_cluster]
        by_source = {r["source_id"]: r for r in state.rows}
        cluster = {
            "cluster_id": exported_cluster,
            "revision": crow["revision"],
            "size": len(crow["members"]),
            "status": "singleton" if len(crow["members"]) == 1 else "verified",
            "seed_id": crow["members"][0],
            "mint_provenance": crow["mint_provenance"],
            "members": [
                {
                    "source_id": m,
                    "outcome": by_source[m]["outcome"],
                    "reason": by_source[m]["reason"],
                    "decision_id": by_source[m]["decision_id"],
                }
                for m in crow["members"]
            ],
        }

    admission: list[dict[str, Any]] = []
    if row["decision_id"] is not None:
        found = store.decisions_by_id([row["decision_id"]])
        decision = found.get(int(row["decision_id"]))
        if decision is not None:
            admission = [
                summarise_decision(store, d, evidence, cache, full=full)
                for d in store.decisions_in_step(decision["step_key"])
            ]

    history_rows = store.assignment_log(source_id)
    history = [
        {
            "seq": h["seq"],
            "step_key": h["step_key"],
            "outcome": h["outcome"],
            "cluster_id": h["cluster_id"],
            "reason": h["reason"],
            "confidence": h["confidence"],
            "decision_id": h["decision_id"],
            "after_export": after(h["step_key"]),
        }
        for h in history_rows
    ]

    ancestry = _merge_ancestry(store, exported_cluster, cutoff) if exported_cluster else []
    later = _later(store, source_id, exported_cluster, cutoff, history_rows, evidence, cache)

    evidence_in = []
    for decision in store.decisions():
        if decision["prompt"] is None:
            continue
        for entry in decision["shown"]:
            if source_id in _shown_members(store, entry, evidence, cache):
                evidence_in.append(
                    {
                        "decision_id": decision["decision_id"],
                        "step_key": decision["step_key"],
                        "kind": decision["kind"],
                        "subject_id": decision["subject_id"],
                        "cluster_id": entry["cluster_id"],
                        "result": decision["result"],
                        "after_export": after(decision["step_key"]),
                    }
                )
                break

    current = store.current_assignments().get(source_id)
    exported_stored = state.view.assignments.get(source_id)
    current_view = None
    if current is not None and assignment_differs(exported_stored, current):
        current_view = {
            "outcome": current["outcome"],
            "cluster_id": current["cluster_id"],
            "reason": current["reason"],
            "decision_id": current["decision_id"],
        }
    last = store.get_meta("last_revision")
    superseded = bool(
        later
        and state.view.revision is not None
        and last is not None
        and state.view.revision != last
    )
    return {
        "manifest": run.manifest,
        "run_state": run_state_of(store, run.manifest),
        "source_id": source_id,
        "order_index": row["order_index"],
        "fields": fields,
        "status": row["outcome"],
        "exported": {k: row[k] for k in row if k not in ("order_index", "source_id")},
        "exported_revision": state.view.revision,
        "last_revision": last,
        "cluster": cluster,
        "admission": admission,
        "merge_ancestry": ancestry,
        "evidence_in": evidence_in,
        "later": later,
        "superseded_by_selection": superseded,
        "current": current_view,
        "history": history,
        "note": _NOTES[row["outcome"]],
        "member_evidence": evidence,
    }


def _merge_ancestry(
    store: ClusterStore, cluster_id: str, cutoff: int | None
) -> list[dict[str, Any]]:
    """Every applied merge that fed `cluster_id` up to the export, recursively: the
    clusters it absorbed, and the clusters those had absorbed before. Restoring a
    state revision also restores its merge lineage."""
    by_step: dict[str, list[dict[str, Any]]] = {}
    for merge in store.merges():
        if merge["outcome"] == "applied":
            by_step.setdefault(merge["step_key"], []).append(merge)
    applied: list[dict[str, Any]] = []
    snapshots: dict[int, list[dict[str, Any]]] = {}
    for step in store.steps():
        if cutoff is not None and step["seq"] > cutoff:
            break
        parts = step["step_key"].split("/")
        if len(parts) == 3 and parts[0] == "resume" and parts[2] == "restore":
            note = step["note"] or ""
            if note.startswith("restored:"):
                revision = int(note.removeprefix("restored:"))
                if revision not in snapshots:
                    raise ClusterInspectError(
                        "store_inconsistent",
                        f"{store.path}: missing restored state revision {revision}",
                        EXIT_RUNTIME,
                    )
                applied = list(snapshots[revision])
        applied.extend(by_step.get(step["step_key"], []))
        if (len(parts) == 2 and parts[0] == "end") or (
            len(parts) == 3 and parts[0] == "resume" and parts[2] == "end"
        ):
            snapshots[int(parts[1])] = list(applied)
    revisions: dict[str, list[dict[str, Any]]] = {}
    for rev in store.revision_log():
        revisions.setdefault(str(rev["cluster_id"]), []).append(rev)
    confidence = {
        did: d["confidence"]
        for did, d in store.decisions_by_id(
            [int(m["decision_id"]) for m in applied if m["decision_id"] is not None]
        ).items()
    }
    found: list[dict[str, Any]] = []
    todo, seen = [cluster_id], {cluster_id}
    while todo:
        winner = todo.pop()
        for merge in applied:
            if merge["winner_id"] != winner:
                continue
            loser = merge["right_id"] if merge["left_id"] == winner else merge["left_id"]
            before: list[str] = []
            for rev in revisions.get(str(loser), []):
                if rev["decision_id"] == merge["decision_id"] and not rev["members"]:
                    break
                before = list(rev["members"])
            found.append(
                {
                    "merge_seq": merge["seq"],
                    "iteration": merge["iteration"],
                    "step_key": merge["step_key"],
                    "decision_id": merge["decision_id"],
                    "winner_id": winner,
                    "absorbed_id": loser,
                    "absorbed_members": before,
                    "confidence": confidence.get(merge["decision_id"]),
                }
            )
            if loser not in seen:
                seen.add(loser)
                todo.append(str(loser))
    return sorted(found, key=lambda m: m["merge_seq"])


def _later(
    store: ClusterStore,
    source_id: str,
    exported_cluster: str | None,
    cutoff: int | None,
    history: Sequence[Mapping[str, Any]],
    evidence: int | None,
    cache: dict[tuple[str, int], list[str]],
) -> list[dict[str, Any]]:
    """Changes after the export to this source's assignment and to its exported cluster,
    each with the decision behind it. `adverse`: the change took the source out of its
    exported cluster, or removed members from that cluster (or dissolved it)."""
    if cutoff is None:
        return []
    entries: list[dict[str, Any]] = []
    for h in history:
        if h["step_seq"] <= cutoff:
            continue
        adverse = exported_cluster is not None and (
            h["outcome"] not in ACCEPTED or h["cluster_id"] != exported_cluster
        )
        entries.append(
            {
                "type": "assignment",
                "step_key": h["step_key"],
                "step_seq": h["step_seq"],
                "outcome": h["outcome"],
                "cluster_id": h["cluster_id"],
                "reason": h["reason"],
                "confidence": h["confidence"],
                "decision_id": h["decision_id"],
                "adverse": adverse,
            }
        )
    if exported_cluster is not None:
        previous: list[str] | None = None
        for rev in store.revision_log():
            if rev["cluster_id"] != exported_cluster:
                continue
            members = list(rev["members"])
            if rev["step_seq"] > cutoff and previous is not None:
                removed = [m for m in previous if m not in members]
                added = [m for m in members if m not in previous]
                dissolved = not rev["live"] and rev["merged_into"] is None
                entries.append(
                    {
                        "type": "cluster_revision",
                        "step_key": rev["step_key"],
                        "step_seq": rev["step_seq"],
                        "cluster_id": exported_cluster,
                        "revision": rev["revision"],
                        "removed": removed,
                        "added": added,
                        "live": rev["live"],
                        "merged_into": rev["merged_into"],
                        "decision_id": rev["decision_id"],
                        "adverse": bool(removed) or dissolved,
                    }
                )
            previous = members
    ids = [int(e["decision_id"]) for e in entries if e["decision_id"] is not None]
    decisions = store.decisions_by_id(ids)
    for entry in entries:
        did = entry["decision_id"]
        decision = decisions.get(int(did)) if did is not None else None
        entry["decision"] = (
            None if decision is None else summarise_decision(store, decision, evidence, cache)
        )
    return sorted(entries, key=lambda e: (e["step_seq"], e["type"]))


__all__ = [
    "ClusterInspectError",
    "ClusterRun",
    "explain_cluster_source",
    "export_cutoff",
    "inspect_cluster_run",
    "open_cluster_run",
    "run_state_of",
    "run_usage",
    "summarise_decision",
]

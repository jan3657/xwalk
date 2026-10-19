"""`xwalk cluster` and `xwalk validate` for clustering jobs, as `ops` operations.

The CLI and `xwalk.ops.cluster` call these; they return the shared `OpResult` envelope
(CONTRACTS.md section 8) and raise `OpError` for usage and directory errors.

Exit codes: 0 complete with nothing unresolved; 1 complete with `needs_review` sources;
2 invalid job, duplicate ids, missing credential or extra; 3 aborted run (fatal provider
error, `--max-calls` reached), sources that failed, or a run directory that belongs to
another run; 130 interrupted.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xwalk.cluster.diagnostics import dedupe
from xwalk.cluster.evaluate import ClusterEvalError, evaluate_cluster_run
from xwalk.cluster.inspection import (
    ClusterInspectError,
    explain_cluster_source,
    inspect_cluster_run,
)
from xwalk.cluster.job import ClusterJobSpec, load_cluster_job
from xwalk.cluster.run import ClusterRunError, order_sources, run_clustering
from xwalk.cluster.store import ClusterStoreError
from xwalk.config import CredentialMissingError, JobValidationError
from xwalk.llm.base import LLMClient
from xwalk.ops import (
    EXIT_ATTENTION,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    OpError,
    OpMessage,
    OpResult,
    _check_out_dir,
    _credential_check,
    _run_exit_code,
)
from xwalk.retrieval.dense import Encoder

OPERATION = "cluster"
ClusterJobRef = str | Path | ClusterJobSpec

_USAGE_CODES = {"duplicate_id", "usage"}
EXPERIMENTAL_NOTE = (
    "flat clustering is experimental: thresholds are uncalibrated and quality has not "
    "been evaluated on a real model (see docs/guide/clustering.md)"
)


def load_job(job: ClusterJobRef, operation: str = OPERATION) -> ClusterJobSpec:
    if isinstance(job, ClusterJobSpec):
        return job
    try:
        return load_cluster_job(job)
    except JobValidationError as exc:
        raise OpError(
            operation,
            "job_invalid",
            str(exc),
            exit_code=EXIT_USAGE,
            errors=[OpMessage(i.code, str(i)) for i in exc.issues],
        ) from None


def _preflight_errors(spec: ClusterJobSpec, *, custom_encoder: bool) -> list[OpMessage]:
    errors: list[OpMessage] = []
    if spec.source.path is not None and not (spec.base_dir / spec.source.path).is_file():
        errors.append(
            OpMessage(
                "missing_file", f"source.path: no such file: {spec.base_dir / spec.source.path}"
            )
        )
    needs: list[tuple[str, str, str]] = []
    if spec.source.required_extra:
        needs.append(("source", *spec.source.required_extra))
    if spec.dense is not None and not custom_encoder:
        needs.append(("dense", "dense", "sentence_transformers"))
    if spec.llm.kind == "litellm":
        needs.append(("llm", "litellm", "litellm"))
    for where, extra, module in needs:
        if importlib.util.find_spec(module) is None:
            errors.append(
                OpMessage(
                    "missing_extra",
                    f"{where}: needs {module!r}, part of xwalk[{extra}]; "
                    f"pip install 'xwalk[{extra}]'",
                )
            )
    return errors


def validate_cluster(
    job: ClusterJobRef, *, check_credentials: bool = True, scan_records: bool = True
) -> OpResult:
    """Offline preflight of a clustering job. Never calls a model."""
    try:
        spec = load_job(job, "validate")
    except OpError as exc:
        return exc.result
    result = OpResult(operation="validate")
    result.errors += _preflight_errors(spec, custom_encoder=False)
    credentials: dict[str, str] = {}
    if check_credentials:
        credentials, missing = _credential_check(spec)  # type: ignore[arg-type]
        result.errors += missing
    templates = None
    try:
        templates = spec.build_templates()
    except Exception as exc:
        result.errors.append(OpMessage("template_invalid", f"templates: {exc}"))
    if scan_records and templates is not None and not result.errors:
        try:
            ordered = order_sources(spec.build_source_records(), templates, spec.order)
        except ClusterRunError as exc:
            result.errors.append(OpMessage(exc.code, str(exc)))
        except Exception as exc:
            result.errors.append(OpMessage("records_unreadable", f"source: {exc}"))
        else:
            result.counts["sources"] = len(ordered)
            empty = sum(1 for r in ordered if not templates.render_query(r).strip())
            if empty:
                result.warnings.append(
                    OpMessage("empty_query", f"{empty} source record(s) render an empty query")
                )
            if not ordered:
                result.errors.append(
                    OpMessage("empty_collection", "the source collection is empty")
                )
    result.exit_code = EXIT_USAGE if result.errors else EXIT_OK
    result.data = {
        "job": spec.name,
        "kind": "cluster",
        "model": spec.llm.model,
        "credentials": credentials,
        "paid_calls": 0,
        "experimental": True,
    }
    verdict = "invalid" if result.errors else "valid"
    result.lines.append(f"clustering job {spec.name!r} is {verdict}")
    if "sources" in result.counts:
        result.lines.append(f"  sources : {result.counts['sources']} records")
    return result


async def cluster_operation(
    job: ClusterJobRef,
    out: str | Path,
    *,
    max_calls: int | None = None,
    llm: LLMClient | None = None,
    encoder: Encoder | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> OpResult:
    if max_calls is not None and max_calls < 0:
        raise OpError(OPERATION, "usage", "--max-calls must be >= 0", exit_code=EXIT_USAGE)
    spec = load_job(job)
    errors = _preflight_errors(spec, custom_encoder=encoder is not None)
    if errors:
        raise OpError(
            OPERATION,
            errors[0].code,
            "; ".join(e.message for e in errors),
            exit_code=EXIT_USAGE,
            errors=errors,
        )
    if llm is None:
        try:
            llm = spec.build_llm()
        except CredentialMissingError as exc:
            raise OpError(OPERATION, "credential_missing", str(exc), exit_code=EXIT_USAGE) from None
    if encoder is None:
        encoder = spec.build_encoder()
    out_dir = Path(out)
    try:
        report = await run_clustering(
            spec.build_source_records(),
            out=out_dir,
            llm=llm,
            templates=spec.build_templates(),
            settings=spec.build_settings(),
            prompts=spec.build_prompts(),
            encoder=encoder,
            max_calls=max_calls,
            progress=progress,
            manifest_extra={"job": spec.name, "model": spec.llm.model},
            check_out_dir=lambda path, fp, comps: _check_out_dir(
                path, fp, comps, operation=OPERATION
            ),
        )
    except ClusterRunError as exc:
        code = EXIT_USAGE if exc.code in _USAGE_CODES else EXIT_RUNTIME
        raise OpError(OPERATION, exc.code, str(exc), exit_code=code, data=exc.data) from None
    except ClusterStoreError as exc:
        raise OpError(OPERATION, "store_mismatch", str(exc), exit_code=EXIT_RUNTIME) from None

    needs_review = report.counts.get("needs_review", 0)
    if report.run_state in ("aborted", "failed"):
        exit_code = EXIT_RUNTIME
    elif needs_review:
        exit_code = EXIT_ATTENTION
    else:
        exit_code = EXIT_OK
    usage = {
        "calls": report.usage.calls,
        "prompt_tokens": report.usage.prompt_tokens,
        "completion_tokens": report.usage.completion_tokens,
        "unknown_calls": report.usage.unknown_calls,
        "cache_hits": report.usage.cache_hits,
        "tokens": report.usage.describe_tokens(),
        "limit": max_calls,
    }
    result = OpResult(
        operation=OPERATION,
        exit_code=exit_code,
        run={
            "dir": str(report.run_dir),
            "run_fingerprint": report.run_fingerprint,
            "run_state": report.run_state,
        },
        counts=dict(report.counts),
        usage=usage,
        artifacts=dict(report.artifacts),
        errors=[
            OpMessage(str(e["code"]), str(e["message"]), e.get("source_id")) for e in report.errors
        ],
        warnings=[OpMessage("experimental", EXPERIMENTAL_NOTE)],
        data={
            "job": spec.name,
            "model": spec.llm.model,
            "experimental": True,
            "stop_reason": report.stop_reason,
            "selected_revision": report.selected_revision,
            "last_revision": report.last_revision,
            "exported_revision": report.exported_revision,
        },
    )
    if needs_review:
        result.warnings.append(
            OpMessage(
                "needs_review",
                f"{needs_review} source(s) need review: see {report.artifacts.get('unresolved')}",
            )
        )
    result.warnings += [OpMessage(w["code"], w["message"]) for w in report.warnings]
    result.warnings = dedupe(result.warnings)
    c = report.counts
    result.lines += [
        f"clustered {c.get('total', 0)} records into {c.get('clusters', 0)} clusters "
        f"in {report.run_dir}",
        f"  run state     : {report.run_state}",
        f"  stop reason   : {report.stop_reason or '-'} "
        f"(exported revision {report.exported_revision})",
    ]
    for outcome in ("assigned", "singleton", "needs_review", "failed", "pending"):
        result.lines.append(f"  {outcome:<14}: {c.get(outcome, 0)}")
    result.lines.append(f"  tokens        : {usage['tokens']} in {usage['calls']} calls")
    return result


EVAL_OPERATION = "cluster eval"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def cluster_eval_operation(
    run_dir: str | Path, gold: str | Path, *, out: str | Path | None = None
) -> OpResult:
    """Score a clustering run's exported partition against a gold partition.

    Reads the exports and the store (read-only); calls no model and no retriever and
    writes no data in the run directory. Exit 0 when scored (a partial run is scored
    with its `pending` records as singletons and a `partial_run` warning); 2 for an
    unknown run directory, a run that is not a clustering run, or an unreadable gold
    file; 3 for an unreadable store or an IO error.
    """
    try:
        report = evaluate_cluster_run(run_dir, gold)
    except ClusterEvalError as exc:
        raise OpError(EVAL_OPERATION, exc.code, str(exc), exit_code=EXIT_USAGE) from None
    except ClusterStoreError as exc:
        raise OpError(
            EVAL_OPERATION, "store_unreadable", str(exc), exit_code=EXIT_RUNTIME
        ) from None
    scores = report["scores"]
    outcome_counts = scores["outcome_counts"]
    result = OpResult(
        operation=EVAL_OPERATION,
        exit_code=EXIT_OK,
        run={
            "dir": report["run"],
            "run_fingerprint": report["run_fingerprint"],
            "run_state": report["run_state"],
        },
        counts={"gold_items": report["gold_items"], **outcome_counts},
        usage={**report["usage"], "paid_calls_made_by_eval": 0},
        warnings=[OpMessage("experimental", EXPERIMENTAL_NOTE)],
        data=report,
    )
    if report["run_state"] != "complete" or outcome_counts.get("pending"):
        result.warnings.append(
            OpMessage(
                "partial_run",
                f"the run is {report['run_state']}; {outcome_counts.get('pending', 0)} "
                "pending record(s) are scored as singletons",
            )
        )
    usage = report["usage"]
    if usage["invocations_without_usage"]:
        result.warnings.append(
            OpMessage(
                "usage_incomplete",
                f"{usage['invocations_without_usage']} invocation(s) never finished and "
                "recorded no usage: the call and token totals are a lower bound",
            )
        )
    if outcome_counts.get("missing"):
        result.warnings.append(
            OpMessage(
                "gold_not_in_run",
                f"{outcome_counts['missing']} gold item(s) are not records of this run; "
                "they are scored as singletons",
            )
        )
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result.artifacts["report"] = str(path)

    exposure = report["exposure"]
    result.lines.append(
        f"clustering run {report['run']} against {report['gold']} "
        f"({report['gold_items']} gold items; run {report['run_state']})"
    )
    for name in ("as_singletons", "accepted_only"):
        row = scores.get(name)
        if row is None:
            result.lines.append(f"  {name:<14}: no accepted items")
            continue
        result.lines.append(
            f"  {name:<14}: pairwise P/R/F1 {_fmt(row['pairwise_precision'])} / "
            f"{_fmt(row['pairwise_recall'])} / {_fmt(row['pairwise_f1'])}   B-cubed P/R/F1 "
            f"{_fmt(row['bcubed_precision'])} / {_fmt(row['bcubed_recall'])} / "
            f"{_fmt(row['bcubed_f1'])}"
        )
    result.lines.append(
        "  outcomes      : " + ", ".join(f"{n} {o}" for o, n in outcome_counts.items() if n)
    )
    result.lines.append(
        f"  co-exposed    : {exposure['gold_pairs_co_exposed']} of {exposure['gold_pairs']} "
        "gold pairs were in one judge prompt together (a diagnostic, not a recall bound)"
    )
    at_least = "at least " if usage["invocations_without_usage"] else ""
    result.lines.append(
        f"  usage         : {at_least}{usage['calls']} calls over "
        f"{usage['invocations']} invocation(s); this evaluation made none"
    )
    return result


INSPECT_OPERATION = "cluster inspect"
EXPLAIN_OPERATION = "cluster explain"
CLUSTER_FILES = (
    "manifest.json",
    "cluster.sqlite",
    "members.csv",
    "clusters.csv",
    "unresolved.csv",
    "decisions.jsonl",
)


def _inspect_error(operation: str, exc: ClusterInspectError) -> OpError:
    return OpError(operation, exc.code, str(exc), exit_code=exc.exit_code)


def _run_envelope(info: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    return {
        "dir": str(run_dir),
        "run_fingerprint": info["manifest"].get("run_fingerprint"),
        "run_state": info["run_state"],
    }


def _dash(value: Any) -> str:
    return "-" if value is None else str(value)


def cluster_inspect_operation(run_dir: str | Path) -> OpResult:
    """Summarise a clustering run: exported outcome counts, cluster sizes, usage over
    every invocation, selected versus last state revision, warnings. Read-only: no model
    or retriever call, nothing written. Exits with the code the run itself produced
    (0 complete, 1 needs review or an unfinished last invocation, 3 aborted or failed,
    130 interrupted); 2 for an unknown or non-clustering directory; 3 for an unreadable
    manifest or store."""
    from xwalk import __version__

    try:
        info = inspect_cluster_run(run_dir)
    except ClusterInspectError as exc:
        raise _inspect_error(INSPECT_OPERATION, exc) from None
    manifest = info["manifest"]
    counts = dict(info["state"].counts)
    run_state = info["run_state"]
    usage = info["usage"]
    last_invocation = info["invocations"][-1] if info["invocations"] else {}
    errors = [
        OpMessage(str(e.get("code")), str(e.get("message")), e.get("source_id"))
        for e in last_invocation.get("errors") or []
    ]
    if run_state == "running":
        exit_code = EXIT_ATTENTION
    else:
        exit_code = _run_exit_code(run_state, counts.get("needs_review", 0), counts["failed"])
    result = OpResult(
        operation=INSPECT_OPERATION,
        exit_code=exit_code,
        run=_run_envelope(info, run_dir),
        counts=counts,
        usage={**usage, "calls_made_by_inspect": 0},
        artifacts={
            name: str(Path(run_dir) / name)
            for name in CLUSTER_FILES
            if (Path(run_dir) / name).exists()
        },
        errors=errors,
        warnings=inspect_warnings(info, run_dir),
        data={
            "job": manifest.get("job"),
            "model": manifest.get("model"),
            "library_version": manifest.get("library_version"),
            "inspected_with": __version__,
            "experimental": True,
            "fingerprint_components": manifest.get("fingerprint_components"),
            **{
                key: info[key]
                for key in (
                    "stop_reason",
                    "selected_revision",
                    "last_revision",
                    "exported_revision",
                    "selected_is_last",
                    "revisions",
                    "changed_after_export",
                    "clusters",
                    "merges",
                    "decisions",
                    "invocations",
                )
            },
        },
    )
    data = result.data
    sizes = ", ".join(f"{size} x{n}" for size, n in data["clusters"]["by_size"].items())
    at_least = "at least " if usage["invocations_without_usage"] else ""
    merges = data["merges"]
    result.lines += [
        f"clustering run {manifest.get('run_fingerprint')} in {run_dir}",
        f"  job           : {data['job']} ({data['model']})",
        f"  written by    : xwalk {data['library_version']}",
        f"  run state     : {run_state}",
        f"  stop reason   : {data['stop_reason'] or '-'} (exported revision "
        f"{_dash(data['exported_revision'])}, last revision {_dash(data['last_revision'])})",
    ]
    for outcome in ("assigned", "singleton", "needs_review", "failed", "pending"):
        result.lines.append(f"  {outcome:<14}: {counts.get(outcome, 0)}")
    result.lines += [
        f"  clusters      : {counts.get('clusters', 0)}",
        f"  cluster sizes : {sizes or '-'} (size x count)",
        f"  merges        : {merges['applied']} applied, {merges['review']} for review, "
        f"{merges['rejected']} rejected",
        f"  decisions     : {data['decisions']['total']} ("
        + ", ".join(f"{k} {n}" for k, n in data["decisions"]["by_kind"].items())
        + ")",
        f"  usage         : {at_least}{usage['calls']} calls over {usage['invocations']} "
        "invocation(s); this inspection made none",
    ]
    return result


def inspect_warnings(info: dict[str, Any], run_dir: str | Path) -> list[OpMessage]:
    """The warnings `cluster inspect` reports, at most one per code."""
    counts = info["state"].counts
    usage = info["usage"]
    warnings = [OpMessage("experimental", EXPERIMENTAL_NOTE)]
    if counts.get("needs_review"):
        warnings.append(
            OpMessage(
                "needs_review",
                f"{counts['needs_review']} source(s) need review: see "
                f"{Path(run_dir) / 'unresolved.csv'}",
            )
        )
    if counts.get("pending"):
        warnings.append(
            OpMessage(
                "partial_run",
                f"the run is {info['run_state']}; {counts['pending']} source(s) were not "
                "reached and are pending (running `xwalk cluster` again resumes it)",
            )
        )
    if info["run_state"] == "running":
        warnings.append(
            OpMessage(
                "invocation_unfinished",
                "the last invocation never finished: it is still running or was killed",
            )
        )
    if usage["invocations_without_usage"]:
        warnings.append(
            OpMessage(
                "usage_incomplete",
                f"{usage['invocations_without_usage']} invocation(s) never finished and "
                "recorded no usage: the call and token totals are a lower bound",
            )
        )
    warnings += [OpMessage(w["code"], w["message"]) for w in info["warnings"]]
    return dedupe(warnings)


def cluster_explain_operation(
    run_dir: str | Path, source_id: str, *, full: bool = False
) -> OpResult:
    """Why one source has its exported outcome: its exported membership, the decisions
    that admitted it, the merge ancestry of its cluster, later decisions against it or
    its cluster that revision selection set aside, and its full assignment history.
    Read-only. Exit 0; 2 for an unknown source or directory; 3 for an unreadable run."""
    try:
        info = explain_cluster_source(run_dir, source_id, full=full)
    except ClusterInspectError as exc:
        raise _inspect_error(EXPLAIN_OPERATION, exc) from None
    data = {k: v for k, v in info.items() if k not in ("manifest", "run_state")}
    result = OpResult(
        operation=EXPLAIN_OPERATION,
        exit_code=EXIT_OK,
        run=_run_envelope(info, run_dir),
        data=data,
    )
    adverse = [e for e in data["later"] if e["adverse"]]
    if adverse and data["superseded_by_selection"]:
        exported = data["exported_revision"]
        result.warnings.append(
            OpMessage(
                "superseded_by_selection",
                f"{len(adverse)} change(s) after the exported revision {exported} went "
                f"against {source_id!r} or its cluster; the run exported revision {exported} "
                f"(last {data['last_revision']}), so they are not in the exports",
                source_id,
            )
        )
    result.lines += _explain_lines(data, info["manifest"].get("run_fingerprint"))
    return result


def _decision_line(d: dict[str, Any]) -> str:
    result = d["result"] or {}
    verdict = next(
        (
            f"{key} {result[key]}"
            for key in ("decision", "chosen", "novel", "preferred", "provenance")
            if key in result
        ),
        "",
    )
    extra = f", error {d['error']}" if d["error"] else ""
    conf = "" if d["confidence"] is None else f" (confidence {d['confidence']})"
    return f"#{d['decision_id']} {d['kind']} {verdict}{conf}{extra}"


def _explain_lines(data: dict[str, Any], fp: Any) -> list[str]:
    exported = data["exported"]
    lines = [f"source {data['source_id']} in clustering run {fp}"]
    cluster = data["cluster"]
    if cluster is None:
        proposal = exported["proposal_cluster_id"]
        lines.append(
            f"  outcome       : {data['status']}, reason {exported['reason']}"
            + (f", proposal {proposal}" if proposal else "")
        )
    else:
        members = ", ".join(m["source_id"] for m in cluster["members"])
        lines.append(
            f"  outcome       : {data['status']} in {cluster['cluster_id']} "
            f"({cluster['size']} members: {members}), reason {exported['reason']}"
        )
    if data["admission"]:
        lines.append("  decided by    : " + "; ".join(_decision_line(d) for d in data["admission"]))
    for merge in data["merge_ancestry"]:
        lines.append(
            f"  merge         : #{merge['decision_id']} {merge['winner_id']} absorbed "
            f"{merge['absorbed_id']} ({', '.join(merge['absorbed_members'])}) in iteration "
            f"{merge['iteration']}"
        )
    for entry in data["later"]:
        if entry["type"] == "assignment":
            what = f"{entry['outcome']} ({entry['reason']})"
        else:
            what = (
                f"cluster revision {entry['revision']}, removed "
                f"{', '.join(entry['removed']) or '-'}, added {', '.join(entry['added']) or '-'}"
            )
        decision = entry["decision"]
        by = f" by {_decision_line(decision)}" if decision else ""
        flag = " [adverse]" if entry["adverse"] else ""
        lines.append(f"  after export  : {what}{by}{flag}")
    current = data["current"]
    if current is not None:
        lines.append(
            f"  last state    : {current['outcome']} {current['cluster_id'] or '-'} "
            f"({current['reason']})"
        )
    lines.append(f"  note          : {data['note']}")
    return lines


__all__ = [
    "EVAL_OPERATION",
    "EXPLAIN_OPERATION",
    "INSPECT_OPERATION",
    "EXPERIMENTAL_NOTE",
    "cluster_eval_operation",
    "cluster_explain_operation",
    "cluster_inspect_operation",
    "cluster_operation",
    "load_job",
    "validate_cluster",
]

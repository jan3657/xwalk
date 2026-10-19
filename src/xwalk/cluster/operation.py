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

from xwalk.cluster.evaluate import ClusterEvalError, evaluate_cluster_run
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
    writes nothing in the run directory. Exit 0 when scored (a partial run is scored
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
    result.lines.append(
        f"  usage         : {report['usage']['calls']} calls over "
        f"{report['usage']['invocations']} invocation(s); this evaluation made none"
    )
    return result


__all__ = [
    "EVAL_OPERATION",
    "EXPERIMENTAL_NOTE",
    "cluster_eval_operation",
    "cluster_operation",
    "load_job",
    "validate_cluster",
]

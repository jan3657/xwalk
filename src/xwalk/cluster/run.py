"""Running a clustering job into a run directory: identity, resume, limits, exports.

    from xwalk.cluster import ClusterSettings, run_clustering
    from xwalk.templates import TemplateSet

    templates = TemplateSet(query="{{ label }}", context="{{ label }}",
                            doc="{{ label }}", candidate="{{ label }}")
    report = asyncio.run(run_clustering(records, out="runs/c1", llm=client,
                                        templates=templates, max_calls=500))
    print(report.run_state, report.counts)

The run directory is bound to one run fingerprint (CONTRACTS.md sections 9 and 10.2):
the ordered source snapshot, the ordering rule, templates, prompts, LLM, pool and policy.
The same fingerprint resumes; another is refused and nothing is overwritten.
"""

from __future__ import annotations

import asyncio
import json
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xwalk import __version__
from xwalk.cluster.engine import FAILED, ClusterEngine, Progress
from xwalk.cluster.exports import write_exports
from xwalk.cluster.pool import PoolIndex
from xwalk.cluster.prompts import ClusterPrompts
from xwalk.cluster.settings import ClusterSettings, OrderRule
from xwalk.cluster.store import ClusterStore
from xwalk.fingerprint import hash_record, hash_value
from xwalk.llm.base import LLMClient, LLMFatalError
from xwalk.llm.budget import CALL_LIMIT_CODE, BudgetedLLM, CallBudget, CallLimitExceeded
from xwalk.records import Record, Usage
from xwalk.retrieval.base import component_differences
from xwalk.retrieval.dense import Encoder
from xwalk.templates import TemplateSet

CLUSTER_FORMAT_VERSION = 1
STORE_FILE = "cluster.sqlite"
MANIFEST_FILE = "manifest.json"

OutDirCheck = Callable[[Path, str, Mapping[str, Any]], None]


class ClusterRunError(RuntimeError):
    """The run directory cannot be used, or the input is invalid. `code` is stable."""

    def __init__(self, code: str, message: str, data: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = dict(data or {})


@dataclass(frozen=True)
class ClusterReport:
    run_dir: Path
    run_fingerprint: str
    run_state: str
    stop_reason: str | None
    selected_revision: int | None
    last_revision: int | None
    exported_revision: int | None
    counts: dict[str, int]
    usage: Usage
    errors: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)
    pool_updates: int = 0


def _order_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def order_sources(
    records: Iterable[Record], templates: TemplateSet, rule: OrderRule = "label"
) -> list[Record]:
    """The processing order. `label`: stable sort on the normalised query text, then id,
    so a permuted input file gives the same run. `input`: the order given."""
    items = list(records)
    seen: set[str] = set()
    for record in items:
        if record.id in seen:
            raise ClusterRunError("duplicate_id", f"source id {record.id!r} appears more than once")
        seen.add(record.id)
    if rule == "input":
        return items
    return sorted(items, key=lambda r: (_order_key(templates.render_query(r)), r.id))


def fingerprint_components(
    ordered: Sequence[Record],
    *,
    templates: TemplateSet,
    prompts: ClusterPrompts,
    llm: LLMClient,
    settings: ClusterSettings,
    pool: Mapping[str, Any],
) -> dict[str, Any]:
    """Everything whose change makes earlier clustering decisions invalid."""
    snapshot = [[r.id, hash_record(r)] for r in ordered]
    return {
        "operation": "cluster",
        "format_version": CLUSTER_FORMAT_VERSION,
        "library_version": __version__,
        "snapshot": hash_value(snapshot),
        "source_count": len(snapshot),
        "templates": templates.fingerprint,
        "prompts": prompts.fingerprint,
        "llm": llm.fingerprint,
        "pool": dict(pool),
        **settings.components(),
    }


def _read_manifest(out: Path) -> dict[str, Any] | None:
    path = out / MANIFEST_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def default_out_dir_check(out: Path, run_fp: str, components: Mapping[str, Any]) -> None:
    """Refuse a directory that holds another run or something that is not a run."""
    if not out.exists():
        return
    if not out.is_dir():
        raise ClusterRunError("out_not_a_directory", f"{out} is not a directory")
    manifest = _read_manifest(out)
    if manifest is None:
        if any(out.iterdir()):
            raise ClusterRunError(
                "out_not_a_run",
                f"{out} is not empty and is not an xwalk run directory; choose another --out",
            )
        return
    if manifest.get("run_fingerprint") == run_fp:
        return
    stored = manifest.get("fingerprint_components")
    differences = component_differences(stored if isinstance(stored, dict) else None, components)
    raise ClusterRunError(
        "run_fingerprint_mismatch",
        f"{out} holds run {manifest.get('run_fingerprint')}, but this job is run {run_fp} "
        f"(changed: {', '.join(sorted(differences)) or 'unknown'}); nothing was overwritten",
        {"differences": {k: {"stored": a, "expected": b} for k, (a, b) in differences.items()}},
    )


def _usage_dict(usage: Usage, limit: int | None) -> dict[str, Any]:
    return {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "unknown_calls": usage.unknown_calls,
        "cache_hits": usage.cache_hits,
        "tokens": usage.describe_tokens(),
        "limit": limit,
    }


async def run_clustering(
    sources: Iterable[Record],
    *,
    out: str | Path,
    llm: LLMClient,
    templates: TemplateSet,
    settings: ClusterSettings | None = None,
    prompts: ClusterPrompts | None = None,
    encoder: Encoder | None = None,
    max_calls: int | None = None,
    progress: Progress | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    check_out_dir: OutDirCheck = default_out_dir_check,
) -> ClusterReport:
    """Cluster `sources` into `out`, resuming a run with the same fingerprint.

    `max_calls` caps upstream requests in this invocation; reaching it aborts the run at
    a step boundary (`run_state="aborted"`, error `call_limit_reached`) and a later call
    resumes. A fatal provider error aborts the same way. KeyboardInterrupt and task
    cancellation write `interrupted` exports and the manifest, then propagate.
    """
    settings = settings or ClusterSettings()
    prompts = prompts or ClusterPrompts()
    if max_calls is not None and max_calls < 0:
        raise ClusterRunError("usage", "max_calls must be >= 0")
    out_dir = Path(out)
    ordered = order_sources(sources, templates, settings.order)
    budgeted = BudgetedLLM(llm, CallBudget(max_calls))
    pool_components = PoolIndex(encoder=encoder).components()
    components = fingerprint_components(
        ordered,
        templates=templates,
        prompts=prompts,
        llm=budgeted,
        settings=settings,
        pool=pool_components,
    )
    run_fp = hash_value(components)
    check_out_dir(out_dir, run_fp, components)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = ClusterStore.open(out_dir / STORE_FILE)
    try:
        store.bind(run_fp, [(r, hash_record(r)) for r in ordered])
        invocation = store.begin_invocation()
        base_manifest = {
            "operation": "cluster",
            "run_fingerprint": run_fp,
            "fingerprint_components": components,
            "library_version": __version__,
            "experimental": True,
            "max_calls": max_calls,
            **dict(manifest_extra or {}),
        }
        _write_manifest(out_dir, {**base_manifest, "run_state": "running"})
        engine = ClusterEngine(
            store=store,
            llm=budgeted,
            templates=templates,
            settings=settings,
            prompts=prompts,
            run_fingerprint=run_fp,
            encoder=encoder,
            progress=progress,
        )
        errors: list[dict[str, Any]] = []
        run_state = "complete"
        stop_reason = selected = last = None
        pool_updates = 0
        try:
            report = await engine.run()
        except LLMFatalError as exc:
            run_state = "aborted"
            code = (
                CALL_LIMIT_CODE if isinstance(exc, CallLimitExceeded) else "fatal_provider_failure"
            )
            errors.append({"code": code, "message": str(exc)})
        except (KeyboardInterrupt, asyncio.CancelledError):
            _finish(
                store,
                out_dir,
                invocation,
                base_manifest,
                "interrupted",
                budgeted,
                max_calls,
                [{"code": "interrupted", "message": "interrupted"}],
                engine,
            )
            raise
        except Exception as exc:
            _finish(
                store,
                out_dir,
                invocation,
                base_manifest,
                "aborted",
                budgeted,
                max_calls,
                [{"code": "exception", "message": f"{type(exc).__name__}: {exc}"}],
                engine,
            )
            raise
        else:
            stop_reason, selected, last = (
                report.stop_reason,
                report.selected_revision,
                report.last_revision,
            )
            pool_updates = report.pool_updates
            failed = [sid for sid, m in engine.members.items() if m.outcome == FAILED]
            if failed:
                run_state = "failed"
                errors += [
                    {
                        "code": "source_failed",
                        "message": engine.members[sid].reason,
                        "source_id": sid,
                    }
                    for sid in sorted(failed, key=engine.order.index)
                ]
        exported = _finish(
            store,
            out_dir,
            invocation,
            base_manifest,
            run_state,
            budgeted,
            max_calls,
            errors,
            engine,
            stop=(stop_reason, selected, last),
        )
    finally:
        store.close()
    artifacts = dict(exported["artifacts"])
    artifacts.update({"manifest": str(out_dir / MANIFEST_FILE), "store": str(out_dir / STORE_FILE)})
    return ClusterReport(
        run_dir=out_dir,
        run_fingerprint=run_fp,
        run_state=run_state,
        stop_reason=stop_reason,
        selected_revision=selected,
        last_revision=last,
        exported_revision=exported["exported_revision"],
        counts=exported["counts"],
        usage=budgeted.usage,
        errors=errors,
        artifacts=artifacts,
        components=components,
        pool_updates=pool_updates,
    )


def _write_manifest(out: Path, manifest: Mapping[str, Any]) -> None:
    tmp = out / (MANIFEST_FILE + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    tmp.replace(out / MANIFEST_FILE)


def _finish(
    store: ClusterStore,
    out: Path,
    invocation: int,
    base: Mapping[str, Any],
    run_state: str,
    budgeted: BudgetedLLM,
    max_calls: int | None,
    errors: Sequence[Mapping[str, Any]],
    engine: ClusterEngine,
    *,
    stop: tuple[str | None, int | None, int | None] = (None, None, None),
) -> dict[str, Any]:
    usage = _usage_dict(budgeted.usage, max_calls)
    exported = write_exports(store, out)
    store.finish_invocation(invocation, run_state, usage, errors)
    stop_reason, selected, last = stop
    _write_manifest(
        out,
        {
            **base,
            "run_state": run_state,
            "stop_reason": stop_reason,
            "selected_revision": selected,
            "last_revision": last,
            "exported_revision": exported["exported_revision"],
            "counts": exported["counts"],
            "usage": usage,
            "errors": list(errors),
            "pool_updates": engine.pool.updates,
            "artifacts": {name: Path(path).name for name, path in exported["artifacts"].items()},
        },
    )
    return exported


__all__ = [
    "CLUSTER_FORMAT_VERSION",
    "MANIFEST_FILE",
    "STORE_FILE",
    "ClusterReport",
    "ClusterRunError",
    "default_out_dir_check",
    "fingerprint_components",
    "order_sources",
    "run_clustering",
]

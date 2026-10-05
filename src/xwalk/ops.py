"""Operations: the one place that validates, prepares, runs, inspects, explains and
exports (docs/claude-upgrade/CONTRACTS.md section 8).

The CLI, the high-level Python route and later MCP and clustering call these functions
and only format what they return. Every operation returns an `OpResult` -- the versioned
machine-readable envelope -- or raises `OpError`, which carries the same envelope for a
failure. Nothing here prints; progress goes through an optional callback.

    from xwalk import ops

    check = ops.validate("job.yaml")              # strict, offline, zero model calls
    result = ops.run("job.yaml", out="runs/a", max_calls=500)
    print(result.exit_code, result.run, result.counts, result.usage["tokens"])

Exit codes: 0 complete; 1 attention (review rows, a partial `--limit` run, rejected
review rows); 2 usage or configuration error; 3 runtime failure (aborted or failed run,
IO error, incompatible index or run directory); 130 interrupted.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xwalk import __version__
from xwalk.batch import (
    PENDING,
    RunState,
    export_history_jsonl,
    export_mapping_csv,
    run_batch,
    usage_to_dict,
)
from xwalk.config import (
    CredentialMissingError,
    JobSpec,
    JobValidationError,
    RetrieverSpec,
    load_job,
)
from xwalk.fingerprint import hash_value
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMClient
from xwalk.llm.budget import CALL_LIMIT_CODE, BudgetedLLM, CallBudget
from xwalk.records import MatchResult, MatchStatus, Record, RetrievalHit
from xwalk.retrieval import bm25 as bm25_index
from xwalk.retrieval import dense as dense_index
from xwalk.retrieval.base import Retriever, SearchRequest, component_differences
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.dense import DenseRetriever, Encoder
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.review import adjudicated, review_history
from xwalk.serde import result_to_dict
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xwalk.cluster.job import ClusterJobSpec

SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_ATTENTION = 1
EXIT_USAGE = 2
EXIT_RUNTIME = 3
EXIT_INTERRUPTED = 130

_STATUS_BY_EXIT = {
    EXIT_OK: "ok",
    EXIT_ATTENTION: "attention",
    EXIT_USAGE: "error",
    EXIT_RUNTIME: "error",
    EXIT_INTERRUPTED: "interrupted",
}

EXPORT_VIEWS = ("raw", "reviewed", "history")

# Bounds for the read operations an agent can call repeatedly (`search`, `list_results`):
# one call can never return an unbounded slice of a collection or a ledger.
MAX_SEARCH_LIMIT = 100
MAX_PAGE_SIZE = 200

REVIEWED_COLUMNS = (
    "source_id",
    "final_matched_id",
    "final_status",
    "review_decision",
    "corrected_target_id",
    "reviewer",
    "review_note",
    "reviewed_at",
    "model_matched_id",
    "model_status",
    "model_reason",
    "confidence",
    "result_key",
)

JobRef = str | Path | JobSpec
EncoderFactory = Callable[[RetrieverSpec], Encoder]


# --- the envelope -------------------------------------------------------------------


@dataclass(frozen=True)
class OpMessage:
    """One error or warning: a stable `code`, a human `message`, optionally the source."""

    code: str
    message: str
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.source_id is not None:
            data["source_id"] = self.source_id
        return data


@dataclass
class OpResult:
    """What an operation produced. `envelope()` is the `--json` output, schema 1.

    `lines` is the human rendering the CLI prints without `--json`; it is not part of
    the envelope. `data` holds operation-specific detail (inspect, explain, validate).
    """

    operation: str
    exit_code: int = EXIT_OK
    run: dict[str, Any] | None = None
    counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[OpMessage] = field(default_factory=list)
    errors: list[OpMessage] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return _STATUS_BY_EXIT.get(self.exit_code, "error")

    @property
    def ok(self) -> bool:
        return self.exit_code in (EXIT_OK, EXIT_ATTENTION)

    def envelope(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "operation": self.operation,
            "status": self.status,
            "exit_code": self.exit_code,
            "run": self.run,
            "counts": dict(self.counts),
            "usage": self.usage,
            "artifacts": dict(self.artifacts),
            "warnings": [w.to_dict() for w in self.warnings],
            "errors": [e.to_dict() for e in self.errors],
            "data": self.data,
        }


class OpError(Exception):
    """An operation failed. `result` is the envelope to report (status ``error``)."""

    def __init__(
        self,
        operation: str,
        code: str,
        message: str,
        *,
        exit_code: int,
        errors: Sequence[OpMessage] | None = None,
        data: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.result = OpResult(
            operation=operation,
            exit_code=exit_code,
            errors=list(errors) if errors else [OpMessage(code, message)],
            data=dict(data or {}),
            run=dict(run) if run is not None else None,
        )


def failure_result(operation: str, exc: BaseException) -> OpResult:
    """The error envelope for an exception that escaped an operation. No traceback.

    `OpError` carries its own envelope; a job validation error, a missing credential or
    optional extra is a usage error (exit 2); an IO error or anything else is a runtime
    failure (exit 3). The CLI and the MCP server both report failures through this.
    """
    from xwalk._extras import MissingExtra

    if isinstance(exc, OpError):
        result = exc.result
        result.operation = operation
        return result
    if isinstance(exc, JobValidationError):
        return OpResult(operation, exit_code=EXIT_USAGE, errors=_job_errors(exc))
    if isinstance(exc, CredentialMissingError):
        code, exit_code = "credential_missing", EXIT_USAGE
    elif isinstance(exc, MissingExtra):
        code, exit_code = "missing_extra", EXIT_USAGE
    elif isinstance(exc, OSError):
        code, exit_code = "io_error", EXIT_RUNTIME
    else:
        code, exit_code = "exception", EXIT_RUNTIME
    message = str(exc) if code != "exception" else f"{type(exc).__name__}: {exc}"
    return OpResult(operation, exit_code=exit_code, errors=[OpMessage(code, message)])


# --- job loading and preflight ----------------------------------------------------


def _job_errors(exc: JobValidationError) -> list[OpMessage]:
    return [OpMessage(issue.code, str(issue)) for issue in exc.issues]


def load_valid_job(job: JobRef, *, operation: str = "validate") -> JobSpec:
    """A strictly validated `JobSpec`, or `OpError` (exit 2) naming every issue."""
    if isinstance(job, JobSpec):
        return job
    try:
        return load_job(job)
    except JobValidationError as exc:
        raise OpError(
            operation, "job_invalid", str(exc), exit_code=EXIT_USAGE, errors=_job_errors(exc)
        ) from None


def _extra_errors(job: JobSpec, *, custom_encoder: bool = False) -> list[OpMessage]:
    """Optional dependencies the job needs and the environment lacks. A caller-supplied
    encoder factory replaces sentence-transformers for dense retrievers."""
    needs: list[tuple[str, str, str]] = []
    for role, spec in (("target", job.target), ("source", job.source)):
        if spec.required_extra:
            needs.append((role, *spec.required_extra))
    for index, retriever in enumerate(job.retrievers):
        if retriever.required_extra and not custom_encoder:
            needs.append((f"retrievers.{index}", *retriever.required_extra))
    if job.llm.kind == "litellm":
        needs.append(("llm", "litellm", "litellm"))
    errors: list[OpMessage] = []
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


def _file_errors(job: JobSpec) -> list[OpMessage]:
    errors: list[OpMessage] = []
    paths = [("prompts.slots", job.base_dir / job.prompts.slots)]
    for role, spec in (("target", job.target), ("source", job.source)):
        if spec.path is not None:
            paths.append((f"{role}.path", job.base_dir / spec.path))
    for where, path in paths:
        if not path.is_file():
            errors.append(OpMessage("missing_file", f"{where}: no such file: {path}"))
    return errors


def _credential_check(job: JobSpec) -> tuple[dict[str, str], list[OpMessage]]:
    """Presence only. The value is never read into the result."""
    variable = job.llm.api_key_env
    if not variable:
        return {}, []
    if os.environ.get(variable):
        return {variable: "set"}, []
    return {variable: "missing"}, [
        OpMessage(
            "credential_missing",
            f"environment variable {variable} (llm.api_key_env) is not set",
        )
    ]


def _scan(
    records: Iterable[Record], role: str, templates: TemplateSet, *, as_doc: bool
) -> tuple[int, list[OpMessage], list[OpMessage]]:
    seen: set[str] = set()
    errors: list[OpMessage] = []
    empty = 0
    count = 0
    for record in records:
        count += 1
        if record.id in seen:
            errors.append(
                OpMessage("duplicate_id", f"{role} id {record.id!r} appears more than once")
            )
        seen.add(record.id)
        text = templates.render_doc(record) if as_doc else templates.render_query(record)
        if not text.strip():
            empty += 1
    warnings = []
    if empty:
        what = "doc" if as_doc else "query"
        warnings.append(
            OpMessage(f"empty_{what}", f"{empty} {role} record(s) render an empty {what} text")
        )
    if count == 0:
        errors.append(OpMessage("empty_collection", f"the {role} collection has no records"))
    return count, errors, warnings


def validate(
    job: JobRef,
    *,
    check_credentials: bool = True,
    scan_records: bool = True,
) -> OpResult:
    """Offline preflight. Never builds an LLM client and never calls a model.

    Checks the job file strictly, the templates and prompt slots, that referenced files
    exist, that optional extras are installed, that the credential variable is *set*
    (its value is never read into the result), and -- with `scan_records` -- that both
    collections load with unique ids.
    """
    from xwalk.prompts.contract import PromptSet, load_slots, validate_contract

    if _is_cluster_job(job):
        from xwalk.cluster.operation import validate_cluster

        return validate_cluster(
            job,  # type: ignore[arg-type]
            check_credentials=check_credentials,
            scan_records=scan_records,
        )
    try:
        spec = load_valid_job(job, operation="validate")
    except OpError as exc:
        return exc.result
    result = OpResult(operation="validate")
    errors = result.errors
    errors += _file_errors(spec)
    errors += _extra_errors(spec)

    templates: TemplateSet | None = None
    try:
        templates = spec.build_templates()
    except Exception as exc:
        errors.append(OpMessage("template_invalid", f"templates: {exc}"))
    if (spec.base_dir / spec.prompts.slots).is_file():
        try:
            validate_contract(PromptSet.from_slots(load_slots(spec.base_dir / spec.prompts.slots)))
        except Exception as exc:
            errors.append(OpMessage("prompts_invalid", f"prompts.slots: {exc}"))

    credentials: dict[str, str] = {}
    if check_credentials:
        credentials, missing = _credential_check(spec)
        errors += missing

    loaders_ok = not any(e.code in ("missing_file", "missing_extra") for e in errors)
    if scan_records and templates is not None and loaders_ok:
        for role, builder, as_doc in (
            ("target", spec.build_target_records, True),
            ("source", spec.build_source_records, False),
        ):
            try:
                count, scan_errors, scan_warnings = _scan(builder(), role, templates, as_doc=as_doc)
            except Exception as exc:
                errors.append(OpMessage("records_unreadable", f"{role}: {exc}"))
                continue
            result.counts[f"{role}s"] = count
            errors += scan_errors
            result.warnings += scan_warnings

    result.exit_code = EXIT_USAGE if errors else EXIT_OK
    result.data = {
        "job": spec.name,
        "model": spec.llm.model,
        "credentials": credentials,
        "retrievers": [r.index_name for r in spec.retrievers],
        "paid_calls": 0,
    }
    verdict = "invalid" if errors else "valid"
    result.lines.append(f"job {spec.name!r} is {verdict}")
    for role in ("targets", "sources"):
        if role in result.counts:
            result.lines.append(f"  {role:<8}: {result.counts[role]} records")
    for variable, state in credentials.items():
        result.lines.append(f"  {variable}: {state}")
    return result


def _is_cluster_job(job: object) -> bool:
    from xwalk.cluster.job import ClusterJobSpec, is_cluster_job

    if isinstance(job, ClusterJobSpec):
        return True
    return isinstance(job, (str, Path)) and is_cluster_job(job)


# --- index preparation (CONTRACTS.md section 7) -------------------------------------


@dataclass(frozen=True)
class PlannedIndex:
    """One retriever's index as the job expects it, computed without building anything.

    It quacks like a `Retriever` for the run fingerprint (name, fingerprint, depth and
    encoder identity) so a run's identity is known before any index is touched.
    """

    spec: RetrieverSpec
    directory: Path
    name: str
    expected: dict[str, Any]
    encoder: Encoder | None = None

    @property
    def fingerprint(self) -> str:
        return hash_value(self.expected)

    @property
    def default_limit(self) -> int:
        return self.spec.limit

    @property
    def encoder_identity(self) -> dict[str, Any] | None:
        return None if self.encoder is None else dense_index.encoder_identity(self.encoder)

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        raise RuntimeError(f"{self.name} is planned, not built; call prepare_indexes")


def _default_encoder_factory(spec: RetrieverSpec) -> Encoder:
    return JobSpec.build_encoder(spec)


def plan_indexes(
    job: JobSpec,
    targets: Sequence[Record],
    templates: TemplateSet,
    index_dir: str | Path,
    *,
    encoder_factory: EncoderFactory | None = None,
) -> list[PlannedIndex]:
    """The expected identity of every index. A dense encoder is loaded, never run."""
    factory = encoder_factory or _default_encoder_factory
    plans: list[PlannedIndex] = []
    for spec in job.retrievers:
        directory = Path(index_dir) / spec.index_name
        if spec.kind == "bm25":
            expected = bm25_index.index_components(
                targets, templates, exact_fields=spec.effective_exact_fields
            )
            plans.append(PlannedIndex(spec, directory, spec.name or "bm25", expected))
        else:
            encoder = factory(spec)
            expected = dense_index.index_components(targets, templates, encoder)
            name = spec.name or f"dense:{encoder.name}"
            plans.append(PlannedIndex(spec, directory, name, expected, encoder))
    return plans


def _index_state(plan: PlannedIndex) -> tuple[str, dict[str, tuple[Any, Any]]]:
    """`absent`, `compatible` or `incompatible`, with the differing components."""
    if not (plan.directory / "xwalk_meta.json").is_file():
        return "absent", {}
    reader = bm25_index.read_components if plan.spec.kind == "bm25" else dense_index.read_components
    differences = component_differences(reader(plan.directory), plan.expected)
    return ("incompatible", differences) if differences else ("compatible", {})


def prepare_indexes(
    plans: Sequence[PlannedIndex],
    targets: Sequence[Record],
    templates: TemplateSet,
    *,
    rebuild: bool = False,
    operation: str = "index",
) -> tuple[list[Retriever], dict[str, str]]:
    """Open compatible indexes, build absent ones, refuse incompatible ones.

    An incompatible index is replaced only with `rebuild=True`. Every index is checked
    before any is touched, so a refusal leaves all of them as they were. Returns the
    retrievers and `{name: "opened" | "built" | "rebuilt"}`.
    """
    states = {plan.name: _index_state(plan) for plan in plans}
    refused = {name: diff for name, (state, diff) in states.items() if state == "incompatible"}
    if refused and not rebuild:
        errors = [
            OpMessage(
                "index_mismatch",
                f"index {plan.directory} differs from the job in: "
                + ", ".join(sorted(refused[plan.name])),
            )
            for plan in plans
            if plan.name in refused
        ]
        raise OpError(
            operation,
            "index_mismatch",
            "an existing index is incompatible with this job; rerun with --rebuild-index "
            "to replace it, or choose another index directory",
            exit_code=EXIT_RUNTIME,
            errors=errors,
            data={
                "differences": {
                    name: {k: {"stored": a, "expected": b} for k, (a, b) in diff.items()}
                    for name, diff in refused.items()
                }
            },
        )

    retrievers: list[Retriever] = []
    actions: dict[str, str] = {}
    for plan in plans:
        state, _ = states[plan.name]
        spec = plan.spec
        if state == "compatible":
            actions[plan.name] = "opened"
            if spec.kind == "bm25":
                retrievers.append(
                    BM25Retriever.open(
                        plan.directory,
                        name=plan.name,
                        expected=plan.expected,
                        default_limit=spec.limit,
                    )
                )
            else:
                assert plan.encoder is not None
                retrievers.append(
                    DenseRetriever.open(
                        plan.directory,
                        plan.encoder,
                        name=plan.name,
                        default_limit=spec.limit,
                        expected=plan.expected,
                    )
                )
            continue
        actions[plan.name] = "built" if state == "absent" else "rebuilt"
        if spec.kind == "bm25":
            retrievers.append(
                BM25Retriever.build(
                    targets,
                    templates,
                    plan.directory,
                    name=plan.name,
                    exact_fields=spec.effective_exact_fields,
                    default_limit=spec.limit,
                )
            )
        else:
            assert plan.encoder is not None
            retrievers.append(
                DenseRetriever.build(
                    targets,
                    templates,
                    plan.directory,
                    plan.encoder,
                    name=plan.name,
                    default_limit=spec.limit,
                )
            )
    for plan, retriever in zip(plans, retrievers, strict=True):
        if retriever.fingerprint != plan.fingerprint:  # pragma: no cover - invariant
            raise RuntimeError(f"index {plan.name} does not have the planned fingerprint")
    return retrievers, actions


def _read_targets(job: JobSpec, operation: str) -> tuple[list[Record], MemoryStore]:
    """The target collection, read once and shared by the store and every index."""
    targets = list(job.build_target_records())
    try:
        store = MemoryStore.from_source(targets)
    except ValueError as exc:
        raise OpError(operation, "duplicate_id", str(exc), exit_code=EXIT_USAGE) from None
    return targets, store


def _preflight(job: JobSpec, operation: str, *, custom_encoder: bool = False) -> None:
    errors = _file_errors(job) + _extra_errors(job, custom_encoder=custom_encoder)
    if errors:
        raise OpError(
            operation,
            errors[0].code,
            "; ".join(e.message for e in errors),
            exit_code=EXIT_USAGE,
            errors=errors,
        )


def index(
    job: JobRef,
    index_dir: str | Path,
    *,
    rebuild: bool = False,
    encoder_factory: EncoderFactory | None = None,
) -> OpResult:
    """Build absent indexes, open compatible ones, refuse incompatible ones."""
    spec = load_valid_job(job, operation="index")
    _preflight(spec, "index", custom_encoder=encoder_factory is not None)
    templates = spec.build_templates()
    targets, _ = _read_targets(spec, "index")
    plans = plan_indexes(spec, targets, templates, index_dir, encoder_factory=encoder_factory)
    retrievers, actions = prepare_indexes(plans, targets, templates, rebuild=rebuild)
    result = OpResult(operation="index", counts={"targets": len(targets)})
    result.artifacts["index"] = str(index_dir)
    result.data["retrievers"] = [
        {"name": r.name, "fingerprint": r.fingerprint, "action": actions[r.name]}
        for r in retrievers
    ]
    result.lines.append(
        f"indexed {len(targets)} target records into {len(retrievers)} retriever(s)"
    )
    for r in retrievers:
        result.lines.append(f"  {r.name}: {r.fingerprint} ({actions[r.name]})")
    return result


async def search_async(
    job: JobRef,
    query: str,
    *,
    index_dir: str | Path,
    limit: int = 10,
    encoder_factory: EncoderFactory | None = None,
) -> OpResult:
    """Retrieve fused candidates for `query`. See `search`."""
    operation = "search"
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise OpError(
            operation,
            "usage",
            f"limit must be between 1 and {MAX_SEARCH_LIMIT}, got {limit}",
            exit_code=EXIT_USAGE,
        )
    if not query.strip():
        raise OpError(operation, "usage", "the query is empty", exit_code=EXIT_USAGE)
    spec = load_valid_job(job, operation=operation)
    _preflight(spec, operation, custom_encoder=encoder_factory is not None)
    templates = spec.build_templates()
    targets, store = _read_targets(spec, operation)
    plans = plan_indexes(spec, targets, templates, index_dir, encoder_factory=encoder_factory)
    retrievers, actions = prepare_indexes(
        plans, targets, templates, rebuild=False, operation=operation
    )

    async def one(retriever: Retriever) -> Sequence[RetrievalHit]:
        depth = max(limit, getattr(retriever, "default_limit", None) or limit)
        return await retriever.search(SearchRequest(text=query, limit=depth))

    outcomes = await asyncio.gather(*(one(r) for r in retrievers), return_exceptions=True)
    groups: list[Sequence[RetrievalHit]] = []
    warnings: list[OpMessage] = []
    for retriever, outcome in zip(retrievers, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            warnings.append(OpMessage("retriever_failure", f"{retriever.name}: {outcome}"))
        else:
            groups.append(outcome)
    # The same fusion the matcher applies, so these are the candidates a record whose
    # query renders to `query` would be shown (before the selector's own truncation).
    fused = reciprocal_rank_fusion(groups, store)[:limit]
    candidates = [
        {
            "rank": rank,
            "id": candidate.id,
            "fused_score": candidate.fused_score,
            "retrievers": {hit.retriever: hit.rank for hit in candidate.evidence},
            "text": templates.render_candidate(candidate.record),
        }
        for rank, candidate in enumerate(fused, start=1)
    ]
    result = OpResult(
        operation=operation,
        counts={"targets": len(targets), "candidates": len(candidates)},
        artifacts={"index": str(index_dir)},
        warnings=warnings,
        data={"query": query, "limit": limit, "candidates": candidates, "indexes": actions},
    )
    result.lines.append(f"{len(candidates)} candidate(s) for {query!r}")
    for row in candidates:
        result.lines.append(f"  {row['rank']:>3}. {row['id']}  {row['text']}")
    return result


def search(
    job: JobRef,
    query: str,
    *,
    index_dir: str | Path,
    limit: int = 10,
    encoder_factory: EncoderFactory | None = None,
) -> OpResult:
    """The fused retrieval candidates for a free-text query. Never calls a model.

    Every retriever in the job is searched (indexes in `index_dir` are opened, built
    when absent, refused when incompatible) and the hit lists are fused by reciprocal
    rank, as the matcher does. At most `limit` (1 to `MAX_SEARCH_LIMIT`) candidates are
    returned, each with its rendered candidate text and per-retriever ranks.
    """
    return asyncio.run(
        search_async(job, query, index_dir=index_dir, limit=limit, encoder_factory=encoder_factory)
    )


# --- run directories (CONTRACTS.md section 9) ---------------------------------------


def _read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _check_out_dir(
    out: Path, run_fp: str, components: Mapping[str, Any], *, operation: str
) -> None:
    """Refuse a run directory that belongs to another run or to something else."""
    if not out.exists():
        return
    if not out.is_dir():
        raise OpError(
            operation, "out_not_a_directory", f"{out} is not a directory", exit_code=EXIT_RUNTIME
        )
    manifest = _read_manifest(out)
    if manifest is None:
        foreign = [p.name for p in out.iterdir() if p.name != "index"]
        if not foreign:
            return
        if (out / "ledger.sqlite").is_file():
            ledger = Ledger.open(out / "ledger.sqlite")
            try:
                if ledger.get_manifest(run_fp) is not None:
                    return
            finally:
                ledger.close()
            stored_fp, stored_components = "unknown", None
        else:
            raise OpError(
                operation,
                "out_not_a_run",
                f"{out} is not empty and is not an xwalk run directory; choose another --out",
                exit_code=EXIT_RUNTIME,
            )
    else:
        stored_fp = str(manifest.get("run_fingerprint", "unknown"))
        stored_components = manifest.get("fingerprint_components")
        if stored_fp == run_fp:
            return
    differences = component_differences(
        stored_components if isinstance(stored_components, dict) else None, components
    )
    changed = ", ".join(sorted(differences)) or "unknown"
    raise OpError(
        operation,
        "run_fingerprint_mismatch",
        f"{out} holds run {stored_fp}, but this job is run {run_fp} (changed: {changed}); "
        f"nothing was overwritten -- choose a new --out",
        exit_code=EXIT_RUNTIME,
        data={
            "stored_fingerprint": stored_fp,
            "fingerprint": run_fp,
            "differences": {k: {"stored": a, "expected": b} for k, (a, b) in differences.items()},
        },
    )


def _run_exit_code(run_state: str, needs_review: int) -> int:
    if run_state == RunState.INTERRUPTED.value:
        return EXIT_INTERRUPTED
    if run_state in (RunState.ABORTED.value, RunState.FAILED.value):
        return EXIT_RUNTIME
    if needs_review or run_state == RunState.PARTIAL.value:
        return EXIT_ATTENTION
    return EXIT_OK


@dataclass(frozen=True)
class _RunView:
    run_dir: Path
    run_fingerprint: str
    manifest: dict[str, Any]


def _open_run(run_dir: str | Path, operation: str) -> tuple[_RunView, Ledger]:
    path = Path(run_dir)
    manifest = _read_manifest(path)
    if manifest is None or not (path / "ledger.sqlite").is_file():
        raise OpError(
            operation,
            "run_not_found",
            f"{path} is not an xwalk run directory (no manifest.json and ledger.sqlite)",
            exit_code=EXIT_USAGE,
        )
    return (
        _RunView(path, str(manifest["run_fingerprint"]), manifest),
        Ledger.open(path / "ledger.sqlite"),
    )


def _summarise_run(view: _RunView, ledger: Ledger, operation: str) -> OpResult:
    """The envelope fields every run-level operation shares, read from the ledger."""
    fp = view.run_fingerprint
    counts = {status.value: n for status, n in ledger.count_by_status(fp).items()}
    pending = ledger.pending_count(fp)
    if pending:
        counts[PENDING] = pending
    counts["total"] = ledger.count(fp) + pending
    invocation = ledger.last_invocation(fp)
    run_state = str(invocation["run_state"]) if invocation else "unknown"
    needs_review = counts.get(MatchStatus.NEEDS_REVIEW.value, 0)
    result = OpResult(
        operation=operation,
        exit_code=_run_exit_code(run_state, needs_review),
        run={"dir": str(view.run_dir), "run_fingerprint": fp, "run_state": run_state},
        counts=counts,
        usage=dict(invocation["usage"]) if invocation and invocation["usage"] else None,
        artifacts={
            name: str(view.run_dir / name)
            for name in ("mapping.csv", "results.jsonl", "manifest.json", "ledger.sqlite")
            if (view.run_dir / name).exists()
        },
    )
    if invocation:
        result.errors += [
            OpMessage(str(e["code"]), str(e["message"]), e.get("source_id"))
            for e in invocation.get("errors") or []
        ]
    duplicates = ledger.duplicate_targets(fp)
    if duplicates:
        result.warnings.append(
            OpMessage(
                "duplicate_targets",
                f"duplicate targets: {len(duplicates)} matched by more than one source "
                "(see manifest.json)",
            )
        )
    if needs_review:
        result.warnings.append(
            OpMessage(
                "needs_review",
                f"{needs_review} row(s) need review: xwalk review export --run {view.run_dir}",
            )
        )
    return result


# --- run ----------------------------------------------------------------------------


async def run_async(
    job: JobRef,
    out: str | Path,
    *,
    index_dir: str | Path | None = None,
    resume: bool = True,
    limit: int | None = None,
    max_calls: int | None = None,
    rebuild_index: bool = False,
    llm: LLMClient | None = None,
    encoder_factory: EncoderFactory | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> OpResult:
    """Match a job into `out`. See `run`."""
    operation = "match"
    if limit is not None and limit < 0:
        raise OpError(operation, "usage", "--limit must be >= 0", exit_code=EXIT_USAGE)
    if max_calls is not None and max_calls < 0:
        raise OpError(operation, "usage", "--max-calls must be >= 0", exit_code=EXIT_USAGE)
    spec = load_valid_job(job, operation=operation)
    _preflight(spec, operation, custom_encoder=encoder_factory is not None)
    out_dir = Path(out)
    index_root = Path(index_dir) if index_dir is not None else out_dir / "index"

    templates = spec.build_templates()
    targets, store = _read_targets(spec, operation)
    if llm is None:
        try:
            llm = spec.build_llm()
        except CredentialMissingError as exc:
            raise OpError(operation, "credential_missing", str(exc), exit_code=EXIT_USAGE) from None
    budgeted = BudgetedLLM(llm, CallBudget(max_calls))

    # The run's identity is known before any index or run directory is touched, so a
    # refusal leaves both exactly as they were.
    plans = plan_indexes(spec, targets, templates, index_root, encoder_factory=encoder_factory)
    components = spec.run_fingerprint_components(store=store, retrievers=plans, llm=budgeted)
    run_fp = hash_value(components)
    _check_out_dir(out_dir, run_fp, components, operation=operation)
    retrievers, actions = prepare_indexes(
        plans, targets, templates, rebuild=rebuild_index, operation=operation
    )
    matcher = spec.build_matcher(store=store, retrievers=retrievers, llm=budgeted)
    if matcher.run_fingerprint != run_fp:  # pragma: no cover - invariant
        raise RuntimeError("the built matcher does not have the planned run fingerprint")

    try:
        report = await run_batch(
            matcher,
            spec.build_source_records(),
            out=out_dir,
            resume=resume,
            limit=limit,
            manifest_extra={"job": spec.name, "model": spec.llm.model, "max_calls": max_calls},
            fingerprint_components=components,
            progress=progress,
        )
    except Exception as exc:
        raise OpError(
            operation,
            "run_exception",
            f"{type(exc).__name__}: {exc}",
            exit_code=EXIT_RUNTIME,
            run={"dir": str(out_dir), "run_fingerprint": run_fp, "run_state": "aborted"},
        ) from exc

    view = _RunView(out_dir, run_fp, _read_manifest(out_dir) or {})
    ledger = Ledger.open(out_dir / "ledger.sqlite")
    try:
        result = _summarise_run(view, ledger, operation)
    finally:
        ledger.close()
    # The budgeted client's accounting includes calls cancelled in flight (as calls
    # with unknown usage), which the batch report cannot see.
    result.usage = usage_to_dict(budgeted.usage)
    result.usage["limit"] = max_calls
    result.errors = [
        OpMessage(CALL_LIMIT_CODE, e.message, e.source_id)
        if budgeted.budget.exhausted and "call limit of" in e.message
        else OpMessage(e.code, e.message, e.source_id)
        for e in report.errors
    ]
    result.artifacts["index"] = str(index_root)
    result.data = {"indexes": actions, "job": spec.name, "model": spec.llm.model}
    result.lines += _run_lines(result, report.total)
    return result


def run(
    job: JobRef,
    out: str | Path,
    *,
    index_dir: str | Path | None = None,
    resume: bool = True,
    limit: int | None = None,
    max_calls: int | None = None,
    rebuild_index: bool = False,
    llm: LLMClient | None = None,
    encoder_factory: EncoderFactory | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> OpResult:
    """Validate, prepare indexes, and match every unfinished source record.

    `out` absent or empty starts a run; the same run fingerprint resumes it; another
    fingerprint or a foreign directory is refused (exit 3) and nothing is overwritten.
    `max_calls` caps upstream LLM requests in this invocation, retries and rewrites
    included; reaching it aborts the run (resume continues). `llm` replaces the job's
    client (credentials are then not needed). KeyboardInterrupt propagates after the
    run directory records `interrupted`.
    """
    return asyncio.run(
        run_async(
            job,
            out,
            index_dir=index_dir,
            resume=resume,
            limit=limit,
            max_calls=max_calls,
            rebuild_index=rebuild_index,
            llm=llm,
            encoder_factory=encoder_factory,
            progress=progress,
        )
    )


def _run_lines(result: OpResult, total: int) -> list[str]:
    assert result.run is not None
    lines = [f"matched {total} records into {result.run['dir']}"]
    lines.append(f"  run state     : {result.run['run_state']}")
    for status, count in sorted(result.counts.items()):
        if status != "total":
            lines.append(f"  {status:<14}: {count}")
    if result.usage is not None:
        lines.append(f"  tokens        : {result.usage['tokens']} in {result.usage['calls']} calls")
    return lines


# --- cluster (experimental; src/xwalk/cluster) ---------------------------------------


async def cluster_async(
    job: str | Path | ClusterJobSpec,
    out: str | Path,
    *,
    max_calls: int | None = None,
    llm: LLMClient | None = None,
    encoder: Encoder | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> OpResult:
    """Cluster a `kind: cluster` job into `out`. See `cluster`."""
    from xwalk.cluster.operation import cluster_operation

    return await cluster_operation(
        job, out, max_calls=max_calls, llm=llm, encoder=encoder, progress=progress
    )


def cluster(
    job: str | Path | ClusterJobSpec,
    out: str | Path,
    *,
    max_calls: int | None = None,
    llm: LLMClient | None = None,
    encoder: Encoder | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> OpResult:
    """Flat equivalence clustering of one collection (experimental).

    `job` is a `kind: cluster` job file or a `ClusterJobSpec`. `out` absent or empty
    starts a run; the same run fingerprint resumes it; another is refused (exit 3).
    `max_calls` caps upstream LLM requests in this invocation; reaching it aborts at a
    step boundary (exit 3, `call_limit_reached`) and a later call resumes. Exports:
    `members.csv`, `clusters.csv`, `unresolved.csv`, `decisions.jsonl`.
    """
    return asyncio.run(
        cluster_async(job, out, max_calls=max_calls, llm=llm, encoder=encoder, progress=progress)
    )


# --- inspect / explain --------------------------------------------------------------


def inspect(run_dir: str | Path) -> OpResult:
    """A run's identity, state, current counts, usage and review overlay.

    The exit code reflects the run (as `match` would report it), so a script can test
    a finished run without re-running it.
    """
    view, ledger = _open_run(run_dir, "inspect")
    try:
        result = _summarise_run(view, ledger, "inspect")
        fp = view.run_fingerprint
        reviews: dict[str, int] = {}
        for entry in review_history(ledger, fp):
            reviews[entry.state] = reviews.get(entry.state, 0) + 1
        invocation = ledger.last_invocation(fp)
        snapshot_size = ledger.snapshot_size(fp)
        result.data = {
            "job": view.manifest.get("job"),
            "model": view.manifest.get("model"),
            "library_version": view.manifest.get("library_version"),
            "inspected_with": __version__,
            "fingerprint_components": view.manifest.get("fingerprint_components"),
            "last_invocation": invocation,
            "snapshot": {"recorded": snapshot_size is not None, "size": snapshot_size},
            "history_results": ledger.history_count(fp),
            "removed_sources": len(ledger.removed_sources(fp)),
            "reviews": reviews,
        }
    finally:
        ledger.close()
    run = result.run or {}
    result.lines += [
        f"run {run.get('run_fingerprint')} in {run.get('dir')}",
        f"  job           : {result.data['job']} ({result.data['model']})",
        f"  written by    : xwalk {result.data['library_version']}",
        f"  run state     : {run.get('run_state')}",
    ]
    for status, count in sorted(result.counts.items()):
        result.lines.append(f"  {status:<14}: {count}")
    if result.usage:
        result.lines.append(
            f"  last usage    : {result.usage.get('tokens')} tokens in "
            f"{result.usage.get('calls')} calls"
        )
    result.lines.append(f"  history       : {result.data['history_results']} results")
    if reviews:
        result.lines.append(
            "  reviews       : " + ", ".join(f"{n} {s}" for s, n in sorted(reviews.items()))
        )
    return result


def _attempt_summary(attempt: Any) -> dict[str, Any]:
    return {
        "index": attempt.index,
        "query": attempt.query,
        "proposal": None
        if attempt.proposal is None
        else {"kind": attempt.proposal.kind, "value": attempt.proposal.value},
        "candidate_count": attempt.candidate_count,
        "top_candidates": [c.record.id for c in attempt.candidates[:5]],
        "chosen_id": attempt.chosen_id,
        "resolution": attempt.resolution,
        "primary_score": attempt.primary_score,
        "verifier_decision": attempt.verifier_decision,
        "verifier_score": attempt.verifier_score,
        "reason": None if attempt.reason is None else attempt.reason.value,
        "error": attempt.error,
        "explanation": attempt.explanation,
        "usage": usage_to_dict(attempt.usage),
    }


def explain(run_dir: str | Path, source_id: str, *, full: bool = False) -> OpResult:
    """Why one source got its answer: the decision, every attempt, the review overlay
    and the source's history. `full=True` adds the complete stored result."""
    view, ledger = _open_run(run_dir, "explain")
    try:
        fp = view.run_fingerprint
        entry = next((e for e in ledger.iter_current(fp) if e.source_id == source_id), None)
        history = [h for h in ledger.iter_history(fp) if h.result.source_id == source_id]
        if entry is None and not history:
            raise OpError(
                "explain",
                "source_not_found",
                f"source {source_id!r} is not in run {fp}",
                exit_code=EXIT_USAGE,
            )
        applied = {
            str(h.review["result_key"]): h.review
            for h in review_history(ledger, fp)
            if h.state == "applied"
        }
        overlay = {row.result_key: row for row in adjudicated(ledger, fp)}
    finally:
        ledger.close()

    result = OpResult(
        operation="explain",
        run={
            "dir": str(view.run_dir),
            "run_fingerprint": fp,
            "run_state": view.manifest.get("run_state", "unknown"),
        },
    )
    current: MatchResult | None = entry.result if entry is not None else None
    data: dict[str, Any] = {
        "source_id": source_id,
        "in_current_snapshot": entry is not None,
        "library_version": view.manifest.get("library_version"),
        "status": PENDING if entry is not None and current is None else None,
        "decision": None,
        "attempts": [],
        "review": None,
        "history": [
            {
                "revision": h.revision,
                "current": h.current,
                "source_hash": h.result.source_hash,
                "status": h.result.status.value,
                "matched_id": h.result.matched_id,
                "reason": h.result.reason.value,
            }
            for h in history
        ],
    }
    if current is not None:
        data["status"] = current.status.value
        data["decision"] = {
            "result_key": current.result_key,
            "revision": entry.revision if entry is not None else None,
            "matched_id": current.matched_id,
            "confidence": current.confidence,
            "status": current.status.value,
            "reason": current.reason.value,
            "explanation": current.explanation,
            "usage": usage_to_dict(current.usage),
        }
        data["attempts"] = [_attempt_summary(a) for a in current.attempts]
        review = applied.get(current.result_key)
        row = overlay.get(current.result_key)
        if review is not None and row is not None:
            data["review"] = {
                "decision": review["decision"],
                "corrected_target_id": review.get("corrected_target_id"),
                "reviewer": review["reviewer"],
                "review_note": review["review_note"],
                "reviewed_at": review["reviewed_at"],
                "final_matched_id": row.final_target_id,
                "final_status": row.final_status.value,
            }
        if full:
            data["result"] = result_to_dict(current)
    elif entry is None:
        result.warnings.append(
            OpMessage(
                "removed_source",
                f"{source_id!r} is not in the current snapshot; only its history remains",
                source_id,
            )
        )
    result.data = data
    result.lines += _explain_lines(data, fp)
    return result


def _explain_lines(data: Mapping[str, Any], fp: str) -> list[str]:
    lines = [f"source {data['source_id']} in run {fp}"]
    decision = data["decision"]
    if decision is None:
        lines.append(f"  status        : {data['status'] or 'not in current snapshot'}")
    else:
        lines += [
            f"  decision      : {decision['status']} -> {decision['matched_id'] or '-'} "
            f"(confidence {decision['confidence']}, reason {decision['reason']})",
            f"  explanation   : {decision['explanation']}",
        ]
    for attempt in data["attempts"]:
        lines.append(
            f"  attempt {attempt['index']}: query {attempt['query']!r}, "
            f"{attempt['candidate_count']} candidates, chose {attempt['chosen_id'] or '-'}, "
            f"score {attempt['primary_score']}, verifier {attempt['verifier_decision'] or '-'}"
            + (f", error {attempt['error']}" if attempt["error"] else "")
        )
    review = data["review"]
    if review is not None:
        lines.append(
            f"  review        : {review['decision']} by {review['reviewer']} -> "
            f"{review['final_status']} {review['final_matched_id'] or '-'}"
        )
    if len(data["history"]) > 1:
        lines.append(f"  history       : {len(data['history'])} results for this source")
    return lines


# --- export -------------------------------------------------------------------------


def list_results(
    run_dir: str | Path,
    *,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
) -> OpResult:
    """One page of a run's current view (the model's decisions, ordered by source id).

    `limit` is 1 to `MAX_PAGE_SIZE`; `status` keeps only rows with that status
    (`matched`, `needs_review`, `unmatched`, `failed` or `pending`). `data` carries the
    rows, `total` (rows matching the filter) and `next_offset` (None on the last page).
    Use `explain` for one row's attempts and review, `export` for the whole table.
    """
    operation = "results"
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise OpError(
            operation,
            "usage",
            f"limit must be between 1 and {MAX_PAGE_SIZE}, got {limit}",
            exit_code=EXIT_USAGE,
        )
    if offset < 0:
        raise OpError(
            operation, "usage", f"offset must be >= 0, got {offset}", exit_code=EXIT_USAGE
        )
    statuses = {s.value for s in MatchStatus} | {PENDING}
    if status is not None and status not in statuses:
        raise OpError(
            operation,
            "usage",
            f"unknown status {status!r}; choose from {sorted(statuses)}",
            exit_code=EXIT_USAGE,
        )
    view, ledger = _open_run(run_dir, operation)
    try:
        fp = view.run_fingerprint
        rows: list[dict[str, Any]] = []
        total = 0
        for row in ledger.iter_current_summaries(fp):
            if status is not None and row["status"] != status:
                continue
            if offset <= total < offset + limit:
                rows.append(row)
            total += 1
        invocation = ledger.last_invocation(fp)
    finally:
        ledger.close()
    end = offset + len(rows)
    result = OpResult(
        operation=operation,
        run={
            "dir": str(view.run_dir),
            "run_fingerprint": fp,
            "run_state": str(invocation["run_state"]) if invocation else "unknown",
        },
        counts={"total": total, "returned": len(rows)},
        data={
            "rows": rows,
            "offset": offset,
            "limit": limit,
            "status": status,
            "total": total,
            "next_offset": end if end < total else None,
        },
    )
    for row in rows:
        result.lines.append(
            f"{row['source_id']}\t{row['status']}\t{row['matched_id'] or ''}\t"
            f"{'' if row['confidence'] is None else row['confidence']}"
        )
    if result.data["next_offset"] is not None:
        result.lines.append(f"(rows {offset + 1}-{end} of {total}; next --offset {end})")
    return result


def _write_reviewed_csv(ledger: Ledger, fp: str, path: Path) -> tuple[int, int]:
    """The current view with the review overlay applied, next to the model's decision.

    Returns `(rows, reviewed_rows)`. The ledger is not modified: the original decision
    stays the record of what the model said.
    """
    applied = {
        str(h.review["result_key"]): h.review
        for h in review_history(ledger, fp)
        if h.state == "applied"
    }
    overlay = {row.result_key: row for row in adjudicated(ledger, fp)}
    rows = reviewed = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEWED_COLUMNS))
        writer.writeheader()
        for entry in ledger.iter_current(fp):
            blank = {column: "" for column in REVIEWED_COLUMNS}
            result = entry.result
            if result is None:
                writer.writerow({**blank, "source_id": entry.source_id, "final_status": PENDING})
                rows += 1
                continue
            row = overlay[result.result_key]
            review = applied.get(result.result_key)
            reviewed += review is not None
            writer.writerow(
                {
                    **blank,
                    "source_id": result.source_id,
                    "final_matched_id": row.final_target_id or "",
                    "final_status": row.final_status.value,
                    "review_decision": "" if review is None else review["decision"],
                    "corrected_target_id": ""
                    if review is None
                    else review.get("corrected_target_id") or "",
                    "reviewer": row.reviewer or "",
                    "review_note": row.review_note,
                    "reviewed_at": row.reviewed_at or "",
                    "model_matched_id": result.matched_id or "",
                    "model_status": result.status.value,
                    "model_reason": result.reason.value,
                    "confidence": "" if result.confidence is None else result.confidence,
                    "result_key": result.result_key,
                }
            )
            rows += 1
    return rows, reviewed


def _write_review_history(ledger: Ledger, fp: str, path: Path) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for entry in review_history(ledger, fp):
            line = {"state": entry.state, "review": entry.review}
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            written += 1
    return written


def export(run_dir: str | Path, view: str, out: str | Path) -> OpResult:
    """Write one explicit view of a run.

    - ``raw``: the current view as the model decided it (`mapping.csv` columns).
    - ``reviewed``: the current view with the review overlay applied; each row keeps the
      model's decision beside the final one.
    - ``history``: every result ever committed (JSONL), and, when reviews exist, every
      review decision with its state in ``<out>.reviews.jsonl``.
    """
    if view not in EXPORT_VIEWS:
        raise OpError(
            "export",
            "usage",
            f"unknown view {view!r}; choose from {', '.join(EXPORT_VIEWS)}",
            exit_code=EXIT_USAGE,
        )
    run_view, ledger = _open_run(run_dir, "export")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = run_view.run_fingerprint
    result = OpResult(
        operation="export",
        run={
            "dir": str(run_view.run_dir),
            "run_fingerprint": fp,
            "run_state": run_view.manifest.get("run_state", "unknown"),
        },
    )
    try:
        if view == "raw":
            rows = export_mapping_csv(ledger, fp, out_path)
            result.counts = {"rows": rows}
        elif view == "reviewed":
            rows, reviewed = _write_reviewed_csv(ledger, fp, out_path)
            result.counts = {"rows": rows, "reviewed": reviewed}
        else:
            rows = export_history_jsonl(ledger, fp, out_path)
            result.counts = {"rows": rows}
            reviews_path = out_path.with_name(out_path.stem + ".reviews.jsonl")
            reviews = sum(1 for _ in review_history(ledger, fp))
            if reviews:
                _write_review_history(ledger, fp, reviews_path)
                result.counts["reviews"] = reviews
                result.artifacts["review_history"] = str(reviews_path)
    finally:
        ledger.close()
    result.artifacts[view] = str(out_path)
    result.data = {"view": view}
    result.lines.append(f"exported {result.counts['rows']} {view} rows to {out_path}")
    return result


# --- review worksheet ---------------------------------------------------------------


def review_export(run_dir: str | Path, out: str | Path) -> OpResult:
    """The review worksheet: current `needs_review` rows with blank decision columns."""
    from xwalk.review import export_review

    view, ledger = _open_run(run_dir, "review export")
    try:
        count = export_review(ledger, view.run_fingerprint, out)
    finally:
        ledger.close()
    result = OpResult(operation="review export", counts={"rows": count})
    result.artifacts["review"] = str(out)
    result.lines.append(f"exported {count} rows to {out}")
    return result


def review_apply(run_dir: str | Path, reviewed: str | Path, job: JobRef) -> OpResult:
    """Record reviewer decisions. Rejected rows make the exit code 1."""
    from xwalk.review import apply_review, read_review

    spec = load_valid_job(job, operation="review apply")
    _, store = _read_targets(spec, "review apply")
    _, ledger = _open_run(run_dir, "review apply")
    try:
        report = apply_review(
            ledger, read_review(reviewed), target_store_fingerprint=store.fingerprint
        )
    finally:
        ledger.close()
    result = OpResult(
        operation="review apply",
        exit_code=EXIT_ATTENTION if report.rejected else EXIT_OK,
        counts={"applied": report.applied, "rejected": len(report.rejected)},
    )
    result.warnings += [
        OpMessage("review_rejected", f"{key}: {why}") for key, why in report.rejected
    ]
    result.lines.append(f"applied {report.applied} decisions")
    return result


# --- init ---------------------------------------------------------------------------


def bundled_example() -> Path:
    """The directory of the bundled quickstart job (job.yaml, data, slots)."""
    return Path(str(resources.files("xwalk") / "resources" / "quickstart"))


def init(dest: str | Path) -> OpResult:
    """Copy the bundled offline quickstart into `dest`. An existing non-empty `dest`
    is refused (exit 3); nothing is overwritten."""
    target = Path(dest)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise OpError(
            "init",
            "destination_not_empty",
            f"{target} already exists and is not empty; choose another directory",
            exit_code=EXIT_RUNTIME,
        )
    source = bundled_example()
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for item in sorted(source.iterdir()):
        if item.is_file():
            shutil.copyfile(item, target / item.name)
            copied.append(item.name)
    result = OpResult(operation="init", counts={"files": len(copied)})
    result.artifacts = {"dir": str(target), "job": str(target / "job.yaml")}
    result.data = {"files": copied}
    result.lines += [
        f"created {target} ({', '.join(copied)})",
        f"next: xwalk validate --job {target / 'job.yaml'}",
    ]
    return result


def interrupted(operation: str, run_dir: str | Path | None) -> OpResult:
    """The envelope for a Ctrl-C: the run directory's recorded state when there is one."""
    result: OpResult | None = None
    if run_dir is not None:
        try:
            view, ledger = _open_run(run_dir, operation)
        except (OpError, OSError):
            result = None
        else:
            try:
                result = _summarise_run(view, ledger, operation)
            finally:
                ledger.close()
    if result is None:
        result = OpResult(operation=operation)
    result.exit_code = EXIT_INTERRUPTED
    result.errors.append(OpMessage("interrupted", "interrupted by the user"))
    return result


__all__ = [
    "EXIT_ATTENTION",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_USAGE",
    "EXPORT_VIEWS",
    "SCHEMA_VERSION",
    "OpError",
    "OpMessage",
    "OpResult",
    "PlannedIndex",
    "bundled_example",
    "cluster",
    "cluster_async",
    "explain",
    "export",
    "index",
    "init",
    "inspect",
    "interrupted",
    "load_valid_job",
    "plan_indexes",
    "prepare_indexes",
    "review_apply",
    "review_export",
    "run",
    "run_async",
    "validate",
]

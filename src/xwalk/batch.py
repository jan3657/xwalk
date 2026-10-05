"""Batch matching: the mapping platform.

Point at a source, get a resumable run directory containing a ledger and three exports.

Lifecycle of one `run_batch` invocation (docs/claude-upgrade/CONTRACTS.md sections 2-4):

1. The source is read once, start to end, as a stream. Every record joins the
   invocation's *snapshot* — its ordered ``(source_id, source_hash)`` list — even when
   it is not processed (already done, or beyond ``limit``). A repeated source id is an
   error.
2. Unfinished records go to a bounded pool of record tasks. A record is finished when
   the ledger holds a result for its current content that is not ``failed``.
3. Each result is committed in its own transaction the moment it is ready. The snapshot
   is committed in one transaction once the source has been read to the end.
4. A ``fatal_provider_failure`` result is never committed; it aborts the run. So does
   any exception in a record task, and cancellation or Ctrl-C interrupts it. In every
   case scheduling stops, in-flight record tasks are cancelled and awaited, the exports
   and the manifest are written from the current view with the run state, and only then
   is the ledger closed.

The current view (one row per snapshot entry) feeds every export; superseded results
stay in the ledger as history (`Ledger.iter_history`, `export_history_jsonl`).
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from xwalk import __version__
from xwalk.fingerprint import hash_record, hash_value, result_key
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet
from xwalk.records import DecisionReason, MatchResult, MatchStatus, Record, Usage
from xwalk.retrieval.base import Retriever
from xwalk.review import adjudicated
from xwalk.serde import result_to_dict
from xwalk.stages.select import SelectorPolicy
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet

MAPPING_COLUMNS = (
    "source_id",
    "matched_id",
    "confidence",
    "status",
    "reason",
    "explanation",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "llm_calls",
    "elapsed_seconds",
    # Appended in 0.2 so positional readers of the 0.1 columns keep working.
    "unknown_calls",
    "cache_hits",
)

# The status of a snapshot entry that has no result yet (`--limit`, an interrupted run).
PENDING = "pending"


class RunState(str, Enum):
    """How an invocation ended. Precedence when several apply: interrupted, aborted,
    failed, partial, complete."""

    COMPLETE = "complete"  # every current source has a result and none failed
    PARTIAL = "partial"  # some current sources are pending (a limit was used)
    FAILED = "failed"  # some current results are `failed`; resume retries them
    ABORTED = "aborted"  # a fatal provider failure or an exception stopped the run
    INTERRUPTED = "interrupted"  # cancelled or Ctrl-C


class DuplicateSourceIdError(ValueError):
    """Two records in one source share an id; the snapshot would be ambiguous."""


@dataclass(frozen=True)
class BatchError:
    code: str
    message: str
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.source_id is not None:
            data["source_id"] = self.source_id
        return data


def usage_to_dict(usage: Usage) -> dict[str, Any]:
    """Usage for a manifest: raw counts plus the display string that never hides
    unknown usage behind a zero."""
    return {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "unknown_calls": usage.unknown_calls,
        "cache_hits": usage.cache_hits,
        "tokens": usage.describe_tokens(),
    }


def run_fingerprint_components(
    *,
    templates: TemplateSet,
    prompts: PromptSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    llm: LLMClient,
    policy: MatchPolicy,
    selector_policy: SelectorPolicy,
    retriever_limit: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Everything whose change should invalidate prior results, as a JSON-safe dict.

    Deliberately excludes credentials, output paths, and concurrency — none of them
    change what a result means. `retriever_limit` and `rrf_k` must match the values
    handed to `Matcher`, or a resumed run will reuse results produced at another depth.
    The dict is stored in the manifest so a refused resume can name what changed.
    """
    components: dict[str, Any] = {
        "library_version": __version__,
        "templates": templates.fingerprint,
        "prompts": prompts.fingerprint,
        "target": store.fingerprint,
        "retrievers": sorted(f"{r.name}:{r.fingerprint}" for r in retrievers),
        # Depth and fusion constant change which candidates exist at all, so they
        # belong here: raising k from 20 to 100 must not silently reuse old results.
        "retrieval": {
            "depths": sorted(
                f"{r.name}:{getattr(r, 'default_limit', retriever_limit)}" for r in retrievers
            ),
            "retriever_limit": retriever_limit,
            "rrf_k": rrf_k,
            "retriever_timeout": policy.retriever_timeout,
        },
        "llm": llm.fingerprint,
        "policy": {
            "max_attempts": policy.max_attempts,
            "accept_at": policy.accept_at,
            "review_floor": policy.review_floor,
            "verify_band": list(policy.verify_band) if policy.verify_band else None,
            "audit_rate": policy.audit_rate,
            "legacy_id_resolution": policy.legacy_id_resolution,
        },
        "selector_policy": {
            "max_candidates": selector_policy.max_candidates,
            "max_candidate_tokens": selector_policy.max_candidate_tokens,
        },
    }
    # The live encoder of each dense retriever: model, revision, prefixes, normalization.
    # An index fingerprint alone misses an encoder reopened with different settings.
    encoders = {
        r.name: dict(identity)
        for r in retrievers
        if (identity := getattr(r, "encoder_identity", None)) is not None
    }
    if encoders:
        components["encoders"] = encoders
    return components


def build_run_fingerprint(
    *,
    templates: TemplateSet,
    prompts: PromptSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    llm: LLMClient,
    policy: MatchPolicy,
    selector_policy: SelectorPolicy,
    retriever_limit: int = 20,
    rrf_k: int = 60,
) -> str:
    """The hash of `run_fingerprint_components`."""
    return hash_value(
        run_fingerprint_components(
            templates=templates,
            prompts=prompts,
            store=store,
            retrievers=retrievers,
            llm=llm,
            policy=policy,
            selector_policy=selector_policy,
            retriever_limit=retriever_limit,
            rrf_k=rrf_k,
        )
    )


@dataclass(frozen=True)
class BatchReport:
    """The outcome of one invocation.

    `total` is the number of sources in the current snapshot, `pending` how many of
    them have no result yet. `usage` covers the calls this invocation made for records
    it finished (including a fatal one); calls made by records cancelled mid-flight are
    not included. The query methods read the current view from the ledger.
    """

    run_fingerprint: str
    out_dir: Path
    total: int
    usage: Usage
    _ledger_path: Path
    run_state: RunState = RunState.COMPLETE
    pending: int = 0
    errors: tuple[BatchError, ...] = field(default_factory=tuple)

    def _ledger(self) -> Ledger:
        return Ledger.open(self._ledger_path)

    def by_status(self) -> dict[MatchStatus, int]:
        ledger = self._ledger()
        try:
            return ledger.count_by_status(self.run_fingerprint)
        finally:
            ledger.close()

    def needs_review(self) -> list[MatchResult]:
        ledger = self._ledger()
        try:
            return [
                r
                for r in ledger.iter_results(self.run_fingerprint)
                if r.status is MatchStatus.NEEDS_REVIEW
            ]
        finally:
            ledger.close()

    def duplicate_targets(self) -> dict[str, list[str]]:
        ledger = self._ledger()
        try:
            return ledger.duplicate_targets(self.run_fingerprint)
        finally:
            ledger.close()


class _FatalResult(Exception):
    """Carries a `fatal_provider_failure` result out of its record task, uncommitted."""

    def __init__(self, result: MatchResult) -> None:
        super().__init__(result.explanation)
        self.result = result


def _fatal_error(result: MatchResult) -> BatchError:
    errors = [a.error for a in result.attempts if a.error]
    message = errors[-1] if errors else (result.explanation or "fatal provider failure")
    return BatchError(
        code=DecisionReason.FATAL_PROVIDER_FAILURE.value,
        message=message,
        source_id=result.source_id,
    )


async def _cancel_and_wait(tasks: set[asyncio.Task[MatchResult]]) -> None:
    """Cancel every task and wait until each has really finished.

    A second cancellation of the caller while waiting does not cut the wait short:
    returning early would let a record task commit after the ledger is closed.
    """
    for task in tasks:
        task.cancel()
    pending = set(tasks)
    while pending:
        try:
            _, pending = await asyncio.wait(pending)
        except asyncio.CancelledError:
            for task in pending:
                task.cancel()


def _run_state(ledger: Ledger, run_fp: str) -> RunState:
    counts = ledger.count_by_status(run_fp)
    if counts.get(MatchStatus.FAILED):
        return RunState.FAILED
    if ledger.pending_count(run_fp):
        return RunState.PARTIAL
    return RunState.COMPLETE


async def run_batch(
    matcher: Matcher,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    limit: int | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    fingerprint_components: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport:
    """Match every unfinished source record and export the current view.

    `limit` caps how many unfinished records this invocation processes; the source is
    still read to the end so the snapshot is complete, and the rest stay pending.
    `resume=False` recomputes every record; the earlier results go to history.

    A fatal provider failure returns a report with `run_state=ABORTED` and the error in
    `errors`. An exception from a record task or the ledger is re-raised, and so is
    cancellation (after the run is recorded as interrupted).
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger.sqlite"
    ledger = Ledger.open(ledger_path)
    run_fp = matcher.run_fingerprint
    concurrency = matcher.policy.concurrency

    try:
        manifest: dict[str, Any] = {
            "run_fingerprint": run_fp,
            "library_version": __version__,
            "target_fingerprint": matcher.store_fingerprint,
            **dict(manifest_extra or {}),
        }
        if fingerprint_components is not None:
            manifest["fingerprint_components"] = dict(fingerprint_components)
        ledger.put_manifest(run_fp, manifest)
        invocation = ledger.begin_invocation(
            run_fp, library_version=__version__, resume=resume, limit=limit
        )
    except BaseException:
        ledger.close()
        raise

    usage = Usage.zero()
    errors: list[BatchError] = []
    in_flight: set[asyncio.Task[MatchResult]] = set()
    snapshot_id: int | None = None
    state: RunState | None = None
    raised: BaseException | None = None

    async def one(record: Record) -> MatchResult:
        result = await matcher.match(record)
        if result.reason is DecisionReason.FATAL_PROVIDER_FAILURE:
            raise _FatalResult(result)  # never committed: resume must retry it
        await ledger.put_result(result)
        return result

    def settle(done: set[asyncio.Task[MatchResult]], *, report: bool = True) -> None:
        """Account for finished tasks. Raises the first unexpected exception, after
        reading every outcome so none is left unretrieved."""
        nonlocal usage
        unexpected: BaseException | None = None
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is None:
                result = task.result()
                usage = usage + result.usage
                if report and progress is not None:
                    progress(result)
            elif isinstance(exc, _FatalResult):
                usage = usage + exc.result.usage
                errors.append(_fatal_error(exc.result))
            elif unexpected is None:
                unexpected = exc
        if unexpected is not None:
            raise unexpected

    async def wait_for_one() -> None:
        done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        in_flight.difference_update(done)
        settle(done)

    try:
        seen: set[str] = set()
        entries: list[tuple[str, str]] = []
        scheduled = 0
        source_read = True
        for record in source:
            if record.id in seen:
                raise DuplicateSourceIdError(
                    f"source id {record.id!r} appears more than once; ids must be unique"
                )
            seen.add(record.id)
            source_hash = hash_record(record)
            entries.append((record.id, source_hash))
            key = result_key(run_fp, record.id, source_hash)
            if resume and ledger.is_settled(key):
                continue
            if limit is not None and scheduled >= limit:
                continue
            while len(in_flight) >= concurrency and not errors:
                await wait_for_one()
            if errors:  # a fatal failure: stop scheduling, leave the snapshot unrecorded
                source_read = False
                break
            scheduled += 1
            in_flight.add(asyncio.create_task(one(record)))

        if source_read:
            snapshot_id = ledger.put_snapshot(run_fp, entries)
        while in_flight and not errors:
            await wait_for_one()
        state = RunState.ABORTED if errors else None
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        state, raised = RunState.INTERRUPTED, exc
    except BaseException as exc:
        state, raised = RunState.ABORTED, exc
        errors.append(BatchError(code="exception", message=f"{type(exc).__name__}: {exc}"))

    try:
        # Cleanup, in contract order: cancel and await every record task, record the
        # outcome, write the exports, and only then close the ledger.
        if in_flight:
            await _cancel_and_wait(in_flight)
            try:
                settle(in_flight, report=False)
            except BaseException as exc:  # already aborting; keep the first cause
                if raised is None:
                    state, raised = RunState.ABORTED, exc
                    errors.append(
                        BatchError(code="exception", message=f"{type(exc).__name__}: {exc}")
                    )
            in_flight.clear()
        if state is None:
            state = _run_state(ledger, run_fp)
        ledger.finish_invocation(
            invocation,
            run_state=state.value,
            snapshot_id=snapshot_id,
            usage=usage_to_dict(usage),
            errors=[e.to_dict() for e in errors],
        )
        export_results_jsonl(ledger, run_fp, out_dir / "results.jsonl")
        export_mapping_csv(ledger, run_fp, out_dir / "mapping.csv")
        export_manifest(ledger, run_fp, out_dir / "manifest.json")
        report = BatchReport(
            run_fingerprint=run_fp,
            out_dir=out_dir,
            total=ledger.count(run_fp) + ledger.pending_count(run_fp),
            usage=usage,
            _ledger_path=ledger_path,
            run_state=state,
            pending=ledger.pending_count(run_fp),
            errors=tuple(errors),
        )
    except BaseException:
        if raised is None:
            raise
        report = None  # the original failure is the one worth reporting
    finally:
        ledger.close()

    if raised is not None:
        raise raised
    assert report is not None
    return report


def run_batch_sync(
    matcher: Matcher,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    limit: int | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    fingerprint_components: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport:
    return asyncio.run(
        run_batch(
            matcher,
            source,
            out=out,
            resume=resume,
            limit=limit,
            manifest_extra=manifest_extra,
            fingerprint_components=fingerprint_components,
            progress=progress,
        )
    )


# --- exports --------------------------------------------------------------------


def export_results_jsonl(ledger: Ledger, run_fingerprint: str, path: str | Path) -> int:
    """Current results, one JSON object per line. Pending sources have no line."""
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for result in ledger.iter_results(run_fingerprint):
            handle.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")
            written += 1
    return written


def export_history_jsonl(ledger: Ledger, run_fingerprint: str, path: str | Path) -> int:
    """Every result ever committed for the run, with its revision and whether it is
    current. Superseded source versions, removed sources and retried failures included."""
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for entry in ledger.iter_history(run_fingerprint):
            line = {
                "current": entry.current,
                "revision": entry.revision,
                "result": result_to_dict(entry.result),
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    return written


def _pending_row(source_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in MAPPING_COLUMNS}
    row.update({"source_id": source_id, "status": PENDING, "reason": PENDING})
    return row


def export_mapping_csv(
    ledger: Ledger,
    run_fingerprint: str,
    path: str | Path,
    *,
    use_review: bool = False,
) -> int:
    """The deliverable: one row per current source, pending ones with status `pending`.

    `use_review=True` writes the adjudicated view instead.
    """
    path = Path(path)
    written = 0
    reviewed = (
        {row.result_key: row for row in adjudicated(ledger, run_fingerprint)} if use_review else {}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MAPPING_COLUMNS))
        writer.writeheader()

        for entry in ledger.iter_current(run_fingerprint):
            result = entry.result
            if result is None:
                writer.writerow(_pending_row(entry.source_id))
            elif use_review:
                row = reviewed[result.result_key]
                blank = {column: "" for column in MAPPING_COLUMNS}
                writer.writerow(
                    {
                        **blank,
                        "source_id": row.source_id,
                        "matched_id": row.final_target_id or "",
                        "confidence": "" if row.confidence is None else row.confidence,
                        "status": row.final_status.value,
                        "reason": "reviewed" if row.reviewer else row.model_status.value,
                        "explanation": row.review_note,
                    }
                )
            else:
                writer.writerow(
                    {
                        "source_id": result.source_id,
                        "matched_id": result.matched_id or "",
                        "confidence": "" if result.confidence is None else result.confidence,
                        "status": result.status.value,
                        "reason": result.reason.value,
                        "explanation": result.explanation,
                        "attempts": len(result.attempts),
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "llm_calls": result.usage.calls,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                        "unknown_calls": result.usage.unknown_calls,
                        "cache_hits": result.usage.cache_hits,
                    }
                )
            written += 1
    return written


def export_manifest(ledger: Ledger, run_fingerprint: str, path: str | Path) -> None:
    """The manifest, regenerated from the ledger alone.

    `run_state`, `usage` and `errors` describe the latest invocation; `counts`,
    `duplicate_targets` and `removed_sources` describe the current view.
    """
    manifest = dict(ledger.get_manifest(run_fingerprint) or {})
    counts: dict[str, int] = {
        status.value: count for status, count in ledger.count_by_status(run_fingerprint).items()
    }
    pending = ledger.pending_count(run_fingerprint)
    if pending:
        counts[PENDING] = pending
    manifest["counts"] = counts
    manifest["duplicate_targets"] = ledger.duplicate_targets(run_fingerprint)
    manifest["removed_sources"] = len(ledger.removed_sources(run_fingerprint))
    manifest["history"] = {"results": ledger.history_count(run_fingerprint)}
    manifest["ledger_schema_version"] = ledger.schema_version
    snapshot_size = ledger.snapshot_size(run_fingerprint)
    manifest["snapshot"] = {"recorded": snapshot_size is not None, "size": snapshot_size}
    invocation = ledger.last_invocation(run_fingerprint)
    if invocation is not None:
        manifest["run_state"] = invocation["run_state"]
        manifest["usage"] = invocation["usage"]
        manifest["errors"] = invocation["errors"]
        manifest["limit"] = invocation["record_limit"]
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

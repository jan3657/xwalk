"""The CLI: argument parsing and formatting over `xwalk.ops`.

Every subcommand parses arguments, calls one operation, and prints its `OpResult`:
human lines on stdout, or with `--json` exactly one JSON envelope (schema 1) on stdout.
Warnings, errors and progress go to stderr, so stdout stays parseable on success and on
failure.

Exit codes (docs/claude-upgrade/CONTRACTS.md section 8):
  0    complete: no review rows, no failed rows
  1    attention: rows need review, a partial run from --limit, or rejected review rows
  2    usage or configuration error: bad flags, invalid job file, missing credential
       or optional extra
  3    runtime failure: aborted or failed run, IO error, incompatible index or run
       directory
  130  interrupted (Ctrl-C)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xwalk import __version__
from xwalk.ops import (
    EXIT_ATTENTION,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXPORT_VIEWS,
    OpError,
    OpMessage,
    OpResult,
    failure_result,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xwalk.config import JobSpec
    from xwalk.decide.base import DecisionClient
    from xwalk.decide.fit import FitPoint
    from xwalk.evaluate.ablate import MatcherConfig
    from xwalk.llm.base import LLMClient
    from xwalk.matcher import Matcher
    from xwalk.prompts.contract import PromptSet

__all__ = [
    "EXIT_ATTENTION",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_USAGE",
    "main",
]


def _run_fingerprint_of(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["run_fingerprint"])


def _build_parser() -> argparse.ArgumentParser:
    # `--json` lives on every leaf parser through this parent. argparse binds an option
    # to whichever parser is active when it is seen, so a flag declared only on the
    # top-level parser but typed after the subcommand would be unrecognised.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        help="print one JSON result object (schema 1) on stdout; logs stay on stderr",
    )

    parser = argparse.ArgumentParser(
        prog="xwalk", description="Match records from any collection to any other."
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", parents=[common], help="copy the bundled quickstart job")
    init.add_argument("dest", nargs="?", default="xwalk-quickstart", help="new directory")

    validate = sub.add_parser(
        "validate",
        aliases=["doctor"],
        parents=[common],
        help="offline preflight of a job: no model calls",
    )
    validate.add_argument("--job", required=True)
    validate.add_argument(
        "--no-credentials",
        dest="check_credentials",
        action="store_false",
        help="do not require the llm.api_key_env variable to be set",
    )
    validate.add_argument(
        "--no-scan",
        dest="scan_records",
        action="store_false",
        help="do not read the collections (skips duplicate-id checks)",
    )

    index = sub.add_parser("index", parents=[common], help="build or open retriever indexes")
    index.add_argument("--job", required=True)
    index.add_argument("--out", required=True, help="index directory")
    index.add_argument("--rebuild-index", action="store_true", help="replace an incompatible index")

    match = sub.add_parser("match", parents=[common], help="run a matching job")
    match.add_argument("--job", required=True)
    match.add_argument("--out", required=True, help="run directory")
    match.add_argument("--index", default=None, help="index directory (default: <out>/index)")
    match.add_argument("--resume", action="store_true", default=True)
    match.add_argument("--no-resume", dest="resume", action="store_false")
    match.add_argument("--limit", type=int, default=None, help="process only the first N records")
    match.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="cap upstream LLM requests in this invocation (retries and rewrites included)",
    )
    match.add_argument("--rebuild-index", action="store_true", help="replace an incompatible index")

    cluster = sub.add_parser(
        "cluster",
        parents=[common],
        help="group one collection into clusters of equivalent records (experimental)",
    )
    cluster.add_argument("--job", required=True, help="a kind: cluster job file")
    cluster.add_argument("--out", required=True, help="run directory")
    cluster.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="cap upstream LLM requests in this invocation; reaching it aborts (resumable)",
    )

    search = sub.add_parser(
        "search", parents=[common], help="retrieve fused candidates for a query: no model calls"
    )
    search.add_argument("query")
    search.add_argument("--job", required=True)
    search.add_argument("--index", required=True, help="index directory (built if absent)")
    search.add_argument("--limit", type=int, default=10, help="candidates to return (max 100)")

    inspect = sub.add_parser("inspect", parents=[common], help="summarise a run directory")
    inspect.add_argument("--run", required=True)

    explain = sub.add_parser("explain", parents=[common], help="explain one source's decision")
    explain.add_argument("source_id")
    explain.add_argument("--run", required=True)
    explain.add_argument("--full", action="store_true", help="include the stored result")

    results = sub.add_parser(
        "results", parents=[common], help="one page of a run's current results"
    )
    results.add_argument("--run", required=True)
    results.add_argument("--status", default=None, help="only rows with this status")
    results.add_argument("--offset", type=int, default=0)
    results.add_argument("--limit", type=int, default=50, help="page size (max 200)")

    export = sub.add_parser("export", parents=[common], help="export a view of a run")
    export.add_argument("--run", required=True)
    export.add_argument("--view", choices=EXPORT_VIEWS, default="raw")
    export.add_argument("--out", required=True)

    ev = sub.add_parser("eval", parents=[common], help="evaluate a run against gold labels")
    ev.add_argument("--run", required=True)
    ev.add_argument("--gold", required=True)
    ev.add_argument("--out", default=None, help="write <out>.json and <out>.txt")

    fit = sub.add_parser(
        "fit",
        parents=[common],
        help="fit decider thresholds on a completed run and gold labels",
    )
    fit.add_argument("--run", required=True)
    fit.add_argument("--gold", required=True)
    fit.add_argument("--precision", type=float, default=0.95)
    fit.add_argument(
        "--job",
        default=None,
        help="the job the run used, so the gates that are not swept match the run",
    )
    fit.add_argument(
        "--holdout",
        action="store_true",
        help="fit on half the labelled rows and report the recommendation on the other half",
    )
    fit.add_argument(
        "--write-job",
        default=None,
        metavar="OUT.yaml",
        help="write a copy of --job with the recommended thresholds in its policy: block",
    )

    comp = sub.add_parser("compare", parents=[common], help="compare completed runs")
    comp.add_argument("--gold", required=True)
    comp.add_argument("--run", action="append", required=True, help="LABEL=PATH, repeatable")

    abl = sub.add_parser("ablate", parents=[common], help="re-run with each component disabled")
    abl.add_argument("--job", required=True)
    abl.add_argument("--gold", required=True)
    abl.add_argument("--out", required=True)

    prompts = sub.add_parser("prompts", help="draft or optimise prompt slots")
    prompt_sub = prompts.add_subparsers(dest="prompts_command")
    draft = prompt_sub.add_parser("draft", parents=[common])
    draft.add_argument("--job", required=True)
    draft.add_argument("--describe", required=True)
    draft.add_argument("--out", required=True)
    optimise = prompt_sub.add_parser("optimize", parents=[common])
    optimise.add_argument("--job", required=True)
    optimise.add_argument("--gold", required=True)
    optimise.add_argument("--out", required=True)
    optimise.add_argument("--role", default="selector")
    optimise.add_argument("--rounds", type=int, default=4)
    optimise.add_argument("--max-calls", type=int, default=None)

    mcp = sub.add_parser(
        "mcp", help="serve bounded operations to an agent over MCP stdio (needs xwalk[mcp])"
    )
    mcp.add_argument(
        "--max-calls-cap",
        type=int,
        default=None,
        help="the largest max_calls a match_records call may request (default 500)",
    )
    mcp.add_argument(
        "--offline-model",
        action="store_true",
        help="answer every model call with a fixed scripted reply instead of the job's "
        "endpoint (demos and tests; results are meaningless)",
    )

    ui = sub.add_parser("ui", help="serve a local web app to explore jobs and runs")
    ui.add_argument(
        "--root", default=".", help="the workspace directory the app may read and write"
    )
    ui.add_argument("--host", default="127.0.0.1", help="address to listen on")
    ui.add_argument("--port", type=int, default=8765, help="port to listen on (0: any free)")
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ui.add_argument(
        "--max-calls-cap",
        type=int,
        default=None,
        help="the largest max_calls a run with the job's endpoint may request (default 500)",
    )
    ui.add_argument(
        "--offline-only",
        action="store_true",
        help="refuse runs against the job's endpoint; only the offline stand-in model",
    )
    ui.add_argument(
        "--max-upload-mb", type=int, default=512, help="the largest file the page may upload"
    )
    ui.add_argument(
        "--public",
        action="store_true",
        help="run as a website: a private workspace per visitor, visitors' own model keys, "
        "no server-side URL downloads (see docs/guide/hosting.md)",
    )
    ui.add_argument(
        "--library",
        action="append",
        default=[],
        help="a directory of pre-parsed ontologies to offer read-only (repeatable; made by "
        "scripts/build_ontology_library.py)",
    )
    ui.add_argument("--lookup-cache", default=None, help="public mode: shared search-index cache")
    ui.add_argument(
        "--session-hours", type=float, default=24.0, help="public mode: delete idle sessions after"
    )
    ui.add_argument(
        "--max-tasks", type=int, default=4, help="public mode: runs at once across visitors"
    )
    ui.add_argument(
        "--frame-ancestors", default=None, help="origins allowed to embed the page in a frame"
    )
    ui.add_argument("--verbose", action="store_true", help="log every request to stderr")

    review = sub.add_parser("review", help="export or apply human review")
    review_sub = review.add_subparsers(dest="review_command")
    rexport = review_sub.add_parser("export", parents=[common])
    rexport.add_argument("--run", required=True)
    rexport.add_argument("--out", required=True)
    apply_ = review_sub.add_parser("apply", parents=[common])
    apply_.add_argument("--run", required=True)
    apply_.add_argument("--reviewed", required=True)
    apply_.add_argument("--job", required=True)

    return parser


# --- handlers: parse -> one operation -> OpResult -----------------------------------


def _cmd_init(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.init(args.dest)


def _build_decider(job: JobSpec) -> DecisionClient:
    """Module-level so a test can swap in a FakeDecider without an API key."""
    return job.build_decider()


def _build_rewrite_llm(job: JobSpec) -> LLMClient:
    """Module-level so a test can swap in a FakeLLM without an API key."""
    return job.build_rewrite_llm()


def _cmd_validate(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.validate(
        args.job, check_credentials=args.check_credentials, scan_records=args.scan_records
    )


def _cmd_index(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.index(args.job, args.out, rebuild=args.rebuild_index)


def _progress(result: Any) -> None:
    print(f"  {result.source_id}: {result.status.value}", file=sys.stderr)


def _cmd_match(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.run(
        args.job,
        args.out,
        index_dir=args.index,
        resume=args.resume,
        limit=args.limit,
        max_calls=args.max_calls,
        rebuild_index=args.rebuild_index,
        progress=_progress,
        # Looked up at call time, so a test can swap in fakes without an API key.
        decider_factory=lambda job: _build_decider(job),
        rewrite_llm_factory=lambda job: _build_rewrite_llm(job),
    )


def _cluster_progress(source_id: str, outcome: str) -> None:
    print(f"  {source_id}: {outcome}", file=sys.stderr)


def _cmd_cluster(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.cluster(args.job, args.out, max_calls=args.max_calls, progress=_cluster_progress)


def _cmd_search(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.search(args.job, args.query, index_dir=args.index, limit=args.limit)


def _cmd_results(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.list_results(args.run, offset=args.offset, limit=args.limit, status=args.status)


def _cmd_inspect(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.inspect(args.run)


def _cmd_explain(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.explain(args.run, args.source_id, full=args.full)


def _cmd_export(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    return ops.export(args.run, args.view, args.out)


def _cmd_eval(args: argparse.Namespace) -> OpResult:
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.evaluate.report import evaluate, render_report, write_report
    from xwalk.ledger import Ledger

    run_dir = Path(args.run)
    gold = load_gold_csv(args.gold)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        report = evaluate(ledger, _run_fingerprint_of(run_dir), gold)
    finally:
        ledger.close()
    result = OpResult(operation="eval", lines=[render_report(report)])
    result.data = {"report": report.as_dict()}
    if args.out:
        write_report(report, args.out)
        result.artifacts = {"json": f"{args.out}.json", "text": f"{args.out}.txt"}
    return result


def _write_fitted_job(job_path: str, out_path: str, point: FitPoint) -> None:
    """Copy the job file with the fitted thresholds set; every other key is left as it was."""
    import yaml

    data = yaml.safe_load(Path(job_path).read_text(encoding="utf-8"))
    policy = data.get("policy") or {}
    policy["accept_at"] = point.accept_at
    policy["property_floor"] = point.property_floor
    policy["choose_at"] = point.choose_at
    data["policy"] = policy
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _cmd_fit(args: argparse.Namespace) -> OpResult:
    from xwalk import ops
    from xwalk.decide import fit as fit_module
    from xwalk.decide.policy import DecisionPolicy
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.ledger import Ledger

    operation = "fit"
    if args.write_job is not None and args.job is None:
        raise OpError(
            operation,
            "usage",
            "--write-job needs --job, the job file to copy",
            exit_code=EXIT_USAGE,
        )

    result = OpResult(operation=operation)
    # Only accept_at, property_floor and choose_at are swept. Every other gate --
    # screen_floor, none_at, rubric_floor, shortlist_floor -- has to be the one the run
    # actually used, or the fitted thresholds are tuned against a policy nobody ran.
    if args.job is not None:
        job = ops.load_valid_job(args.job, operation=operation)
        if job.decider is None:
            raise OpError(
                operation,
                "wrong_job_path",
                f"{args.job} has no decider: block; fit works on the decider path",
                exit_code=EXIT_USAGE,
            )
        base = job.build_decision_policy()
    else:
        base = DecisionPolicy()
        result.warnings.append(
            OpMessage(
                "default_policy", "--job not given; fitting against default policy thresholds"
            )
        )

    run_dir = Path(args.run)
    if not (run_dir / "manifest.json").is_file() or not (run_dir / "ledger.sqlite").is_file():
        raise OpError(
            operation,
            "run_not_found",
            f"{run_dir} is not an xwalk run directory (no manifest.json and ledger.sqlite)",
            exit_code=EXIT_USAGE,
        )
    gold = load_gold_csv(args.gold)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    fingerprint = _run_fingerprint_of(run_dir)
    try:
        results = list(ledger.iter_results(fingerprint))
    finally:
        ledger.close()
    report = fit_module.fit_thresholds(results, gold, base=base, target_precision=args.precision)
    result.lines.append(fit_module.render_fit(report))
    recommended = report.recommended
    result.data = {
        "recommended": None
        if recommended is None
        else {
            "accept_at": recommended.accept_at,
            "property_floor": recommended.property_floor,
            "choose_at": recommended.choose_at,
            "precision": recommended.precision,
            "coverage": recommended.coverage,
        },
        "target_precision": args.precision,
    }
    if args.holdout:
        _, point = fit_module.fit_holdout(
            results,
            gold,
            base=base,
            seed_fingerprint=fingerprint,
            target_precision=args.precision,
        )
        result.lines.append(fit_module.render_holdout(point, recommended))
    if args.write_job is not None:
        if recommended is None:
            result.errors.append(
                OpMessage(
                    "no_recommendation",
                    f"no recommendation, so nothing to write to {args.write_job}",
                )
            )
            result.exit_code = EXIT_ATTENTION
            return result
        _write_fitted_job(args.job, args.write_job, recommended)
        result.lines.append(f"wrote {args.write_job}")
        result.artifacts["job"] = str(args.write_job)
        if Path(args.write_job).resolve().parent != Path(args.job).resolve().parent:
            result.warnings.append(
                OpMessage(
                    "relative_paths",
                    "the copy is in another directory; relative paths in it now resolve from there",
                )
            )
    result.exit_code = EXIT_OK if recommended is not None else EXIT_ATTENTION
    return result


def _refuse_decider_job(job: JobSpec, operation: str) -> None:
    if job.decider is not None:
        raise OpError(
            operation,
            "wrong_job_path",
            "this command works on the LLM path; the job has a decider: block",
            exit_code=EXIT_USAGE,
        )


def _cmd_compare(args: argparse.Namespace) -> OpResult:
    from xwalk.evaluate.compare import RunSummary, compare_runs, summarise_run
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.ledger import Ledger

    gold = load_gold_csv(args.gold)
    summaries: list[RunSummary] = []
    for entry in args.run:
        if "=" not in entry:
            raise OpError(
                "compare", "usage", f"--run expects LABEL=PATH, got {entry!r}", exit_code=EXIT_USAGE
            )
        label, _, path = entry.partition("=")
        run_dir = Path(path)
        ledger = Ledger.open(run_dir / "ledger.sqlite")
        try:
            summaries.append(summarise_run(ledger, _run_fingerprint_of(run_dir), gold, label=label))
        finally:
            ledger.close()
    table = compare_runs(summaries)
    return OpResult(operation="compare", lines=[table], data={"table": table})


def _cmd_ablate(args: argparse.Namespace) -> OpResult:
    from dataclasses import asdict

    from xwalk import ops
    from xwalk.evaluate.ablate import MatcherConfig, ablate
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.stores.memory import MemoryStore

    job = ops.load_valid_job(args.job, operation="ablate")
    _refuse_decider_job(job, "ablate")
    gold = load_gold_csv(args.gold)
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = MemoryStore.from_source(targets)
    llm = job.build_llm()
    index_dir = Path(args.out) / "index"
    all_retrievers = job.build_retrievers(targets, templates, index_dir)

    def factory(config: MatcherConfig) -> Matcher:
        retrievers = [r for r in all_retrievers if r.name in config.retriever_names]
        spec = job.model_copy(
            update={
                "policy": job.policy.model_copy(update=asdict(config.policy)),
                "selector": job.selector.model_copy(update=asdict(config.selector_policy)),
            }
        )
        return spec.build_matcher(store=store, retrievers=retrievers, llm=llm)

    base = MatcherConfig(
        name="baseline",
        # From the built retrievers, not from the specs: `factory` filters `all_retrievers`
        # on `r.name`, and a retriever can name itself something the spec does not say. An
        # unnamed `kind: dense` spec reads as "dense" but builds as "dense:<model>", so
        # spec-derived names would silently drop it from the baseline and every variant.
        retriever_names=tuple(r.name for r in all_retrievers),
        policy=job.build_policy(),
        selector_policy=job.build_selector_policy(),
    )
    out = Path(args.out) / "ablation.json"
    report = asyncio.run(ablate(factory, base, list(job.build_source_records()), gold, out=out))
    result = OpResult(operation="ablate", artifacts={"ablation": str(out)})
    result.data = {
        "rows": [
            {"name": row.name, "description": row.description, "delta": row.delta}
            for row in report.rows
        ]
    }
    result.lines = [
        f"  {row.name:<16} {row.description:<34} delta {row.delta:+.3f}" for row in report.rows
    ]
    return result


def _cmd_prompts(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    command = getattr(args, "prompts_command", None)
    if command not in ("draft", "optimize"):
        raise OpError(
            "prompts",
            "usage",
            "prompts needs a subcommand: draft or optimize",
            exit_code=EXIT_USAGE,
        )

    job = ops.load_valid_job(args.job, operation=f"prompts {command}")

    _refuse_decider_job(job, f"prompts {command}")

    if command == "draft":
        from xwalk.prompts.author import draft_slots, slots_diff, write_slots
        from xwalk.prompts.contract import load_slots

        sources = list(job.build_source_records())[:8]
        targets = list(job.build_target_records())[:8]
        existing = None
        slots_path = job.base_dir / job.prompts.slots
        if slots_path.exists():
            existing = load_slots(slots_path)
        draft = asyncio.run(
            draft_slots(
                job.build_llm(),
                description=args.describe,
                source_samples=sources,
                target_samples=targets,
                existing=existing,
            )
        )
        result = OpResult(operation="prompts draft", artifacts={"slots": str(args.out)})
        result.warnings = [OpMessage("draft_warning", w) for w in draft.warnings]
        if existing is not None:
            result.lines.append(slots_diff(existing, draft.slots) or "(no changes)")
        write_slots(draft.slots, args.out)
        result.lines.append(f"wrote {args.out}")
        return result

    from xwalk.evaluate.failures import PromptRole
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.prompts.contract import load_slots
    from xwalk.prompts.optimize import OptimizeConfig, optimize_prompt
    from xwalk.stores.memory import MemoryStore

    try:
        role = PromptRole(args.role)
    except ValueError:
        raise OpError(
            "prompts optimize",
            "usage",
            f"unknown --role {args.role!r}; choose from {[r.value for r in PromptRole]}",
            exit_code=EXIT_USAGE,
        ) from None
    gold = load_gold_csv(args.gold)
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = MemoryStore.from_source(targets)
    llm = job.build_llm()
    retrievers = job.build_retrievers(targets, templates, Path(args.out) / "index")

    def factory(prompts: PromptSet) -> Matcher:
        # The prompts MUST be threaded through: a factory that ignores them re-runs the
        # original slots every round and the optimiser measures nothing.
        return job.build_matcher(store=store, retrievers=retrievers, llm=llm, prompts=prompts)

    report = asyncio.run(
        optimize_prompt(
            matcher_factory=factory,
            source_records=list(job.build_source_records()),
            gold=gold,
            initial=load_slots(job.base_dir / job.prompts.slots),
            optimiser_llm=llm,
            config=OptimizeConfig(role=role, rounds=args.rounds, max_calls=args.max_calls),
            work_dir=args.out,
            progress=lambda line: print(line, file=sys.stderr),
        )
    )
    result = OpResult(operation="prompts optimize", artifacts={"work_dir": str(args.out)})
    result.data = {"stopped_because": str(report.stopped_because)}
    result.lines.append(f"stopped: {report.stopped_because}")
    if report.test_report is not None:
        precision = report.test_report.accepted_precision
        result.data["test_accepted_precision"] = precision
        result.lines.append(f"test accepted precision: {precision}")
    return result


def _cmd_review(args: argparse.Namespace) -> OpResult:
    from xwalk import ops

    command = getattr(args, "review_command", None)
    if command == "export":
        return ops.review_export(args.run, args.out)
    if command == "apply":
        return ops.review_apply(args.run, args.reviewed, args.job)
    raise OpError(
        "review", "usage", "review needs a subcommand: export or apply", exit_code=EXIT_USAGE
    )


_DISPATCH: dict[str, Callable[[argparse.Namespace], OpResult]] = {
    "init": _cmd_init,
    "validate": _cmd_validate,
    "doctor": _cmd_validate,
    "index": _cmd_index,
    "match": _cmd_match,
    "cluster": _cmd_cluster,
    "search": _cmd_search,
    "inspect": _cmd_inspect,
    "results": _cmd_results,
    "explain": _cmd_explain,
    "export": _cmd_export,
    "eval": _cmd_eval,
    "fit": _cmd_fit,
    "compare": _cmd_compare,
    "ablate": _cmd_ablate,
    "prompts": _cmd_prompts,
    "review": _cmd_review,
}


# --- output -------------------------------------------------------------------------


def _operation_name(args: argparse.Namespace) -> str:
    command = str(args.command)
    sub = getattr(args, "prompts_command", None) or getattr(args, "review_command", None)
    return f"{command} {sub}" if sub else command


def _emit(result: OpResult, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.envelope(), ensure_ascii=False, default=str))
        return
    for line in result.lines:
        print(line)
    for warning in result.warnings:
        print(f"warning: {warning.message}", file=sys.stderr)
    for error in result.errors:
        where = f"{error.source_id}: " if error.source_id else ""
        print(f"error: {where}{error.code}: {error.message}", file=sys.stderr)


def _failure(operation: str, exc: BaseException) -> OpResult:
    """Map an exception that escaped an operation to an envelope. No traceback."""
    return failure_result(operation, exc)


def _serve_mcp(args: argparse.Namespace) -> int:
    """Run the MCP stdio server until the client closes stdin. stdout is the protocol
    channel from here on, so a failure to start is reported on stderr only."""
    try:
        from xwalk.mcp_server import serve

        serve(max_calls_cap=args.max_calls_cap, offline_model=args.offline_model)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as exc:
        result = failure_result("mcp", exc)
        for error in result.errors:
            print(f"error: {error.code}: {error.message}", file=sys.stderr)
        return result.exit_code
    return EXIT_OK


def _serve_ui(args: argparse.Namespace) -> int:
    """Serve the web app until Ctrl-C. Status lines go to stderr."""
    try:
        from xwalk.ui import serve
        from xwalk.ui.api import DEFAULT_MAX_CALLS_CAP

        serve(
            args.root,
            host=args.host,
            port=args.port,
            max_calls_cap=DEFAULT_MAX_CALLS_CAP
            if args.max_calls_cap is None
            else args.max_calls_cap,
            offline_only=args.offline_only,
            max_upload_mb=args.max_upload_mb,
            open_browser=not args.no_browser,
            verbose=args.verbose,
            public=args.public,
            library=args.library,
            lookup_cache=args.lookup_cache,
            session_ttl=args.session_hours * 3600,
            max_tasks=args.max_tasks,
            frame_ancestors=args.frame_ancestors,
        )
    except KeyboardInterrupt:
        return EXIT_OK
    except (OSError, ValueError) as exc:
        print(f"error: ui: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    as_json = "--json" in arguments
    parser = _build_parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exc:
        # argparse exits the process on `--help` and on an invalid choice. A library
        # entry point that raises SystemExit can only be called from a shell.
        code = int(exc.code or 0)
        if as_json and code:
            _emit(
                OpResult(
                    "usage",
                    exit_code=EXIT_USAGE,
                    errors=[OpMessage("usage", "invalid arguments (see stderr)")],
                ),
                as_json=True,
            )
        return EXIT_USAGE if code else EXIT_OK

    if args.version:
        print(__version__)
        return EXIT_OK
    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    if args.command == "mcp":
        return _serve_mcp(args)
    if args.command == "ui":
        return _serve_ui(args)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE

    as_json = bool(getattr(args, "json", False))
    operation = _operation_name(args)
    try:
        result = handler(args)
    except KeyboardInterrupt:
        from xwalk import ops

        run_dir = getattr(args, "out", None) if args.command == "match" else None
        result = ops.interrupted(operation, run_dir)
    except Exception as exc:  # a CLI must not dump a traceback
        result = _failure(operation, exc)
    _emit(result, as_json=as_json)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""The CLI: a thin shell over the SDK.

Every subcommand parses arguments, calls one library function, prints, and returns an
exit code. Nothing worth testing lives here.

Exit codes:
  0  success
  1  the run completed but something needs attention (non-empty review bucket)
  2  usage error
  3  runtime failure (missing key, unreadable file, unsupported platform)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xwalk import __version__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xwalk.batch import MatcherLike
    from xwalk.config import JobSpec
    from xwalk.decide.base import DecisionClient
    from xwalk.evaluate.ablate import MatcherConfig
    from xwalk.matcher import Matcher
    from xwalk.prompts.contract import PromptSet

EXIT_OK = 0
EXIT_ATTENTION = 1
EXIT_USAGE = 2
EXIT_RUNTIME = 3


def _run_fingerprint_of(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["run_fingerprint"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xwalk", description="Match records from any collection to any other."
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    index = sub.add_parser("index", help="build retriever indexes for a job")
    index.add_argument("--job", required=True)
    index.add_argument("--out", required=True, help="index directory")

    match = sub.add_parser("match", help="run a matching job")
    match.add_argument("--job", required=True)
    match.add_argument("--out", required=True, help="run directory")
    match.add_argument("--index", default=None, help="index directory (default: <out>/index)")
    match.add_argument("--resume", action="store_true", default=True)
    match.add_argument("--no-resume", dest="resume", action="store_false")
    match.add_argument("--limit", type=int, default=None, help="process only the first N records")

    ev = sub.add_parser("eval", help="evaluate a completed run against gold labels")
    ev.add_argument("--run", required=True)
    ev.add_argument("--gold", required=True)
    ev.add_argument("--out", default=None, help="write <out>.json and <out>.txt")

    fit = sub.add_parser("fit", help="fit decider thresholds on a completed run and gold labels")
    fit.add_argument("--run", required=True)
    fit.add_argument("--gold", required=True)
    fit.add_argument("--precision", type=float, default=0.95)
    fit.add_argument(
        "--job",
        default=None,
        help="the job the run used, so the gates that are not swept match the run",
    )

    comp = sub.add_parser("compare", help="compare completed runs")
    comp.add_argument("--gold", required=True)
    comp.add_argument("--run", action="append", required=True, help="LABEL=PATH, repeatable")

    abl = sub.add_parser("ablate", help="re-run with each component disabled")
    abl.add_argument("--job", required=True)
    abl.add_argument("--gold", required=True)
    abl.add_argument("--out", required=True)

    # Flags live on the leaf subparsers, never on the group. argparse binds an option
    # to whichever parser is active when it is seen, so a flag declared on the group
    # but typed after the subcommand is simply unrecognised.
    prompts = sub.add_parser("prompts", help="draft or optimise prompt slots")
    prompt_sub = prompts.add_subparsers(dest="prompts_command")
    draft = prompt_sub.add_parser("draft")
    draft.add_argument("--job", required=True)
    draft.add_argument("--describe", required=True)
    draft.add_argument("--out", required=True)
    optimise = prompt_sub.add_parser("optimize")
    optimise.add_argument("--job", required=True)
    optimise.add_argument("--gold", required=True)
    optimise.add_argument("--out", required=True)
    optimise.add_argument("--role", default="selector")
    optimise.add_argument("--rounds", type=int, default=4)
    optimise.add_argument("--max-calls", type=int, default=None)

    review = sub.add_parser("review", help="export or apply human review")
    review_sub = review.add_subparsers(dest="review_command")
    export = review_sub.add_parser("export")
    export.add_argument("--run", required=True)
    export.add_argument("--out", required=True)
    apply_ = review_sub.add_parser("apply")
    apply_.add_argument("--run", required=True)
    apply_.add_argument("--reviewed", required=True)
    apply_.add_argument("--job", required=True)

    return parser


def _cmd_index(args: argparse.Namespace) -> int:
    from xwalk.config import load_job

    job = load_job(args.job)
    templates = job.build_templates()
    records = list(job.build_target_records())
    retrievers = job.build_retrievers(records, templates, args.out)
    print(f"indexed {len(records)} target records into {len(retrievers)} retriever(s)")
    for retriever in retrievers:
        print(f"  {retriever.name}: {retriever.fingerprint}")
    return EXIT_OK


def _build_decider(job: JobSpec) -> DecisionClient:
    """Module-level so a test can swap in a FakeDecider without an API key."""
    return job.build_decider()


def _cmd_match(args: argparse.Namespace) -> int:
    from xwalk.batch import run_batch
    from xwalk.config import load_job
    from xwalk.decide.cache import CachingDecider
    from xwalk.ledger import Ledger

    job = load_job(args.job)
    index_dir = Path(args.index or Path(args.out) / "index")
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, templates, index_dir)

    records = list(job.build_source_records())
    if args.limit is not None:
        records = records[: args.limit]

    cache_ledger: Ledger | None = None
    try:
        if job.decider is not None:
            # The cache lives in the run's own ledger. run_batch opens the same file
            # again; two connections are safe because every ledger write is a single
            # autocommit statement issued from one thread, so neither holds a
            # transaction open across the other's writes.
            cache_ledger = Ledger.open(Path(args.out) / "ledger.sqlite")
            decider = CachingDecider(_build_decider(job), cache_ledger)
            matcher: MatcherLike = job.build_decision_matcher(
                store=store, retrievers=retrievers, decider=decider
            )
            manifest_extra = {"job": job.name, "model": decider.model, "path": "decider"}
        else:
            llm = job.build_llm()
            matcher = job.build_matcher(store=store, retrievers=retrievers, llm=llm)
            manifest_extra = {"job": job.name, "model": llm.model, "path": "llm"}

        report = asyncio.run(
            run_batch(
                matcher, records, out=args.out, resume=args.resume, manifest_extra=manifest_extra
            )
        )
    finally:
        if cache_ledger is not None:
            cache_ledger.close()

    print(f"matched {report.total} records into {report.out_dir}")
    for status, count in sorted(report.by_status().items(), key=lambda kv: kv[0].value):
        print(f"  {status.value:<14}: {count}")
    duplicates = report.duplicate_targets()
    if duplicates:
        print(f"  duplicate targets: {len(duplicates)} (see manifest.json)")
    review_count = len(report.needs_review())
    if review_count:
        print(f"\n{review_count} rows need review: xwalk review export --run {args.out}")
        return EXIT_ATTENTION
    return EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
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
    print(render_report(report))
    if args.out:
        write_report(report, args.out)
    return EXIT_OK


def _cmd_fit(args: argparse.Namespace) -> int:
    from xwalk.config import load_job
    from xwalk.decide.fit import fit_thresholds, render_fit
    from xwalk.decide.policy import DecisionPolicy
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.ledger import Ledger

    # Only accept_at and property_floor are swept. Every other gate -- screen_floor,
    # choose_at, none_at, rubric_floor, shortlist_floor -- has to be the one the run
    # actually used, or the fitted pair is tuned against a policy nobody ran.
    if args.job is not None:
        job = load_job(args.job)
        if job.decider is None:
            raise ValueError(f"{args.job} has no decider: block; fit works on the decider path")
        base = job.build_decision_policy()
    else:
        base = DecisionPolicy()
        print("note: --job not given; fitting against default policy thresholds", file=sys.stderr)

    run_dir = Path(args.run)
    gold = load_gold_csv(args.gold)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        results = list(ledger.iter_results(_run_fingerprint_of(run_dir)))
    finally:
        ledger.close()
    report = fit_thresholds(results, gold, base=base, target_precision=args.precision)
    print(render_fit(report))
    return EXIT_OK if report.recommended is not None else EXIT_ATTENTION


def _cmd_compare(args: argparse.Namespace) -> int:
    from xwalk.evaluate.compare import RunSummary, compare_runs, summarise_run
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.ledger import Ledger

    gold = load_gold_csv(args.gold)
    summaries: list[RunSummary] = []
    for entry in args.run:
        if "=" not in entry:
            print(f"--run expects LABEL=PATH, got {entry!r}", file=sys.stderr)
            return EXIT_USAGE
        label, _, path = entry.partition("=")
        run_dir = Path(path)
        ledger = Ledger.open(run_dir / "ledger.sqlite")
        try:
            summaries.append(summarise_run(ledger, _run_fingerprint_of(run_dir), gold, label=label))
        finally:
            ledger.close()
    print(compare_runs(summaries))
    return EXIT_OK


def _cmd_ablate(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from xwalk.config import load_job
    from xwalk.evaluate.ablate import MatcherConfig, ablate
    from xwalk.evaluate.gold import load_gold_csv

    job = load_job(args.job)
    if job.decider is not None:
        print(
            "error: this command works on the LLM path; the job has a decider: block",
            file=sys.stderr,
        )
        return EXIT_USAGE
    gold = load_gold_csv(args.gold)
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
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
    report = asyncio.run(
        ablate(
            factory,
            base,
            list(job.build_source_records()),
            gold,
            out=Path(args.out) / "ablation.json",
        )
    )
    for row in report.rows:
        print(f"  {row.name:<16} {row.description:<34} delta {row.delta:+.3f}")
    return EXIT_OK


def _cmd_prompts(args: argparse.Namespace) -> int:
    from xwalk.config import load_job

    if getattr(args, "prompts_command", None) not in ("draft", "optimize"):
        print("prompts needs a subcommand: draft or optimize", file=sys.stderr)
        return EXIT_USAGE

    job = load_job(args.job)
    if job.decider is not None:
        print(
            "error: this command works on the LLM path; the job has a decider: block",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.prompts_command == "draft":
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
        for warning in draft.warnings:
            print(f"warning: {warning}")
        if existing is not None:
            print(slots_diff(existing, draft.slots) or "(no changes)")
        write_slots(draft.slots, args.out)
        print(f"wrote {args.out}")
        return EXIT_OK

    from xwalk.evaluate.failures import PromptRole
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.prompts.contract import load_slots
    from xwalk.prompts.optimize import OptimizeConfig, optimize_prompt

    gold = load_gold_csv(args.gold)
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
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
            config=OptimizeConfig(
                role=PromptRole(args.role), rounds=args.rounds, max_calls=args.max_calls
            ),
            work_dir=args.out,
            progress=print,
        )
    )
    print(f"stopped: {report.stopped_because}")
    if report.test_report is not None:
        print(f"test accepted precision: {report.test_report.accepted_precision}")
    return EXIT_OK


def _cmd_review(args: argparse.Namespace) -> int:
    from xwalk.ledger import Ledger
    from xwalk.review import apply_review, export_review, read_review

    if getattr(args, "review_command", None) not in ("export", "apply"):
        print("review needs a subcommand: export or apply", file=sys.stderr)
        return EXIT_USAGE

    run_dir = Path(args.run)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        run_fp = _run_fingerprint_of(run_dir)
        if args.review_command == "export":
            count = export_review(ledger, run_fp, args.out)
            print(f"exported {count} rows to {args.out}")
            return EXIT_OK

        from xwalk.config import load_job

        job = load_job(args.job)
        report = apply_review(
            ledger,
            read_review(args.reviewed),
            target_store_fingerprint=job.build_store().fingerprint,
        )
        print(f"applied {report.applied} decisions")
        for key, why in report.rejected:
            print(f"  rejected {key}: {why}", file=sys.stderr)
        return EXIT_ATTENTION if report.rejected else EXIT_OK
    finally:
        ledger.close()


_DISPATCH: dict[str, Any] = {
    "index": _cmd_index,
    "match": _cmd_match,
    "eval": _cmd_eval,
    "fit": _cmd_fit,
    "compare": _cmd_compare,
    "ablate": _cmd_ablate,
    "prompts": _cmd_prompts,
    "review": _cmd_review,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        # argparse exits the process on `--help` and on an invalid choice. A library
        # entry point that raises SystemExit can only be called from a shell.
        return int(exc.code or 0)

    if args.version:
        print(__version__)
        return EXIT_OK
    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(handler(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as exc:  # a CLI must not dump a traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())

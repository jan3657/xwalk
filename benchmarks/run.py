"""Run the pilot benchmark.

Offline smoke checks (no network, no credentials, SYNTHETIC judge):

    python -m benchmarks.run --smoke

Real evaluation (calls a provider; costs money; results are only meaningful from here):

    export XWALK_BENCH_BASE_URL=... XWALK_BENCH_MODEL=... XWALK_BENCH_API_KEY_ENV=MY_KEY
    python -m benchmarks.run --real --max-calls 2000 --out benchmarks/results/real-<date>

Without the ``XWALK_BENCH_*`` overrides the real path uses each example job's own ``llm``
section. ``--max-calls`` is required for ``--real`` and caps *every* LLM-using method
separately (each one is an invocation).

Writes ``results.json`` (everything, raw), ``summary.md`` (generated tables) and the
clustering prediction files under ``--out``. Scratch runs, indexes and ledgers go to
``--work`` (a temporary directory unless given) and are never committed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from benchmarks import methods
from benchmarks.cluster_eval import evaluate_files
from benchmarks.clustering import write_assignments
from benchmarks.data import REPO_ROOT, Variant, load_manifest, materialise, split_counts
from benchmarks.metrics import matching_metrics
from benchmarks.pairwise import run_pairwise
from benchmarks.synthetic_llm import SyntheticJudgeLLM, normalise
from xwalk import __version__
from xwalk.config import load_job
from xwalk.llm.base import LLMClient

DATASETS = ("cafeteria_fcd", "ncbi_disease")
SYNTHETIC_WARNING = (
    "SYNTHETIC: every LLM answer here comes from benchmarks/synthetic_llm.py, a "
    "deterministic string-similarity heuristic, not a language model. These numbers "
    "check that the runners, accounting and metrics work end to end. They are not "
    "evidence about xwalk's matching quality with a real model, and token counts are "
    "chars/4 estimates."
)
PENDING = {
    "dense_retrieval": "skipped: needs xwalk[dense] and sentence-transformer weights (a "
    "large download); not run in this environment",
    "linktransformer": "pending: benchmarks/adapters/linktransformer.py documents the "
    "adapter; running it needs the linktransformer package and model weights",
}

ENV_OVERRIDES = {
    "XWALK_BENCH_BASE_URL": "base_url",
    "XWALK_BENCH_MODEL": "model",
    "XWALK_BENCH_API_KEY_ENV": "api_key_env",
    "XWALK_BENCH_PROFILE": "profile",
    "XWALK_BENCH_LLM_KIND": "kind",
}


# --- provenance ---------------------------------------------------------------------


def code_revision() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "xwalk_version": __version__,
    }


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.machine(),
    }


# --- LLM selection ------------------------------------------------------------------


def apply_env_overrides(job_path: Path) -> dict[str, Any]:
    """Point the variant's job at the endpoint named by XWALK_BENCH_* (real mode)."""
    data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            data["llm"][key] = os.environ[env]
    job_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dict(data["llm"])


def llm_factory(mode: str, variant: Variant) -> Callable[[], LLMClient]:
    if mode == "smoke":
        return lambda: SyntheticJudgeLLM(query_field=variant.query_field)
    spec = load_job(variant.job_path)
    return spec.build_llm  # raises CredentialMissingError naming the variable


# --- measurement --------------------------------------------------------------------


def measured(fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Wall time and Python peak memory (tracemalloc) of one call.

    Wall time is taken with tracemalloc active, which slows allocation-heavy code; it is
    comparable between methods of one run, not an absolute figure. tracemalloc sees
    Python allocations only, not tantivy's native (Rust) memory.
    """
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = fn()
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return value, {
        "wall_seconds": round(elapsed, 3),
        "peak_python_mib": round(peak / 2**20, 2),
    }


def _metrics(variant: Variant, output: methods.MethodOutput) -> dict[str, Any]:
    return {
        split: matching_metrics(output.predictions, variant.gold, variant.ids_in_split(split))
        for split in ("all", "test")
    }


# --- tracks -------------------------------------------------------------------------


def run_matching(
    variant: Variant, work: Path, make_llm: Callable[[], LLMClient], max_calls: int | None
) -> tuple[dict[str, Any], dict[str, methods.MethodOutput]]:
    runs: dict[str, Callable[[], methods.MethodOutput]] = {
        "exact": lambda: methods.exact(variant),
        "fuzzy_difflib": lambda: methods.fuzzy(variant),
        "bm25_top1": lambda: methods.bm25_only(variant, work),
        "retrieval_one_llm_pass": lambda: methods.selector_one_pass(
            variant, work, make_llm(), max_calls=max_calls
        ),
        "xwalk_single_attempt": lambda: methods.xwalk_run(
            variant,
            work / "xwalk_single_attempt",
            make_llm(),
            policy_overrides={"max_attempts": 1, "verify_band": None},
            max_calls=max_calls,
        ),
        "xwalk_full": lambda: methods.xwalk_run(
            variant, work / "xwalk_full", make_llm(), max_calls=max_calls
        ),
    }
    results: dict[str, Any] = {}
    outputs: dict[str, methods.MethodOutput] = {}
    for name, fn in runs.items():
        output, cost = measured(fn)
        outputs[name] = output
        results[name] = {
            "metrics": _metrics(variant, output),
            "usage": output.usage,
            "cost": cost,
            "config": output.config,
            **({"run": output.extra} if output.extra else {}),
        }
    return results, outputs


def gold_clusters(variant: Variant) -> dict[str, str]:
    """Equivalence gold for mention clustering: mentions with the same (expanded) gold
    id set are one cluster; a mention with no gold is its own singleton. Uses the gold
    before any catalog construction, since clustering involves no catalog."""
    out = {}
    for source in variant.sources:
        ids = variant.base_gold.get(source.id) or frozenset()
        out[source.id] = "g:" + "|".join(sorted(ids)) if ids else f"s:{source.id}"
    return out


def greedy_leader(
    variant: Variant, order: Sequence[str], threshold: float = 0.85
) -> dict[str, str]:
    """Order-sensitive baseline: join the first leader within `threshold` difflib ratio."""
    text = {s.id: normalise(str(s.fields.get(variant.query_field) or "")) for s in variant.sources}
    leaders: list[str] = []
    out: dict[str, str] = {}
    for item in order:
        for leader in leaders:
            if SequenceMatcher(None, text[item], text[leader]).ratio() >= threshold:
                out[item] = leader
                break
        else:
            leaders.append(item)
            out[item] = item
    return {s.id: out[s.id] for s in variant.sources}


def run_clustering(
    variant: Variant, matching: dict[str, methods.MethodOutput], out_dir: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = out_dir / "gold_clusters.csv"
    write_assignments(gold_path, gold_clusters(variant))
    ids = [s.id for s in variant.sources]

    predictions: dict[str, list[Path]] = {}

    exact_path = out_dir / "pred_exact_mention.csv"
    write_assignments(
        exact_path,
        {s.id: normalise(str(s.fields.get(variant.query_field) or "")) for s in variant.sources},
    )
    predictions["exact_mention"] = [exact_path]

    leader_paths = []
    for label, order in (("input_order", ids), ("reversed", list(reversed(ids)))):
        path = out_dir / f"pred_greedy_fuzzy_{label}.csv"
        write_assignments(path, greedy_leader(variant, order))
        leader_paths.append(path)
    predictions["greedy_fuzzy_leader"] = leader_paths

    xw = matching.get("xwalk_full")
    if xw is not None:
        clusters: dict[str, str] = {}
        outcomes: dict[str, str] = {}
        for p in xw.predictions:
            if p.status == "matched" and p.predicted_id:
                clusters[p.source_id] = f"t:{p.predicted_id}"
                outcomes[p.source_id] = ""
            elif p.status == "unmatched":
                clusters[p.source_id] = f"s:{p.source_id}"
                outcomes[p.source_id] = "singleton"
            else:
                clusters[p.source_id] = f"r:{p.source_id}"
                outcomes[p.source_id] = "needs_review" if p.status == "needs_review" else "failed"
        path = out_dir / "pred_via_xwalk_matching.csv"
        write_assignments(path, clusters, {k: v for k, v in outcomes.items() if v})
        predictions["via_xwalk_matching"] = [path]

    results: dict[str, Any] = {}
    for name, paths in predictions.items():
        scored, cost = measured(functools.partial(evaluate_files, gold_path, paths))
        scored["gold"] = gold_path.name
        scored["predictions"] = [p.name for p in paths]
        results[name] = {**scored, "cost": cost}
    results["_note"] = (
        "Baselines only. xwalk clustering (task 04) is scored by passing its exported "
        "assignments to `python -m benchmarks.cluster_eval`; via_xwalk_matching groups "
        "mentions by their accepted catalog target and is not a clustering algorithm."
    )
    return results


# --- summary ------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Benchmark summary ({report['kind']})", ""]
    if report["kind"] == "SYNTHETIC":
        lines += [f"> {SYNTHETIC_WARNING}", ""]
    lines += [
        f"Generated {report['generated_at']} at commit `{report['code']['commit']}` "
        f"(dirty: {report['code']['dirty']}). Raw values: `results.json`.",
        "",
    ]
    for variant, track in report["tracks"]["matching"].items():
        lines += [
            f"## Matching: {variant} (all rows)",
            "",
            "| method | acc. precision | coverage | review | final F1 | no-match P | "
            "no-match R | false accept on no-match | cand. recall | truncated | calls | "
            "tokens | wall s | peak MiB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, row in track.items():
            m = row["metrics"]["all"]
            usage = row["usage"] or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(m["accepted_precision"]),
                        _fmt(m["coverage"]),
                        _fmt(m["review_rate"]),
                        _fmt(m["final_f1"]),
                        _fmt(m["no_match_precision"]),
                        _fmt(m["no_match_recall"]),
                        _fmt(m["false_accept_on_no_match"]),
                        _fmt(m["candidate_recall"]),
                        _fmt(m["truncated"]),
                        _fmt(usage.get("calls")),
                        _fmt(usage.get("tokens")),
                        _fmt(row["cost"]["wall_seconds"]),
                        _fmt(row["cost"]["peak_python_mib"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    for variant, track in report["tracks"]["pairwise"].items():
        lines += [
            f"## Pairwise: {variant} (BM25 top-{track['block_k']} pairs; "
            f"{track['blocking_misses']} gold targets never blocked)",
            "",
            "| method | pairs | positives | precision | recall | F1 |",
            "|---|---|---|---|---|---|",
        ]
        for name, row in track["methods"].items():
            lines.append(
                f"| {name} | {row['pairs']} | {row['positives']} | {_fmt(row['precision'])} | "
                f"{_fmt(row['recall'])} | {_fmt(row['f1'])} |"
            )
        lines.append("")
    for dataset, track in report["tracks"]["clustering"].items():
        lines += [
            f"## Clustering: {dataset} (as_singletons convention)",
            "",
            "| prediction | pairwise F1 | B-cubed F1 | false-merge clusters | split gold "
            "clusters | pred. singletons | review | order: min F1 between runs |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, row in track.items():
            if name.startswith("_"):
                continue
            s = row["scores"]["as_singletons"]
            order = row["order_sensitivity"] or {}
            lines.append(
                f"| {name} | {_fmt(s['pairwise_f1'])} | {_fmt(s['bcubed_f1'])} | "
                f"{s['false_merge_clusters']} | {s['split_gold_clusters']} | "
                f"{s['predicted_singletons']} | "
                f"{row['scores']['outcome_counts']['needs_review']} | "
                f"{_fmt(order.get('min_pairwise_f1_between_runs'))} |"
            )
        lines.append("")
    lines += ["## Not run", ""] + [f"- {k}: {v}" for k, v in report["not_run"].items()]
    return "\n".join(lines) + "\n"


# --- entry point --------------------------------------------------------------------


def run(
    mode: str,
    out: Path,
    work: Path,
    *,
    datasets: Sequence[str] = DATASETS,
    pair_k: int = 5,
    max_calls: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "SYNTHETIC" if mode == "smoke" else "REAL",
        "warning": SYNTHETIC_WARNING if mode == "smoke" else None,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code": code_revision(),
        "environment": environment(),
        "command": " ".join(["python -m benchmarks.run", *sys.argv[1:]]),
        "limit": limit,
        "data": {},
        "llm": {},
        "tracks": {"matching": {}, "pairwise": {}, "clustering": {}},
        "not_run": dict(PENDING),
    }
    out.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        manifest = load_manifest(dataset)
        variants = materialise(manifest, work / "data", limit=limit)
        report["data"][dataset] = {
            "manifest": str(Path(manifest["_path"]).relative_to(REPO_ROOT)),
            "sha256": manifest["sha256"],
            "variants": {
                v.name: {
                    "targets": len(v.targets),
                    "sources": len(v.sources),
                    "gold_no_match": sum(1 for ids in v.gold.labels.values() if not ids),
                    "splits": split_counts(v),
                    "construction": v.construction,
                }
                for v in variants
            },
        }
        for variant in variants:
            if mode == "real":
                report["llm"][variant.name] = apply_env_overrides(variant.job_path)
            else:
                report["llm"][variant.name] = {
                    "client": "benchmarks.synthetic_llm.SyntheticJudgeLLM",
                    "fingerprint": SyntheticJudgeLLM(query_field=variant.query_field).fingerprint,
                }
            make_llm = llm_factory(mode, variant)
            if mode == "real":
                make_llm()  # a missing credential stops here, before anything runs
            variant_work = work / "runs" / variant.name
            matching, outputs = run_matching(variant, variant_work, make_llm, max_calls)
            report["tracks"]["matching"][variant.name] = matching
            report["tracks"]["pairwise"][variant.name] = run_pairwise(
                variant, variant_work, make_llm(), pair_k, max_calls
            )
            if variant.construction.get("method") == "none":
                report["tracks"]["clustering"][dataset] = run_clustering(
                    variant, outputs, out / "clustering" / dataset
                )
    (out / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.run", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="offline, SYNTHETIC judge")
    mode.add_argument("--real", action="store_true", help="call the configured provider")
    parser.add_argument("--out", type=Path, help="results directory")
    parser.add_argument("--work", type=Path, help="scratch directory (default: temporary)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--pair-k", type=int, default=5, help="pairwise block size")
    parser.add_argument(
        "--max-calls", type=int, help="per-method LLM call cap (required with --real)"
    )
    parser.add_argument("--limit", type=int, help="first N source records only (quick check)")
    args = parser.parse_args(argv)

    if args.real and not args.max_calls:
        parser.error("--real needs --max-calls: a real run spends money")
    mode_name = "smoke" if args.smoke else "real"
    out = args.out or REPO_ROOT / "benchmarks" / "results" / mode_name
    temp = None
    if args.work is None:
        temp = tempfile.mkdtemp(prefix="xwalk-bench-")
        work = Path(temp)
    else:
        work = args.work
        marker = work / ".xwalk-bench-work"
        if work.exists() and any(work.iterdir()) and not marker.exists():
            print(
                f"error: --work {work} is not empty and is not a benchmark work dir",
                file=sys.stderr,
            )
            return 2
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        marker.touch()
    try:
        report = run(
            mode_name,
            out,
            work,
            datasets=args.datasets,
            pair_k=args.pair_k,
            max_calls=args.max_calls,
            limit=args.limit,
        )
    except Exception as exc:  # report, never a traceback for a missing credential
        from xwalk.config import CredentialMissingError

        if isinstance(exc, CredentialMissingError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"[{report['kind']}] wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

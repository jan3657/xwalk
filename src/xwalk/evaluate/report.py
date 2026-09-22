"""The evaluation entry point.

One call, one printable summary, with the ceiling recommendation first -- "where should I
spend effort?" is the question this phase exists to answer, and it must not be buried.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xwalk.evaluate.ceiling import CeilingBucket, CeilingReport, ceiling_report
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import (
    EvalReport,
    ThresholdPoint,
    evaluate_results,
    threshold_curve,
)
from xwalk.evaluate.partition import Partition, Partitioner, partition_of
from xwalk.ledger import Ledger


@dataclass(frozen=True)
class FullReport:
    metrics: EvalReport
    ceiling: CeilingReport
    thresholds: tuple[ThresholdPoint, ...]
    partition_sizes: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "ceiling": self.ceiling.as_dict(),
            "thresholds": [
                {
                    "threshold": p.threshold,
                    "coverage": p.coverage,
                    "precision": p.precision,
                    "accepted": p.accepted,
                    "correct": p.correct,
                }
                for p in self.thresholds
            ],
            "partition_sizes": dict(self.partition_sizes),
        }


def evaluate(
    ledger: Ledger,
    run_fingerprint: str,
    gold: GoldSet,
    *,
    partitioner: Partitioner | None = None,
    partition_overrides: Mapping[str, Partition] | None = None,
    threshold_steps: int = 21,
) -> FullReport:
    results = list(ledger.iter_results(run_fingerprint))

    sizes: dict[str, int] = {}
    if partitioner is not None:
        counts = {p.value: 0 for p in Partition}
        for result in results:
            if result.source_id not in gold:
                continue
            counts[partition_of(result.source_id, partitioner, partition_overrides).value] += 1
        sizes = counts

    return FullReport(
        metrics=evaluate_results(results, gold),
        ceiling=ceiling_report(results, gold),
        thresholds=tuple(threshold_curve(results, gold, steps=threshold_steps)),
        partition_sizes=sizes,
    )


def _pct(value: float | None, *, reason: str = "") -> str:
    if value is None:
        return f"n/a{f' ({reason})' if reason else ''}"
    return f"{value:.1%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _usd(value: float | None) -> str:
    # Six places: a per-record cost of a hundredth of a cent is normal, and rounding it
    # to two would print every decider run as $0.00.
    return "n/a" if value is None else f"${value:.6f}"


def render_report(report: FullReport) -> str:
    m = report.metrics
    c = report.ceiling

    lines = [
        "## Where to spend effort",
        f"  {c.recommendation}",
        "",
        "## Headline",
        f"  accepted precision   : {_pct(m.accepted_precision, reason='nothing matched')}",
        f"  automatic coverage   : {_pct(m.automatic_coverage, reason='nothing labelled')}",
        f"  review rate          : {_pct(m.review_rate, reason='nothing labelled')}",
        f"  unmatched rate       : {_pct(m.unmatched_rate, reason='nothing labelled')}",
        f"  error rate           : {_pct(m.error_rate, reason='nothing labelled')}",
        f"  unresolved rate      : {_pct(m.unresolved_rate, reason='nothing labelled')}",
        f"  recall at any status : {_pct(m.recall_at_any_status, reason='no gold ids')}",
        "",
        "## No-match handling",
        f"  no-match precision   : "
        f"{_pct(m.no_match_precision, reason='nothing predicted no-match')}",
        f"  no-match recall      : {_pct(m.no_match_recall, reason='no gold no-match labels')}",
        "",
        "## Failure decomposition",
    ]
    for bucket in (
        CeilingBucket.FOUND,
        CeilingBucket.MISJUDGED,
        CeilingBucket.TRUNCATED,
        CeilingBucket.NEVER_RETRIEVED,
    ):
        lines.append(f"  {bucket.value:<18}: {c.buckets.get(bucket, 0)}")
    if c.by_retriever:
        lines.append(
            "  gold surfaced by   : "
            + ", ".join(f"{name} ({count})" for name, count in sorted(c.by_retriever.items()))
        )

    lines += [
        "",
        "## Cost",
        f"  model calls / record : {_num(m.mean_llm_calls)}",
        f"  tokens / record      : {_num(m.mean_tokens)}",
        f"  cost / record        : {_usd(m.mean_cost_usd)}",
        f"  seconds / record     : {_num(m.mean_seconds)}",
        f"  duplicate targets    : {m.duplicate_target_conflicts}",
    ]

    if report.partition_sizes:
        lines += ["", "## Partitions", "  " + json.dumps(dict(report.partition_sizes))]

    if m.calibration_warning:
        lines += ["", "## Calibration warning", f"  {m.calibration_warning}"]

    return "\n".join(lines)


def write_report(report: FullReport, path_stem: str | Path) -> None:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stem.with_suffix(".txt").write_text(render_report(report), encoding="utf-8")

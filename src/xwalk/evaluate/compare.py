"""Comparing runs. This is how you actually "select an LLM".

`misjudged` sits beside precision on purpose: precision alone cannot tell you whether a
worse model is worse at judging or was simply handed a worse candidate list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.evaluate.ceiling import CeilingBucket, ceiling_report
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import evaluate_results
from xwalk.ledger import Ledger

_COLUMNS = (
    ("label", 24),
    ("precision", 10),
    ("coverage", 10),
    ("review", 8),
    ("errors", 8),
    ("calls", 7),
    ("tokens", 9),
    ("secs", 7),
    ("misjudged", 10),
)


@dataclass(frozen=True)
class RunSummary:
    label: str
    run_fingerprint: str
    labelled: int
    accepted_precision: float | None
    automatic_coverage: float | None
    review_rate: float | None
    error_rate: float | None
    mean_llm_calls: float | None
    mean_tokens: float | None
    mean_seconds: float | None
    misjudged: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "run_fingerprint": self.run_fingerprint,
            "labelled": self.labelled,
            "accepted_precision": self.accepted_precision,
            "automatic_coverage": self.automatic_coverage,
            "review_rate": self.review_rate,
            "error_rate": self.error_rate,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_tokens": self.mean_tokens,
            "mean_seconds": self.mean_seconds,
            "misjudged": self.misjudged,
        }


def summarise_run(ledger: Ledger, run_fingerprint: str, gold: GoldSet, *, label: str) -> RunSummary:
    results = list(ledger.iter_results(run_fingerprint))
    metrics = evaluate_results(results, gold)
    ceiling = ceiling_report(results, gold)
    return RunSummary(
        label=label,
        run_fingerprint=run_fingerprint,
        labelled=metrics.labelled,
        accepted_precision=metrics.accepted_precision,
        automatic_coverage=metrics.automatic_coverage,
        review_rate=metrics.review_rate,
        error_rate=metrics.error_rate,
        mean_llm_calls=metrics.mean_llm_calls,
        mean_tokens=metrics.mean_tokens,
        mean_seconds=metrics.mean_seconds,
        misjudged=ceiling.buckets.get(CeilingBucket.MISJUDGED, 0),
    )


def _cell(value: Any, width: int) -> str:
    if value is None:
        text = "n/a"
    elif isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return text[:width].ljust(width)


def compare_runs(summaries: Sequence[RunSummary]) -> str:
    if not summaries:
        return "no runs to compare"

    scored = [s for s in summaries if s.accepted_precision is not None]
    best = max(scored, key=lambda s: s.accepted_precision or 0.0) if scored else None

    header = "  " + " ".join(name.ljust(width) for name, width in _COLUMNS)
    lines = [header, "  " + "-" * (len(header) - 2)]
    for summary in summaries:
        marker = "* " if best is not None and summary is best else "  "
        lines.append(
            marker
            + " ".join(
                _cell(value, width)
                for value, (_, width) in zip(
                    (
                        summary.label,
                        summary.accepted_precision,
                        summary.automatic_coverage,
                        summary.review_rate,
                        summary.error_rate,
                        summary.mean_llm_calls,
                        summary.mean_tokens,
                        summary.mean_seconds,
                        summary.misjudged,
                    ),
                    _COLUMNS,
                    strict=True,
                )
            )
        )
    return "\n".join(lines)


def compare_runs_dict(summaries: Sequence[RunSummary]) -> list[dict[str, Any]]:
    return [s.as_dict() for s in summaries]

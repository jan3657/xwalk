"""Ablation: flip one flag, report the delta.

The standard set includes halving the selector budget precisely because the ceiling
decomposition separates budget misses from retrieval misses -- an ablation that confirms
a diagnosis is worth more than one that only measures a component.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import EvalReport, evaluate_results
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import Record
from xwalk.stages.select import SelectorPolicy


@dataclass(frozen=True)
class MatcherConfig:
    name: str
    retriever_names: tuple[str, ...]
    policy: MatchPolicy
    selector_policy: SelectorPolicy


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    apply: Callable[[MatcherConfig], MatcherConfig]


def _drop_retriever(name: str) -> Callable[[MatcherConfig], MatcherConfig]:
    """Built outside the loop on purpose: a closure over the loop variable would make
    every ablation drop the last retriever."""

    def drop(config: MatcherConfig) -> MatcherConfig:
        return replace(
            config,
            name=f"no_{name}",
            retriever_names=tuple(n for n in config.retriever_names if n != name),
        )

    return drop


def standard_ablations(retriever_names: Sequence[str]) -> list[Ablation]:
    ablations: list[Ablation] = [
        Ablation(
            name=f"no_{name}",
            description=f"drop the {name} retriever",
            apply=_drop_retriever(name),
        )
        for name in retriever_names
    ]

    ablations.append(
        Ablation(
            name="no_verifier",
            description="never buy a second opinion",
            apply=lambda c: replace(
                c, name="no_verifier", policy=replace(c.policy, verify_band=None)
            ),
        )
    )
    ablations.append(
        Ablation(
            name="no_retries",
            description="one attempt only; no reformulation",
            apply=lambda c: replace(c, name="no_retries", policy=replace(c.policy, max_attempts=1)),
        )
    )
    ablations.append(
        Ablation(
            name="half_budget",
            description="halve the selector budget",
            apply=lambda c: replace(
                c,
                name="half_budget",
                selector_policy=replace(
                    c.selector_policy,
                    max_candidates=max(1, c.selector_policy.max_candidates // 2),
                ),
            ),
        )
    )
    return ablations


@dataclass(frozen=True)
class AblationRow:
    name: str
    description: str
    accepted_precision: float | None
    automatic_coverage: float | None
    review_rate: float | None
    mean_llm_calls: float | None
    delta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "accepted_precision": self.accepted_precision,
            "automatic_coverage": self.automatic_coverage,
            "review_rate": self.review_rate,
            "mean_llm_calls": self.mean_llm_calls,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class AblationReport:
    rows: tuple[AblationRow, ...]
    objective: str

    def as_dict(self) -> dict[str, Any]:
        return {"objective": self.objective, "rows": [r.as_dict() for r in self.rows]}


async def ablate(
    matcher_factory: Callable[[MatcherConfig], Matcher],
    base: MatcherConfig,
    records: Sequence[Record],
    gold: GoldSet,
    *,
    ablations: Sequence[Ablation] | None = None,
    objective: str = "accepted_precision",
    out: str | Path | None = None,
) -> AblationReport:
    chosen = list(ablations if ablations is not None else standard_ablations(base.retriever_names))

    async def run(config: MatcherConfig) -> tuple[EvalReport, float | None]:
        matcher = matcher_factory(config)
        results = [await matcher.match(record) for record in records]
        report = evaluate_results(results, gold)
        value = getattr(report, objective, None)
        return report, None if value is None else float(value)

    baseline_report, baseline_score = await run(base)
    rows = [
        AblationRow(
            name="baseline",
            description="all components enabled",
            accepted_precision=baseline_report.accepted_precision,
            automatic_coverage=baseline_report.automatic_coverage,
            review_rate=baseline_report.review_rate,
            mean_llm_calls=baseline_report.mean_llm_calls,
            delta=0.0,
        )
    ]

    for ablation in chosen:
        report, score = await run(ablation.apply(base))
        delta = 0.0
        if score is not None and baseline_score is not None:
            delta = score - baseline_score
        elif score is None and baseline_score is not None:
            # The ablation destroyed the metric entirely. Reporting 0.0 would read as
            # "no effect", which is the opposite of what happened.
            delta = -baseline_score
        rows.append(
            AblationRow(
                name=ablation.name,
                description=ablation.description,
                accepted_precision=report.accepted_precision,
                automatic_coverage=report.automatic_coverage,
                review_rate=report.review_rate,
                mean_llm_calls=report.mean_llm_calls,
                delta=delta,
            )
        )

    result = AblationReport(rows=tuple(rows), objective=objective)
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result

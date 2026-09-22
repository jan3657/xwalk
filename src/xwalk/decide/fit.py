"""Fit the decider policy's thresholds on a finished run and its gold labels.

Nothing here calls a model. It re-derives statuses from the signals already in the
ledger, so fitting is free and repeatable. The probe showed Jev's probabilities move by
a few hundredths between identical calls, so every point also reports how many labelled
rows sit within `margin` of the threshold: a policy that only looks good because of
where the jitter landed is visible as a large `near_threshold`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from xwalk.decide.policy import DecisionPolicy, Signals, derive_status
from xwalk.evaluate.gold import GoldSet
from xwalk.records import MatchResult, MatchStatus

DEFAULT_ACCEPT_GRID: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
DEFAULT_PROPERTY_GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)


@dataclass(frozen=True)
class FitPoint:
    accept_at: float
    property_floor: float
    accepted: int
    correct: int
    precision: float | None
    coverage: float
    near_threshold: int


@dataclass(frozen=True)
class FitReport:
    points: list[FitPoint]
    recommended: FitPoint | None
    target_precision: float
    labelled: int


def _signals_of(result: MatchResult) -> Signals | None:
    if not result.attempts:
        return None
    attempt = result.attempts[0]
    return Signals.from_flat(
        attempt.signals,
        candidate_count=attempt.candidate_count,
        retrieval_failed=result.status is MatchStatus.FAILED,
        resolution=attempt.resolution,
    )


def fit_thresholds(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    base: DecisionPolicy,
    target_precision: float = 0.95,
    accept_grid: Sequence[float] = DEFAULT_ACCEPT_GRID,
    property_grid: Sequence[float] = DEFAULT_PROPERTY_GRID,
    margin: float = 0.05,
) -> FitReport:
    labelled: list[tuple[MatchResult, Signals]] = []
    for result in results:
        if result.source_id not in gold:
            continue
        signals = _signals_of(result)
        if signals is not None:
            labelled.append((result, signals))

    points: list[FitPoint] = []
    for accept_at in accept_grid:
        for property_floor in property_grid:
            policy = replace(base, accept_at=accept_at, property_floor=property_floor)
            accepted = correct = near = 0
            for result, signals in labelled:
                chosen = signals.screen_chosen
                if chosen is not None and abs(chosen - accept_at) <= margin:
                    near += 1
                status, _ = derive_status(signals, policy)
                if status is MatchStatus.MATCHED:
                    accepted += 1
                    if gold.is_correct(result.source_id, result.matched_id):
                        correct += 1
            points.append(
                FitPoint(
                    accept_at=accept_at,
                    property_floor=property_floor,
                    accepted=accepted,
                    correct=correct,
                    precision=(correct / accepted) if accepted else None,
                    coverage=(accepted / len(labelled)) if labelled else 0.0,
                    near_threshold=near,
                )
            )
    eligible = [p for p in points if p.precision is not None and p.precision >= target_precision]
    recommended = max(
        eligible,
        key=lambda p: (p.accepted, p.accept_at, p.property_floor),
        default=None,
    )
    return FitReport(
        points=points,
        recommended=recommended,
        target_precision=target_precision,
        labelled=len(labelled),
    )


def render_fit(report: FitReport) -> str:
    lines = [
        f"labelled rows: {report.labelled}; target precision: {report.target_precision:.2f}",
        "",
        f"{'accept_at':>9} {'prop_floor':>10} {'accepted':>8} {'correct':>7} "
        f"{'precision':>9} {'coverage':>8} {'near':>4}",
    ]
    for p in report.points:
        precision = "  -" if p.precision is None else f"{p.precision:9.2f}"
        lines.append(
            f"{p.accept_at:9.2f} {p.property_floor:10.2f} {p.accepted:8d} {p.correct:7d} "
            f"{precision:>9} {p.coverage:8.2f} {p.near_threshold:4d}"
        )
    lines.append("")
    if report.recommended is None:
        lines.append(
            "no grid point meets the target precision; lower the target or improve the questions"
        )
    else:
        r = report.recommended
        lines.append(
            f"recommended: accept_at={r.accept_at:.2f} property_floor={r.property_floor:.2f} "
            f"({r.correct}/{r.accepted} correct, coverage {r.coverage:.2f}, "
            f"{r.near_threshold} rows within the jitter margin of accept_at)"
        )
    return "\n".join(lines)

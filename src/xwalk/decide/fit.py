"""Fit the decider policy's thresholds on a finished run and its gold labels.

Nothing here calls a model. It re-derives statuses from the signals already in the
ledger, so fitting is free and repeatable. The probe showed Jev's probabilities move by
a few hundredths between identical calls, so every point also reports how many labelled
rows sit within `margin` of the threshold: a policy that only looks good because of
where the jitter landed is visible as a large `near_threshold`.

A point fitted and scored on the same rows flatters itself. `fit_holdout` splits the
labelled ids in two, fits on one half and reports the recommended point on the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from xwalk.decide.policy import DecisionPolicy, Signals, derive_status
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.partition import Partitioner
from xwalk.records import MatchResult, MatchStatus

DEFAULT_ACCEPT_GRID: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
DEFAULT_PROPERTY_GRID: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)
DEFAULT_CHOOSE_GRID: tuple[float, ...] = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class FitPoint:
    accept_at: float
    property_floor: float
    choose_at: float
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


def _labelled(results: Iterable[MatchResult], gold: GoldSet) -> list[tuple[MatchResult, Signals]]:
    labelled: list[tuple[MatchResult, Signals]] = []
    for result in results:
        if result.source_id not in gold:
            continue
        signals = _signals_of(result)
        if signals is not None:
            labelled.append((result, signals))
    return labelled


def _score(
    labelled: Sequence[tuple[MatchResult, Signals]],
    gold: GoldSet,
    policy: DecisionPolicy,
    margin: float,
) -> FitPoint:
    accepted = correct = near = 0
    for result, signals in labelled:
        chosen = signals.screen_chosen
        if chosen is not None and abs(chosen - policy.accept_at) <= margin:
            near += 1
        status, _ = derive_status(signals, policy)
        if status is MatchStatus.MATCHED:
            accepted += 1
            if gold.is_correct(result.source_id, result.matched_id):
                correct += 1
    return FitPoint(
        accept_at=policy.accept_at,
        property_floor=policy.property_floor,
        choose_at=policy.choose_at,
        accepted=accepted,
        correct=correct,
        precision=(correct / accepted) if accepted else None,
        coverage=(accepted / len(labelled)) if labelled else 0.0,
        near_threshold=near,
    )


def fit_thresholds(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    base: DecisionPolicy,
    target_precision: float = 0.95,
    accept_grid: Sequence[float] = DEFAULT_ACCEPT_GRID,
    property_grid: Sequence[float] = DEFAULT_PROPERTY_GRID,
    choose_grid: Sequence[float] = DEFAULT_CHOOSE_GRID,
    margin: float = 0.05,
) -> FitReport:
    labelled = _labelled(results, gold)
    # The run's own choose_at is always swept, so every point the sweep offered before
    # choose_at joined it is still on offer.
    choose_values = sorted({*choose_grid, base.choose_at})

    points: list[FitPoint] = []
    for accept_at in accept_grid:
        for property_floor in property_grid:
            for choose_at in choose_values:
                policy = replace(
                    base, accept_at=accept_at, property_floor=property_floor, choose_at=choose_at
                )
                points.append(_score(labelled, gold, policy, margin))
    eligible = [p for p in points if p.precision is not None and p.precision >= target_precision]
    # Most accepted wins; on a tie, the stricter gates.
    recommended = max(
        eligible,
        key=lambda p: (p.accepted, p.accept_at, p.property_floor, p.choose_at),
        default=None,
    )
    return FitReport(
        points=points,
        recommended=recommended,
        target_precision=target_precision,
        labelled=len(labelled),
    )


def holdout_split(source_ids: Iterable[str], seed_fingerprint: str) -> tuple[list[str], list[str]]:
    """Split ids into (dev, holdout) halves whose sizes differ by at most one.

    Ids are ordered by a hash draw salted with `seed_fingerprint` and dealt alternately,
    so the split depends only on the ids and the seed, never on input order.
    """
    partitioner = Partitioner(salt=seed_fingerprint)
    ordered = sorted(set(source_ids), key=lambda sid: (partitioner.draw(sid), sid))
    return sorted(ordered[0::2]), sorted(ordered[1::2])


def fit_holdout(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    base: DecisionPolicy,
    seed_fingerprint: str,
    target_precision: float = 0.95,
    accept_grid: Sequence[float] = DEFAULT_ACCEPT_GRID,
    property_grid: Sequence[float] = DEFAULT_PROPERTY_GRID,
    choose_grid: Sequence[float] = DEFAULT_CHOOSE_GRID,
    margin: float = 0.05,
) -> tuple[FitReport, FitPoint | None]:
    """Fit on the dev half; score the recommendation on the holdout half it never saw.

    Returns the dev report and the holdout point, or None when nothing was recommended.
    """
    labelled = _labelled(results, gold)
    dev_ids, holdout_ids = holdout_split((r.source_id for r, _ in labelled), seed_fingerprint)
    dev, holdout = set(dev_ids), set(holdout_ids)
    report = fit_thresholds(
        [r for r, _ in labelled if r.source_id in dev],
        gold,
        base=base,
        target_precision=target_precision,
        accept_grid=accept_grid,
        property_grid=property_grid,
        choose_grid=choose_grid,
        margin=margin,
    )
    r = report.recommended
    if r is None:
        return report, None
    policy = replace(
        base, accept_at=r.accept_at, property_floor=r.property_floor, choose_at=r.choose_at
    )
    unseen = [(result, signals) for result, signals in labelled if result.source_id in holdout]
    return report, _score(unseen, gold, policy, margin)


def render_fit(report: FitReport) -> str:
    lines = [
        f"labelled rows: {report.labelled}; target precision: {report.target_precision:.2f}",
        "",
        f"{'accept_at':>9} {'prop_floor':>10} {'choose_at':>9} {'accepted':>8} {'correct':>7} "
        f"{'precision':>9} {'coverage':>8} {'near':>4}",
    ]
    for p in report.points:
        precision = "  -" if p.precision is None else f"{p.precision:9.2f}"
        lines.append(
            f"{p.accept_at:9.2f} {p.property_floor:10.2f} {p.choose_at:9.2f} "
            f"{p.accepted:8d} {p.correct:7d} "
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
            f"choose_at={r.choose_at:.2f} "
            f"({r.correct}/{r.accepted} correct, coverage {r.coverage:.2f}, "
            f"{r.near_threshold} rows within the jitter margin of accept_at)"
        )
    return "\n".join(lines)


def _thresholds(point: FitPoint) -> tuple[float, float, float]:
    return point.accept_at, point.property_floor, point.choose_at


def render_holdout(point: FitPoint | None, full: FitPoint | None = None) -> str:
    """One line: the dev-half recommendation scored on the holdout half.

    `full` is the recommendation fitted on every labelled row. When the dev half chose
    different thresholds the line says so, because the holdout numbers then do not
    describe the point `--write-job` writes.
    """
    if point is None:
        return "holdout: no holdout point (nothing was recommended on the dev half)"
    precision = "-" if point.precision is None else f"{point.precision:.2f}"
    line = (
        f"holdout: accept_at={point.accept_at:.2f} property_floor={point.property_floor:.2f} "
        f"choose_at={point.choose_at:.2f} accepted={point.accepted} correct={point.correct} "
        f"precision={precision} coverage={point.coverage:.2f}"
    )
    if full is not None and _thresholds(full) != _thresholds(point):
        line += (
            " (holdout of the dev-half recommendation, which differs from the full-data "
            "recommendation)"
        )
    return line

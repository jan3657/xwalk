"""Policy for the decider path: thresholds on calibrated probabilities, and nothing else.

The model returns probabilities; this module turns them into one of four outcomes. The
defaults are starting points from the September 2026 probe. `xwalk fit` replaces them
with values fitted on gold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from xwalk.records import DecisionReason, MatchStatus
from xwalk.stages.keying import Resolution

_PROP = "prop_"


@dataclass(frozen=True)
class DecisionPolicy:
    screen_floor: float = 0.30
    shortlist_size: int = 15
    shortlist_floor: float = 0.20
    none_at: float = 0.70
    choose_at: float = 0.50
    accept_at: float = 0.85
    rubric_floor: float | None = None
    property_floor: float = 0.50
    chunk_size: int = 50
    max_candidates: int = 300
    concurrency: int = 32
    retriever_timeout: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "screen_floor",
            "shortlist_floor",
            "none_at",
            "choose_at",
            "accept_at",
            "property_floor",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.shortlist_floor > self.accept_at:
            raise ValueError(
                f"shortlist_floor ({self.shortlist_floor}) must not exceed "
                f"accept_at ({self.accept_at})"
            )
        for name in ("shortlist_size", "chunk_size", "max_candidates", "concurrency"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")


@dataclass(frozen=True)
class Signals:
    """Everything the decider path learned about one record, as numbers."""

    candidate_count: int
    retrieval_failed: bool
    screen_best: float | None = None
    screen_chosen: float | None = None
    resolution: str | None = None
    p_choice: float | None = None
    p_none: float | None = None
    choice_confidence: float | None = None
    rubric: float | None = None
    rubric_levels: int | None = None
    rubric_confidence: float | None = None
    properties: Mapping[str, float] = field(default_factory=dict)

    _NUMERIC = (
        "screen_best",
        "screen_chosen",
        "p_choice",
        "p_none",
        "choice_confidence",
        "rubric",
        "rubric_levels",
        "rubric_confidence",
    )

    def flat(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in self._NUMERIC:
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        for name, value in sorted(self.properties.items()):
            out[f"{_PROP}{name}"] = float(value)
        return out

    @classmethod
    def from_flat(
        cls,
        flat: Mapping[str, float],
        *,
        candidate_count: int,
        retrieval_failed: bool,
        resolution: str | None,
    ) -> Signals:
        levels = flat.get("rubric_levels")
        return cls(
            candidate_count=candidate_count,
            retrieval_failed=retrieval_failed,
            screen_best=flat.get("screen_best"),
            screen_chosen=flat.get("screen_chosen"),
            resolution=resolution,
            p_choice=flat.get("p_choice"),
            p_none=flat.get("p_none"),
            choice_confidence=flat.get("choice_confidence"),
            rubric=flat.get("rubric"),
            rubric_levels=None if levels is None else int(levels),
            rubric_confidence=flat.get("rubric_confidence"),
            properties={k[len(_PROP) :]: v for k, v in flat.items() if k.startswith(_PROP)},
        )


def effective_rubric_floor(policy: DecisionPolicy, levels: int) -> float:
    if policy.rubric_floor is not None:
        return policy.rubric_floor
    return float(max(0, levels - 2))


def derive_status(signals: Signals, policy: DecisionPolicy) -> tuple[MatchStatus, DecisionReason]:
    if signals.retrieval_failed:
        return MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE
    if signals.candidate_count == 0 or signals.screen_best is None:
        return MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES
    if signals.screen_best < policy.screen_floor:
        return MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR
    if signals.resolution == Resolution.ABSTAIN.value:
        if (signals.p_none or 0.0) >= policy.none_at:
            return MatchStatus.UNMATCHED, DecisionReason.SELECTOR_ABSTAINED
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT
    if signals.resolution != Resolution.EXACT_KEY.value or signals.screen_chosen is None:
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT

    accepted = (
        signals.screen_chosen >= policy.accept_at
        and (signals.p_choice or 0.0) >= policy.choose_at
        and signals.rubric is not None
        and signals.rubric_levels is not None
        and signals.rubric >= effective_rubric_floor(policy, signals.rubric_levels)
        and all(p >= policy.property_floor for p in signals.properties.values())
    )
    if accepted:
        return MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD
    return MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD


def render_explanation(signals: Signals) -> str:
    """A reviewer-facing line built only from numbers, so it is the same every run."""
    if signals.candidate_count == 0 or signals.screen_best is None:
        return "no candidates"
    parts = [f"same {signals.screen_best:.2f}"]
    if signals.screen_chosen is not None and signals.screen_chosen != signals.screen_best:
        parts.append(f"chosen {signals.screen_chosen:.2f}")
    if signals.resolution == Resolution.ABSTAIN.value:
        parts.append(f"NONE {signals.p_none or 0.0:.2f}")
    elif signals.p_choice is not None:
        parts.append(f"p_choice {signals.p_choice:.2f}")
        if signals.p_none is not None:
            parts.append(f"p_none {signals.p_none:.2f}")
    if signals.rubric is not None and signals.rubric_levels:
        parts.append(f"rubric {signals.rubric:.1f}/{signals.rubric_levels - 1}")
    for name, value in sorted(signals.properties.items()):
        parts.append(f"{name} {value:.2f}")
    return "; ".join(parts)

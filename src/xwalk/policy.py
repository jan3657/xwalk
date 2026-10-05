"""Policy: cost controls, classification thresholds, and status derivation.

`verify_band` is a *cost control* — buy a second opinion only when the first is
uncertain. `accept_at` and `review_floor` are *classification*. The paper repo conflates
these; separating them lets each be tuned without disturbing the other.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.records import Attempt, DecisionReason, MatchStatus
from xwalk.stages.keying import Resolution

_EXACT_RESOLUTIONS = frozenset({Resolution.EXACT_KEY.value})
_FAILURE_REASONS = frozenset(
    {
        DecisionReason.RETRIEVER_FAILURE,
        DecisionReason.PROVIDER_FAILURE,
        DecisionReason.FATAL_PROVIDER_FAILURE,
    }
)
# A verifier verdict that lets a score stand. "error" (the verifier call failed) does not.
_UNCHALLENGED = frozenset({None, "support"})


@dataclass(frozen=True)
class MatchPolicy:
    max_attempts: int = 4
    accept_at: float = 0.6
    review_floor: float = 0.4
    verify_band: tuple[float, float] | None = (0.6, 0.8)
    audit_rate: float = 0.0
    concurrency: int = 32
    legacy_id_resolution: bool = False
    retriever_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if not 0.0 <= self.review_floor <= self.accept_at <= 1.0:
            raise ValueError(
                f"need 0 <= review_floor ({self.review_floor}) <= accept_at ({self.accept_at}) <= 1"
            )
        if not 0.0 <= self.audit_rate <= 1.0:
            raise ValueError(f"audit_rate must be in [0, 1], got {self.audit_rate}")
        if self.verify_band is not None:
            low, high = self.verify_band
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(
                    f"verify_band must be an ordered pair in [0, 1], got {self.verify_band}"
                )
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")


def should_verify(score: float | None, policy: MatchPolicy) -> bool:
    """Verification is bought only where the primary score is genuinely uncertain."""
    if score is None or policy.verify_band is None:
        return False
    low, high = policy.verify_band
    return low <= score <= high


def should_audit(
    score: float | None,
    policy: MatchPolicy,
    run_fingerprint: str,
    source_id: str,
) -> bool:
    """Sample a fraction of otherwise-automatic high-confidence matches.

    High-confidence errors exist and are by definition invisible. Sampling decides
    *which* results get a second opinion, never whether the opinion counts — an audit
    verdict is honoured exactly like any other verdict.

    Seeded from (run_fingerprint, source_id) so audits are reproducible across resumes
    rather than re-rolled each time a run restarts.
    """
    if policy.audit_rate <= 0.0 or score is None:
        return False
    if policy.verify_band is not None and score <= policy.verify_band[1]:
        return False  # already covered by verification
    digest = hashlib.sha256(f"{run_fingerprint}\x00{source_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return draw < policy.audit_rate


def _is_failure(attempt: Attempt) -> bool:
    return attempt.reason in _FAILURE_REASONS


def is_valid_score(score: float | None) -> bool:
    """A usable confidence: a finite number in [0, 1]. Stages already reject anything
    else; this keeps a score that reached an Attempt by any other route from passing a
    threshold (NaN compares false, +inf compares above every threshold)."""
    return score is not None and math.isfinite(score) and 0.0 <= score <= 1.0


def _has_choice(attempt: Attempt) -> bool:
    return attempt.chosen_id is not None and is_valid_score(attempt.primary_score)


def is_acceptable(attempt: Attempt, policy: MatchPolicy) -> bool:
    """Good enough to stop early: a valid exact-key score at or above `accept_at`, from
    an attempt that did not fail, with no verifier verdict against it."""
    return (
        attempt.reason is None
        and is_valid_score(attempt.primary_score)
        and (attempt.primary_score or 0.0) >= policy.accept_at
        and attempt.resolution in _EXACT_RESOLUTIONS
        and attempt.verifier_decision in _UNCHALLENGED
    )


def derive_status(
    attempts: Sequence[Attempt],
    policy: MatchPolicy,
) -> tuple[MatchStatus, DecisionReason, Attempt | None]:
    """Reduce a list of attempts to one status, one reason, and the winning attempt."""
    if not attempts:
        return MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE, None

    if any(a.reason is DecisionReason.FATAL_PROVIDER_FAILURE for a in attempts):
        # Configuration-level failure: nothing this record produced can be trusted to
        # be complete, and a batch run stops on it.
        return MatchStatus.FAILED, DecisionReason.FATAL_PROVIDER_FAILURE, None

    if all(_is_failure(a) for a in attempts):
        reason = attempts[-1].reason or DecisionReason.PROVIDER_FAILURE
        return MatchStatus.FAILED, reason, None

    scored = [a for a in attempts if _has_choice(a)]
    if scored:
        # Highest score wins. On a tie, an attempt that completed beats one whose
        # verifier call failed (a retry that re-scored and verified the same choice),
        # then the earlier attempt wins.
        best = min(scored, key=lambda a: (-(a.primary_score or 0.0), _is_failure(a), a.index))

        if best.verifier_decision in ("disagree", "no_match"):
            return MatchStatus.NEEDS_REVIEW, DecisionReason.VERIFIER_DISAGREEMENT, best

        if best.verifier_decision not in _UNCHALLENGED:
            # The score fell in the verify band but the verifier call failed: an
            # unverified in-band score is at best a candidate for review.
            return MatchStatus.NEEDS_REVIEW, DecisionReason.PROVIDER_FAILURE, best

        if best.resolution not in _EXACT_RESOLUTIONS:
            # legacy or otherwise inexact resolution never becomes an automatic match
            return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT, best

        score = best.primary_score or 0.0
        if score >= policy.accept_at:
            return MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD, best
        if score >= policy.review_floor:
            return MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD, best
        return MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR, best

    if any(a.resolution == Resolution.UNRESOLVED.value for a in attempts):
        last = next(a for a in reversed(attempts) if a.resolution == Resolution.UNRESOLVED.value)
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT, last

    if any(a.candidate_count > 0 for a in attempts):
        last = next(a for a in reversed(attempts) if a.candidate_count > 0)
        return MatchStatus.UNMATCHED, DecisionReason.SELECTOR_ABSTAINED, last

    return MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES, attempts[-1]

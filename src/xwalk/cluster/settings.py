"""Clustering settings: how much is retrieved and shown, and what each verdict must reach.

Every threshold is provisional. Calibrate on a small adjudicated development set before
relying on a run: a verifier's confidence is a routing signal, never proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

OrderRule = Literal["label", "input"]
MintProvenance = Literal["empty_pool", "pool_exhausted", "retrieval_exhausted", "bounded"]

# Strongest first. `mint_requires` names the weakest provenance that may mint.
PROVENANCE_STRENGTH: dict[str, int] = {
    "empty_pool": 3,
    "pool_exhausted": 3,
    "retrieval_exhausted": 2,
    "bounded": 1,
}


@dataclass(frozen=True)
class PoolSettings:
    """Retrieval over the growing cluster pool and the size of what the model sees."""

    retrieve_limit: int = 20
    shown_limit: int = 8
    expand_limit: int = 80
    max_expansion_pages: int = 2
    scan_below: int = 40
    member_evidence: int = 5
    evidence_chars: int = 200
    mint_requires: MintProvenance = "bounded"

    def __post_init__(self) -> None:
        for name in ("retrieve_limit", "shown_limit", "expand_limit", "member_evidence"):
            if getattr(self, name) < 1:
                raise ValueError(f"pool.{name} must be >= 1")
        for name in ("max_expansion_pages", "scan_below"):
            if getattr(self, name) < 0:
                raise ValueError(f"pool.{name} must be >= 0")
        if self.evidence_chars < 20:
            raise ValueError("pool.evidence_chars must be >= 20")
        if self.expand_limit < self.retrieve_limit:
            raise ValueError("pool.expand_limit must be >= pool.retrieve_limit")
        if self.mint_requires not in PROVENANCE_STRENGTH or self.mint_requires == "empty_pool":
            raise ValueError(
                "pool.mint_requires must be pool_exhausted, retrieval_exhausted or bounded"
            )


@dataclass(frozen=True)
class ClusterPolicy:
    """Verdict thresholds and refinement bounds. Provisional defaults."""

    verify_assignments: bool = True
    assign_accept_at: float = 0.7
    assign_review_floor: float = 0.5
    novelty_accept_at: float = 0.7
    merge_accept_at: float = 0.8
    merge_review_floor: float = 0.6
    comparative_accept_at: float = 0.8
    max_refine_iterations: int = 2
    consolidate: bool = True
    reassign: bool = True

    def __post_init__(self) -> None:
        for name in (
            "assign_accept_at",
            "assign_review_floor",
            "novelty_accept_at",
            "merge_accept_at",
            "merge_review_floor",
            "comparative_accept_at",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"policy.{name} must be within [0, 1], got {value}")
        if self.assign_review_floor > self.assign_accept_at:
            raise ValueError("policy.assign_review_floor must not exceed assign_accept_at")
        if self.merge_review_floor > self.merge_accept_at:
            raise ValueError("policy.merge_review_floor must not exceed merge_accept_at")
        if self.max_refine_iterations < 0:
            raise ValueError("policy.max_refine_iterations must be >= 0")


@dataclass(frozen=True)
class ClusterSettings:
    order: OrderRule = "label"
    pool: PoolSettings = PoolSettings()
    policy: ClusterPolicy = ClusterPolicy()

    def __post_init__(self) -> None:
        if self.order not in ("label", "input"):
            raise ValueError(f"order must be 'label' or 'input', got {self.order!r}")

    def components(self) -> dict[str, Any]:
        """The settings as fingerprint components."""
        return {"order": self.order, "pool": asdict(self.pool), "policy": asdict(self.policy)}

    @property
    def max_calls_per_member_decision(self) -> int:
        """Select pages plus one verification or novelty call."""
        return 2 + self.pool.max_expansion_pages


__all__ = [
    "PROVENANCE_STRENGTH",
    "ClusterPolicy",
    "ClusterSettings",
    "MintProvenance",
    "OrderRule",
    "PoolSettings",
]

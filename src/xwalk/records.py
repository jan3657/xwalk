"""Core value types. Everything in xwalk is built from these."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


@dataclass(frozen=True)
class Record:
    """One row from either collection. `fields` is whatever the source produced."""

    id: str
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError(f"Record.id must be str, got {type(self.id).__name__}")
        if not self.id.strip():
            raise ValueError("Record.id must be non-empty")
        if not isinstance(self.fields, Mapping):
            raise TypeError("Record.fields must be a Mapping")


@dataclass(frozen=True)
class RetrievalHit:
    """One retriever's opinion about one target record. `rank` is 1-based."""

    record_id: str
    retriever: str
    raw_score: float | None
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"RetrievalHit.rank is 1-based, got {self.rank}")


@dataclass(frozen=True)
class Candidate:
    """A fused candidate. `evidence` records every retriever that surfaced it."""

    record: Record
    fused_score: float
    evidence: tuple[RetrievalHit, ...]

    def __post_init__(self) -> None:
        for hit in self.evidence:
            if hit.record_id != self.record.id:
                raise ValueError(
                    f"Candidate evidence names {hit.record_id!r} "
                    f"but the record is {self.record.id!r}"
                )

    @property
    def id(self) -> str:
        return self.record.id


@dataclass(frozen=True)
class Usage:
    """Token and call accounting. Additive so attempts can be summed into a result."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @classmethod
    def zero(cls) -> Usage:
        return cls()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Any) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: Any) -> Usage:
        if other == 0:
            return self
        return self.__add__(other)


@dataclass(frozen=True)
class RetryProposal:
    """A lead for the next attempt.

    `kind="candidate"` means "look again at a record we already retrieved" — no new
    search. `kind="query"` means "run a genuinely new search". Conflating the two makes
    the loop re-retrieve records it already has in hand.
    """

    kind: Literal["candidate", "query"]
    value: str
    source: Literal["scorer", "rewriter"]


class MatchStatus(Enum):
    MATCHED = "matched"
    NEEDS_REVIEW = "needs_review"
    UNMATCHED = "unmatched"
    FAILED = "failed"


class DecisionReason(Enum):
    ACCEPT_THRESHOLD = "accept_threshold"
    BELOW_ACCEPT_THRESHOLD = "below_accept_threshold"
    BELOW_REVIEW_FLOOR = "below_review_floor"
    SELECTOR_ABSTAINED = "selector_abstained"
    NO_CANDIDATES = "no_candidates"
    UNRESOLVED_OUTPUT = "unresolved_output"
    RETRIEVER_FAILURE = "retriever_failure"
    PROVIDER_FAILURE = "provider_failure"
    VERIFIER_DISAGREEMENT = "verifier_disagreement"


@dataclass(frozen=True)
class Attempt:
    """One pass through retrieve -> select -> gate. The complete audit trail."""

    index: int
    query: str
    proposal: RetryProposal | None
    candidates: tuple[Candidate, ...]
    candidate_count: int
    candidates_truncated: int
    issued_keys: Mapping[str, str]
    raw_selection: str | None
    chosen_id: str | None
    resolution: str
    primary_score: float | None
    explanation: str
    verifier_decision: str | None
    verifier_score: float | None
    verifier_preferred_id: str | None
    audited: bool
    dropped_proposals: tuple[tuple[str, str], ...]
    reason: DecisionReason | None
    error: str | None
    usage: Usage


@dataclass(frozen=True)
class MatchResult:
    result_key: str
    source_id: str
    source_hash: str
    matched_id: str | None
    matched_record: Record | None
    confidence: float | None
    status: MatchStatus
    reason: DecisionReason
    explanation: str
    candidates: tuple[Candidate, ...]
    attempts: tuple[Attempt, ...]
    usage: Usage
    run_fingerprint: str

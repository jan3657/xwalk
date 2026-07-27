"""Core value types. Everything in xwalk is built from these."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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

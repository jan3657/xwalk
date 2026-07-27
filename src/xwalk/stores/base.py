"""Target storage, deliberately separate from retrieval.

Retrievers return IDs and ranks; the store turns IDs into Records. Keeping them apart
is what makes "plug in an external search backend for scale" actually work — otherwise
the retriever still has to hold every target record in memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from xwalk.records import Record


@runtime_checkable
class TargetStore(Protocol):
    @property
    def fingerprint(self) -> str:
        """Digest of the snapshot. Feeds the run fingerprint."""

    def get(self, record_id: str) -> Record:
        """Return one record, or raise KeyError."""

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        """Return records in request order, silently skipping unknown IDs."""


__all__ = ["TargetStore"]

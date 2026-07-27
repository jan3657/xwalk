"""The retrieval contract. Implement this to plug in any backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from xwalk.records import Record, RetrievalHit


class RetrieverError(Exception):
    """A retriever failed. The matcher degrades to the remaining retrievers."""


@dataclass(frozen=True)
class SearchRequest:
    """A request object rather than `search(query, k)`.

    Filters and field-aware retrieval can then be added without a breaking protocol
    change, and a backend that ignores `source_record` stays correct.
    """

    text: str
    limit: int
    filters: Mapping[str, Any] | None = None
    source_record: Record | None = None


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier. Appears in RetrievalHit.retriever and in diagnostics."""

    @property
    def fingerprint(self) -> str:
        """Digest of index content and configuration. Feeds the run fingerprint."""

    @property
    def default_limit(self) -> int:
        """This retriever's own retrieval depth.

        Depth belongs here rather than on the matcher: a BM25 index and a dense index
        have no reason to share a `k`. The matcher supplies a fallback only.
        """

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        """Return hits ranked best-first with 1-based contiguous ranks."""


__all__ = ["Retriever", "RetrieverError", "SearchRequest"]

"""The retrieval contract. Implement this to plug in any backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from xwalk.records import Record, RetrievalHit


class RetrieverError(Exception):
    """A retriever failed. The matcher degrades to the remaining retrievers."""


class IndexMismatchError(ValueError):
    """A persisted index does not match what the job expects (CONTRACTS.md section 7).

    `differences` maps each differing component to `(stored, expected)`. The index was
    not modified; rebuild it explicitly (`--rebuild-index`) or point at another directory.
    """

    def __init__(self, index_dir: object, differences: Mapping[str, tuple[Any, Any]]) -> None:
        self.index_dir = index_dir
        self.differences = dict(differences)
        names = ", ".join(sorted(self.differences)) or "unknown"
        super().__init__(
            f"index {index_dir} is incompatible with this job (differs in: {names}); "
            f"rebuild it with --rebuild-index or use another index directory"
        )


def component_differences(
    stored: Mapping[str, Any] | None, expected: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Top-level components that differ, as `{name: (stored, expected)}`. Nested
    mappings are compared key by key (`encoder.query_prefix`)."""
    if stored is None:
        return {"components": (None, "index metadata predates component validation")}
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(stored) | set(expected)):
        old, new = stored.get(key), expected.get(key)
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            for sub, pair in component_differences(old, new).items():
                differences[f"{key}.{sub}"] = pair
        elif old != new:
            differences[key] = (old, new)
    return differences


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


__all__ = [
    "IndexMismatchError",
    "Retriever",
    "RetrieverError",
    "SearchRequest",
    "component_differences",
]

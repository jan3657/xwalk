"""The growing cluster pool's retrieval index, updated incrementally.

The July plan rebuilt a Tantivy index from scratch on every pool commit. With one mint
per novel source that is quadratic in the number of clusters (CONTRACTS.md section 10.8).
Here a mint appends one document and a join re-indexes one cluster's document; BM25
statistics (document count, total length, document frequencies) are running counters,
so a score is computed from the current state without any rebuild.

Scores do not depend on insertion history: each document's score is summed over the
query terms in query order, so an index built incrementally and one built in one pass
from the same documents rank identically (resume relies on this).

The heads implement the `Retriever` protocol and are fused with the matcher's
`reciprocal_rank_fusion`, so the clustering pool and target retrieval share one fusion
rule. An optional dense head uses the existing `Encoder` protocol (brute-force cosine;
fine for thousands of clusters, not for millions).
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.fingerprint import hash_value
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest
from xwalk.retrieval.dense import Encoder, encoder_identity
from xwalk.retrieval.fusion import reciprocal_rank_fusion

POOL_INDEX_VERSION = 1
_TOKEN = re.compile(r"\w+", re.UNICODE)
_K1 = 1.2
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class LexicalPoolHead:
    """Incremental BM25 over cluster documents. A `Retriever`."""

    def __init__(self, *, name: str = "pool-bm25", default_limit: int = 20) -> None:
        self._name = name
        self._default_limit = default_limit
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_terms: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._total_len = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return hash_value(self.components())

    @property
    def default_limit(self) -> int:
        return self._default_limit

    def components(self) -> dict[str, Any]:
        return {"engine": "pool-bm25", "version": POOL_INDEX_VERSION, "k1": _K1, "b": _B}

    def __len__(self) -> int:
        return len(self._doc_len)

    def upsert(self, doc_id: str, text: str) -> None:
        self.remove(doc_id)
        terms: dict[str, int] = {}
        tokens = tokenize(text)
        for token in tokens:
            terms[token] = terms.get(token, 0) + 1
        for term, tf in terms.items():
            self._postings.setdefault(term, {})[doc_id] = tf
        self._doc_terms[doc_id] = terms
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)

    def remove(self, doc_id: str) -> None:
        terms = self._doc_terms.pop(doc_id, None)
        if terms is None:
            return
        for term in terms:
            postings = self._postings[term]
            del postings[doc_id]
            if not postings:
                del self._postings[term]
        self._total_len -= self._doc_len.pop(doc_id)

    def search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        count = len(self._doc_len)
        if not count or limit < 1:
            return []
        average = self._total_len / count if self._total_len else 1.0
        scores: dict[str, float] = {}
        for term in dict.fromkeys(tokenize(text)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                norm = _K1 * (1.0 - _B + _B * self._doc_len[doc_id] / average)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (_K1 + 1.0) / (tf + norm)
        best = heapq.nsmallest(limit, scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            RetrievalHit(record_id=doc_id, retriever=self._name, raw_score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(best, start=1)
        ]

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return self.search_sync(request.text, request.limit)


class DensePoolHead:
    """Brute-force cosine over unit vectors, appended per document. A `Retriever`."""

    def __init__(self, encoder: Encoder, *, default_limit: int = 20) -> None:
        self._encoder = encoder
        self._name = f"pool-dense:{encoder.name}"
        self._default_limit = default_limit
        self._vectors: dict[str, list[float]] = {}
        self._texts: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return hash_value(self.components())

    @property
    def default_limit(self) -> int:
        return self._default_limit

    def components(self) -> dict[str, Any]:
        return {
            "engine": "pool-dense",
            "version": POOL_INDEX_VERSION,
            "encoder": encoder_identity(self._encoder),
        }

    def upsert(self, doc_id: str, text: str) -> None:
        if self._texts.get(doc_id) == text:
            return
        self._texts[doc_id] = text
        self._vectors[doc_id] = self._encoder.encode([text])[0]

    def remove(self, doc_id: str) -> None:
        self._texts.pop(doc_id, None)
        self._vectors.pop(doc_id, None)

    def search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        if not self._vectors or limit < 1:
            return []
        query = self._encoder.encode([text], is_query=True)[0]
        scored = (
            (doc_id, sum(a * b for a, b in zip(query, vector, strict=False)))
            for doc_id, vector in self._vectors.items()
        )
        best = heapq.nsmallest(limit, scored, key=lambda item: (-item[1], item[0]))
        return [
            RetrievalHit(record_id=doc_id, retriever=self._name, raw_score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(best, start=1)
        ]

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return self.search_sync(request.text, request.limit)


@dataclass(frozen=True)
class PoolHit:
    """One retrieved cluster: 1-based fused rank, fused score, per-head ranks."""

    cluster_id: str
    rank: int
    score: float
    heads: dict[str, int]


class _IdStore:
    """The `TargetStore` fusion needs: every indexed id is a record with no fields.

    Holds the live mapping by reference; copying it per search would make each query
    O(pool size) and a run quadratic."""

    def __init__(self, ids: Mapping[str, str]) -> None:
        self._ids = ids

    @property
    def fingerprint(self) -> str:  # pragma: no cover - never fingerprinted
        return "pool"

    def get(self, record_id: str) -> Record:
        if record_id not in self._ids:
            raise KeyError(record_id)
        return Record(id=record_id, fields={})

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        return [Record(id=rid, fields={}) for rid in record_ids if rid in self._ids]


class PoolIndex:
    """Every live cluster's document, searchable through one or two heads."""

    def __init__(self, *, encoder: Encoder | None = None) -> None:
        self.lexical = LexicalPoolHead()
        self.dense = DensePoolHead(encoder) if encoder is not None else None
        self._docs: dict[str, str] = {}
        self.updates = 0  # documents added or replaced, for the scaling check

    def components(self) -> dict[str, Any]:
        return {
            "lexical": self.lexical.components(),
            "dense": None if self.dense is None else self.dense.components(),
            "fusion": "rrf-60",
        }

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, cluster_id: object) -> bool:
        return cluster_id in self._docs

    def upsert(self, cluster_id: str, text: str) -> None:
        if self._docs.get(cluster_id) == text:
            return
        self._docs[cluster_id] = text
        self.lexical.upsert(cluster_id, text)
        if self.dense is not None:
            self.dense.upsert(cluster_id, text)
        self.updates += 1

    def remove(self, cluster_id: str) -> None:
        self._docs.pop(cluster_id, None)
        self.lexical.remove(cluster_id)
        if self.dense is not None:
            self.dense.remove(cluster_id)

    def search(self, text: str, limit: int, *, exclude: Iterable[str] = ()) -> list[PoolHit]:
        excluded = set(exclude)
        depth = limit + sum(1 for cid in excluded if cid in self._docs)
        groups = [self.lexical.search_sync(text, depth)]
        if self.dense is not None:
            groups.append(self.dense.search_sync(text, depth))
        fused = reciprocal_rank_fusion(groups, _IdStore(self._docs))
        hits: list[PoolHit] = []
        for candidate in fused:
            if candidate.id in excluded:
                continue
            heads = {hit.retriever: hit.rank for hit in candidate.evidence}
            hits.append(PoolHit(candidate.id, len(hits) + 1, candidate.fused_score, heads))
            if len(hits) == limit:
                break
        return hits


__all__ = ["POOL_INDEX_VERSION", "DensePoolHead", "LexicalPoolHead", "PoolHit", "PoolIndex"]

"""The cluster pool index: incremental updates, history-independent ranking."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from xwalk.cluster.pool import LexicalPoolHead, PoolIndex
from xwalk.retrieval.base import Retriever, SearchRequest

DOCS = {
    "K1": "dark chocolate\nchocolate, dark",
    "K2": "milk chocolate",
    "K3": "chocolate",
    "K4": "heart attack\nmyocardial infarction",
    "K5": "white chocolate chips",
}


def _ranking(index: PoolIndex, text: str, limit: int = 10) -> list[tuple[str, float]]:
    return [(h.cluster_id, h.score) for h in index.search(text, limit)]


def test_heads_satisfy_the_retriever_protocol():
    assert isinstance(LexicalPoolHead(), Retriever)


async def test_lexical_head_searches_through_the_protocol():
    head = LexicalPoolHead()
    head.upsert("K1", "dark chocolate")
    hits = await head.search(SearchRequest(text="chocolate", limit=5))
    assert [h.record_id for h in hits] == ["K1"] and hits[0].rank == 1


def test_incremental_index_ranks_exactly_like_one_built_in_one_pass():
    incremental = PoolIndex()
    order = list(DOCS)
    random.Random(7).shuffle(order)
    # grow, edit and shrink in an arbitrary order, as a run does
    incremental.upsert("K9", "temporary noise chocolate")
    for cid in order:
        incremental.upsert(cid, DOCS[cid].split("\n")[0])
    for cid in order:
        incremental.upsert(cid, DOCS[cid])
    incremental.remove("K9")

    fresh = PoolIndex()
    for cid in sorted(DOCS):
        fresh.upsert(cid, DOCS[cid])

    for query in ("dark chocolate", "chocolate", "heart attack", "chips", "nothing"):
        got, want = _ranking(incremental, query), _ranking(fresh, query)
        assert [cid for cid, _ in got] == [cid for cid, _ in want]
        for (_, a), (_, b) in zip(got, want, strict=True):
            assert math.isclose(a, b, rel_tol=1e-12)


def test_removed_documents_are_not_returned_and_ties_break_by_id():
    index = PoolIndex()
    index.upsert("Kb", "apple")
    index.upsert("Ka", "apple")
    assert [h.cluster_id for h in index.search("apple", 5)] == ["Ka", "Kb"]
    index.remove("Ka")
    assert [h.cluster_id for h in index.search("apple", 5)] == ["Kb"]
    assert len(index) == 1


def test_exclusion_does_not_shorten_the_result():
    index = PoolIndex()
    for cid, text in DOCS.items():
        index.upsert(cid, text)
    hits = index.search("chocolate", 2, exclude={"K3", "K1"})
    assert len(hits) == 2 and {h.cluster_id for h in hits}.isdisjoint({"K1", "K3"})
    assert [h.rank for h in hits] == [1, 2]


def test_an_unchanged_document_is_not_reindexed():
    index = PoolIndex()
    index.upsert("K1", "dark chocolate")
    index.upsert("K1", "dark chocolate")
    assert index.updates == 1
    index.upsert("K1", "dark chocolate\nchocolate, dark")
    assert index.updates == 2


class ToyEncoder:
    """Deterministic 'embedding': which of a few concept words a text mentions."""

    AXES = (("heart", "myocardial", "cardiac"), ("chocolate", "cocoa"), ("fruit", "apple"))

    @property
    def name(self) -> str:
        return "toy"

    @property
    def dimension(self) -> int:
        return len(self.AXES)

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            raw = [float(any(w in lowered for w in axis)) for axis in self.AXES]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            vectors.append([v / norm for v in raw])
        return vectors


def test_dense_head_finds_a_synonym_with_no_shared_token():
    lexical_only = PoolIndex()
    with_dense = PoolIndex(encoder=ToyEncoder())
    for index in (lexical_only, with_dense):
        index.upsert("K4", "myocardial infarction")
        index.upsert("K2", "milk chocolate")
    assert lexical_only.search("cardiac arrest heart", 5) == []
    hits = with_dense.search("cardiac arrest heart", 5)
    assert hits[0].cluster_id == "K4" and "pool-dense:toy" in hits[0].heads
    assert with_dense.components()["dense"]["encoder"]["name"] == "toy"

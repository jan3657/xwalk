import pytest

from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.stores.memory import MemoryStore


@pytest.fixture
def store():
    return MemoryStore.from_source(
        [Record(id=letter, fields={"label": letter}) for letter in "ABCDE"]
    )


def hit(record_id, retriever, rank, score=1.0):
    return RetrievalHit(record_id=record_id, retriever=retriever, raw_score=score, rank=rank)


def test_single_retriever_preserves_its_order(store):
    hits = [hit("A", "bm25", 1), hit("B", "bm25", 2), hit("C", "bm25", 3)]
    assert [c.id for c in reciprocal_rank_fusion([hits], store)] == ["A", "B", "C"]


def test_agreement_beats_a_single_first_place(store):
    """B is second for both retrievers; A is first for one and absent from the other.

    RRF: A = 1/61 = 0.01639. B = 2 * 1/62 = 0.03226. Consensus wins, which is the
    entire reason for fusing rather than concatenating.
    """
    bm25 = [hit("A", "bm25", 1), hit("B", "bm25", 2)]
    dense = [hit("C", "dense", 1), hit("B", "dense", 2)]
    assert reciprocal_rank_fusion([bm25, dense], store)[0].id == "B"


def test_score_matches_the_rrf_formula(store):
    fused = reciprocal_rank_fusion([[hit("A", "bm25", 1)]], store, k=60)
    assert fused[0].fused_score == pytest.approx(1 / 61)


def test_k_is_configurable(store):
    fused = reciprocal_rank_fusion([[hit("A", "bm25", 1)]], store, k=10)
    assert fused[0].fused_score == pytest.approx(1 / 11)


def test_evidence_collects_every_retriever_that_surfaced_the_record(store):
    bm25 = [hit("A", "bm25", 1)]
    dense = [hit("A", "dense", 4)]
    [candidate] = reciprocal_rank_fusion([bm25, dense], store)
    assert {(e.retriever, e.rank) for e in candidate.evidence} == {("bm25", 1), ("dense", 4)}


def test_evidence_order_is_deterministic(store):
    bm25 = [hit("A", "bm25", 1)]
    dense = [hit("A", "dense", 4)]
    forward = reciprocal_rank_fusion([bm25, dense], store)[0].evidence
    backward = reciprocal_rank_fusion([dense, bm25], store)[0].evidence
    assert [(e.retriever, e.rank) for e in forward] == [(e.retriever, e.rank) for e in backward]


def test_ties_break_on_record_id(store):
    a = [hit("B", "r1", 1)]
    b = [hit("A", "r2", 1)]
    assert [c.id for c in reciprocal_rank_fusion([a, b], store)] == ["A", "B"]


def test_ids_missing_from_the_store_are_dropped(store):
    hits = [hit("A", "bm25", 1), hit("GONE", "bm25", 2)]
    assert [c.id for c in reciprocal_rank_fusion([hits], store)] == ["A"]


def test_empty_input_yields_no_candidates(store):
    assert reciprocal_rank_fusion([], store) == []
    assert reciprocal_rank_fusion([[], []], store) == []


def test_duplicate_hits_from_one_retriever_count_once(store):
    """A malformed backend returning the same ID twice must not double its score."""
    hits = [hit("A", "bm25", 1), hit("A", "bm25", 2)]
    [candidate] = reciprocal_rank_fusion([hits], store)
    assert candidate.fused_score == pytest.approx(1 / 61)
    assert len(candidate.evidence) == 1


def test_rejects_a_non_positive_k(store):
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([[hit("A", "bm25", 1)]], store, k=0)

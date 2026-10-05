import pytest

from xwalk.records import Candidate, Record, RetrievalHit, Usage


def test_record_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        Record(id="", fields={"name": "x"})


def test_record_rejects_non_string_id():
    with pytest.raises(TypeError, match="str"):
        Record(id=42, fields={})  # type: ignore[arg-type]


def test_record_is_frozen():
    record = Record(id="A1", fields={"name": "glucose"})
    with pytest.raises(AttributeError):
        record.id = "A2"  # type: ignore[misc]


def test_retrieval_hit_rejects_rank_below_one():
    with pytest.raises(ValueError, match="1-based"):
        RetrievalHit(record_id="A1", retriever="bm25", raw_score=1.0, rank=0)


def test_retrieval_hit_allows_missing_score():
    hit = RetrievalHit(record_id="A1", retriever="external", raw_score=None, rank=1)
    assert hit.raw_score is None


def test_candidate_id_delegates_to_record():
    record = Record(id="CHEBI:17234", fields={"label": "glucose"})
    hit = RetrievalHit(record_id="CHEBI:17234", retriever="bm25", raw_score=3.2, rank=1)
    candidate = Candidate(record=record, fused_score=0.016, evidence=(hit,))
    assert candidate.id == "CHEBI:17234"


def test_candidate_rejects_evidence_for_a_different_record():
    record = Record(id="A1", fields={})
    hit = RetrievalHit(record_id="B2", retriever="bm25", raw_score=1.0, rank=1)
    with pytest.raises(ValueError, match="evidence"):
        Candidate(record=record, fused_score=0.5, evidence=(hit,))


def test_usage_zero_is_all_zeroes():
    usage = Usage.zero()
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (0, 0, 0)
    assert usage.total_tokens == 0


def test_usage_adds_componentwise():
    total = Usage(prompt_tokens=10, completion_tokens=5, calls=1) + Usage(
        prompt_tokens=3, completion_tokens=2, calls=1
    )
    assert total == Usage(prompt_tokens=13, completion_tokens=7, calls=2)


def test_usage_sum_starts_from_zero():
    parts = [Usage(prompt_tokens=1, completion_tokens=1, calls=1) for _ in range(3)]
    assert sum(parts, Usage.zero()) == Usage(prompt_tokens=3, completion_tokens=3, calls=3)


def test_usage_adds_unknown_calls_and_cache_hits():
    total = Usage(calls=1, unknown_calls=1) + Usage(cache_hits=2)
    assert (total.calls, total.unknown_calls, total.cache_hits) == (1, 1, 2)


def test_unknown_usage_is_never_displayed_as_a_bare_zero():
    assert Usage(prompt_tokens=3, completion_tokens=1, calls=1).describe_tokens() == "4"
    assert Usage.unreported(2).describe_tokens() == "0 (+2 calls with unknown usage)"

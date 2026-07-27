from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)
from xwalk.serde import result_from_dict, result_to_dict


def sample_result() -> MatchResult:
    record = Record(id="T1", fields={"label": "glucose", "synonyms": ["dextrose"]})
    hit = RetrievalHit(record_id="T1", retriever="bm25", raw_score=3.5, rank=1)
    candidate = Candidate(record=record, fused_score=0.016, evidence=(hit,))
    attempt = Attempt(
        index=0,
        query="glucose",
        proposal=RetryProposal(kind="query", value="dextrose", source="rewriter"),
        candidates=(candidate,),
        candidate_count=1,
        candidates_truncated=3,
        issued_keys={"C01": "T1"},
        raw_selection='{"chosen_key": "C01"}',
        chosen_id="T1",
        resolution="exact_key",
        primary_score=0.91,
        explanation="exact synonym match",
        verifier_decision="support",
        verifier_score=0.88,
        verifier_preferred_id=None,
        audited=True,
        dropped_proposals=(("C99", "candidate key 'C99' was not issued this attempt"),),
        reason=DecisionReason.ACCEPT_THRESHOLD,
        error=None,
        usage=Usage(prompt_tokens=100, completion_tokens=20, calls=2),
    )
    return MatchResult(
        result_key="rk1",
        source_id="s1",
        source_hash="sh1",
        matched_id="T1",
        matched_record=record,
        confidence=0.91,
        status=MatchStatus.MATCHED,
        reason=DecisionReason.ACCEPT_THRESHOLD,
        explanation="exact synonym match",
        candidates=(candidate,),
        attempts=(attempt,),
        usage=Usage(prompt_tokens=100, completion_tokens=20, calls=2),
        run_fingerprint="fp1",
    )


def test_round_trip_preserves_everything():
    original = sample_result()
    assert result_from_dict(result_to_dict(original)) == original


def test_enums_serialise_as_their_values():
    data = result_to_dict(sample_result())
    assert data["status"] == "matched"
    assert data["reason"] == "accept_threshold"


def test_evidence_survives_the_round_trip():
    restored = result_from_dict(result_to_dict(sample_result()))
    assert restored.candidates[0].evidence[0].retriever == "bm25"


def test_a_none_matched_record_round_trips():
    original = sample_result()
    unmatched = MatchResult(
        **{
            **original.__dict__,
            "matched_id": None,
            "matched_record": None,
            "status": MatchStatus.UNMATCHED,
            "reason": DecisionReason.NO_CANDIDATES,
        }
    )
    assert result_from_dict(result_to_dict(unmatched)).matched_record is None


def test_a_none_proposal_round_trips():
    original = sample_result()
    attempt = Attempt(**{**original.attempts[0].__dict__, "proposal": None})
    result = MatchResult(**{**original.__dict__, "attempts": (attempt,)})
    assert result_from_dict(result_to_dict(result)).attempts[0].proposal is None


def test_the_serialised_form_is_json_safe():
    import json

    json.dumps(result_to_dict(sample_result()))  # must not raise

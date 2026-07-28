from xwalk.evaluate.ceiling import CeilingBucket, ceiling_report, classify_ceiling
from xwalk.evaluate.gold import GoldSet
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    Usage,
)

GOLD = GoldSet({"s1": frozenset({"T1"}), "s2": frozenset(), "s3": frozenset({"T3"})})


def cand(record_id, retrievers=("bm25",)):
    return Candidate(
        record=Record(id=record_id, fields={}),
        fused_score=0.5,
        evidence=tuple(
            RetrievalHit(record_id=record_id, retriever=name, raw_score=1.0, rank=i)
            for i, name in enumerate(retrievers, start=1)
        ),
    )


def attempt(index, candidates, issued, truncated=0):
    return Attempt(
        index=index,
        query="q",
        proposal=None,
        candidates=tuple(candidates),
        candidate_count=len(candidates),
        candidates_truncated=truncated,
        issued_keys=issued,
        raw_selection=None,
        chosen_id=None,
        resolution="abstain",
        primary_score=None,
        explanation="",
        verifier_decision=None,
        verifier_score=None,
        verifier_preferred_id=None,
        audited=False,
        dropped_proposals=(),
        reason=None,
        error=None,
        usage=Usage.zero(),
        elapsed_seconds=0.0,
        finish_reason="stop",
    )


def result(source_id, matched_id, attempts, status=MatchStatus.MATCHED):
    return MatchResult(
        result_key=f"rk-{source_id}",
        source_id=source_id,
        source_hash="h",
        matched_id=matched_id,
        matched_record=None if matched_id is None else Record(id=matched_id, fields={}),
        confidence=0.9,
        status=status,
        reason=DecisionReason.ACCEPT_THRESHOLD,
        explanation="",
        candidates=tuple(attempts[-1].candidates) if attempts else (),
        attempts=tuple(attempts),
        usage=Usage.zero(),
        elapsed_seconds=0.0,
        run_fingerprint="fp1",
    )


# --- the four outcomes -----------------------------------------------------------


def test_a_correct_match_is_found():
    r = result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.FOUND


def test_gold_absent_from_every_candidate_list_is_never_retrieved():
    r = result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.NEVER_RETRIEVED


def test_gold_retrieved_but_not_issued_a_key_is_truncated():
    """A budget miss, not a retrieval miss. Raise max_candidates, not the retriever."""
    r = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)])
    assert classify_ceiling(r, GOLD) is CeilingBucket.TRUNCATED


def test_gold_presented_but_not_chosen_is_misjudged():
    r = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.MISJUDGED


def test_the_three_failure_buckets_are_mutually_exclusive():
    cases = [
        result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)]),
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})]),
    ]
    assert len({classify_ceiling(c, GOLD) for c in cases}) == 3


# --- edges ------------------------------------------------------------------------


def test_a_no_match_gold_label_gets_its_own_bucket():
    r = result("s2", None, [attempt(0, [], {})], status=MatchStatus.UNMATCHED)
    assert classify_ceiling(r, GOLD) is CeilingBucket.NO_GOLD


def test_an_unlabelled_result_gets_its_own_bucket():
    r = result("s99", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])
    assert classify_ceiling(r, GOLD) is CeilingBucket.UNLABELLED


def test_gold_retrieved_in_a_later_attempt_still_counts_as_retrieved():
    """Reformulation recovered it; retrieval is not the bottleneck."""
    r = result(
        "s1",
        "T9",
        [
            attempt(0, [cand("T9")], {"C01": "T9"}),
            attempt(1, [cand("T1"), cand("T9")], {"C01": "T1", "C02": "T9"}),
        ],
    )
    assert classify_ceiling(r, GOLD) is CeilingBucket.MISJUDGED


def test_a_result_with_no_attempts_is_never_retrieved():
    r = result("s1", None, [], status=MatchStatus.FAILED)
    assert classify_ceiling(r, GOLD) is CeilingBucket.NEVER_RETRIEVED


def test_multi_gold_counts_any_listed_id_as_retrieved():
    gold = GoldSet({"s1": frozenset({"T1", "T2"})})
    r = result("s1", "T9", [attempt(0, [cand("T2")], {"C01": "T2"})])
    assert classify_ceiling(r, gold) is CeilingBucket.MISJUDGED


# --- the report -------------------------------------------------------------------


def test_report_counts_every_bucket():
    results = [
        result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})]),
        result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
    ]
    report = ceiling_report(results, GOLD)
    assert report.buckets[CeilingBucket.FOUND] == 1
    assert report.buckets[CeilingBucket.NEVER_RETRIEVED] == 1


def test_report_attributes_retrieval_per_retriever():
    """Which retriever surfaced the gold record, when any did."""
    results = [
        result("s1", "T1", [attempt(0, [cand("T1", ("bm25", "dense"))], {"C01": "T1"})]),
        result("s3", "T3", [attempt(0, [cand("T3", ("dense",))], {"C01": "T3"})]),
    ]
    report = ceiling_report(results, GOLD)
    assert report.by_retriever["dense"] == 2
    assert report.by_retriever["bm25"] == 1


def test_report_records_which_attempt_first_surfaced_the_gold():
    results = [
        result(
            "s1",
            "T1",
            [attempt(0, [cand("T9")], {"C01": "T9"}), attempt(1, [cand("T1")], {"C01": "T1"})],
        ),
    ]
    assert ceiling_report(results, GOLD).by_attempt[1] == 1


def test_report_excludes_unlabelled_rows_from_the_failure_totals():
    results = [result("s99", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])]
    assert ceiling_report(results, GOLD).evaluable == 0


def test_report_as_dict_is_json_safe():
    import json

    results = [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])]
    json.dumps(ceiling_report(results, GOLD).as_dict())


def test_report_recommends_the_budget_when_truncation_dominates():
    results = [
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)]),
        result("s3", "T9", [attempt(0, [cand("T9"), cand("T3")], {"C01": "T9"}, truncated=1)]),
    ]
    assert "budget" in ceiling_report(results, GOLD).recommendation.lower()


def test_report_recommends_retrieval_when_never_retrieved_dominates():
    results = [
        result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
        result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]),
    ]
    assert "retriev" in ceiling_report(results, GOLD).recommendation.lower()


def test_report_recommends_prompts_when_misjudgement_dominates():
    results = [
        result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})]),
        result("s3", "T9", [attempt(0, [cand("T9"), cand("T3")], {"C01": "T9", "C02": "T3"})]),
    ]
    assert "prompt" in ceiling_report(results, GOLD).recommendation.lower()


def test_a_clean_run_says_so_rather_than_recommending_at_random():
    results = [result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])]
    assert "no ceiling failures" in ceiling_report(results, GOLD).recommendation

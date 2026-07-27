from xwalk.records import RetryProposal
from xwalk.stages.proposals import normalise_query, route_proposals

ISSUED = ("C01", "C02", "C03")


def prop(kind, value, source="scorer"):
    return RetryProposal(kind=kind, value=value, source=source)


def test_candidate_proposals_naming_issued_keys_are_kept():
    routed = route_proposals([prop("candidate", "C02")], ISSUED, seen_queries=set())
    assert routed.candidate_keys == ("C02",)


def test_candidate_proposals_naming_unissued_keys_are_dropped():
    routed = route_proposals([prop("candidate", "C99")], ISSUED, seen_queries=set())
    assert routed.candidate_keys == ()
    assert routed.dropped[0][0].value == "C99"


def test_a_dropped_candidate_is_never_promoted_to_a_query():
    """A candidate proposal naming something not in the set is a hallucination, not a
    new lead. Promoting it would run a full retrieval on invented text."""
    routed = route_proposals([prop("candidate", "C99")], ISSUED, seen_queries=set())
    assert routed.queries == ()


def test_the_drop_reason_is_recorded():
    routed = route_proposals([prop("candidate", "C99")], ISSUED, seen_queries=set())
    assert "not issued" in routed.dropped[0][1]


def test_query_proposals_enter_the_query_queue():
    routed = route_proposals([prop("query", "tumour protein p53")], ISSUED, seen_queries=set())
    assert routed.queries == ("tumour protein p53",)


def test_queries_already_tried_are_dropped():
    routed = route_proposals([prop("query", "TP53")], ISSUED, seen_queries={"tp53"})
    assert routed.queries == ()
    assert "already tried" in routed.dropped[0][1]


def test_query_deduplication_ignores_case_and_whitespace():
    routed = route_proposals([prop("query", "  TP53  ")], ISSUED, seen_queries={"tp53"})
    assert routed.queries == ()


def test_duplicate_queries_within_one_batch_are_collapsed():
    routed = route_proposals(
        [prop("query", "p53"), prop("query", "P53")], ISSUED, seen_queries=set()
    )
    assert routed.queries == ("p53",)


def test_blank_proposals_are_dropped():
    routed = route_proposals([prop("query", "   ")], ISSUED, seen_queries=set())
    assert routed.queries == () and len(routed.dropped) == 1


def test_candidate_keys_are_deduplicated_and_order_preserved():
    routed = route_proposals(
        [prop("candidate", "C03"), prop("candidate", "C01"), prop("candidate", "C03")],
        ISSUED,
        seen_queries=set(),
    )
    assert routed.candidate_keys == ("C03", "C01")


def test_candidate_key_matching_is_case_insensitive():
    routed = route_proposals([prop("candidate", "c02")], ISSUED, seen_queries=set())
    assert routed.candidate_keys == ("C02",)


def test_normalise_query_lowercases_and_collapses_whitespace():
    assert normalise_query("  TP53   gene ") == "tp53 gene"


def test_empty_input_routes_to_nothing():
    routed = route_proposals([], ISSUED, seen_queries=set())
    assert routed.candidate_keys == () and routed.queries == () and routed.dropped == ()

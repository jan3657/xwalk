"""Policy replay is a reduction of paid-for evidence, never a new match run."""

from dataclasses import replace

import pytest

from tests.test_ceiling import attempt, cand, result
from tests.test_matcher import (
    SOURCE,
    ScriptedRetriever,
    build,
    score_reply,
    select_reply,
    verify_reply,
)
from xwalk.decide.policy import DecisionPolicy
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import threshold_curve
from xwalk.llm.fake import FakeLLM
from xwalk.policy import MatchPolicy
from xwalk.records import DecisionReason, MatchStatus


def scored(rid="T1", score=0.7, **changes):
    return replace(
        attempt(0, [cand("T1"), cand("T2")], {"C01": "T1", "C02": "T2"}),
        chosen_id=rid,
        resolution="exact_key",
        primary_score=score,
        **changes,
    )


def test_threshold_curve_keeps_verifier_disagreement():
    a = scored(verifier_decision="disagree")
    row = result("s1", "T1", [a], MatchStatus.NEEDS_REVIEW)
    points = threshold_curve([row], GoldSet({"s1": frozenset({"T1"})}))
    assert all(p.accepted == 0 for p in points)
    assert all(p.conditional_on_recorded_attempts for p in points)


@pytest.mark.parametrize("reply", ["match", "review", "unresolved", "abstain", "empty", "failure"])
async def test_same_policy_replays_generated_matcher_ledgers(tmp_path, reply):
    from xwalk.evaluate.replay import replay_results
    from xwalk.ledger import Ledger
    from xwalk.llm.base import LLMRetryableError

    policy = MatchPolicy(max_attempts=1)
    replies = {
        "match": [select_reply("C01"), score_reply(0.95)],
        "review": [select_reply("C01"), score_reply(0.7), verify_reply("disagree")],
        "unresolved": ["{}"],
        "abstain": [select_reply(None)],
        "empty": [],
        "failure": [LLMRetryableError("down")],
    }
    retriever = ScriptedRetriever({"glucose": [] if reply == "empty" else ["T1"]})
    row = await build(FakeLLM(replies[reply]), retriever, policy=policy).match(SOURCE)
    ledger = Ledger.open(tmp_path / "ledger.sqlite")
    try:
        await ledger.put_result(row)
        rows = list(ledger.iter_results(row.run_fingerprint))
    finally:
        ledger.close()
    replay = replay_results(rows, GoldSet({"s1": frozenset({"T1"})}), recorded_policy=policy)
    assert [(r.status, r.reason, r.matched_id, r.confidence) for r in replay.results] == [
        (row.status, row.reason, row.matched_id, row.confidence)
    ]
    assert replay.status_transitions == {row.status.value: {row.status.value: 1}}


@pytest.mark.parametrize(
    "case", ["match", "property", "none", "low_screen", "empty", "provider", "retriever"]
)
async def test_same_policy_replays_generated_decider_ledgers(tmp_path, case):
    from tests.test_decide_matcher import SOURCE as DSOURCE
    from tests.test_decide_matcher import Scripted, _matcher, confident
    from xwalk.decide.base import DecisionError
    from xwalk.decide.fake import FakeDecider
    from xwalk.evaluate.replay import replay_results
    from xwalk.ledger import Ledger
    from xwalk.records import Record

    policy = DecisionPolicy()
    source = DSOURCE
    retrievers = None
    decider = FakeDecider(handler=confident)
    if case == "property":
        policy = replace(policy, property_floor=1.0)
    elif case == "none":
        policy = replace(policy, screen_floor=0.0)
        source = Record("s", {"mention": "other", "alias": "", "note": ""})
        retrievers = [Scripted({"other": ["T2"]})]
    elif case == "low_screen":
        policy = replace(policy, screen_floor=1.0)
    elif case == "empty":
        retrievers = [Scripted({})]
    elif case == "provider":

        def fail(state, questions):
            raise DecisionError("down")

        decider = FakeDecider(handler=fail)
    elif case == "retriever":
        retrievers = [Scripted({}, fail=True)]
    row = await _matcher(decider=decider, retrievers=retrievers, policy=policy).match(source)
    ledger = Ledger.open(tmp_path / "ledger.sqlite")
    try:
        await ledger.put_result(row)
        rows = list(ledger.iter_results("fp"))
    finally:
        ledger.close()
    replay = replay_results(rows, GoldSet({source.id: frozenset({"T1"})}), recorded_policy=policy)
    assert [(r.status, r.reason, r.matched_id, r.confidence) for r in replay.results] == [
        (row.status, row.reason, row.matched_id, row.confidence)
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"verify_band": None},
        {"max_attempts": 9},
        {"audit_rate": 0.2},
        {"retriever_timeout": 2},
        {"legacy_id_resolution": True},
    ],
)
def test_replay_refuses_unrecorded_acquisition(changes):
    from xwalk.evaluate.replay import replay_results

    base = MatchPolicy()
    with pytest.raises(ValueError, match="unrecorded calls"):
        replay_results([], GoldSet({}), recorded_policy=base, policy=replace(base, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"screen_floor": 0.5},
        {"shortlist_size": 3},
        {"shortlist_floor": 0.1},
        {"chunk_size": 2},
        {"max_candidates": 2},
    ],
)
def test_decider_replay_refuses_routing_changes(changes):
    from xwalk.evaluate.replay import replay_results

    base = DecisionPolicy()
    with pytest.raises(ValueError, match="unrecorded calls"):
        replay_results([], GoldSet({}), recorded_policy=base, policy=replace(base, **changes))


def test_decider_replay_preserves_property_gate_and_provider_failure():
    from xwalk.evaluate.replay import replay_result

    a = scored(reason=DecisionReason.BELOW_ACCEPT_THRESHOLD)
    a = replace(
        a,
        signals={
            "screen_best": 0.9,
            "screen_chosen": 0.9,
            "p_choice": 0.9,
            "rubric": 2.0,
            "rubric_levels": 3.0,
            "prop_form": 0.1,
        },
    )
    row = result("s1", "T1", [a], MatchStatus.NEEDS_REVIEW)
    assert replay_result(row, DecisionPolicy(accept_at=0.2)).status is MatchStatus.NEEDS_REVIEW
    failed = replace(row, attempts=(replace(a, reason=DecisionReason.PROVIDER_FAILURE),))
    assert replay_result(failed, DecisionPolicy(accept_at=0.2)).status is MatchStatus.FAILED


def test_replay_reports_historical_reducer_change_and_gold_transitions():
    from xwalk.evaluate.replay import replay_results

    row = result(
        "s1",
        "T2",
        [scored("T2", 0.9, verifier_decision="disagree"), replace(scored("T1", 0.8), index=1)],
        MatchStatus.NEEDS_REVIEW,
    )
    replay = replay_results(
        [row],
        GoldSet({"s1": frozenset({"T1"})}),
        recorded_policy=MatchPolicy(),
        recorded_semantics_version=1,
    )
    assert replay.status_transitions == {"needs_review": {"matched": 1}}
    assert replay.gold_transitions == {"review": {"accepted_correct": 1}}
    assert "historical" in " ".join(replay.warnings)
    assert replay.as_dict()["status_semantics_version"] == 2


def test_stage_diagnostics_are_separate_from_ceiling_and_retry_is_descriptive():
    from xwalk.evaluate.replay import replay_results

    rows = [
        result("below", "T1", [scored(score=0.5)], MatchStatus.NEEDS_REVIEW),
        result("verifier", "T1", [scored(verifier_decision="disagree")], MatchStatus.NEEDS_REVIEW),
        result("selector", "T2", [scored("T2", 0.9)]),
        result("lost", "T2", [scored("T1", 0.5), replace(scored("T2", 0.9), index=1)]),
        result("recovered", "T1", [scored("T2", 0.5), replace(scored("T1", 0.9), index=1)]),
    ]
    gold = GoldSet({r.source_id: frozenset({"T1"}) for r in rows})
    report = replay_results(rows, gold, recorded_policy=MatchPolicy())
    assert report.misjudged["scorer_below"] == 2
    assert report.misjudged["verifier_disagreed"] == 1
    assert report.misjudged["selector_wrong"] == 2
    assert report.misjudged["lost_by_retry"] == 1
    assert report.retry_metrics["recovered"] == 1
    assert report.retry_metrics["broken_by_retry"] == 1
    assert "descriptive" in report.as_dict()["retry_definition"]


@pytest.mark.parametrize("decider", [False, True])
async def test_evaluation_never_calls_clients_or_retrievers(tmp_path, monkeypatch, decider):
    from xwalk.decide.fake import FakeDecider
    from xwalk.evaluate.replay import replay_results
    from xwalk.evaluate.report import evaluate
    from xwalk.ledger import Ledger
    from xwalk.llm.openai_compat import OpenAICompatClient
    from xwalk.retrieval.bm25 import BM25Retriever

    policy = MatchPolicy(max_attempts=1)
    row = await build(
        FakeLLM([select_reply("C01"), score_reply(0.9)]),
        ScriptedRetriever({"glucose": ["T1"]}),
        policy=policy,
    ).match(SOURCE)

    if decider:
        from tests.test_decide_matcher import SOURCE as DSOURCE
        from tests.test_decide_matcher import _matcher

        matcher = _matcher()
        policy = matcher.policy
        row = await matcher.match(DSOURCE)

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation made a call")

    for cls, method in [
        (FakeLLM, "complete"),
        (OpenAICompatClient, "complete"),
        (FakeDecider, "decide"),
        (ScriptedRetriever, "search"),
        (BM25Retriever, "search"),
    ]:
        monkeypatch.setattr(cls, method, forbidden)
    ledger = Ledger.open(tmp_path / "ledger.sqlite")
    gold = GoldSet({row.source_id: frozenset({"T1"})})
    try:
        await ledger.put_result(row)
        assert evaluate(ledger, row.run_fingerprint, gold).metrics.accepted_precision == 1.0
        assert (
            replay_results(
                list(ledger.iter_results(row.run_fingerprint)), gold, recorded_policy=policy
            )
            .results[0]
            .status
            is MatchStatus.MATCHED
        )
    finally:
        ledger.close()


def test_retry_counts_match_benchmark_definitions_including_correct_review():
    from benchmarks.metrics import Prediction, matching_metrics
    from xwalk.evaluate.replay import retry_metrics

    rows = [
        result("review", "T1", [scored(), replace(scored(), index=1)], MatchStatus.NEEDS_REVIEW),
        result("recovered", "T1", [scored("T2"), replace(scored(), index=1)]),
        result("one", "T1", [scored()], MatchStatus.NEEDS_REVIEW),
    ]
    gold = GoldSet({r.source_id: frozenset({"T1"}) for r in rows})
    predictions = [
        Prediction(
            r.source_id,
            r.status.value,
            r.matched_id,
            first_choice_id=r.attempts[0].chosen_id,
            has_first_choice=True,
            attempts=len(r.attempts),
        )
        for r in rows
    ]
    expected = matching_metrics(predictions, gold)
    assert retry_metrics(rows, gold) == {k: expected[k] for k in ("recovered", "broken_by_retry")}


@pytest.mark.parametrize("decider", [False, True])
def test_replay_keeps_degraded_negative_and_fatal_precedence(decider):
    from xwalk.evaluate.replay import replay_result

    a = scored(score=0.1, retrieval_failures=("bm25: down",))
    policy = MatchPolicy()
    if decider:
        policy = DecisionPolicy()
        a = replace(a, signals={"screen_best": 0.1, "screen_chosen": 0.1})
    row = result("s1", None, [a], MatchStatus.FAILED)
    replayed = replay_result(row, policy)
    assert (replayed.status, replayed.reason) == (
        MatchStatus.FAILED,
        DecisionReason.RETRIEVER_FAILURE,
    )
    a = replace(a, primary_score=0.99, reason=DecisionReason.FATAL_PROVIDER_FAILURE)
    assert (
        replay_result(replace(row, attempts=(a,)), policy).reason
        is DecisionReason.FATAL_PROVIDER_FAILURE
    )


def test_no_attempts_are_unavailable_for_replay():
    from xwalk.evaluate.replay import replay_result

    with pytest.raises(ValueError, match="unavailable"):
        replay_result(result("s1", "T1", []), MatchPolicy())


def test_lower_threshold_can_recover_a_suppressed_choice_from_attempts():
    from xwalk.evaluate.replay import replay_results

    row = result("s1", None, [scored(score=0.3)], MatchStatus.UNMATCHED)
    gold = GoldSet({"s1": frozenset({"T1"})})
    replay = replay_results(
        [row],
        gold,
        recorded_policy=MatchPolicy(),
        policy=MatchPolicy(accept_at=0.2, review_floor=0.1),
    )
    assert replay.results[0].matched_id == "T1"
    assert replay.results[0].status is MatchStatus.MATCHED
    assert replay.gold_transitions == {"incorrect_no_match": {"accepted_correct": 1}}
    assert row.matched_id is None


def test_decider_selector_attribution_requires_gold_in_the_chooser_shortlist():
    from xwalk.evaluate.replay import decision_diagnostics

    a = replace(scored("T2", 0.9), signals={"screen_C02": 0.9})
    row = result("s1", "T2", [a])
    gold = GoldSet({"s1": frozenset({"T1"})})
    assert decision_diagnostics([row], gold, DecisionPolicy())["selector_wrong"] == 0
    shown = replace(row, attempts=(replace(a, signals={"screen_C01": 0.8, "screen_C02": 0.9}),))
    assert decision_diagnostics([shown], gold, DecisionPolicy())["selector_wrong"] == 1

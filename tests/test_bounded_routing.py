"""Batch 6 regressions derived from advisor probes P3/P4/P6/P8 and A#8."""

import dataclasses
import json
from pathlib import Path

import pytest

from tests.test_matcher import (
    SOURCE,
    ScriptedRetriever,
    build,
    rewrite_reply,
    score_reply,
    select_reply,
    verify_reply,
)
from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.policy import MatchPolicy
from xwalk.records import MatchStatus, RetryProposal


def base_cases():
    return {
        "P3": (
            [
                select_reply("C01"),
                score_reply(0.5, better_candidate_keys=["C02"]),
                score_reply(0.5, better_candidate_keys=["C01"]),
                score_reply(0.5, better_candidate_keys=["C02"]),
                score_reply(0.5),
            ],
            {"glucose": ["T1", "T2"]},
        ),
        "P6": (
            [
                select_reply("C01"),
                score_reply(0.7),
                verify_reply("disagree", preferred_key="C02"),
                rewrite_reply("q2"),
                select_reply("C01"),
                score_reply(0.7),
                verify_reply("disagree", preferred_key="C02"),
                rewrite_reply("q3"),
                select_reply("C02"),
                score_reply(0.7),
                verify_reply(),
            ],
            {q: ["T1", "T2"] for q in ("glucose", "q2", "q3")},
        ),
        "P8": (
            [
                select_reply("C01"),
                score_reply(0.5, better_candidate_keys=["C01", "C02"]),
                score_reply(0.5),
                score_reply(0.5),
                rewrite_reply(),
            ],
            {"glucose": ["T1", "T2"]},
        ),
        "A8": (
            [
                select_reply("C01"),
                score_reply(0.2, better_candidate_keys=["C02"], better_queries=["dextrose"]),
                score_reply(0.2, better_candidate_keys=["C01"]),
                score_reply(0.2, better_candidate_keys=["C02"]),
                score_reply(0.2),
            ],
            {"glucose": ["T2", "T3"], "dextrose": ["T1"]},
        ),
    }


def observations(result, llm, retriever):
    return {
        "requests": [dataclasses.asdict(r) for r in llm.requests],
        "queries": retriever.queries,
        "chosen": [a.chosen_id for a in result.attempts],
        "status": result.status.value,
        "reason": result.reason.value,
        "matched_id": result.matched_id,
        "confidence": result.confidence,
    }


@pytest.mark.parametrize("case", base_cases())
async def test_flags_off_match_captured_base_requests_and_behavior(case):
    captured = json.loads(
        (Path(__file__).parent / "fixtures/batch6_base_requests.json").read_text()
    )
    replies, queries = base_cases()[case]
    llm = FakeLLM(replies)
    retriever = ScriptedRetriever(queries)
    result = await build(llm, retriever).match(SOURCE)
    assert observations(result, llm, retriever) == captured[case]


@pytest.mark.parametrize("trace", [True, False])
@pytest.mark.parametrize("stage", ["score", "verification"])
async def test_p4_forced_failure_retries_live_candidate_with_provenance(trace, stage):
    replies = [select_reply("C01"), score_reply(0.5, better_candidate_keys=["C02"])]
    if stage == "verification":
        replies.append(score_reply(0.7))
    replies.append(LLMRetryableError("503 on forced attempt"))
    replies.append(score_reply(0.7 if stage == "verification" else 0.9))
    if stage == "verification":
        replies.append(verify_reply())
    llm = FakeLLM(replies)
    retriever = ScriptedRetriever({"glucose": ["T1", "T2"]})
    result = await build(llm, retriever, trace=trace, policy=MatchPolicy(max_attempts=3)).match(
        SOURCE
    )
    expected = ["selection", "score", "score", "score"]
    if stage == "verification":
        expected = ["selection", "score", "score", "verification", "score", "verification"]
    assert [r.schema_name for r in llm.requests] == expected
    assert retriever.queries == ["glucose"]
    assert result.status is MatchStatus.MATCHED
    assert result.matched_id == "T2"
    assert [a.chosen_id for a in result.attempts] == ["T1", "T2", "T2"]
    assert (
        result.attempts[1].proposal
        == result.attempts[2].proposal
        == RetryProposal("candidate", "C02", "scorer")
    )
    assert result.attempts[1].issued_keys == result.attempts[2].issued_keys
    assert result.attempts[1].raw_selection == result.attempts[2].raw_selection
    assert llm.requests[-2 if stage == "verification" else -1] == llm.requests[2]
    assert result.usage.calls == len(llm.requests)


async def capture_wire(case):
    import httpx

    from tests.test_matcher import PROMPTS, STORE, TEMPLATES
    from xwalk.batch import run_fingerprint_components
    from xwalk.llm.openai_compat import OpenAICompatClient
    from xwalk.stages.select import SelectorPolicy

    replies, queries = base_cases()[case]
    script = iter(replies)
    bodies = []

    def handler(request):
        bodies.append(request.content.decode())
        return httpx.Response(200, json={"choices": [{"message": {"content": next(script)}}]})

    llm = OpenAICompatClient(
        base_url="https://llm.test/v1",
        model="m",
        api_key="test",
        profile="unknown",
        transport=httpx.MockTransport(handler),
    )
    try:
        ret = ScriptedRetriever(queries)
        matcher = build(llm, ret)
        result = await matcher.match(SOURCE)
        components = run_fingerprint_components(
            templates=TEMPLATES,
            prompts=PROMPTS,
            store=STORE,
            retrievers=[ret],
            llm=llm,
            policy=MatchPolicy(),
            selector_policy=SelectorPolicy(),
            selector=matcher._selector,
            scorer=matcher._scorer,
            verifier=matcher._verifier,
            rewriter=matcher._rewriter,
        )
        return {"bodies": bodies, "identity": components, "status": result.status.value}
    finally:
        await llm.aclose()


@pytest.mark.parametrize("case", base_cases())
async def test_flags_off_match_base_http_bytes_and_full_identity(case):
    captured = json.loads((Path(__file__).parent / "fixtures/batch6_base_wire.json").read_text())
    assert await capture_wire(case) == captured[case]


FLAGS = ("deduplicate_candidates", "alternate_queues", "chase_verifier_leads")


@pytest.mark.parametrize("flag", FLAGS)
def test_routing_flags_config_identity_and_replay(flag):
    from tests.test_matcher import PROMPTS, STORE, TEMPLATES
    from xwalk.batch import run_fingerprint_components
    from xwalk.config import PolicySpec
    from xwalk.evaluate.replay import validate_replay_policy
    from xwalk.stages.select import SelectorPolicy

    assert getattr(PolicySpec(), flag) is False
    assert getattr(PolicySpec(**{flag: True}), flag) is True
    assert getattr(MatchPolicy(**PolicySpec(**{flag: True}).model_dump()), flag) is True
    llm = FakeLLM([])
    matcher = build(llm, ScriptedRetriever({}))

    def components(policy):
        return run_fingerprint_components(
            templates=TEMPLATES,
            prompts=PROMPTS,
            store=STORE,
            retrievers=matcher._retrievers,
            llm=llm,
            policy=policy,
            selector_policy=SelectorPolicy(),
            selector=matcher._selector,
            scorer=matcher._scorer,
            verifier=matcher._verifier,
            rewriter=matcher._rewriter,
        )

    off = components(MatchPolicy())
    assert all(f not in off["policy"] for f in FLAGS)
    on_policy = MatchPolicy(**{flag: True})
    on = components(on_policy)
    assert on["policy"].pop(flag) is True
    assert on == off
    with pytest.raises(ValueError, match=flag):
        validate_replay_policy(MatchPolicy(), on_policy)


@pytest.mark.parametrize("self_lead", [False, True], ids=["P3", "P8"])
async def test_dedup_completed_and_pending_scoring_work(self_lead):
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(
                0.5, better_candidate_keys=(["C01", "C02", "C03"] if self_lead else ["C02", "C03"])
            ),
            score_reply(0.5, better_candidate_keys=["C01", "C03"]),
            score_reply(0.5),
            rewrite_reply(),
        ]
    )
    ret = ScriptedRetriever({"glucose": ["T1", "T2", "T3"]})
    result = await build(llm, ret, policy=MatchPolicy(deduplicate_candidates=True)).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == [
        "selection",
        "score",
        "score",
        "score",
        "rewrite",
    ]
    assert [a.chosen_id for a in result.attempts] == ["T1", "T2", "T3"]
    assert ("C01", "scoring work already completed") in result.attempts[1].dropped_proposals
    assert ("C03", "scoring work already pending") in result.attempts[1].dropped_proposals


async def test_p3_two_candidate_ping_pong_stops_after_two_scores():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.5, better_candidate_keys=["C02"]),
            score_reply(0.5, better_candidate_keys=["C01"]),
            rewrite_reply(),
        ]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(deduplicate_candidates=True),
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == ["selection", "score", "score", "rewrite"]
    assert len(result.attempts) == 2


async def test_p8_self_lead_is_not_scored_again():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.5, better_candidate_keys=["C01", "C02"]),
            score_reply(0.5),
            rewrite_reply(),
        ]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(deduplicate_candidates=True),
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == ["selection", "score", "score", "rewrite"]
    assert [a.chosen_id for a in result.attempts] == ["T1", "T2"]


@pytest.mark.parametrize("dedup", [False, True])
async def test_a8_alternation_runs_useful_query_by_third_attempt(dedup):
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.2, better_candidate_keys=["C02"], better_queries=["dextrose"]),
            score_reply(0.2, better_candidate_keys=["C01"]),
            select_reply("C01"),
            score_reply(0.95),
        ]
    )
    ret = ScriptedRetriever({"glucose": ["T2", "T3"], "dextrose": ["T1"]})
    result = await build(
        llm,
        ret,
        policy=MatchPolicy(alternate_queues=True, deduplicate_candidates=dedup, max_attempts=3),
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == [
        "selection",
        "score",
        "score",
        "selection",
        "score",
    ]
    assert ret.queries == ["glucose", "dextrose"]
    assert [a.chosen_id for a in result.attempts] == ["T2", "T3", "T1"]
    assert result.status is MatchStatus.MATCHED


@pytest.mark.parametrize("lead_source", ["scorer", "verifier"])
async def test_queued_lead_keeps_original_map_after_an_intervening_search(lead_source):
    replies = [
        select_reply("C01"),
        score_reply(0.5, better_candidate_keys=["C02"], better_queries=["q2"]),
    ]
    if lead_source == "scorer":
        replies.append(score_reply(0.5, better_candidate_keys=["C03"]))
    else:
        replies.extend([score_reply(0.7), verify_reply("disagree", preferred_key="C03")])
    replies.extend([select_reply("C01"), score_reply(0.5), score_reply(0.95)])
    llm = FakeLLM(replies)
    ret = ScriptedRetriever({"glucose": ["T1", "T2", "T3"], "q2": ["T3", "T1", "T2"]})
    result = await build(
        llm,
        ret,
        policy=MatchPolicy(
            alternate_queues=True, chase_verifier_leads=True, deduplicate_candidates=True
        ),
    ).match(SOURCE)
    assert [a.chosen_id for a in result.attempts] == ["T1", "T2", "T3", "T3"]
    final = result.attempts[-1]
    assert final.issued_keys == result.attempts[0].issued_keys
    assert final.issued_keys != result.attempts[2].issued_keys
    assert final.query == "glucose"
    assert final.proposal == RetryProposal("candidate", "C03", lead_source)
    assert result.status is MatchStatus.MATCHED
    assert "[C03]" in llm.requests[-1].user


@pytest.mark.parametrize(
    "queries, scores",
    [
        ({"glucose": ["T1"], "q2": ["T1"]}, 1),
        ({"glucose": ["T1"], "q2": ["T2"]}, 2),
        ({"glucose": ["T1", "T2"], "q2": ["T1", "T3"]}, 2),
    ],
)
async def test_dedup_requires_same_record_and_same_scoring_context(queries, scores):
    replies = [select_reply("C01"), score_reply(0.5, better_queries=["q2"]), select_reply("C01")]
    if scores == 2:
        replies.append(score_reply(0.5))
    llm = FakeLLM(replies)
    result = await build(
        llm,
        ScriptedRetriever(queries),
        policy=MatchPolicy(max_attempts=2, deduplicate_candidates=True),
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests].count("score") == scores
    assert len(result.attempts) == 2


@pytest.mark.parametrize("bad_stage", ["score", "verification", "provider"])
async def test_incomplete_semantics_do_not_mark_work_examined(bad_stage):
    replies = [select_reply("C01")]
    if bad_stage == "score":
        replies.extend([score_reply("bad"), rewrite_reply("q2")])
    elif bad_stage == "verification":
        replies.extend([score_reply(0.7), '{"decision":"unknown"}', rewrite_reply("q2")])
    else:
        replies.append(LLMRetryableError("outage"))
    replies.extend([select_reply("C01"), score_reply(0.95)])
    llm = FakeLLM(replies)
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1"], "q2": ["T1"]}),
        policy=MatchPolicy(max_attempts=2, deduplicate_candidates=True),
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests].count("score") == 2
    assert result.status is MatchStatus.MATCHED


async def test_p6_verifier_lead_is_scored_verified_and_trace_invariant():
    runs = []
    for trace in (True, False):
        llm = FakeLLM(
            [
                select_reply("C01"),
                score_reply(0.7),
                verify_reply("disagree", preferred_key="C02"),
                score_reply(0.7),
                verify_reply(),
            ]
        )
        ret = ScriptedRetriever({"glucose": ["T1", "T2"]})
        result = await build(
            llm, ret, trace=trace, policy=MatchPolicy(**dict.fromkeys(FLAGS, True))
        ).match(SOURCE)
        assert [r.schema_name for r in llm.requests] == [
            "selection",
            "score",
            "verification",
            "score",
            "verification",
        ]
        assert result.status is MatchStatus.MATCHED
        assert result.matched_id == "T2"
        assert result.attempts[1].proposal == RetryProposal("candidate", "C02", "verifier")
        runs.append(
            (
                llm.requests,
                dataclasses.replace(
                    result,
                    candidates=(),
                    elapsed_seconds=0,
                    attempts=tuple(
                        dataclasses.replace(a, candidates=(), elapsed_seconds=0)
                        for a in result.attempts
                    ),
                ),
            )
        )
    assert runs[0] == runs[1]


@pytest.mark.parametrize(
    "decision, preferred, extra",
    [
        ("unknown", "C02", {}),
        ("support", "C02", {}),
        ("no_match", "C02", {}),
        ("disagree", "C01", {}),
        ("disagree", "C99", {}),
        ("disagree", "T2", {}),
        ("disagree", "C02", {"confidence_score": "bad"}),
    ],
)
async def test_only_well_formed_disagreement_with_another_issued_key_is_chased(
    decision, preferred, extra
):
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.7),
            verify_reply(decision, preferred_key=preferred, **extra),
            rewrite_reply(),
        ]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(chase_verifier_leads=True),
    ).match(SOURCE)
    assert len(result.attempts) == 1
    assert [r.schema_name for r in llm.requests].count("score") == 1


@pytest.mark.parametrize("bound", [1, 2, 6])
async def test_verifier_leads_enter_once_per_record_and_respect_attempt_bound(bound):
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.7),
            verify_reply("disagree", preferred_key="C02"),
            score_reply(0.7),
            verify_reply("disagree", preferred_key="C01"),
            score_reply(0.7),
            verify_reply("disagree", preferred_key="C02"),
            rewrite_reply(),
        ]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(max_attempts=bound, chase_verifier_leads=True),
    ).match(SOURCE)
    assert len(result.attempts) == min(bound, 3)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert [a.proposal.value for a in result.attempts if a.proposal] == ["C02", "C01"][: bound - 1]


async def test_infrastructure_retry_bypasses_dedup_and_alternation():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.5, better_candidate_keys=["C02"], better_queries=["q2"]),
            score_reply(0.7),
            LLMRetryableError("503"),
            score_reply(0.7),
            verify_reply(),
        ]
    )
    ret = ScriptedRetriever({"glucose": ["T1", "T2"], "q2": ["T3"]})
    result = await build(
        llm, ret, policy=MatchPolicy(max_attempts=3, **dict.fromkeys(FLAGS, True))
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == [
        "selection",
        "score",
        "score",
        "verification",
        "score",
        "verification",
    ]
    assert ret.queries == ["glucose"]
    assert result.status is MatchStatus.MATCHED


async def test_query_selection_does_not_duplicate_pending_candidate_work():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.5, better_candidate_keys=["C02", "C03"], better_queries=["q2"]),
            score_reply(0.5),
            select_reply("C03"),
            score_reply(0.95),
        ]
    )
    ret = ScriptedRetriever({q: ["T1", "T2", "T3"] for q in ("glucose", "q2")})
    result = await build(
        llm, ret, policy=MatchPolicy(alternate_queues=True, deduplicate_candidates=True)
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == [
        "selection",
        "score",
        "score",
        "selection",
        "score",
    ]
    assert result.attempts[2].primary_score is None
    assert result.attempts[2].dropped_proposals == (
        ("C03", "scoring work already pending or completed"),
    )
    assert result.attempts[3].proposal == RetryProposal("candidate", "C03", "scorer")
    assert result.matched_id == "T3"


async def test_verifier_lead_provider_retry_keeps_verifier_provenance():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.7),
            verify_reply("disagree", preferred_key="C02"),
            LLMRetryableError("outage"),
            score_reply(0.7),
            verify_reply(),
        ]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(**dict.fromkeys(FLAGS, True)),
    ).match(SOURCE)
    assert [r.schema_name for r in llm.requests] == [
        "selection",
        "score",
        "verification",
        "score",
        "score",
        "verification",
    ]
    assert (
        result.attempts[1].proposal
        == result.attempts[2].proposal
        == RetryProposal("candidate", "C02", "verifier")
    )
    assert result.status is MatchStatus.MATCHED


async def test_dedup_and_alternation_are_trace_invariant():
    runs = []
    for trace in (True, False):
        llm = FakeLLM(
            [
                select_reply("C01"),
                score_reply(
                    0.2, better_candidate_keys=["C01", "C02", "C03"], better_queries=["q2"]
                ),
                score_reply(0.2, better_candidate_keys=["C01", "C03"]),
                select_reply("C01"),
                score_reply(0.95),
            ]
        )
        ret = ScriptedRetriever({"glucose": ["T2", "T3", "T1"], "q2": ["T1"]})
        result = await build(
            llm, ret, trace=trace, policy=MatchPolicy(**dict.fromkeys(FLAGS, True))
        ).match(SOURCE)
        assert result.matched_id == "T1"
        assert len(result.attempts) == 3
        runs.append(
            (
                llm.requests,
                dataclasses.replace(
                    result,
                    candidates=(),
                    elapsed_seconds=0,
                    attempts=tuple(
                        dataclasses.replace(a, candidates=(), elapsed_seconds=0)
                        for a in result.attempts
                    ),
                ),
            )
        )
    assert runs[0] == runs[1]


async def test_infrastructure_retries_remain_bounded_with_all_flags_on():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.5, better_candidate_keys=["C02"]),
            LLMRetryableError("503"),
            LLMRetryableError("503"),
            LLMRetryableError("503"),
        ]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(**dict.fromkeys(FLAGS, True)),
    ).match(SOURCE)
    assert len(result.attempts) == 4
    assert [r.schema_name for r in llm.requests] == [
        "selection",
        "score",
        "score",
        "score",
        "score",
    ]


@pytest.mark.parametrize(
    "reply, expected",
    [
        (select_reply("C01", 0.23), 0.23),
        (select_reply(None, 0.95), 0.95),
        (select_reply(None, 0.2), 0.2),
        (select_reply("C99", 0.8), 0.8),
        ('{"confidence_score": 0.4}', 0.4),
        (select_reply("C01", "0.9"), None),
        (select_reply(None, float("nan")), None),
        ("not JSON", None),
    ],
)
async def test_selector_confidence_is_stored_independently_of_scoring(reply, expected):
    from xwalk.policy import derive_status
    from xwalk.serde import result_from_dict, result_to_dict

    llm = FakeLLM([reply, score_reply(0.95)])
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    attempt = result.attempts[0]
    assert attempt.selector_confidence == expected
    blob = result_to_dict(result)
    assert blob["attempts"][0]["selector_confidence"] == expected
    assert result_from_dict(blob) == result
    del blob["attempts"][0]["selector_confidence"]
    assert result_from_dict(blob).attempts[0].selector_confidence is None
    # Advisory storage changes no status, reason or primary confidence.
    classifications = []
    for confidence in (None, 0.0, 1.0):
        status, reason, best = derive_status(
            [dataclasses.replace(attempt, selector_confidence=confidence)],
            MatchPolicy(max_attempts=1),
        )
        classifications.append((status, reason, best.primary_score if best else None))
    assert classifications[0] == classifications[1] == classifications[2]


async def test_selector_confidence_survives_scoring_failure_and_is_absent_on_forced_work():
    llm = FakeLLM(
        [
            select_reply("C01", 0.12),
            LLMRetryableError("503"),
            select_reply("C01", 0.34),
            score_reply(0.5, better_candidate_keys=["C02"]),
            score_reply(0.95),
        ]
    )
    result = await build(llm, ScriptedRetriever({"glucose": ["T1", "T2"]})).match(SOURCE)
    assert [a.selector_confidence for a in result.attempts] == [0.12, 0.34, None]
    assert result.confidence == 0.95
    assert result.status is MatchStatus.MATCHED


async def test_selector_failure_has_no_confidence():
    result = await build(
        FakeLLM([LLMRetryableError("503")]),
        ScriptedRetriever({"glucose": ["T1"]}),
        policy=MatchPolicy(max_attempts=1),
    ).match(SOURCE)
    assert result.attempts[0].selector_confidence is None

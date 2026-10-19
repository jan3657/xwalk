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

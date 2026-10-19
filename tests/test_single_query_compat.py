"""Single-query wire bytes and identities pinned to the pre-batch-5 implementation.

See fixtures/single_query_587176e/README.md for baseline capture instructions.
"""

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from tests.test_matcher import (
    PROMPTS,
    SOURCE,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    rewrite_reply,
    score_reply,
    select_reply,
    verify_reply,
)
from xwalk.batch import run_fingerprint_components
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import MatchStatus
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

FIXTURES = Path(__file__).parent / "fixtures" / "single_query_587176e"
QUERY_CASES = {
    "legacy": (),
    "same_query": ("{{ mention }}",),
    "single_alternate": ("alternate",),
}
SCENARIOS = {
    "confident": [select_reply("C01"), score_reply(0.95)],
    "verified": [select_reply("C01"), score_reply(0.7), verify_reply("support")],
    "rewritten": [
        select_reply("C01", 0.5),
        score_reply(0.3),
        rewrite_reply("dextrose"),
        select_reply("C01", 0.95),
        score_reply(0.95),
    ],
}


async def capture_single_query(scenario, query_case):
    """Shared harness, also run with baseline src and baseline test constants."""
    bodies = []
    replies = iter(SCENARIOS[scenario])

    def handler(request):
        bodies.append(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    llm = OpenAICompatClient(
        base_url="https://llm.test/v1",
        model="m",
        api_key="test",
        profile="unknown",
        transport=httpx.MockTransport(handler),
    )
    try:
        templates = dataclasses.replace(TEMPLATES, _compiled={}, queries=QUERY_CASES[query_case])
        retriever = ScriptedRetriever({"glucose": ["T2", "T1"], "dextrose": ["T1", "T2"]})
        policy = MatchPolicy()
        selector_policy = SelectorPolicy()
        stages = {
            "selector": Selector(llm, PROMPTS, templates, policy=selector_policy),
            "scorer": Scorer(llm, PROMPTS, templates, review_floor=policy.review_floor),
            "verifier": Verifier(llm, PROMPTS, templates),
            "rewriter": QueryRewriter(llm, PROMPTS, templates),
        }
        components = run_fingerprint_components(
            templates=templates,
            prompts=PROMPTS,
            store=STORE,
            retrievers=[retriever],
            llm=llm,
            policy=policy,
            selector_policy=selector_policy,
            **stages,
        )
        result = await Matcher(
            templates=templates,
            retrievers=[retriever],
            store=STORE,
            policy=policy,
            run_fingerprint="compatibility-capture",
            **stages,
        ).match(SOURCE)
        assert result.status is MatchStatus.MATCHED
        assert result.matched_id == ("T1" if scenario == "rewritten" else "T2")
        assert retriever.queries == (
            ["glucose", "dextrose"] if scenario == "rewritten" else ["glucose"]
        )
        assert len(bodies) == len(SCENARIOS[scenario])
        return bodies, {
            "run": components,
            "stages": {name: stage.identity for name, stage in stages.items()},
        }
    finally:
        await llm.aclose()


@pytest.mark.parametrize("query_case", QUERY_CASES)
@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_single_query_matches_baseline_bytes_and_identity(scenario, query_case):
    bodies, identity = await capture_single_query(scenario, query_case)
    expected_bodies = sorted((FIXTURES / scenario).glob("*.body"))
    assert len(expected_bodies) == len(bodies)
    for body, fixture in zip(bodies, expected_bodies, strict=True):
        assert body == fixture.read_bytes(), fixture.name

    baseline = json.loads((FIXTURES / f"{query_case}.identity.json").read_text())
    # Rendered-budget v2 intentionally invalidates old run and selector identities.
    # Everything else, including template/prompt fingerprints and stage overrides,
    # must equal the baseline. Do not discard arbitrary new components or versions.
    assert "budget_version" not in baseline["run"]["selector_policy"]
    assert "budget_version" not in baseline["stages"]["selector"]
    baseline["run"]["selector_policy"]["budget_version"] = 2
    baseline["stages"]["selector"]["budget_version"] = 2
    assert identity == baseline

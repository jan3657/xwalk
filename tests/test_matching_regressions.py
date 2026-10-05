"""Package-level regressions for the 5 October 2026 audit (cases 1-6) and for
generation-parameter precedence. See docs/claude-upgrade/CONTRACTS.md sections 1, 3, 5
and 6 for the behaviour these pin."""

import dataclasses
import json

import httpx
import pytest

from tests.conftest import FIXTURES
from tests.test_matcher import (
    PROMPTS,
    SOURCE,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    build,
    rewrite_reply,
    score_reply,
    select_reply,
    verify_reply,
)
from xwalk.config import load_job
from xwalk.llm import openai_compat
from xwalk.llm.base import LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy, derive_status
from xwalk.records import Candidate, DecisionReason, MatchStatus, Record, RetrievalHit, Usage
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.keying import assign_keys
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector
from xwalk.templates import TemplateSet

ONE_ATTEMPT = MatchPolicy(max_attempts=1)


def _raw_score(token: str) -> str:
    """A scorer reply whose confidence is spelled verbatim (json.dumps cannot emit NaN)."""
    return '{"confidence_score": ' + token + ', "explanation": ""}'


# --- case 1: invalid confidence never passes an acceptance gate ------------------


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1.7", "-0.2", '"0.9"', "null"])
async def test_an_invalid_scorer_confidence_is_never_accepted(token):
    llm = FakeLLM([select_reply("C01"), _raw_score(token)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]}), policy=ONE_ATTEMPT).match(
        SOURCE
    )
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.UNRESOLVED_OUTPUT
    assert result.confidence is None
    assert result.attempts[0].primary_score is None
    assert "confidence_score" in (result.attempts[0].error or "")


async def test_a_missing_scorer_confidence_is_never_accepted():
    llm = FakeLLM([select_reply("C01"), json.dumps({"explanation": "sure"})])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]}), policy=ONE_ATTEMPT).match(
        SOURCE
    )
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.confidence is None


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1.7"])
async def test_an_invalid_selector_confidence_is_unresolved_not_scored(token):
    reply = '{"chosen_key": "C01", "confidence_score": ' + token + ', "explanation": ""}'
    llm = FakeLLM([reply])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]}), policy=ONE_ATTEMPT).match(
        SOURCE
    )
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.UNRESOLVED_OUTPUT
    assert len(llm.requests) == 1  # no scorer call on an unclassifiable selection


async def test_an_invalid_verifier_confidence_is_dropped_not_stored():
    verify = '{"decision": "support", "confidence_score": NaN, "explanation": ""}'
    llm = FakeLLM([select_reply("C01"), score_reply(0.7), verify])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]}), policy=ONE_ATTEMPT).match(
        SOURCE
    )
    assert result.attempts[0].verifier_score is None
    assert result.status is MatchStatus.MATCHED  # the decision itself was valid


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.1])
def test_derive_status_refuses_a_non_finite_or_out_of_range_stored_score(bad):
    """Defence in depth: a score that reached an Attempt by any route cannot match."""
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = build(llm, ScriptedRetriever({"glucose": ["T1"]})).match_sync(SOURCE)
    poisoned = dataclasses.replace(result.attempts[0], primary_score=bad)
    status, _, _ = derive_status([poisoned], MatchPolicy())
    assert status is not MatchStatus.MATCHED


# --- case 2 and section 5: usage accounting --------------------------------------


async def test_rewrite_usage_is_counted_and_attributed_to_an_attempt():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3),
            rewrite_reply("dextrose"),
            select_reply("C01", 0.95),
            score_reply(0.95),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T2"], "dextrose": ["T1"]})
    result = await build(llm, retriever).match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert result.usage.calls == len(llm.requests) == 5
    assert result.usage.total_tokens == 5 * 120
    assert result.usage.unknown_calls == 0
    assert result.usage == sum((a.usage for a in result.attempts), Usage.zero())
    # the rewrite ran after attempt 0, so attempt 0 carries it
    assert result.attempts[0].usage.calls == 3
    assert result.attempts[1].usage.calls == 2


async def test_failed_calls_are_counted_with_unknown_usage():
    llm = FakeLLM(
        [
            LLMRetryableError("429"),
            select_reply("C01"),
            score_reply(0.7),
            LLMRetryableError("503"),
            select_reply("C01"),
            score_reply(0.7),
            verify_reply("support"),
        ]
    )
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=3)
    ).match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert result.usage.calls == len(llm.requests) == 7
    assert result.usage.unknown_calls == 2
    assert result.usage.total_tokens == 5 * 120  # only reported tokens are summed
    assert result.usage == sum((a.usage for a in result.attempts), Usage.zero())


# --- case 3 and section 3: verifier failures stay inside the record ------------


async def test_verifier_retry_exhaustion_is_returned_not_raised():
    llm = FakeLLM([select_reply("C01"), score_reply(0.7), LLMRetryableError("429 exhausted")])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]}), policy=ONE_ATTEMPT).match(
        SOURCE
    )
    assert result.status is not MatchStatus.MATCHED
    assert result.reason is DecisionReason.PROVIDER_FAILURE
    attempt = result.attempts[0]
    assert attempt.verifier_decision == "error"
    assert attempt.primary_score == 0.7  # the score is kept
    assert "verifier" in (attempt.error or "")
    assert result.usage.calls == 3 and result.usage.unknown_calls == 1


async def test_a_verifier_failure_is_retried_like_any_provider_failure():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.7),
            LLMRetryableError("429"),
            select_reply("C01"),
            score_reply(0.7),
            verify_reply("support"),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert retriever.queries == ["glucose", "glucose"]  # same query, not a rewrite
    assert result.status is MatchStatus.MATCHED


async def test_an_unverified_in_band_score_is_at_best_needs_review():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.3),
            rewrite_reply("dextrose"),
            select_reply("C01"),
            score_reply(0.75),
            LLMRetryableError("429"),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T2"], "dextrose": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.PROVIDER_FAILURE
    assert result.matched_id == "T1"


async def test_an_audit_verifier_failure_leaves_the_decision_unchanged():
    llm = FakeLLM([select_reply("C01"), score_reply(0.99), LLMRetryableError("429")])
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1"]}),
        policy=MatchPolicy(audit_rate=1.0, max_attempts=1),
    ).match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert result.attempts[0].verifier_decision is None
    assert "audit" in (result.attempts[0].error or "")
    assert result.usage.calls == 3


@pytest.mark.parametrize("failing_call", [0, 1, 2])
async def test_a_fatal_error_in_any_gating_stage_fails_the_record(failing_call):
    script: list = [select_reply("C01"), score_reply(0.7), verify_reply("support")]
    script[failing_call] = LLMFatalError("invalid api key")
    llm = FakeLLM(script[: failing_call + 1])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert result.reason is DecisionReason.FATAL_PROVIDER_FAILURE
    assert result.matched_id is None
    assert len(result.attempts) == 1
    assert result.usage.calls == len(llm.requests)


# --- case 4: rewriter failures ---------------------------------------------------


async def test_a_fatal_rewriter_error_fails_the_record_without_raising():
    llm = FakeLLM([select_reply("C01"), score_reply(0.3), LLMFatalError("invalid api key")])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert result.reason is DecisionReason.FATAL_PROVIDER_FAILURE
    assert "rewriter" in (result.attempts[-1].error or "")
    assert result.usage.calls == 3 and result.usage.unknown_calls == 1


async def test_a_recoverable_rewriter_error_ends_the_loop_with_the_existing_status():
    llm = FakeLLM([select_reply("C01"), score_reply(0.5), LLMRetryableError("429")])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.BELOW_ACCEPT_THRESHOLD
    assert len(result.attempts) == 1
    assert "rewriter" in (result.attempts[0].error or "")
    assert result.usage.calls == 3 and result.usage.unknown_calls == 1


# --- case 5: multiline candidate evidence stays in its own block ----------------


class ParagraphTemplates(TemplateSet):
    """A caller-supplied renderer that keeps paragraph breaks in long definitions.

    The stock `TemplateSet` collapses blank lines, but `render_candidate` is an
    extension point; nothing downstream may assume its output has none.
    """

    def render_candidate(self, record: Record) -> str:
        head = super().render_candidate(record)
        return f"{head}\nDefinition: {record.fields.get('definition', '')}"


MULTILINE_TEMPLATES = ParagraphTemplates(
    query="{{ mention }}",
    context="",
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)
MULTILINE = [
    Candidate(
        record=Record(
            id="CHEBI:17234",
            fields={
                "label": "glucose",
                "definition": (
                    "An aldohexose used as a source of energy.\n\n"
                    "Note: the D-enantiomer only; L-glucose is a separate entry."
                ),
            },
        ),
        fused_score=0.9,
        evidence=(RetrievalHit("CHEBI:17234", "bm25", 1.0, 1),),
    ),
    Candidate(
        record=Record(id="CHEBI:37624", fields={"label": "L-glucose", "definition": "Mirror."}),
        fused_score=0.8,
        evidence=(RetrievalHit("CHEBI:37624", "bm25", 0.9, 2),),
    ),
]


def _sections(prompt: str) -> tuple[str, str]:
    after = prompt.split("## Proposed match", 1)[1]
    chosen, others = after.split("## Other candidates", 1)
    return chosen, others


@pytest.mark.parametrize("stage", ["score", "verify"])
async def test_a_multiline_chosen_candidate_keeps_its_evidence(stage):
    keyed = assign_keys(MULTILINE, MULTILINE_TEMPLATES)
    source = Record(id="s1", fields={"mention": "glucose"})
    if stage == "score":
        llm = FakeLLM([score_reply(0.9)])
        await Scorer(llm, PROMPTS, MULTILINE_TEMPLATES).score(source, "", keyed, "C01")
    else:
        llm = FakeLLM([verify_reply("support")])
        await Verifier(llm, PROMPTS, MULTILINE_TEMPLATES).verify(source, "", keyed, "C01")
    chosen, others = _sections(llm.requests[0].user)
    assert "L-glucose is a separate entry" in chosen
    assert "L-glucose is a separate entry" not in others
    assert "[C02]" in others and "[C02]" not in chosen


# --- case 6: the trace flag never changes a decision -----------------------------


def _strip_trace(attempt):
    return dataclasses.replace(attempt, candidates=(), elapsed_seconds=0.0)


async def test_trace_settings_do_not_change_matching():
    outcomes = {}
    for keep in (True, False):
        llm = FakeLLM(
            [
                select_reply("C01", 0.5),
                score_reply(0.3, better_candidate_keys=["C02"]),
                score_reply(0.95),
            ]
        )
        matcher = Matcher(
            templates=TEMPLATES,
            retrievers=[ScriptedRetriever({"glucose": ["T1", "T2"]})],
            store=STORE,
            selector=Selector(llm, PROMPTS, TEMPLATES),
            scorer=Scorer(llm, PROMPTS, TEMPLATES),
            verifier=Verifier(llm, PROMPTS, TEMPLATES),
            rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
            policy=MatchPolicy(max_attempts=2),
            run_fingerprint="trace",
            keep_candidates_in_trace=keep,
        )
        result = await matcher.match(SOURCE)
        outcomes[keep] = (result, llm.requests)

    (on, on_requests), (off, off_requests) = outcomes[True], outcomes[False]
    assert on.status is MatchStatus.MATCHED
    assert (off.status, off.matched_id, off.confidence, off.reason) == (
        on.status,
        on.matched_id,
        on.confidence,
        on.reason,
    )
    assert [_strip_trace(a) for a in off.attempts] == [_strip_trace(a) for a in on.attempts]
    assert off_requests == on_requests
    assert off.candidates == () and all(a.candidates == () for a in off.attempts)
    assert on.attempts[0].candidates  # the trace really was kept when asked


# --- section 6: generation parameters reach the outgoing request ----------------

_ANY_STAGE_REPLY = json.dumps(
    {
        "chosen_key": "C01",
        "confidence_score": 0.95,
        "explanation": "",
        "decision": "support",
        "queries": [],
    }
)


def _capturing_transport(bodies: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": _ANY_STAGE_REPLY}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    return httpx.MockTransport(handler)


async def test_job_generation_settings_reach_every_outgoing_request(tmp_path, monkeypatch):
    bodies: list[dict] = []
    real_client = httpx.AsyncClient

    def with_mock_transport(**kwargs):
        return real_client(timeout=kwargs.get("timeout"), transport=_capturing_transport(bodies))

    monkeypatch.setattr(openai_compat.httpx, "AsyncClient", with_mock_transport)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    job = load_job(FIXTURES / "job_tiny.yaml")
    job = job.model_copy(
        update={"llm": job.llm.model_copy(update={"temperature": 0.7, "max_tokens": 2048})}
    )
    targets = list(job.build_target_records())
    retrievers = job.build_retrievers(targets, job.build_templates(), tmp_path)
    llm = job.build_llm()
    matcher = job.build_matcher(store=job.build_store(), retrievers=retrievers, llm=llm)
    result = await matcher.match(next(iter(job.build_source_records())))

    assert result.status is MatchStatus.MATCHED
    assert len(bodies) == 2
    assert all(b["temperature"] == 0.7 for b in bodies)
    assert all(b["max_tokens"] == 2048 for b in bodies)


async def test_an_explicit_stage_argument_overrides_the_client():
    bodies: list[dict] = []
    llm = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        temperature=0.7,
        max_tokens=2048,
        transport=_capturing_transport(bodies),
    )
    keyed = assign_keys(MULTILINE, MULTILINE_TEMPLATES)
    source = Record(id="s1", fields={"mention": "glucose"})
    await Scorer(llm, PROMPTS, MULTILINE_TEMPLATES, max_tokens=256, temperature=0.1).score(
        source, "", keyed, "C01"
    )
    await Verifier(llm, PROMPTS, MULTILINE_TEMPLATES).verify(source, "", keyed, "C01")
    assert (bodies[0]["temperature"], bodies[0]["max_tokens"]) == (0.1, 256)
    assert (bodies[1]["temperature"], bodies[1]["max_tokens"]) == (0.7, 2048)


async def test_an_unset_request_uses_the_client_defaults():
    bodies: list[dict] = []
    llm = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        transport=_capturing_transport(bodies),
    )
    await llm.complete(LLMRequest(system="", user="u"))
    assert (bodies[0]["temperature"], bodies[0]["max_tokens"]) == (0.0, 1024)

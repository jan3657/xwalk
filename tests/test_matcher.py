import asyncio
import json

from xwalk.llm.base import LLMFatalError, LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import DecisionReason, MatchStatus, Record, RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context=(
        "{% if context_left %}{{ context_left }} [{{ mention }}] {{ context_right }}{% endif %}"
    ),
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)
PROMPTS = PromptSet.from_slots(
    PromptSlots(
        entity_noun="mention",
        target_noun="term",
        domain_brief="test",
        rubric=[
            {"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.4, "name": "Weak", "when": "vague"},
        ],
    )
)
STORE = MemoryStore.from_source(
    [
        Record(id="T1", fields={"label": "glucose"}),
        Record(id="T2", fields={"label": "fructose"}),
        Record(id="T3", fields={"label": "sucrose"}),
    ]
)
SOURCE = Record(
    id="s1", fields={"mention": "glucose", "context_left": "blood", "context_right": "levels"}
)


class ScriptedRetriever:
    """Returns a fixed hit list per query text; unknown queries return nothing."""

    def __init__(self, by_query, *, name="bm25", fail=False, hang=False, default_limit=20):
        self._by_query = by_query
        self._name = name
        self._fail = fail
        self._hang = hang
        self._default_limit = default_limit
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return f"scripted-{self._name}"

    @property
    def default_limit(self) -> int:
        return self._default_limit

    async def search(self, request: SearchRequest):
        self.queries.append(request.text)
        if self._fail:
            raise RetrieverError("index unavailable")
        if self._hang:
            await asyncio.sleep(3600)
        ids = self._by_query.get(request.text, [])
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(ids, start=1)
        ]


def select_reply(key, score=0.9, explanation="ok"):
    return json.dumps({"chosen_key": key, "confidence_score": score, "explanation": explanation})


def score_reply(score, **extra):
    return json.dumps({"confidence_score": score, "explanation": "", **extra})


def verify_reply(decision="support", **extra):
    return json.dumps({"decision": decision, "explanation": "", **extra})


def rewrite_reply(*queries):
    return json.dumps({"queries": list(queries), "explanation": ""})


def build(llm, retriever, *, policy=None):
    policy = policy or MatchPolicy()
    return Matcher(
        templates=TEMPLATES,
        retrievers=[retriever],
        store=STORE,
        selector=Selector(
            llm, PROMPTS, TEMPLATES, legacy_id_resolution=policy.legacy_id_resolution
        ),
        scorer=Scorer(llm, PROMPTS, TEMPLATES, review_floor=policy.review_floor),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=policy,
        run_fingerprint="fp1",
    )


# --- happy path -----------------------------------------------------------------


async def test_a_confident_match_returns_immediately():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    retriever = ScriptedRetriever({"glucose": ["T1", "T2"]})
    result = await build(llm, retriever).match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert result.matched_id == "T1"
    assert result.confidence == 0.95
    assert len(result.attempts) == 1


async def test_the_matched_record_is_attached():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.matched_record is not None
    assert result.matched_record.fields["label"] == "glucose"


async def test_the_first_query_comes_from_the_query_template():
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose"]


async def test_the_context_reaches_the_selector():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert "blood [glucose] levels" in llm.requests[0].user


async def test_usage_sums_across_stages():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.usage.calls == 2


async def test_the_result_key_is_stable_for_the_same_record_and_run():
    def make():
        return build(
            FakeLLM([select_reply("C01"), score_reply(0.95)]),
            ScriptedRetriever({"glucose": ["T1"]}),
        )

    first = await make().match(SOURCE)
    second = await make().match(SOURCE)
    assert first.result_key == second.result_key


async def test_the_result_key_changes_when_a_field_value_changes():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    a = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    llm2 = FakeLLM([select_reply("C01"), score_reply(0.95)])
    changed = Record(id="s1", fields={**SOURCE.fields, "context_right": "was low"})
    b = await build(llm2, ScriptedRetriever({"glucose": ["T1"]})).match(changed)
    assert a.result_key != b.result_key


# --- retry paths ----------------------------------------------------------------


async def test_a_low_score_triggers_a_rewrite_and_a_second_attempt():
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
    assert retriever.queries == ["glucose", "dextrose"]
    assert result.matched_id == "T1"
    assert len(result.attempts) == 2


async def test_a_scorer_query_proposal_is_used_before_the_rewriter_is_called():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3, better_queries=["dextrose"]),
            select_reply("C01", 0.95),
            score_reply(0.95),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T2"], "dextrose": ["T1"]})
    result = await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]
    assert result.status is MatchStatus.MATCHED


async def test_a_candidate_proposal_is_rescored_without_new_retrieval():
    """The proposed candidate is already in hand; re-running retrieval would be waste."""
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3, better_candidate_keys=["C02"]),
            score_reply(0.95),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1", "T2"]})
    result = await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose"]  # exactly one retrieval
    assert result.matched_id == "T2"
    assert result.status is MatchStatus.MATCHED


async def test_a_hallucinated_candidate_proposal_is_dropped_not_searched():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3, better_candidate_keys=["C99"]),
            rewrite_reply("dextrose"),
            select_reply("C01", 0.9),
            score_reply(0.9),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "dextrose": ["T2"]})
    await build(llm, retriever).match(SOURCE)
    assert "C99" not in retriever.queries


async def test_the_loop_stops_at_max_attempts():
    llm = FakeLLM(
        [select_reply("C01", 0.3), score_reply(0.3), rewrite_reply("q2")] * 2
        + [select_reply("C01", 0.3), score_reply(0.3)]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "q2": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert len(result.attempts) == 2


async def test_a_query_is_never_searched_twice():
    llm = FakeLLM(
        [
            select_reply("C01", 0.3),
            score_reply(0.3, better_queries=["glucose"]),  # the query already tried
            rewrite_reply("dextrose"),
            select_reply("C01", 0.9),
            score_reply(0.9),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "dextrose": ["T2"]})
    await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]


async def test_the_best_attempt_wins_on_exhaustion():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.5),
            rewrite_reply("q2"),
            select_reply("C01", 0.2),
            score_reply(0.2),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "q2": ["T2"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert result.confidence == 0.5
    assert result.matched_id == "T1"


# --- abstention and no candidates -----------------------------------------------


async def test_no_candidates_anywhere_is_unmatched_with_that_reason():
    llm = FakeLLM([rewrite_reply()])
    retriever = ScriptedRetriever({})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=1)).match(SOURCE)
    assert result.status is MatchStatus.UNMATCHED
    assert result.reason is DecisionReason.NO_CANDIDATES


async def test_an_explicit_abstention_is_unmatched_with_its_own_reason():
    llm = FakeLLM([select_reply(None), rewrite_reply()])
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=1)).match(SOURCE)
    assert result.reason is DecisionReason.SELECTOR_ABSTAINED


async def test_the_scorer_is_not_called_after_an_abstention():
    """With max_attempts=1 the loop stops after the single attempt, so the only call is
    the selector's. What this pins is that no scoring prompt was ever sent."""
    llm = FakeLLM([select_reply(None)])
    await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    assert len(llm.requests) == 1
    assert all("## Rubric" not in r.user for r in llm.requests)


async def test_unresolvable_output_routes_to_review():
    llm = FakeLLM([select_reply("T1"), rewrite_reply()])  # an id, not a key
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.UNRESOLVED_OUTPUT


# --- verification ---------------------------------------------------------------


async def test_a_score_in_the_verify_band_triggers_verification():
    llm = FakeLLM([select_reply("C01"), score_reply(0.7), verify_reply("support")])
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(verify_band=(0.6, 0.8))
    ).match(SOURCE)
    assert result.attempts[0].verifier_decision == "support"
    assert result.status is MatchStatus.MATCHED


async def test_a_score_above_the_band_is_not_verified():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.attempts[0].verifier_decision is None


async def test_verifier_disagreement_forces_review():
    llm = FakeLLM(
        [select_reply("C01"), score_reply(0.7), verify_reply("disagree", preferred_key="C02")]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(verify_band=(0.6, 0.8), max_attempts=1),
    ).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.VERIFIER_DISAGREEMENT


async def test_the_verifiers_preferred_key_is_recorded_but_not_chased():
    """Re-entering selection on the verifier's preference would make termination depend
    on two models negotiating. Bounded loop, honest flag, human decides."""
    llm = FakeLLM(
        [select_reply("C01"), score_reply(0.7), verify_reply("disagree", preferred_key="C02")]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(verify_band=(0.6, 0.8), max_attempts=1),
    ).match(SOURCE)
    assert result.attempts[0].verifier_preferred_id == "T2"
    assert result.matched_id == "T1"  # unchanged


async def test_an_audit_verdict_is_honoured_not_merely_counted():
    llm = FakeLLM([select_reply("C01"), score_reply(0.99), verify_reply("no_match")])
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1"]}),
        policy=MatchPolicy(audit_rate=1.0, verify_band=(0.6, 0.8), max_attempts=1),
    ).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.attempts[0].audited is True


async def test_audit_is_off_by_default():
    llm = FakeLLM([select_reply("C01"), score_reply(0.99)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.attempts[0].audited is False


# --- error handling -------------------------------------------------------------


async def test_one_retriever_failing_degrades_to_the_others():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    bad = ScriptedRetriever({}, name="dense", fail=True)
    matcher = Matcher(
        templates=TEMPLATES,
        retrievers=[good, bad],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(),
        run_fingerprint="fp1",
    )
    result = await matcher.match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert "dense" in (result.attempts[0].error or "")


async def test_a_retriever_timeout_degrades_rather_than_failing_the_match():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    slow = ScriptedRetriever({}, name="slow", hang=True)
    matcher = Matcher(
        templates=TEMPLATES,
        retrievers=[good, slow],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(retriever_timeout=0.05),
        run_fingerprint="fp1",
    )
    result = await matcher.match(SOURCE)
    assert result.status is MatchStatus.MATCHED


async def test_every_retriever_failing_is_a_retriever_failure_not_a_non_match():
    llm = FakeLLM([])
    bad = ScriptedRetriever({}, name="bm25", fail=True)
    matcher = Matcher(
        templates=TEMPLATES,
        retrievers=[bad],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(max_attempts=1),
        run_fingerprint="fp1",
    )
    result = await matcher.match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert result.reason is DecisionReason.RETRIEVER_FAILURE


async def test_a_provider_failure_is_failed_not_unmatched():
    llm = FakeLLM([LLMRetryableError("429 exhausted")])
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert result.reason is DecisionReason.PROVIDER_FAILURE


async def test_a_fatal_provider_error_stops_the_loop_immediately():
    llm = FakeLLM([LLMFatalError("invalid api key")])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert len(result.attempts) == 1  # no point retrying a bad key


async def test_one_failed_attempt_does_not_sink_a_later_good_one():
    llm = FakeLLM([LLMRetryableError("429"), select_reply("C01", 0.95), score_reply(0.95)])
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert result.status is MatchStatus.MATCHED


# --- sync facade ----------------------------------------------------------------


def test_match_sync_works_outside_an_event_loop():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = build(llm, ScriptedRetriever({"glucose": ["T1"]})).match_sync(SOURCE)
    assert result.status is MatchStatus.MATCHED

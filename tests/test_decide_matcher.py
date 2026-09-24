import pytest

from xwalk.decide.base import DecisionFatalError, DecisionRetryableError
from xwalk.decide.fake import FakeDecider
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import DecisionReason, MatchStatus, Record, RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.stages.choose import Chooser
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.screen import Screener
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ note }}",
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
    queries=("{{ mention }}", "{{ alias }}"),
)
SLOTS = PromptSlots(
    entity_noun="mention",
    target_noun="term",
    domain_brief="test",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact"},
        {"score": 0.4, "name": "Weak", "when": "vague"},
    ],
    properties=[{"name": "form", "question": "Same form?"}],
)
QUESTIONS = QuestionSet.from_slots(SLOTS)
STORE = MemoryStore.from_source(
    [
        Record(id="T1", fields={"label": "glucose"}),
        Record(id="T2", fields={"label": "fructose"}),
        Record(id="T3", fields={"label": "sucrose"}),
    ]
)
SOURCE = Record(id="s1", fields={"mention": "glucose", "alias": "dextrose", "note": ""})


class Scripted:
    def __init__(self, by_query, *, name="bm25", fail=False):
        self._by_query, self._name, self._fail = by_query, name, fail
        self.queries: list[str] = []

    @property
    def name(self):
        return self._name

    @property
    def fingerprint(self):
        return "s"

    @property
    def default_limit(self):
        return 10

    async def search(self, request: SearchRequest):
        self.queries.append(request.text)
        if self._fail:
            raise RetrieverError("index unavailable")
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(self._by_query.get(request.text, []), start=1)
        ]


def confident(state, questions):
    """Says yes to any candidate whose text contains "glucose", no to the rest."""
    from xwalk.decide.base import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

    answers = {}
    candidates = state.get("candidates", {})
    for name, q in questions.items():
        if isinstance(q, Noul):
            key = name[len("n_") :] if name.startswith("n_") else None
            text = (
                candidates.get(key, state.get("candidate", ""))
                if key
                else state.get("candidate", "")
            )
            answers[name] = NoulAnswer(noul=0.95 if "glucose" in str(text) else 0.05)
        elif isinstance(q, Choice):
            hits = [k for k in q.criteria if "glucose" in str(candidates.get(k, ""))]
            choice = hits[0] if hits else "NONE"
            answers[name] = ChoiceAnswer(
                choice=choice,
                probabilities={
                    k: (0.9 if k == choice else 0.1 / max(1, len(q.criteria) - 1))
                    for k in q.criteria
                },
                confidence=0.85,
            )
        elif isinstance(q, Score):
            top = len(q.criteria) - 1
            answers[name] = ScoreAnswer(
                score=float(top),
                probabilities={str(i): (1.0 if i == top else 0.0) for i in range(top + 1)},
                confidence=0.9,
                legend={str(i): str(c) for i, c in enumerate(q.criteria)},
            )
    return answers


def _matcher(decider=None, retrievers=None, policy=None, rewriter=None):
    decider = decider or FakeDecider(handler=confident)
    return DecisionMatcher(
        templates=TEMPLATES,
        retrievers=retrievers or [Scripted({"glucose": ["T1", "T2"], "dextrose": ["T3"]})],
        store=STORE,
        screener=Screener(decider, QUESTIONS, TEMPLATES, chunk_size=2),
        chooser=Chooser(decider, QUESTIONS, TEMPLATES),
        gate=PropertyGate(decider, QUESTIONS, TEMPLATES),
        policy=policy
        or DecisionPolicy(
            accept_at=0.5,
            screen_floor=0.1,
            shortlist_floor=0.0,
            property_floor=0.0,
            rubric_floor=0.0,
        ),
        run_fingerprint="fp",
        rewriter=rewriter,
    )


async def test_the_confident_case_matches_with_signals_and_a_numeric_explanation():
    decider = FakeDecider(handler=confident)
    retriever = Scripted({"glucose": ["T1", "T2"], "dextrose": ["T3"]})
    result = await _matcher(decider, [retriever]).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]
    assert result.status is MatchStatus.MATCHED
    assert result.matched_id == "T1"
    assert result.reason is DecisionReason.ACCEPT_THRESHOLD
    assert result.confidence == result.signals["screen_chosen"]
    assert "prop_form" in result.signals
    assert result.explanation.startswith("same ")
    attempt = result.attempts[0]
    assert attempt.chosen_id == "T1" and attempt.resolution == "exact_key"
    # Fusion order, not retrieval order: T3 is rank 1 for the second query, so it ties
    # T1 on RRF score and beats T2 (rank 2, one query) whatever the screen then says.
    assert attempt.issued_keys == {"C001": "T1", "C002": "T3", "C003": "T2"}
    assert attempt.primary_score == result.confidence
    assert '"choose"' in (attempt.raw_selection or "") and '"gate"' in (attempt.raw_selection or "")
    # 2 screen chunks + choose + gate
    assert result.usage.calls == 4 and len(decider.calls) == 4
    assert len(result.candidates) == 3
    # shortlist screen probabilities are kept by key so recall can be measured later
    assert result.signals["screen_C001"] == 0.95


async def test_nothing_similar_is_unmatched_without_choose_or_gate():
    decider = FakeDecider(handler=confident)
    retriever = Scripted({"unobtainium": ["T2", "T3"]})
    result = await _matcher(decider, [retriever]).match(
        Record(id="s", fields={"mention": "unobtainium", "alias": "", "note": ""})
    )
    assert result.status is MatchStatus.UNMATCHED
    assert result.reason is DecisionReason.BELOW_REVIEW_FLOOR
    assert result.matched_id is None
    assert len(decider.calls) == 1  # one screen chunk of two candidates; no choose, no gate


async def test_no_candidates():
    result = await _matcher(retrievers=[Scripted({})]).match(SOURCE)
    assert result.status is MatchStatus.UNMATCHED and result.reason is DecisionReason.NO_CANDIDATES


async def test_every_retriever_failing_is_failed_not_unmatched():
    result = await _matcher(retrievers=[Scripted({}, fail=True)]).match(SOURCE)
    assert result.status is MatchStatus.FAILED and result.reason is DecisionReason.RETRIEVER_FAILURE


async def test_a_retryable_provider_error_is_a_failed_result():
    decider = FakeDecider(handler=lambda s, q: DecisionRetryableError("exhausted"))
    # a handler that raises is still a FakeDecider; the matcher maps the error
    result = await _matcher(decider).match(SOURCE)
    assert result.status is MatchStatus.FAILED and result.reason is DecisionReason.PROVIDER_FAILURE
    assert "exhausted" in (result.attempts[0].error or "")


async def test_a_failed_screen_chunk_still_bills_for_its_siblings():
    """One chunk of two failed; the other was paid for, and the result must say so."""

    def second_chunk_fails(state, questions):
        if "n_C003" in questions:
            raise DecisionRetryableError("busy")
        return confident(state, questions)

    decider = FakeDecider(handler=second_chunk_fails)
    result = await _matcher(decider).match(SOURCE)
    assert result.status is MatchStatus.FAILED and result.reason is DecisionReason.PROVIDER_FAILURE
    assert result.usage.calls == 1
    assert "busy" in (result.attempts[0].error or "")


async def test_a_fatal_screen_chunk_error_propagates():
    def second_chunk_is_fatal(state, questions):
        if "n_C003" in questions:
            raise DecisionFatalError("bad key")
        return confident(state, questions)

    with pytest.raises(DecisionFatalError):
        await _matcher(FakeDecider(handler=second_chunk_is_fatal)).match(SOURCE)


async def test_a_fatal_provider_error_propagates():
    decider = FakeDecider(handler=lambda s, q: DecisionFatalError("bad key"))
    with pytest.raises(DecisionFatalError):
        await _matcher(decider).match(SOURCE)


async def test_max_candidates_caps_what_the_screen_sees():
    decider = FakeDecider(handler=confident)
    policy = DecisionPolicy(
        max_candidates=2,
        accept_at=0.5,
        screen_floor=0.1,
        shortlist_floor=0.0,
        property_floor=0.0,
        rubric_floor=0.0,
    )
    result = await _matcher(decider, policy=policy).match(SOURCE)
    assert result.attempts[0].candidate_count == 2
    assert result.attempts[0].candidates_truncated == 1


def test_match_sync_runs_the_loop():
    result = _matcher().match_sync(SOURCE)
    assert result.status is MatchStatus.MATCHED


# --- conditional query rewrite (A3) -------------------------------------------------

TWO_QUERIES = '{"queries": ["glucose", "dextrose"], "explanation": "the formal names"}'
# "blood sugar" retrieves only fructose, so the first screen misses; the rewrite's
# queries find glucose (T1) and sucrose (T3).
MISS = Record(id="s2", fields={"mention": "blood sugar", "alias": "bs", "note": ""})
MISS_INDEX = {"blood sugar": ["T2"], "glucose": ["T1", "T2"], "dextrose": ["T3"]}


def _rewriter(llm):
    from xwalk.prompts.contract import PromptSet
    from xwalk.stages.rewrite import QueryRewriter

    return QueryRewriter(llm, PromptSet.from_slots(SLOTS), TEMPLATES, max_queries=3)


def _notes(result):
    return result.attempts[0].error or ""


async def test_no_rewrite_when_the_first_screen_clears_the_floor():
    from xwalk.llm.fake import FakeLLM

    llm = FakeLLM([TWO_QUERIES])
    plain = await _matcher(
        retrievers=[Scripted({"glucose": ["T1", "T2"], "dextrose": ["T3"]})]
    ).match(SOURCE)
    retriever = Scripted({"glucose": ["T1", "T2"], "dextrose": ["T3"]})
    result = await _matcher(retrievers=[retriever], rewriter=_rewriter(llm)).match(SOURCE)
    assert llm.requests == []
    assert retriever.queries == ["glucose", "dextrose"]
    assert "rewrite" not in _notes(result)
    assert result.attempts[0].query == "glucose"
    assert (result.status, result.matched_id, result.usage) == (
        plain.status,
        plain.matched_id,
        plain.usage,
    )


async def test_a_miss_below_the_floor_is_rewritten_once_and_matches():
    from xwalk.llm.fake import FakeLLM

    llm = FakeLLM([TWO_QUERIES])
    retriever = Scripted(MISS_INDEX)
    result = await _matcher(retrievers=[retriever], rewriter=_rewriter(llm)).match(MISS)
    assert len(llm.requests) == 1
    # the rewriter sees the queries already tried, never the screened candidates
    assert "blood sugar" in llm.requests[0].user
    assert "fructose" not in llm.requests[0].user
    assert retriever.queries == ["blood sugar", "bs", "glucose", "dextrose"]
    assert result.status is MatchStatus.MATCHED and result.matched_id == "T1"
    assert "rewrite: 2 queries proposed, 2 new candidates" in _notes(result)
    assert {c.id for c in result.candidates} == {"T1", "T2", "T3"}


async def test_no_candidates_triggers_the_rewrite():
    from xwalk.llm.fake import FakeLLM

    llm = FakeLLM([TWO_QUERIES])
    retriever = Scripted({"glucose": ["T1"]})
    result = await _matcher(retrievers=[retriever], rewriter=_rewriter(llm)).match(MISS)
    assert len(llm.requests) == 1
    assert result.matched_id == "T1"
    assert "rewrite: 2 queries proposed, 1 new candidates" in _notes(result)


async def test_the_rewrite_runs_at_most_once_per_record():
    from xwalk.llm.fake import FakeLLM

    llm = FakeLLM([TWO_QUERIES])  # a second call would exhaust the script
    retriever = Scripted({"blood sugar": ["T2"], "glucose": ["T2"], "dextrose": ["T3"]})
    result = await _matcher(retrievers=[retriever], rewriter=_rewriter(llm)).match(MISS)
    assert len(llm.requests) == 1
    assert retriever.queries == ["blood sugar", "bs", "glucose", "dextrose"]
    assert result.status is MatchStatus.UNMATCHED
    assert "failed" not in _notes(result)


async def test_the_second_screen_is_merged_keeping_the_higher_probability():
    from xwalk.decide.base import Noul, NoulAnswer
    from xwalk.llm.fake import FakeLLM

    phase = {"n": 1}
    # all above the screener's shortlist floor (0.2) so every merged value is traced,
    # all of the first screen below this policy's screen floor (0.5)
    first = {"fructose": 0.3, "sucrose": 0.25}
    second = {"fructose": 0.22, "sucrose": 0.4, "glucose": 0.95}
    policy = DecisionPolicy(
        accept_at=0.5, screen_floor=0.5, shortlist_floor=0.0, property_floor=0.0, rubric_floor=0.0
    )

    def phased(state, questions):
        answers = confident(state, questions)
        table = first if phase["n"] == 1 else second
        for name, q in questions.items():
            if isinstance(q, Noul) and name.startswith("n_"):
                text = state["candidates"][name[2:]]
                answers[name] = NoulAnswer(noul=next(p for k, p in table.items() if k in text))
        return answers

    def rewrite(request):
        phase["n"] = 2
        return TWO_QUERIES

    llm = FakeLLM(handler=rewrite)
    retriever = Scripted({"blood sugar": ["T2", "T3"], "glucose": ["T1", "T2"], "dextrose": ["T3"]})
    result = await _matcher(
        FakeDecider(handler=phased), [retriever], policy=policy, rewriter=_rewriter(llm)
    ).match(MISS)
    issued = result.attempts[0].issued_keys
    merged = {rid: result.signals[f"screen_{key}"] for key, rid in issued.items()}
    assert merged == {"T1": 0.95, "T2": 0.3, "T3": 0.4}
    assert sorted(issued.values()) == ["T1", "T2", "T3"]  # one key per record
    assert result.matched_id == "T1"
    assert result.signals["screen_best"] == 0.95
    assert "rewrite: 2 queries proposed, 1 new candidates" in _notes(result)


async def test_the_attempt_lists_every_query_tried():
    from xwalk.llm.fake import FakeLLM

    result = await _matcher(
        retrievers=[Scripted(MISS_INDEX)], rewriter=_rewriter(FakeLLM([TWO_QUERIES]))
    ).match(MISS)
    assert result.attempts[0].query == "blood sugar | bs | glucose | dextrose"


async def test_usage_sums_both_screens_and_the_rewrite_call():
    from xwalk.llm.fake import FakeLLM
    from xwalk.records import Usage

    decider = FakeDecider(handler=confident)
    llm = FakeLLM([TWO_QUERIES], prompt_tokens=100, completion_tokens=20)
    result = await _matcher(decider, [Scripted(MISS_INDEX)], rewriter=_rewriter(llm)).match(MISS)
    # screen 1: one chunk (T2); screen 2: two chunks (T1, T2 | T3); choose; gate
    assert len(decider.calls) == 5
    decided = sum(
        (Usage(prompt_tokens=decider._prompt_tokens, calls=1) for _ in decider.calls),
        Usage.zero(),
    )
    assert result.usage == decided + Usage(prompt_tokens=100, completion_tokens=20, calls=1)
    assert result.attempts[0].usage == result.usage


async def test_a_rewrite_error_is_a_note_and_the_first_screen_stands():
    from xwalk.llm.fake import FakeLLM

    decider = FakeDecider(handler=confident)
    retriever = Scripted(MISS_INDEX)
    llm = FakeLLM([RuntimeError("rate limited")])
    result = await _matcher(decider, [retriever], rewriter=_rewriter(llm)).match(MISS)
    assert len(llm.requests) == 1
    assert retriever.queries == ["blood sugar", "bs"]
    assert result.status is MatchStatus.UNMATCHED
    assert result.reason is DecisionReason.BELOW_REVIEW_FLOOR
    assert [c.id for c in result.candidates] == ["T2"]
    assert "rewrite: failed (RuntimeError)" in _notes(result)
    assert result.attempts[0].query == "blood sugar"
    assert len(decider.calls) == 1


async def test_a_rewrite_with_nothing_new_is_skipped_with_a_note():
    from xwalk.llm.fake import FakeLLM

    retriever = Scripted(MISS_INDEX)
    llm = FakeLLM(['{"queries": ["Blood Sugar", "bs"], "explanation": "same"}'])
    result = await _matcher(retrievers=[retriever], rewriter=_rewriter(llm)).match(MISS)
    assert retriever.queries == ["blood sugar", "bs"]
    assert "rewrite: skipped (no new queries)" in _notes(result)
    assert result.status is MatchStatus.UNMATCHED

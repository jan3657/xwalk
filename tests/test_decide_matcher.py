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
QUESTIONS = QuestionSet.from_slots(
    PromptSlots(
        entity_noun="mention",
        target_noun="term",
        domain_brief="test",
        rubric=[
            {"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.4, "name": "Weak", "when": "vague"},
        ],
        properties=[{"name": "form", "question": "Same form?"}],
    )
)
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


def _matcher(decider=None, retrievers=None, policy=None):
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

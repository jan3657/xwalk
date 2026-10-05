from xwalk.decide.base import ChoiceAnswer
from xwalk.decide.fake import FakeDecider
from xwalk.decide.questions import NONE_KEY, QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.choose import Chooser
from xwalk.stages.keying import Resolution
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="",
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
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
    )
)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


def _cands(*labels):
    return [
        Candidate(
            record=Record(id=f"T{i}", fields={"label": label}),
            fused_score=1.0,
            evidence=(
                RetrievalHit(record_id=f"T{i}", retriever="bm25", raw_score=1.0, rank=i + 1),
            ),
        )
        for i, label in enumerate(labels)
    ]


async def test_chooses_by_key_and_resolves_to_the_record():
    fake = FakeDecider()
    outcome = await Chooser(fake, QUESTIONS, TEMPLATES).choose(
        SOURCE, "", _cands("fructose", "glucose")
    )
    assert outcome.record_id == "T1"
    assert outcome.resolution is Resolution.EXACT_KEY
    assert outcome.p_choice > 0.5 and outcome.p_none < 0.5
    state, questions = fake.calls[0]
    assert list(state["candidates"]) == ["C01", "C02"]
    assert list(questions) == ["best"]
    assert list(questions["best"].criteria) == ["C01", "C02", NONE_KEY]


async def test_none_is_an_abstention():
    fake = FakeDecider()
    outcome = await Chooser(fake, QUESTIONS, TEMPLATES).choose(
        Record(id="s", fields={"mention": "unobtainium"}), "", _cands("glucose")
    )
    assert outcome.record_id is None
    assert outcome.resolution is Resolution.ABSTAIN
    assert outcome.p_none == outcome.p_choice


async def test_an_unissued_key_is_unresolved_not_a_record():
    def handler(state, questions):
        return {"best": ChoiceAnswer(choice="C99", probabilities={"C99": 1.0}, confidence=1.0)}

    outcome = await Chooser(FakeDecider(handler=handler), QUESTIONS, TEMPLATES).choose(
        SOURCE, "", _cands("glucose")
    )
    assert outcome.record_id is None
    assert outcome.resolution is Resolution.UNRESOLVED


async def test_empty_shortlist_abstains_without_a_call():
    fake = FakeDecider()
    outcome = await Chooser(fake, QUESTIONS, TEMPLATES).choose(SOURCE, "", [])
    assert outcome.resolution is Resolution.ABSTAIN and fake.calls == []

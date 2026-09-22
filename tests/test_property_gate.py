from xwalk.decide.fake import FakeDecider
from xwalk.decide.questions import QuestionSet
from xwalk.prompts.contract import PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.property_gate import PropertyGate
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="",
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)
SLOTS = PromptSlots(
    entity_noun="mention",
    target_noun="term",
    domain_brief="test",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact"},
        {"score": 0.7, "name": "High", "when": "close"},
        {"score": 0.4, "name": "Weak", "when": "vague"},
    ],
    properties=[{"name": "processing_state", "question": "Same processing state?"}],
)
CHOSEN = Candidate(
    record=Record(id="T1", fields={"label": "glucose"}),
    fused_score=1.0,
    evidence=(RetrievalHit(record_id="T1", retriever="bm25", raw_score=1.0, rank=1),),
)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


async def test_asks_the_rubric_and_every_property_in_one_call():
    fake = FakeDecider()
    outcome = await PropertyGate(fake, QuestionSet.from_slots(SLOTS), TEMPLATES).gate(
        SOURCE, "", CHOSEN
    )
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert state == {
        "source": {"fields": {"mention": "glucose"}},
        "candidate": "ID: T1 Label: glucose",
    }
    assert set(questions) == {"rubric", "prop_processing_state"}
    assert outcome.rubric_levels == 3
    assert 0.0 <= outcome.rubric_score <= 2.0
    assert list(outcome.properties) == ["processing_state"]
    assert outcome.usage.calls == 1
    assert "rubric" in outcome.raw


async def test_no_properties_means_only_the_rubric():
    fake = FakeDecider()
    slots = SLOTS.model_copy(update={"properties": []})
    outcome = await PropertyGate(fake, QuestionSet.from_slots(slots), TEMPLATES).gate(
        SOURCE, "", CHOSEN
    )
    assert set(fake.calls[0][1]) == {"rubric"}
    assert outcome.properties == {}

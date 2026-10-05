import json
from pathlib import Path

import pytest

from tests.test_matcher import STORE, TEMPLATES, ScriptedRetriever, score_reply, select_reply
from xwalk.evaluate.failures import PromptRole
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.partition import Partition, Partitioner
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, PromptSlots, PropertyQuestion
from xwalk.prompts.optimize import OptimizeConfig, estimate_calls, optimize_prompt
from xwalk.records import Record
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector

SLOTS = PromptSlots(
    entity_noun="mention",
    target_noun="term",
    domain_brief="v0",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact"},
        {"score": 0.4, "name": "Weak", "when": "vague"},
    ],
)
SOURCES = [Record(id=f"s{i:02d}", fields={"mention": "glucose"}) for i in range(30)]
GOLD = GoldSet({f"s{i:02d}": frozenset({"T1"}) for i in range(30)})
RETRIEVER_MAP = {"glucose": ["T1", "T2"]}


def matcher_llm(accuracy):
    """Answers correctly for the first `accuracy` share of each block of ten records.

    Deterministic and independent of how many matchers get built. Keying correctness on
    factory-call order (as the plan originally did) makes a "successful" prompt-train run
    produce no failures, so the optimiser exits before round one ever happens.
    """
    state = {"n": 0}

    def handler(request):
        if "## Candidates" in request.user:
            index = state["n"]
            state["n"] += 1
            return select_reply("C01" if (index % 10) < round(accuracy * 10) else "C02")
        return score_reply(0.95)

    return FakeLLM(handler=handler)


def make_factory(accuracy_by_brief, default=0.5):
    """Correctness is a property of the prompt under test, not of call ordering."""

    def factory(prompts: PromptSet) -> Matcher:
        llm = matcher_llm(accuracy_by_brief.get(prompts.slots.domain_brief, default))
        return Matcher(
            templates=TEMPLATES,
            retrievers=[ScriptedRetriever(RETRIEVER_MAP)],
            store=STORE,
            selector=Selector(llm, prompts, TEMPLATES),
            scorer=Scorer(llm, prompts, TEMPLATES),
            verifier=Verifier(llm, prompts, TEMPLATES),
            rewriter=QueryRewriter(llm, prompts, TEMPLATES),
            policy=MatchPolicy(max_attempts=1),
            run_fingerprint=prompts.slots.domain_brief,
        )

    return factory


def optimiser_llm(*briefs):
    replies = [
        json.dumps(
            {
                "entity_noun": "mention",
                "target_noun": "term",
                "domain_brief": brief,
                "rubric": [
                    {"score": 1.0, "name": "Certain", "when": "exact"},
                    {"score": 0.4, "name": "Weak", "when": "vague"},
                ],
                "hard_rules": [],
                "disambiguation_steps": "",
            }
        )
        for brief in briefs
    ]
    return FakeLLM(replies)


# --- partition discipline ---------------------------------------------------------


async def test_the_test_partition_is_evaluated_exactly_once(tmp_path):
    """The single most important property of this module."""
    seen: list[str] = []

    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0, "v2": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("v1", "v2"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=2),
        work_dir=tmp_path,
        progress=seen.append,
    )
    assert sum(1 for line in seen if "test partition" in line.lower()) == 1
    assert report.test_report is not None


async def test_test_scores_are_absent_from_every_round_result(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    for round_result in report.rounds:
        assert not hasattr(round_result, "test")


async def test_only_prompt_train_failures_reach_the_optimising_model(tmp_path):
    llm = optimiser_llm("v1")
    partitioner = Partitioner(fractions=(0.34, 0.33, 0.33), salt="fixed")
    test_ids = {s.id for s in SOURCES if partitioner.assign(s.id) is Partition.TEST}
    await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=llm,
        partitioner=partitioner,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    prompt = llm.requests[0].user
    assert test_ids and not any(tid in prompt for tid in test_ids)


async def test_partition_overrides_are_honoured(tmp_path):
    overrides = {s.id: Partition.PROMPT_TRAIN for s in SOURCES[:10]}
    overrides.update({s.id: Partition.VALIDATION for s in SOURCES[10:20]})
    overrides.update({s.id: Partition.TEST for s in SOURCES[20:]})
    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        partition_overrides=overrides,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert report.test_report.labelled == 10


# --- round selection --------------------------------------------------------------


async def test_an_improving_round_is_retained(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"better": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("better"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert report.best_slots.domain_brief == "better"
    assert report.rounds[0].improved is True


async def test_properties_survive_an_optimiser_that_omits_them(tmp_path):
    """The optimiser never sees the decider path, so it drops `properties`. Carry it."""
    with_properties = SLOTS.model_copy(
        update={"properties": [PropertyQuestion(name="form", question="Same form?")]}
    )
    report = await optimize_prompt(
        matcher_factory=make_factory({"better": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=with_properties,
        optimiser_llm=optimiser_llm("better"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert report.best_slots.domain_brief == "better"
    assert [p.name for p in report.best_slots.properties] == ["form"]


async def test_a_regressing_round_is_discarded(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"worse": 0.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("worse"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert report.best_slots.domain_brief == "v0"
    assert report.rounds[0].improved is False


async def test_the_baseline_is_measured_before_any_round(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({}, default=1.0),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert report.baseline.accepted_precision == 1.0


async def test_optimisation_stops_early_after_patience_rounds_without_improvement(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("a", "b", "c", "d"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, patience=2),
        work_dir=tmp_path,
    )
    assert len(report.rounds) == 2
    assert "patience" in report.stopped_because


async def test_optimisation_stops_when_there_are_no_failures_to_learn_from(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({}, default=1.0),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("unused"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4),
        work_dir=tmp_path,
    )
    assert report.rounds == ()
    assert "no failures" in report.stopped_because.lower()


# --- safety -----------------------------------------------------------------------


async def test_a_round_producing_invalid_slots_is_skipped_not_fatal(tmp_path):
    llm = FakeLLM(
        [
            "not json at all",
            json.dumps(
                {
                    "entity_noun": "mention",
                    "target_noun": "term",
                    "domain_brief": "recovered",
                    "rubric": [
                        {"score": 1.0, "name": "C", "when": "x"},
                        {"score": 0.4, "name": "W", "when": "y"},
                    ],
                    "hard_rules": [],
                    "disambiguation_steps": "",
                }
            ),
        ]
    )
    report = await optimize_prompt(
        matcher_factory=make_factory({"recovered": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=llm,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=2),
        work_dir=tmp_path,
    )
    assert any(r.warnings for r in report.rounds)
    assert report.best_slots.domain_brief == "recovered"


async def test_the_call_budget_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="max_calls"):
        await optimize_prompt(
            matcher_factory=make_factory({}),
            source_records=SOURCES,
            gold=GOLD,
            initial=SLOTS,
            optimiser_llm=optimiser_llm("v1"),
            config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, max_calls=5),
            work_dir=tmp_path,
        )


def test_estimate_calls_scales_with_rounds_and_records():
    small = estimate_calls(OptimizeConfig(rounds=1), 10, 10, 10)
    large = estimate_calls(OptimizeConfig(rounds=4), 10, 10, 10)
    assert large > small


def test_estimate_calls_counts_the_test_partition_once():
    """Not once per round -- that would be both wasteful and methodologically wrong.

    Compared as the test partition's *contribution*: the per-round optimiser call means
    the totals themselves differ with `rounds`, so equating them would be simply false.
    """
    one = OptimizeConfig(rounds=1)
    four = OptimizeConfig(rounds=4)
    contribution_one = estimate_calls(one, 0, 0, 100) - estimate_calls(one, 0, 0, 0)
    contribution_four = estimate_calls(four, 0, 0, 100) - estimate_calls(four, 0, 0, 0)
    assert contribution_one == contribution_four == 200


async def test_every_round_is_persisted_for_inspection(tmp_path):
    await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert (Path(tmp_path) / "round_01" / "slots.yaml").exists()
    assert (Path(tmp_path) / "best" / "slots.yaml").exists()
    assert (Path(tmp_path) / "report.json").exists()


async def test_the_final_report_is_json_serialisable(tmp_path):
    report = await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=optimiser_llm("v1"),
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    json.dumps(report.as_dict())


async def test_the_source_fields_of_a_failing_record_reach_the_model(tmp_path):
    """A failure case without its source record is a bare id the model cannot act on."""
    llm = optimiser_llm("v1")
    await optimize_prompt(
        matcher_factory=make_factory({"v1": 1.0}),
        source_records=SOURCES,
        gold=GOLD,
        initial=SLOTS,
        optimiser_llm=llm,
        config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=1),
        work_dir=tmp_path,
    )
    assert "mention: glucose" in llm.requests[0].user

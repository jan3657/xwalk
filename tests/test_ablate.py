from tests.test_matcher import (
    PROMPTS,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    score_reply,
    select_reply,
)
from xwalk.evaluate.ablate import MatcherConfig, ablate, standard_ablations
from xwalk.evaluate.gold import GoldSet
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import Record
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

SOURCES = [Record(id=f"s{i}", fields={"mention": "glucose"}) for i in range(6)]
GOLD = GoldSet({f"s{i}": frozenset({"T1"}) for i in range(6)})


def factory(config: MatcherConfig) -> Matcher:
    def handler(request):
        if "## Candidates" in request.user:
            return select_reply("C01")
        return score_reply(0.95)

    llm = FakeLLM(handler=handler)
    retrievers = [
        ScriptedRetriever({"glucose": ["T1", "T2"]}, name=name) for name in config.retriever_names
    ] or [ScriptedRetriever({}, name="none")]
    return Matcher(
        templates=TEMPLATES,
        retrievers=retrievers,
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES, policy=config.selector_policy),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=config.policy,
        run_fingerprint=config.name,
    )


BASE = MatcherConfig(
    name="baseline",
    retriever_names=("bm25", "dense"),
    policy=MatchPolicy(),
    selector_policy=SelectorPolicy(),
)


def test_standard_ablations_cover_every_retriever():
    names = {a.name for a in standard_ablations(("bm25", "dense"))}
    assert "no_bm25" in names and "no_dense" in names


def test_standard_ablations_cover_verifier_retries_and_budget():
    names = {a.name for a in standard_ablations(("bm25",))}
    assert {"no_verifier", "no_retries", "half_budget"} <= names


def test_dropping_a_retriever_removes_it_from_the_config():
    ablation = next(a for a in standard_ablations(("bm25", "dense")) if a.name == "no_dense")
    assert ablation.apply(BASE).retriever_names == ("bm25",)


def test_each_retriever_ablation_drops_its_own_retriever():
    """A closure that captured the loop variable would drop the last one every time."""
    for name in ("bm25", "dense"):
        ablation = next(a for a in standard_ablations(("bm25", "dense")) if a.name == f"no_{name}")
        assert name not in ablation.apply(BASE).retriever_names


def test_disabling_the_verifier_clears_the_band():
    ablation = next(a for a in standard_ablations(("bm25",)) if a.name == "no_verifier")
    assert ablation.apply(BASE).policy.verify_band is None


def test_disabling_retries_sets_one_attempt():
    ablation = next(a for a in standard_ablations(("bm25",)) if a.name == "no_retries")
    assert ablation.apply(BASE).policy.max_attempts == 1


def test_halving_the_budget_halves_max_candidates():
    ablation = next(a for a in standard_ablations(("bm25",)) if a.name == "half_budget")
    assert ablation.apply(BASE).selector_policy.max_candidates == 15


def test_ablations_never_mutate_the_base_config():
    for ablation in standard_ablations(("bm25", "dense")):
        ablation.apply(BASE)
    assert BASE.retriever_names == ("bm25", "dense")
    assert BASE.policy.max_attempts == 4


async def test_ablate_reports_a_row_per_ablation_plus_the_baseline():
    report = await ablate(factory, BASE, SOURCES, GOLD)
    names = [row.name for row in report.rows]
    assert names[0] == "baseline"
    assert len(names) == 1 + len(standard_ablations(BASE.retriever_names))


async def test_ablate_reports_a_delta_against_the_baseline():
    report = await ablate(factory, BASE, SOURCES, GOLD)
    assert report.rows[0].delta == 0.0


async def test_dropping_every_retriever_shows_a_negative_delta():
    from xwalk.evaluate.ablate import Ablation

    kill = Ablation(
        name="no_retrieval",
        description="drop all retrievers",
        apply=lambda c: MatcherConfig(**{**c.__dict__, "retriever_names": ()}),
    )
    report = await ablate(factory, BASE, SOURCES, GOLD, ablations=[kill])
    assert report.rows[1].delta < 0


async def test_ablate_writes_a_report_when_given_a_path(tmp_path):
    import json

    await ablate(factory, BASE, SOURCES, GOLD, out=tmp_path / "ablation.json")
    json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))


async def test_ablate_rows_carry_cost_so_a_cheaper_ablation_is_visible():
    report = await ablate(factory, BASE, SOURCES, GOLD)
    assert all(row.mean_llm_calls is not None for row in report.rows)

from pathlib import Path

import pytest

from xwalk.config import load_job
from xwalk.prompts.contract import PromptSet, load_slots, validate_contract

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(p for p in (ROOT / "examples").iterdir() if (p / "job.yaml").exists())


def _skip_if_loader_missing(job):
    if job.target.kind == "owl":
        pytest.importorskip("rdflib")


def test_there_are_examples():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_job_file_loads(example):
    assert load_job(example / "job.yaml").name


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_templates_compile(example):
    load_job(example / "job.yaml").build_templates()


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_slots_satisfy_the_prompt_contract(example):
    job = load_job(example / "job.yaml")
    validate_contract(PromptSet.from_slots(load_slots(job.base_dir / job.prompts.slots)))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_sample_target_records_load(example):
    job = load_job(example / "job.yaml")
    _skip_if_loader_missing(job)
    records = list(job.build_target_records())
    assert records and all(r.id for r in records)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_sample_source_records_load(example):
    assert list(load_job(example / "job.yaml").build_source_records())


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_doc_template_renders_non_empty_text_for_every_target(example):
    """An empty rendered doc means that record can never be retrieved."""
    job = load_job(example / "job.yaml")
    _skip_if_loader_missing(job)
    templates = job.build_templates()
    empty = [r.id for r in job.build_target_records() if not templates.render_doc(r)]
    assert not empty, f"{len(empty)} target records render an empty doc: {empty[:5]}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_query_template_renders_non_empty_text_for_every_source(example):
    job = load_job(example / "job.yaml")
    templates = job.build_templates()
    empty = [r.id for r in job.build_source_records() if not templates.render_query(r)]
    assert not empty, f"{len(empty)} source records render an empty query: {empty[:5]}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_bm25_indexes_the_sample_and_finds_something(example, tmp_path):
    import asyncio

    from xwalk.retrieval.base import SearchRequest
    from xwalk.retrieval.bm25 import BM25Retriever

    job = load_job(example / "job.yaml")
    _skip_if_loader_missing(job)
    templates = job.build_templates()
    targets = list(job.build_target_records())
    retriever = BM25Retriever.build(
        targets, templates, tmp_path / "idx", exact_fields=("label", "synonyms")
    )
    # Not "the first mention retrieves something": a lexical miss is a legitimate,
    # interesting outcome (CRAFT's "cholinergic" shares no token with any ChEBI label in
    # the slice, which is exactly the never_retrieved case the ceiling report exists to
    # surface). What must hold is that the index is functional at all.
    found = 0
    for source in list(job.build_source_records())[:10]:
        hits = asyncio.run(
            retriever.search(SearchRequest(text=templates.render_query(source), limit=5))
        )
        found += bool(hits)
    assert found, "no query out of the first ten retrieved anything -- the index is broken"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_gold_file_labels_every_sample_mention(example):
    """A sample whose gold file has drifted from its mentions measures nothing."""
    from xwalk.evaluate import load_gold_csv

    job = load_job(example / "job.yaml")
    gold_path = example / "sample" / "gold.csv"
    if not gold_path.exists():
        pytest.skip("this example ships no gold file")
    gold = load_gold_csv(gold_path)
    unlabelled = [r.id for r in job.build_source_records() if r.id not in gold]
    assert not unlabelled, f"{len(unlabelled)} mentions have no gold row: {unlabelled[:5]}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_no_example_file_exceeds_one_megabyte(example):
    large = [p for p in example.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000]
    assert not large, f"files over 1 MB: {large}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_no_job_file_contains_an_inline_api_key(example):
    assert "api_key:" not in (example / "job.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_example_has_a_readme(example):
    assert (example / "README.md").exists()


def test_the_nlm_gene_example_uses_numeric_target_ids():
    """The case opaque keys exist for. If this stops holding, the example lost its point."""
    job = load_job(ROOT / "examples/nlm_gene/job.yaml")
    ids = [r.id for r in job.build_target_records()]
    assert any(i.split(":")[-1].isdigit() for i in ids)


def test_the_ncbi_disease_expander_recovers_an_aliased_gold_id():
    """The replacement for the paper repo's expand_ctd_eval_ids hook, exercised."""
    from examples.ncbi_disease.gold_normalize import make_expander, normalize

    assert normalize("D003924") == "MESH:D003924"
    expand = make_expander(str(ROOT / "examples/ncbi_disease/sample/targets.tsv"))
    assert expand(frozenset({"MESH:D003924"})) >= frozenset({"MESH:D003924"})

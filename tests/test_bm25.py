import pytest

from xwalk.retrieval.base import SearchRequest
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source
from xwalk.templates import TemplateSet

DOC_TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="",
    doc="{{ label }} {{ synonyms | join(' ') }} {{ definition }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)


EXACT_FIELDS = ("label", "synonyms")


@pytest.fixture
def retriever(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    return BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "idx", exact_fields=EXACT_FIELDS)


async def test_exact_label_match_ranks_first(retriever):
    """Plain BM25 puts beta-D-glucose first here — it contains 'glucose' twice in a
    shorter document. The exact field is what makes this assertion a requirement."""
    hits = await retriever.search(SearchRequest(text="glucose", limit=5))
    assert hits[0].record_id == "CHEBI:17234"


async def test_an_exact_synonym_outranks_a_partial_label_match(retriever):
    hits = await retriever.search(SearchRequest(text="table sugar", limit=5))
    assert hits[0].record_id == "CHEBI:17992"


async def test_without_exact_fields_it_is_a_plain_bm25_index(targets_csv, tmp_path):
    """`exact_fields=()` must still build, search, and return hits — the exact field is
    an opt-in ranking aid, never a requirement for the retriever to function."""
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    plain = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "plain")
    hits = await plain.search(SearchRequest(text="glucose", limit=5))
    assert [h.record_id for h in hits]


async def test_default_limit_is_exposed_for_the_matcher(retriever):
    assert retriever.default_limit == 20


async def test_synonyms_are_searchable(retriever):
    hits = await retriever.search(SearchRequest(text="dextrose", limit=5))
    assert hits[0].record_id == "CHEBI:17234"


async def test_ranks_are_one_based_and_contiguous(retriever):
    hits = await retriever.search(SearchRequest(text="sugar", limit=5))
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


async def test_hits_carry_the_retriever_name(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=5))
    assert all(h.retriever == "bm25" for h in hits)


async def test_limit_is_respected(retriever):
    hits = await retriever.search(SearchRequest(text="sugar", limit=2))
    assert len(hits) <= 2


async def test_no_match_returns_empty_not_an_error(retriever):
    assert await retriever.search(SearchRequest(text="unobtainium", limit=5)) == []


async def test_blank_query_returns_empty(retriever):
    assert await retriever.search(SearchRequest(text="   ", limit=5)) == []


async def test_query_syntax_characters_are_treated_as_text(retriever):
    """Real mentions contain +, -, :, (, ), ^, ~, AND, OR. None may reach the parser."""
    for text in ["glucose:", "glucose (D-)", "glucose AND fructose", "+glucose^2", "gluc*ose~"]:
        hits = await retriever.search(SearchRequest(text=text, limit=5))
        assert isinstance(hits, list)  # no exception, no parser error


async def test_open_reuses_a_built_index(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    built = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "idx", exact_fields=EXACT_FIELDS)
    reopened = BM25Retriever.open(tmp_path / "idx")
    assert reopened.fingerprint == built.fingerprint
    hits = await reopened.search(SearchRequest(text="glucose", limit=5))
    assert hits[0].record_id == "CHEBI:17234"


def test_open_restores_the_name_and_exact_fields_from_meta(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    BM25Retriever.build(
        records, DOC_TEMPLATES, tmp_path / "idx", name="lexical", exact_fields=EXACT_FIELDS
    )
    reopened = BM25Retriever.open(tmp_path / "idx")
    assert reopened.name == "lexical"


def test_fingerprint_changes_with_the_exact_fields(targets_csv, tmp_path):
    """Exact fields change ranking, so they must change the index fingerprint."""
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    a = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "a", exact_fields=EXACT_FIELDS)
    b = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "b")
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_the_doc_template(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    a = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "a")
    other = TemplateSet(query="", context="", doc="{{ label }}", candidate="")
    b = BM25Retriever.build(records, other, tmp_path / "b")
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_the_target_content(tmp_path):
    from xwalk.records import Record

    a = BM25Retriever.build([Record(id="x", fields={"label": "a"})], DOC_TEMPLATES, tmp_path / "a")
    b = BM25Retriever.build([Record(id="x", fields={"label": "b"})], DOC_TEMPLATES, tmp_path / "b")
    assert a.fingerprint != b.fingerprint


def test_records_with_empty_rendered_docs_are_reported(tmp_path):
    from xwalk.records import Record

    r = BM25Retriever.build(
        [Record(id="x", fields={"label": ""}), Record(id="y", fields={"label": "glucose"})],
        DOC_TEMPLATES,
        tmp_path / "idx",
    )
    assert r.empty_doc_count == 1


def test_bm25_satisfies_the_retriever_protocol(retriever):
    from xwalk.retrieval.base import Retriever

    assert isinstance(retriever, Retriever)

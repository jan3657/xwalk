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


async def test_rebuilding_over_an_existing_index_replaces_it(targets_csv, tmp_path):
    """`xwalk match` rebuilds into `<out>/index` on every invocation, so appending
    instead of replacing would duplicate every document on the second run. The
    fingerprint hashes records, not the index, so nothing downstream would notice."""
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    for _ in range(3):
        retriever = BM25Retriever.build(
            records, DOC_TEMPLATES, tmp_path / "idx", exact_fields=EXACT_FIELDS
        )

    hits = await retriever.search(SearchRequest(text="glucose", limit=20))
    assert len({h.record_id for h in hits}) == len(hits), "the same record was returned twice"


async def test_a_rebuild_does_not_spend_the_retrieval_limit_on_duplicates(
    targets_eval_csv, tmp_path
):
    """The duplicate hit itself is harmless -- fusion collapses it. The harm is that
    duplicates consume the `limit`, so fewer distinct records reach the model and recall
    drops with nothing in the run reporting it."""
    records = list(csv_source(targets_eval_csv, id_column="id", multivalue_columns=["synonyms"]))
    request = SearchRequest(text="sugar", limit=10)

    first = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "idx")
    before = {h.record_id for h in await first.search(request)}

    second = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "idx")
    after = {h.record_id for h in await second.search(request)}

    assert after == before


def test_bm25_satisfies_the_retriever_protocol(retriever):
    from xwalk.retrieval.base import Retriever

    assert isinstance(retriever, Retriever)


SPELLINGS = [
    ("A1", "anesthetic"),
    ("A2", "anaesthetic agents"),
    ("G1", "glucose"),
]
LABEL_ONLY = TemplateSet(query="{{ mention }}", context="", doc="{{ label }}", candidate="")


def _spellings_index(path, **options):
    from xwalk.records import Record

    records = [Record(id=rid, fields={"label": label}) for rid, label in SPELLINGS]
    return BM25Retriever.build(records, LABEL_ONLY, path, **options)


async def _ids(retriever, text):
    return {h.record_id for h in await retriever.search(SearchRequest(text=text, limit=10))}


async def test_the_default_analyzer_does_not_stem(tmp_path):
    assert await _ids(_spellings_index(tmp_path / "idx"), "anesthetics") == set()


async def test_the_en_stem_analyzer_matches_an_inflected_query(tmp_path):
    retriever = _spellings_index(tmp_path / "idx", analyzer="en_stem")
    assert await _ids(retriever, "anesthetics") == {"A1"}


async def test_fuzzy_terms_match_both_spellings(tmp_path):
    retriever = _spellings_index(tmp_path / "idx", fuzzy_distance=1)
    assert await _ids(retriever, "anaesthetic") == {"A1", "A2"}


def test_fingerprint_changes_with_the_analyzer_and_fuzzy_distance(tmp_path):
    fingerprints = {
        _spellings_index(tmp_path / "a").fingerprint,
        _spellings_index(tmp_path / "b", analyzer="en_stem").fingerprint,
        _spellings_index(tmp_path / "c", fuzzy_distance=1).fingerprint,
    }
    assert len(fingerprints) == 3


async def test_open_restores_the_analyzer_and_fuzzy_distance(tmp_path):
    """Both on: "anesthetics" stems to "anesthet", and its fuzzy clause must be stemmed
    too to reach "anaesthet" -- the raw word is three edits away from either stem."""
    _spellings_index(tmp_path / "idx", analyzer="en_stem", fuzzy_distance=1)
    reopened = BM25Retriever.open(tmp_path / "idx")
    assert await _ids(reopened, "anesthetics") == {"A1", "A2"}
    assert await _ids(reopened, "glucose") == {"G1"}


def test_opening_with_a_different_analyzer_is_refused(tmp_path):
    _spellings_index(tmp_path / "idx", analyzer="en_stem")
    with pytest.raises(ValueError, match="analyzer 'en_stem'.*'default'"):
        BM25Retriever.open(tmp_path / "idx", analyzer="default")


def test_opening_with_a_different_fuzzy_distance_is_refused(tmp_path):
    _spellings_index(tmp_path / "idx")
    with pytest.raises(ValueError, match="fuzzy_distance 0.*1"):
        BM25Retriever.open(tmp_path / "idx", fuzzy_distance=1)


async def test_an_index_without_the_new_metadata_opens_with_the_defaults(tmp_path):
    """Indexes built before the analyzer option existed carry no `analyzer` or
    `fuzzy_distance` key; they are default-tokenised, non-fuzzy indexes."""
    import json

    built = _spellings_index(tmp_path / "idx")
    meta_path = tmp_path / "idx" / "xwalk_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["analyzer"], meta["fuzzy_distance"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    reopened = BM25Retriever.open(tmp_path / "idx", analyzer="default", fuzzy_distance=0)
    assert reopened.fingerprint == built.fingerprint
    assert await _ids(reopened, "anesthetic") == {"A1"}


def test_the_default_fingerprint_is_unchanged_by_the_new_options(tmp_path):
    """Pinned: defaults must not re-key existing indexes or cached runs."""
    from xwalk.fingerprint import hash_record, hash_value
    from xwalk.records import Record

    records = [Record(id=rid, fields={"label": label}) for rid, label in SPELLINGS]
    expected = hash_value(
        {
            "engine": "tantivy-bm25",
            "doc_template": LABEL_ONLY.doc,
            "exact_fields": [],
            "records": sorted(hash_record(r) for r in records),
        }
    )
    assert _spellings_index(tmp_path / "idx").fingerprint == expected

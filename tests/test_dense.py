import asyncio
import math

import pytest

from xwalk.records import Record
from xwalk.retrieval.base import Retriever, SearchRequest
from xwalk.retrieval.dense import DenseRetriever
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="", doc="{{ label }}", candidate="ID: {{ id }}"
)

RECORDS = [
    Record(id="T1", fields={"label": "glucose"}),
    Record(id="T2", fields={"label": "fructose"}),
    Record(id="T3", fields={"label": "automobile"}),
]


class ToyEncoder:
    """Deterministic, dependency-free: a character-histogram embedding.

    Enough structure for 'glucose' to sit closer to 'fructose' than to 'automobile',
    which is all the retriever's contract requires. Tests must not depend on a
    downloaded model.
    """

    name = "toy"
    dimension = 26

    def encode(self, texts, *, is_query: bool = False):
        vectors = []
        for text in texts:
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@pytest.fixture
def retriever(tmp_path):
    return DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())


async def test_an_exact_match_ranks_first(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=3))
    assert hits[0].record_id == "T1"


async def test_ranks_are_one_based_and_contiguous(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=3))
    assert [h.rank for h in hits] == [1, 2, 3]


async def test_a_semantically_closer_record_outranks_a_distant_one(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=3))
    order = [h.record_id for h in hits]
    assert order.index("T2") < order.index("T3")


async def test_hits_carry_the_encoder_derived_name(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=1))
    assert hits[0].retriever == "dense:toy"


async def test_limit_is_respected(retriever):
    assert len(await retriever.search(SearchRequest(text="glucose", limit=2))) == 2


async def test_a_blank_query_returns_nothing(retriever):
    assert await retriever.search(SearchRequest(text="   ", limit=3)) == []


async def test_scores_are_reported(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=1))
    assert hits[0].raw_score is not None


async def test_open_reuses_a_built_index(tmp_path):
    built = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())
    reopened = DenseRetriever.open(tmp_path / "idx", ToyEncoder())
    assert reopened.fingerprint == built.fingerprint
    hits = await reopened.search(SearchRequest(text="glucose", limit=1))
    assert hits[0].record_id == "T1"


def test_opening_a_non_index_directory_is_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="xwalk"):
        DenseRetriever.open(tmp_path / "empty", ToyEncoder())


def test_opening_with_a_different_encoder_is_refused(tmp_path):
    """A vector index built by one encoder is meaningless to another."""
    DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())

    class OtherEncoder(ToyEncoder):
        name = "other"

    with pytest.raises(ValueError, match="encoder"):
        DenseRetriever.open(tmp_path / "idx", OtherEncoder())


def test_a_dimension_mismatch_is_refused(tmp_path):
    class BadEncoder(ToyEncoder):
        dimension = 12

    with pytest.raises(ValueError, match="dimension"):
        DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", BadEncoder())


def test_fingerprint_changes_with_the_encoder(tmp_path):
    class OtherEncoder(ToyEncoder):
        name = "other"

    a = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "a", ToyEncoder())
    b = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "b", OtherEncoder())
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_the_records(tmp_path):
    a = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "a", ToyEncoder())
    b = DenseRetriever.build(RECORDS[:2], TEMPLATES, tmp_path / "b", ToyEncoder())
    assert a.fingerprint != b.fingerprint


def test_dense_satisfies_the_retriever_protocol(retriever):
    assert isinstance(retriever, Retriever)


def test_dense_declares_its_own_retrieval_depth(tmp_path):
    """`default_limit` is part of the Retriever protocol and the matcher reads it on
    every attempt. A dense retriever without one breaks the loop at runtime."""
    built = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), default_limit=7
    )
    assert built.default_limit == 7
    assert DenseRetriever.open(tmp_path / "idx", ToyEncoder()).default_limit == 7


def test_two_dense_retrievers_can_coexist_with_distinct_names(tmp_path):
    """This is what replaces the paper repo's hardcoded 'second dense model'."""

    class OtherEncoder(ToyEncoder):
        name = "domain"

    a = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "a", ToyEncoder())
    b = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "b", OtherEncoder())
    assert a.name != b.name


def test_a_query_prefix_is_applied_only_to_queries(tmp_path):
    seen = []

    class RecordingEncoder(ToyEncoder):
        def encode(self, texts, *, is_query: bool = False):
            seen.append((is_query, list(texts)))
            return super().encode(texts, is_query=is_query)

    retriever = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", RecordingEncoder())
    asyncio.run(retriever.search(SearchRequest(text="glucose", limit=1)))
    assert seen[0][0] is False  # documents
    assert seen[-1][0] is True  # query


async def test_an_empty_record_set_builds_and_searches_without_crashing(tmp_path):
    """An empty target collection is a user error worth surviving, not a ZeroDivision."""
    built = DenseRetriever.build([], TEMPLATES, tmp_path / "idx", ToyEncoder())
    assert await built.search(SearchRequest(text="glucose", limit=3)) == []


@pytest.mark.dense
def test_the_sentence_transformer_encoder_loads():
    pytest.importorskip("sentence_transformers")
    from xwalk.retrieval.dense import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = encoder.encode(["glucose"], is_query=True)
    assert len(vectors) == 1 and len(vectors[0]) == encoder.dimension


def test_the_module_imports_without_the_extra():
    """Importing must work; only *using* SentenceTransformerEncoder demands torch."""
    import xwalk.retrieval.dense as module

    assert hasattr(module, "DenseRetriever")

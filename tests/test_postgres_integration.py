"""`xwalk.retrieval.postgres` against a real Neon branch over HTTPS.

Runs only with `-m integration` and `DATABASE_URL` set (skipped otherwise). It creates a
throwaway schema, loads five records with a 26-d letter-histogram encoder (no embedding
calls), searches both retrievers and the store, and drops the schema again.
"""

from __future__ import annotations

import asyncio
import math
import os
import secrets
from collections.abc import Iterator, Sequence

import pytest

from xwalk.records import Record
from xwalk.retrieval.base import IndexMismatchError, SearchRequest
from xwalk.retrieval.postgres import (
    NeonHTTPExecutor,
    PgVectorRetriever,
    PostgresBM25Retriever,
    PostgresIndexWriter,
    PostgresStore,
    bm25_components,
    pgvector_components,
    read_index_meta,
    terms_components,
)
from xwalk.retrieve import retrieve
from xwalk.templates import TemplateSet

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL is not set"),
]

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ mention }}",
    doc="{{ label }} {{ synonyms | join(' ') }} {{ definition }}",
    candidate="{{ id }} {{ label }}",
)
RECORDS = [
    Record("G:1", {"label": "ADA", "synonyms": ["ADA1"], "definition": "adenosine deaminase"}),
    Record("G:2", {"label": "ADK", "synonyms": [], "definition": "adenosine kinase"}),
    Record("G:3", {"label": "IL-2", "synonyms": ["interleukin 2"], "definition": "cytokine"}),
    Record("G:4", {"label": "glucose", "synonyms": ["dextrose"], "definition": "a sugar"}),
    Record("G:5", {"label": "beta-D-glucose", "synonyms": [], "definition": "glucose anomer"}),
]
EXACT = ("label", "synonyms")
NAMES = ("label", "synonyms")


class Histogram:
    name = "histogram"
    dimension = 26

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture(scope="module")
def loaded() -> Iterator[tuple[NeonHTTPExecutor, str]]:
    executor = NeonHTTPExecutor.from_env()
    schema = f"xwalk_it_{secrets.token_hex(4)}"
    writer = PostgresIndexWriter(executor, schema=schema, dimension=26)
    encoder = Histogram()
    try:
        writer.create_schema()
        ont = "genes"
        writer.write_meta(ont, "terms", terms_components(RECORDS), name="terms", status="loading")
        writer.upsert_terms(ont, RECORDS, TEMPLATES, exact_fields=EXACT)
        rows = writer.vector_rows(RECORDS, TEMPLATES, name_fields=NAMES)
        assert writer.present_vectors(ont, rows) == set()
        writer.upsert_vectors(ont, rows, encoder.encode([r.text for r in rows]))
        assert len(writer.present_vectors(ont, rows)) == len(rows)
        writer.create_indexes(ont)
        writer.analyze()
        writer.write_meta(
            ont, "terms", terms_components(RECORDS), name="terms", status="ready", record_count=5
        )
        writer.write_meta(
            ont,
            "bm25",
            bm25_components(RECORDS, TEMPLATES, exact_fields=EXACT),
            name="bm25",
            status="ready",
            record_count=5,
        )
        writer.write_meta(
            ont,
            "dense",
            pgvector_components(RECORDS, TEMPLATES, encoder, name_fields=NAMES),
            name="dense",
            status="ready",
            record_count=5,
            vector_count=len(rows),
            dimension=26,
        )
        yield executor, schema
    finally:
        writer.drop_schema()
        executor.close()


def test_bm25_exact_match_comes_first_and_text_matches_follow(
    loaded: tuple[NeonHTTPExecutor, str],
) -> None:
    executor, schema = loaded
    expected = bm25_components(RECORDS, TEMPLATES, exact_fields=EXACT)
    retriever = PostgresBM25Retriever.open(executor, "genes", schema=schema, expected=expected)
    hits = asyncio.run(retriever.search(SearchRequest(text="Glucose", limit=5)))
    assert [h.record_id for h in hits][:2] == ["G:4", "G:5"]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    ada = asyncio.run(retriever.search(SearchRequest(text="ada1", limit=5)))
    assert ada[0].record_id == "G:1"
    il2 = asyncio.run(retriever.search(SearchRequest(text="IL-2", limit=5)))
    assert il2[0].record_id == "G:3"
    assert asyncio.run(retriever.search(SearchRequest(text="(+/-)", limit=5))) == []


def test_dense_returns_each_record_once(loaded: tuple[NeonHTTPExecutor, str]) -> None:
    executor, schema = loaded
    encoder = Histogram()
    expected = pgvector_components(RECORDS, TEMPLATES, encoder, name_fields=NAMES)
    retriever = PgVectorRetriever.open(executor, "genes", encoder, schema=schema, expected=expected)
    hits = asyncio.run(retriever.search(SearchRequest(text="dextrose", limit=5)))
    ids = [h.record_id for h in hits]
    assert ids[0] == "G:4"
    assert len(ids) == len(set(ids)) == 5
    assert hits[0].raw_score == pytest.approx(1.0, abs=1e-3)


def test_store_and_fusion_end_to_end(loaded: tuple[NeonHTTPExecutor, str]) -> None:
    executor, schema = loaded
    meta = read_index_meta(executor, schema)
    store = PostgresStore.open(executor, "genes", schema=schema, meta=meta)
    assert [r.id for r in store.get_many(["G:2", "nope", "G:1"])] == ["G:2", "G:1"]
    assert store.get("G:3") == RECORDS[2]
    retrievers = [
        PostgresBM25Retriever.open(executor, "genes", schema=schema, meta=meta, store=store),
        PgVectorRetriever.open(executor, "genes", Histogram(), schema=schema, meta=meta),
    ]
    candidates, notes, failed = asyncio.run(
        retrieve(["ADA"], Record("q", {"mention": "ADA"}), retrievers, store, timeout=30)
    )
    assert not notes and not failed
    assert candidates[0].id == "G:1"


def test_a_mismatched_expectation_is_refused(loaded: tuple[NeonHTTPExecutor, str]) -> None:
    executor, schema = loaded
    with pytest.raises(IndexMismatchError):
        PostgresBM25Retriever.open(
            executor,
            "genes",
            schema=schema,
            expected=bm25_components(RECORDS[:4], TEMPLATES, exact_fields=EXACT),
        )

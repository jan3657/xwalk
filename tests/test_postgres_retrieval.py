"""Postgres-backed retrieval (`xwalk.retrieval.postgres`), offline.

A fake executor stands in for the database: it records every statement and answers by
the tag comment each statement starts with, so these tests pin the SQL shape and the
parameters without a server. `NeonHTTPExecutor` is tested through `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record
from xwalk.retrieval.base import IndexMismatchError, Retriever, SearchRequest
from xwalk.retrieval.bm25 import exact_form
from xwalk.retrieval.postgres import (
    IndexNotReadyError,
    NeonHTTPExecutor,
    PgVectorRetriever,
    PostgresBM25Retriever,
    PostgresError,
    PostgresIndexWriter,
    PostgresStore,
    bm25_components,
    pgvector_components,
    read_index_meta,
    tag_of,
    terms_components,
    vector_literal,
)
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ mention }}",
    doc="{{ label }} {{ synonyms | join(' ') }}",
    candidate="{{ id }} {{ label }}",
)
RECORDS = [
    Record("G:1", {"label": "ADA", "synonyms": ["ADA1"]}),
    Record("G:2", {"label": "adenosine kinase", "synonyms": ["ADK"]}),
    Record("G:3", {"label": "ADAR", "synonyms": []}),
]
FAKE_URL = "postgresql://owner:npg_SECRETpassword123@ep-test-pooler.c-1.eu.aws.neon.tech/db?sslmode=require"


class Toy:
    """3-d encoder; counts calls."""

    name = "toy"
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.calls.append((list(texts), is_query))
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeExecutor:
    """Answers each statement by its tag; records (sql, params) and transactions."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.answers: dict[str, Any] = dict(answers or {})
        self.calls: list[tuple[str, list[Any]]] = []
        self.transactions: list[tuple[list[tuple[str, list[Any]]], bool]] = []

    def _answer(self, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        self.calls.append((sql, list(params)))
        answer = self.answers.get(tag_of(sql) or "", [])
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return list(answer(sql, list(params)))
        return list(answer)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return self._answer(sql, params)

    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]:
        self.transactions.append(([(s, list(p)) for s, p in statements], read_only))
        return [self._answer(s, p) for s, p in statements]

    def tagged(self, tag: str) -> list[tuple[str, list[Any]]]:
        return [(s, p) for s, p in self.calls if tag_of(s) == tag]


def _meta_row(kind: str, components: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "ontology": "genes",
        "kind": kind,
        "name": extra.pop("name", kind),
        "status": extra.pop("status", "ready"),
        "components": components,
        "fingerprint": hash_value(components),
        "default_limit": extra.pop("default_limit", 20),
        "record_count": len(RECORDS),
        "vector_count": extra.pop("vector_count", None),
        "dimension": extra.pop("dimension", None),
        **extra,
    }


def _bm25_expected() -> dict[str, Any]:
    return bm25_components(RECORDS, TEMPLATES, exact_fields=("label", "synonyms"))


def _dense_expected(encoder: Toy) -> dict[str, Any]:
    return pgvector_components(RECORDS, TEMPLATES, encoder, name_fields=("label", "synonyms"))


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- components --------------------------------------------------------------------


def test_bm25_components_name_the_engine_records_template_and_exact_fields() -> None:
    components = _bm25_expected()
    assert components["engine"] == "postgres-lakebase-bm25"
    assert components["records"] == hash_value(sorted(hash_record(r) for r in RECORDS))
    assert components["record_count"] == 3
    assert components["doc_template"] == TEMPLATES.doc
    assert components["exact_fields"] == ["label", "synonyms"]
    assert components["text_config"] == "simple"
    assert components != bm25_components(RECORDS, TEMPLATES, exact_fields=("label",))
    assert components != bm25_components(RECORDS[:2], TEMPLATES, exact_fields=("label", "synonyms"))


def test_pgvector_components_carry_the_encoder_identity_and_name_fields() -> None:
    components = _dense_expected(Toy())
    assert components["engine"] == "pgvector"
    assert components["vector_type"] == "halfvec"
    assert components["metric"] == "inner_product"
    assert components["encoder"] == {"name": "toy", "dimension": 3}
    assert components["name_fields"] == ["label", "synonyms"]
    assert components["records"] == _bm25_expected()["records"]


def test_terms_components_fingerprint_equals_the_memory_store_fingerprint() -> None:
    from xwalk.stores.memory import MemoryStore

    components = terms_components(RECORDS)
    assert components["records"] == MemoryStore.from_source(RECORDS).fingerprint


def test_vector_literal_round_trips_float16_exactly() -> None:
    values = [0.1, -0.333333, 1e-5, 0.0, 0.9999, -1.0, 0.012345678]
    text = vector_literal(values)
    assert text.startswith("[") and text.endswith("]")
    parsed = [float(v) for v in text[1:-1].split(",")]
    for given, back in zip(values, parsed, strict=True):
        assert struct.pack("<e", given) == struct.pack("<e", back)
    assert len(text) < 9 * len(values)
    with pytest.raises(ValueError, match="finite"):
        vector_literal([math.nan])


# --- BM25 --------------------------------------------------------------------------


def _bm25(executor: FakeExecutor, **kw: Any) -> PostgresBM25Retriever:
    executor.answers.setdefault("meta_read", [_meta_row("bm25", _bm25_expected())])
    return PostgresBM25Retriever.open(executor, "genes", **kw)


def test_bm25_search_sql_shape_and_params() -> None:
    executor = FakeExecutor({"bm25_search": []})
    retriever = _bm25(executor, schema="xw")
    _run(retriever.search(SearchRequest(text="  IL-2  Receptor", limit=7)))
    ((sql, params),) = executor.tagged("bm25_search")
    assert "xw.terms" in sql
    assert "<@>" in sql and "to_bm25query" in sql and "to_tsvector('simple'" in sql
    assert "exact @>" in sql
    assert params == ["genes", "  IL-2  Receptor", exact_form("  IL-2  Receptor"), 7]
    assert params[2] == "il-2 receptor"


def test_bm25_hits_are_ranked_1_based_contiguous_exact_first_and_deduplicated() -> None:
    rows = [
        {"id": "G:1", "exact": 1, "score": -1.5, "fields": RECORDS[0].fields},
        {"id": "G:3", "exact": 0, "score": -2.0, "fields": RECORDS[2].fields},
        {"id": "G:1", "exact": 0, "score": -1.5, "fields": RECORDS[0].fields},
        {"id": "G:2", "exact": 0, "score": -0.5, "fields": RECORDS[1].fields},
    ]
    executor = FakeExecutor({"bm25_search": rows})
    retriever = _bm25(executor)
    hits = _run(retriever.search(SearchRequest(text="ADA", limit=20)))
    assert [h.record_id for h in hits] == ["G:1", "G:3", "G:2"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert {h.retriever for h in hits} == {"bm25"}
    assert hits[1].raw_score == pytest.approx(2.0)  # positive BM25, not the <@> distance


def test_bm25_cuts_to_the_limit_after_deduplication() -> None:
    rows = [{"id": f"G:{i}", "exact": 0, "score": -1.0 / i, "fields": {}} for i in range(1, 9)]
    executor = FakeExecutor({"bm25_search": rows})
    hits = _run(_bm25(executor).search(SearchRequest(text="x", limit=3)))
    assert [h.rank for h in hits] == [1, 2, 3]


def test_bm25_blank_query_makes_no_call() -> None:
    executor = FakeExecutor()
    retriever = _bm25(executor)
    assert _run(retriever.search(SearchRequest(text="   ", limit=5))) == []
    assert executor.tagged("bm25_search") == []


def test_bm25_open_refuses_a_missing_index() -> None:
    executor = FakeExecutor({"meta_read": []})
    with pytest.raises(IndexNotReadyError, match="no bm25 index") as info:
        PostgresBM25Retriever.open(executor, "genes")
    assert info.value.state == "missing"


def test_bm25_open_refuses_a_load_in_progress() -> None:
    executor = FakeExecutor({"meta_read": [_meta_row("bm25", _bm25_expected(), status="loading")]})
    with pytest.raises(IndexNotReadyError) as info:
        PostgresBM25Retriever.open(executor, "genes")
    assert info.value.state == "loading"


def test_bm25_open_refuses_a_mismatch_and_names_the_differences() -> None:
    executor = FakeExecutor({"meta_read": [_meta_row("bm25", _bm25_expected())]})
    expected = bm25_components(RECORDS[:2], TEMPLATES, exact_fields=("label", "synonyms"))
    with pytest.raises(IndexMismatchError) as info:
        PostgresBM25Retriever.open(executor, "genes", expected=expected)
    assert set(info.value.differences) == {"record_count", "records"}


def test_bm25_open_without_record_check_compares_configuration_only() -> None:
    executor = FakeExecutor({"meta_read": [_meta_row("bm25", _bm25_expected())]})
    expected = bm25_components(RECORDS[:2], TEMPLATES, exact_fields=("label", "synonyms"))
    retriever = PostgresBM25Retriever.open(
        executor, "genes", expected=expected, verify_records=False
    )
    # The identity is still the stored one, records digest included.
    assert retriever.fingerprint == hash_value(_bm25_expected())
    other = bm25_components(RECORDS, TEMPLATES, exact_fields=("label",))
    with pytest.raises(IndexMismatchError) as info:
        PostgresBM25Retriever.open(executor, "genes", expected=other, verify_records=False)
    assert set(info.value.differences) == {"exact_fields"}


def test_bm25_open_matches_and_takes_the_stored_identity() -> None:
    executor = FakeExecutor({"meta_read": [_meta_row("bm25", _bm25_expected(), default_limit=9)]})
    retriever = PostgresBM25Retriever.open(executor, "genes", expected=_bm25_expected(), name="kw")
    assert isinstance(retriever, Retriever)
    assert retriever.name == "kw"
    assert retriever.default_limit == 9
    assert retriever.fingerprint == hash_value(_bm25_expected())


def test_bm25_open_refuses_a_stored_fingerprint_that_does_not_hash_its_components() -> None:
    row = _meta_row("bm25", _bm25_expected())
    row["fingerprint"] = "tampered"
    executor = FakeExecutor({"meta_read": [row]})
    with pytest.raises(IndexMismatchError) as info:
        PostgresBM25Retriever.open(executor, "genes")
    assert "fingerprint" in info.value.differences


def test_open_accepts_meta_read_once_for_many_ontologies() -> None:
    executor = FakeExecutor({"meta_read": [_meta_row("bm25", _bm25_expected())]})
    meta = read_index_meta(executor, "xwalk")
    PostgresBM25Retriever.open(executor, "genes", meta=meta)
    PostgresBM25Retriever.open(executor, "genes", meta=meta)
    assert len(executor.tagged("meta_read")) == 1


def test_a_missing_schema_reads_as_no_index() -> None:
    executor = FakeExecutor(
        {"meta_read": PostgresError('relation "xwalk.index_meta" does not exist', code="42P01")}
    )
    assert read_index_meta(executor, "xwalk") == {}


# --- pgvector ----------------------------------------------------------------------


def _dense(executor: FakeExecutor, encoder: Toy, **kw: Any) -> PgVectorRetriever:
    executor.answers.setdefault(
        "meta_read", [_meta_row("dense", _dense_expected(encoder), dimension=3, vector_count=8)]
    )
    return PgVectorRetriever.open(executor, "genes", encoder, **kw)


def test_dense_search_is_one_read_only_transaction_with_ef_search() -> None:
    encoder = Toy()
    rows = [
        {"id": "G:2", "score": 0.9, "fields": RECORDS[1].fields},
        {"id": "G:1", "score": 0.5, "fields": RECORDS[0].fields},
    ]
    executor = FakeExecutor({"pgvector_search": rows})
    retriever = _dense(executor, encoder, overfetch=4)
    hits = _run(retriever.search(SearchRequest(text="adenosine", limit=5)))
    assert encoder.calls == [(["adenosine"], True)]
    ((statements, read_only),) = executor.transactions
    assert read_only is True
    (set_sql, set_params), (search_sql, params) = statements
    assert "hnsw.ef_search" in set_sql
    assert int(set_params[0]) >= 20
    assert "<#>" in search_sql and "group by" in search_sql.lower()
    assert params == ["genes", "[1,0,0]", 20, 5]
    assert [(h.record_id, h.rank, h.raw_score) for h in hits] == [("G:2", 1, 0.9), ("G:1", 2, 0.5)]


def test_dense_hits_keep_each_record_once() -> None:
    rows = [
        {"id": "G:2", "score": 0.9, "fields": {}},
        {"id": "G:2", "score": 0.8, "fields": {}},
        {"id": "G:3", "score": 0.7, "fields": {}},
    ]
    executor = FakeExecutor({"pgvector_search": rows})
    hits = _run(_dense(executor, Toy()).search(SearchRequest(text="x", limit=5)))
    assert [(h.record_id, h.rank) for h in hits] == [("G:2", 1), ("G:3", 2)]


def test_dense_blank_query_encodes_nothing() -> None:
    encoder = Toy()
    executor = FakeExecutor()
    assert _run(_dense(executor, encoder).search(SearchRequest(text=" ", limit=5))) == []
    assert encoder.calls == [] and executor.transactions == []


def test_dense_open_refuses_another_encoder() -> None:
    class Other(Toy):
        settings = {"query_prefix": "q: "}

    executor = FakeExecutor()
    _dense(executor, Toy())
    with pytest.raises(IndexMismatchError) as info:
        PgVectorRetriever.open(executor, "genes", Other(), expected=_dense_expected(Other()))
    assert "encoder.query_prefix" in info.value.differences


def test_dense_open_refuses_a_dimension_the_encoder_does_not_produce() -> None:
    class Wide(Toy):
        dimension = 4

    executor = FakeExecutor()
    _dense(executor, Toy())
    with pytest.raises(ValueError, match="dimension"):
        PgVectorRetriever.open(executor, "genes", Wide())


def test_dense_wrong_length_query_vector_is_refused() -> None:
    class Broken(Toy):
        def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    executor = FakeExecutor()
    retriever = _dense(executor, Broken())
    with pytest.raises(ValueError, match="length 2"):
        _run(retriever.search(SearchRequest(text="x", limit=5)))


# --- store -------------------------------------------------------------------------


def _store(executor: FakeExecutor) -> PostgresStore:
    executor.answers.setdefault("meta_read", [_meta_row("terms", terms_components(RECORDS))])
    return PostgresStore.open(executor, "genes")


def test_store_get_many_keeps_request_order_and_skips_unknown_ids() -> None:
    rows = [{"id": r.id, "fields": r.fields} for r in RECORDS]
    executor = FakeExecutor({"store_get": rows})
    store = _store(executor)
    assert isinstance(store, TargetStore)
    got = store.get_many(["G:3", "nope", "G:1"])
    assert [r.id for r in got] == ["G:3", "G:1"]
    assert got[1] == RECORDS[0]
    ((sql, params),) = executor.tagged("store_get")
    assert params[0] == "genes" and sorted(params[1]) == ["G:1", "G:3", "nope"]
    with pytest.raises(KeyError):
        _store(FakeExecutor({"store_get": []})).get("nope")


def test_store_fingerprint_is_the_snapshot_digest() -> None:
    store = _store(FakeExecutor())
    assert store.fingerprint == terms_components(RECORDS)["records"]


def test_retrievers_prime_the_store_so_fusion_needs_no_extra_round_trip() -> None:
    rows = [{"id": "G:1", "exact": 1, "score": -1.0, "fields": RECORDS[0].fields}]
    executor = FakeExecutor({"bm25_search": rows})
    store = _store(executor)
    executor.answers["meta_read"] = [_meta_row("bm25", _bm25_expected())]
    retriever = PostgresBM25Retriever.open(executor, "genes", store=store)
    _run(retriever.search(SearchRequest(text="ADA", limit=5)))
    assert [r.id for r in store.get_many(["G:1"])] == ["G:1"]
    assert executor.tagged("store_get") == []


# --- writer ------------------------------------------------------------------------


def test_writer_schema_uses_lakebase_bm25_halfvec_and_refuses_another_dimension() -> None:
    executor = FakeExecutor({"vector_dimension": [{"dimension": 3}]})
    writer = PostgresIndexWriter(executor, schema="xw", dimension=3)
    writer.create_schema()
    sql = "\n".join(s for s, _ in executor.calls)
    assert "create extension if not exists lakebase_text" in sql
    assert "create extension if not exists vector" in sql
    assert "halfvec(3)" in sql
    assert "xw.index_meta" in sql
    with pytest.raises(ValueError, match="halfvec\\(3\\)"):
        PostgresIndexWriter(executor, schema="xw", dimension=4).create_schema()


def test_writer_refuses_unsafe_identifiers() -> None:
    with pytest.raises(ValueError, match="schema"):
        PostgresIndexWriter(FakeExecutor(), schema="x; drop table y", dimension=3)
    writer = PostgresIndexWriter(FakeExecutor(), schema="xw", dimension=3)
    with pytest.raises(ValueError, match="ontology"):
        writer.create_indexes("bad'slug")


def test_writer_upserts_terms_as_one_jsonb_param_with_exact_forms() -> None:
    executor = FakeExecutor()
    writer = PostgresIndexWriter(executor, schema="xw", dimension=3)
    writer.upsert_terms("genes", RECORDS, TEMPLATES, exact_fields=("label", "synonyms"))
    ((sql, params),) = executor.tagged("terms_upsert")
    assert "on conflict" in sql.lower()
    rows = json.loads(params[1])
    assert rows[0]["id"] == "G:1"
    assert rows[0]["exact"] == ["ada", "ada1"]
    assert rows[0]["doc"] == TEMPLATES.render_doc(RECORDS[0])
    assert rows[0]["digest"] == hash_record(RECORDS[0])
    assert rows[0]["fields"] == RECORDS[0].fields


def test_writer_vector_rows_are_doc_then_names() -> None:
    rows = PostgresIndexWriter.vector_rows(
        RECORDS[:1], TEMPLATES, name_fields=("label", "synonyms")
    )
    assert [(r.record_id, r.ord, r.kind, r.text) for r in rows] == [
        ("G:1", 0, "doc", TEMPLATES.render_doc(RECORDS[0])),
        ("G:1", 1, "name", "ADA"),
        ("G:1", 2, "name", "ADA1"),
    ]


# --- Neon HTTP executor ------------------------------------------------------------


def _neon(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> NeonHTTPExecutor:
    return NeonHTTPExecutor(
        FAKE_URL, transport=httpx.MockTransport(handler), backoff_base=0.0, **kw
    )


def _result(names: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "fields": [{"name": n, "dataTypeID": 25} for n in names],
        "rows": rows,
        "command": "SELECT",
        "rowCount": len(rows),
        "rowAsArray": True,
    }


def test_neon_executor_posts_to_the_sql_endpoint_in_array_mode() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_result(["a", "b"], [[1, "x"], [2, None]]))

    executor = _neon(handler)
    rows = executor.execute("select $1::int a, $2::text b", [1, "x"])
    assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": None}]
    (request,) = seen
    assert str(request.url) == "https://ep-test-pooler.c-1.eu.aws.neon.tech/sql"
    assert request.headers["Neon-Connection-String"] == FAKE_URL
    assert request.headers["Neon-Array-Mode"] == "true"
    assert json.loads(request.content) == {
        "query": "select $1::int a, $2::text b",
        "params": [1, "x"],
    }
    assert executor.stats.requests == 1 and executor.stats.bytes_sent == len(request.content)


def test_neon_executor_transaction_is_one_request_with_batch_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"results": [_result(["a"], [[1]]), _result(["b"], [["y"]])]}
        )

    results = _neon(handler).transaction(
        [("select 1 a", []), ("select $1 b", ["y"])], read_only=True
    )
    assert results == [[{"a": 1}], [{"b": "y"}]]
    (request,) = seen
    assert request.headers["Neon-Batch-Read-Only"] == "true"
    assert json.loads(request.content)["queries"][1] == {"query": "select $1 b", "params": ["y"]}


def test_neon_executor_raises_postgres_errors_without_retrying_them() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            400,
            json={
                "message": 'relation "t" does not exist',
                "code": "42P01",
                "neon:retryable": False,
            },
        )

    with pytest.raises(PostgresError) as info:
        _neon(handler).execute("select * from t")
    assert info.value.code == "42P01"
    assert 'relation "t" does not exist' in str(info.value)
    assert len(calls) == 1


def test_neon_executor_retries_transient_failures() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        if len(calls) == 2:
            return httpx.Response(503, text="upstream unavailable")
        return httpx.Response(200, json=_result(["a"], [[1]]))

    assert _neon(handler, max_retries=2).execute("select 1 a") == [{"a": 1}]
    assert len(calls) == 3


def test_neon_executor_gives_up_after_max_retries_with_a_clean_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text=f"bad gateway for {FAKE_URL}")

    with pytest.raises(PostgresError) as info:
        _neon(handler, max_retries=1).execute("select 1")
    message = str(info.value)
    assert "502" in message
    assert "npg_SECRETpassword123" not in message
    assert "SECRETpassword" not in repr(info.value)


def test_neon_executor_scrubs_the_password_from_server_messages_and_repr() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"message": "auth failed for npg_SECRETpassword123", "code": "28P01"},
        )

    executor = _neon(handler, max_retries=0)
    with pytest.raises(PostgresError) as info:
        executor.execute("select 1")
    assert "SECRETpassword" not in str(info.value)
    assert "SECRETpassword" not in repr(executor)
    assert "ep-test-pooler" in repr(executor)


def test_neon_executor_refuses_a_non_postgres_url() -> None:
    with pytest.raises(ValueError, match="postgres") as info:
        NeonHTTPExecutor("https://user:npg_SECRETpassword123@example.com/x")
    assert "SECRETpassword" not in str(info.value)


def test_neon_executor_reports_a_non_json_success_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy</html>")

    with pytest.raises(PostgresError, match="not JSON"):
        _neon(handler, max_retries=0).execute("select 1")


def test_writer_drop_ontology_removes_rows_meta_and_its_partial_indexes() -> None:
    executor = FakeExecutor()
    PostgresIndexWriter(executor, schema="xw", dimension=3).drop_ontology("genes")
    sql = "\n".join(s for s, _ in executor.calls)
    assert "drop index if exists xw.terms_bm25_genes_" in sql
    assert "drop index if exists xw.terms_hnsw_genes_" in sql
    for table in ("term_vectors", "terms", "index_meta"):
        assert f"delete from xw.{table} where ontology = $1" in sql
    assert all(p == ["genes"] for s, p in executor.calls if "delete" in s)

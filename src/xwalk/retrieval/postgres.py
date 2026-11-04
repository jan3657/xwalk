"""Retrieval over Postgres: BM25, pgvector and a target store in one database.

For target collections too large to ship with an application (hundreds of thousands of
records, millions of vectors) the records, the BM25 index and the vectors can live in
Postgres. Three objects read them, each speaking an existing xwalk protocol:

- `PostgresBM25Retriever` (`Retriever`): BM25 through Neon's `lakebase_text` extension
  (a `lakebase_bm25` index over a `tsvector`, ranked by `<@>`), plus an exact
  whole-string match on label and synonyms that outranks any text match, like
  `BM25Retriever(exact_fields=...)`.
- `PgVectorRetriever` (`Retriever`): pgvector `halfvec` vectors under an HNSW index,
  inner product on unit vectors, with several vectors per record (the doc and each
  name) max-pooled so each record appears once, like `DenseRetriever(name_fields=...)`.
- `PostgresStore` (`TargetStore`): records by id.

All SQL goes through a small `SQLExecutor` protocol (`execute`, `transaction`) using
Postgres's own `$1` placeholders. `NeonHTTPExecutor` implements it over Neon's HTTPS SQL
endpoint with httpx (a base dependency), so it works where TCP to Postgres does not.

Identity works as for the file-based indexes: the loader (`PostgresIndexWriter`)
stores each index's components in an `index_meta` table, every retriever's
`fingerprint` is the hash of its stored components, and `open(..., expected=...)`
refuses a difference with `IndexMismatchError`. A load that has not finished is refused
too (`IndexNotReadyError`). Nothing here falls back to another engine.

Docs: docs/reference/retrieval.md, "Postgres retrieval".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
import re
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlsplit

import httpx

from xwalk.fingerprint import hash_record, hash_value, redact_url
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import (
    IndexMismatchError,
    RetrieverError,
    SearchRequest,
    component_differences,
)
from xwalk.retrieval.bm25 import NORMALIZATION_VERSION, exact_form
from xwalk.retrieval.dense import Encoder, encoder_identity, record_names
from xwalk.templates import TEMPLATE_RENDER_VERSION, TemplateSet, has_null_field

DEFAULT_SCHEMA = "xwalk"
BM25_ENGINE = "postgres-lakebase-bm25"
VECTOR_ENGINE = "pgvector"
TERMS_ENGINE = "postgres-terms"
FORMAT_VERSION = 1
DEFAULT_TEXT_CONFIG = "simple"
# Rows fetched from the HNSW index per record asked for: with per-name vectors a record
# owns several rows, so `limit` rows could name fewer than `limit` records.
DEFAULT_OVERFETCH = 5
MIN_EF_SEARCH = 40  # pgvector's own default

_IDENT = re.compile(r"[a-z_][a-z0-9_]{0,40}")
_ONTOLOGY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_TAG = re.compile(r"\s*/\* xwalk:(\w+) \*/")
_RECORD_KEYS = ("records", "record_count")


# --- the executor protocol ---------------------------------------------------------

Row = dict[str, Any]
Statement = tuple[str, Sequence[Any]]


@runtime_checkable
class SQLExecutor(Protocol):
    """Runs SQL with Postgres `$1, $2, ...` placeholders; rows come back as dicts.

    Values are JSON-shaped: text, numbers, booleans, None, lists (arrays) and dicts
    (jsonb). Cast in SQL where the type matters (`::float8`, `::int`): a bigint or
    numeric may arrive as a string over HTTP. Implementations must be thread-safe.
    """

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        """One statement, autocommitted."""

    def transaction(
        self, statements: Sequence[Statement], *, read_only: bool = False
    ) -> list[list[Row]]:
        """Several statements in one transaction (all or nothing), one result each."""


class PostgresError(RetrieverError):
    """A statement or the transport failed. `code` is the SQLSTATE when the server gave
    one. Messages never carry the connection string or its password."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable


def tag_of(sql: str) -> str | None:
    """The `/* xwalk:<tag> */` a statement starts with (every statement here has one, so
    a fake executor or a log can tell them apart)."""
    found = _TAG.match(sql)
    return found.group(1) if found else None


@dataclass
class ExecutorStats:
    """What an executor has sent so far (cumulative; thread-safe updates)."""

    requests: int = 0
    statements: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    seconds: float = 0.0


def _json_param(value: Any) -> Any:
    if isinstance(value, bytes | bytearray | memoryview):
        return "\\x" + bytes(value).hex()  # bytea hex, as Neon's own driver sends it
    if isinstance(value, tuple):
        return [_json_param(v) for v in value]
    return value


class NeonHTTPExecutor:
    """`SQLExecutor` over Neon's HTTPS SQL endpoint (`POST https://<host>/sql`).

    The protocol (that of `@neondatabase/serverless`, checked against the live endpoint
    2026-11-04): the connection string travels in the `Neon-Connection-String` header;
    the body is `{"query", "params"}` for one statement or `{"queries": [...]}` for a
    transaction (one request, rolled back as a whole on any error);
    `Neon-Array-Mode: true` returns rows as arrays beside `fields`; a SQL error is HTTP
    400 with a JSON body (`message`, `code`, `neon:retryable`). Requests and responses
    may be up to 64 MB; a gzip request body is refused, and one statement per query.

    Transport errors, HTTP 429/5xx and errors the server marks retryable are retried
    `max_retries` times with jittered backoff. A retried write may have been applied
    once already, so writes sent through this should be idempotent.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        trust_env: bool = True,
        endpoint: str | None = None,
    ) -> None:
        parts = urlsplit(connection_string)
        if parts.scheme not in {"postgres", "postgresql"} or not parts.hostname:
            raise ValueError(
                "NeonHTTPExecutor needs a postgres:// or postgresql:// connection string "
                f"with a host (got scheme {parts.scheme!r})"
            )
        self._connection_string = connection_string
        secrets = [connection_string]
        if parts.password:
            secrets += [parts.password, unquote(parts.password)]
        self._secrets = tuple(s for s in secrets if len(s) >= 6)
        self._host = parts.hostname
        self._endpoint = endpoint or f"https://{parts.hostname}/sql"
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._client = httpx.Client(timeout=timeout, transport=transport, trust_env=trust_env)
        self._lock = threading.Lock()
        self.stats = ExecutorStats()

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, name: str = "DATABASE_URL", **kwargs: Any
    ) -> NeonHTTPExecutor:
        import os

        source = os.environ if env is None else env
        value = (source.get(name) or "").strip()
        if not value:
            raise ValueError(f"{name} is not set")
        return cls(value, **kwargs)

    def __repr__(self) -> str:
        return f"NeonHTTPExecutor(host={self._host!r})"

    @property
    def host(self) -> str:
        return self._host

    def close(self) -> None:
        self._client.close()

    def scrub(self, text: str) -> str:
        """`text` without the connection string or password, and every URL redacted."""
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        return re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+", lambda m: redact_url(m[0]), text)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        payload = self._post({"query": sql, "params": [_json_param(p) for p in params]}, {}, 1)
        return _rows(payload)

    def transaction(
        self, statements: Sequence[Statement], *, read_only: bool = False
    ) -> list[list[Row]]:
        body = {
            "queries": [
                {"query": sql, "params": [_json_param(p) for p in params]}
                for sql, params in statements
            ]
        }
        headers = {"Neon-Batch-Read-Only": "true"} if read_only else {}
        payload = self._post(body, headers, len(statements))
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != len(statements):
            raise PostgresError("Neon answered a transaction without one result per statement")
        return [_rows(result) for result in results]

    def _post(self, body: dict[str, Any], headers: dict[str, str], statements: int) -> Any:
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request_headers = {
            "Neon-Connection-String": self._connection_string,
            "Neon-Array-Mode": "true",
            "Content-Type": "application/json",
            **headers,
        }
        last: PostgresError | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                time.sleep(self._backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random()))
            started = time.monotonic()
            try:
                response = self._client.post(
                    self._endpoint, content=content, headers=request_headers
                )
            except httpx.HTTPError as exc:
                self._book(len(content), 0, started, statements)
                last = PostgresError(
                    self.scrub(
                        f"could not reach Neon at {self._host}: {type(exc).__name__}: {exc}"
                    ),
                    retryable=True,
                )
                continue
            self._book(len(content), len(response.content), started, statements)
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise PostgresError(
                        f"Neon answered HTTP 200 with a body that is not JSON "
                        f"({len(response.content)} bytes)",
                        status=200,
                    ) from None
            last = self._error(response)
            if not last.retryable:
                raise last
        assert last is not None
        raise last

    def _error(self, response: httpx.Response) -> PostgresError:
        status = response.status_code
        data: Any = None
        with contextlib.suppress(ValueError):
            data = response.json()
        if isinstance(data, dict) and isinstance(data.get("message"), str):
            code = data.get("code") or None
            message = data["message"]
            detail = data.get("detail")
            if isinstance(detail, str) and detail:
                message = f"{message} ({detail})"
            retryable = bool(data.get("neon:retryable")) or status == 429 or status >= 500
            return PostgresError(
                self.scrub(f"Postgres error{f' {code}' if code else ''}: {message}")[:500],
                code=code if isinstance(code, str) else None,
                status=status,
                retryable=retryable,
            )
        text = self.scrub(response.text.strip())[:200]
        return PostgresError(
            f"Neon answered HTTP {status}{f': {text}' if text else ''}",
            status=status,
            retryable=status == 429 or status >= 500,
        )

    def _book(self, sent: int, received: int, started: float, statements: int) -> None:
        with self._lock:
            self.stats.requests += 1
            self.stats.statements += statements
            self.stats.bytes_sent += sent
            self.stats.bytes_received += received
            self.stats.seconds += time.monotonic() - started


def _rows(result: Any) -> list[Row]:
    if not isinstance(result, dict):
        raise PostgresError("Neon answered with a result that is not an object")
    fields = result.get("fields") or []
    names = [str(f.get("name")) for f in fields if isinstance(f, dict)]
    rows = result.get("rows") or []
    out: list[Row] = []
    for row in rows:
        if isinstance(row, dict):  # a server that ignored Neon-Array-Mode
            out.append(dict(row))
        else:
            out.append(dict(zip(names, row, strict=False)))
    return out


# --- identity ----------------------------------------------------------------------


def _digests(records: Iterable[Record]) -> tuple[list[str], bool]:
    digests: list[str] = []
    nulls = False
    for record in records:
        digests.append(hash_record(record))
        nulls = nulls or has_null_field(record)
    digests.sort()
    return digests, nulls


def terms_components(records: Iterable[Record]) -> dict[str, Any]:
    """The identity of the stored records. `records` is the same digest as
    `MemoryStore.from_source(records).fingerprint`."""
    digests, _ = _digests(records)
    return {
        "engine": TERMS_ENGINE,
        "format_version": FORMAT_VERSION,
        "record_count": len(digests),
        "records": hash_value(digests),
    }


def bm25_components(
    records: Iterable[Record],
    templates: TemplateSet,
    *,
    exact_fields: Sequence[str] = (),
    text_config: str = DEFAULT_TEXT_CONFIG,
) -> dict[str, Any]:
    """The identity of a Postgres BM25 index, computed from the records (nothing is
    queried). Mirrors `xwalk.retrieval.bm25.index_components` under another engine."""
    digests, nulls = _digests(records)
    components: dict[str, Any] = {
        "engine": BM25_ENGINE,
        "format_version": FORMAT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "doc_template": templates.doc,
        "exact_fields": list(exact_fields),
        "text_config": text_config,
        "record_count": len(digests),
        "records": hash_value(digests),
    }
    if nulls:
        components["template_render_version"] = TEMPLATE_RENDER_VERSION
    return components


def pgvector_components(
    records: Iterable[Record],
    templates: TemplateSet,
    encoder: Encoder,
    *,
    name_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The identity of a pgvector index (encodes nothing): records, doc template,
    encoder identity (name, dimension, settings), name fields, vector type, metric."""
    digests, nulls = _digests(records)
    components: dict[str, Any] = {
        "engine": VECTOR_ENGINE,
        "format_version": FORMAT_VERSION,
        "vector_type": "halfvec",
        "metric": "inner_product",
        "doc_template": templates.doc,
        "record_count": len(digests),
        "records": hash_value(digests),
        "encoder": encoder_identity(encoder),
        "name_fields": list(name_fields or ()),
    }
    if nulls:
        components["template_render_version"] = TEMPLATE_RENDER_VERSION
    return components


class IndexNotReadyError(LookupError):
    """No index of this kind for this ontology (`state == "missing"`), or its load has
    not finished (`state == "loading"`: running, or interrupted; rerun the loader)."""

    def __init__(self, ontology: str, kind: str, state: str, schema: str) -> None:
        self.ontology = ontology
        self.kind = kind
        self.state = state
        self.schema = schema
        if state == "missing":
            message = f"no {kind} index for {ontology!r} in schema {schema!r}; load it first"
        else:
            message = (
                f"the {kind} index for {ontology!r} in schema {schema!r} is incomplete "
                f"(status {state!r}: a load is running or was interrupted); rerun the loader"
            )
        super().__init__(message)


@dataclass(frozen=True)
class IndexMeta:
    """One row of `<schema>.index_meta`."""

    ontology: str
    kind: str  # "terms" | "bm25" | "dense"
    name: str
    status: str  # "loading" | "ready"
    components: dict[str, Any]
    fingerprint: str
    default_limit: int
    record_count: int | None = None
    vector_count: int | None = None
    dimension: int | None = None


MetaMap = Mapping[tuple[str, str], IndexMeta]


def _schema(schema: str) -> str:
    if not _IDENT.fullmatch(schema):
        raise ValueError(f"schema {schema!r} must match {_IDENT.pattern}")
    return schema


def _ontology(ontology: str) -> str:
    if not _ONTOLOGY.fullmatch(ontology):
        raise ValueError(f"ontology {ontology!r} must match {_ONTOLOGY.pattern}")
    return ontology


def _text_config(config: str) -> str:
    if not re.fullmatch(r"[a-z_]{1,40}", config):
        raise ValueError(f"text_config {config!r} must be a plain text search configuration")
    return config


def index_name(kind: str, ontology: str) -> str:
    """The deterministic name of an ontology's partial index (`bm25` or `hnsw`)."""
    slug = re.sub(r"[^a-z0-9]+", "_", ontology.lower()).strip("_")[:30]
    return f"terms_{kind}_{slug}_{hash_value(ontology)[:8]}"


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


def read_index_meta(
    executor: SQLExecutor, schema: str = DEFAULT_SCHEMA, ontologies: Sequence[str] | None = None
) -> dict[tuple[str, str], IndexMeta]:
    """Every stored index identity (or those of `ontologies`), keyed by (ontology, kind),
    in one query. A schema or table that does not exist reads as no indexes."""
    schema = _schema(schema)
    sql = (
        "/* xwalk:meta_read */ select ontology, kind, name, status, components, fingerprint, "
        "default_limit, record_count, vector_count, dimension "
        f"from {schema}.index_meta where ($1::text[] is null or ontology = any($1::text[]))"
    )
    try:
        rows = executor.execute(sql, [list(ontologies) if ontologies is not None else None])
    except PostgresError as exc:
        if exc.code in {"42P01", "3F000"}:  # undefined table / schema
            return {}
        raise
    found: dict[tuple[str, str], IndexMeta] = {}
    for row in rows:
        components = row.get("components")
        if isinstance(components, str):
            components = json.loads(components)
        meta = IndexMeta(
            ontology=str(row["ontology"]),
            kind=str(row["kind"]),
            name=str(row.get("name") or row["kind"]),
            status=str(row.get("status") or ""),
            components=dict(components or {}),
            fingerprint=str(row.get("fingerprint") or ""),
            default_limit=int(row.get("default_limit") or 20),
            record_count=_int(row.get("record_count")),
            vector_count=_int(row.get("vector_count")),
            dimension=_int(row.get("dimension")),
        )
        found[(meta.ontology, meta.kind)] = meta
    return found


def _checked_meta(
    executor: SQLExecutor,
    schema: str,
    ontology: str,
    kind: str,
    *,
    expected: Mapping[str, Any] | None,
    verify_records: bool,
    meta: MetaMap | None,
) -> IndexMeta:
    """The stored meta row, refused when missing, unfinished, self-inconsistent, or
    different from `expected` (records digest and count left out unless
    `verify_records`)."""
    schema = _schema(schema)
    _ontology(ontology)
    found = (meta if meta is not None else read_index_meta(executor, schema, [ontology])).get(
        (ontology, kind)
    )
    where = f"postgres:{schema}/{ontology}/{kind}"
    if found is None:
        raise IndexNotReadyError(ontology, kind, "missing", schema)
    if found.status != "ready":
        raise IndexNotReadyError(ontology, kind, found.status or "unknown", schema)
    computed = hash_value(found.components)
    if found.fingerprint != computed:
        raise IndexMismatchError(where, {"fingerprint": (found.fingerprint, computed)})
    if expected is not None:
        stored = dict(found.components)
        wanted = dict(expected)
        if not verify_records:
            for key in _RECORD_KEYS:
                stored.pop(key, None)
                wanted.pop(key, None)
        differences = component_differences(stored, wanted)
        if differences:
            raise IndexMismatchError(where, differences)
    return found


# --- store -------------------------------------------------------------------------


class PostgresStore:
    """`TargetStore` over `<schema>.terms` for one ontology.

    Records seen by a retriever built with `store=` are remembered (a bounded LRU), so
    fusion's `get_many` after a search usually needs no round trip of its own."""

    def __init__(
        self,
        executor: SQLExecutor,
        *,
        ontology: str,
        fingerprint: str,
        schema: str = DEFAULT_SCHEMA,
        record_count: int | None = None,
        cache_size: int = 4096,
    ) -> None:
        self._executor = executor
        self._schema = _schema(schema)
        self._ontology = _ontology(ontology)
        self._fingerprint = fingerprint
        self.record_count = record_count
        self._cache: OrderedDict[str, Record] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        executor: SQLExecutor,
        ontology: str,
        *,
        schema: str = DEFAULT_SCHEMA,
        expected: Mapping[str, Any] | None = None,
        verify_records: bool = True,
        meta: MetaMap | None = None,
        cache_size: int = 4096,
    ) -> PostgresStore:
        """Open the stored records; `expected` from `terms_components`."""
        found = _checked_meta(
            executor,
            schema,
            ontology,
            "terms",
            expected=expected,
            verify_records=verify_records,
            meta=meta,
        )
        return cls(
            executor,
            ontology=ontology,
            fingerprint=str(found.components.get("records", found.fingerprint)),
            schema=schema,
            record_count=found.record_count,
            cache_size=cache_size,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def remember(self, records: Iterable[Record]) -> None:
        with self._lock:
            for record in records:
                self._cache[record.id] = record
                self._cache.move_to_end(record.id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def get(self, record_id: str) -> Record:
        found = self.get_many([record_id])
        if not found:
            raise KeyError(record_id)
        return found[0]

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        with self._lock:
            known = {rid: self._cache[rid] for rid in record_ids if rid in self._cache}
        missing = sorted({rid for rid in record_ids if rid not in known})
        if missing:
            rows = self._executor.execute(
                f"/* xwalk:store_get */ select id, fields from {self._schema}.terms "
                "where ontology = $1 and id = any($2::text[])",
                [self._ontology, missing],
            )
            fetched = [_record(row) for row in rows]
            self.remember(fetched)
            known.update((r.id, r) for r in fetched)
        return [known[rid] for rid in record_ids if rid in known]


def _record(row: Mapping[str, Any]) -> Record:
    fields = row.get("fields")
    if isinstance(fields, str):
        fields = json.loads(fields)
    return Record(str(row["id"]), dict(fields or {}))


# --- retrievers --------------------------------------------------------------------


def _hits(
    rows: Iterable[Mapping[str, Any]], name: str, limit: int, score: Any
) -> tuple[list[RetrievalHit], list[Record]]:
    """Best-first rows to hits: first occurrence of each record, cut to `limit`, ranks
    1..n. Also the records carried by the rows (for the store)."""
    seen: set[str] = set()
    hits: list[RetrievalHit] = []
    records: list[Record] = []
    for row in rows:
        record_id = str(row["id"])
        if record_id in seen:
            continue
        seen.add(record_id)
        if row.get("fields") is not None:
            records.append(_record(row))
        if len(hits) < limit:
            hits.append(
                RetrievalHit(
                    record_id=record_id,
                    retriever=name,
                    raw_score=score(row),
                    rank=len(hits) + 1,
                )
            )
    return hits, records


class PostgresBM25Retriever:
    """BM25 over `<schema>.terms` with `lakebase_text`, plus the exact-match boost.

    Records whose exact forms (`exact_form` of each label/synonym, stored by the loader)
    contain `exact_form(query)` come first, ordered by their BM25 score; then the BM25
    hits with a positive score. `raw_score` is the BM25 score (the negation of `<@>`).
    The query text is tokenised by Postgres (`to_tsvector(text_config, text)`); it is
    never parsed as query syntax, so it needs no sanitising.
    """

    def __init__(
        self,
        executor: SQLExecutor,
        *,
        ontology: str,
        name: str,
        fingerprint: str,
        schema: str = DEFAULT_SCHEMA,
        exact_fields: Sequence[str] = (),
        text_config: str = DEFAULT_TEXT_CONFIG,
        default_limit: int = 20,
        store: PostgresStore | None = None,
    ) -> None:
        self._executor = executor
        self._schema = _schema(schema)
        self._ontology = _ontology(ontology)
        self._name = name
        self._fingerprint = fingerprint
        self._exact_fields = tuple(exact_fields)
        self._default_limit = default_limit
        self._store = store
        config = _text_config(text_config)
        index = f"'{self._schema}.{index_name('bm25', self._ontology)}'::regclass"
        score = f"t.tsv <@> to_bm25query(to_tsvector('{config}', $2::text), {index})"
        self._sql = (
            "/* xwalk:bm25_search */ with exact_hits as ("
            f" select t.id, 1 as exact, ({score})::float8 as score, t.fields"
            f" from {self._schema}.terms t"
            " where t.ontology = $1 and $3::text <> '' and t.exact @> array[$3::text]"
            " order by score, t.id limit $4"
            "), text_hits as ("
            f" select t.id, 0 as exact, ({score})::float8 as score, t.fields"
            f" from {self._schema}.terms t where t.ontology = $1"
            f" order by {score} limit $4"
            ") select id, exact, score, fields from exact_hits"
            " union all select id, exact, score, fields from text_hits where score < 0"
            " order by exact desc, score asc, id asc"
        )

    @classmethod
    def open(
        cls,
        executor: SQLExecutor,
        ontology: str,
        *,
        schema: str = DEFAULT_SCHEMA,
        name: str | None = None,
        expected: Mapping[str, Any] | None = None,
        verify_records: bool = True,
        default_limit: int | None = None,
        meta: MetaMap | None = None,
        store: PostgresStore | None = None,
    ) -> PostgresBM25Retriever:
        """Open a loaded index. `expected` from `bm25_components`; with
        `verify_records=False` only the configuration is compared (for a caller that
        does not hold the records); the identity still includes the stored digest."""
        found = _checked_meta(
            executor,
            schema,
            ontology,
            "bm25",
            expected=expected,
            verify_records=verify_records,
            meta=meta,
        )
        components = found.components
        return cls(
            executor,
            ontology=ontology,
            name=name or found.name,
            fingerprint=found.fingerprint,
            schema=schema,
            exact_fields=tuple(components.get("exact_fields") or ()),
            text_config=str(components.get("text_config") or DEFAULT_TEXT_CONFIG),
            default_limit=default_limit if default_limit is not None else found.default_limit,
            store=store,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def default_limit(self) -> int:
        return self._default_limit

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        if not text.strip() or limit < 1:
            return []
        exact = exact_form(text) if self._exact_fields else ""
        rows = self._executor.execute(self._sql, [self._ontology, text, exact, limit])
        hits, records = _hits(rows, self._name, limit, lambda r: -float(r["score"] or 0.0))
        if self._store is not None:
            self._store.remember(records)
        return hits

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return await asyncio.to_thread(self._search_sync, request.text, request.limit)


def vector_literal(vector: Sequence[float]) -> str:
    """A pgvector text literal for `vector`, rounded to float16 first and written with
    five significant digits, which reads back as exactly that float16 (`halfvec`
    stores no more). About 8 characters a value instead of 18 for a float64 repr."""
    for value in vector:
        if not math.isfinite(value):
            raise ValueError("a vector must hold only finite values")
    halves = struct.unpack(f"<{len(vector)}e", struct.pack(f"<{len(vector)}e", *vector))
    return "[" + ",".join(f"{v:.5g}" for v in halves) + "]"


class PgVectorRetriever:
    """Nearest neighbours over `<schema>.term_vectors` (halfvec, HNSW, inner product).

    A record may own several vectors (its doc and each name); the `limit * overfetch`
    nearest rows are fetched from the index and each record keeps its best, so each
    record appears once and the cut to `limit` is over records. `raw_score` is the
    inner product. The HNSW search is approximate; `ef_search` is raised to at least the
    number of rows fetched.
    """

    def __init__(
        self,
        executor: SQLExecutor,
        encoder: Encoder,
        *,
        ontology: str,
        name: str,
        fingerprint: str,
        dimension: int,
        schema: str = DEFAULT_SCHEMA,
        default_limit: int = 20,
        overfetch: int = DEFAULT_OVERFETCH,
        ef_search: int | None = None,
        store: PostgresStore | None = None,
    ) -> None:
        if overfetch < 1:
            raise ValueError("overfetch must be at least 1")
        self._executor = executor
        self._encoder = encoder
        self._schema = _schema(schema)
        self._ontology = _ontology(ontology)
        self._name = name
        self._fingerprint = fingerprint
        self._dimension = dimension
        self._default_limit = default_limit
        self._overfetch = overfetch
        self._ef_search = ef_search
        self._store = store
        s = self._schema
        self._sql = (
            "/* xwalk:pgvector_search */ select b.record_id as id, (-b.d)::float8 as score,"
            " t.fields from ("
            " select r.record_id, min(r.d) as d from ("
            f" select v.record_id, v.embedding <#> $2::halfvec as d from {s}.term_vectors v"
            " where v.ontology = $1 order by v.embedding <#> $2::halfvec limit $3"
            " ) r group by r.record_id order by min(r.d), r.record_id limit $4"
            f") b left join {s}.terms t on t.ontology = $1 and t.id = b.record_id"
            " order by b.d, b.record_id"
        )

    @classmethod
    def open(
        cls,
        executor: SQLExecutor,
        ontology: str,
        encoder: Encoder,
        *,
        schema: str = DEFAULT_SCHEMA,
        name: str | None = None,
        expected: Mapping[str, Any] | None = None,
        verify_records: bool = True,
        default_limit: int | None = None,
        overfetch: int = DEFAULT_OVERFETCH,
        ef_search: int | None = None,
        meta: MetaMap | None = None,
        store: PostgresStore | None = None,
    ) -> PgVectorRetriever:
        """Open loaded vectors for `encoder`; `expected` from `pgvector_components`.
        The encoder's dimension must equal the stored one (ValueError otherwise)."""
        found = _checked_meta(
            executor,
            schema,
            ontology,
            "dense",
            expected=expected,
            verify_records=verify_records,
            meta=meta,
        )
        stored = found.components.get("encoder") or {}
        dimension = found.dimension or int(stored.get("dimension") or 0)
        if dimension != encoder.dimension:
            raise ValueError(
                f"the vectors for {ontology!r} have dimension {dimension} but encoder "
                f"{encoder.name!r} produces {encoder.dimension}"
            )
        return cls(
            executor,
            encoder,
            ontology=ontology,
            name=name or found.name,
            fingerprint=found.fingerprint,
            dimension=dimension,
            schema=schema,
            default_limit=default_limit if default_limit is not None else found.default_limit,
            overfetch=overfetch,
            ef_search=ef_search,
            store=store,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def default_limit(self) -> int:
        return self._default_limit

    @property
    def encoder_identity(self) -> dict[str, Any]:
        """The live encoder's identity, for the run fingerprint (as `DenseRetriever`)."""
        return encoder_identity(self._encoder)

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        if not text.strip() or limit < 1:
            return []
        query = self._encoder.encode([text], is_query=True)[0]
        if len(query) != self._dimension:
            raise ValueError(
                f"encoder {self._encoder.name!r} returned a query vector of length "
                f"{len(query)}; the index holds {self._dimension}"
            )
        fetch = limit * self._overfetch
        ef = max(self._ef_search or 0, fetch, MIN_EF_SEARCH)
        _, rows = self._executor.transaction(
            [
                (
                    "/* xwalk:pgvector_settings */ select set_config('hnsw.ef_search', $1, true)",
                    [str(ef)],
                ),
                (self._sql, [self._ontology, vector_literal(query), fetch, limit]),
            ],
            read_only=True,
        )
        hits, records = _hits(rows, self._name, limit, lambda r: float(r["score"]))
        if self._store is not None:
            self._store.remember(records)
        return hits

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return await asyncio.to_thread(self._search_sync, request.text, request.limit)


# --- writing -----------------------------------------------------------------------


@dataclass(frozen=True)
class VectorRow:
    """One vector to store: `ord` 0 is the record's rendered doc, 1.. its names."""

    record_id: str
    ord: int
    kind: str  # "doc" | "name"
    text: str


def _term_row(
    record: Record, templates: TemplateSet, exact_fields: Sequence[str]
) -> dict[str, Any]:
    fields = record.fields
    synonyms = fields.get("synonyms")
    if synonyms is None:
        synonym_list: list[str] = []
    elif isinstance(synonyms, str):
        synonym_list = [synonyms]
    else:
        synonym_list = [str(s) for s in synonyms if s is not None]
    exact: list[str] = []
    for name in exact_fields:
        value = fields.get(name)
        if value is None:
            continue
        for item in [value] if isinstance(value, str) else list(value):
            form = exact_form(str(item))
            if form and form not in exact:
                exact.append(form)
    return {
        "id": record.id,
        "label": str(fields.get("label") or ""),
        "synonyms": synonym_list,
        "definition": str(fields.get("definition") or ""),
        "fields": dict(fields),
        "doc": templates.render_doc(record),
        "exact": exact,
        "digest": hash_record(record),
    }


@dataclass
class PostgresIndexWriter:
    """Creates the schema and writes records, vectors and identities. Every write is
    idempotent (upserts keyed by ontology and id), so a load can be rerun or resumed.

    Layout (one schema, every ontology in the same tables):

    - `terms(ontology, id, label, synonyms, definition, fields jsonb, doc, tsv, exact,
      digest)`, primary key (ontology, id); a GIN index on `exact`; one partial
      `lakebase_bm25` index on `tsv` per ontology.
    - `term_vectors(ontology, record_id, ord, kind, text, embedding halfvec(D))`,
      primary key (ontology, record_id, ord); one partial HNSW index
      (`halfvec_ip_ops`) per ontology.
    - `index_meta(ontology, kind, name, status, components, fingerprint, default_limit,
      record_count, vector_count, dimension, updated_at)`, primary key (ontology, kind).
    """

    executor: SQLExecutor
    schema: str = DEFAULT_SCHEMA
    dimension: int = 1024
    text_config: str = DEFAULT_TEXT_CONFIG
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    _schema_ok: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _schema(self.schema)
        _text_config(self.text_config)
        if not 1 <= self.dimension <= 4000:
            raise ValueError("halfvec HNSW indexes support 1 to 4000 dimensions")

    def _run(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        return self.executor.execute(sql, params)

    def stored_dimension(self) -> int | None:
        rows = self._run(
            "/* xwalk:vector_dimension */ select atttypmod as dimension from pg_attribute "
            f"where attrelid = to_regclass('{self.schema}.term_vectors') "
            "and attname = 'embedding'"
        )
        return int(rows[0]["dimension"]) if rows and rows[0].get("dimension") is not None else None

    def create_schema(self) -> None:
        stored = self.stored_dimension()
        if stored is not None and stored != self.dimension:
            raise ValueError(
                f"{self.schema}.term_vectors holds halfvec({stored}) vectors, not "
                f"halfvec({self.dimension}); load with --dimensions {stored}, or drop the "
                f"schema (or use another one) to change the dimension"
            )
        s, d = self.schema, self.dimension
        statements = [
            "create extension if not exists vector",
            "create extension if not exists lakebase_text",
            f"create schema if not exists {s}",
            f"create table if not exists {s}.terms ("
            " ontology text not null, id text not null, label text not null default '',"
            " synonyms text[] not null default '{}', definition text not null default '',"
            " fields jsonb not null, doc text not null, tsv tsvector not null,"
            " exact text[] not null default '{}', digest text not null,"
            " primary key (ontology, id))",
            f"create index if not exists terms_exact_gin on {s}.terms using gin (exact)",
            f"create table if not exists {s}.term_vectors ("
            " ontology text not null, record_id text not null, ord int not null,"
            " kind text not null check (kind in ('doc', 'name')), text text not null,"
            f" embedding halfvec({d}) not null, primary key (ontology, record_id, ord))",
            f"create table if not exists {s}.index_meta ("
            " ontology text not null, kind text not null, name text not null,"
            " status text not null, components jsonb not null, fingerprint text not null,"
            " default_limit int not null, record_count int, vector_count int, dimension int,"
            " updated_at timestamptz not null default now(), primary key (ontology, kind))",
        ]
        for sql in statements:
            self._run(f"/* xwalk:schema */ {sql}")
        self._schema_ok = True

    def drop_schema(self) -> None:
        self._run(f"/* xwalk:schema_drop */ drop schema if exists {self.schema} cascade")

    def create_indexes(self, ontology: str) -> None:
        """The per-ontology partial BM25 and HNSW indexes (no-ops when present). Create
        them after the bulk load: building once is faster than updating per row."""
        _ontology(ontology)
        s = self.schema
        literal = "'" + ontology.replace("'", "''") + "'"
        self._run(
            f"/* xwalk:index_bm25 */ create index if not exists {index_name('bm25', ontology)} "
            f"on {s}.terms using lakebase_bm25 (tsv) where ontology = {literal}"
        )
        self._run(
            f"/* xwalk:index_hnsw */ create index if not exists {index_name('hnsw', ontology)} "
            f"on {s}.term_vectors using hnsw (embedding halfvec_ip_ops) "
            f"with (m = {int(self.hnsw_m)}, ef_construction = {int(self.hnsw_ef_construction)}) "
            f"where ontology = {literal}"
        )

    def drop_ontology(self, ontology: str) -> None:
        """Remove one ontology entirely: its partial indexes, vectors, terms and stored
        identities (readers then report it missing). Other ontologies are untouched."""
        _ontology(ontology)
        s = self.schema
        for kind in ("bm25", "hnsw"):
            self._run(
                f"/* xwalk:index_drop */ drop index if exists {s}.{index_name(kind, ontology)}"
            )
        for tag, table in (
            ("vectors_drop", "term_vectors"),
            ("terms_drop", "terms"),
            ("meta_drop", "index_meta"),
        ):
            self._run(f"/* xwalk:{tag} */ delete from {s}.{table} where ontology = $1", [ontology])

    def vacuum(self) -> None:
        """Plain VACUUM (no lock beyond the usual) so deleted rows' space is reused, and
        trailing empty pages are returned (it ran over Neon's HTTP endpoint on
        2026-11-04: after dropping a 2,000-gene slice the database went from 62.7 MB
        to 31.6 MB)."""
        for table in ("terms", "term_vectors"):
            self._run(f"/* xwalk:vacuum */ vacuum {self.schema}.{table}")

    def analyze(self) -> None:
        for table in ("terms", "term_vectors"):
            self._run(f"/* xwalk:analyze */ analyze {self.schema}.{table}")

    def upsert_terms(
        self,
        ontology: str,
        records: Sequence[Record],
        templates: TemplateSet,
        *,
        exact_fields: Sequence[str] = (),
    ) -> int:
        """Insert or update `records` (unchanged rows, by digest, are left alone)."""
        _ontology(ontology)
        if not records:
            return 0
        rows = [_term_row(r, templates, exact_fields) for r in records]
        s = self.schema
        self._run(
            f"/* xwalk:terms_upsert */ insert into {s}.terms as t (ontology, id, label, "
            "synonyms, definition, fields, doc, tsv, exact, digest) "
            "select $1, x.id, x.label, x.synonyms, x.definition, x.fields, x.doc, "
            f"to_tsvector('{self.text_config}', x.doc), x.exact, x.digest "
            "from jsonb_to_recordset($2::jsonb) as x(id text, label text, synonyms text[], "
            "definition text, fields jsonb, doc text, exact text[], digest text) "
            "on conflict (ontology, id) do update set label = excluded.label, "
            "synonyms = excluded.synonyms, definition = excluded.definition, "
            "fields = excluded.fields, doc = excluded.doc, tsv = excluded.tsv, "
            "exact = excluded.exact, digest = excluded.digest "
            "where t.digest is distinct from excluded.digest",
            [ontology, json.dumps(rows, ensure_ascii=False)],
        )
        return len(rows)

    @staticmethod
    def vector_rows(
        records: Iterable[Record], templates: TemplateSet, *, name_fields: Sequence[str] = ()
    ) -> list[VectorRow]:
        """Per record: its doc (ord 0), then each distinct name (`record_names`)."""
        rows: list[VectorRow] = []
        for record in records:
            rows.append(VectorRow(record.id, 0, "doc", templates.render_doc(record)))
            for i, text in enumerate(record_names(record, name_fields) if name_fields else []):
                rows.append(VectorRow(record.id, i + 1, "name", text))
        return rows

    def present_vectors(self, ontology: str, rows: Sequence[VectorRow]) -> set[tuple[str, int]]:
        """Which of `rows` are already stored with the same text (resuming skips them)."""
        if not rows:
            return set()
        found = self._run(
            "/* xwalk:vectors_present */ select v.record_id, v.ord "
            f"from {self.schema}.term_vectors v join unnest($2::text[], $3::int[], $4::text[]) "
            "as k(record_id, ord, text) on v.record_id = k.record_id and v.ord = k.ord "
            "and v.text = k.text where v.ontology = $1",
            [
                _ontology(ontology),
                [r.record_id for r in rows],
                [r.ord for r in rows],
                [r.text for r in rows],
            ],
        )
        return {(str(f["record_id"]), int(f["ord"])) for f in found}

    def upsert_vectors(
        self, ontology: str, rows: Sequence[VectorRow], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Store one batch (one statement: a crash loses at most this batch)."""
        _ontology(ontology)
        if len(rows) != len(vectors):
            raise ValueError(f"{len(rows)} rows but {len(vectors)} vectors")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"a vector has length {len(vector)}; the table holds halfvec({self.dimension})"
                )
        if not rows:
            return 0
        self._run(
            f"/* xwalk:vectors_upsert */ insert into {self.schema}.term_vectors "
            "(ontology, record_id, ord, kind, text, embedding) "
            "select $1, k.record_id, k.ord, k.kind, k.text, k.e::halfvec "
            "from unnest($2::text[], $3::int[], $4::text[], $5::text[], $6::text[]) "
            "as k(record_id, ord, kind, text, e) "
            "on conflict (ontology, record_id, ord) do update set kind = excluded.kind, "
            "text = excluded.text, embedding = excluded.embedding",
            [
                ontology,
                [r.record_id for r in rows],
                [r.ord for r in rows],
                [r.kind for r in rows],
                [r.text for r in rows],
                [vector_literal(v) for v in vectors],
            ],
        )
        return len(rows)

    def prune(
        self, ontology: str, record_ids: Sequence[str], vector_keys: Sequence[tuple[str, int]]
    ) -> tuple[int, int]:
        """Delete terms and vectors of `ontology` that are not in the current load.
        Returns (terms deleted, vectors deleted)."""
        _ontology(ontology)
        s = self.schema
        terms = self._run(
            f"/* xwalk:terms_prune */ with d as (delete from {s}.terms where ontology = $1 "
            "and not (id = any($2::text[])) returning 1) select count(*)::int as n from d",
            [ontology, list(record_ids)],
        )
        vectors = self._run(
            f"/* xwalk:vectors_prune */ with d as (delete from {s}.term_vectors v "
            "where v.ontology = $1 and not exists (select 1 from unnest($2::text[], $3::int[]) "
            "as k(record_id, ord) where k.record_id = v.record_id and k.ord = v.ord) "
            "returning 1) select count(*)::int as n from d",
            [ontology, [k[0] for k in vector_keys], [k[1] for k in vector_keys]],
        )
        return int(terms[0]["n"]) if terms else 0, int(vectors[0]["n"]) if vectors else 0

    def write_meta(
        self,
        ontology: str,
        kind: str,
        components: Mapping[str, Any],
        *,
        name: str,
        status: str,
        default_limit: int = 20,
        record_count: int | None = None,
        vector_count: int | None = None,
        dimension: int | None = None,
    ) -> None:
        """Record an index identity. Write `status="loading"` before changing an
        ontology's rows and `"ready"` once they are complete: readers refuse anything
        but ready, so an interrupted load is never searched as if it were whole."""
        if kind not in {"terms", "bm25", "dense"}:
            raise ValueError(f"unknown index kind {kind!r}")
        if status not in {"loading", "ready"}:
            raise ValueError(f"unknown status {status!r}")
        self._run(
            f"/* xwalk:meta_write */ insert into {self.schema}.index_meta (ontology, kind, "
            "name, status, components, fingerprint, default_limit, record_count, "
            "vector_count, dimension, updated_at) "
            "values ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, now()) "
            "on conflict (ontology, kind) do update set name = excluded.name, "
            "status = excluded.status, components = excluded.components, "
            "fingerprint = excluded.fingerprint, default_limit = excluded.default_limit, "
            "record_count = excluded.record_count, vector_count = excluded.vector_count, "
            "dimension = excluded.dimension, updated_at = now()",
            [
                _ontology(ontology),
                kind,
                name,
                status,
                json.dumps(dict(components)),
                hash_value(dict(components)),
                default_limit,
                record_count,
                vector_count,
                dimension,
            ],
        )

    def sizes(self) -> dict[str, int]:
        """Bytes: the whole database, and each table with its indexes and TOAST."""
        rows = self._run(
            "/* xwalk:sizes */ select pg_database_size(current_database())::float8 as database, "
            f"coalesce(pg_total_relation_size(to_regclass('{self.schema}.terms')), 0)::float8 "
            "as terms, "
            f"coalesce(pg_total_relation_size(to_regclass('{self.schema}.term_vectors')), 0)"
            "::float8 as term_vectors, "
            f"coalesce(pg_total_relation_size(to_regclass('{self.schema}.index_meta')), 0)"
            "::float8 as index_meta"
        )
        return {k: int(v) for k, v in (rows[0] if rows else {}).items()}


__all__ = [
    "BM25_ENGINE",
    "DEFAULT_SCHEMA",
    "ExecutorStats",
    "IndexMeta",
    "IndexNotReadyError",
    "NeonHTTPExecutor",
    "PgVectorRetriever",
    "PostgresBM25Retriever",
    "PostgresError",
    "PostgresIndexWriter",
    "PostgresStore",
    "SQLExecutor",
    "VECTOR_ENGINE",
    "VectorRow",
    "bm25_components",
    "index_name",
    "pgvector_components",
    "read_index_meta",
    "tag_of",
    "terms_components",
    "vector_literal",
]

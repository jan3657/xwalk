"""Per-ontology indexes: BM25 built at cold start into the cache dir, dense opened from
the committed indexes, and an honest status for each (docs/ner-site/PLAN.md section 5).

With ``NER_SITE_BACKEND=neon`` (docs/ner-site/NEON.md) the records, BM25 and vectors are
read from Postgres instead: one ``PostgresStore``, ``PostgresBM25Retriever`` and
``PgVectorRetriever`` per ontology, opened against the identity the loader
(``ner-site/tools/load_neon.py``) stored, under the same names and statuses.

Nothing here raises for a missing, stale or unreadable dense index or a missing key:
the ontology runs on BM25 alone and its ``status["dense"]`` says why, in words an
operator can act on. A BM25 build that fails is reported the same way.
"""

from __future__ import annotations

import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from ner_site.ontologies import FULL_ROUTES, ROUTES, TEMPLATES, load_terms
from ner_site.schema import RetrieverStatus
from xwalk.records import Record
from xwalk.retrieval import bm25, dense
from xwalk.retrieval.base import IndexMismatchError, Retriever
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.dense import DenseRetriever, Encoder
from xwalk.retrieval.embeddings_api import default_embeddings_encoder
from xwalk.retrieval.postgres import (
    IndexMeta,
    IndexNotReadyError,
    NeonHTTPExecutor,
    PgVectorRetriever,
    PostgresBM25Retriever,
    PostgresError,
    PostgresStore,
    SQLExecutor,
    bm25_components,
    pgvector_components,
    read_index_meta,
    terms_components,
)
from xwalk.stores.base import TargetStore
from xwalk.stores.memory import MemoryStore

BM25_NAME = "bm25"
# The key of this retriever in every match's `retrievers`; fixed on build and on open.
DENSE_NAME = "dense-qwen3"
EXACT_FIELDS = ("label", "synonyms")
NAME_FIELDS = ("label", "synonyms")
DEFAULT_LIMIT = 20
BUILD_TOOL = "ner-site/tools/build_indexes.py"
LOAD_TOOL = "ner-site/tools/load_neon.py"
NEON_SCHEMA = "xwalk"
NEON_DIMENSIONS = 1024
NO_DATABASE = "DATABASE_URL is not set (NER_SITE_BACKEND=neon)"
UNREACHABLE = "neon database could not be queried"
# Per-request database calls: short, one retry. A failure degrades the entity (a note),
# like an embeddings failure; it never fails the request.
NEON_TIMEOUT_S = 15.0
NEON_RETRIES = 1
Backend = Literal["bundled", "neon"]
Catalog = Literal["samples", "full"]

# Query embeddings at request time: a short timeout and no retry, so a slow or
# rate-limited provider degrades a request (BM25 only) instead of eating the function's
# whole duration. With no retry the encoder never sleeps, so a 429 with a long
# Retry-After (which it would otherwise honour for up to 300 s) fails at once. Neither
# setting changes what a vector means, so an index built with the defaults still matches.
QUERY_ENCODER_SETTINGS: dict[str, Any] = {"timeout": 15.0, "max_retries": 0}

_BUILD_LOCK = threading.Lock()
_MEMO_LOCK = threading.Lock()
_MEMO: dict[tuple[str | int | None, ...], Mapping[str, OntologyIndex]] = {}


class PrefetchEncoder:
    """The shared query encoder, able to embed many queries in one request.

    ``prefetch`` sends every query not yet embedded in one ``encode`` call; a later
    ``encode(..., is_query=True)`` of those texts (each dense search embeds its own query)
    is served from memory. One request of 25 queries took 3 to 10 s through OpenRouter on
    4 November 2026, where 25 single requests, 4 at a time, took 52 to 67 s. Name,
    dimension and settings are the wrapped encoder's, so index identities are unchanged.
    Thread-safe; keeps at most ``MAX_CACHED`` query vectors."""

    MAX_CACHED = 4096

    def __init__(self, inner: Encoder) -> None:
        self._inner = inner
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def inner(self) -> Encoder:
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def __getattr__(self, attr: str) -> Any:  # settings, usage, ...: the wrapped encoder's
        return getattr(self._inner, attr)

    def cached(self, text: str) -> bool:
        with self._lock:
            return text in self._cache

    def _remember(self, texts: Sequence[str], vectors: Sequence[list[float]]) -> None:
        with self._lock:
            for text, vector in zip(texts, vectors, strict=True):
                self._cache[text] = list(vector)
                self._cache.move_to_end(text)
            while len(self._cache) > self.MAX_CACHED:
                self._cache.popitem(last=False)

    def prefetch(self, texts: Sequence[str]) -> int:
        """Embed every query in `texts` not yet in memory, in one call; the number of
        calls made (0 or 1)."""
        with self._lock:
            missing = list(dict.fromkeys(t for t in texts if t and t not in self._cache))
        if not missing:
            return 0
        self._remember(missing, self._inner.encode(missing, is_query=True))
        return 1

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        if is_query:
            with self._lock:
                found = [self._cache.get(t) for t in texts]
            if all(v is not None for v in found):
                return [list(v) for v in found if v is not None]
        vectors = self._inner.encode(texts, is_query=is_query)
        if is_query:
            self._remember(texts, vectors)
        return vectors


@dataclass
class OntologyIndex:
    slug: str
    store: TargetStore
    retrievers: list[Retriever]  # BM25 first, then dense when available
    status: dict[str, RetrieverStatus]  # {"bm25": ..., "dense": ...}
    # The dense retriever's query encoder (shared by every ontology), for batching.
    encoder: PrefetchEncoder | None = None


class IndexSettings(Protocol):
    """What `get_indexes` reads from WS-D's `Settings`."""

    @property
    def cache_dir(self) -> Path: ...

    @property
    def dense_dir(self) -> Path | None: ...

    @property
    def api_key(self) -> str | None: ...

    # The Neon backend (read with defaults, so a settings object without them is bundled):
    # backend, database_url, neon_schema, neon_dimensions, catalog.


def _ok() -> RetrieverStatus:
    return RetrieverStatus(status="ok")


def _unavailable(reason: str) -> RetrieverStatus:
    return RetrieverStatus(status="unavailable", reason=reason)


def open_bm25(records: Sequence[Record], index_dir: Path) -> BM25Retriever:
    """Reuse the index at `index_dir` when its components match `records`; otherwise
    (absent, stale, unreadable) replace it with a fresh build. Serialised by a lock so
    two threads on a cold instance never write the same directory at once."""
    expected = bm25.index_components(records, TEMPLATES, exact_fields=EXACT_FIELDS)
    with _BUILD_LOCK:
        try:
            return BM25Retriever.open(
                index_dir, name=BM25_NAME, expected=expected, default_limit=DEFAULT_LIMIT
            )
        except (OSError, ValueError, KeyError):  # FileNotFoundError, IndexMismatchError, JSON
            pass
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return BM25Retriever.build(
            records,
            TEMPLATES,
            index_dir,
            name=BM25_NAME,
            exact_fields=EXACT_FIELDS,
            default_limit=DEFAULT_LIMIT,
        )


def _display(index_dir: Path, dense_dir: Path) -> str:
    """The index path relative to the dense dir's parent (``indexes/<slug>/dense`` for the
    default layout), so a reason never exposes the function's absolute paths."""
    try:
        return index_dir.relative_to(dense_dir.parent).as_posix()
    except ValueError:
        return index_dir.as_posix()


def open_dense(
    slug: str,
    records: Sequence[Record],
    *,
    dense_dir: Path | None,
    encoder: Callable[[], Encoder] | None,
) -> tuple[DenseRetriever | None, RetrieverStatus]:
    """The dense retriever and its status; `(None, unavailable(reason))` instead of raising.

    `encoder` supplies the query encoder; None means there is no key to embed with."""
    if dense_dir is None:
        return None, _unavailable("no dense index directory configured (NER_SITE_DENSE_DIR)")
    index_dir = dense_dir / slug / "dense"
    shown = _display(index_dir, dense_dir)
    if not (index_dir / "xwalk_meta.json").exists():
        return None, _unavailable(f"no index at {shown}; run {BUILD_TOOL}")
    if encoder is None:
        return None, _unavailable(
            "OPENROUTER_API_KEY is not set; query embeddings need it (BM25 only)"
        )
    try:
        live = encoder()
        expected = dense.index_components(records, TEMPLATES, live, name_fields=NAME_FIELDS)
        retriever = DenseRetriever.open(
            index_dir, live, name=DENSE_NAME, default_limit=DEFAULT_LIMIT, expected=expected
        )
    except IndexMismatchError as exc:
        differs = ", ".join(sorted(exc.differences)) or "unknown"
        return None, _unavailable(
            f"index at {shown} is stale (differs in: {differs}); rebuild it with {BUILD_TOOL}"
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, _unavailable(
            f"index at {shown} could not be opened ({type(exc).__name__}: {exc}); "
            f"rebuild it with {BUILD_TOOL}"
        )
    return retriever, _ok()


def open_indexes(
    *,
    cache_dir: Path,
    dense_dir: Path | None,
    api_key: str | None,
    encoder_factory: Callable[[], Encoder] | None = None,
) -> Mapping[str, OntologyIndex]:
    """One `OntologyIndex` per routed ontology, in `ROUTES` order. Never raises for
    an unavailable retriever; see each index's `status`.

    The four dense retrievers share one query encoder (one connection pool), created on
    first need: from `encoder_factory` when given (tests, offline use; no key needed),
    otherwise the default Qwen3 encoder with `api_key`."""
    shared: list[PrefetchEncoder] = []

    def shared_encoder() -> Encoder:
        if not shared:
            shared.append(
                PrefetchEncoder(
                    encoder_factory()
                    if encoder_factory is not None
                    else default_embeddings_encoder(api_key=api_key, **QUERY_ENCODER_SETTINGS)
                )
            )
        return shared[0]

    have_encoder = encoder_factory is not None or bool(api_key)
    opened: dict[str, OntologyIndex] = {}
    for slug in ROUTES.values():
        records = load_terms(slug)
        store = MemoryStore.from_source(records)
        retrievers: list[Retriever] = []
        status: dict[str, RetrieverStatus] = {}
        bm25_dir = Path(cache_dir) / slug / "bm25"
        try:
            retrievers.append(open_bm25(records, bm25_dir))
            status["bm25"] = _ok()
        except Exception as exc:  # noqa: BLE001 - reported, and dense may still serve
            status["bm25"] = _unavailable(
                f"BM25 index could not be built ({type(exc).__name__}: {exc})"
            )
        retriever, status["dense"] = open_dense(
            slug,
            records,
            dense_dir=dense_dir,
            encoder=shared_encoder if have_encoder else None,
        )
        if retriever is not None:
            retrievers.append(retriever)
        encoder = shared[0] if retriever is not None and shared else None
        opened[slug] = OntologyIndex(slug, store, retrievers, status, encoder)
    return opened


def make_executor(database_url: str) -> SQLExecutor:
    """The executor the Neon backend queries with (HTTPS; works on Vercel and anywhere
    TCP to Postgres is blocked). Tests replace it."""
    return NeonHTTPExecutor(database_url, timeout=NEON_TIMEOUT_S, max_retries=NEON_RETRIES)


def _neon_reason(slug: str, exc: Exception) -> str:
    if isinstance(exc, IndexNotReadyError):
        if exc.state == "missing":
            return f"no neon index for {slug} (schema {exc.schema}); run {LOAD_TOOL}"
        return (
            f"neon index for {slug} is incomplete (status {exc.state}: a load is running or "
            f"was interrupted); rerun {LOAD_TOOL}"
        )
    if isinstance(exc, IndexMismatchError):
        differs = ", ".join(sorted(exc.differences)) or "unknown"
        return f"neon index for {slug} is stale (differs in: {differs}); reload it with {LOAD_TOOL}"
    if isinstance(exc, PostgresError):
        return f"{UNREACHABLE} ({exc})"
    return f"neon index for {slug} could not be opened ({type(exc).__name__}: {exc})"


def _all_unavailable(slugs: Sequence[str], reason: str) -> dict[str, OntologyIndex]:
    return {
        slug: OntologyIndex(
            slug,
            MemoryStore.from_source([]),
            [],
            {"bm25": _unavailable(reason), "dense": _unavailable(reason)},
        )
        for slug in slugs
    }


def open_neon_indexes(
    *,
    executor: SQLExecutor,
    api_key: str | None,
    schema: str = NEON_SCHEMA,
    dimensions: int = NEON_DIMENSIONS,
    catalog: Catalog = "samples",
    encoder_factory: Callable[[], Encoder] | None = None,
) -> Mapping[str, OntologyIndex]:
    """One `OntologyIndex` per routed ontology from Postgres; never raises for an
    unavailable index (see `status`). One query reads every stored identity.

    The samples are checked against the bundled records (digest included). The full
    catalogs are not shipped with the site, so their configuration is checked (engine,
    doc template, exact and name fields, encoder) and the loader's record digest is
    taken as the identity (``verify_records=False``)."""
    slugs = list((ROUTES if catalog == "samples" else FULL_ROUTES).values())
    try:
        meta: Mapping[tuple[str, str], IndexMeta] = read_index_meta(executor, schema, slugs)
    except Exception as exc:  # noqa: BLE001 - reported as the reason for every ontology
        return _all_unavailable(slugs, _neon_reason("", exc))

    shared: list[PrefetchEncoder] = []

    def query_encoder() -> Encoder:
        if not shared:
            shared.append(
                PrefetchEncoder(
                    encoder_factory()
                    if encoder_factory is not None
                    else default_embeddings_encoder(
                        api_key=api_key, dimensions=dimensions, **QUERY_ENCODER_SETTINGS
                    )
                )
            )
        return shared[0]

    have_encoder = encoder_factory is not None or bool(api_key)
    verify = catalog == "samples"
    opened: dict[str, OntologyIndex] = {}
    for slug in slugs:
        records: list[Record] = load_terms(slug) if verify else []
        common: dict[str, Any] = {"schema": schema, "verify_records": verify, "meta": meta}
        status: dict[str, RetrieverStatus] = {}
        retrievers: list[Retriever] = []
        try:
            store: TargetStore = PostgresStore.open(
                executor, slug, expected=terms_components(records), **common
            )
        except Exception as exc:  # noqa: BLE001 - no records, so neither retriever can serve
            reason = _neon_reason(slug, exc)
            opened[slug] = OntologyIndex(
                slug,
                MemoryStore.from_source([]),
                [],
                {"bm25": _unavailable(reason), "dense": _unavailable(reason)},
            )
            continue
        assert isinstance(store, PostgresStore)
        try:
            retrievers.append(
                PostgresBM25Retriever.open(
                    executor,
                    slug,
                    name=BM25_NAME,
                    expected=bm25_components(records, TEMPLATES, exact_fields=EXACT_FIELDS),
                    default_limit=DEFAULT_LIMIT,
                    store=store,
                    **common,
                )
            )
            status["bm25"] = _ok()
        except Exception as exc:  # noqa: BLE001
            status["bm25"] = _unavailable(_neon_reason(slug, exc))
        if not have_encoder:
            status["dense"] = _unavailable(
                "OPENROUTER_API_KEY is not set; query embeddings need it (BM25 only)"
            )
        else:
            try:
                live = query_encoder()
                retrievers.append(
                    PgVectorRetriever.open(
                        executor,
                        slug,
                        live,
                        name=DENSE_NAME,
                        expected=pgvector_components(
                            records, TEMPLATES, live, name_fields=NAME_FIELDS
                        ),
                        default_limit=DEFAULT_LIMIT,
                        store=store,
                        **common,
                    )
                )
                status["dense"] = _ok()
            except Exception as exc:  # noqa: BLE001
                status["dense"] = _unavailable(_neon_reason(slug, exc))
        encoder = shared[0] if status.get("dense") == _ok() and shared else None
        opened[slug] = OntologyIndex(slug, store, retrievers, status, encoder)
    return opened


def _transient(indexes: Mapping[str, OntologyIndex]) -> bool:
    """A database that could not be reached: worth asking again on the next request."""
    return any(
        (s.reason or "").startswith(UNREACHABLE)
        for index in indexes.values()
        for s in index.status.values()
    )


def get_indexes(settings: IndexSettings) -> Mapping[str, OntologyIndex]:
    """`open_indexes` (or, with ``backend == "neon"``, `open_neon_indexes`) for
    `settings`, opened once per process (a warm instance reuses it). Thread-safe:
    concurrent first calls build once. A Neon open that could not reach the database is
    not remembered. `reset()` clears it for tests."""
    backend = getattr(settings, "backend", "bundled")
    if backend == "neon":
        return _get_neon(settings)
    dense_dir = settings.dense_dir
    key: tuple[str | int | None, ...] = (
        "bundled",
        str(settings.cache_dir),
        None if dense_dir is None else str(dense_dir),
        settings.api_key,
    )
    with _MEMO_LOCK:
        found = _MEMO.get(key)
        if found is None:
            found = open_indexes(
                cache_dir=Path(settings.cache_dir),
                dense_dir=None if dense_dir is None else Path(dense_dir),
                api_key=settings.api_key,
            )
            _MEMO[key] = found
        return found


def _get_neon(settings: IndexSettings) -> Mapping[str, OntologyIndex]:
    database_url: str | None = getattr(settings, "database_url", None)
    schema: str = getattr(settings, "neon_schema", NEON_SCHEMA)
    dimensions: int = getattr(settings, "neon_dimensions", NEON_DIMENSIONS)
    catalog: Catalog = getattr(settings, "catalog", "samples")
    key: tuple[str | int | None, ...] = (
        "neon",
        database_url,
        schema,
        dimensions,
        catalog,
        settings.api_key,
    )
    with _MEMO_LOCK:
        found = _MEMO.get(key)
        if found is not None:
            return found
        if not database_url:
            slugs = list((ROUTES if catalog == "samples" else FULL_ROUTES).values())
            found = _all_unavailable(slugs, NO_DATABASE)
        else:
            found = open_neon_indexes(
                executor=make_executor(database_url),
                api_key=settings.api_key,
                schema=schema,
                dimensions=dimensions,
                catalog=catalog,
            )
        if not _transient(found):
            _MEMO[key] = found
        return found


def reset() -> None:
    with _MEMO_LOCK:
        _MEMO.clear()


def summarise_status(indexes: Mapping[str, OntologyIndex]) -> dict[str, RetrieverStatus]:
    """The site-wide ``retrieval`` block for ``/api/health`` and responses: a retriever
    is ``ok`` only when it is ok for every ontology; otherwise ``unavailable`` with the
    failing ontologies' reasons as ``"<slug>: <reason>"`` joined by `` | ``."""
    if not indexes:
        reason = "no ontology indexes are loaded"
        return {"bm25": _unavailable(reason), "dense": _unavailable(reason)}
    summary: dict[str, RetrieverStatus] = {}
    for kind in ("bm25", "dense"):
        failing: list[str] = []
        for slug, index in indexes.items():
            status = index.status.get(kind)
            if status is None or status.status != "ok":
                why = status.reason if status is not None else None
                failing.append(f"{slug}: {why or 'unavailable'}")
        summary[kind] = _unavailable(" | ".join(failing)) if failing else _ok()
    return summary


__all__ = [
    "BM25_NAME",
    "DENSE_NAME",
    "EXACT_FIELDS",
    "LOAD_TOOL",
    "NAME_FIELDS",
    "NEON_SCHEMA",
    "IndexSettings",
    "OntologyIndex",
    "get_indexes",
    "make_executor",
    "open_neon_indexes",
    "open_bm25",
    "open_dense",
    "open_indexes",
    "reset",
    "summarise_status",
]

"""Per-ontology indexes: BM25 built at cold start into the cache dir, dense opened from
the committed indexes, and an honest status for each (docs/ner-site/PLAN.md section 5).

Nothing here raises for a missing, stale or unreadable dense index or a missing key:
the ontology runs on BM25 alone and its ``status["dense"]`` says why, in words an
operator can act on. A BM25 build that fails is reported the same way.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ner_site.ontologies import ROUTES, TEMPLATES, load_terms
from ner_site.schema import RetrieverStatus
from xwalk.records import Record
from xwalk.retrieval import bm25, dense
from xwalk.retrieval.base import IndexMismatchError, Retriever
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.dense import DenseRetriever, Encoder
from xwalk.retrieval.embeddings_api import default_embeddings_encoder
from xwalk.stores.memory import MemoryStore

BM25_NAME = "bm25"
# The key of this retriever in every match's `retrievers`; fixed on build and on open.
DENSE_NAME = "dense-qwen3"
EXACT_FIELDS = ("label", "synonyms")
NAME_FIELDS = ("label", "synonyms")
DEFAULT_LIMIT = 20
BUILD_TOOL = "ner-site/tools/build_indexes.py"

# Query embeddings at request time: a short timeout and no retry, so a slow or
# rate-limited provider degrades a request (BM25 only) instead of eating the function's
# whole duration. With no retry the encoder never sleeps, so a 429 with a long
# Retry-After (which it would otherwise honour for up to 300 s) fails at once. Neither
# setting changes what a vector means, so an index built with the defaults still matches.
QUERY_ENCODER_SETTINGS: dict[str, Any] = {"timeout": 15.0, "max_retries": 0}

_BUILD_LOCK = threading.Lock()
_MEMO_LOCK = threading.Lock()
_MEMO: dict[tuple[str, str | None, str | None], Mapping[str, OntologyIndex]] = {}


@dataclass
class OntologyIndex:
    slug: str
    store: MemoryStore
    retrievers: list[Retriever]  # BM25 first, then dense when available
    status: dict[str, RetrieverStatus]  # {"bm25": ..., "dense": ...}


class IndexSettings(Protocol):
    """What `get_indexes` reads from WS-D's `Settings`."""

    @property
    def cache_dir(self) -> Path: ...

    @property
    def dense_dir(self) -> Path | None: ...

    @property
    def api_key(self) -> str | None: ...


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
    shared: list[Encoder] = []

    def shared_encoder() -> Encoder:
        if not shared:
            shared.append(
                encoder_factory()
                if encoder_factory is not None
                else default_embeddings_encoder(api_key=api_key, **QUERY_ENCODER_SETTINGS)
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
        opened[slug] = OntologyIndex(slug, store, retrievers, status)
    return opened


def get_indexes(settings: IndexSettings) -> Mapping[str, OntologyIndex]:
    """`open_indexes` for `settings`, opened once per process (a warm instance reuses it).
    Thread-safe: concurrent first calls build once. `reset()` clears it for tests."""
    dense_dir = settings.dense_dir
    key = (str(settings.cache_dir), None if dense_dir is None else str(dense_dir), settings.api_key)
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
    "NAME_FIELDS",
    "IndexSettings",
    "OntologyIndex",
    "get_indexes",
    "open_bm25",
    "open_dense",
    "open_indexes",
    "reset",
    "summarise_status",
]

"""Fused retrieval per entity (docs/ner-site/PLAN.md section 5).

Mapping is retrieval only: BM25 and dense fused by reciprocal rank through
``xwalk.retrieve.retrieve``. Match ids only ever come from the ontology's store; no model
text becomes an id. A failing retriever becomes a note on the entity, never an exception.

Embedding requests: pass a `MapStats` and read ``stats.embedding_requests`` afterwards.
It counts the dense searches this call dispatched (one per entity routed to an ontology
whose dense retriever is up, failed and timed-out searches included; a search skipped
because the deadline had passed is not dispatched). Each dense search is one embeddings
request unless the encoder retried, so it is a lower bound on HTTP requests; the encoder
is shared across requests on a warm instance, so its own cumulative `usage` cannot be
attributed to one request.

Time: every retriever call is bounded by `timeout`, and with `deadline` the whole call
is bounded too. A dense (remote) search still running at the deadline is cut off, and one
not yet started is skipped; the entity keeps its BM25 matches and gets the note
``"<retriever>: cut off by the request deadline"``. Local retrievers (BM25) get
`LOCAL_GRACE_S` past the deadline, so entities queued behind slow dense searches still
get their BM25 matches. A cut-off search's thread is abandoned, not joined: it finishes
in the background and its result goes nowhere (see ``ner_site.http``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ner_site.indexes import OntologyIndex
from ner_site.ontologies import TEMPLATES, route, term_url
from ner_site.schema import Entity, EntityResult, Match
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.retrieval.base import Retriever, SearchRequest
from xwalk.retrieval.dense import DenseRetriever
from xwalk.retrieval.postgres import PgVectorRetriever
from xwalk.retrieve import retrieve

OTHER_NOTE = "no ontology for type 'other'"
ALL_FAILED_NOTE = "every retriever failed; no matches"
DEADLINE_NOTE = "cut off by the request deadline"
LOCAL_GRACE_S = 5.0  # what local (BM25) searches may run past the deadline
# How deep the fused ranking is listed: `matches` (the visitor's top_k) plus `more`, up to
# here. A selector (ner_site.select) chooses among the same depth, so its pick is always
# one of the listed candidates.
MAX_LISTED = 30
# `retrieve`'s own timeout is a backstop only: `_Bounded` always fires first.
_BACKSTOP_S = 1.0
# The retrievers that embed the query (one embeddings request per search): the bundled
# index and the Neon one. They are the ones cut off at the deadline and counted.
DENSE_TYPES = (DenseRetriever, PgVectorRetriever)


@dataclass
class MapStats:
    """Filled in by `map_entities`. `warnings` summarises the retriever degradation
    notes across entities as ``"<note> (<entity ids>)"``, first occurrence order.

    `dense_entities` lists, in entity order, every entity whose ontology had a dense
    retriever and that reached the search; `dense_failed` maps those whose dense search
    failed, timed out or was cut off by the deadline to the first such note."""

    embedding_requests: int = 0
    warnings: list[str] = field(default_factory=list)
    # Every routed entity's fused candidates, best first, before the top_k cut: what a
    # selector (``ner_site.select``) chooses among.
    candidates: dict[str, list[Candidate]] = field(default_factory=dict)
    dense_entities: list[str] = field(default_factory=list)
    dense_failed: dict[str, str] = field(default_factory=dict)


class _CutOff(Exception):
    """The request deadline ended (or prevented) a search."""


class _TimedOut(Exception):
    """A retriever call took longer than its own timeout."""


class _Bounded:
    """One retriever for one entity's search, bounded by its timeout and the deadline,
    recording what happened. Same name, depth and hits as the retriever it wraps."""

    def __init__(self, inner: Retriever, *, timeout: float, deadline_at: float | None) -> None:
        self._inner = inner
        self._timeout = timeout
        self._deadline_at = deadline_at
        self.dispatched = False
        self.failure: str | None = None

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    @property
    def default_limit(self) -> int:
        return self._inner.default_limit

    def _fail(self, message: str) -> str:
        self.failure = message
        return message

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        limit = self._timeout
        if self._deadline_at is not None:
            remaining = self._deadline_at - time.monotonic()
            if remaining <= 0:
                raise _CutOff(self._fail(DEADLINE_NOTE))
            limit = min(limit, remaining)
        self.dispatched = True
        try:
            return await asyncio.wait_for(self._inner.search(request), timeout=limit)
        except asyncio.TimeoutError:  # builtins.TimeoutError from 3.11; distinct on 3.10
            if limit < self._timeout:
                raise _CutOff(self._fail(DEADLINE_NOTE)) from None
            raise _TimedOut(self._fail(f"timed out after {self._timeout:g}s")) from None
        except Exception as exc:
            self._fail(str(exc))
            raise


def _result(
    entity: Entity, ontology: str | None, matches: list[Match], notes: list[str]
) -> EntityResult:
    return EntityResult(**entity.model_dump(), ontology=ontology, matches=matches, notes=notes)


async def map_entities(
    entities: Sequence[Entity],
    indexes: Mapping[str, OntologyIndex],
    *,
    top_k: int,
    concurrency: int = 4,
    timeout: float = 30.0,
    deadline: float | None = None,
    stats: MapStats | None = None,
    clean: Callable[[str], str] | None = None,
    routes: Mapping[str, str] | None = None,
) -> list[EntityResult]:
    """One `EntityResult` per entity, in the order given. `timeout` bounds each
    retriever call; `deadline` (seconds from now) bounds the whole call (see the module
    docstring); at most `concurrency` entities are retrieved at once. `clean` is applied
    to every retriever note before it reaches an entity, a warning or `stats` (the
    service scrubs secrets and caps the length there). `routes` maps entity types to
    ontology slugs (``ontologies.ROUTES`` by default; the full catalogs use
    ``FULL_ROUTES``)."""
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    stats = stats if stats is not None else MapStats()
    tidy = clean if clean is not None else (lambda note: note)
    deadline_at = None if deadline is None else time.monotonic() + deadline
    semaphore = asyncio.Semaphore(concurrency)
    degraded: dict[str, list[str]] = {}  # note -> entity ids
    dense_seen: set[str] = set()

    async def one(entity: Entity) -> EntityResult:
        slug = route(entity.type, routes)
        if slug is None:
            return _result(entity, None, [], [f"no ontology for type '{entity.type}'"])
        index = indexes.get(slug)
        if index is None:
            return _result(entity, slug, [], [f"ontology {slug!r} is not loaded"])
        if not index.retrievers:
            return _result(entity, slug, [], [f"no retriever is available for {slug!r}"])
        source = Record(entity.id, {"mention": entity.text})
        query = TEMPLATES.render_query(source)
        if not query:
            return _result(entity, slug, [], ["empty mention; nothing to search"])
        bounded = [
            _Bounded(
                r,
                timeout=timeout,
                deadline_at=(
                    deadline_at
                    if deadline_at is None or isinstance(r, DENSE_TYPES)
                    else deadline_at + LOCAL_GRACE_S
                ),
            )
            for r in index.retrievers
        ]
        dense = [
            b for b, r in zip(bounded, index.retrievers, strict=True) if isinstance(r, DENSE_TYPES)
        ]
        async with semaphore:
            try:
                candidates, notes, all_failed = await retrieve(
                    [query], source, bounded, index.store, timeout=timeout + _BACKSTOP_S
                )
            except Exception as exc:  # noqa: BLE001 - a note, never a 5xx
                candidates, notes, all_failed = [], [f"retrieval failed: {exc}"], True
        stats.embedding_requests += sum(b.dispatched for b in dense)
        if dense:
            dense_seen.add(entity.id)
            failed = next((b for b in dense if b.failure is not None), None)
            if failed is not None:
                stats.dense_failed[entity.id] = tidy(f"{failed.name}: {failed.failure}")
        stats.candidates[entity.id] = list(candidates)
        notes = [tidy(note) for note in notes]
        for note in notes:
            degraded.setdefault(note, []).append(entity.id)
        entity_notes = list(notes)
        if all_failed:
            entity_notes.append(ALL_FAILED_NOTE)
        listed = [
            Match(
                id=c.id,
                label=str(c.record.fields.get("label", "")),
                score=c.fused_score,
                retrievers={h.retriever: h.rank for h in c.evidence},
                url=term_url(c.id),
            )
            for c in candidates[: max(top_k, MAX_LISTED)]
        ]
        result = _result(entity, slug, listed[:top_k], entity_notes)
        return result.model_copy(update={"more": listed[top_k:]})

    results = list(await asyncio.gather(*(one(e) for e in entities)))
    order = {e.id: i for i, e in enumerate(entities)}
    stats.dense_entities = [e.id for e in entities if e.id in dense_seen]
    for note, ids in degraded.items():
        ids.sort(key=lambda eid: order.get(eid, 0))
        stats.warnings.append(f"{note} ({', '.join(ids)})")
    return results


__all__ = [
    "ALL_FAILED_NOTE",
    "DEADLINE_NOTE",
    "DENSE_TYPES",
    "LOCAL_GRACE_S",
    "OTHER_NOTE",
    "MapStats",
    "map_entities",
]

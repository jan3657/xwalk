"""Fused retrieval per entity (docs/ner-site/PLAN.md section 5).

Mapping is retrieval only: BM25 and dense fused by reciprocal rank through
``xwalk.retrieve.retrieve``. Match ids only ever come from the ontology's store; no model
text becomes an id. A failing retriever becomes a note on the entity, never an exception.

Embedding requests: pass a `MapStats` and read ``stats.embedding_requests`` afterwards.
It counts the dense searches this call dispatched (one per entity routed to an ontology
whose dense retriever is up, failed and timed-out searches included). Each dense search
is one embeddings request unless the encoder retried, so it is a lower bound on HTTP
requests; the encoder is shared across requests on a warm instance, so its own
cumulative `usage` cannot be attributed to one request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ner_site.indexes import OntologyIndex
from ner_site.ontologies import TEMPLATES, route, term_url
from ner_site.schema import Entity, EntityResult, Match
from xwalk.records import Record
from xwalk.retrieval.dense import DenseRetriever
from xwalk.retrieve import retrieve

OTHER_NOTE = "no ontology for type 'other'"
ALL_FAILED_NOTE = "every retriever failed; no matches"


@dataclass
class MapStats:
    """Filled in by `map_entities`. `warnings` summarises the retriever degradation
    notes across entities as ``"<note> (<entity ids>)"``, first occurrence order."""

    embedding_requests: int = 0
    warnings: list[str] = field(default_factory=list)


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
    stats: MapStats | None = None,
) -> list[EntityResult]:
    """One `EntityResult` per entity, in the order given. `timeout` bounds each
    retriever call; at most `concurrency` entities are retrieved at once."""
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    stats = stats if stats is not None else MapStats()
    semaphore = asyncio.Semaphore(concurrency)
    degraded: dict[str, list[str]] = {}  # note -> entity ids

    async def one(entity: Entity) -> EntityResult:
        slug = route(entity.type)
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
        async with semaphore:
            stats.embedding_requests += sum(isinstance(r, DenseRetriever) for r in index.retrievers)
            try:
                candidates, notes, all_failed = await retrieve(
                    [query], source, index.retrievers, index.store, timeout=timeout
                )
            except Exception as exc:  # noqa: BLE001 - a note, never a 5xx
                candidates, notes, all_failed = [], [f"retrieval failed: {exc}"], True
        for note in notes:
            degraded.setdefault(note, []).append(entity.id)
        entity_notes = list(notes)
        if all_failed:
            entity_notes.append(ALL_FAILED_NOTE)
        matches = [
            Match(
                id=c.id,
                label=str(c.record.fields.get("label", "")),
                score=c.fused_score,
                retrievers={h.retriever: h.rank for h in c.evidence},
                url=term_url(c.id),
            )
            for c in candidates[:top_k]
        ]
        return _result(entity, slug, matches, entity_notes)

    results = list(await asyncio.gather(*(one(e) for e in entities)))
    order = {e.id: i for i, e in enumerate(entities)}
    for note, ids in degraded.items():
        ids.sort(key=lambda eid: order.get(eid, 0))
        stats.warnings.append(f"{note} ({', '.join(ids)})")
    return results

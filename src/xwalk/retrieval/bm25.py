"""BM25 retrieval over Tantivy.

Tantivy is a hard dependency with a declared support matrix (manylinux x86-64 and
aarch64, macOS arm64, Windows x86-64). There is deliberately no automatic fallback to
another engine: a silent switch would change ranking behaviour and quietly undermine
reproducibility. On an unsupported platform the user selects an alternate retriever
explicitly.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import tantivy

from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest
from xwalk.templates import TemplateSet

_META_FILE = "xwalk_meta.json"

# Tantivy's query parser gives these characters meaning. Real mentions contain them —
# "glucose (D-)", "IL-2", "5-HT2A", "Ca2+" — so they are stripped rather than escaped.
# BM25 over a tokenised text field does not need the operators.
_QUERY_SYNTAX = re.compile(r"[+\-!(){}\[\]^\"~*?:\\/]|(?<!\w)(?:AND|OR|NOT|IN)(?!\w)")
_WS = re.compile(r"\s+")


def sanitise_query(text: str) -> str:
    """Reduce arbitrary text to something the query parser cannot misinterpret."""
    return _WS.sub(" ", _QUERY_SYNTAX.sub(" ", text)).strip()


# An exact whole-string hit on a label or synonym must outrank a document that merely
# happens to contain the query term more often. 10.0 is comfortably above any BM25 score
# the tokenised `text` field produces on a collection of this shape.
_EXACT_BOOST = 10.0

# Tantivy's `writer()` defaults are heap_size=128 MB and num_threads=0, where 0 means
# "one indexing thread per core" — and the heap is allocated *per thread*. On a 64-core
# machine that is an 8 GB arena, which costs ~7 s to set up whether the index holds five
# documents or five million. These defaults are deliberately modest; raise them for large
# collections. Tantivy's own floor is 15 MB.
_WRITER_HEAP_BYTES = 50_000_000
_WRITER_THREADS = 1


def _build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("record_id", stored=True)
    # `raw` = no tokenisation: the whole field value is one term, so "glucose" matches
    # the label "glucose" but not the label "beta-D-glucose".
    builder.add_text_field("exact", stored=False, tokenizer_name="raw")
    builder.add_text_field("text", stored=False)
    return builder.build()


def _exact_forms(record: Record, exact_fields: Sequence[str]) -> list[str]:
    """Every whole-string form of a record that should count as an exact match."""
    forms: list[str] = []
    for field_name in exact_fields:
        value = record.fields.get(field_name)
        if value is None:
            continue
        candidates = [value] if isinstance(value, str) else list(value)
        for candidate in candidates:
            text = str(candidate).strip().lower()
            if text and text not in forms:
                forms.append(text)
    return forms


class BM25Retriever:
    """Tantivy-backed BM25, with an opt-in exact-match field. Reuse via `open`."""

    def __init__(
        self,
        index: tantivy.Index,
        *,
        name: str,
        fingerprint: str,
        exact_fields: Sequence[str] = (),
        default_limit: int = 20,
        empty_doc_count: int = 0,
    ) -> None:
        self._index = index
        self._name = name
        self._fingerprint = fingerprint
        self._exact_fields = tuple(exact_fields)
        self._default_limit = default_limit
        self.empty_doc_count = empty_doc_count

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def default_limit(self) -> int:
        return self._default_limit

    @classmethod
    def build(
        cls,
        records: Iterable[Record],
        templates: TemplateSet,
        index_dir: str | Path,
        *,
        name: str = "bm25",
        exact_fields: Sequence[str] = (),
        default_limit: int = 20,
        writer_heap_bytes: int = _WRITER_HEAP_BYTES,
        writer_threads: int = _WRITER_THREADS,
    ) -> BM25Retriever:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        index = tantivy.Index(_build_schema(), path=str(index_dir))
        writer = index.writer(heap_size=writer_heap_bytes, num_threads=writer_threads)
        # `tantivy.Index` on an existing path opens it, and the writer appends. Building
        # is a replace, not an append: `xwalk match` rebuilds into `<out>/index` on every
        # invocation, so appending would duplicate every document on the second run --
        # and the fingerprint hashes records rather than the index, so nothing downstream
        # would notice. Duplicates then spend the retrieval limit, costing recall.
        writer.delete_all_documents()

        digests: list[str] = []
        empty = 0
        for record in records:
            text = templates.render_doc(record)
            if not text:
                empty += 1
            document = tantivy.Document(record_id=record.id, text=text)
            for form in _exact_forms(record, exact_fields):
                document.add_text("exact", form)
            writer.add_document(document)
            digests.append(hash_record(record))
        writer.commit()
        index.reload()

        fingerprint = hash_value(
            {
                "engine": "tantivy-bm25",
                "doc_template": templates.doc,
                "exact_fields": list(exact_fields),
                "records": sorted(digests),
            }
        )
        (index_dir / _META_FILE).write_text(
            json.dumps(
                {
                    "engine": "tantivy-bm25",
                    "name": name,
                    "fingerprint": fingerprint,
                    "exact_fields": list(exact_fields),
                    "default_limit": default_limit,
                    "doc_count": len(digests),
                    "empty_doc_count": empty,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(
            index,
            name=name,
            fingerprint=fingerprint,
            exact_fields=exact_fields,
            default_limit=default_limit,
            empty_doc_count=empty,
        )

    @classmethod
    def open(cls, index_dir: str | Path, *, name: str | None = None) -> BM25Retriever:
        """Reopen a built index. `name` overrides the one recorded at build time."""
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{index_dir} is not an xwalk BM25 index (no {_META_FILE}); build it first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = tantivy.Index(_build_schema(), path=str(index_dir))
        index.reload()
        return cls(
            index,
            name=name or meta["name"],
            fingerprint=meta["fingerprint"],
            exact_fields=meta.get("exact_fields", ()),
            default_limit=meta.get("default_limit", 20),
            empty_doc_count=meta.get("empty_doc_count", 0),
        )

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        cleaned = sanitise_query(text)
        if not cleaned:
            return []
        searcher = self._index.searcher()
        text_query = self._index.parse_query(cleaned, ["text"])
        if self._exact_fields:
            exact_query = tantivy.Query.boost_query(
                tantivy.Query.term_query(self._index.schema, "exact", cleaned.lower()),
                _EXACT_BOOST,
            )
            query = tantivy.Query.boolean_query(
                [
                    (tantivy.Occur.Should, exact_query),
                    (tantivy.Occur.Should, text_query),
                ]
            )
        else:
            query = text_query
        result = searcher.search(query, limit)
        hits: list[RetrievalHit] = []
        for rank, (score, address) in enumerate(result.hits, start=1):
            doc = searcher.doc(address)
            record_id = doc["record_id"][0]
            hits.append(
                RetrievalHit(
                    record_id=record_id,
                    retriever=self._name,
                    raw_score=float(score),
                    rank=rank,
                )
            )
        return hits

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        # Tantivy's search is synchronous CPU work; a thread keeps the event loop free
        # so N retrievers genuinely run concurrently.
        return await asyncio.to_thread(self._search_sync, request.text, request.limit)

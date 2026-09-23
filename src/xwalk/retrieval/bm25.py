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
_QUERY_SYNTAX = re.compile(r"[`+\-!(){}\[\]^\"~*?:\\/]|(?<!\w)(?:AND|OR|NOT|IN)(?!\w)")
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

# `default` is tantivy's own tokenizer for the `text` field (split, lower-case) and is
# what every index built before this option existed uses. `en_stem` adds ASCII folding and
# English stemming, so "anesthetics" meets "anesthetic".
ANALYZERS = ("default", "en_stem")

# A fuzzy clause widens recall for spelling variants ("anaesthetic" / "anesthetic"), but an
# edit-distance match is weaker evidence than the term itself, and short terms are too
# close to too many others to be worth expanding.
_FUZZY_BOOST = 0.5
_FUZZY_MIN_LENGTH = 5


def _en_stem() -> tantivy.TextAnalyzer:
    return (
        tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
        .filter(tantivy.Filter.lowercase())
        .filter(tantivy.Filter.ascii_fold())
        .filter(tantivy.Filter.stemmer("english"))
        .build()
    )


def _open_index(index_dir: Path, analyzer: str) -> tantivy.Index:
    """Open (or create) the index, registering the analyzer before any writer or query
    parser needs it. `default` is built in and needs no registration."""
    index = tantivy.Index(_build_schema(analyzer), path=str(index_dir))
    if analyzer == "en_stem":
        index.register_tokenizer("en_stem", _en_stem())
    return index


def _build_schema(analyzer: str = "default") -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("record_id", stored=True)
    # `raw` = no tokenisation: the whole field value is one term, so "glucose" matches
    # the label "glucose" but not the label "beta-D-glucose".
    builder.add_text_field("exact", stored=False, tokenizer_name="raw")
    builder.add_text_field("text", stored=False, tokenizer_name=analyzer)
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
        analyzer: str = "default",
        fuzzy_distance: int = 0,
    ) -> None:
        self._index = index
        self._name = name
        self._fingerprint = fingerprint
        self._exact_fields = tuple(exact_fields)
        self._default_limit = default_limit
        self.empty_doc_count = empty_doc_count
        self._fuzzy_distance = fuzzy_distance
        # Fuzzy terms bypass the query parser, so they must be put into the index's term
        # space by hand: with `en_stem` the index holds "anesthet", not "anesthetics".
        self._term_analyzer = _en_stem() if analyzer == "en_stem" else None

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
        analyzer: str = "default",
        fuzzy_distance: int = 0,
        writer_heap_bytes: int = _WRITER_HEAP_BYTES,
        writer_threads: int = _WRITER_THREADS,
    ) -> BM25Retriever:
        if analyzer not in ANALYZERS:
            raise ValueError(f"unknown BM25 analyzer {analyzer!r}; expected one of {ANALYZERS}")
        if fuzzy_distance < 0:
            raise ValueError(f"fuzzy_distance must be 0 or more, got {fuzzy_distance}")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        index = _open_index(index_dir, analyzer)
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

        identity: dict[str, object] = {
            "engine": "tantivy-bm25",
            "doc_template": templates.doc,
            "exact_fields": list(exact_fields),
            "records": sorted(digests),
        }
        # Only non-default options enter the hash, so a default index keeps the
        # fingerprint it had before these options existed.
        if analyzer != "default":
            identity["analyzer"] = analyzer
        if fuzzy_distance:
            identity["fuzzy_distance"] = fuzzy_distance
        fingerprint = hash_value(identity)
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
                    "analyzer": analyzer,
                    "fuzzy_distance": fuzzy_distance,
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
            analyzer=analyzer,
            fuzzy_distance=fuzzy_distance,
        )

    @classmethod
    def open(
        cls,
        index_dir: str | Path,
        *,
        name: str | None = None,
        analyzer: str | None = None,
        fuzzy_distance: int | None = None,
    ) -> BM25Retriever:
        """Reopen a built index. `name` overrides the one recorded at build time.

        `analyzer` and `fuzzy_distance`, when given, are what the caller expects; an index
        built with different ones is refused rather than searched with the wrong settings.
        An index whose metadata predates these options was built with the defaults.
        """
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{index_dir} is not an xwalk BM25 index (no {_META_FILE}); build it first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        built_analyzer = meta.get("analyzer", "default")
        built_fuzzy = meta.get("fuzzy_distance", 0)
        if analyzer is not None and analyzer != built_analyzer:
            raise ValueError(
                f"index at {index_dir} was built with analyzer {built_analyzer!r} but "
                f"analyzer {analyzer!r} was requested; rebuild it"
            )
        if fuzzy_distance is not None and fuzzy_distance != built_fuzzy:
            raise ValueError(
                f"index at {index_dir} was built with fuzzy_distance {built_fuzzy} but "
                f"fuzzy_distance {fuzzy_distance} was requested; rebuild it"
            )
        index = _open_index(index_dir, built_analyzer)
        index.reload()
        return cls(
            index,
            name=name or meta["name"],
            fingerprint=meta["fingerprint"],
            exact_fields=meta.get("exact_fields", ()),
            default_limit=meta.get("default_limit", 20),
            empty_doc_count=meta.get("empty_doc_count", 0),
            analyzer=built_analyzer,
            fuzzy_distance=built_fuzzy,
        )

    def _fuzzy_queries(self, cleaned: str) -> list[tantivy.Query]:
        """One down-weighted edit-distance clause per distinct query term long enough to
        be worth expanding. None unless the index was built with `fuzzy_distance > 0`."""
        if self._fuzzy_distance <= 0:
            return []
        terms: dict[str, None] = {}
        for word in cleaned.lower().split():
            if len(word) < _FUZZY_MIN_LENGTH:
                continue
            analyzed = self._term_analyzer.analyze(word) if self._term_analyzer else [word]
            terms.update(dict.fromkeys(analyzed))
        return [
            tantivy.Query.boost_query(
                tantivy.Query.fuzzy_term_query(
                    self._index.schema,
                    "text",
                    term,
                    distance=self._fuzzy_distance,
                    transposition_cost_one=True,
                ),
                _FUZZY_BOOST,
            )
            for term in terms
        ]

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        cleaned = sanitise_query(text)
        if not cleaned:
            return []
        searcher = self._index.searcher()
        try:
            text_query = self._index.parse_query(cleaned, ["text"])
        except Exception:
            fallback = re.sub(r"[^\w\s]", " ", cleaned).strip()
            text_query = self._index.parse_query(fallback, ["text"])
        clauses: list[tuple[tantivy.Occur, tantivy.Query]] = []
        if self._exact_fields:
            exact_query = tantivy.Query.boost_query(
                tantivy.Query.term_query(self._index.schema, "exact", cleaned.lower()),
                _EXACT_BOOST,
            )
            clauses.append((tantivy.Occur.Should, exact_query))
        clauses.append((tantivy.Occur.Should, text_query))
        clauses.extend((tantivy.Occur.Should, fuzzy) for fuzzy in self._fuzzy_queries(cleaned))
        query = tantivy.Query.boolean_query(clauses) if len(clauses) > 1 else text_query
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

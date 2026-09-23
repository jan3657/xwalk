# Retrieval

Retrieval in xwalk is three separate jobs, kept apart on purpose. A **retriever**
returns record IDs and 1-based ranks — never the records themselves. A **target store**
turns IDs into `Record`s. **Fusion** combines the per-retriever rankings into scored
candidates. This split is what makes "plug in an external search backend for scale"
actually work: the retriever never has to hold your target collection in memory, and
the store never has to know how ranking happened.

Two retrievers ship with xwalk — BM25 over Tantivy and dense retrieval behind an
`Encoder` protocol — plus reciprocal rank fusion to combine them. Anything that speaks
the `Retriever` protocol slots in beside or instead of them.

## The Retriever protocol

`xwalk.retrieval.base.Retriever` is a `runtime_checkable` `Protocol` with four members.
A custom retriever implements exactly these:

```python
class MyRetriever:
    @property
    def name(self) -> str:
        """Stable identifier. Appears in RetrievalHit.retriever and in diagnostics."""

    @property
    def fingerprint(self) -> str:
        """Digest of index content and configuration. Feeds the run fingerprint."""

    @property
    def default_limit(self) -> int:
        """This retriever's own retrieval depth."""

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        """Return hits ranked best-first with 1-based contiguous ranks."""
```

`default_limit` lives on the retriever rather than the matcher because a BM25 index and
a dense index have no reason to share a `k`; the matcher supplies a fallback only. The
matcher reads this property on every attempt — a retriever without it breaks the loop.

`search` takes a request object rather than `search(query, k)` so that filters and
field-aware retrieval can be added without a breaking protocol change. A backend that
ignores `source_record` stays correct.

```python
@dataclass(frozen=True)
class SearchRequest:
    text: str
    limit: int
    filters: Mapping[str, Any] | None = None
    source_record: Record | None = None
```

Each hit is a frozen `RetrievalHit(record_id, retriever, raw_score, rank)`. `rank` is
1-based and validated; `raw_score` may be `None` for backends that have none.

Raise `RetrieverError` when the backend fails. The matcher catches it and degrades to
the remaining retrievers rather than failing the record.

## BM25Retriever

`xwalk.retrieval.bm25.BM25Retriever` is BM25 over Tantivy, with an opt-in exact-match
field. Tantivy is a hard dependency with a declared support matrix (see
[docs/platforms.md](../platforms.md)); there is deliberately no automatic fallback to
another engine, because a silent switch would change ranking and quietly undermine
reproducibility.

### Building and reopening

```python
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
    writer_heap_bytes: int = 50_000_000,
    writer_threads: int = 1,
) -> BM25Retriever
```

```python
@classmethod
def open(
    cls,
    index_dir: str | Path,
    *,
    name: str | None = None,
    analyzer: str | None = None,
    fuzzy_distance: int | None = None,
) -> BM25Retriever
```

`build` renders each record through `templates.render_doc` and indexes it. Records that
render to empty text are still indexed (they remain reachable via `exact_fields`) and
counted in `empty_doc_count` — a non-zero count usually means the `doc` template names
fields your source does not produce.

`open` reopens a built index without re-reading the records; `name` overrides the one
recorded at build time. Opening a directory without `xwalk_meta.json` raises
`FileNotFoundError` telling you to build first. `analyzer` and `fuzzy_distance`, when
passed, must match what the index was built with, or `open` raises `ValueError` naming
both values; left as `None`, the recorded ones are used. An index whose metadata predates
these options is treated as `default` / `0`.

### What is written to disk

`index_dir` holds Tantivy's own index — its `meta.json` plus segment files, all managed
by Tantivy — and one file xwalk adds: `xwalk_meta.json`, recording `engine`
(`"tantivy-bm25"`), `name`, `fingerprint`, `exact_fields`, `default_limit`,
`doc_count`, `empty_doc_count`, `analyzer`, and `fuzzy_distance`. The fingerprint digests
the engine, the `doc` template, `exact_fields`, the sorted per-record hashes, and
`analyzer` and `fuzzy_distance` when they differ from their defaults, so any change to
content or configuration changes the run fingerprint while a default index keeps the
fingerprint it had before those options existed.

### `exact_fields`

An exact whole-string hit on a label or synonym must outrank a document that merely
happens to contain the query term more often. When `exact_fields` is set, every value
of those fields (strings, or each element of a list field) is stripped, lowercased,
de-duplicated, and stored in an untokenised `exact` field — `raw` tokenisation means
the whole field value is one term, so `"glucose"` matches the label `glucose` but not
`beta-D-glucose`. At search time a term query on `exact` is combined with the ordinary
`text` query under a boost of **10.0**, comfortably above any BM25 score the tokenised
`text` field produces on a collection of this shape. Omit `exact_fields` for plain
BM25.

### `analyzer` and `fuzzy_distance`

Both are off by default. `analyzer` picks the tokenizer for the `text` field (the `exact`
field is always `raw`): `default` is Tantivy's own (split, drop tokens over 40 characters, lowercase); `en_stem` is a
simple tokenizer followed by lowercase, ASCII folding and the English stemmer, so
`anesthetics` finds `anesthetic`. `fuzzy_distance` is 0 to 2 (Tantivy's Levenshtein limit;
`build` refuses anything else). Above 0 it adds, for each distinct
whitespace-separated query term of five or more characters, a Levenshtein fuzzy term
query on `text` (transpositions cost 1) boosted by **0.5**, OR-ed with the text query,
so `anaesthetic` also finds `anesthetic`. Fuzzy terms bypass the query parser, so each is
lowercased and, under `en_stem`, run through the same analyzer, landing in the index's
term space (`anesthet`, not `anesthetics`).

### `sanitise_query`

```python
def sanitise_query(text: str) -> str
```

Tantivy's query parser gives `+ - ! ( ) { } [ ] ^ " ~ * ? : \ /` and the bare words
`AND OR NOT IN` meaning. Real mentions contain them — "glucose (D-)", "IL-2", "5-HT2A",
"Ca2+" — so they are **stripped rather than escaped**: BM25 over a tokenised text field
does not need the operators, and escaping would keep parser semantics in play for text
that was never a query language. Every `search` call sanitises its text first; a query
that reduces to nothing returns zero hits without error.

### Writer defaults

Tantivy's own `writer()` defaults are a 128 MB heap and `num_threads=0`, meaning one
indexing thread per core — and the heap is allocated *per thread*. On a 64-core machine
that is an 8 GB arena costing ~7 s to set up whether the index holds five documents or
five million. xwalk's defaults (`writer_heap_bytes=50_000_000`, `writer_threads=1`) are
deliberately modest; raise them for large collections. Tantivy's own floor is 15 MB.

Search itself is synchronous CPU work run via `asyncio.to_thread`, so N retrievers
genuinely run concurrently under the matcher.

## DenseRetriever

`xwalk.retrieval.dense.DenseRetriever` does nearest-neighbour search over encoder
vectors. There is deliberately no "second dense model" concept: running two encoders
means constructing two `DenseRetriever`s and handing both to the matcher.

### The `Encoder` protocol

```python
@runtime_checkable
class Encoder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        """Return one unit-normalised vector per text."""
```

Because the encoder is a protocol, an API embedding service can be plugged in with no
torch installed at all.

### `SentenceTransformerEncoder`

The default encoder. Requires `xwalk[dense]`.

```python
SentenceTransformerEncoder(
    model_name: str,
    *,
    device: str | None = None,
    normalize: bool = True,
    query_prefix: str = "",
    doc_prefix: str = "",
    batch_size: int = 32,
)
```

`name` is the model name; `dimension` comes from the model. `query_prefix` /
`doc_prefix` are prepended per side for models that expect instruction prefixes.

### Building and reopening

```python
@classmethod
def build(
    cls,
    records: Iterable[Record],
    templates: TemplateSet,
    index_dir: str | Path,
    encoder: Encoder,
    *,
    name: str | None = None,
    default_limit: int = 20,
    batch: int = 256,
) -> DenseRetriever
```

```python
@classmethod
def open(
    cls,
    index_dir: str | Path,
    encoder: Encoder,
    *,
    name: str | None = None,
    default_limit: int | None = None,
) -> DenseRetriever
```

`build` encodes rendered docs in batches of `batch` and validates that every vector's
length matches `encoder.dimension`, raising `ValueError` otherwise. When `name` is
omitted it defaults to `dense:{encoder.name}`. Three files are written to `index_dir`:

| File | Contents |
|---|---|
| `vectors.f32` | vectors as packed little-endian float32, one row per record |
| `record_ids.json` | record IDs in row order |
| `xwalk_meta.json` | engine, name, encoder, dimension, limit, fingerprint, doc count |

`open` requires the same encoder the index was built with, because a vector index is
meaningless to a different encoder. It raises `FileNotFoundError` when
`xwalk_meta.json` is absent, `ValueError` when `encoder.name` differs from the recorded
encoder, and `ValueError` when `encoder.dimension` differs from the recorded dimension.
`default_limit=None` keeps the value recorded at build time (falling back to 20).

### FAISS vs pure Python

If `faiss` and `numpy` import, search uses `IndexFlatIP` over the stored vectors. This
is a *speed* fallback with identical semantics on normalised vectors — `IndexFlatIP` is
exact — not a silent ranking change. Without FAISS, search is an exact inner-product
scan in pure Python, sorted by descending score with ascending row index as the
tie-break. An empty or whitespace-only query, or an empty index, returns no hits.
`xwalk[dense]` pulls `faiss-cpu` along with sentence-transformers and torch; only the
encoder strictly needs the extra, so a custom `Encoder` runs FAISS-less on the pure
Python path.

## Fusion

`xwalk.retrieval.fusion.reciprocal_rank_fusion` combines rankings from retrievers whose
raw scores are not comparable — a BM25 score and a cosine similarity have no shared
scale — by using rank position alone.

```python
def reciprocal_rank_fusion(
    hit_groups: Sequence[Sequence[RetrievalHit]],
    store: TargetStore,
    *,
    k: int = 60,
) -> list[Candidate]
```

A record's fused score is the sum of `1 / (k + rank)` over the retrievers that surfaced
it. `k` defaults to **60**; `k <= 0` raises `ValueError`. Each retriever gets one vote
per record — its best rank, by arrival order within its hit group. The result is sorted
by descending fused score, then ascending record ID: a stable, reproducible order.
Within a candidate, evidence hits are sorted by `(retriever, rank)` so the trace is
byte-identical regardless of retriever completion order.

IDs absent from the store are **dropped**, not raised: a stale index may legitimately
name a record that has since been deleted.

## Target stores

`xwalk.stores.base.TargetStore` is a `runtime_checkable` `Protocol` with three members:

```python
class TargetStore(Protocol):
    @property
    def fingerprint(self) -> str:
        """Digest of the snapshot. Feeds the run fingerprint."""

    def get(self, record_id: str) -> Record:
        """Return one record, or raise KeyError."""

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        """Return records in request order, silently skipping unknown IDs."""
```

`get_many` skips unknown IDs rather than raising for the same reason fusion drops them:
a retriever backed by a stale index can name a record that no longer exists.

`xwalk.stores.memory.MemoryStore` is the dict-backed default for collections that fit
in RAM, built eagerly from any iterable of `Record`s:

```python
store = MemoryStore.from_source(records)
```

`from_source` raises `ValueError` on a duplicate record ID — two targets sharing an ID
would make every downstream decision ambiguous. The fingerprint is the hash of the
**sorted** per-record digests, so it depends only on content, never on source order.
`MemoryStore` also supports `len(store)`, iteration over records, and `id in store`.

## Gotchas

- **A query can sanitise to nothing.** `sanitise_query("(+/-)")` is empty, and BM25
  returns zero hits with no error. If a source column is entirely punctuation, expect
  `NO_CANDIDATES`, not a crash.
- **`exact_fields` is fixed at build time.** `open` restores it from `xwalk_meta.json`;
  changing it means rebuilding the index. The exact match is case-insensitive and
  whole-string only.
- **`empty_doc_count > 0` is a template smell.** Those records were indexed but are
  unreachable through the `text` field; check that the `doc` template names fields the
  source actually emits.
- **Indexing defaults are tuned for small collections.** For millions of documents,
  raise `writer_heap_bytes` and `writer_threads`; the shipped defaults trade indexing
  throughput for not allocating a multi-gigabyte arena on many-core machines.
- **Dense `open` is strict about the encoder.** The same model under a different
  `name`, or a different dimension, is rejected. This is deliberate: silently serving
  vectors to the wrong encoder would corrupt ranking without any visible failure.
- **Raw scores never cross retrievers.** `RetrievalHit.raw_score` is a BM25 score on
  one retriever and an inner product on another; fusion uses ranks only, which is why
  boosting inside one retriever (the 10.0 exact boost) affects only that retriever's
  ranking, not its weight in fusion.
- **One vote per retriever in fusion.** Duplicate hits for the same record from the
  same retriever are ignored after the first; feed each retriever's hits best-first,
  as the protocol requires, and the best rank is the one counted.

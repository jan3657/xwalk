# Retrieval

Retrieval in xwalk is three separate jobs, kept apart on purpose. A **retriever**
returns record IDs and 1-based ranks — never the records themselves. A **target store**
turns IDs into `Record`s. **Fusion** combines the per-retriever rankings into scored
candidates. This split is what makes "plug in an external search backend for scale"
actually work: the retriever never has to hold your target collection in memory, and
the store never has to know how ranking happened.

Two retrievers ship with xwalk — BM25 over Tantivy and dense retrieval behind an
`Encoder` protocol — plus reciprocal rank fusion to combine them, and the same pair over
Postgres for collections too large to hold in memory ([Postgres retrieval](#postgres-retrieval)). Anything that speaks
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
    expected: Mapping[str, Any] | None = None,
    default_limit: int | None = None,
    analyzer: str | None = None,
    fuzzy_distance: int | None = None,
) -> BM25Retriever
```

`build` renders each record through `templates.render_doc` and indexes it. Records that
render to empty text are still indexed (they remain reachable via `exact_fields`) and
counted in `empty_doc_count` — a non-zero count usually means the `doc` template names
fields your source does not produce.

`build` replaces whatever the directory held; it never appends (0.1.0 appended, which
duplicated every document on a rebuild).

`open` reopens a built index without re-reading the records; `name` and `default_limit`
override the recorded ones. Opening a directory without `xwalk_meta.json` raises
`FileNotFoundError` telling you to build first. With `expected` — the dict
`index_components(records, templates, exact_fields=..., analyzer=..., fuzzy_distance=...)`
returns, computed without building anything — an index whose stored components differ,
or that predates stored components (built before 0.2), raises `IndexMismatchError` (a
`ValueError` whose `differences` maps each component to `(stored, expected)`) and is left
untouched. `xwalk match` and `xwalk index` always pass `expected` (see the [job
file](job-file.md) for the command-level behaviour). `analyzer` and `fuzzy_distance`,
when passed, must match what the index was built with, or `open` raises `ValueError`
naming both values; left as `None`, the recorded ones are used. An index whose metadata
predates these options is treated as `default` / `0`.

Independently of `expected`, `open` refuses an index whose stored
`normalization_version` is not the current one (or that has no stored components) with
`IndexMismatchError` naming `normalization_version`: its exact forms were written by a
different normaliser, so searching it would silently miss exact hits. Rebuild it with
`build` (or `--rebuild-index` from the CLI). Version 2 (the current one) introduced
`exact_form`; every index built with version 1 must be rebuilt.

### What is written to disk

`index_dir` holds Tantivy's own index — its `meta.json` plus segment files, all managed
by Tantivy — and one file xwalk adds: `xwalk_meta.json`, recording `engine`
(`"tantivy-bm25"`), `name`, `fingerprint`, `components`, `exact_fields`,
`default_limit`, `doc_count`, `empty_doc_count`, `analyzer`, and `fuzzy_distance`.
`components` is the index identity: engine, index format version, normalisation version,
the `doc` template, `exact_fields`, the record count, a digest of the sorted per-record
hashes, and `analyzer` and `fuzzy_distance` when they differ from their defaults. The
fingerprint is the hash of `components`, so any change to content or configuration
changes the run fingerprint, while a default index keeps the identity it had before
those options existed.

### `exact_fields`

An exact whole-string hit on a label or synonym must outrank a document that merely
happens to contain the query term more often. When `exact_fields` is set, every value
of those fields (strings, or each element of a list field) is passed through
`exact_form`, de-duplicated, and stored in an untokenised `exact` field — `raw`
tokenisation means the whole field value is one term, so `"glucose"` matches the label
`glucose` but not `beta-D-glucose`. At search time the **original** query text (not the
sanitised one) goes through the same `exact_form`, and a term query on `exact` is
combined with the ordinary `text` query under a boost of **10.0**. The boost multiplies
a BM25 term score, so its contribution grows with the collection; on collections of
realistic size it outranks any tokenised `text` match. Omit `exact_fields` for plain
BM25.

```python
def exact_form(text: str) -> str
```

The one exact-match normaliser, used for indexed forms and queries alike: Unicode NFC,
`casefold`, strip, and each internal whitespace run collapsed to one space.
Punctuation and accents are kept: `IL-2`, `glucose (D-)` and `Ca2+` match themselves
exactly (in any case), but `IL-2` does not match `IL 2`, `C++` does not match `C`, and
`Gruyere` does not match `Gruyère` (NFC and NFD spellings of `Gruyère` are the same
form). Changing it changes what an index answers, so it is versioned by
`NORMALIZATION_VERSION` in the index components (currently 2; version 1 lowercased the
indexed forms but matched them against the sanitised query, so punctuated labels never
matched exactly).

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
that was never a query language. Every `search` call sanitises its text for the
tokenised `text` query (and the fuzzy terms) only; the exact lookup uses the original
text through `exact_form`. When sanitising empties the query (`IN`, `-`, `( )`) the
exact lookup still runs alone, so a target labelled `IN` is found; without
`exact_fields`, or with no exact hit, such a query returns zero hits without error.

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
torch installed at all; `OpenAIEmbeddingsEncoder` below is the one xwalk ships.

### `SentenceTransformerEncoder`

The default encoder. Requires `xwalk-match[dense]`.

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

### `OpenAIEmbeddingsEncoder`

`xwalk.retrieval.embeddings_api.OpenAIEmbeddingsEncoder` calls `POST {base_url}/embeddings`
on any service that speaks the OpenAI embeddings shape: OpenRouter, OpenAI, Voyage,
DeepInfra, a local vLLM or Ollama. It needs only httpx (a base dependency), so dense
retrieval runs without torch or a downloaded model.

```python
OpenAIEmbeddingsEncoder(
    base_url: str,
    model: str,
    *,
    dimensions: int,
    api_key: str | None = None,
    send_dimensions: bool = True,
    truncate: bool = False,
    normalize: bool = True,
    query_prefix: str = "",
    doc_prefix: str = "",
    query_input_type: str | None = None,
    doc_input_type: str | None = None,
    provider: Mapping[str, Any] | None = None,
    batch_size: int = 64,
    timeout: float = 60.0,
    max_retries: int = 5,
)
```

- `dimensions` is declared, never probed, so computing an index's identity spends no
  call. With `send_dimensions` it is sent as the request's `dimensions` (for models
  that can shorten their output); every returned vector must have exactly that length,
  or the batch fails with `EmbeddingAPIError`. A provider that ignores the parameter is
  caught here instead of silently building an index of another shape. `truncate: true`
  cuts a longer vector to `dimensions` and renormalises it; use it only for models
  trained for truncation (Matryoshka: Qwen3-Embedding, Voyage 4, OpenAI
  `text-embedding-3-*`, Nemotron 3.5 Embed).
- `query_input_type` / `doc_input_type` are sent as `input_type` on each side (for
  example `search_query` / `search_document` on OpenRouter); the prefixes are prepended
  to the text, as for `SentenceTransformerEncoder`.
- `provider` is passed through verbatim as OpenRouter's provider routing preferences
  (`order`, `only`, `allow_fallbacks`, ...). Pinning a provider makes vectors
  reproducible when several serve the same model.
- A blank text becomes a zero vector and is not sent (gateways reject empty strings).
- HTTP 408/409/429/5xx/529, transport errors, a non-JSON body and an HTTP 200 carrying
  a transient `error` object are retried with jittered backoff (`Retry-After` is
  honoured); 400/401/402/403/404 are not. A 413 halves the batch. Failures raise
  `EmbeddingAPIError`, a `RetrieverError`: during matching the attempt degrades to the
  remaining retrievers like any failing retriever.
- `usage` reports requests, texts, prompt tokens and cost as the endpoint stated them;
  one response without a count makes that total `None` (unknown), never zero.
  `xwalk index` prints what each build spent.

`default_embeddings_encoder(api_key=...)` returns the default encoder a job gets with
`encoder: openai_embeddings` and no `model`: Qwen3-Embedding 8B through OpenRouter at
1024 dimensions with its query instruction, preferring the DeepInfra provider with
fallback (`DEFAULT_MODEL`, `DEFAULT_BASE_URL`, `DEFAULT_DIMENSIONS`,
`DEFAULT_QUERY_PREFIX`, `DEFAULT_PROVIDER` in the same module). Keyword arguments
override any setting. `probe_dimension()` spends one call to report the native vector
length; nothing else probes.

Identity (`settings`) covers the redacted base URL, `send_dimensions`, `truncate`,
`normalize`, both prefixes, both input types and `provider`, plus `adapter:
"openai_embeddings"` so a local and a remote model of the same name never share an
index. The key, batch size, timeout and retries are not identity. The remote model
cannot be pinned, so `revision` is `"unknown"`.

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
    name_fields: Sequence[str] | None = None,
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
    expected: Mapping[str, Any] | None = None,
) -> DenseRetriever
```

With `name_fields`, `build` also encodes each distinct non-blank value of those fields
(each item of a list field) as a document of its own, and a record scores by the best of
its vectors, the whole-doc vector included. Search scores every vector, keeps each
record's best and cuts to `limit` records, so each record appears once. The identity gains
`name_fields` and `format_version` 2 only when it is set; an index without it keeps the
identity it had. `xwalk_meta.json` records `doc_count` (records) and `vector_count` (rows).

`build` encodes rendered docs in batches of `batch` and validates that every vector's
length matches `encoder.dimension`, raising `ValueError` otherwise. When `name` is
omitted it defaults to `dense:{encoder.name}`. Three files are written to `index_dir`:

| File | Contents |
|---|---|
| `vectors.f32` | vectors as packed little-endian float32, one row per record |
| `record_ids.json` | record IDs in row order |
| `xwalk_meta.json` | engine, name, encoder, dimension, limit, fingerprint, `components`, doc count |

`open` requires the same encoder the index was built with, because a vector index is
meaningless to a different encoder. It raises `FileNotFoundError` when
`xwalk_meta.json` is absent, `ValueError` when `encoder.name` differs from the recorded
encoder, and `ValueError` when `encoder.dimension` differs from the recorded dimension.
`default_limit=None` keeps the value recorded at build time (falling back to 20). With
`expected` (from `xwalk.retrieval.dense.index_components(records, templates, encoder)`,
which encodes nothing) every stored component is compared — engine, format version, `doc`
template, record digest, and the encoder identity: model name, revision (`"unknown"`
when unpinned), dimension, `normalize`, both prefixes and max sequence length — and a
difference raises `IndexMismatchError` before any vector is read.

### FAISS vs pure Python

If `faiss` and `numpy` import, search uses `IndexFlatIP` over the stored vectors. This
is a *speed* fallback with identical semantics on normalised vectors — `IndexFlatIP` is
exact — not a silent ranking change. Without FAISS, search is an exact inner-product
scan in pure Python, sorted by descending score with ascending row index as the
tie-break. An empty or whitespace-only query, or an empty index, returns no hits.
`xwalk-match[dense]` pulls `faiss-cpu` along with sentence-transformers and torch; only the
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

## Postgres retrieval

`xwalk.retrieval.postgres` keeps the records, a BM25 index and the vectors in one
Postgres database, for target collections too large to ship with an application. It
needs no optional extra: SQL travels through a small executor protocol, and the one
executor xwalk ships speaks Neon's HTTPS SQL endpoint with httpx. The engines are Neon's
own: `lakebase_text` for BM25 (Neon blocks new installs of ParadeDB's `pg_search`; it is
listed in `neon.deprecated_extensions`) and pgvector `halfvec` with HNSW. A worked
deployment, with sizes, throughput and costs measured on Neon, is
the Neon deployment guide in the separate `xwalk-ui` repository.

### The executor protocol

```python
class SQLExecutor(Protocol):
    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]: ...
    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]: ...
```

Placeholders are Postgres's own `$1, $2, ...`. Values are JSON-shaped (text, numbers,
booleans, `None`, lists for arrays, dicts for jsonb); every statement xwalk sends casts
where the type matters, because over HTTP a bigint or numeric arrives as a string.
`transaction` runs its statements in one transaction, all or nothing. Implementations
must be thread-safe (searches run in threads). Every statement xwalk sends starts with
a tag comment, `/* xwalk:bm25_search */`; `tag_of(sql)` reads it, which is what a fake
executor in tests dispatches on.

```python
NeonHTTPExecutor(
    connection_string: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 2,
    backoff_base: float = 0.5,
    transport: httpx.BaseTransport | None = None,
    trust_env: bool = True,
    endpoint: str | None = None,
)
```

`POST https://<host>/sql` with the connection string in the `Neon-Connection-String`
header, `{"query", "params"}` for a statement or `{"queries": [...]}` for a
transaction, and `Neon-Array-Mode: true` (rows as arrays beside `fields`, so duplicate
column names cannot collide). A SQL error is `PostgresError` (a `RetrieverError`) with
the SQLSTATE in `code`; transport errors, HTTP 429/5xx and errors the server marks
`neon:retryable` are retried with jittered backoff, so writes sent through it should be
idempotent (the writer's are). Messages and `repr` never carry the connection string or
its password, and URLs in them are redacted. `from_env(name="DATABASE_URL")` reads the
variable; `stats` counts requests, statements and bytes.

### Retrievers and store

```python
@classmethod
def open(
    cls,
    executor: SQLExecutor,
    ontology: str,
    *,
    schema: str = "xwalk",
    name: str | None = None,
    expected: Mapping[str, Any] | None = None,
    verify_records: bool = True,
    default_limit: int | None = None,
    meta: Mapping[tuple[str, str], IndexMeta] | None = None,
    store: PostgresStore | None = None,
) -> PostgresBM25Retriever
```

`PgVectorRetriever.open(executor, ontology, encoder, ...)` takes the same keywords plus
`overfetch` (default 5) and `ef_search`, and `PostgresStore.open(executor, ontology, ...)`
the identity keywords. One schema holds any number of collections, each named by
`ontology`.

- **`PostgresBM25Retriever`** ranks with `tsv <@> to_bm25query(...)` over a partial
  `lakebase_bm25` index per ontology, so BM25 statistics are per collection, as with one
  Tantivy index each. The query text is tokenised by `to_tsvector('simple', text)` (the
  stored identity's `text_config`); it is never parsed as query syntax, so no
  sanitising is needed. With `exact_fields` (stored by the writer) records whose
  `exact_form` of a label or synonym equals `exact_form(query)` come first, ordered by
  their BM25 score, then the text matches: the same "an exact name outranks any text
  match" rule as `BM25Retriever(exact_fields=...)`, without a numeric boost. `raw_score`
  is the BM25 score (the negation of `<@>`); rows scoring 0 are dropped.
- **`PgVectorRetriever`** embeds the query with any `Encoder`, fetches the
  `limit * overfetch` nearest rows by inner product (`<#>`) from a partial HNSW index
  (`halfvec_ip_ops`), keeps each record's best row and cuts to `limit` records, so each
  record appears once even with per-name vectors, as `DenseRetriever(name_fields=...)`.
  The search is one read-only transaction that first raises `hnsw.ef_search` to at
  least the rows fetched. HNSW is approximate, unlike `DenseRetriever`'s exact scan;
  on small collections Postgres's planner picks an exact scan by itself.
- **`PostgresStore`** is a `TargetStore` over the stored records. A retriever opened
  with `store=` hands it the records its search returned, so fusion's `get_many` after
  a search usually needs no extra round trip.

Ranks are 1-based and contiguous; an empty or whitespace query makes no call.

### Identity

The writer stores each index's components in `<schema>.index_meta`, one row per
(ontology, kind) with kind `terms`, `bm25` or `dense`, and the retriever's `fingerprint`
is `hash_value` of its stored components. `bm25_components(records, templates,
exact_fields=..., text_config=...)` names the engine (`postgres-lakebase-bm25`), format
and normalisation versions, the doc template, the exact fields and the record digest;
`pgvector_components(records, templates, encoder, name_fields=...)` names the engine
(`pgvector`), vector type, metric, doc template, record digest, the encoder identity
(name, dimension and settings) and the name fields; `terms_components(records)` gives
the store's fingerprint, the same digest as `MemoryStore.from_source(records)`.
`open(..., expected=...)` refuses any difference with `IndexMismatchError`; an absent
index, or one whose load has not finished (status `loading`), raises
`IndexNotReadyError` (`state` is `missing` or `loading`); a stored fingerprint that
does not hash its own components is refused too. A caller that does not hold the
records passes `verify_records=False`: then everything but the record digest and count
is compared, and the identity still contains the stored digest. `read_index_meta(executor,
schema)` reads every identity in one query, for opening many collections at once.

### Writing

`PostgresIndexWriter(executor, schema="xwalk", dimension=1024)` creates the schema
(`create_schema`: the two extensions, the `terms`, `term_vectors` and `index_meta`
tables) and writes it: `upsert_terms` (rendered doc, `tsvector`, exact forms, digest),
`vector_rows` (each record's doc as `ord` 0, then each distinct name),
`present_vectors` (what is stored already, for resuming), `upsert_vectors` (one batch
per statement, vectors sent as text rounded to float16), `prune`, `create_indexes`
(the partial BM25 and HNSW indexes, after the bulk load), `write_meta` (status
`loading` before changing a collection, `ready` after), `drop_ontology`, `vacuum` and
`sizes`. Every write is an upsert, so a load can be rerun or resumed. A table already
holding vectors of another dimension is refused with a message naming both.

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

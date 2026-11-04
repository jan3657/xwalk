# NER site on Neon Postgres

Status: implemented and verified against the project's Neon branch on 4 November 2026
(a subagent of the orchestrating session; PostgreSQL 18.6, `aws-eu-central-1`, Neon free
plan). Amends [PLAN.md](PLAN.md), see its section 12. Library reference:
[Postgres retrieval](../reference/retrieval.md#postgres-retrieval).

The bundled backend maps entities onto four 200-term samples: BM25 built in `/tmp` at
cold start, dense vectors committed with the function. That cannot hold the full
catalogs (629,558 records, 1,934,527 per-name vectors). With `NER_SITE_BACKEND=neon` the
records, BM25 and vectors live in Postgres instead, and the function holds nothing but
code. Responses keep the same shape; `/api/health` adds `"backend": "neon"` and a `neon`
block (schema, catalog).

## What runs where

| Piece | Bundled (default) | Neon |
|---|---|---|
| Records | `xwalk/ui/ontologies/*.jsonl` in the package | `xwalk.terms` |
| BM25 | Tantivy, built into `/tmp` at cold start | `lakebase_text`: a partial `lakebase_bm25` index per ontology |
| Dense | `indexes/<slug>/dense/` committed, exact scan | pgvector `halfvec(D)`, a partial HNSW index per ontology |
| Store | `MemoryStore` | `PostgresStore` (records primed by the searches) |
| Query embeddings | OpenRouter, Qwen3-Embedding 8B, 1024-d | the same, at `NER_SITE_NEON_DIMENSIONS` |
| Built by | `tools/build_indexes.py` | `tools/load_neon.py` |

**pg_search is not available.** The plan named ParadeDB's `pg_search` for BM25. Neon
lists it in `neon.deprecated_extensions` and refuses `create extension pg_search`
("extension "pg_search" is deprecated and no longer allowed"). Neon's replacement,
`lakebase_text` 0.1.3, is used instead
([Neon docs](https://neon.com/docs/extensions/lakebase-text)): BM25 over a `tsvector`
column, `tsv <@> to_bm25query(query_tsvector, 'index'::regclass)` returning the negated
score (lower is better), with Block-Max WAND top-k. It accepts partial indexes, checked
with `explain`: the query uses `Index Scan using terms_bm25_<slug>_...`.

## The executor and Neon's HTTP protocol

This container cannot open TCP to Postgres (port 5432 times out); Vercel can, but HTTPS
works in both, so everything goes through `xwalk.retrieval.postgres.SQLExecutor`
(`execute(sql, params)`, `transaction(statements, read_only=)`) and its one
implementation, `NeonHTTPExecutor` on httpx. The protocol is the one
`@neondatabase/serverless` 1.2.0 speaks (read from its published source). Each claim
below was checked against the live endpoint on 2026-11-04:

| Claim | Checked |
|---|---|
| `POST https://<host>/sql`, connection string in `Neon-Connection-String`, body `{"query", "params"}` | yes; answer has `fields`, `rows`, `command`, `rowCount` |
| `{"queries": [...]}` runs one transaction, answer `{"results": [...]}` | yes; an insert followed by `1/0` left the table empty (rolled back) |
| `Neon-Batch-Read-Only`, `Neon-Batch-Isolation-Level` headers on a batch | read-only accepted; isolation level sent by the driver, not needed here |
| `Neon-Array-Mode: true`: rows as arrays | yes (`rowAsArray: true`) |
| `Neon-Raw-Text-Output: true`: every value as text | yes (`"t"` for true, `"{1,2}"` for an array) |
| Without raw text: JSON-typed values; int8 as a string, float8 a number, jsonb an object, arrays lists, bytea `"\\x0102"` | yes |
| JSON array and object params bind to `text[]`/`int[]` and `jsonb`; bytea as `"\\x..."` hex | yes |
| SQL error: HTTP 400 with `message`, `code` (SQLSTATE), `detail`, `hint`, ..., `neon:retryable` | yes (`42P01`, `22012`, `42601`) |
| A wrong password is also HTTP 400, `code` empty, `neon:retryable: true` | yes |
| One statement per query (`select 1; select 2` refused: 42601) | yes |
| Request and response up to 64 MB (Neon docs) | a 20 MB request answered 200 in 1.5 s; 64 MB not tried |
| `Content-Encoding: gzip` request bodies | refused (400 "could not parse the HTTP request body") |
| A cold endpoint | the first request after idle took 5.3 s; later ones about 120 ms from here |

`NeonHTTPExecutor` sends array mode (so duplicate column names cannot collide) without
raw text, retries transport errors, 429, 5xx and `neon:retryable` errors with jittered
backoff (`max_retries`, default 2; the site uses 1 and a 15 s timeout), and scrubs the
connection string, the password and every URL's userinfo from messages and `repr`.

## Schema

One schema (`xwalk` by default; `--schema`, `NER_SITE_NEON_SCHEMA`) holds every
ontology, keyed by its slug (`chebi-sample` ... for the samples, `chebi`, `ctd-disease`,
`ncbi-gene`, `foodon` for the full catalogs).

```sql
create extension if not exists vector;          -- 0.8.6
create extension if not exists lakebase_text;   -- 0.1.3
create table xwalk.terms (
  ontology text, id text, label text, synonyms text[], definition text,
  fields jsonb,          -- the whole record; PostgresStore returns it
  doc text,              -- TEMPLATES.render_doc(record), as the bundled index
  tsv tsvector,          -- to_tsvector('simple', doc)
  exact text[],          -- exact_form() of label and synonyms
  digest text,           -- hash_record(record)
  primary key (ontology, id));
create index terms_exact_gin on xwalk.terms using gin (exact);
create table xwalk.term_vectors (
  ontology text, record_id text, ord int,      -- 0: the doc, 1..: each distinct name
  kind text check (kind in ('doc', 'name')), text text,
  embedding halfvec(1024),
  primary key (ontology, record_id, ord));
create table xwalk.index_meta (
  ontology text, kind text,                    -- terms | bm25 | dense
  name text, status text,                      -- loading | ready
  components jsonb, fingerprint text, default_limit int,
  record_count int, vector_count int, dimension int, updated_at timestamptz,
  primary key (ontology, kind));
-- per ontology, after its rows are loaded:
create index terms_bm25_<slug>_<hash> on xwalk.terms
  using lakebase_bm25 (tsv) where ontology = '<slug>';
create index terms_hnsw_<slug>_<hash> on xwalk.term_vectors
  using hnsw (embedding halfvec_ip_ops) with (m = 16, ef_construction = 64)
  where ontology = '<slug>';
```

Partial indexes keep BM25 statistics per ontology (one Tantivy index each, as before)
and let the planner use the ontology's own HNSW graph; at 20,000 synthetic vectors the
dense query used `Index Scan using terms_hnsw_big_...` and answered in 133 ms (about one
round trip). On the 200-term samples the planner prefers an exact scan, which is
cheaper there and exact.

## Identity

The loader writes `status: loading` for an ontology's three identities (`terms`,
`bm25`, `dense`) before touching its rows and `ready` after the last write, with the
components:

- `bm25`: `bm25_components(records, TEMPLATES, exact_fields=("label", "synonyms"))`:
  engine `postgres-lakebase-bm25`, format and normalisation versions, doc template,
  exact fields, `text_config: simple`, record count and digest.
- `dense`: `pgvector_components(records, TEMPLATES, encoder, name_fields=("label",
  "synonyms"))`: engine `pgvector`, `halfvec`, inner product, doc template, record
  digest, the encoder identity (Qwen3-Embedding 8B, OpenRouter base URL, dimension,
  prefixes, provider preference, ...), name fields.
- `terms`: the record digest (the same value as `MemoryStore`'s fingerprint).

The site opens each with `expected`: for the samples computed from the bundled records
(digest included), so editing a sample makes the Neon index stale; for the full
catalogs, which the site does not ship, with `verify_records=False` (configuration
only; the identity still contains the loader's digest). A missing ontology, an
unfinished load, a mismatch or an unreachable database becomes the status reason, never
an exception and never another engine:

- `no neon index for chebi-sample (schema xwalk); run ner-site/tools/load_neon.py`
- `neon index for foodon-sample is incomplete (status loading: a load is running or was interrupted); rerun ner-site/tools/load_neon.py`
- `neon index for ctd-disease-sample is stale (differs in: doc_template); reload it with ner-site/tools/load_neon.py`
- `DATABASE_URL is not set (NER_SITE_BACKEND=neon)` (reported once, "(all 4 ontologies)")
- `neon database could not be queried (...)`: not memoised, so the next request retries

`NER_SITE_NEON_DIMENSIONS` must equal the loaded dimension; otherwise dense is stale
with `encoder.dimension` among the differences.

## Loading

```bash
set -a; . ./.env.local; set +a        # DATABASE_URL, and OPENROUTER_API_KEY for embeddings
python ner-site/tools/load_neon.py --dry-run                  # samples: counts, size, tokens, cost
python ner-site/tools/load_neon.py                            # the four samples
python ner-site/tools/load_neon.py --source full --dry-run    # downloads the sources, estimates
python ner-site/tools/load_neon.py --source full --only ncbi-gene --limit 2000
python ner-site/tools/load_neon.py --source full --only ncbi-gene --drop   # remove it again
python ner-site/tools/load_neon.py --benchmark-upload 6000 --concurrency 8  # synthetic, dropped
```

Options: `--source samples|full`, `--only chebi|ctd-disease|ncbi-gene|foodon`
(repeatable), `--limit N` (first N records each), `--dimensions D` (default 1024; 512
works the same), `--batch` (texts per embeddings request, default 128),
`--concurrency` (batches in flight, default 4), `--schema`, `--cache-dir` (downloads),
`--max-bytes` (default 400 MB), `--dry-run`, `--drop`, `--benchmark-upload N`.

- **Sources.** Samples: the bundled JSONL. Full: CTD `CTD_diseases.tsv.gz` (DiseaseName,
  DiseaseID, Definition, Synonyms split on `|`), ChEBI `chebi.obo.gz` through
  `obo_source(id_prefix="CHEBI:")`, FoodOn `foodon.owl` through
  `owl_source(id_prefix="FOODON:")` (needs `xwalk[ontology]`), NCBI Gene
  `{Homo_sapiens,Mus_musculus,Rattus_norvegicus}.gene_info.gz` (Symbol, Synonyms with
  `-` for none, description, organism from `#tax_id`). Records keep the samples' fields
  (label, synonyms, definition; genes add organism and tax_id). Downloads go to their own
  directory, are written under a `.part` name until complete, are only parsed, and are
  reused by later runs.
- **Same index as bundled.** `ner_site.ontologies.TEMPLATES` renders the docs; vectors
  are each record's doc plus each distinct label and synonym
  (`record_names(record, ("label", "synonyms"))`), the retriever names are `bm25` and
  `dense-qwen3`.
- **Resumable.** Terms are upserted (unchanged digests are not rewritten); vectors are
  checked against what is stored with the same text, window by window, and only the
  missing ones are embedded; each batch is one embeddings request and one insert, so a
  crash loses at most the batches in flight. Records that left the source are pruned.
  Indexes are created after the rows (`if not exists`), then `analyze`, then `ready`.
  A rerun of the sample load embedded nothing.
- **Guard.** `--source full` refuses to start when the size estimate plus the database's
  current size exceeds `--max-bytes`, before writing anything.
- **Usage.** It prints progress per window and, at the end, the encoder's requests,
  texts, tokens and cost as the provider reported them ("unknown" when it did not), the
  database requests and bytes, the vector upload rate, and the table sizes.

## Vector format

Measured 2026-11-04 by the orchestrator on cached 1024-d Qwen3 vectors (512 and 256 by
truncation and renormalisation, f16 by casting), dense retrieval with per-name vectors;
the 200 query embeddings cost $0.00004. MRR is the mean over ncbi_disease, nlm_gene and
cafeteria_fcd; CHEBI saturates at 1.0 for every format.

| Format | Mean MRR |
|---|---|
| float32, 1024 | 0.836 |
| halfvec (f16), 1024 | 0.836 (identical on every dataset) |
| f16, 512 (truncated) | 0.831 |
| f16, 256 (truncated) | 0.834 |
| binary 1024, then f16 rerank of the top 400 vectors | 0.836 |
| binary only | 0.831 |

The differences are within noise, but the pilots have only 200 targets each, so lower
dimensions or binary codes may lose more on 200,000-record catalogs. Decision: default
`--dimensions 1024` stored as `halfvec(1024)`, the same quality as the committed
float32 index; `--dimensions` stays configurable and 512 works the same way. A future
option: a binary-quantised prefilter (`binary_quantize(embedding)::bit(D)` under its own
HNSW index, Hamming distance) with a `halfvec` rerank of a few hundred candidates.

Vectors are sent as pgvector text rounded to float16 first and written with five
significant digits, which reads back as exactly that float16 (about 9.9 KB of request a
1024-d vector). Gzip bodies are refused by the endpoint, and pgvector has no SQL
function that turns bytea into `halfvec`, so text it is.

## Sizes (measured on Neon, 2026-11-04)

| What | Bytes |
|---|---|
| One 1024-d vector (heap, TOAST, key, HNSW), 6,000 synthetic vectors | 5,677 (2,945 before the HNSW index) |
| One 512-d vector, same test | 2,615 (1,247 before the HNSW index) |
| The four samples: 3,132 vectors (`term_vectors` with its indexes) | 17.3 MB, 5,523 a vector |
| The four samples: 800 terms (`terms` with BM25, GIN and key indexes) | 1.7 MB, 2,125 a term |
| A 2,000-gene slice of NCBI Gene: 10,964 vectors and 2,000 terms | 61.5 MB and 3.6 MB (estimate 62.6 and 3.5) |
| The database after the sample load, with only `xwalk` left | 31.3 MB (`pg_database_size`) |

A 1024-d `halfvec` is 2,052 bytes, just over Postgres's 2 KB inline limit, so it moves to
TOAST, and the HNSW graph holds a second copy; that is why a vector costs about 5.7 KB,
not 2 KB, and why the full catalog needs about 11 GB at 1024 dimensions, not 4 GB.
`load_neon.py` estimates sizes from these measurements (5.98 bytes a dimension minus
446 per vector; 1,300 bytes plus 6.6 per doc character per term).

## Throughput and latency

Upload, from this container (Europe, behind an HTTPS proxy) to Frankfurt, synthetic
1024-d vectors, no index during the upload:

| Batch | In flight | MB/s | Vectors/s |
|---|---|---|---|
| 128 | 1 | 4.4 | 447 |
| 128 | 4 | 9.3 | 936 |
| 128 | 8 | 10.9 | 1,096 |
| 512 | 4 | 10.1 | 1,019 |
| 128 | 8 (6,000 vectors) | 9.7 | 974 |
| 128 at 512-d | 8 | 10.3 | 2,094 |

HNSW build after the upload: 6,000 vectors in 5.6 s at 1024-d (1,071 a second), 3.3 s at
512-d. In real loads embedding is the bottleneck: the 2,000-gene slice (10,964 texts,
`--concurrency 4`) took 70.7 s, of which embedding was 190.6 request-seconds and upload
30.0 request-seconds.

Per-request latency of the Neon retrievers from this container (10 searches each,
medians): a round trip (`select 1`) 120 to 125 ms; BM25 search 123 ms; dense search, DB
part only, 142 to 150 ms; the query embedding through OpenRouter 264 ms (164 to 625);
a `store_get` 122 ms (rarely needed: the searches prime the store). Opening the four
ontologies on a cold instance costs one `index_meta` query (0.6 s for health, including
a warm-up). From Vercel's default `iad1` region each query also crosses the Atlantic (a
Virginia to Frankfurt round trip is typically around 90 ms; an assumption, not
measured); running the function in `fra1`, or the database in `aws-us-east-1`, avoids
it. Not measured from Vercel.

## Costs

Embeddings (Qwen3-Embedding 8B through OpenRouter, as reported): the four samples,
3,132 texts, 65,316 tokens, $0.000653; the 2,000-gene slice, 10,964 texts, 94,773
tokens, $0.000948. That is $0.01 a million tokens; the dry run estimates tokens at two
characters each, which errs high for prose and is close for gene symbols.

## The full load

The full catalogs do not fit the free plan. `--source full --dry-run` (sources
downloaded once, 1 min 44 s including the FoodOn OWL parse):

| Ontology | Records | Vectors | Est. tokens | Est. cost | Est. size at 1024-d |
|---|---|---|---|---|---|
| chebi | 219,011 | 852,943 | 42.4 M | $0.42 | 5,536 MB |
| ctd-disease | 13,323 | 104,630 | 3.1 M | $0.03 | 643 MB |
| ncbi-gene | 368,540 | 911,653 | 15.0 M | $0.15 | 5,866 MB |
| foodon | 28,684 | 65,301 | 2.5 M | $0.02 | 441 MB |
| total | 629,558 | 1,934,527 | 62.9 M | $0.63 | 12,486 MB (12.5 GB) |

At 512 dimensions the total is about 6.6 GB (vectors 5.2 GB, terms 1.35 GB). Projected
time, from the measurements
above: about 35 minutes of upload at 1,000 vectors a second, HNSW builds of at least
30 minutes (they slow down as a graph grows; not measured beyond 20,000 vectors), and
embedding at about 155 texts a second with `--concurrency 4` (3.5 hours; more in flight
helps if the provider allows it). Once the plan allows about 13 GB (a Neon paid plan;
raise `maintenance_work_mem` for the HNSW builds if the plan lets you):

```bash
python ner-site/tools/load_neon.py --source full --max-bytes 14000000000 --concurrency 8
```

Then set `NER_SITE_BACKEND=neon` and `NER_SITE_NEON_CATALOG=full` on the deployment.
The loader is resumable, so it can be stopped and rerun, and `--only` loads one ontology
at a time.

## The full load, done 4 November 2026 (Launch plan)

The org moved to Neon's Launch plan (pay as you go: $0.35 a GB-month of storage, $0.106 a
CU-hour of compute, scale to zero after 5 minutes). Loaded into schema `xwalk_full` at
`halfvec(768)`, through the unpooled endpoint with `maintenance_work_mem = 1GB` set on the
database (`alter database neondb set ...`; pooled sessions do not see it). **NCBI Gene was
not loaded** (the operator's choice); the site does not route genes with the full
catalogs.

| Ontology | Records | Vectors | Tokens (reported) | Cost (reported) | Time |
|---|---|---|---|---|---|
| ctd-disease | 13,323 | 104,630 | 1,897,340 | $0.0190 | 14.7 min |
| foodon | 28,684 | 65,301 | 1,344,408 | $0.0134 | 8.7 min |
| chebi | 219,011 | 852,943 | 44,057,139 | $0.4406 | 54 min, then the index (below) |
| total | 261,018 | 1,022,874 | 47,298,887 | $0.4730 | about 80 min |

CTD was also loaded once at 1024 dimensions for the truncation test and dropped
(another $0.0190). Embedding ran at 120 to 330 vectors a second with `--concurrency 8`,
bound by OpenRouter.

**Vector format.** `halfvec(768)`: on the fully loaded CTD vocabulary 768 dimensions
matched 1024 at every cut-off and 512 lost 0.07 MRR; a dimension just under the 2 KB
inline limit (1000, 1016) costs exactly what 1024 does, and `STORAGE PLAIN` saves 2%
(`benchmarks/results/neon-truncation-2026-11-04/FINDINGS.md`). Stored:
`term_vectors` 4,006.7 MB for 1,022,874 vectors (3,917 bytes each, heap and HNSW),
`terms` 480.1 MB, the database 4,541 MB in all (the samples in `xwalk` included), so
about $1.60 a month of storage.

**The ChEBI HNSW build.** Over 852,943 vectors it outlived the loader's 120 s request:
each of the three retries started the build again and the abandoned request cancelled
it, so the loader stopped with a read timeout after storing every vector. Built by hand
in one request (`set local maintenance_work_mem = '3GB'`, `set local
max_parallel_maintenance_workers = 4`, the loader's `create index` statement): 110 s.
A rerun of the loader then embedded nothing and marked ChEBI ready. The loader now builds
indexes through a second executor with an hour's timeout and no retries.

**Retrieval on the full catalogs**
(`benchmarks/results/site-retrieval-full-2026-11-04/FINDINGS.md`): fused MRR ChEBI 0.749,
CTD 0.572, FoodOn 0.766 (samples: 1.000, 0.704, 0.914), recall@20 0.96, 0.70, 0.94.

## Site configuration

| Name | Default | Meaning |
|---|---|---|
| `NER_SITE_BACKEND` | `bundled` | `neon` reads records, BM25 and vectors from Postgres |
| `DATABASE_URL` | unset | the Neon connection string (read only with `neon`; scrubbed from every response and log line) |
| `NER_SITE_NEON_SCHEMA` | `xwalk` | the schema the loader wrote |
| `NER_SITE_NEON_DIMENSIONS` | 1024 | must equal the loaded `--dimensions` |
| `NER_SITE_NEON_CATALOG` | `samples` | `full` routes to `chebi`, `ctd-disease`, `ncbi-gene`, `foodon` |

Before the Vercel deployment can use it, `ner-site/requirements.txt` must pin an xwalk
commit that contains `xwalk.retrieval.postgres`.

## Verified on 4 November 2026

- Offline: `tests/test_postgres_retrieval.py` (fake executor, `httpx.MockTransport`) and
  `tests/test_site_neon.py` (an in-memory stand-in that answers each tagged statement
  with the SQL's semantics: the loader end to end, resume, prune, the size guard, the
  parsers, the site's statuses, health, mapping).
- Live, over HTTPS: `tests/test_postgres_integration.py` (a throwaway schema, created and
  dropped) and the sample-index test in `tests/test_site_neon.py`.
- The sample load, then `NER_SITE_BACKEND=neon python ner-site/tools/serve_local.py`:
  `/api/health` 200 with `backend: neon`, BM25 ok, dense ok; the page's example text
  gave 10 entities, every one matched, 1 LLM call (641 + 186 tokens), 10 embedding
  requests, mapping 1.6 s of 5.2 s. The same 10 entities mapped through the bundled
  backend gave the same top 5, in the same order, for every entity.
- A 2,000-gene slice of the full NCBI Gene source loaded, served with
  `NER_SITE_NEON_CATALOG=full` (ADA and NAT2 first in both retrievers; the three
  unloaded ontologies reported as missing), and dropped again.

Not done: a load beyond the free plan, any request from Vercel, a psycopg (TCP)
executor.

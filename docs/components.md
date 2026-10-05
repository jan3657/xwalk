# What each component does

xwalk is roughly sixty modules in eight layers. This page says what every one of them
is for, in one or two sentences, so you can find the right file without reading all of
them. Each entry links to the reference page that documents its API in full.

The layers depend downward only: prompts and stages depend on core types, the matcher
depends on stages, batch depends on the matcher, and evaluation depends on nothing but
the ledger. Nothing depends on the CLI or the MCP server; both only format what
`ops.py` returns. [Architecture](architecture.md) has the compact layer map, the
invariants and a contributor map from a change to its modules and tests.

```
cli/main.py, mcp_server.py ── ops.py ── config.py ──┐
                                                    ▼
  batch.py ──► matcher.py ──► stages/ ──► prompts/ ──► llm/
                  │              │
                  ├──► retrieval/│         records.py  policy.py
                  ├──► stores/   └───────► templates.py  keying.py
                  └──► ledger.py
                          ▲
                  evaluate/ ── review.py     sources/ ──► records.py
```

---

## Core types — [reference](reference/core-types.md)

The vocabulary everything else is written in. All of these are frozen dataclasses, so a
result you hold is a result nobody can edit behind your back.

| Module | What it does |
|---|---|
| `records.py` | The value types: `Record` (one row from either collection), `RetrievalHit`, `Candidate`, `Usage`, `RetryProposal`, `Attempt`, `MatchResult`, and the `MatchStatus` / `DecisionReason` enums. `Attempt` is the per-try audit trail; `MatchResult` carries all of them. |
| `policy.py` | `MatchPolicy` — the thresholds and cost controls — plus `should_verify`, `should_audit`, and `derive_status`, which reduces a list of attempts to one final status and reason. This is where "what counts as a match" is decided, and it is the only place. |
| `templates.py` | `TemplateSet`: the four Jinja templates (`query`, `context`, `doc`, `candidate`) that carry your entire domain mapping. Compiles eagerly so a typo fails at construction, and renders a missing record field as an empty string rather than aborting a 100k-row run. |
| `fingerprint.py` | Deterministic hashing. `hash_record`, `run_fingerprint`, and `result_key` are what decide whether a previously computed result still applies — the whole resume mechanism rests here. |
| `serde.py` | Converts a `MatchResult` tree to and from plain JSON-safe dicts, so the ledger can store one blob per result without knowing the shape of a result. |
| `ledger.py` | The SQLite run store. Every completed result is committed the moment it finishes, so a crash at record 9,000 of 10,000 costs you only the records in flight. Keeps the current view (one row per source in the latest snapshot), the full history, run invocations and the review overlay. `mapping.csv`, `results.jsonl` and `manifest.json` are exports regenerated from here — this file is the source of truth. |
| `_extras.py` | The one place a missing optional dependency is explained. `require()` raises `MissingExtra` naming the extra and the exact `pip install` line, instead of an `ImportError` from a library you have never heard of. |

## Getting records in — [reference](reference/sources.md)

A record source is just `Iterable[Record]`. That is the entire extension contract: a
list, a generator over ten million rows, or a function querying your warehouse all
qualify equally.

| Module | What it does |
|---|---|
| `sources/base.py` | Defines `RecordSource = Iterable[Record]`. That is the whole file, and that is the point. |
| `sources/tabular.py` | `csv_source` and `jsonl_source`. Both are lazy generators. Handles multi-value columns (`dextrose\|grape sugar` becomes a list) and refuses a blank id rather than inventing one. TSV is `csv_source(delimiter="\t")`. |
| `sources/ontology.py` | `obo_source` (parsed directly, no dependency) and `owl_source` (rdflib, needs `xwalk[ontology]`). Both emit the *same* field names — `label`, `synonyms`, `definition`, `parents`, `obsolete` — so one `doc` template works against either format. |
| `sources/sql.py` | `sql_source`: streams a query through SQLAlchemy in chunks rather than materialising it, and disposes the engine even if you abandon the generator. Needs `xwalk[sql]`. |
| `stores/base.py` | The `TargetStore` protocol: `get`, `get_many`, `fingerprint`. Kept separate from retrieval on purpose — retrievers return ids and ranks, the store turns ids into records, and that split is what lets an external search backend scale without holding every record in memory. |
| `stores/memory.py` | `MemoryStore`, the dict-backed default. Refuses duplicate target ids outright; skips unknown ids on lookup, because a stale index may legitimately name a deleted record. |

## Retrieval — [reference](reference/retrieval.md)

| Module | What it does |
|---|---|
| `retrieval/base.py` | The `Retriever` protocol (`name`, `fingerprint`, `default_limit`, `async search`), the `SearchRequest` value object, and `RetrieverError`. A request object rather than `search(query, k)` so filters can be added later without breaking every implementation. |
| `retrieval/bm25.py` | Lexical retrieval over Tantivy. `exact_fields` adds a heavily boosted whole-string match so an exact label or synonym hit outranks a document that merely repeats the query term. `sanitise_query` strips query-syntax characters rather than escaping them, because real mentions contain `(`, `-`, `+` and `:`. |
| `retrieval/dense.py` | Vector retrieval behind a pluggable `Encoder` protocol, with `SentenceTransformerEncoder` as the default. Uses FAISS when available purely for speed — on unit-normalised vectors the ranking is identical either way. Needs `xwalk[dense]`. |
| `retrieval/fusion.py` | Reciprocal rank fusion. Combines rankings whose raw scores share no scale (BM25 against cosine) using rank position alone, with deterministic tie-breaking so two identical runs produce byte-identical candidate order. |

## The pipeline — [reference](reference/pipeline.md)

| Module | What it does |
|---|---|
| `stages/keying.py` | Assigns opaque per-attempt keys (`C01`, `C02`, …) to candidates and resolves a model's answer back to a record id by exact dictionary lookup. This is the module that makes a malformed answer safe: it can fail to resolve, but it cannot resolve to the *wrong* record. |
| `stages/select.py` | `Selector` — shows the model a keyed candidate list and gets back one key or null. `SelectorPolicy` and `apply_budget` decide how much of the fused list actually fits, deterministically. |
| `stages/gate.py` | `Scorer` and `Verifier`. The scorer produces the confidence that gets thresholded, judged against the whole source record rather than the retrieval query. The verifier is a second, independent opinion bought only when the first one is uncertain — it returns a verdict, not a number to average. |
| `stages/rewrite.py` | `QueryRewriter` — proposes new queries when retrieval has run dry. Produces query proposals only, never candidate proposals. |
| `stages/proposals.py` | Routes the scorer's retry leads into "re-examine this candidate" versus "run this new search", and drops the rest with a recorded reason. Conflating the two would make the loop re-retrieve records it is already holding. |
| `matcher.py` | The loop: retrieve, select, score, maybe verify, decide whether to try again. Orchestration only — every actual decision lives in a stage or in `policy.py`. |
| `batch.py` | Point it at a source, get a resumable run directory. Bounds concurrency, commits each result as it lands, and writes the three exports at the end. `build_run_fingerprint` lives here. |

## LLM clients — [reference](reference/llm.md)

| Module | What it does |
|---|---|
| `llm/base.py` | The `LLMClient` protocol, `LLMRequest` / `LLMResponse` / `LLMCapabilities`, and the error hierarchy. Capabilities decide what the adapter *asks* a provider for; they never decide whether to trust the answer. |
| `llm/openai_compat.py` | Chat Completions over httpx: retry with backoff, capability profiles per provider family, and a structured-output fallback that asks once and remembers. The API key is deliberately excluded from the fingerprint so rotating a key does not invalidate a resumable run. |
| `llm/litellm.py` | A thin adapter over LiteLLM for the hundred providers it proxies. No retry loop and no profiles — it cannot know what any given backend supports, so it asks for nothing by default. Needs `xwalk[litellm]`. |
| `llm/fake.py` | A scripted client. This is the reason the entire matcher loop is testable offline, and it is why the test suite runs in seconds with no credentials. |
| `llm/budget.py` | `CallBudget` and `BudgetedLLM`: a per-invocation cap on upstream requests, reserved before each dispatch (retries included), so concurrent records cannot overshoot it. This is `--max-calls`. |
| `llm/cache.py` | `CachingLLM` — serves an identical repeated request from the ledger. Delegates its identity to the wrapped client, so caching changes how an answer was obtained and never what it means. |
| `llm/parsing.py` | Getting JSON out of what a model actually returns: thinking blocks (including the truncated and dangling-tag cases), fenced code blocks, trailing commas. Anything unsalvageable raises, and the matcher routes it to review — a malformed answer is a signal, not a non-match. |

## Prompts — [reference](reference/prompts.md)

| Module | What it does |
|---|---|
| `prompts/base/*.j2` | The four skeletons (`select`, `score`, `verify`, `rewrite`). They own the machine-readable contract: the section headings and the output shape. No model is ever asked to write these. |
| `prompts/contract.py` | `PromptSlots` (your domain content), `PromptSet` (slots plus skeletons), the four JSON schemas, and `validate_contract`, which renders all four prompts against fixtures and checks the contract survived. A bad slots file can produce a weak rubric; it cannot produce a prompt whose output will not parse. |
| `prompts/author.py` | `draft_slots` — asks a model to propose slots for your domain from a description and a handful of sample records, then repairs what is mechanically repairable and validates the rest. Plus `write_slots` and `slots_diff`. |
| `prompts/optimize.py` | Iterative prompt improvement against gold labels, over three partitions with distinct jobs: prompt-train supplies the failures shown to the optimising model, validation picks the round to keep, and test is scored exactly once at the end. |

## Evaluation and review — [reference](reference/evaluation.md)

Nothing in this layer calls an LLM or a retriever. It reads the ledger, which is what
makes scoring a run free, repeatable, and runnable on a machine with no credentials.

| Module | What it does |
|---|---|
| `evaluate/gold.py` | Loads gold labels while preserving the distinction that matters: an empty `gold_ids` cell means "the correct answer is no match", an absent row means unlabelled. Alias expansion and id normalisation are hooks you supply — the library never learns your identifier scheme. |
| `evaluate/metrics.py` | The operational numbers: accepted precision, automatic coverage, review rate, no-match precision and recall, cost and latency per record, a threshold curve re-derived from recorded confidences, and a warning when those confidences do not separate correct from incorrect. |
| `evaluate/ceiling.py` | The three-way failure decomposition — never retrieved, truncated, misjudged — because each one sends you somewhere different: the doc template, the selector budget, or the prompts. |
| `evaluate/report.py` | `evaluate` and `render_report`: one call, one printable summary that leads with where to spend effort rather than with a headline number. |
| `evaluate/failures.py` | Chooses which failures are worth showing an optimising model, per prompt role. A selector learns nothing from a case where the gold record was never retrieved; a scorer learns a great deal from exactly that case. |
| `evaluate/partition.py` | Salted-hash three-way splitting. Deterministic and unseeded, so adding labelled records never reshuffles the existing split. |
| `evaluate/compare.py` | Side-by-side comparison of completed runs. This is how you actually choose a model: precision next to cost next to the misjudged count. |
| `evaluate/ablate.py` | Re-runs with one component disabled and reports the delta. Includes halving the selector budget, so an ablation can confirm the diagnosis the ceiling decomposition made. |
| `review.py` | Human review as an immutable overlay. Three layers are kept and never collapsed: what the model said, what the reviewer decided, and the adjudicated view derived from both. |

## Configuration and CLI — [job file](reference/job-file.md) · [CLI](reference/cli.md)

| Module | What it does |
|---|---|
| `config.py` | `JobSpec` — a job file is serialized constructor arguments and nothing more. Every field maps to something you would otherwise pass by hand, no config field gates behaviour the SDK cannot express, and credentials are environment variable *names*, never values. |
| `ops.py` | The operations layer: `validate`, `index`, `run`, `search`, `inspect`, `explain`, `results`, `export`, `review_export`, `review_apply`, `init`, `cluster`. Each returns an `OpResult`, the versioned `--json` envelope with its exit code; strict job loading, index reuse checks, run-directory collision checks and the call budget live here. |
| `cli/main.py` | Argument parsing and dispatch for the sixteen subcommands. Each one parses, calls one `ops` operation (or, for `eval`, `compare`, `ablate` and `prompts`, one library function), prints, and returns an exit code. |
| `mcp_server.py` | `xwalk mcp`: six bounded tools over the Model Context Protocol, each calling one `ops` operation. Needs `xwalk[mcp]`. See the [MCP guide](guide/mcp.md). |

## Clustering (experimental) — [guide](guide/clustering.md)

| Module | What it does |
|---|---|
| `cluster/engine.py` | The stream and refinement steps: assign, create, defer or fail each record; verified merges; reassignment with the incumbent always shown; stopping and the selected revision. |
| `cluster/pool.py` | The cluster pool's retrieval index, updated per change instead of rebuilt per new cluster; fused with the matcher's RRF. |
| `cluster/store.py` | Its own SQLite schema: one transaction per step, versioned cluster revisions, every decision with the clusters retrieved and shown. |
| `cluster/run.py`, `cluster/operation.py` | Run identity, resume, call limits, exports; the `xwalk cluster` operation and `kind: cluster` validation. |

---

## Where to go next

- New to xwalk: [getting started](guide/getting-started.md).
- Want to know *why* the loop is shaped this way: [concepts](concepts.md).
- Want to plug in your own retriever, source, or model: [extending xwalk](guide/extending.md).
- Something is behaving oddly: [troubleshooting](guide/troubleshooting.md).

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 onward. Before
1.0.0, minor versions may contain breaking changes.

## [Unreleased]

### Fixed

- A `NaN` or `Infinity` confidence from the scorer was clamped to `1.0` and accepted as
  a match. Every stage now accepts only a finite JSON number in `[0, 1]`; anything else
  (missing, strings, booleans, non-finite, out of range) is invalid output and routes
  to review. `derive_status` also refuses a non-finite stored score.
- `keep_candidates_in_trace=False` changed decisions: a scorer's candidate proposal was
  re-examined against an empty candidate list. Live candidate state is no longer read
  back from the trace; trace on/off gives identical results and identical requests.
- A candidate whose rendered text contained blank lines had the rest of its text moved
  into the "other candidates" section of the scorer and verifier prompts. Candidate
  blocks are now kept per key until rendering.
- Verifier and rewriter provider errors escaped `Matcher.match`. Every stage now follows
  one error policy (see `docs/reference/pipeline.md`): recoverable errors stay inside
  the record (a failed gating verification can never become `matched`), and fatal
  errors yield `failed` with the new reason `fatal_provider_failure`.
- Rewriter calls were missing from usage totals. Every stage call is now attributed to
  an attempt, and `result.usage` equals the sum of its attempts.
- Job `llm.temperature`/`llm.max_tokens` changed only the run fingerprint: stages
  hard-coded `temperature=0.0` and 512/256 output tokens. The job's values now reach
  every matching request.
- Editing a source record and resuming exported two rows for it (`mapping.csv`,
  `results.jsonl`, manifest counts, `duplicate_targets`, `BatchReport.total`). Each
  `run_batch` invocation now records the source collection as a snapshot, and every
  export and report uses the current view: exactly one row per current source. Earlier
  results are kept as history (`Ledger.iter_history`, `export_history_jsonl`).
- A failed record, or a fatal provider error, could keep a run from finishing cleanly:
  `run_batch` committed `fatal_provider_failure` results, gathered tasks without
  cancelling them, and closed the ledger while sibling tasks could still write to it.
  A fatal result is now never committed and aborts the run; on an abort, an exception
  or a cancellation, in-flight records are cancelled and awaited, the exports and the
  manifest are written with the run state, and only then is the ledger closed. A write
  after `close()` raises `LedgerClosedError`.
- A `failed` result was never retried on resume. It now is; the failure stays in
  history.
- A ledger write that failed at `COMMIT` left its transaction open, to be committed by
  the next write. Every ledger write is now one explicit transaction, rolled back on
  failure.
- `xwalk match` returned 0 for a run with failed records or a fatal provider error. It
  now returns 3 (an interim mapping until the shared operations of task 03).
- Credentials passed to `LiteLLMClient` as keyword arguments (`api_key`, headers, ...)
  and userinfo or key parameters in an `OpenAICompatClient` base URL fed the LLM
  fingerprint, so rotating a key invalidated resumable runs. They are excluded now.
- The run fingerprint ignored dense encoder settings: reopening an index with another
  query prefix, document prefix or normalisation reused old results. The live encoder's
  identity (`DenseRetriever.encoder_identity`) is now part of it. `JobSpec.run_fingerprint`
  also used the default fallback retrieval depth instead of the one handed to `Matcher`.

### Added

- Run states: `RunState` (`complete`, `partial`, `failed`, `aborted`, `interrupted`) on
  `BatchReport.run_state` and in `manifest.json`, plus `BatchReport.pending` and
  `BatchReport.errors`.
- `run_batch(limit=...)`: the snapshot is the whole source, only the first N unfinished
  records are processed, the rest are `pending` in `mapping.csv` and the run is
  `partial`. A repeated source id raises `DuplicateSourceIdError`.
- Ledger schema versioning (`LEDGER_SCHEMA_VERSION = 2`, `UnsupportedLedgerError`),
  result revisions and history, source snapshots, an invocation log,
  `removed_sources`, and `xwalk.review.review_history`.
- `run_fingerprint_components` (and `JobSpec.run_fingerprint_components`); `run_batch`
  stores them in the manifest when given (`fingerprint_components=`).
- `unknown_calls` and `cache_hits` columns in `mapping.csv`; the manifest's `usage` with
  `Usage.describe_tokens()`.
- `SentenceTransformerEncoder(revision=...)` and its `settings`.
- `Usage.unknown_calls` (calls whose tokens were not reported, including failed calls
  and responses without a usage block) and `Usage.cache_hits`; `Usage.describe_tokens()`.
  `calls` now includes `OpenAICompatClient`'s internal retries. Retries inside LiteLLM
  remain unobservable (documented).
- `DecisionReason.FATAL_PROVIDER_FAILURE`.
- `llm.seed` in job files and a `seed=` argument on both clients.
- `temperature=` on every stage, alongside `max_tokens=`, as explicit per-stage
  overrides.
- `xwalk.llm.parsing.parse_confidence`, `xwalk.llm.base.failure_usage`, and a `usage`
  attribute on `LLMError`.

### Changed

- **Compatibility:** opening a 0.1.1 ledger upgrades it in place after copying it to
  `ledger.sqlite.v1-backup`; running 0.1.1 against the upgraded file is not supported,
  and the backup is the way back. The
  upgrade only adds tables and defaulted columns. A ledger from a newer xwalk is refused
  unmodified.
- **Compatibility:** `Ledger.iter_results`, `count`, `count_by_status`,
  `duplicate_targets`, `export_review`, `adjudicated` and every export now read the
  current view, not every row ever committed. `BatchReport.total` is the number of
  current sources.
- **Compatibility:** `xwalk match --limit N` reads the whole source; records beyond the
  limit appear in `mapping.csv` with status `pending`, and the exit code is 1
  (`partial`) instead of 0. Repeating the command continues with the next N.
- **Compatibility:** `mapping.csv` gained two trailing columns, `unknown_calls` and
  `cache_hits`.
- **Compatibility:** a review is bound to the result revision it was made against. It
  stops applying (state `stale`) when the source is edited or removed or the result is
  recomputed; `apply_review` refuses rows whose result is no longer current or now
  proposes another target.
- **Compatibility:** the run fingerprint changed for jobs with dense retrievers or a
  retriever `limit` other than 20, and for LiteLLM clients given credential keyword
  arguments; such runs recompute once.
- **Compatibility:** `LLMRequest.temperature`/`max_tokens` default to `None` (use the
  client's value). Stage `max_tokens` defaults changed from 512 (256 for the rewriter)
  to `None`, so default requests now send the client's `max_tokens` (1024 unless
  configured).
- **Compatibility:** an out-of-range confidence (for example `1.7`) is no longer clamped;
  it is invalid output. The selector treats it like a missing confidence (`UNRESOLVED`).
- **Compatibility:** `KeyedCandidates` takes `blocks` (key -> rendered block) instead of
  `rendered`; `rendered` is now a derived property.
- **Compatibility:** a fatal provider error yields reason `fatal_provider_failure`
  instead of `provider_failure`. A cache hit reports `Usage(cache_hits=1)` instead of
  `Usage.zero()`. Ledgers written by 0.1.x read back with `unknown_calls=0` and
  `cache_hits=0`.
- The CI platform matrix now executes rather than merely being declared. macOS 14,
  Windows, and Linux on 3.10/3.11/3.12, plus aarch64 and each optional extra, all pass
  as of v0.1.1 — which retires the second of the limitations listed under 0.1.0.
- README badges, and a release runbook that covers a dropped tag trigger and a refused
  trusted-publisher exchange.

## [0.1.1] — 2026-08-04

The first release published to PyPI. 0.1.0 was tagged but never uploaded, and its tree
carries the three bugs below; install 0.1.1 or later.

### Fixed

- `BM25Retriever.build` appended to an existing index directory instead of replacing it,
  so every rebuild added a second copy of every document. `xwalk match` rebuilds into
  `<out>/index` on each invocation, so this triggered on the second run of any job.
  Fusion collapses the duplicate hits, which is why it was invisible — but the duplicates
  still spend the retrieval `limit`, so fewer distinct records reach the model and recall
  silently drops. The retriever fingerprint hashes records rather than the index, so
  nothing downstream detected it either. Building now replaces the index contents.
- `xwalk ablate` aborted with "at least one retriever is required" on any job configuring
  a single retriever. `standard_ablations` emitted a `no_<name>` variant that dropped the
  sole retriever, leaving `Matcher` nothing to build with, and the failure killed the
  whole run before any variant reported. Retriever-drop variants are now generated only
  when more than one retriever is configured — dropping the only one does not answer
  "what did this retriever add?", it removes retrieval entirely.
- `xwalk ablate` silently excluded any retriever whose built name differs from its job
  spec. The baseline took names from the spec (`name or kind`) while the factory filtered
  on the built retriever's `name`, and an unnamed `kind: dense` spec reads as `dense` but
  builds as `dense:<model>`. Every row was measured against a configuration missing that
  retriever, with no error. Names now come from the built retrievers.

### Added

- Full documentation under [`docs/`](docs/README.md): guides for getting started,
  evaluation, review, prompt optimisation, extending, and troubleshooting; a
  component-by-component map; and API reference pages for every subsystem.

## [0.1.0] — 2026-07-28

First public release. The API is expected to move before 1.0.0.

### Added

**Matching**

- `Matcher`: retrieve → select → score → verify, with query reformulation between
  attempts and per-stage degradation rather than whole-record failure.
- Opaque candidate keys (`C01`, `C02`, …) with exact-only resolution. The model never
  answers with an identifier, so a malformed answer cannot resolve to a real target
  record — it becomes `UNRESOLVED_OUTPUT` and routes to review.
- `MatchStatus` and `DecisionReason` as separate axes, with deterministic status
  derivation. `verify_band` is a cost control; `accept_at` and `review_floor` are
  classification. Tuning one does not disturb the other.
- Contract-safe prompt skeletons: domain knowledge enters through validated YAML slots,
  and `validate_contract` refuses any prompt whose machine-readable output contract has
  been damaged.

**Retrieval**

- `BM25Retriever` on Tantivy, with a raw-tokenizer exact-match field so a whole-string
  label or synonym hit outranks a document that merely contains the term.
- `DenseRetriever` with a pluggable `Encoder` protocol — an API embedding service needs
  no torch — behind `xwalk[dense]`.
- Reciprocal rank fusion with per-retriever evidence retained on every candidate.

**Sources and providers**

- CSV, TSV, and JSONL sources; OBO (no dependency) and OWL (`xwalk[ontology]`); SQL
  (`xwalk[sql]`), streamed rather than materialised.
- `OpenAICompatClient` for any OpenAI-compatible endpoint, with declared-not-trusted
  capabilities, structured-output fallback, and exponential backoff.
- `LiteLLMClient` behind `xwalk[litellm]`.

**Runs**

- SQLite WAL run ledger with resume: re-running skips completed records and re-runs
  anything whose source record or run configuration changed.
- Run fingerprinting over templates, prompts, store, retrievers, model, and policy —
  deliberately excluding credentials, output paths, and concurrency.
- An immutable review overlay that refuses to apply against a mismatched snapshot.
  Review never overwrites model output.

**Evaluation**

- Operational metrics: accepted precision, automatic coverage, review rate, recall at any
  status, no-match precision and recall, cost and latency per record.
- Three-way retrieval-ceiling decomposition — never retrieved, truncated by the selector
  budget, misjudged — because collapsing a budget miss into "retrieval failure" sends
  users to fix the wrong thing.
- Threshold curves re-derived from recorded confidences, and a calibration warning when
  those confidences do not separate correct from incorrect.
- Component ablation and run comparison.

**Prompt optimisation**

- LLM-assisted drafting of slots — never prompt text — validated before anything is
  written.
- A three-partition optimiser: prompt-train supplies the failures shown to the model,
  validation chooses the retained round, and test is evaluated exactly once at the end.
- Cost estimated and printed before spending, with `max_calls` as a hard stop.

**Interfaces**

- A YAML job spec that is serialized constructor arguments and nothing more. Credentials
  are named environment variables; an inline `api_key` is rejected at load time.
- `xwalk` CLI: `index`, `match`, `eval`, `compare`, `ablate`, `prompts`, `review`.
- Five worked examples, four of them slices of the datasets from the OntoRAG paper.

### Known limitations

- Validated end to end against a live provider on a 20-record example, not at production
  volume.
- The CI platform matrix (macOS, Windows, aarch64) is declared and scripted but has not
  yet executed.
- Pre-1.0: expect the API to change.

[Unreleased]: https://github.com/jan3657/xwalk/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/jan3657/xwalk/releases/tag/v0.1.1
[0.1.0]: https://github.com/jan3657/xwalk/releases/tag/v0.1.0

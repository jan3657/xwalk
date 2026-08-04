# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 onward. Before
1.0.0, minor versions may contain breaking changes.

## [Unreleased]

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

[Unreleased]: https://github.com/jan3657/xwalk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jan3657/xwalk/releases/tag/v0.1.0

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 onward. Before
1.0.0, minor versions may contain breaking changes.

## [Unreleased]

### Added

- A decider path: a `decider:` block on a job routes matching through a decision model
  that answers typed yes/no questions instead of writing prose — screen every retrieved
  candidate, choose among the survivors, then gate the choice on the identity-bearing
  properties declared in `properties:`. `xwalk fit` fits the acceptance thresholds on a
  gold sample, results carry their `signals`, and the written output has a `cost_usd`
  column, so a run's bill is a number in the results rather than an estimate.
- `scripts/retrieval_recall.py`: an offline harness for recall@k of a job's retrievers
  against a gold sample, with no model calls.
- `xwalk fit --holdout` fits on half the labelled rows and scores the recommendation on
  the other half; `--write-job OUT.yaml` writes a copy of the job with the fitted
  `accept_at`, `property_floor` and `choose_at` (the copy loses the YAML comments).
- BM25 retriever options `analyzer: en_stem` and `fuzzy_distance`, both opt-in; the
  sample jobs keep the defaults because they did not lift recall enough.
- A dense `intfloat/multilingual-e5-small` retriever (`limit: 20`) beside BM25 on the
  four sample decider jobs; the Ref_zivila job keeps BM25 alone. The four sample
  `job_jev.yaml` files and `scripts/run_jev_eval.sh` now need `xwalk[dense]` and the
  `intfloat/multilingual-e5-small` model in the HuggingFace cache. Tokens per record rise
  about 1.5-2.8x on those samples (against B1 alone) in exchange for the accuracy gain.
- An opt-in `decider.rewrite` block: when the screen finds nothing, an LLM proposes new
  queries and Jev screens what they retrieve. No shipped job sets it.
- `examples/ref_zivila/gold_adjudicated.csv` (39 rows from the adjudicated
  disagreements, built by `scripts/adjudication_to_gold.py`) and
  `scripts/hop_headroom.py`, which measures how many retrieval misses a graph hop could repair.

### Changed

- The screen's shared preamble moves into the decision state (`questions_version` 2), so
  each candidate's question is short. The run fingerprint covers the question version,
  so a decider run recorded before this change does not resume and re-runs from the start.
- The four sample decider jobs carry thresholds fitted with `xwalk fit --holdout`.
  Ref_zivila's fitted threshold was measured but not applied: its job keeps `accept_at`
  0.85 pending a spot-check (see the recall-and-cost spec's Results).
- The CI platform matrix now executes rather than merely being declared. macOS 14,
  Windows, and Linux on 3.10/3.11/3.12, plus aarch64 and each optional extra, all pass
  as of v0.1.1 — which retires the second of the limitations listed under 0.1.0.
- README badges, and a release runbook that covers a dropped tag trigger and a refused
  trusted-publisher exchange.
- `TemplateSet.fingerprint` and `PromptSet.fingerprint` now cover their newly defaulted
  fields (`queries` and `properties`). Runs recorded before this change have a different
  fingerprint, so they will not resume and will re-run from the start; a version bump
  would have had the same effect.

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

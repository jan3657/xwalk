# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 onward. Before
1.0.0, minor versions may contain breaking changes.

## [Unreleased]

Merged after 0.2.0rc1 was cut: the decider path, developed on the `jev-decider` and
`jev-recall-cost` branches against 0.1.1 and adapted to the 0.2 contracts
(`docs/claude-upgrade/CONTRACTS.md`). Accuracy and cost figures quoted in
`docs/reference/decide.md` were measured on those branches against the live Jev endpoint
before the merge; nothing was re-measured for this entry, and the merged tree was tested
offline only (FakeDecider, no paid calls).

### Added

- `xwalk ui`: a local web app over `xwalk.ops` (`xwalk.ui`), standard library only, for
  exploring jobs and runs: workspace discovery, validation, record previews through the
  templates, index search, background match and cluster runs with live progress and
  cancellation, results with per-record explanations (attempts, keyed candidates, scores,
  verifier, raw output), in-page review applied through `ops.review_apply`, evaluation
  against gold labels, and downloads. Runs default to offline stand-ins (the scripted
  quickstart reply, `FakeDecider`, and a word-overlap clustering judge), labelled
  `offline_model`; the job's endpoint requires a call limit and can be disabled with
  `--offline-only`. Local by default, with a per-process API token, a Host check and
  workspace path confinement. See `docs/guide/ui.md`.
- A decider path: a `decider:` block on a job routes matching through a decision model
  (TypeSafe's Jev, `JevClient`) that answers typed questions with probabilities instead
  of writing prose -- screen every retrieved candidate, choose among the survivors, then
  gate the choice on the rubric and on the identity-bearing properties declared in the
  slots' `properties:`. `DecisionMatcher` returns the same `MatchResult` as `Matcher`;
  results carry their calibrated `signals`. `FakeDecider` runs it offline.
- `xwalk fit` fits `accept_at`, `property_floor` and `choose_at` on a completed decider
  run and gold labels; `--holdout` fits on half the labelled rows and scores the
  recommendation on the other half; `--write-job OUT.yaml` writes a copy of the job with
  the fitted thresholds (the copy loses the YAML comments). It supports `--json`.
- `Usage.cost_usd` (provider-reported cost; 0.0 when a provider reports none) and a
  trailing `cost_usd` column in `mapping.csv`; `xwalk eval` reports `mean_cost_usd`.
- `templates.queries`: several retrieval queries per source, fused by reciprocal rank
  (decider jobs).
- BM25 retriever options `analyzer: en_stem` and `fuzzy_distance`, both opt-in and part
  of the index identity only when set; the sample jobs keep the defaults.
- An opt-in `decider.rewrite` block: when the screen finds nothing, an LLM proposes new
  queries and Jev screens what they retrieve. No shipped job sets it.
- `xwalk.evaluate.recall` and `scripts/retrieval_recall.py`: offline recall@k of a job's
  retrievers against gold, with no model calls.
- Decider example jobs (`job_jev.yaml`) for cafeteria_fcd, chebi, ncbi_disease, nlm_gene
  and Ref_zivila FoodOn. The four sample jobs add a dense `intfloat/multilingual-e5-small`
  retriever beside BM25 and so need `xwalk[dense]`. cafeteria_fcd and ncbi_disease keep
  their decider `properties:` in `slots_jev.yaml`, so the slots files pinned by the
  benchmark manifests are unchanged.
- `examples/ref_zivila/gold_adjudicated.csv` (built by `scripts/adjudication_to_gold.py`)
  and the evaluation scripts `scripts/run_jev_eval.sh`, `compare_gates.py`,
  `screen_recall.py` and `hop_headroom.py`.

### Changed

- Decider jobs run through `xwalk.ops` like LLM jobs: strict validation of the
  `decider:` block and the decider `policy:` (unknown keys with did-you-mean, ranges,
  `shortlist_floor <= accept_at`), `--json` envelopes, run states, exit codes, run
  directories bound to one fingerprint (whose components are recorded in the manifest,
  without secrets), index identity checks, and `validate` (credentials checked for
  presence only). Decisions are cached in the run's ledger (`llm_cache` table), so a
  resumed or `--no-resume` run replays them; the ledger schema is unchanged (v2).
- Decider provider errors follow CONTRACTS.md section 3: a fatal error (auth, unknown
  model, a spent call limit) from the decider or the rewrite LLM is a
  `fatal_provider_failure` that aborts the run without committing the record (exit 3);
  a malformed response or exhausted retries is a per-record `provider_failure`. On the
  branch a fatal decider error raised out of `DecisionMatcher.match`.
- `--max-calls` covers the decider path: Jev requests (each HTTP retry counts) and
  rewrite LLM requests share one limit. A failed decision request counts as a call with
  unknown usage instead of nothing; cache replays count as `cache_hits`.
- A job needs exactly one of `llm:` or `decider:`. A decider job refuses `selector:`; an
  `llm:` job refuses `templates.queries` with more than one entry (the LLM matcher
  searches one query per attempt). `xwalk ablate` and `xwalk prompts` refuse decider
  jobs (exit 2).
- `TemplateSet.fingerprint` and `PromptSet.fingerprint` are unchanged for jobs that do
  not use `queries` or slot `properties`, so 0.2.0rc1 runs keep resuming.

## [0.2.0rc1] — 2026-10-05

Release candidate for 0.2.0. Not published: tagging and uploading are the maintainer's
decision (see `docs/releasing.md`). Install it from the repository or a built wheel; once
on PyPI, `pip install --pre xwalk`.

### Release notes

**Why upgrade.** 0.2 makes runs trustworthy to repeat and to automate. Matching bugs that
could accept a non-finite confidence, drop candidates from prompts, or let provider
errors escape are fixed. A resumed run now exports exactly one current row per source,
keeps everything else as history, aborts cleanly on a fatal provider error, and never
mixes configurations in one run directory or reuses an incompatible index. Every command
goes through one operations layer with strict job validation, `--json` output and a hard
per-invocation `--max-calls` limit.

**New.** `xwalk init` (an offline quickstart bundled in the wheel), `validate`,
`inspect`, `explain`, `export --view raw|reviewed|history`, `search`, `results`;
`xwalk.ops` for Python; `--max-calls`; an optional MCP server (`xwalk[mcp]`).

**Experimental.** `xwalk cluster` (flat equivalence clustering) is tested with a scripted
model only and has not been evaluated on a real model. Its thresholds are provisional.

**Not measured yet.** The benchmark harness (repository only) has run offline smoke
checks and non-LLM baselines. No real-model accuracy, cost or speed claim is made for
this release; real-model results are pending (`docs/benchmarks.md`).

**Before upgrading, read "Changed".** Breaking changes are marked **Compatibility**. The
ones most likely to affect you: exit codes (invalid job, missing credential or extra
now `2`; failed rows `3`; Ctrl-C `130`); job files with unknown keys no longer load;
indexes built by 0.1 are refused until rebuilt with `--rebuild-index`; opening a 0.1.1
ledger upgrades it in place after saving `ledger.sqlite.v1-backup`; run fingerprints
changed for some jobs, so those runs recompute once.

**Known limitations.** Matching is many-to-one only. `eval`, `compare`, `ablate` and
`prompts` are not yet in `xwalk.ops`, and `ablate`/`prompts optimize` build their own
indexes and take no `--max-calls`. Retries inside LiteLLM are not counted. Cluster runs
have no `inspect`, `explain` or review-apply. The MCP host configurations are documented
but untested.

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
  now returns 3.
- Misspelled or misplaced job-file keys were ignored: `accept_att: 0.9` ran with the
  default `accept_at`. Every job spec now forbids unknown fields (with a did-you-mean
  hint), range-checks numbers, rejects fields that do not apply to the declared `kind`,
  and refuses duplicate retriever names; `load_job` raises `JobValidationError` listing
  every issue.
- `xwalk match` rebuilt every index on every invocation and read the target collection
  twice. Targets are read once; a compatible persisted index is opened without calling
  the encoder, an absent one is built, and an incompatible one is refused (exit 3)
  unless `--rebuild-index` is given. Index metadata now stores its identity components
  (doc template, record digest, BM25 `exact_fields` and normalization version; dense
  encoder name, revision, dimension, `normalize`, prefixes, max sequence length).
- `xwalk match --out D` silently mixed a different configuration into an existing run
  directory. A directory holding another run fingerprint, or unrelated files, is now
  refused (exit 3) naming the changed fingerprint components; nothing is overwritten.
- Ctrl-C and unexpected exceptions in the CLI: Ctrl-C now exits 130 after the run
  directory records `interrupted`; any other exception becomes an error result
  (exit 3) instead of escaping or printing a traceback.
- The usage reported for an interrupted or aborted invocation omitted calls that were
  in flight when it stopped. `xwalk match`/`ops.run` now count them as calls with
  unknown usage. (`BatchReport.usage` still covers finished records only.)
- Credentials passed to `LiteLLMClient` as keyword arguments (`api_key`, headers, ...)
  and userinfo or key parameters in an `OpenAICompatClient` base URL fed the LLM
  fingerprint, so rotating a key invalidated resumable runs. They are excluded now.
- The run fingerprint ignored dense encoder settings: reopening an index with another
  query prefix, document prefix or normalisation reused old results. The live encoder's
  identity (`DenseRetriever.encoder_identity`) is now part of it. `JobSpec.run_fingerprint`
  also used the default fallback retrieval depth instead of the one handed to `Matcher`.

- Resuming a finished `xwalk cluster` run with `failed` records made no calls and kept
  them failed (exit 3). Resume now retries them from the exported state, as `match`
  retries failed records, and stores the result as a new selected state revision;
  earlier decisions stay as history.
- A reached `--max-calls` limit was listed once per in-flight record in a match run's
  `errors`. It is now one `call_limit_reached` entry.
- The `dev` extra allowed pytest 8.0 with pytest-asyncio 0.23.0, which crash at test
  collection. It now requires `pytest>=8.2` and `pytest-asyncio>=0.24`.

### Added

- Optional MCP server: `pip install 'xwalk[mcp]'` (official SDK, `mcp>=2.3,<3`), then
  `xwalk mcp` serves six tools over stdio -- `validate_job`, `search_candidates`,
  `match_records`, `get_run`, `list_results`, `explain_result` -- each calling one
  `xwalk.ops` operation and returning the `--json` envelope as structured content with
  a declared output schema. `match_records` requires `max_calls` (capped by
  `--max-calls-cap`, default 500) and processes at most 100 records per call;
  `list_results` pages at most 200 rows; `search_candidates` returns at most 100.
  `--offline-model` answers with a fixed scripted reply for demos and tests. The base
  install does not import `mcp`; without it `xwalk mcp` exits 2 naming the extra. See
  `docs/guide/mcp.md`.
- `xwalk search QUERY --job --index` / `ops.search`: the fused retrieval candidates for
  a query, without a model call. `xwalk results --run [--status] [--offset] [--limit]` /
  `ops.list_results`: one bounded page of a run's current view
  (`Ledger.iter_current_summaries`). `ops.failure_result` maps an escaped exception to
  an error envelope (shared by the CLI and the MCP server).
- Experimental flat equivalence clustering: `xwalk cluster --job cluster.yaml --out DIR`
  (`kind: cluster` job files), `ops.cluster`, and `xwalk.cluster.run_clustering`. Groups
  one collection's equivalent records with LLM decisions under bounded candidate lists
  and `--max-calls`; outcomes `assigned`, `singleton`, `needs_review`, `failed`; verified
  cluster merges without transitive closure; bounded refinement with a recorded stop
  reason and the selected revision exported; one SQLite transaction per step, so an
  interrupted run resumes to the same result; exports `members.csv`, `clusters.csv`,
  `unresolved.csv`, `decisions.jsonl`. Tested with a scripted judge only; not evaluated
  on a real model. See `docs/guide/clustering.md`.
- `xwalk validate` accepts clustering jobs; `xwalk match` refuses one with
  `wrong_job_kind` (exit 2).
- `xwalk.ops`: one operations layer (`validate`, `index`, `run`/`run_async`, `inspect`,
  `explain`, `export`, `review_export`, `review_apply`, `init`) returning `OpResult`, the
  versioned machine-readable result; the CLI only formats it. The high-level Python
  route is `ops.run("job.yaml", out=..., max_calls=...)` (see `examples/quickstart.py`).
- `--json` on every command: exactly one JSON object (schema 1) on stdout, on success
  and on failure; warnings, errors and progress go to stderr.
- New commands: `xwalk init [DIR]` copies the bundled quickstart job;
  `xwalk validate --job` (alias `doctor`) is an offline preflight that never calls a
  model and reports credentials by name and presence only; `xwalk inspect --run`;
  `xwalk explain SOURCE_ID --run`; `xwalk export --run --view raw|reviewed|history --out`.
  The reviewed export applies the review overlay and keeps the model's decision in the
  same row; the history export adds every review decision with its state.
- `xwalk match --max-calls N` and `xwalk.llm.budget` (`CallBudget`, `BudgetedLLM`,
  `CallLimitExceeded`): a per-invocation cap on upstream requests, reserved before each
  dispatch so concurrent records cannot exceed it, counting client retries and
  rewrites. Reaching it aborts the run (exit 3, error `call_limit_reached`); resume
  continues. `xwalk match --rebuild-index`, `xwalk index --rebuild-index`.
- Retriever specs gain `revision` and `normalize` (dense); `JobSpec.build_encoder`;
  `IndexMismatchError`, `index_components` and `read_components` for both engines;
  `BM25Retriever.open(expected=..., default_limit=...)`,
  `DenseRetriever.open(expected=...)`.
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
- A pilot benchmark harness in `benchmarks/` (repository only, not packaged): pinned
  data manifests with constructed no-match variants, separate matching, pairwise and
  clustering tracks, a clustering scorer for any `item_id,cluster_id[,outcome]` file
  (`python -m benchmarks.cluster_eval`), and `python -m benchmarks.run --smoke` for
  offline SYNTHETIC checks. Real-model results are pending. See `docs/benchmarks.md`.

### Changed

- **Compatibility (exit codes):** an invalid or unreadable job file, a missing
  credential variable, a missing optional extra, and a bad flag value (`--role`) now
  exit 2 instead of 3. Ctrl-C exits 130. `xwalk inspect` exits with the code the run
  would have produced (0, 1 or 3).
- **Compatibility:** job files that set unknown keys, kind-inapplicable fields (for
  example `query_prefix` on a bm25 retriever) or an unknown `llm.profile` no longer load.
- **Compatibility:** a job file (match or cluster) that gives the same key twice in one
  mapping no longer loads (`job_yaml_invalid`, "duplicate key ..."); YAML would otherwise
  keep the last value silently. YAML syntax errors report a position instead of quoting
  the offending line.
- **Compatibility:** a prompt slots file that gives the same key twice no longer loads
  (`load_slots` raises `ValueError`; `xwalk validate` reports `prompts_invalid`).
- **Compatibility:** a retriever `name` must start with a letter or digit and use only
  letters, digits, `.`, `_` and `-` (at most 64 characters). It names the index
  directory, and `../x` placed the index outside it.
- **Compatibility:** `xwalk cluster --max-calls N` with `0 < N < 2 + pool.max_expansion_pages`
  is refused (exit 2): a step commits only when it finishes, so such a cap never made
  progress.
- **Compatibility:** `xwalk match` warnings (review bucket, duplicate targets) go to
  stderr; stdout carries the summary (or the JSON envelope).
- **Compatibility:** index fingerprints, and therefore run fingerprints, changed; an
  index built by an earlier version is refused by `xwalk match` until rebuilt with
  `--rebuild-index` (or into a fresh directory). `BM25Retriever.build` and
  `DenseRetriever.build` materialise their input records as a list.
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

Tagged and released on GitHub. The PyPI upload failed (`invalid-publisher`: no Trusted
Publisher was registered), so xwalk 0.1.x is not on PyPI; this entry previously called it
published. 0.1.0 was tagged but never uploaded, and its tree carries the three bugs below.

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

[Unreleased]: https://github.com/jan3657/xwalk/compare/v0.2.0rc1...HEAD
[0.2.0rc1]: https://github.com/jan3657/xwalk/compare/v0.1.1...v0.2.0rc1
[0.1.1]: https://github.com/jan3657/xwalk/releases/tag/v0.1.1
[0.1.0]: https://github.com/jan3657/xwalk/releases/tag/v0.1.0

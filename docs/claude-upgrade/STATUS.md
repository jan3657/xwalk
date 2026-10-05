# Execution status

Task 00 completed 5 October 2026. Baseline: `BASELINE.md`. Contracts: `CONTRACTS.md`.

| Task | State | Starting balance | Ending balance | Branch or commit | Evidence |
|---|---|---|---|---|---|
| 00 Baseline and contracts | Done | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | BASELINE.md: 872 passed / 7 skipped (base), ontology 6, sql 10, ruff+mypy clean, build+wheel OK; dense/integration not run |
| 01 Matching correctness | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11, fresh venv `pip install -e ".[dev]"`. New `tests/test_matching_regressions.py` (audit cases 1-6 + generation precedence): 28 of 34 failed before the fix, 34 passed after. `python -m pytest -q -m "not integration"`: 946 passed, 14 skipped, 1 deselected; `python -m pytest -q`: 946 passed, 15 skipped; `ruff check src tests examples scripts`: all passed; `ruff format --check ...`: 110 files formatted; `mypy`: no issues in 64 files. FakeLLM/MockTransport only, no paid calls; dense/ontology/sql/integration not run. `audit/diagnose_audited_source.py` no longer runs (imports removed private `_clamp`); superseded by the package tests |
| 02 Persistence and lifecycle | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"`. New `tests/test_lifecycle.py` (21), `tests/test_ledger_compat.py` (6, on the preserved 0.1.1 fixture), plus additions to test_ledger/test_review/test_batch/test_fingerprint/test_cli. Before the fix: 18 failed in test_batch+test_ledger and 3 new modules failed at import (new API); the two credential tests failed against the old clients. After: `python -m pytest -q`: 996 passed, 15 skipped; `-m "not integration"`: 996 passed, 14 skipped, 1 deselected; ruff check/format clean (113 files); mypy clean (64 files). FakeLLM/ScriptedRetriever only, no paid calls; dense/ontology/sql/integration not run |
| 03 Shared operations and CLI | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"` (+ hatchling, now in the dev extra). Reproduced before the fix on `ae55e4c`: a misspelled `policy.accept_att` loaded with `accept_at=0.6`. New tests: `test_job_validation.py` (19), `test_call_budget.py` (7), `test_ops.py` (30), `test_cli_json.py` (11), plus test_bm25/test_packaging (wheel contents) additions; 4 CLI tests migrated to the contract exit codes / stderr warnings. `python -m pytest -q`: 1065 passed, 15 skipped; `-m "not integration"`: 1065 passed, 14 skipped, 1 deselected; ruff check/format clean (120 files); mypy clean (67 files). FakeLLM/MockTransport/toy encoders only, no paid calls; dense extra not installed (dense paths tested with fake encoders); ontology/sql/integration not run |
| 04 Flat clustering | Done, **experimental** (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"`. New `src/xwalk/cluster/` (engine, incremental pool index, SQLite store, prompts, job, ops operation), `xwalk cluster`, `ops.cluster`. New tests: `test_cluster_engine.py` (30), `test_cluster_ops.py` (16), `test_cluster_pool.py` (7), fixture `tests/fixtures/cluster_tiny/`; all with a scripted judge (FakeLLM handler), no paid calls. `python -m pytest -q`: 1124 passed, 15 skipped; `-m "not integration"`: 1124 passed, 14 skipped, 1 deselected; ruff check/format and mypy clean. Gates: all implementation gates met; real labelled sample **not** evaluated (no endpoint) -> experimental. Decision record: `CLUSTERING_DECISIONS.md` (incl. scaling table) |
| 05 Benchmark and performance | Done (awaiting review; real eval pending) | unreported | unreported | worktree branch `worktree-agent-ad4549e848904883e` (off `d8fa82f`) | Python 3.11 venv `pip install -e ".[dev,ontology]"`. `python -m benchmarks.run --smoke` (SYNTHETIC judge, offline) on 4 pilot variants, raw output `benchmarks/results/smoke/` at `cd88923`; new `tests/test_benchmarks.py` (17). Profile + experiment in `benchmarks/results/perf/`: per-call Jinja recompilation in `PromptSet._render` is ~70% of xwalk's own CPU (0.440 s -> 0.188 s median with a cached template, prompts identical; ~1.5% at 200 ms/call). No library change made (fix is in `prompts/`, outside this task's file set). `python -m pytest -q`: 1097 passed, 3 skipped; `-m "not integration"`: 1097 passed, 2 skipped, 1 deselected; ruff check/format clean on src tests examples scripts benchmarks (133 files); mypy clean (67 files; benchmarks/ also strict-clean, 12 files). Without rdflib the cafeteria test skips. No paid calls; dense/LinkTransformer/real-model not run |
| 06 Optional MCP | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"` + `mcp==2.3.0` (latest on PyPI at implementation; extra `xwalk[mcp]` = `mcp>=2.3,<3`). New `src/xwalk/mcp_server.py` (MCPServer, 6 tools), `xwalk mcp`, `ops.search`/`ops.list_results`/`ops.failure_result`, `xwalk search`/`xwalk results`. New tests: `test_ops_read.py` (16, base), `test_mcp.py` (9: 7 marked `mcp` via the SDK's in-process and stdio-subprocess clients plus a raw JSON-RPC stdout check; 2 missing-extra tests unmarked). `python -m pytest -q`: 1171 passed, 16 skipped; `-m "not integration"`: 1171 passed, 15 skipped, 1 deselected; `-m mcp`: 7 passed; without mcp installed `test_mcp.py`: 2 passed, 7 skipped; ruff check/format clean on src tests benchmarks examples scripts (151 files); mypy clean (79 files) with and without mcp installed. Offline scripted model only, no paid calls; host configs (Claude Code/Desktop) documented but not exercised |
| 07 Documentation and release | Done (awaiting review; not tagged or published) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv. Version 0.2.0rc1. `python -m build`: sdist + wheel, `twine check` PASSED; wheel has the 4 prompt skeletons, the 5 quickstart files, `py.typed`, `ops.py`, `mcp_server.py`, LICENSE, no tests. Fresh venv in the session scratchpad (outside the repo), `pip install <wheel>`: `pip check` clean, `xwalk --version` 0.2.0rc1, no heavy module importable (torch/numpy/rdflib/sqlalchemy/faiss/litellm/mcp), `xwalk mcp` exits 2 naming the extra. README quickstart run command for command from that install in an empty directory: `xwalk init demo`, `xwalk validate --job demo/job.yaml --no-credentials`, `python quickstart.py` (prints `complete {'matched': 3, 'unmatched': 1, 'total': 4} model calls: 7`), `xwalk inspect --run demo/run`, `xwalk explain --run demo/run s4`, `xwalk export --run demo/run --view reviewed --out demo/reviewed.csv`: all exit 0; also via `scripts/check_readme_quickstart.py --bin <venv>/bin`. `python -m pytest -q`: 1179 passed, 16 skipped; ruff check/format clean (src tests benchmarks examples scripts, 152 files); mypy clean (80 files). No paid calls; real-provider route documented, not executed |
| 08 Independent review | Done: matching ready as 0.2.0rc1, clustering experimental, MCP optional (not tagged or published) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` (reviewed `c97b748`) | `REVIEW.md`. On `c97b748`: pytest 1179 passed / 16 skipped; ruff, format and mypy clean; build + `twine check` PASSED; fresh-venv wheel install clean (`pip check`, no heavy imports); README quickstart OK via the script and by hand; py3.12 with dev, mcp, ontology and sql extras: 1201 passed (1 environmental failure: no pip in uv venv); py3.10 with lowest runtime deps: 1171 passed. Probes in the scratchpad: 3,000 random trace on/off and usage scenarios (0 violations), lifecycle/limit/reviews, fatal/bug/cancel with concurrent work (no write after close), 0.1.1 fixture regenerated from f737099 (identical), generation params on the wire, call budget exact under concurrency, 429s and rewrites, dense index reuse with 0 encodes and mismatch refusal, CLI JSON matrix (23), raw MCP JSON-RPC from the wheel, clustering shuffles, chains, noisy judges, budget and cancel resume (partition invariant held). Fixed: duplicate YAML keys in job files (compat), YAML errors no longer quote file content, `inspect` exits 3 for failed rows without an invocation record, stale benchmarks note. Tests +6 (all failed before the fix); after: pytest 1188 passed / 16 skipped (incl. doc checks over REVIEW.md), gates clean. No paid calls |

Balances come from the actual promotional credit display, not a model estimate. Protect $55 for consolidation.

## Decisions recorded in task 00

- Starting commit `2dc58b1`; `src/` identical to audited `f737099`. All 8 diagnostic
  cases reproduce (output identical to `audit/observed.json`).
- Current-source export and history: CONTRACTS.md §2 (snapshot per invocation, one
  current row per source, history kept).
- Error and cancellation policy: CONTRACTS.md §3-4.
- Generation parameter precedence: CONTRACTS.md §6.
- Shared operation and JSON contracts, exit codes, run-dir collisions: CONTRACTS.md §8-9.
- First clustering relation and outcomes: CONTRACTS.md §10 (equivalence only;
  assigned/singleton/needs_review/failed).
- PyPI: xwalk is **not** published (404 on PyPI and TestPyPI). Trusted Publisher
  registration must be checked by Jan before tagging.
- Pilot benchmark: `examples/cafeteria_fcd/sample` (+ constructed no-match cases) and
  `examples/ncbi_disease/sample`. No inference endpoint configured; real eval pending.
- Integration branch: `release/0.2` (to be created by Jan from the accepted task 00
  commit). Next task: 01.

## Task 01 notes (for task 02 and later)

- `Matcher.match` never raises for provider errors. A fatal provider error returns
  `status=failed, reason=fatal_provider_failure`. `run_batch` still commits that result
  like any other: task 02 must implement the CONTRACTS.md section 3 abort (do not commit
  it, stop scheduling, cancel).
- `Usage` gained `unknown_calls` and `cache_hits` (serde reads old blobs with 0).
  `mapping.csv`/manifest/CLI do not show them yet; displays should use
  `Usage.describe_tokens()` (task 02/03).
- Per-stage Python overrides (`Scorer(max_tokens=...)`) are not part of
  `build_run_fingerprint` (batch.py, task 02). Job-built matchers never set them, so their
  effective values equal the client's, which the LLM fingerprint already covers.
- Unverified in-band score: when it is the only attempt the record is `failed`/
  `provider_failure` (retried on resume); otherwise at best `needs_review`/
  `provider_failure`.
- Retries inside LiteLLM (`num_retries`) are not observable; documented in `llm.md`.

## Task 02 notes (for task 03 and later)

- Snapshot/current view/history implemented as CONTRACTS.md section 2, with two
  recorded additions: reviews bind to the result *revision*, and a snapshot is recorded
  only when the source was read to the end (an aborted or interrupted invocation keeps
  the previous snapshot; with none, the latest row per source id). Ledger schema v2;
  0.1.1 ledgers are upgraded on open after a file copy to `ledger.sqlite.v1-backup`
  (any command that opens a ledger, including `eval`/`review`, triggers it). Fixture and
  its generator: `tests/fixtures/ledger_v0_1_1/` (made with the f737099 code).
- `RunState` adds `failed` to the contract's states (CONTRACTS.md section 8 updated):
  interrupted > aborted > failed > partial > complete. `run_batch` *returns* a report
  for a fatal abort (`run_state=ABORTED`, `errors=[BatchError("fatal_provider_failure",
  message, source_id)]`) and *re-raises* other exceptions and cancellation after writing
  exports and a manifest with `aborted`/`interrupted`. Task 03 must map these to the
  JSON envelope (`errors` already has `code`/`message`/`source_id`) and exit codes; the
  CLI has only an interim mapping (aborted/failed -> 3, partial -> 1) and does not yet
  catch the re-raised exception or return 130 on Ctrl-C.
- `manifest.json` now has `run_state`, `usage` (with `tokens` = `describe_tokens()`),
  `errors`, `limit`, `counts` (incl. `pending`), `removed_sources`, `history.results`,
  `snapshot`, `ledger_schema_version`, and `fingerprint_components` when the caller passes
  them (`xwalk match` does). Use `fingerprint_components` for the run-dir collision diff
  (CONTRACTS.md section 9); `run_batch` itself still lets several fingerprints share one
  ledger and does not refuse a mismatched `--out`.
- `xwalk match` now streams `job.build_source_records()` into `run_batch(limit=...)`;
  targets are still read twice (store and index build), and indexes are still rebuilt
  on every `match` (finding 13). `DenseRetriever.encoder_identity` feeds the run
  fingerprint from the live encoder, but the index meta/`open` validation of section 7
  (prefixes, normalize, revision) is untouched and belongs to task 03.
- An invocation left `running` by a killed process is marked `interrupted` when the next
  invocation begins. `BatchReport.usage` excludes calls made by records cancelled
  mid-flight (their usage is lost with the cancelled task).
- `mapping.csv` gained trailing `unknown_calls`, `cache_hits`; pending rows have
  status/reason `pending`. `results.jsonl` has no line for pending sources.
- Not done: no CLI for history (`Ledger.iter_history`/`export_history_jsonl` exist);
  `review export` still uses the CLI's `manifest.json` lookup.

## Task 03 notes (for task 04 and later)

- **Ops layer**: `src/xwalk/ops.py`. Operations return `OpResult` (`envelope()` is the
  `--json` schema 1: the CONTRACTS.md section 8 keys plus `data`) or raise `OpError`,
  whose `.result` is the error envelope; the CLI (`cli/main.py`) only formats. Reuse for
  clustering: `load_valid_job` (strict, exit 2), `_preflight` (files/extras),
  `_read_targets` (read once, duplicate ids -> exit 2), `plan_indexes` +
  `prepare_indexes` (open/build/refuse by components; a `PlannedIndex` stands in for a
  retriever when computing a fingerprint before touching disk), `_check_out_dir`
  (fingerprint collision with component diff), `_summarise_run`/`_run_exit_code` (exit
  mapping from run state + review count). A `cluster` operation should follow `run_async`:
  plan -> fingerprint -> collision check -> prepare -> execute -> summarise, and add one
  `cluster` subparser in `_build_parser` and `_DISPATCH`.
- **Call limits**: wrap the client in `BudgetedLLM(llm, CallBudget(max_calls))`
  (`xwalk.llm.budget`) once per invocation and give *every* clustering stage the wrapped
  client. Reservation is synchronous before dispatch; `OpenAICompatClient` reserves per
  HTTP request (retries, fallback) via `attach_call_budget`; other clients per
  `complete()`. A refusal raises `CallLimitExceeded` (an `LLMFatalError`, usage zero).
  Report `BudgetedLLM.usage` (includes in-flight-cancelled calls as `unknown_calls`), not
  a sum of committed results. Put a `CachingLLM` *outside* the budget wrapper or cache
  hits would spend the allowance. Limit is per invocation.
- **Usage/JSON**: `usage_to_dict` (batch.py) for envelopes; `tokens` is
  `describe_tokens()`. Error codes in use: see docs/reference/cli.md "Machine-readable
  output"; add clustering codes there.
- **Index identity**: `bm25.index_components` / `dense.index_components` (+
  `read_components`, `IndexMismatchError`, `component_differences`). The clustering pool
  index (CONTRACTS section 10.8, incremental) needs its own components, including an
  append count/digest, if it is persisted.
- **Exit-code migrations** (documented in CHANGELOG): invalid/unreadable job, missing
  credential, missing extra, bad `--role` now 2 (were 3); Ctrl-C 130; `inspect` exits with
  the run's code; match warnings moved to stderr.
- **Known limits**: `BatchReport.usage` still omits calls cancelled in flight (ops reports
  the complete figure); LiteLLM internal retries are neither counted nor limited; `eval`,
  `compare`, `ablate` and `prompts` are formatted as envelopes but still live in the CLI
  (not in ops) and `ablate`/`prompts optimize` always rebuild their own indexes and take
  no `--max-calls` (optimize keeps its own estimate-based `--max-calls`); `review apply`
  validation errors are reported as code `exception` (exit 3); a run directory created by
  0.1.x has no `fingerprint_components`, so a refusal there reports the component diff as
  unknown; indexes built before 0.2 are refused until `--rebuild-index`.
- Next recommended task: 04 (flat equivalence clustering), reusing the above.

## Task 04 notes (for task 05 and later)

- **Experimental.** Every implementation gate of the task is met with a scripted judge
  (`tests/cluster_helpers.Oracle`, a FakeLLM handler that answers from ground truth by
  parsing the real prompts). No real model was run: the labelled-sample gate is open and
  thresholds are uncalibrated. Task 05 can reuse `Oracle` for synthetic clustering
  benchmarks and should add a real labelled sample (B-cubed/pairwise metrics are not
  implemented yet) when an endpoint and budget exist.
- Decisions and the eight reconciled conflicts: `CLUSTERING_DECISIONS.md`. Contract
  additions (converged export, `mint_requires`, seed outcome): CONTRACTS.md section 10.
- Default `pool.mint_requires` is `bounded` (after the configured expansion). The draft
  default `retrieval_exhausted` deferred every late novel record whenever a common
  token was shared; found by a test, kept as an option.
- Scaling: the pool index is incremental pure-Python BM25 (+ optional brute-force dense
  head). 16,000 query+mint steps took 1.7 s; Tantivy rebuild-per-mint took 5.8 s for 250.
  Two quadratic per-search costs were found by the measurement and removed. Engine
  overhead with the scripted judge: about 1 ms per call (1,200 sources, 4,101 calls,
  4.0 s).
- Ops/CLI: `ops.cluster`/`cluster_async` delegate to `xwalk.cluster.operation`;
  `ops.validate` delegates for `kind: cluster` files; `config.parse_job` refuses a
  clustering job (`wrong_job_kind`); `config._issues_from` takes the root spec so
  "did you mean" works for cluster jobs. The CLI's Ctrl-C path reports a cluster run
  only as exit 130 (no run summary; the manifest records `interrupted`).
- Not done: `inspect`/`explain`/`review apply` for cluster runs, a clustering review
  overlay, concurrency (processing is sequential), cluster names, LLM-judged or gold
  metrics for clusters, any hierarchy.
- Next recommended task: 05 (benchmarks), with a small labelled clustering sample
  evaluated on a real endpoint before clustering loses its experimental label.

## Task 05 notes (for task 04, 07 and later)

- Harness: `benchmarks/` (repo only; the wheel packages `src/xwalk` only). Command:
  `python -m benchmarks.run --smoke` (about a minute; `--datasets ncbi_disease --limit 5`
  for seconds). Method, data and reading: `docs/benchmarks.md`.
- Everything that involves a model answer is SYNTHETIC (`benchmarks/synthetic_llm.py`, a
  string-similarity judge that never sees gold). Real results are PENDING:
  `python -m benchmarks.run --real --max-calls N` with `XWALK_BENCH_BASE_URL`,
  `XWALK_BENCH_MODEL`, `XWALK_BENCH_API_KEY_ENV` (or the example jobs' `llm` sections).
- Clustering (task 04): export assignments as `item_id,cluster_id,outcome` (CSV or
  JSONL; outcomes per CONTRACTS section 10) and score with
  `python -m benchmarks.cluster_eval --gold <gold_clusters.csv> --pred <file> [--pred
  <other order>] [--meta usage.json]`. Pilot gold clusters are written by the smoke run
  under `benchmarks/results/smoke/clustering/<dataset>/`. The food pilot is nearly all
  singletons (48 clusters / 50 mentions); use the disease pilot. Wiring xwalk clustering
  into `benchmarks/run.py` is a small follow-up once its API is final.
- Data findings: 11 of 50 `ncbi_disease` mentions have gold ids absent from the sample
  catalog (OMIM:215600, OMIM:261600) even after alias expansion; 12 of 50 are bare
  abbreviations (CT, FAP, MHP) with zero BM25 hits.
- Performance follow-up (measured here; applied afterwards in commit `d75463b`): cache compiled prompt
  templates in `xwalk/prompts/contract.py` `PromptSet._render` (it builds a Jinja
  Environment and compiles per call). Patch and numbers: `docs/benchmarks.md`
  "Performance"; reproduce with `python -m benchmarks.profile_workload --experiment
  prompt-template-cache --llm trivial`. CachingLLM miss coalescing was not justified (2/111
  and 8/130 duplicate requests; `xwalk match` does not use `CachingLLM`).
- Not done: dense retrieval (extra not installed), LinkTransformer (unexecuted adapter in
  `benchmarks/adapters/`), WDC/Splink/dedupe, real-model evaluation.

## Task 06 notes (for task 07 and later)

- **SDK**: `mcp` 2.x (2.3.0 tested). FastMCP is now `mcp.server.mcpserver.MCPServer`;
  results use snake_case fields (`structured_content`, `is_error`). The extra pins
  `<3`; 1.x is not supported. Tools are registered with `server.add_tool` (not the
  decorator) so `mypy --strict` passes with and without the SDK installed (`mcp.*` is in
  the `ignore_missing_imports` override; CI lint installs `.[dev,mcp]` to type-check
  against the real SDK).
- **Shape**: every tool returns `Annotated[CallToolResult, <Envelope model>]`: the
  `--json` envelope as structured content (validated by the SDK against the declared
  output schema) plus the same JSON as text. `isError` only when the operation raised
  (`OpError`, unexpected exception, schema-invalid arguments, `max_calls` over the cap);
  a performed operation (invalid job, run aborted at its limit, partial run) is a normal
  result whose `status`/`exit_code` say so, identical to the CLI. Recorded in
  `docs/guide/mcp.md`.
- **Bounds**: `match_records` requires `max_calls` (<= `--max-calls-cap`, default 500,
  enforced through the existing `BudgetedLLM`/`CallBudget`) and `limit` <= 100 records
  (`run_batch(limit=...)`: the rest stay `pending`, the run `partial`; calling again
  resumes). `ops.MAX_SEARCH_LIMIT` 100, `ops.MAX_PAGE_SIZE` 200. No background jobs.
  `explain_result` does not expose `--full`.
- **Offline**: jobs cannot name an offline model, so the server takes
  `--offline-model` (a launch flag the human sets, not a tool argument): the quickstart's
  fixed scripted reply via `FakeLLM(model="xwalk-offline")`, flagged on every result
  with an `offline_model` warning. Its LLM fingerprint differs from any real client, so
  an offline run directory is refused by a real-model run (and vice versa).
- **stdout**: the SDK's stdio transport moves fd 1 to stderr while serving; a startup
  failure (missing extra) prints to stderr only. Raw-protocol test confirms every stdout
  line is JSON-RPC.
- Not done: clustering tool (cluster runs have no inspect/explain yet), a path sandbox
  (`--root`) for tool arguments, progress notifications during `match_records`, the
  host configurations are untested. CLI vs MCP consistency is tested for `inspect`,
  `results` and `validate` envelopes (byte-identical JSON objects).
- Next recommended task: 07 (documentation and release).

## Task 07 notes (for task 08 and the release)

- **Docs**: README rewritten problem-first with an offline quickstart (FakeLLM script) and
  a separate real-provider route; the 0.1 constructor tour moved intact to
  `docs/guide/python-sdk.md`; new `docs/architecture.md` (layers, 11 invariants,
  contributor map); `docs/superpowers/README.md` summarises the superseded plans.
  Reconciled: index build/open behaviour (job-file, retrieval reference), troubleshooting
  exit codes and fatal errors, components (ops, MCP, budget, subcommand count), the
  prompt-optimise `max_calls` wording, getting-started install and extras, concepts
  (pending, cardinality).
- **README check**: the quickstart is marked `<!-- quickstart:begin/end -->`;
  `scripts/check_readme_quickstart.py` runs it exactly; `tests/test_docs.py` runs it
  in-repo, CI `build` and release `build` run it against the clean wheel install.
- **Release**: `release.yml` gained a `gates` job on the tagged commit (build needs it),
  wider wheel-content checks, artefact SHA-256s in the run summary, `pip check`, the README
  check. Not executed on GitHub (no tag pushed).
- **PyPI**: still 404 on 2026-10-05. The v0.1.1 publish log (run 30898407571) shows
  `invalid-publisher` with claims repo `jan3657/xwalk`, workflow `release.yml`,
  environment `pypi`; the workflow still sends these, so Jan must register the pending
  Trusted Publisher (docs/releasing.md step 3) before tagging `v0.2.0rc1`. The 0.1.1
  changelog entry no longer says it was published.
- **Test fix**: two cluster oscillation tests assumed which of two tied clusters the
  ambiguous record joins; the tie-break follows hash ids that include the version, so the
  bump broke one. They now read the joined cluster from the scripted judge.
- **Open**: the README install line uses the git URL of the default branch, which has
  0.2 only once this branch is merged; the README's relative links will not resolve on
  a PyPI project page; `xwalk inspect` shows the job's model name for a run made with an
  injected offline model. CONTRIBUTING's lint paths omit `benchmarks` (CI includes it).
- Next: task 08 (independent review), then the user's tag/publish decision.

## Task 08 notes (for the release)

- Verdict and evidence: `REVIEW.md`. No release blocker; remaining Medium items are a
  cluster run with `failed` members not retrying on resume, and the MCP tool paths
  having no sandbox (both outside the supported matching scope).
- Next: Jan merges the branch, registers the PyPI pending Trusted Publisher (`jan3657/xwalk`,
  `release.yml`, environment `pypi`), then tags `v0.2.0rc1` on the merged commit.

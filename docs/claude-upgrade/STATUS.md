# Execution status

Task 00 completed 5 October 2026. Baseline: `BASELINE.md`. Contracts: `CONTRACTS.md`.

| Task | State | Starting balance | Ending balance | Branch or commit | Evidence |
|---|---|---|---|---|---|
| 00 Baseline and contracts | Done | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | BASELINE.md: 872 passed / 7 skipped (base), ontology 6, sql 10, ruff+mypy clean, build+wheel OK; dense/integration not run |
| 01 Matching correctness | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11, fresh venv `pip install -e ".[dev]"`. New `tests/test_matching_regressions.py` (audit cases 1-6 + generation precedence): 28 of 34 failed before the fix, 34 passed after. `python -m pytest -q -m "not integration"`: 946 passed, 14 skipped, 1 deselected; `python -m pytest -q`: 946 passed, 15 skipped; `ruff check src tests examples scripts`: all passed; `ruff format --check ...`: 110 files formatted; `mypy`: no issues in 64 files. FakeLLM/MockTransport only, no paid calls; dense/ontology/sql/integration not run. `audit/diagnose_audited_source.py` no longer runs (imports removed private `_clamp`); superseded by the package tests |
| 02 Persistence and lifecycle | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"`. New `tests/test_lifecycle.py` (21), `tests/test_ledger_compat.py` (6, on the preserved 0.1.1 fixture), plus additions to test_ledger/test_review/test_batch/test_fingerprint/test_cli. Before the fix: 18 failed in test_batch+test_ledger and 3 new modules failed at import (new API); the two credential tests failed against the old clients. After: `python -m pytest -q`: 996 passed, 15 skipped; `-m "not integration"`: 996 passed, 14 skipped, 1 deselected; ruff check/format clean (113 files); mypy clean (64 files). FakeLLM/ScriptedRetriever only, no paid calls; dense/ontology/sql/integration not run |
| 03 Shared operations and CLI | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"` (+ hatchling, now in the dev extra). Reproduced before the fix on `ae55e4c`: a misspelled `policy.accept_att` loaded with `accept_at=0.6`. New tests: `test_job_validation.py` (19), `test_call_budget.py` (7), `test_ops.py` (30), `test_cli_json.py` (11), plus test_bm25/test_packaging (wheel contents) additions; 4 CLI tests migrated to the contract exit codes / stderr warnings. `python -m pytest -q`: 1065 passed, 15 skipped; `-m "not integration"`: 1065 passed, 14 skipped, 1 deselected; ruff check/format clean (120 files); mypy clean (67 files). FakeLLM/MockTransport/toy encoders only, no paid calls; dense extra not installed (dense paths tested with fake encoders); ontology/sql/integration not run |
| 04 Flat clustering | Not started | | | | |
| 05 Benchmark and performance | Not started | | | | |
| 06 Optional MCP | Not started | | | | |
| 07 Documentation and release | Not started | | | | |
| 08 Independent review | Not started | | | | |

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

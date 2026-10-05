# Execution status

Task 00 completed 5 October 2026. Baseline: `BASELINE.md`. Contracts: `CONTRACTS.md`.

| Task | State | Starting balance | Ending balance | Branch or commit | Evidence |
|---|---|---|---|---|---|
| 00 Baseline and contracts | Done | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | BASELINE.md: 872 passed / 7 skipped (base), ontology 6, sql 10, ruff+mypy clean, build+wheel OK; dense/integration not run |
| 01 Matching correctness | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11, fresh venv `pip install -e ".[dev]"`. New `tests/test_matching_regressions.py` (audit cases 1-6 + generation precedence): 28 of 34 failed before the fix, 34 passed after. `python -m pytest -q -m "not integration"`: 946 passed, 14 skipped, 1 deselected; `python -m pytest -q`: 946 passed, 15 skipped; `ruff check src tests examples scripts`: all passed; `ruff format --check ...`: 110 files formatted; `mypy`: no issues in 64 files. FakeLLM/MockTransport only, no paid calls; dense/ontology/sql/integration not run. `audit/diagnose_audited_source.py` no longer runs (imports removed private `_clamp`); superseded by the package tests |
| 02 Persistence and lifecycle | Not started | | | | |
| 03 Shared operations and CLI | Not started | | | | |
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

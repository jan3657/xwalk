# Execution status

Task 00 completed 5 October 2026. Baseline: `BASELINE.md`. Contracts: `CONTRACTS.md`.

| Task | State | Starting balance | Ending balance | Branch or commit | Evidence |
|---|---|---|---|---|---|
| 00 Baseline and contracts | Done | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | BASELINE.md: 872 passed / 7 skipped (base), ontology 6, sql 10, ruff+mypy clean, build+wheel OK; dense/integration not run |
| 01 Matching correctness | Not started | | | | |
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

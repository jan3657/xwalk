# Execution status

Task 00 completed 5 October 2026. Baseline: `BASELINE.md`. Contracts: `CONTRACTS.md`.

| Task | State | Starting balance | Ending balance | Branch or commit | Evidence |
|---|---|---|---|---|---|
| Engine batch 5: FoodOn policy (item 7) | Done (awaiting review; mbox fallback) | unreported | unreported | `engine/batch5-retrieval` | Test-first: 4 policy regressions failed; all 5 FoodOn jobs now pass strict schema/template/prompt loading without private data. Four LLM jobs use audit_rate 0.1 and verify_band [0.6, 0.8]; Jev retains its separate gate policy. Focused example tests: 10 passed. Final offline gate: 2172 passed, 40 skipped, 4 deselected; 8 failures and 40 errors, all 48 socket-bind PermissionErrors from the sandbox. Ruff check/format clean (226 files); mypy clean (128 files). Three commits in `/private/tmp/xwalk-b5.mbox`, with identical changes in this worktree (git metadata writes denied). No paid calls or downloads. |
| Engine batch 5: L2 rendered budget | Done (awaiting review; mbox fallback) | unreported | unreported | `engine/batch5-retrieval` | Test-first: 7 regressions failed. Render-once prefix budgeting includes key widths/separators; oversized-first review makes zero model calls; reason round-trips and replays; budget version 2 changes run identity. Focused select/matcher/policy/identity/replay: 187 passed; reduction/policy/select/matcher: 164 passed. Offline gate: 2167 passed, 40 skipped, 4 deselected; 8 failures and 40 errors, all sandbox socket-bind denials. Ruff check/format clean (226 files); mypy clean (128 files). No paid calls or downloads. Git metadata denied; commits prepared in scratch clone. |
| Engine batch 5: B2 retrieval | Done (awaiting review) | unreported | unreported | `engine/batch5-retrieval` | Test-first: 5 regressions failed; multi-query RRF, original-list retries/dedup, trace invariance and single-query compatibility (including captured HTTP bytes). Focused Matcher/config: 84 passed. Offline gate: 2156 passed, 40 skipped, 4 deselected; 8 failures and 40 errors, all sandbox socket-bind denials. Ruff check/format clean (226 files); mypy clean (128 files). HF/Transformers offline; FakeLLM/MockTransport only, no paid calls. |
| Engine batch 4: E1/E2 evaluation semantics | Done (reviewed by orchestrator gates; not pushed) | unreported | unreported | `engine/batch4-eval` | Test-first: 5 metrics/ceiling regressions, 28 replay regressions and 6 ops regressions failed before implementation; later shortlist attribution and artifact-rendering regressions also failed before fixes. Shared status-based no-match, unavailable/unreachable ceiling buckets, both production reducers in conditional replay, recorded/gold transitions, stage diagnostics and descriptive retry metrics; ops/CLI JSON envelope; docs and changelog updated. Python 3.12 venv with `PYTHONPATH=src` verified to import this worktree. Focused evaluation/benchmark tests: 125 passed, 1 skipped. Final offline gate: 2103 passed, 40 skipped, 4 deselected, 8 failed and 40 errors in 47.37 s; all 48 failures/errors are localhost socket binds denied by sandbox. Outside the Codex sandbox (orchestrator, Python 3.12 `.venv`, `PYTHONPATH=src`): `python -m pytest -q -m "not integration"`: 2151 passed, 40 skipped, 4 deselected; ruff check/format clean (222 files); mypy clean (126 files). Implemented by GPT-6 Astra via Codex; committed by the orchestrator because the Codex sandbox cannot write git metadata. No paid calls or downloads. |
| 00 Baseline and contracts | Done | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | BASELINE.md: 872 passed / 7 skipped (base), ontology 6, sql 10, ruff+mypy clean, build+wheel OK; dense/integration not run |
| 01 Matching correctness | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11, fresh venv `pip install -e ".[dev]"`. New `tests/test_matching_regressions.py` (audit cases 1-6 + generation precedence): 28 of 34 failed before the fix, 34 passed after. `python -m pytest -q -m "not integration"`: 946 passed, 14 skipped, 1 deselected; `python -m pytest -q`: 946 passed, 15 skipped; `ruff check src tests examples scripts`: all passed; `ruff format --check ...`: 110 files formatted; `mypy`: no issues in 64 files. FakeLLM/MockTransport only, no paid calls; dense/ontology/sql/integration not run. `audit/diagnose_audited_source.py` no longer runs (imports removed private `_clamp`); superseded by the package tests |
| 02 Persistence and lifecycle | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"`. New `tests/test_lifecycle.py` (21), `tests/test_ledger_compat.py` (6, on the preserved 0.1.1 fixture), plus additions to test_ledger/test_review/test_batch/test_fingerprint/test_cli. Before the fix: 18 failed in test_batch+test_ledger and 3 new modules failed at import (new API); the two credential tests failed against the old clients. After: `python -m pytest -q`: 996 passed, 15 skipped; `-m "not integration"`: 996 passed, 14 skipped, 1 deselected; ruff check/format clean (113 files); mypy clean (64 files). FakeLLM/ScriptedRetriever only, no paid calls; dense/ontology/sql/integration not run |
| 03 Shared operations and CLI | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"` (+ hatchling, now in the dev extra). Reproduced before the fix on `ae55e4c`: a misspelled `policy.accept_att` loaded with `accept_at=0.6`. New tests: `test_job_validation.py` (19), `test_call_budget.py` (7), `test_ops.py` (30), `test_cli_json.py` (11), plus test_bm25/test_packaging (wheel contents) additions; 4 CLI tests migrated to the contract exit codes / stderr warnings. `python -m pytest -q`: 1065 passed, 15 skipped; `-m "not integration"`: 1065 passed, 14 skipped, 1 deselected; ruff check/format clean (120 files); mypy clean (67 files). FakeLLM/MockTransport/toy encoders only, no paid calls; dense extra not installed (dense paths tested with fake encoders); ontology/sql/integration not run |
| 04 Flat clustering | Done, **experimental** (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"`. New `src/xwalk/cluster/` (engine, incremental pool index, SQLite store, prompts, job, ops operation), `xwalk cluster`, `ops.cluster`. New tests: `test_cluster_engine.py` (30), `test_cluster_ops.py` (16), `test_cluster_pool.py` (7), fixture `tests/fixtures/cluster_tiny/`; all with a scripted judge (FakeLLM handler), no paid calls. `python -m pytest -q`: 1124 passed, 15 skipped; `-m "not integration"`: 1124 passed, 14 skipped, 1 deselected; ruff check/format and mypy clean. Gates: all implementation gates met; real labelled sample **not** evaluated (no endpoint) -> experimental. Decision record: `CLUSTERING_DECISIONS.md` (incl. scaling table) |
| 05 Benchmark and performance | Done (awaiting review; real eval pending) | unreported | unreported | worktree branch `worktree-agent-ad4549e848904883e` (off `d8fa82f`) | Python 3.11 venv `pip install -e ".[dev,ontology]"`. `python -m benchmarks.run --smoke` (SYNTHETIC judge, offline) on 4 pilot variants, raw output `benchmarks/results/smoke/` at `cd88923`; new `tests/test_benchmarks.py` (17). Profile + experiment in `benchmarks/results/perf/`: per-call Jinja recompilation in `PromptSet._render` is ~70% of xwalk's own CPU (0.440 s -> 0.188 s median with a cached template, prompts identical; ~1.5% at 200 ms/call). No library change made (fix is in `prompts/`, outside this task's file set). `python -m pytest -q`: 1097 passed, 3 skipped; `-m "not integration"`: 1097 passed, 2 skipped, 1 deselected; ruff check/format clean on src tests examples scripts benchmarks (133 files); mypy clean (67 files; benchmarks/ also strict-clean, 12 files). Without rdflib the cafeteria test skips. No paid calls; dense/LinkTransformer/real-model not run |
| 06 Optional MCP | Done (awaiting review) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv `pip install -e ".[dev]"` + `mcp==2.3.0` (latest on PyPI at implementation; extra `xwalk[mcp]` = `mcp>=2.3,<3`). New `src/xwalk/mcp_server.py` (MCPServer, 6 tools), `xwalk mcp`, `ops.search`/`ops.list_results`/`ops.failure_result`, `xwalk search`/`xwalk results`. New tests: `test_ops_read.py` (16, base), `test_mcp.py` (9: 7 marked `mcp` via the SDK's in-process and stdio-subprocess clients plus a raw JSON-RPC stdout check; 2 missing-extra tests unmarked). `python -m pytest -q`: 1171 passed, 16 skipped; `-m "not integration"`: 1171 passed, 15 skipped, 1 deselected; `-m mcp`: 7 passed; without mcp installed `test_mcp.py`: 2 passed, 7 skipped; ruff check/format clean on src tests benchmarks examples scripts (151 files); mypy clean (79 files) with and without mcp installed. Offline scripted model only, no paid calls; host configs (Claude Code/Desktop) documented but not exercised |
| 07 Documentation and release | Done (awaiting review; not tagged or published) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` | Python 3.11 venv. Version 0.2.0rc1. `python -m build`: sdist + wheel, `twine check` PASSED; wheel has the 4 prompt skeletons, the 5 quickstart files, `py.typed`, `ops.py`, `mcp_server.py`, LICENSE, no tests. Fresh venv in the session scratchpad (outside the repo), `pip install <wheel>`: `pip check` clean, `xwalk --version` 0.2.0rc1, no heavy module importable (torch/numpy/rdflib/sqlalchemy/faiss/litellm/mcp), `xwalk mcp` exits 2 naming the extra. README quickstart run command for command from that install in an empty directory: `xwalk init demo`, `xwalk validate --job demo/job.yaml --no-credentials`, `python quickstart.py` (prints `complete {'matched': 3, 'unmatched': 1, 'total': 4} model calls: 7`), `xwalk inspect --run demo/run`, `xwalk explain --run demo/run s4`, `xwalk export --run demo/run --view reviewed --out demo/reviewed.csv`: all exit 0; also via `scripts/check_readme_quickstart.py --bin <venv>/bin`. `python -m pytest -q`: 1179 passed, 16 skipped; ruff check/format clean (src tests benchmarks examples scripts, 152 files); mypy clean (80 files). No paid calls; real-provider route documented, not executed |
| 08 Independent review | Done: matching ready as 0.2.0rc1, clustering experimental, MCP optional (not tagged or published) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` (reviewed `c97b748`) | `REVIEW.md`. On `c97b748`: pytest 1179 passed / 16 skipped; ruff, format and mypy clean; build + `twine check` PASSED; fresh-venv wheel install clean (`pip check`, no heavy imports); README quickstart OK via the script and by hand; py3.12 with dev, mcp, ontology and sql extras: 1201 passed (1 environmental failure: no pip in uv venv); py3.10 with lowest runtime deps: 1171 passed. Probes in the scratchpad: 3,000 random trace on/off and usage scenarios (0 violations), lifecycle/limit/reviews, fatal/bug/cancel with concurrent work (no write after close), 0.1.1 fixture regenerated from f737099 (identical), generation params on the wire, call budget exact under concurrency, 429s and rewrites, dense index reuse with 0 encodes and mismatch refusal, CLI JSON matrix (23), raw MCP JSON-RPC from the wheel, clustering shuffles, chains, noisy judges, budget and cancel resume (partition invariant held). Fixed: duplicate YAML keys in job files (compat), YAML errors no longer quote file content, `inspect` exits 3 for failed rows without an invocation record, stale benchmarks note. Tests +6 (all failed before the fix); after: pytest 1188 passed / 16 skipped (incl. doc checks over REVIEW.md), gates clean. No paid calls |
| 08a Review follow-up | Done (not tagged or published) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` (`a64d20e`, `53aea88`, `19cb3d1`, `c087538`) | Fixed REVIEW.md items: cluster resume retries failed sources (resume round from the exported state, history kept, interrupted round reproduces an uninterrupted one); cluster `--max-calls` below `2 + pool.max_expansion_pages` refused (exit 2); retriever names validated as safe identifiers; duplicate keys in slots files refused (shared `xwalk/_yaml.py` loader); `call_limit_reached` reported once; dev extra `pytest>=8.2`, `pytest-asyncio>=0.24` (lowest pair and 0.24 with pytest 8.3.5 each ran the suite in scratch venvs: 1205 passed / 24 skipped; old lowest pair INTERNALERROR). New tests +26: 21 failed before their fix, 5 guard that ordinary retriever names still load. Python 3.11 `.venv`: pytest 1214 passed / 16 skipped; ruff check/format clean (153 files); mypy clean (81 files); README quickstart check OK. Lax type coercion left open. No paid calls |
| 09 Decider-path merge | Done (awaiting review; not tagged or published) | unreported | unreported | `claude/zealous-heisenberg-nri5m0` (merge of `5e567d8` = `origin/jev-recall-cost~1`) | Python 3.11 `.venv`. 17 conflicts resolved, 0.2 contracts kept. New `tests/test_decide_ops.py` (16: ops envelope, cache replay as `cache_hits` in the v2 `llm_cache` table, fatal abort without commit + resume, `--max-calls` abort + resume, fingerprint mismatch refusal, strict decider validation with did-you-mean, credential presence, per-HTTP-retry call counting, shared budget). Adapted branch tests (0.2 behaviour): 2 matcher + 2 screen tests (fatal -> result not raise; failed request counted as unknown-usage call), bm25 pinned default fingerprint (0.2 components), fit note now a stderr warning, a queries test moved to a decider job, the missing-`llm` message. `python -m pytest -q`: 1443 passed, 28 skipped (rdflib, sqlalchemy, sentence-transformers not installed; live Jev/OpenAI tests skipped without `XWALK_TEST_API_KEY`; Ref_zivila data absent); ruff check/format clean (src tests benchmarks examples scripts); mypy clean (101 files); README quickstart check OK against a wheel built from the merge. FakeDecider/MockTransport only, no paid calls |
| 10 Browser UI (`xwalk ui`) | Done (awaiting review; not tagged or published) | unreported | unreported | `claude/eloquent-goldberg-sotveq` | Requested by Jan outside the planned task list. New `src/xwalk/ui/` (stdlib `http.server`, JSON API over `xwalk.ops`, static vanilla-JS page shipped as package data; no new dependency), `xwalk ui`, `docs/guide/ui.md`. New `tests/test_ui.py` (43): API, offline stand-ins, HTTP token/Host/path confinement, CLI; mutation probes (path confinement, Host check, token check each disabled in turn) made 6, 1 and 2 tests fail. `test_packaging` asserts the static files are in the wheel. Python 3.11 `.venv`: `python -m pytest -q -m "not integration"`: 1482 passed, 31 skipped, 4 deselected; ruff check/format clean (184 files); mypy clean (105 files). Browser walkthrough with Playwright/Chromium against a scratch workspace (quickstart, cluster_tiny fixture, ncbi_disease sample): create demo, validate, data preview, search, offline match, results, explain, evaluate, export download, offline clustering, review apply; no console errors. Offline stand-ins only, no paid calls; the endpoint route was exercised only up to its refusals |
| 11 UI: bring your own data | Done (awaiting review; not tagged or published) | unreported | unreported | `claude/eloquent-goldberg-sotveq` (restarted from `main` after PR #1 merged) | Requested by Jan. New `src/xwalk/ui/files.py` (uploads, inspection, normalization), `library.py` (built-in parsed samples, imports, OBO Foundry catalog, downloads), `projects.py` (job builder for map/cluster, retrieval-only lookup), `workbench.py` (endpoints), `static/workbench.js`; parsed samples in `src/xwalk/ui/ontologies/` built by `scripts/build_ui_ontologies.py` (ran with rdflib in the dev venv only). New `tests/test_ui_workbench.py` (33); mutation probes on filename sanitizing, upload-only deletion and the credential-name check each made 1 test fail. Python 3.11 `.venv`: `python -m pytest -q -m "not integration"`: 1515 passed, 31 skipped, 4 deselected (base install; 1533 passed / 12 skipped with rdflib); ruff check/format clean (190 files); mypy clean (110 files). Browser walkthrough (Playwright/Chromium): start page, quick map (candidates and offline), library import and browse, map a file onto ontologies, file onto file, clustering pasted text, files, settings; no console errors. Real-data check: the full HPO downloaded from its GitHub release (OBO PURLs answer 403 in this sandbox) parsed to 19,894 terms; quick map found Myocardial infarction for "heart attack" and Hypertension for "high blood pressure" (retrieval only). Offline stand-ins only, no paid calls |
| 12 UI: public hosting | Done (awaiting review; not deployed) | unreported | unreported | `claude/eloquent-goldberg-sotveq` | Requested by Jan. `xwalk ui --public` (`src/xwalk/ui/hosting.py`: per-visitor cookie sessions with idle expiry, shared read-only `--library`, shared `--lookup-cache`, per-session model keys never in os.environ, https-and-public-address check on model endpoints, no URL downloads, run limits, hidden paths, `/healthz`), `scripts/build_ontology_library.py`, root `Dockerfile` + `.dockerignore`, `docs/guide/hosting.md`. New `tests/test_ui_hosting.py` (20); mutation probes (address check off: 6 failed; session lookup falling back to another visitor: 1 failed). `python -m pytest -q -m "not integration"`: 1538 passed, 31 skipped, 4 deselected; ruff check/format clean (193 files); mypy clean (112 files). Docker daemon unavailable here, so the image was **not built**; its steps were replayed in a fresh Python 3.12 venv with a non-editable install (build script on HPO from GitHub: 19,894 terms, index built; the container command served `/healthz` and the page). Two-visitor browser test of public mode: isolation held, HPO quick map used the prebuilt index. Hosting facts (Cloud Run free tier, HF Docker Spaces now paid, Render 512 MB, Streamlit limits, Oracle Always Free) from web search on 7 Oct 2026, cited in the guide. No paid calls |
| 13 UI: workbench at `/app` | Done (awaiting review; not tagged or published) | unreported | unreported | `feat/react-workbench` (`908848e`) | Requested via a Claude Design handoff (xwalk Design System). React 19 + Vite + TypeScript in `web/` (tokens, 30 components, Runs / Results + inspector / Review queue / Export), built into `src/xwalk/ui/static/app` and committed; `xwalk ui` serves it at `/app` with the page's token and CSP; classic page untouched. Backend additions: `GET /api/reviews` (`review.review_history` + `review.adjudicated`), `usage`/`run_fingerprint`/`answered_by` on workspace runs, `source_context` on mapping rows, `duplicate_targets` on `/api/run`. New `tests/test_ui_app.py` (21; 9 failed before the change for the stated reasons), wheel assertion in `test_packaging`; mutation probe: removing the asset-name check made 2 traversal tests fail. `web/`: vitest 16 passed, `tsc` clean, clean `npm ci && npm run build` byte-identical to the committed build (CI job `web` checks this). Python 3.13 system interpreter (not the 3.11 `.venv`): `python -m pytest -q -m "not integration"`: 1602 passed, 24 skipped, 4 deselected; ruff check/format clean; mypy clean (115 files). Browser walkthrough (Playwright/Chromium, 1440/1024/760 px) on an offline quickstart and a review-demo run: runs, results, reviewed view, inspector, export dialog, a real review save, an intercepted failed save and a SnapshotMismatch conflict (queue did not advance); no console or CSP errors. Offline stand-in only, no paid calls |
| 14 UI: workbench redesign | Done (awaiting review; not tagged or published) | unreported | unreported | `feat/react-workbench` (`908848e`) | Requested by Jan with a reference mock-up (redesign.html, three "tiers"); the balanced tier was used as inspiration, its invented figures were not. `web/` only, no backend change: new palette and effects tokens, all components restyled (squares for model status, diamonds for human decisions, mono capitals, hairline grids), new `CommandLine` and `Drawer`, `DataTable` gains `onActivate` (click/Enter opens, j/k move) and `flush` columns and keeps the selection in view; Shell is a top bar plus status line; Results is a lineage (source ⟶ strand ⟶ target) with metric cells as filters and a bottom inspector drawer; Review draws the source ⟶ proposal strand and redraws it per decision, confirms inline, shakes on a refused save and strobes on a conflict. vitest 22 passed (6 new: CLI quoting, CommandLine copy, Enter/click activation, strand caps); `tsc` clean. Python 3.13 system interpreter: `python -m pytest -q -m "not integration"`: 1602 passed, 24 skipped, 4 deselected; ruff check/format clean; mypy clean (115 files). Browser check (Playwright/Chromium, 1440/1024/760 px) on the offline quickstart, review-demo and a partial run: runs, results (model output, reviewed, hover dimming), inspector drawer, review with a replace decision, export dialog; no console or CSP errors. No paid calls |
| 15 Branch consolidation | Done (not tagged or published) | unreported | unreported | `main` (fast-forward to `ac32d68`, then merge of `origin/jev-recall-cost` `c940c77`) | Requested by Jan on 19 Oct 2026. `feat/react-workbench` fast-forwarded into `main`; the Ref_zivila WIP commit merged **without** its top-level `scratch/` probes (they stay on `jev-recall-cost`); ruff fixes in `examples/ref_zivila/` (unused import, a long condition, formatting). A stale git-ignored `tests/fixtures/cluster_tiny/runs/ui-1` from a 7 Oct UI session made `test_ui.py::test_workspace_lists_jobs_runs_and_gold_files` fail locally; removed. The Ref_zivila jobs validate only with the private `data/ref_zivila/` inputs, which are not in the repository. Python 3.12 `.venv`: `python -m pytest -q -m "not integration"`: 1603 passed, 31 skipped, 4 deselected; ruff check/format clean (205 files); mypy clean (120 files). No paid calls |
| Engine revision B1: one exact normaliser | Done (awaiting review; not pushed) | unreported | unreported | `engine/b1-exact-normaliser` (off `feat/engine-revision`) | Per the advisors' DECISIONS.md ruling. `xwalk.retrieval.bm25.exact_form` (NFC, casefold, strip, collapse whitespace; punctuation and accents kept) on indexed exact forms and on the original query text; sanitising only feeds the text query, and an emptied text query leaves the exact lookup running alone. `NORMALIZATION_VERSION` 1 -> 2; `BM25Retriever.open` also refuses another normalisation version without `expected`. New tests: 24 in `test_bm25.py` (16 failed before the fix, incl. `IL-2` behind 30 `IL-2 receptor` distractors, `IN`, NFC/NFD, `IL-2` vs `IL 2`; case/accent guard passed before too) and 1 in `test_ops.py` (exit 3 naming `normalization_version`, then `--rebuild-index`). Python 3.12 `.venv` with `PYTHONPATH=src`: `python -m pytest -q -m "not integration"`: 1623 passed, 31 skipped, 4 deselected; with rdflib on `PYTHONPATH` (scratch target dir, venv untouched) `tests/test_retrieval_recall.py`: 12 passed, 1 xfailed (pre-existing strict xfail), shipped BM25 baselines reproduced unchanged; ruff check/format clean; mypy clean. No paid calls |
| 16 Engine revision, batch 1 (boundaries) | Done (awaiting review; not pushed) | unreported | unreported | `engine/batch1-boundaries` (off `feat/engine-revision` = `a757eda`) | Advisor findings 1-3, 5 and 11 (rulings R1, R2, R3, R6, R7 in the scratchpad `advisors/DECISIONS.md`). One commit per item, each test-first (every new regression test was run and seen failing for its stated reason before the fix). R1: malformed 200 bodies in `OpenAICompatClient` and `LiteLLMClient` raise only `LLMError` (empty choices retryable, not fatal; 200 error bodies by code; invalid token counts -> unknown usage, answer kept); `_call_stage` unchanged. R2: Jev rubric score in `[0, levels-1]`, policy re-check; missing `chosen_key` unresolved (selector and clusterer); rewrite `queries` list[str]. R3: null fields omitted before rendering, `TEMPLATE_RENDER_VERSION` 2 in `TemplateSet.fingerprint` (every run fingerprint changes) and in index identity only when a target has a null field. R6: screen collects every failed chunk, fatal wins, cancelled-after-dispatch billed; Jev usage missing/invalid -> unknown. R7: LiteLLM `api_key` passed. After review (`advisors/review_batch1.md`, changes required), four follow-up commits: null fields also dropped from the base prompts' source block and the Jev source state; LiteLLM `base_url` sent as `api_base` and hashed through `redact_url`; LiteLLM public error names classified first (`APIResponseValidationError`, `APIError` retryable; context-window, content-policy and similar fatal); half-reported usage unknown; `choices[0].error` retryable with its message; failed-POST-then-malformed-200 accounting pinned. Existing tests changed on purpose: two "no choices is fatal" tests, the decide-fit fixture rubric (2.5 on a 3-level scale), the cluster resume history check (prefix -> subset, ids are fingerprint hashes). Python 3.12 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider` (branch rebased onto the B1 BM25 commits): 1766 passed, 31 skipped, 4 deselected; ruff check/format clean (205 files); mypy clean (120 files). FakeLLM/FakeDecider/MockTransport only, no paid calls |
| 16 UI hosting security (F2, F3, F5, F6) | Done (awaiting review; not pushed) | unreported | unreported | `sec/ui-hosting` (off `feat/engine-revision` `a757eda`: `718fbf0`, `1ff4366`, `71cf411`, `877c5af`) | Requested by Jan from the security review (`reviews/security.md`). F5: `Secure` cookie from `X-Forwarded-Proto` only with new `--trust-proxy` (public only; set in the Dockerfile). F2: page views set a signed cookie only, the first API request opens the session; `--sessions-per-ip` (20 per 10 min, `429 too_many_sessions`); unused sessions dropped after 10 min idle or when the pool is full. F3: new `xwalk.ui.egress` pins public-mode model requests to the checked addresses (SNI/Host unchanged, env proxies ignored). F6: local downloads https-only to public hosts, pinned, at most 5 redirects each re-checked; `--allow-local-downloads` / build-script `--allow-local-urls`; catalog PURLs now https. New tests in `test_ui_hosting.py`, `test_ui_workbench.py`, new `test_ui_egress.py` (local TLS server, fake resolver); each new behavioural test was run against the old code first and failed for the stated reason. Python 3.12 `.venv`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 1642 passed, 31 skipped, 4 deselected; ruff check/format clean (207 files); mypy clean (121 files). Review follow-up (`02f9ce4`, `ddb3116`, and the `cryptography` dev-extra commit): sessions leased for the length of each request (no eviction/sweep mid-request; synchronised regression), ChEBI from `https://ftp.ebi.ac.uk/...` because its PURL redirects to http (other nine PURL targets https per the PURL service config, not fetched), `cryptography` in the dev extra so the TLS tests always run. After: 1645 passed, 31 skipped, 4 deselected; ruff/format/mypy clean (121 files). Real platforms' `X-Forwarded-For` formats (Cloud Run, Spaces) not checked; https PURL reachability not checked from here. No paid calls |
| Cluster C1: bugs and evaluation | Done (awaiting review; not pushed) | unreported | unreported | `cluster/bugs-and-eval` (off `feat/engine-revision` `7032c9a`) | From the clustering review (F6, F7, F8, proposal 1) and the second advisor's review. Fixed: exports after a seed leaves its cluster (lone member `singleton`, reason `sole_remaining_member`; `seed_id` always a member; both files decide by size); `retrieval_exhausted` compared hits with the next page's depth; the pool's retrieval heads were overwritten in the run fingerprint by the pool settings (lexical and dense runs shared a fingerprint; now a `retrieval` component, cluster format 2, format-1 directories refused with `run_format_outdated`). Added: `xwalk cluster eval` / `ops.cluster_eval` (pairwise and B-cubed under both conventions, outcome counts, usage, `gold_pairs_co_exposed`, documented as a diagnostic and **not** a recall bound; read-only store, no calls, no writes), metric code moved to `xwalk.evaluate.clustering` (benchmarks re-export it), matching-run commands refuse a cluster run with `cluster_run`, the benchmark's clustering track runs `xwalk cluster` (smoke: offline lexical judge, SYNTHETIC). New tests: `test_cluster_identity.py` (9), `test_cluster_eval.py` (14), 3 in `test_cluster_engine.py` (review probes edge A and B plus the seed-switch variant), 1 in `test_benchmarks.py` plus smoke-run assertions; each new test was run and seen failing before its change, except the metric re-export test, the pending-WAL read test (written after its fix) and the `cluster` usage test (already passing). Clustering decisions, ordering and retrieval defaults unchanged. Python 3.12.9 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 1651 passed, 31 skipped, 4 deselected; ruff check/format clean (209 files); mypy clean (122 files). Scripted judges only, no paid calls; nothing run against a real model, so the labelled-sample gate stays open. Review follow-ups (approved with fixes): CONTRACTS 10.4 amended for `sole_remaining_member`; run flags before `cluster eval` refused; live-WAL stores snapshotted with `backup()`; corrupt store -> `store_unreadable`; unfinished invocations -> `usage_incomplete`; blank `cluster_id` on accepted rows refused. **Still open** from the advisors' "Now" list: cluster `inspect`/`explain` (read-only, merge lineage, selected-vs-last revision), factual run warnings (bounded retrieval and evidence, deterministic-response assumption), and order sensitivity for `xwalk_clustering` in the benchmark (one run, one order; a second `order: input` run would double its paid calls) |
| Security F1: sandboxed Jinja | Done (awaiting review; not pushed) | unreported | unreported | `sec/jinja-sandbox` (off `feat/engine-revision` = `118c138`) | Security review finding F1 (scratchpad `reviews/security.md`). `TemplateSet` and the prompt skeletons compile in `ImmutableSandboxedEnvironment`; `checked_parse` refuses a constant dunder attribute/subscript at build time (`UnsafeTemplateError`, a `TemplateError`; `validate` -> `template_invalid` for match, decider and cluster jobs); computed names are refused by the sandbox at render (per-record `TemplateError`; `_EmptyUndefined` raises for a sandbox refusal instead of rendering ""); a hostile skeleton is a `ContractError`. Compiled record templates cached per process (`lru_cache`, 256). New `tests/test_template_sandbox.py` (66; 37 failed before the fix, the rest are the equivalence and slot-as-data guards): every bundled job (examples, quickstart, fixtures, cluster), the workbench `_TEMPLATES`, chemistry templates and every slots file render identically to an unsandboxed environment. Scratch before/after capture over the same set (3,565 renderings incl. template and prompt fingerprints, examples' sample records where loadable): identical, so run identity is unchanged and `TEMPLATE_RENDER_VERSION` stays 2. Micro-benchmark: render 5.6 us sandboxed vs 5.5 us plain; `TemplateSet` rebuild 0.8 us cached vs ~670 us uncompiled. Python 3.12 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 1833 passed, 31 skipped, 4 deselected; ruff check/format clean (206 files); mypy clean (120 files). MCP path sandbox (F4) still open. No paid calls |
| 18 Engine revision, batch 2 (identity, I1) | Done (awaiting review; not pushed) | unreported | unreported | `engine/batch2-identity` (off `feat/engine-revision` = `87c8ede`): `8ae3736` | Advisor ruling I1 / astra_review proposal 11. Every stage exposes `identity` (client fingerprint, prompts, templates, effective `temperature`/`max_tokens` via the client's optional `generation_defaults`, behavioural settings); `build_run_fingerprint`/`build_decision_run_fingerprint` take the stages and add a `stages` component only for settings that differ from the job-built defaults; `OpenAICompatClient`/`LiteLLMClient`/`FakeLLM` hash request-affecting capabilities (`json_schema`, `strict_schema` with a schema, `seed`) only when they differ from the profile/default; wrappers delegate; a decider job's `rewrite` adds the deviations of the client `ops.run` hands it (`rewrite_llm=`). No identity version bump for I1 itself (batch 3 changes every fingerprint anyway): job-built fingerprints unchanged by I1, checked on `job_tiny`, `job_tiny_jev` (with and without `decider.rewrite`) and a litellm variant, before (`118c138`) and after. Python-API runs with stage or capability overrides get new fingerprints (CHANGELOG says resumes are refused). New `tests/test_effective_identity.py` (33): captured MockTransport request bodies vs fingerprints on both paths, credential rotation, concurrency, budget/cache wrappers, refusal before any call via `ops.run`, `run_batch` non-reuse. Seen failing first (TypeError on the new API); mutation probes: dropping the `stages` component failed 11, dropping capability hashing 5, dropping `generation_defaults` 2, not passing `rewrite_llm` in ops 1. Python 3.12 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 1946 passed, 31 skipped, 4 deselected; ruff check/format clean (213 files); mypy clean (123 files). Review (`advisors/review_batch2.md`, changes required) follow-up after merging `feat/engine-revision` (`b2c5f34`, batch 3; both `status_semantics` entries kept): `eea9847` makes every stage argument of the four fingerprint builders required (`rewriter=None` explicit on the decider path), redacts the decider rewrite block's `base_url` before hashing, computes the job-side comparison from pure `openai_compat_fingerprint`/`litellm_fingerprint` (no client built), adds LiteLLM request-capture tests. 11 new tests; the redaction, required-stage and no-client tests were seen failing first for their stated reasons; the LiteLLM tests passed on the fixed code and 3 fail when its capability hashing is removed. Job fixtures: fingerprints identical with and without this branch on top of `b2c5f34` (they all differ from pre-batch-3 values because of `STATUS_SEMANTICS_VERSION`). After: 2012 passed, 31 skipped, 4 deselected; ruff check/format clean (214 files); mypy clean (123 files). MockTransport/FakeLLM/FakeDecider only, no paid calls |
| 17 Engine revision, batch 3 (reduction and retrieval health) | Done (awaiting review; not pushed) | unreported | unreported | `engine/batch3-reduction` (off `feat/engine-revision` = `118c138`), `d6d3dfb` | Rulings R4 and R5 (scratchpad `advisors/DECISIONS.md`, `astra_vote.md` precedence table). Test-first: new `tests/test_reduction.py` (56; 46 failed before the change for their stated reasons, incl. P1/P1b/P2 giving `needs_review` with the rejected or unverified candidate). R4: `derive_status` reports an acceptable attempt (`is_acceptable`, which now also requires a chosen target) as `matched`; with none, the old ranking is unchanged (property test over 20,000 random attempt lists against the pre-batch reducer kept verbatim as an oracle; it reaches every branch). R5: precedence table on both matchers; no hits with any retriever down is `retriever_failure`; a negative result with any provider failure or degraded search is `failed` with the latest blocking reason; `Attempt.retrieval_failures` (additive serde, old blobs read `()`); decider `Signals.retrieval_degraded` and rewrite-search failures kept; `STATUS_SEMANTICS_VERSION` 2 in both run fingerprints (one line each in `batch.py`). No existing test changed. The review probes (`fable-scratch/probe_policy.py`, `astra-scratch/probe_matcher_advisor.py`) now give `matched` for P1/P1b/P2 and `failed` for the outage cases. Python 3.12 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 1823 passed, 31 skipped, 4 deselected; ruff check/format clean (206 files); mypy clean (120 files). FakeLLM/FakeDecider only, no paid calls |
| DX fixes (review dx P1, P3, P4, P6, P7 part) | Done (awaiting review; not pushed) | unreported | unreported | `dx/fixes` (off `feat/engine-revision` `7032c9a`) | From the developer-experience review (`dx.md`). Provider errors: a call limit spent on retries names the last provider error instead of advising a higher `--max-calls` (`CallBudget.acquire` returns a dispatch index and `BudgetedLLM`, `BudgetedDecider` and the OpenAI-compatible and Jev clients report each request's outcome with `settle`; the refusal names the error of the latest-dispatched settled request, so overtaken failures are not blamed; a `run_batch` hook records the one run-level message, so match, inspect and manifest agree), failed rows summarised by distinct error with counts in run/inspect envelopes, stderr and summary, `uncapped_failures` warning plus one stderr warning after three failed records in a row. Ledger: `Ledger.open(read_only=True)` (symlinks resolved; `mode=ro` with a pending WAL; a `query_only` connection when writable, which removes the sidecars it created; WAL-aware `mode=ro` whenever a -wal/-shm exists and the log is pending or the directory unwritable, with a temporary-copy fallback; never `immutable=1`; an unwritable ledger without sidecars is copied; 0.1.1 ledgers through an upgraded temporary file; nothing copied into memory) used by inspect, explain, results, export, review export, eval, fit, compare and the UI's reads; new `xwalk upgrade --run`. sdist excludes process notes, `web/`, benchmark results and the Ref_zivila research scripts (CONTRACTS/REVIEW/CLUSTERING_DECISIONS kept); `run_pipeline.sh` takes its checkout as an argument; the shipped pages' links are validated inside the extracted sdist. Help text on every flag; `validate` calls an environment-only problem 'valid; the environment is not ready'; `utf-8-sig` for CSV/TSV/JSONL/OBO sources and gold; `template_unknown_field` warning; UI test servers poll every 10 ms. After the advisors' review (changes-required, 6 items) follow-up commits `fb27d7b`, `b6aa9b8`, `75fab7b`; after the second review (3 items) `af64021`, `639def1`; after the third (3 items, following the fast-forward to `feat/engine-revision`) `f59f20b` (a 200 settles after its body is read) and `d973dc0`. New tests: `test_run_errors.py` (11), `test_ledger_compat.py` +9, `test_ops.py` +6, `test_cli.py` +3, `test_sources.py` +4, `test_packaging.py` +3; every behaviour test failed before its fix, and four guards (healthy call-limit advice, a really invalid job, shipped templates, the existing compat cases) passed before and after (the sdist test was run with hatchling on `PYTHONPATH` from a scratch target dir; the `.venv` has none, so it skips there like the wheel test). Python 3.12 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 2039 passed, 40 skipped, 4 deselected (after the merge) (UI teardown fix: about 65 s down to 44-62 s depending on load); ruff check/format clean; mypy clean (120 files). Not done here: the ops.py split (P5) and the release re-cut (P2). No paid calls |
| MCP path sandbox (F4) | Done (awaiting review; not pushed) | unreported | unreported | `sec/mcp-paths` (off `feat/engine-revision` `3f80253`) | From the security review (`reviews/security.md` F4) and the REVIEW.md open item. New `xwalk.confine` (`Roots.resolve`: `Path.resolve()` + `is_relative_to`; `Roots.check_job`: `source/target.path`, `prompts.slots`, dense `model` local paths, sqlite DB; network SQL URLs refused). Every MCP tool path (`job`, `out`, `index_dir`, `run`) and the job's referenced files must resolve inside the roots before anything is compiled, read or written; refusal is a tool error `path_outside_roots` (exit 2). `xwalk mcp --root DIR` (repeatable, replaces the cwd default); `build_server`/`serve` take `roots=`. Test-first: new `tests/test_mcp_roots.py` (SDK-free, 23) and 6 `mcp`-marked tests in `test_mcp.py` (traversal, symlink escape, absolute outside, writes outside, job referring outside, `--root` over stdio, bad `--root`); with containment disabled 18 of them fail. Python 3.12 venv, `PYTHONPATH=src`: `pytest -q -m "not integration"` 1870 passed, 37 skipped, 4 deselected (no SDK); ruff check/format clean (213 files); mypy clean (124 files) with and without the SDK. With `mcp` 2.3.0 in a scratch dir: `-m mcp` 13 passed (1 skip is the sql module, no sqlalchemy); `test_mcp.py` + `test_mcp_roots.py` 38 passed. No paid calls. Review follow-up (second commit): a symlink at any depth inside an `out`/`index_dir`/`run` directory that leads outside is refused (`Roots.resolve(tree=True)`, `os.walk` without following links); differently cased spellings of a root are accepted via `samefile` on existing ancestors; 4 new tests, all seen failing first (the MCP one had built an index through `index/bm25 -> outside`). Not addressed: check-then-open race (documented); the UI's `Workspace` does not check job-referenced paths either |
| 16 UI: workbench fixes after the UI/UX review | Done (awaiting review; not pushed) | unreported | unreported | `ui/workbench-fixes` (off `feat/engine-revision`; `063c558`, `f3db7fa`, `27fea7f`, `1e06a21`, `a5facdf`, then this STATUS commit after `2613f46`) | Requested by Jan from the ui-ux review (F1-F7, F9). F1: `.gitignore` `data/` -> `/data/`; the six `web/src/components/data/` components rewritten from the committed bundle (rebuilt JS and CSS identical to the old build once CSS-module hashes are normalised; Playwright screenshots of Runs/Results/Review differ only in the server port); a clean export of the branch passes `npm ci`, typecheck, tests and a build byte-identical to the committed one; `web/tsconfig.tsbuildinfo` untracked. F2: clustering-run URLs show a notice instead of a white page; `RunView` is a union; error boundaries inside and around the shell. F3: per-source indexed ledger reads (`current_entry`, `iter_source_history`, `iter_reviews_for`, read-only, beside the existing scans), `ops.review_rows`, `review.adjudicated_reviewed`, next-record prefetch; `tests/test_review_scale.py` (15; the previous explain kept as an oracle; the decode-count tests failed before the change); `scripts/bench_review_step.py` (FakeLLM, 2,000 rows, in-process): explain 749 -> 7 ms, decision 382 -> 28 ms, overlay 379 -> 25 ms; over HTTP on the review's 2,189-row workspace 798 -> 4, 379 -> 12, 373 -> 24 ms. F4: loading placeholder in the proposal. F5/F6/F9: the queue opens on the first undecided record and scrolls with j/k; Escape leaves fields; j/k/Enter from a fresh page; `GET /api/target` and 422 `unknown_target` on save (`review.apply_review` and the CLI unchanged). F7: `--text-faint` #66645E, `--accent-ink`, ink on orange; axe-core color-contrast 272 -> 0 violations over 7 views. Gates (Python 3.12 `.venv`, `PYTHONPATH=src`): pytest 1641 passed, 31 skipped, 4 deselected; ruff check/format clean; mypy clean (121 files); web: typecheck clean, vitest 41 passed, build matches the commit. Not done (by request): classic/app convergence, dark theme. No paid calls |
| Cluster C2: inspect, explain, warnings, docs | Done (awaiting review; not pushed) | unreported | unreported | `cluster/inspect-explain` (off `feat/engine-revision` `3f80253`: `adf445d`, `f2be6e9`, `30666f3`) | Advisors' clustering "Now" items 3 and 4 and review finding 7. Added: `xwalk cluster inspect` / `ops.cluster_inspect` (exported outcome counts, clusters by size, merges, decisions by kind, usage over every invocation with `invocations_without_usage`, selected vs last revision and `changed_after_export`; exits with the run's own code) and `xwalk cluster explain` / `ops.cluster_explain` (exported membership, the decisions of the admitting step with the members each prompt listed, merge ancestry, later adverse changes set aside by revision selection, assignment history, pending/failed notes); both read-only (`open_readonly`, no engine, pool or client) with `manifest_unreadable` / `store_unreadable` / `store_mismatch` exit 3. `exports.exported_state` now feeds both `write_exports` and the views. The `cluster_run` refusal names the new commands. Factual warnings, once per code, on `xwalk cluster` and `cluster inspect`: `retrieval_bounded`, `evidence_bounded`, `resume_determinism`, `selected_not_last` (thresholds in the guide and `cluster/diagnostics.py`). Regressions from the advisors' `evidence_probe.py` (11-member cluster with 15 pairwise-rejected later members; exported membership a later reverification rejected). Docs corrected per astra_cluster.md (evidence bound, resume determinism, order effects, suffix vs hyphen, co-exposure, SYNTHETIC labels); the order figure 29/300 was re-run on this branch with the review's probe, the 72/72 and 0/15 suffix counts are cited from the advisor, not re-run. New tests: `test_cluster_inspect.py` (24), `test_cluster_warnings.py` (9), the redirect test extended; all seen failing before their change except the six run-flag-refusal cases (already refused by argparse as unknown subcommands). Clustering decisions, ordering and retrieval defaults unchanged. Python 3.12.9 `.venv`, `PYTHONPATH=src`: `python -m pytest -q -m "not integration" -p no:cacheprovider`: 1880 passed, 31 skipped, 4 deselected; ruff check/format clean (215 files); mypy clean (125 files). Scripted judges only, no paid calls. Still open: order sensitivity for `xwalk_clustering` in the benchmark, the labelled real-sample gate |
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
- Per-stage Python overrides (`Scorer(max_tokens=...)`) were not part of
  `build_run_fingerprint`. Closed by engine revision batch 2 (I1): the fingerprint
  takes the stages and records the settings that differ from the job-built defaults
  under `stages`; job-built matchers add nothing, so their fingerprints are unchanged.
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
  benchmarks and should add a real labelled sample when an endpoint and budget exist.
  (B-cubed/pairwise metrics exist since task 05 and are in the package since C1:
  `xwalk cluster eval`.)
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

## Task 08a notes (review follow-up)

- Cluster resume of a finished run: steps `resume/<r>/restore`, `resume/<r>/<source>`,
  `resume/<r>/end` with `r` = last revision + 1; the round starts from the exported
  (selected) state and its end revision becomes the selected one. `run_state` and
  `source_failed` errors now follow the exported state, not the in-memory last state.
- Still open from REVIEW.md: MCP path sandbox (`--root`; since closed on `sec/mcp-paths`), pydantic lax coercion, and the
  task 07 carry-overs (README install line, PyPI relative links, `inspect` model name,
  CONTRIBUTING lint paths).

## Task 09 notes (decider-path merge)

- Merged `5e567d8` (the `jev-recall-cost` branch without its final WIP commit). Decider
  jobs run through `ops.run` (`decider=`/`decider_factory=` inject a client; the CLI keeps
  `_build_decider`/`_build_rewrite_llm` for tests), so `--json`, run states, exit codes,
  run-directory binding (decider fingerprint components in the manifest, secret-free) and
  index identity apply. `xwalk fit` returns an `OpResult`; `ablate`/`prompts` refuse a
  decider job with `wrong_job_path` (exit 2). MCP `--offline-model` uses a FakeDecider for
  decider jobs.
- Contract 3 on the decider path: `DecisionMatcher.match` no longer raises; a fatal decider
  or rewrite-LLM error is `fatal_provider_failure` (run aborts, record not committed). A
  malformed Jev response is the new `DecisionResponseError` (subclass of
  `DecisionFatalError`, so branch tests still match) and is a per-record
  `provider_failure`, like malformed LLM output.
- Contract 5: `JevClient.attach_call_budget` reserves per HTTP request; `BudgetedDecider`
  (`xwalk/decide/budget.py`) mirrors `BudgetedLLM`; `CallBudget.share()` lets the decider
  and the rewrite LLM draw on one `--max-calls` without double counting. Failed requests
  are `unknown_calls`; cache replays are `cache_hits`. `Usage.cost_usd` added (serde
  reads old blobs as 0.0); `mapping.csv` gains a trailing `cost_usd` column.
- Ledger: no schema change. Decisions are cached in the existing `llm_cache` table, so
  v0.1.1 and v2 ledgers open as before.
- Identity: `TemplateSet.fingerprint` includes `queries` only when set and
  `PromptSet.fingerprint` excludes slot `properties`, so 0.2.0rc1 LLM runs keep their
  fingerprints. BM25 `analyzer`/`fuzzy_distance` enter the index components only when not
  the defaults. The benchmark-pinned `slots.yaml` of cafeteria_fcd and ncbi_disease are
  unchanged; their decider `properties:` moved to `slots_jev.yaml`.
- Strictness added: decider blocks forbid unknown keys and range-check thresholds at load
  (`rubric_floor` is a level, so only `>= 0`); `selector:` refused on a decider job;
  `templates.queries` with more than one entry refused on an `llm:` job.
- Open: the accuracy/cost numbers in `docs/reference/decide.md` were measured on the branch
  before the merge and not re-run; the live Jev tests were skipped (no key). The
  job-file reference's stale "Gotchas" about unvalidated fields were corrected in passing.

## Engine batch 4 review follow-up (19 October 2026)

- Fixed all three findings in `review_b4.md`, each with a regression observed failing
  before its fix: preserve an absent chooser's resolution in ledger replay; require
  an explicit LLM path before treating a clipped slate as complete; disclose recorded
  and current reducer versions and historical/unknown warnings in ordinary evaluation,
  including `ops.evaluate_run` and saved JSON/text reports.
- Focused evaluation tests: 87 passed. Full non-integration gate with `PYTHONPATH=src`,
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `-p no:cacheprovider`: 2,112 passed,
  40 skipped, 4 deselected; 8 failures and 40 setup errors all from sandbox socket-bind
  denials. Ruff check and format pass (222 files); mypy passes (126 source files).
- No paid calls or push. Balances unreported. This worktree refuses Git's index lock;
  separate commits and mail patches are prepared in `/private/tmp/xwalk-b4-review-commits`
  without changing this branch's ref. Source fixes remain in this worktree.

## Engine batch 5 review follow-up (19 October 2026)

- Finding 1: selector budgeting renders sequentially and stops on the first overflow,
  retaining the count-capped key width. Three regressions failed before the fix;
  selector, keying and matcher tests now pass (110 tests).
- First-fix gates: 2,174 passed, 40 skipped, 4 deselected; 8 failures and 40 setup
  errors are sandbox socket-bind denials. Ruff check/format and mypy pass.
- No paid calls. Balances unreported.
- Finding 2: replaced the same-implementation HTTP comparison with nine cases using
  exact request bodies captured from an archive of `587176e` (confident, verified and
  rewritten paths, each with three single-query configurations). Complete run and
  stage identities allow only the two documented budget-version additions. Capture
  provenance and reproduction commands accompany the fixtures.
- Tests failed before fixtures were supplied. An injected selector-prompt suffix made
  all nine oracle cases fail while the old comparison still passed; an unexpected
  fingerprint component also failed all nine. Mutations were process-local only.
- Final focused tests: 118 passed. Full non-integration gate: 2,182 passed, 40 skipped,
  4 deselected; all 8 failures and 40 setup errors are sandbox socket-bind denials.
  Ruff check/format pass (227 files); mypy passes (128 source files).
- Worktree Git index writes were denied. Two commits with matching source changes are
  prepared in `/private/tmp/xwalk-b5fix-commits` and exported as
  `/private/tmp/xwalk-b5fix.mbox`; the worktree branch ref is unchanged.

## Engine batch 6: bounded routing (19 October 2026)

| Item | State | Validation | Balances |
|---|---|---|---|
| Forced candidate retry | complete | P4 failed first on re-selection; 53 focused tests pass. Full offline non-integration gate: 2,194 passed, 40 skipped, 4 deselected; 8 failures and 40 setup errors are sandbox socket-bind denials. Ruff check/format pass (228 files); mypy passes (128 files). | unreported |

| Opt-in bounded routing | complete | 29 new tests failed before implementation; 87 routing/matcher tests pass. Full offline non-integration gate: 2,228 passed, 40 skipped, 4 deselected; 8 failures and 40 setup errors are sandbox socket-bind denials. Ruff check/format pass (228 files); mypy passes (128 files). Captured base HTTP bytes and identities match with flags off. | unreported |

No paid calls. An initial full-suite invocation was interrupted during an optional
Hugging Face lookup; the completed gate uses `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
Worktree Git index writes are denied; commits are assembled in
`/private/tmp/xwalk-b6-commits` and exported to `/private/tmp/xwalk-b6.mbox`.

# Task 00 baseline report

Date: 5 October 2026. Branch `claude/zealous-heisenberg-nri5m0`.

## Source revision

- Starting HEAD: `2dc58b1` (docs: add claude-upgrade handoff), package version 0.1.1.
- Audited commit: `f737099`. `git diff --stat f737099 HEAD` touches only
  `docs/claude-upgrade/`; `src/`, `tests/`, `examples/` and `scripts/` are identical to the
  audited snapshot. All audit findings therefore apply to unchanged code.
- No root `CLAUDE.md` existed before this task.

## Environment

Linux, Python 3.11.15, fresh venv, `pip install -e ".[dev]"` (pytest, pytest-asyncio,
ruff 0.16.10, mypy, types-PyYAML; tantivy 0.26.2). A second venv had
`pip install -e ".[dev,ontology,sql]"`. The `dense` extra (torch, sentence-transformers,
faiss) was not installed, to avoid large downloads. No install failed through the proxy.
Only Python 3.11 on Linux was exercised; CI's macOS/Windows/aarch64 and 3.10/3.12 cells
were not.

## Commands and outcomes (mirroring `.github/workflows/ci.yml`)

| Command | Outcome |
|---|---|
| `python -c "import tantivy"` | `tantivy v0.26.2, index_format v7` |
| `python -m pytest tests/test_bm25.py -q` | 20 passed |
| `python -m pytest -q -m "not integration and not dense and not ontology and not sql"` | **before fix: 1 failed**, 871 passed, 7 skipped, 8 deselected. After fix: 872 passed, 7 skipped, 8 deselected |
| `python -m pytest -q` (base env) | after fix: 872 passed, 15 skipped |
| `python -m pytest -q -m "not integration"` (CONTRIBUTING gate, base env, after docs added) | 878 passed, 14 skipped, 1 deselected |
| `python -m pytest -q -m ontology` (ontology+sql env) | 6 passed |
| `python -m pytest -q -m sql` (ontology+sql env) | 10 passed |
| `-m dense` | not run (extra not installed) |
| `-m integration` | not run (no endpoint; paid APIs out of scope) |
| `ruff check src tests examples scripts` | All checks passed |
| `ruff format --check src tests examples scripts` | 109 files already formatted |
| `mypy` (strict, config in pyproject) | Success: no issues found in 64 source files |
| base-install-light check (no torch/rdflib/... in `pip list`) | OK |
| `python -m build` + `twine check dist/*` | sdist and wheel built, both PASSED |
| wheel contents check (prompt skeletons, `py.typed`) | OK |
| fresh venv `pip install dist/*.whl`, `xwalk --version`, prompt-contract smoke | `0.1.1`, `fresh install OK` |

### Prerequisite fix (reported separately)

The handoff package itself broke the docs gate: `tests/test_docs.py::
test_the_docs_index_links_to_every_page` requires every `docs/**/*.md` to be linked from
`docs/README.md`, and `docs/claude-upgrade/` is not. Like `docs/superpowers/`, these are
internal planning documents, so the orphan check now also skips `claude-upgrade`
(one-line change in `tests/test_docs.py`). Link and code-fence checks still cover the
new pages. No library code was changed.

## Audit diagnostic

`python docs/claude-upgrade/audit/diagnose_audited_source.py .` ran without error. Its
JSON output is **identical** to `audit/observed.json` (compared with `==` after loading).

## Package-level reproduction

A scratch script (not committed) drove the installed package with the existing test
fixtures (`tests/test_matcher.py` helpers, `FakeLLM`, `ScriptedRetriever`, real
`run_batch`/`Ledger`). Results:

| Probe | Observed |
|---|---|
| Scorer returns `NaN` | `matched`, confidence 1.0 |
| Candidate proposal, trace on vs off | on: `matched T2`; off: `unmatched`, second attempt `no_candidates` |
| Verifier raises `LLMRetryableError` | escapes `Matcher.match` |
| Rewriter raises `LLMFatalError` | escapes `Matcher.match` |
| Two-attempt run with a rewrite | 5 provider calls, `usage.calls == 4` |
| Edit source s1, resume into same dir | `mapping.csv` has 2 rows for s1; `report.total == 2` |
| Misspelled job keys (`polcy`, `policy.acept_at`) | accepted silently |
| Job `temperature=0.7`, `max_tokens=2048` | request body sends `temperature 0.0`, `max_tokens 512` (scorer) |

## Finding status

| # | Finding | Status | Evidence | Proposed behavioral test (owner) |
|---|---|---|---|---|
| 1 | NaN confidence clamps to 1.0 and is accepted | Confirmed | diagnostic + package probe | Scorer/verifier replies with `NaN`, `Infinity`, `-Infinity`, `"0.9"`, `1.7`: never `matched`; non-finite → unresolved/needs_review (01) |
| 2 | Rewrite usage omitted from totals | Confirmed | diagnostic + probe (5 vs 4) | Scripted multi-attempt run: `result.usage.calls == len(fake.requests)` and equals sum of attempt usage (01) |
| 3 | Verifier retry exhaustion escapes | Confirmed | diagnostic + probe | In-band score + verifier `LLMRetryableError` → result returned, never `matched`, reason `provider_failure` (01) |
| 4 | Fatal rewriter error escapes | Confirmed | diagnostic + probe | Rewriter `LLMFatalError` → `failed`/`fatal_provider_failure`, no raise; `run_batch` aborts without committing it (01, 02) |
| 5 | Multiline chosen-candidate evidence leaks into "others" | Confirmed | diagnostic (source unchanged) | Chosen record with blank lines in a field: verifier/scorer prompt keeps the whole text in the chosen block (01) |
| 6 | Trace flag changes decisions | Confirmed | diagnostic + probe | Same script with `keep_candidates_in_trace` True/False: identical status, ids, requests (01) |
| 7 | Changed source duplicates current export | Confirmed | diagnostic + probe | Edit s1, resume: one current s1 row, old version in history; remove s2: absent from current (02) |
| 8 | Ledger written after close on concurrent failure | Confirmed | diagnostic; `run_batch` gathers without cancelling and closes in `finally` | One record raises while a sibling is mid-call: no write after close, sibling cancelled and awaited (02) |
| 9 | CLI exit ignores failed records | Confirmed by inspection | `_cmd_match` only checks `needs_review` | Scripted failed record → exit 3 and JSON `errors` (03) |
| 10 | Unknown job fields ignored | Confirmed | probe | Misspelled key and invalid threshold rejected with exit 2 (03) |
| 11 | Generation settings overridden by stage requests | Confirmed | probe | Capture outgoing body: job `temperature`/`max_tokens` sent (01) |
| 12 | Dense identity incomplete (prefixes, normalize, revision) | Confirmed by inspection | `DenseRetriever.build` fingerprint omits them; `open` checks only name/dimension | Change each setting → open refuses, naming the component (03) |
| 13 | Index rebuilt on every `match` | Confirmed by inspection | `_cmd_match` always calls `build_retrievers` → `.build` | Second `match` against compatible index: zero encoder calls (03) |
| 14 | Targets loaded twice, all sources loaded before `--limit` | Confirmed by inspection | `_cmd_match` | Count target/source reads (03, 05) |
| 15 | BM25 exact-boost normalization (`IL-2`) | Not reproduced (not probed) | — | Query `IL-2` vs indexed `IL2`/`IL-2` with Tantivy (05) |
| 16 | Dense memory copies, concurrent cache misses, batch barriers | Not reproduced (not probed) | barrier pattern visible in `run_batch` | Profiling per task 05; barrier removed by contract §4 (02) |
| 17 | v0.1.1 not on PyPI | Confirmed | see below | n/a (07, user) |

## PyPI and release workflow

- `pip index versions xwalk`: "No matching distribution found". `https://pypi.org/pypi/xwalk/json`
  and `/simple/xwalk/` return 404 (while `/pypi/requests/json` returns 200 through the same
  proxy). TestPyPI also 404. **xwalk is not currently published on PyPI**; the name appears
  unclaimed.
- `.github/workflows/release.yml`: tag `v*` or manual dispatch; build job builds, runs
  `twine check`, checks tag = packaged version, checks wheel contents and licence, tests
  the installed wheel in a clean venv and asserts no heavy deps; publish job uses the
  `pypi` environment and Trusted Publishing (`id-token: write`,
  `pypa/gh-action-pypi-publish@release/v1`). Likely cause of the recorded publish failure
  is a missing/mismatched Trusted Publisher (pending publisher) registration on PyPI for
  the `pypi` environment; this was not verifiable from here. Jan must register a pending
  publisher (owner `jan3657`, repo `xwalk`, workflow `release.yml`, environment `pypi`)
  before the next tag. The build job itself reproduces locally.

## Pilot real-data benchmark choice

- Primary (Jan's domain): `examples/cafeteria_fcd/sample` — 50 food mentions to a FoodOn
  subset (`targets.owl`), multi-ID gold. It has no no-match rows, so task 05 adds ~10
  no-match cases by removing those mentions' gold targets from the catalog (documented
  construction, near-neighbour distractors retained).
- Public pinned corpus: `examples/ncbi_disease/sample` — 50 disease mentions to CTD records.
- Both are small, frozen in the repo, and need no extras except `ontology` for OWL.
- Endpoint: none configured in this environment (no `XWALK_TEST_*` variables; paid APIs
  are out of scope). Real evaluation stays **pending** until Jan supplies an endpoint and
  a separate inference budget, or runs it on his cluster. Offline FakeLLM runs only test
  plumbing.

## Proposed extra features (ranked)

1. **Resume/compatibility explanation.** User: anyone whose rerun is refused or
   silently starts over. Problem: a changed hash gives no reason. Reuse: the dict already
   built in `build_run_fingerprint` and index meta; store components in the manifest and
   diff them. Check: change `accept_at` and the model; refusal names exactly those two.
   Cost: small, one helper plus tests. Fits task 03.
2. **Token-free threshold sweep.** User: someone choosing `accept_at`/`review_floor`.
   Problem: no way to see precision vs coverage vs review rate per threshold. Reuse:
   ledger results, `evaluate/metrics.py`, gold loader. Check: on a fixture run, the sweep
   at the configured thresholds reproduces `xwalk eval` numbers. Cost: small–medium.
   Fits task 05 if budget remains.
3. **Imported candidate pairs as a retriever.** User: large structured jobs blocked by
   Splink/dedupe. Problem: xwalk must do its own retrieval. Reuse: the `Retriever`
   protocol and fusion. Check: a pairs CSV drives candidates; run fingerprint includes the
   file digest. Cost: medium; defer past 0.2 unless a concrete user appears.

## What blocks task 01

Nothing. Task 01 can start from this commit following `CONTRACTS.md` §1, §3, §5, §6 and
the 01 row of §11. Expected compatibility changes it should report: new
`DecisionReason.FATAL_PROVIDER_FAILURE`, new `Usage` fields, optional
`LLMRequest.temperature/max_tokens`, and changed default request parameters.

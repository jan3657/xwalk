# Task 08: independent integrated review of 0.2.0rc1

Reviewer: a separate session that did not write the implementation. Date: 5 October 2026.
Evidence: commands run in this session. Probe scripts were kept in the session scratchpad,
outside the repository. Each finding below was reproduced against the checkout.

## Verdict

**The matching release is ready as 0.2.0rc1, with clustering marked experimental and MCP
optional.** No release-blocking defect was found. I fixed three small contract gaps (one
compatibility change). I found no hidden failed gate.

## Candidate and proposed release scope

- Reviewed candidate: `c97b748` (0.2.0rc1, branch `claude/zealous-heisenberg-nri5m0`).
- Proposed release commit: the review commit on top of it. It contains the fixes below,
  this report and the STATUS row. The version stays `0.2.0rc1` because nothing has been
  tagged or published.
- Scope: the matching SDK and CLI, with the lifecycle, operations, JSON and exit-code
  contracts (supported). `xwalk cluster` (experimental). `xwalk mcp` (optional extra,
  supported for local stdio use, no path sandbox). The benchmark harness lives in the
  repository only and is not in the wheel.

## Commands run and results

Unless stated otherwise, everything ran on Python 3.11 in the repo `.venv` (mcp 2.3.0
installed). Nothing made a paid or network LLM call.

| Check | Result |
|---|---|
| `python -m pytest -q` on `c97b748` | 1179 passed, 16 skipped (sqlalchemy, rdflib, sentence-transformers and the live-endpoint test are absent) |
| `ruff check` / `ruff format --check` / `mypy` (src tests examples scripts benchmarks) on `c97b748` | clean / 152 formatted / no issues in 80 files |
| `python -m build`, `twine check` | sdist and wheel PASSED. The wheel holds prompts, quickstart, `py.typed`, `ops.py` and `mcp_server.py`, and no tests or benchmarks |
| Fresh venv in the scratchpad, `pip install <wheel>` | `pip check` clean, `xwalk --version` 0.2.0rc1. Importing xwalk, ops and the CLI loads no torch, numpy, rdflib, sqlalchemy, faiss, litellm or mcp |
| `scripts/check_readme_quickstart.py --bin <fresh venv>` | OK (before and after the fixes) |
| README quickstart typed by hand in an empty directory | `init`, `validate --no-credentials`, `quickstart.py` (`complete {'matched': 3, 'unmatched': 1, 'total': 4} model calls: 7`), `inspect`, `explain s4`, `export --view reviewed`: all exit 0. A rerun made 0 calls |
| Python 3.12 venv with `.[dev,mcp,ontology,sql]` | 1201 passed, 2 skipped, 1 failed. The failure is `test_packaging` needing `pip` in a `uv venv`, which is environmental |
| Python 3.10 with **lowest** runtime deps (pydantic 2.0, jinja2 3.1.0, httpx 0.27.0, PyYAML 6.0, tantivy 0.22.0) | 1171 passed, 23 skipped |
| PyPI checks | `mcp` 2.3.0 is the latest on PyPI. `mcp.server.mcpserver.MCPServer` exists, so `mcp>=2.3,<3` matches the API used. The `xwalk` project returns 404 (not published) |
| Wheel with `[mcp]` extra in a fresh venv, then raw JSON-RPC over `xwalk mcp --offline-model` | see MCP below |
| After the fixes | pytest 1188 passed, 16 skipped (the 3 tests beyond +6 are doc-link checks parametrised over the new REVIEW.md). ruff, format and mypy clean. Rebuilt wheel `twine check` PASSED and the README check passed from a fresh venv |

### Adversarial probes (results)

- **Trace invariance and usage.** I ran 1,500 seeded random scenarios twice (6% bad JSON,
  NaN confidences, hallucinated keys and proposals, 1-4 attempts, audit 0/0.5/1), then
  1,500 more with fatal errors added. Trace on and off gave identical status, id,
  confidence, reason, attempts and outgoing requests. In every case
  `usage == sum(attempts) == len(requests)`, and nothing matched with a confidence
  outside [0, 1]. **0 violations.**
- **Confidence parsing.** NaN, ±Infinity, out-of-range values, strings and booleans
  (`true`) are all rejected (`parse_confidence`).
- **Lifecycle, through `ops.run`.**
  - Edited sources keep exactly one current row and keep their history.
  - A removed source drops out of the current view, stays in history and is counted in
    `removed_sources`.
  - `--limit 1` gives `partial` / exit 1 with the rest `pending`.
  - Rerunning an unchanged run makes 0 calls. `--no-resume` recomputes every source and
    keeps one row per source.
  - A review made on a source that was later edited is not applied to the reviewed
    export.
- **One failure while others are in flight** (concurrency 4, 25 records, a slow async
  client).
  - Fatal provider error: `aborted`, exit 3, 12 committed. 4 calls were counted as
    unknown usage (1 fatal call plus 3 cancelled in flight).
  - Programming error: `OpError run_exception`, exit 3.
  - Task cancellation: `interrupted`, `inspect` exits 130.
  - In all three, no ledger write or transaction happened after `close()`. Resume
    completed all 25 records.
- **0.1.1 ledger.** I regenerated the fixture from `f737099` with `make_fixture.py` and
  it matched the committed one row for row. HEAD opens it, writes
  `ledger.sqlite.v1-backup` (byte-identical to the original) and exports the current,
  reviewed and history views. The review survives.
- **Strict config.** Misspellings at the top level, in policy, llm, retriever and target
  are rejected with "did you mean". Out-of-range values, inverted `verify_band`, NaN,
  bad Jinja and fractional ints are rejected too. Duplicate keys were accepted (fixed,
  below).
- **Generation parameters.** Job `temperature 0.3, max_tokens 77, seed 5` reached every
  captured HTTP body (9 of 9).
- **Call budget.** 12 trials with concurrency 8, 30 sources, 20% HTTP 429/503 and
  rewrites, with caps from 5 to 82. HTTP requests equalled the cap every time, never
  more, and reported `calls` equalled the HTTP count.
- **Index identity.** Dense, with a counting toy encoder:
  - Build: 5 document encodings.
  - Reopen and match: 0 document encodings.
  - Changed `query_prefix`: refused, exit 3, naming `encoder.query_prefix`.
  - `rebuild=True`: only the dense index was rebuilt.
  - Unpinned model revisions are recorded as `"unknown"`, as the contract allows.
- **CLI JSON.** I ran 23 invocations across every command, including argparse errors.
  Each printed exactly one JSON object on stdout, with the envelope's `exit_code` equal
  to the process exit code. Output-dir collisions were refused with exit 3: a foreign
  directory (`out_not_a_run`), a file (`out_not_a_directory`) and another run
  (`run_fingerprint_mismatch`, with the changed component named).
- **Secrets.** A base_url carrying `user:pass@` and `?api_key=`, plus an API key in the
  environment: neither appeared in the run directory, the `--json` output or stderr.
  The llm fingerprint component is a hash of the redacted URL. All YAML loading uses
  `safe_load` (now a SafeLoader subclass). No `eval`, `exec`, pickle or `shell=True`.
- **MCP** (raw protocol, wheel install).
  - Every stdout line was JSON-RPC (26 of 26), initialize reported xwalk 0.2.0rc1, and
    all 6 tools declare an output schema.
  - Bounds are enforced: `limit` ≤ 100 for search and match, page ≤ 200, `max_calls`
    required and ≤ cap.
  - Paging walks all rows with `next_offset` reaching null.
  - `isError` is true for schema violations, a `max_calls` over the cap, `OpError`s and
    unknown tools. A performed operation, such as an invalid job or a run that hit its
    limit, is a normal result. That matches `docs/guide/mcp.md`.
- **Clustering**, scripted judges (27 labels in 10 concepts, including related-but-distinct
  pairs and two novel items).
  - Four input shuffles (`order: label`) gave identical partitions equal to gold, with
    no impure cluster.
  - `order: input` with a naive selector stayed pure.
  - Contradictory chain (A~B, B~C, C~D, D~E only): no cluster contained a non-equivalent
    pair, with or without the naive selector.
  - 25 noisy judges (random verdicts, bad JSON, NaN, 503s): every source appeared
    exactly once, and accepted clusters were exactly `assigned` plus `singleton`, with no
    overlap.
  - Budget-interrupted runs resumed to the uninterrupted partition (caps 3, 7 and 13).
  - 8 runs cancelled at random points (3-9 times each) all resumed to the uninterrupted
    partition.
- **Benchmarks.** I spot-checked `summary.md` against `results.json`
  (`ncbi_disease_nomatch` `xwalk_full`: 0.870 / F1 0.833 / 139 calls / 3 false accepts)
  and the perf table against `perf/*.json`. All values match. Every model-dependent
  number is labelled SYNTHETIC. README, CHANGELOG and docs make no quality, cost or
  speed claim.

### Checks that could not run

- A real-model run, a live provider and the real-provider README route: there is no
  endpoint or budget.
- The `dense` extra with `sentence-transformers`: not installed. Dense paths were tested
  with toy encoders.
- MCP inside Claude Code or Claude Desktop hosts: not exercised.
- The GitHub release workflow: no tag was pushed.
- macOS and Windows.
- LiteLLM internal retries: neither counted nor capped, as already documented.

## Confirmed fixes in this review (with regression tests)

1. **A key given twice in a job file loaded silently, with the last value winning.**
   `policy: {max_attempts: 3, max_attempts: 1}` validated as valid. This is the same
   class of silent override that `extra="forbid"` exists to prevent. Match and cluster
   job files now load with a SafeLoader subclass that refuses duplicate mapping keys and
   names the first line. `<<` merge overrides are still allowed. Tests:
   `test_job_validation.py` (3) and `test_cluster_ops.py` (1). **Compatibility:** such
   files no longer load (CHANGELOG, job-file reference).
2. **YAML syntax errors quoted the offending file line.** Because MCP `validate_job`
   accepts any path, an agent could read fragments of non-job files, for example a
   credentials `.ini`. The message now gives the problem and line/column only. Test:
   `test_a_yaml_error_does_not_echo_the_file_content`.
3. **`inspect` exited 1 for a run with `failed` rows and no recorded invocation** (an
   upgraded 0.1.1 run, whose `run_state` is "unknown"). CONTRACTS.md section 8 requires
   exit 3 for any current failed row. Test:
   `test_inspecting_an_old_run_with_a_failed_row_exits_3`.
4. Documentation: `docs/benchmarks.md` still called the prompt-template cache a
   follow-up, but it shipped in `d75463b`. It now says so and notes that the smoke
   results predate it.

All six new tests failed against the unfixed `src/` and pass after the fix.

## Real benchmark evidence versus pending

- **Real (no model involved):** the string baselines (exact, difflib), BM25 top-1 and
  candidate recall on the four 50-record pilot variants. The prompt-render CPU profile
  and the template-cache timing (0.440 → 0.188 s trivial model, about 1.5% at 200 ms per
  call).
- **Synthetic (plumbing only):** every number involving a model answer, including
  xwalk_full, retrieval+LLM, the pairwise scorer and clustering predictions.
- **Pending:** any real-model matching or clustering quality, cost or latency.
  LinkTransformer, WDC, Splink and dedupe comparisons. Dense retrieval. Clustering
  B-cubed on a real judge.

## Remaining issues

Follow-up (same day, after this review): the items marked **Fixed** were fixed on this
branch, each with a regression test that failed first. Gates after the follow-up: pytest
1214 passed / 16 skipped; ruff check and format (src tests benchmarks examples scripts),
mypy and `scripts/check_readme_quickstart.py` clean.

| Severity | Issue | Blocks release? |
|---|---|---|
| Medium | **Fixed in `a64d20e`.** Clustering: a completed run with `failed` members cannot be retried by resuming. The rerun makes 0 calls and stays `failed`/exit 3; only a new `--out` (full recompute) helps, and the guide does not say so. Matching does retry failed rows on resume. Now a resume round restores the exported state, re-decides each failed source and stores a new selected state revision; history is kept and an interrupted round resumes to the same rows (`docs/guide/clustering.md`). | No (clustering is experimental). Fix or document before clustering leaves experimental |
| Medium | MCP tool path arguments are unsandboxed: `out`/`index_dir` can create directories anywhere, and the `job` path can be any file. Writes never overwrite foreign files (`out` refuses non-empty non-run directories), and since fix 2 file content is no longer echoed, but unknown-field messages still name the keys of any YAML mapping file. This is documented as "no `--root` sandbox". | No, for local stdio use by the file's owner. Add `--root` before advertising MCP for untrusted agents |
| Low | **Fixed in `a64d20e`.** Clustering `--max-calls` below the calls one step needs (2+) never makes progress. Each invocation spends its budget and aborts with nothing persisted. Small budgets waste work (cap 3: 89 calls versus 66 uninterrupted). A positive cap below `2 + pool.max_expansion_pages` is now refused up front (`usage`, exit 2). | No. Document a minimum, or refuse `max_calls < settings.max_calls_per_member_decision` |
| Low | Pydantic lax mode coerces types: `accept_at: "0.6"` and `max_attempts: true` (→ 1) are accepted. Still open (not trivial to change without breaking existing job files). | No |
| Low | **Fixed in `19cb3d1`.** Several in-flight records hitting the call limit repeat `call_limit_reached` in `errors` (3 identical entries seen). Now reported once. | No (cosmetic) |
| Low | **Fixed in `c087538`.** Dev-only lower bounds: `pytest>=8.0` with `pytest-asyncio>=0.23` resolves to an incompatible pair (INTERNALERROR at collection). Runtime lower bounds are fine. Now `pytest>=8.2`, `pytest-asyncio>=0.24`; the lowest pair (8.2.0/0.24.0) ran the suite in a scratch venv (0.23.5 errors under pytest 8.3.5). | No. Raise to `pytest-asyncio>=0.24` |
| Low | **Fixed in `53aea88`.** A retriever `name` such as `../x` places its index outside the index directory (job files are the user's own configuration). `slots.yaml` still uses plain `safe_load` (duplicate keys pass there). Names are now validated as safe identifiers, and slots files use the job file's duplicate-key loader. | No |
| Low | Carried over from task 07: the README install line points at the default branch, relative links break on PyPI, `inspect` shows the job's model for an injected offline model, and CONTRIBUTING's lint paths omit `benchmarks`. | No |

## Supported versus experimental

- **Supported:** matching (Python SDK and `xwalk.ops`); `init`, `validate`/`doctor`,
  `index`, `match`, `search`, `inspect`, `explain`, `results`, `export`, `review`;
  `--json` and the exit codes; `--max-calls`; resume, snapshot and history; the 0.1.1
  ledger upgrade; BM25; OpenAI-compatible client; MCP stdio (optional extra, local use).
- **Experimental or unevaluated:** `xwalk cluster` (scripted judge only, provisional
  thresholds); dense retrieval (toy encoders only in this review); LiteLLM retry
  accounting; benchmark harness results (synthetic); MCP host configurations (untested).

## Spending

Promotional balance: **unreported**. Starting and ending balances were not supplied to
this session, and no estimate is given in their place.

## Next user actions

1. Review this branch's commits and merge `claude/zealous-heisenberg-nri5m0` (or open a
   pull request from it). No pull request was opened.
2. **Register the PyPI Trusted Publisher before tagging.** PyPI → Account → Publishing →
   "Add a new pending publisher" with: project name `xwalk`, owner `jan3657`, repository
   `xwalk`, workflow `release.yml`, environment `pypi`. On GitHub, create the `pypi`
   environment (Settings → Environments), ideally with a required reviewer. The v0.1.1
   publish failed with `invalid-publisher` for exactly these claims (`docs/releasing.md`
   step 3).
3. Tag the merged commit: `git tag v0.2.0rc1 <commit> && git push origin v0.2.0rc1`.
   The release workflow runs the gates on that commit, checks that the tag matches the
   version, builds, verifies the wheel and the README quickstart, then waits for
   approval in the `pypi` environment.
4. After publishing, check `pip install --pre xwalk` in a clean environment.
5. Before 0.2.0 final, optionally run one small real-model evaluation
   (`python -m benchmarks.run --real --max-calls N`) and address the two Medium items if
   clustering or MCP is to be promoted.

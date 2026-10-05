# Architecture

A compact map for contributors and agents: what each layer owns, the invariants every
change must keep, and where to look for a given change. [Components](components.md)
describes every module; [concepts](concepts.md) explains why the loop is shaped as it is.
The binding 0.2 decisions are in
[CONTRACTS.md](claude-upgrade/CONTRACTS.md).

## Layers

```
 CLI (cli/main.py)    MCP server (mcp_server.py)    your Python code
          \                   |                       /
           +----------  ops.py: validate, index, run, inspect, explain,
                        export, search, results, review, init, cluster
                                     |
     config.py (JobSpec, strict)     |      cluster/ (experimental, own store)
                                     v
     batch.py run_batch  --->  matcher.py Matcher  --->  stages/ select, score,
       (snapshot, workers,        (attempt loop)          verify, rewrite, keying
        cancellation, exports)        |                        |
              |                  retrieval/ bm25, dense,   llm/ clients, budget,
              v                  fusion; stores/           cache, parsing, fake
         ledger.py (SQLite: results, history, snapshots, reviews)
              |
     review.py overlay      evaluate/ (reads the ledger only)
```

- **Front ends only format.** `cli/main.py` and `mcp_server.py` call one `xwalk.ops`
  operation and print or return its `OpResult` (the `--json` envelope, schema 1). Logic
  that decides anything lives below `ops`. (`eval`, `compare`, `ablate` and `prompts`
  are still assembled in the CLI.)
- **`ops` owns orchestration**: strict job loading, preflight (files, extras,
  credentials by name), reading targets once, planning and opening/building indexes by
  their identity components, run-directory collision checks, the per-invocation call
  budget, and the exit-code mapping.
- **`Matcher` decides one record**; `run_batch` schedules many with a bounded worker
  pool, commits each result, and writes exports from the ledger's current view.
- **The ledger is the run.** Exports (`mapping.csv`, `results.jsonl`,
  `manifest.json`) are regenerated from it.
- **Clustering** (`cluster/`, experimental) reuses keying, retrieval fusion, the LLM
  layer and `ops` helpers, but has its own SQLite store and never calls `Matcher` on one
  collection against itself.

## Invariants

Each one has tests; a change that needs to weaken one needs a reviewed contract change.

1. **A malformed model answer never resolves to a real target.** Candidates carry opaque
   per-attempt keys; resolution is exact lookup only (`stages/keying.py`).
2. **Failure fails toward review.** Unparseable output, an unknown verdict, a
   non-finite or out-of-range confidence, or an unverified in-band score cannot become
   `matched` (`policy.derive_status`, `llm/parsing.py`).
3. **Provider errors never escape `Matcher.match`.** Recoverable errors stay in the
   record; a fatal one yields `failed`/`fatal_provider_failure`, which `run_batch` does
   not commit and treats as a run abort (CONTRACTS §3-4).
4. **One current row per source.** Every export reads the current view of the latest
   complete source snapshot; everything ever committed stays in history (CONTRACTS §2).
5. **Run identity is a fingerprint** of everything that changes what a result means
   (templates, prompts, targets, retriever identities, LLM identity and effective
   generation settings, policy, library version), never credentials, paths or
   concurrency. A run directory holds one fingerprint; a different one is refused.
6. **Indexes are reused only when their stored components match**; an incompatible
   index is refused, never silently rebuilt or reused. `build` replaces an index
   directory's contents, never appends.
7. **No silent retriever substitution** and **a light base install**: heavy
   dependencies sit behind extras, imported lazily, failing with `MissingExtra`.
8. **Calls are counted honestly.** Every dispatched upstream request counts, retries
   included; the call budget is reserved before dispatch; unknown token usage is
   reported as unknown, never zero.
9. **Evaluation makes no model or retriever calls**; it reads the ledger.
10. **Reviews are an overlay** bound to a result revision; model output is never edited.
11. **`--json` prints exactly one JSON object on stdout**; logs and progress go to
    stderr (the MCP stdio transport depends on this).

## Contributor map

| If you change... | Look at | Tests to run first |
|---|---|---|
| How a record is decided (attempt loop, routing, rewrites) | `matcher.py`, `stages/*.py`, `policy.py` | `test_matcher.py`, `test_matching_regressions.py`, `test_gate.py`, `test_select.py`, `test_rewrite.py`, `test_policy.py` |
| Candidate keys or resolution | `stages/keying.py`, `stages/proposals.py` | `test_keying.py`, `test_proposals.py` |
| Status or reason semantics | `policy.py`, `records.py` | `test_policy.py`, `test_records.py`, `test_matching_regressions.py` |
| Prompts, slots, skeletons | `prompts/contract.py`, `prompts/base/*.j2` | `test_prompt_contract.py`, `test_packaging.py` |
| Model output parsing | `llm/parsing.py` | `test_parsing.py` |
| An LLM client, usage or retries | `llm/openai_compat.py`, `llm/litellm.py`, `llm/base.py` | `test_openai_compat.py`, `test_litellm.py`, `test_fake_llm.py` |
| Call limits or caching | `llm/budget.py`, `llm/cache.py` | `test_call_budget.py`, `test_llm_cache.py` |
| Retrieval or index identity | `retrieval/bm25.py`, `retrieval/dense.py`, `retrieval/fusion.py` | `test_bm25.py`, `test_dense.py`, `test_fusion.py` |
| Loading records | `sources/*.py`, `stores/*.py` | `test_sources.py`, `test_ontology_source.py`, `test_sql_source.py`, `test_stores.py` |
| Batch scheduling, resume, cancellation | `batch.py` | `test_batch.py`, `test_lifecycle.py` |
| Ledger schema, current view, history | `ledger.py`, `serde.py` | `test_ledger.py`, `test_ledger_compat.py`, `test_serde.py`, `test_lifecycle.py` |
| Run fingerprints | `fingerprint.py`, `batch.build_run_fingerprint`, `config.py` | `test_fingerprint.py` |
| Review | `review.py` | `test_review.py` |
| Job file keys or validation | `config.py` (and `cluster/job.py`) | `test_config.py`, `test_job_validation.py` |
| An operation, JSON envelope or exit code | `ops.py` | `test_ops.py`, `test_ops_read.py`, `test_cli_json.py` |
| CLI flags or output | `cli/main.py` | `test_cli.py`, `test_cli_json.py` |
| MCP tools | `mcp_server.py` | `test_mcp.py` (needs `xwalk[mcp]` for most) |
| Clustering | `cluster/*.py` | `test_cluster_engine.py`, `test_cluster_pool.py`, `test_cluster_ops.py` |
| Metrics, gold, reports | `evaluate/*.py` | `test_metrics.py`, `test_gold.py`, `test_ceiling.py`, `test_eval_report.py`, `test_compare.py`, `test_ablate.py` |
| Prompt drafting or optimisation | `prompts/author.py`, `prompts/optimize.py`, `evaluate/failures.py`, `evaluate/partition.py` | `test_author.py`, `test_optimize.py`, `test_failures.py`, `test_partition.py` |
| Packaging, bundled quickstart, version | `pyproject.toml`, `resources/quickstart/` | `test_packaging.py`, `test_docs.py` |
| Documentation | `README.md`, `docs/` | `test_docs.py` (links, Python blocks, index, README quickstart) |
| Benchmarks (repository only) | `benchmarks/` | `test_benchmarks.py` |

Every commit passes the four gates in [CONTRIBUTING.md](../CONTRIBUTING.md). Tests use
`FakeLLM` and deterministic fixtures; nothing in the default suite calls a paid API.

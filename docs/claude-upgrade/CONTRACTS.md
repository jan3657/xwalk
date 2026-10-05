# xwalk 0.2 contracts

Decided in task 00 (5 October 2026) against commit `2dc58b1` (source identical to the
audited `f737099`). Later sessions read this file instead of the historical plans. Change a
contract only in a reviewed commit that edits this file and says why.

## 1. Runtime state versus traces

- The matcher's live candidate state (fused candidates, issued keys, the `KeyedCandidates`
  reused for candidate proposals) is never read back from `Attempt`/`MatchResult`.
- `keep_candidates_in_trace` only controls what is serialized. With identical inputs and
  scripted responses, trace on/off must give identical status, matched id, confidence,
  attempts (ignoring the `candidates` field) and identical outgoing LLM requests.

## 2. Current snapshot versus history

- **Source snapshot**: each `run_batch` invocation reads the full source collection and
  records its ordered `(source_id, source_hash)` list as the run's current snapshot.
  Duplicate source IDs in one snapshot are a validation error.
- **Current view** = for each snapshot entry, the result whose `result_key` matches
  `(run_fingerprint, source_id, source_hash)`. Every export (`mapping.csv`,
  `results.jsonl`, manifest counts, `duplicate_targets`, `BatchReport`, reviewed export)
  uses the current view. Exactly one row per current source.
- **History** = every result ever committed, including superseded source versions,
  sources removed from the collection, and failed attempts that were later retried. It
  is kept in the ledger, never deleted, and exposed by an explicit history query/export.
- Removed source: absent from the current view, still in history.
- `--limit N`: the snapshot is still the full collection; only the first N unfinished
  records are processed. Unprocessed snapshot entries count as `pending`; the run state
  is `partial` (see section 8).
- A result with status `failed` is retried on resume. The old failed result moves to
  history; the new one becomes current.
- Reviews stay keyed by `result_key`. A review whose result is no longer current is
  shown as stale in history and is not applied to the current reviewed export.
- The ledger schema change is additive with an explicit `schema_version`; a v0.1.1
  ledger opens read-compatible (all its rows become history, its last snapshot is
  inferred as "all source ids with their latest row"). Unknown future versions are
  refused, never migrated in place.

## 3. Provider errors

| Error | Where | Outcome |
|---|---|---|
| Recoverable (`LLMRetryableError` after client retries, timeout, unparseable output, retriever failure) | selector, scorer | attempt `reason=provider_failure` / `retriever_failure`; same query retried within `max_attempts` (existing) |
| Recoverable | gating verifier (score in `verify_band`) | attempt keeps its score, `verifier_decision="error"`, `reason=provider_failure`; retried like above; an unverified in-band score can never become `matched` (best possible status `needs_review`) |
| Recoverable | audit verifier (score outside band) | decision unchanged; error noted on the attempt |
| Recoverable | rewriter | loop ends with no new query; status derived from existing attempts; error noted on the last attempt |
| Fatal (`LLMFatalError`: auth, unknown model, invalid request) | any stage, including verifier and rewriter | `Matcher.match` returns `failed` with new reason `fatal_provider_failure` (never raises for provider errors) |
| Ledger/IO error, programming error | anywhere | raised |

`run_batch` treats a `fatal_provider_failure` result as a run abort: it does **not**
commit that result (so resume retries the record), stops scheduling, and cancels as below.
Malformed or invalid model output is a recoverable per-record outcome, never fatal.

## 4. Cancellation and cleanup

On a fatal result, an exception in any record task, `KeyboardInterrupt` or task
cancellation: stop creating tasks; cancel all in-flight record tasks; `await` them all
(`return_exceptions=True`); discard uncommitted partial results; write exports from the
current view and the manifest with `run_state` (`aborted` or `interrupted`); then close
the ledger. No ledger write may happen after `close()`. Committed results stay valid for
resume. Scheduling uses a bounded worker pool fed from the source iterator, not chunk
barriers.

## 5. Usage accounting

- `Usage` gains `unknown_calls`. `calls` counts every upstream request that was
  dispatched, including client-internal HTTP retries and calls that failed after
  dispatch. Token fields sum only provider-reported tokens. A call whose tokens are not
  reported (error, interruption, provider omitted usage) increments `unknown_calls`.
- Every stage call is attributed to an attempt. Rewriter calls are attributed to the
  attempt after which the rewrite ran (`dataclasses.replace`). Invariant:
  `result.usage == sum(a.usage for a in result.attempts)` and equals the number of
  scripted FakeLLM calls.
- Cache hits are not upstream calls; they are counted separately as `cache_hits`.
- Displays print tokens as "N (+k calls with unknown usage)" when `unknown_calls > 0`,
  never as a silent zero. Estimates, if ever shown, are labelled as estimates.
- Old ledgers deserialize with `unknown_calls=0`, `cache_hits=0`.

## 6. Generation parameter precedence

Highest first: explicit per-stage constructor argument in Python (e.g.
`Rewriter(max_tokens=256)`) > job `llm.*` (`temperature`, `max_tokens`, optional `seed`)
> client constructor defaults > library defaults. `LLMRequest.temperature/max_tokens`
become optional (`None` = use the client value); stages stop hardcoding 512/256 and
`0.0`. The run fingerprint hashes the *effective* per-stage values. The test is the
captured outgoing request body, not the fingerprint.

## 7. Semantic index identity

- An index fingerprint covers engine and index format version, doc template, sorted
  record digests, and engine settings: BM25 `exact_fields` and normalization version;
  dense encoder model name, model revision (resolved hash, or `"unknown"`), dimension,
  `normalize`, `query_prefix`, `doc_prefix`, max sequence length. Meta files store the
  components, not only the hash.
- The expected fingerprint is computed from the job and target records without encoding.
- Default for `match`/`index` against an index directory: absent → build; compatible →
  open without calling the encoder; incompatible → refuse (exit 3) naming the differing
  components. `--rebuild-index` replaces it explicitly. Never silently rebuild over or
  reuse an incompatible index.
- Targets are read once per command and shared by store and index build.

## 8. Operations, output and exit codes

- One operations module (`src/xwalk/ops.py`) owns validate, prepare (index), run,
  inspect, explain and export. CLI, Python high-level API and MCP call it; they only
  format.
- Job validation is strict: pydantic `extra="forbid"` on every spec, range checks
  (0 ≤ thresholds ≤ 1, `review_floor ≤ accept_at`, positive limits), duplicate IDs
  rejected, optional extras checked. Validation never makes a paid call.
- `--json` on every command prints exactly one JSON object to stdout:
  `{"schema_version": 1, "operation", "status", "exit_code", "run": {"dir",
  "run_fingerprint", "run_state"}, "counts", "usage", "artifacts", "warnings", "errors":
  [{"code", "message", "source_id"?}]}`. Progress and logs always go to stderr.
- `run_state`: `complete` | `partial` (pending sources) | `aborted` (fatal) |
  `interrupted`.
- Exit codes (existing 0-3 kept, meaning tightened):
  - `0` complete, no review rows, no failed rows.
  - `1` attention: complete with `needs_review` rows, a `partial` run from `--limit`,
    or rejected review rows.
  - `2` usage or configuration error (bad flags, invalid job file). Invalid job files
    move from 3 to 2: documented as a migration in the changelog.
  - `3` runtime failure: `aborted`, IO error, incompatible index/run directory, or any
    current row with status `failed`. Precedence 3 > 1 > 0.
  - `130` interrupted by the user.

## 9. Run directories and repeated invocation

- A run directory is bound to one run fingerprint (recorded in the manifest with its
  components). The run id is that fingerprint.
- `match --out D`: D absent or empty → new run. D holds the same fingerprint → resume.
  D holds a different fingerprint → refuse (exit 3) and print which fingerprint
  components changed; the user chooses a new `--out`. D is a non-empty non-xwalk
  directory → refuse. Nothing is overwritten implicitly.
- Repeating an unchanged complete run makes zero LLM calls and regenerates identical
  exports (idempotent). `--no-resume` recomputes every record; earlier results go to
  history.
- Source edits do not change the run fingerprint; they are handled by the snapshot.

## 10. Flat equivalence clustering (first release)

Relation: **equivalence only** — two records are equivalent when they denote the same
entity or concept and their labels are interchangeable. Broader/narrower, part-of and
"related" are not equivalence and must stay in separate clusters. Cardinality one.
Reconciling the July spec (`docs/superpowers/specs/2026-07-28-hierarchical-clustering-design.md`)
and its M1 plan:

1. No taxonomy seeding, subsumption levels, multi-parent or taxonomy mutation in 0.2.
   All clusters are minted in-run. (Assigning to immutable taxonomy nodes is a later
   milestone and is a different operation from merging.)
2. Run identity = fingerprint of the ordered source snapshot `(id, hash)`, the ordering
   rule (default stable sort on normalized label, then id), prompts, LLM, retrieval and
   policy settings. Cluster IDs = hash of `(run_fingerprint, seed_member_id)`; names
   never affect identity.
3. Cluster state is versioned: append-only `cluster_revisions(cluster_id, revision,
   member_ids, representation_text)`. Every decision row stores the
   `(cluster_id, revision)` pairs **retrieved** and, separately, **shown**, plus the
   rendered prompt and raw response, so the pool used for any decision is reconstructable.
4. Per-source final outcome, exactly one of: `assigned` (verified member of a cluster),
   `singleton` (new cluster created after retrieval expansion and a passed novelty
   verification), `needs_review`, `failed`. `deferred` is transient and becomes
   `needs_review` at the end. Accepted partition = `assigned` + `singleton` grouped by
   cluster. `needs_review`/`failed` are in no accepted cluster; they are exported with
   their best proposal. Singletons are never described as verified equivalences.
5. Novelty requires the configured retrieval expansion first; mint provenance records it.
6. No transitive merges: a member is verified against the cluster's current
   representation with bounded actual member evidence (up to N member labels and
   fields), never against a single member or exemplar counts. Cluster merges need their
   own verified cluster-vs-cluster decision; a failed reverification goes to review.
7. Refinement: at most `max_refine_iterations` (default 2) of retry-deferred, consolidate,
   reassign. Reassignment always injects the incumbent into the shown list (replacing
   the lowest-ranked candidate if truncated) and switches only on a comparative verdict
   at margin; it never mints. Stop reason `converged` | `max_iterations` | `oscillation`.
   Exports use the best stable revision (fewest `needs_review`, ties earliest), not the
   last transient state.
8. The pool index is incremental (append on mint); no full rebuild per mint.
9. Call/candidate limits from section 5/policy apply to every clustering call.
10. Code lives in `src/xwalk/cluster/`; it does not call `Matcher` with the same
    collection on both sides.

## 11. File ownership and task order

Order: 00 → 01 → 02 → 03 → 04 (05 fixtures/runners may start in parallel with 04) → 06
(optional) → 07 → 08. At most two sessions at once, never two on matcher or ledger.
Integration branch: `release/0.2`, created by Jan from the accepted task 00 commit; each
task branches from its tip and is merged back after review. Until it exists, task 01
starts from `claude/zealous-heisenberg-nri5m0`.

| Task | Owns |
|---|---|
| 01 | `matcher.py`, `stages/*`, `llm/base.py`, `llm/parsing.py`, `llm/openai_compat.py`, `llm/litellm.py`, `llm/fake.py`, `records.py` (Usage fields, new reason), `policy.py` (`derive_status`), `serde.py` (Usage compat), their tests |
| 02 | `ledger.py`, `batch.py`, `fingerprint.py`, `review.py` current view, ledger fixture of v0.1.1, `tests/test_batch.py`, `tests/test_ledger.py`, `tests/test_review.py` |
| 03 | new `ops.py`, `cli/main.py`, `config.py`, index open/validate in `retrieval/bm25.py` and `retrieval/dense.py`, `docs/reference/cli.md`, `docs/reference/job-file.md` |
| 04 | new `src/xwalk/cluster/`, `tests/test_cluster_*.py`, one `cluster` hook in `ops.py`/CLI |
| 05 | new `benchmarks/` (manifests, runners), `scripts/` additions |
| 07 | `README.md`, `docs/`, `CHANGELOG.md`, version bump |

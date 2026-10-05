# Command line

The SDK is primary. The CLI is a shell over it: every subcommand parses arguments, calls one
operation in `xwalk.ops`, prints the `OpResult` it returns, and exits with its code. Nothing
decision-shaped lives in `xwalk/cli/main.py`, which is why anything the CLI does you can do
from Python — `ops.run`, `ops.inspect`, `ops.explain`, `ops.export` return the same result
the CLI prints — and why the things it does not expose (a separate optimiser model, custom
gold normalisation, your own retriever) are not missing features but calls you make
yourself.

Everything below takes a [job file](job-file.md); see that page for what its fields mean.

```
xwalk --version
xwalk <command> [flags]
```

Running `xwalk` with no command prints help and returns `2`. `--help` — on the group or on
any subcommand — returns `0`; argparse's process exit is caught, so `main()` is callable as
a library function and never raises `SystemExit` at you.

## Exit codes

| Code | Name | Meaning |
|---|---|---|
| `0` | success | the command did what it said; for a run: complete, no review rows, no failed rows |
| `1` | attention | completed, but something needs a human: a non-empty review bucket, a `partial` run from `--limit`, or rejected review rows |
| `2` | usage / configuration | bad invocation (no command, unknown command or flag, bad flag value, malformed `--run`), an invalid or unreadable job file, a missing credential variable or optional extra, duplicate record ids, an unknown source id or a path that is not a run directory |
| `3` | runtime | a real failure: an `aborted` run (fatal provider error, `--max-calls` reached, an exception), any current row `failed`, an IO error, an incompatible index or run directory, a stale review snapshot |
| `130` | interrupted | Ctrl-C; the run directory records `interrupted` and resumes from there |

When several apply, `3` wins over `1`, and `1` over `0`. Since 0.2 an invalid job file and a
missing credential exit `2` (they exited `3` in 0.1).

Without `--json`, the result prints as lines on stdout; warnings print as `warning: ...` and
errors as `error: <code>: <message>` on stderr. A CLI must not dump a traceback, so it does
not.

## Machine-readable output: `--json`

Every command takes `--json` (on the leaf subcommand: `xwalk review export --json ...`).
stdout then carries exactly one JSON object, on success and on failure, including argparse
errors; progress and logs stay on stderr, so `xwalk match ... --json | jq .` always parses.

```json
{
  "schema_version": 1,
  "operation": "match",
  "status": "attention",
  "exit_code": 1,
  "run": {"dir": "runs/chebi", "run_fingerprint": "3f2a9c0d1e4b5a67", "run_state": "complete"},
  "counts": {"matched": 41, "needs_review": 6, "unmatched": 3, "total": 50},
  "usage": {"calls": 112, "prompt_tokens": 52011, "completion_tokens": 9223,
            "unknown_calls": 0, "cache_hits": 0, "tokens": "61234", "limit": 500},
  "artifacts": {"mapping.csv": "runs/chebi/mapping.csv", "index": "runs/chebi/index"},
  "warnings": [{"code": "needs_review", "message": "6 row(s) need review: ..."}],
  "errors": [],
  "data": {"indexes": {"bm25": "opened"}, "job": "chebi", "model": "gpt-4o-mini"}
}
```

| Key | Meaning |
|---|---|
| `schema_version` | `1`; a breaking change to this shape bumps it |
| `operation` | the command (`match`, `validate`, `review export`, ...) |
| `status` | `ok` (exit 0), `attention` (1), `error` (2 or 3), `interrupted` (130) |
| `exit_code` | the process exit code |
| `run` | `{dir, run_fingerprint, run_state}` for run-level commands, else `null` |
| `counts` | current-view status counts plus `pending` and `total`, or the command's own counts |
| `usage` | calls and tokens of this invocation; `tokens` never hides unknown usage behind a zero; `null` when nothing ran |
| `artifacts` | files written or read, by name |
| `warnings`, `errors` | lists of `{code, message, source_id?}`; codes are stable identifiers |
| `data` | command-specific detail: `validate` credentials and checks, `inspect` fingerprint components and invocation, `explain` decision and attempts, `search` candidates, `results` one page of rows |

Error codes include `job_not_found`, `job_yaml_invalid`, `unknown_field`, `missing_field`,
`invalid_value`, `missing_file`, `missing_extra`, `credential_missing`, `duplicate_id`,
`index_mismatch`, `run_fingerprint_mismatch`, `out_not_a_run`, `run_not_found`,
`source_not_found`, `fatal_provider_failure`, `call_limit_reached`, `exception`,
`interrupted` and `usage`. Clustering adds `wrong_job_kind` (a `kind: cluster` job given
to `match`), `source_failed`, `out_not_a_directory`, `store_mismatch`, and the
`experimental` warning. `search` adds the `retriever_failure` warning; the MCP server's
`--offline-model` adds the `offline_model` warning.

## `init`

Copies the bundled quickstart job — `job.yaml`, `slots.yaml`, `targets.csv`,
`sources.csv`, `README.md` — into a new directory.

| Argument | Required | Default | Meaning |
|---|---|---|---|
| `DEST` | no | `xwalk-quickstart` | directory to create; an existing non-empty one is refused (exit `3`) |

```console
$ xwalk init demo
created demo (README.md, job.yaml, slots.yaml, sources.csv, targets.csv)
next: xwalk validate --job demo/job.yaml
```

The data is a five-term vocabulary and four mentions, one with no correct target. Matching
it needs the endpoint named in `job.yaml`, or the Python route with a scripted model
(`examples/quickstart.py`), which runs offline.

## `validate` (alias `doctor`)

An offline preflight. It never builds an LLM client and never calls a model.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--no-credentials` | no | — | do not require the `llm.api_key_env` variable to be set |
| `--no-scan` | no | — | do not read the collections (skips duplicate-id and empty-text checks) |

Checks, reporting every problem rather than the first: the job file strictly (unknown or
misspelled keys with a did-you-mean hint, missing fields, out-of-range values, fields that
do not apply to the `kind`, duplicate retriever names); that the templates compile and the
prompt slots satisfy the contract; that referenced files exist; that optional extras the
job needs are installed; that the credential variable is *set* — reported as `set` or
`missing`, its value is never read into the output; and, unless `--no-scan`, that both
collections load, are non-empty and have unique ids (empty rendered texts are warnings).

```console
$ xwalk validate --job job.yaml
job 'quickstart' is invalid
  targets : 5 records
  sources : 4 records
  OPENAI_API_KEY: missing
error: credential_missing: environment variable OPENAI_API_KEY (llm.api_key_env) is not set
```

Returns `0` when valid, `2` otherwise.

## `index`

Prepares every retriever index the job declares, and prints each one's fingerprint and
what was done to it.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--out` | yes | — | index directory; each retriever gets `<out>/<name or kind>` |
| `--rebuild-index` | no | — | replace an incompatible index instead of refusing |

For each retriever the expected identity is computed from the job and the target records
without encoding anything: engine and format version, doc template, a digest of the
records, BM25 `exact_fields` and normalization version, and for dense the encoder name,
model revision (`"unknown"` unless `revision` is pinned), dimension, `normalize`, prefixes
and max sequence length. Then: an absent index is **built**; one whose stored identity
matches is **opened** without calling the encoder; one that differs is **refused** with
exit `3`, naming the differing components, unless `--rebuild-index` replaces it. Every
index is checked before any is touched.

```console
$ xwalk index --job examples/chebi/job.yaml --out runs/chebi/index
indexed 200 target records into 1 retriever(s)
  bm25: 9ce1441b96ffef65 (built)
```

Writes a Tantivy index plus `xwalk_meta.json` for `bm25`, or `vectors.f32`,
`record_ids.json` and `xwalk_meta.json` for `dense`; the meta file stores the identity
components. An index written before 0.2 has no stored components and is refused until
rebuilt.

`match` prepares indexes the same way, so `index` is not a required first step — it is how
you pay the indexing cost separately, confirm the target loader works, and read the
fingerprint before spending anything on a provider.

## `match`

The run. Validates the job, reads the targets once, prepares the indexes (as `index` does),
builds the client and matcher, then matches every unfinished source record into a
resumable run directory.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--out` | yes | — | run directory |
| `--index` | no | `<out>/index` | index directory |
| `--resume` | no | on | skip records already completed under this run fingerprint; `failed` records are retried |
| `--no-resume` | no | — | re-run everything; earlier results stay in the ledger's history |
| `--limit` | no | all | process only the first N unfinished source records, in file order; the rest are reported as `pending` |
| `--max-calls` | no | none | cap the upstream LLM requests of this invocation |
| `--rebuild-index` | no | — | replace an incompatible index instead of refusing |

**Run directories.** A run directory belongs to one run fingerprint, which is computed
before any index or file is touched. `--out` absent or empty starts a run; the same
fingerprint resumes it (an unchanged, complete run makes zero LLM calls and rewrites
identical exports); a different fingerprint is refused with exit `3` and the changed
fingerprint components (`policy.accept_at`, `prompts`, `llm`, ...) — choose a new `--out`.
A non-empty directory that is not an xwalk run is refused too. Nothing is overwritten
implicitly. Editing source records does not change the fingerprint; the run's snapshot
handles it.

**Call limits.** `--max-calls N` reserves one unit before every upstream request — the
OpenAI-compatible client's internal HTTP retries, structured-output fallbacks, verifier
and rewriter calls included — so concurrent records can never dispatch more than `N`.
When a request would exceed it, nothing is dispatched, the record is not recorded, the run
is `aborted` with one error `call_limit_reached` (however many records it stopped; exit `3`), and a later `match` resumes from
there. The limit is per invocation. Retries hidden inside LiteLLM (`num_retries`) cannot be
observed or limited; set `num_retries=0` there. The reported `usage` counts calls that were
in flight when the run stopped as calls with unknown usage, never as free.

Prints the number of current source records, the run state (`complete`, `partial`,
`failed`, `aborted`, `interrupted`), a count per status sorted by status name (with
`pending` and `total`), and tokens (with calls of unknown usage shown, never as zero).
Per-record progress, the duplicate-target and review-bucket warnings, and errors that
stopped the run go to stderr.

```console
$ export OPENAI_API_KEY=sk-…
$ xwalk match --job examples/chebi/job.yaml --out runs/chebi --max-calls 500
matched 50 records into runs/chebi
  run state     : complete
  matched       : 41
  needs_review  : 6
  total         : 50
  unmatched     : 3
  tokens        : 61234 in 112 calls
```

and on stderr, after the progress lines:

```console
warning: duplicate targets: 2 matched by more than one source (see manifest.json)
warning: 6 row(s) need review: xwalk review export --run runs/chebi
```

Writes into the run directory:

| File | What it is |
|---|---|
| `mapping.csv` | the deliverable, one row per current source record: `source_id, matched_id, confidence, status, reason, explanation, attempts, prompt_tokens, completion_tokens, llm_calls, elapsed_seconds, unknown_calls, cache_hits`; records not yet processed have status `pending` |
| `results.jsonl` | one full current result per line, attempts and candidates included |
| `manifest.json` | run fingerprint and its components, library version, target fingerprint, job name, model, run state, usage, errors, status counts (with `pending`), duplicate targets, removed sources, ledger schema version |
| `ledger.sqlite` | the source of truth; makes the run resumable and evaluation free |

Returns `1` — not `0` — whenever the review bucket is non-empty or the run is `partial`
(`--limit`). That is a healthy outcome, not an error; see [Gotchas](#gotchas). Returns
`3` when the run is `aborted` (for example an invalid API key: the fatal record is not
recorded and is retried on the next run) or `failed` (some records failed, for example
the provider was unavailable; rerun to retry them).

The run directory always shows the current state of each source record. Edit a source
record and rerun: the record is matched again and `mapping.csv` keeps one row for it.
Remove a source record and rerun: it disappears from the exports (the manifest counts it
under `removed_sources`). Earlier results are never deleted; they stay in the ledger's
history (`xwalk export --view history`).

## `cluster`

Experimental. Groups the records of one collection into clusters of equivalent records,
from a `kind: cluster` job. The workflow, the job keys, the outputs and the outcomes are
described in [Clustering one collection](../guide/clustering.md).

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | a `kind: cluster` job file |
| `--out` | yes | — | run directory |
| `--max-calls` | no | none | cap the upstream LLM requests of this invocation; reaching it aborts at a step boundary (exit `3`, `call_limit_reached`) and the next invocation resumes. A positive value below `2 + pool.max_expansion_pages` (one record's decision) is refused with exit `2` |

Run directories follow the `match` rules, with one difference: the ordered source
snapshot is part of the run identity, so an edited source collection is a new run and an
existing `--out` is refused. Resuming a finished run retries its `failed` records. Writes `members.csv`, `clusters.csv`, `unresolved.csv`,
`decisions.jsonl`, `manifest.json` and `cluster.sqlite`. Per-record progress goes to
stderr. Exit codes: `0` complete, `1` complete with `needs_review` records, `2` usage or
job error, `3` aborted run, failed records or a foreign run directory, `130` interrupted.
`--json` data: `stop_reason`, `selected_revision`, `last_revision`, `exported_revision`,
`experimental`.

`validate` (and `doctor`) accept clustering jobs too: strict schema, source file,
extras, credential presence and duplicate source ids, with no model call.

## `search`

Retrieves the fused target candidates for a free-text query: every retriever in the job is
searched and the hit lists are fused by reciprocal rank, as the matcher does. Calls no
model, so it is the cheap way to check that retrieval surfaces the right target before a
paid run.

| Argument / flag | Required | Default | Meaning |
|---|---|---|---|
| `QUERY` | yes | — | the text to retrieve for |
| `--job` | yes | — | path to the job file |
| `--index` | yes | — | index directory; built when absent, opened when compatible, refused (exit `3`) when incompatible |
| `--limit` | no | `10` | candidates to return, `1` to `100` (`ops.MAX_SEARCH_LIMIT`) |

```console
$ xwalk search dextrose --job job.yaml --index run/index
1 candidate(s) for 'dextrose'
    1. CHEBI:17234  ID: CHEBI:17234 Label: glucose Synonyms: dextrose; grape sugar
```

`data.candidates` lists `rank`, `id`, `fused_score`, `retrievers` (name to 1-based rank)
and `text` (the job's `candidate` template). A retriever that fails is a
`retriever_failure` warning, not an error.

## `inspect`

Summarises a run directory from its ledger: identity, state, current counts, usage of the
last invocation, history size and review states.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--run` | yes | — | run directory |

```console
$ xwalk inspect --run runs/chebi
run 3f2a9c0d1e4b5a67 in runs/chebi
  job           : chebi (gpt-4o-mini)
  written by    : xwalk 0.2.0
  run state     : complete
  matched       : 41
  needs_review  : 6
  total         : 50
  unmatched     : 3
  last usage    : 61234 tokens in 112 calls
  history       : 50 results
  reviews       : 6 applied
```

With `--json`, `data` adds the fingerprint components, the last invocation record, the
snapshot, removed sources and review counts by state (`applied`, `superseded`, `stale`).
Calls nothing. Exits with the code the run would have produced (`0`, `1` or `3`), so a
script can test a finished run; a path that is not a run directory is `2`.

## `explain`

Why one source record got its answer.

| Argument / flag | Required | Default | Meaning |
|---|---|---|---|
| `SOURCE_ID` | yes | — | the source record id |
| `--run` | yes | — | run directory |
| `--full` | no | — | add the complete stored result (`data.result`) |

```console
$ python -m examples.quickstart runs/quickstart      # the offline scripted run
$ xwalk explain s2 --run runs/quickstart
source s2 in run b7645b1166d7c9ca
  decision      : matched -> CHEBI:17234 (confidence 0.9, reason accept_threshold)
  explanation   :
  attempt 0: query 'dextrose', 1 candidates, chose CHEBI:17234, score 0.9, verifier -
```

(The scripted model gives no explanation; a real one does.)

Shows the decision (status, matched id, confidence, reason, explanation, result key and
revision, usage), every attempt (query, proposal, candidate count and top candidates,
chosen id, score, verifier decision, reason, error, usage), the applied review with the
final answer, and every result the source has had (`history`). A source removed from the
collection shows its history only. An unknown id exits `2`.

## `results`

One page of a run's current view, ordered by source id: the model's decisions, without
the review overlay (use `export --view reviewed` for that).

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--run` | yes | — | run directory |
| `--status` | no | all | `matched`, `needs_review`, `unmatched`, `failed` or `pending` |
| `--offset` | no | `0` | rows to skip |
| `--limit` | no | `50` | page size, `1` to `200` (`ops.MAX_PAGE_SIZE`) |

Each row has `source_id`, `status`, `reason`, `matched_id`, `confidence`, `revision` and
`result_key`. `data.total` counts the rows that match the filter and `data.next_offset` is
the next page's offset, `null` on the last page. Calls nothing and exits `0`; a path that
is not a run directory or an out-of-range flag is `2`.

## `mcp`

Serves `validate_job`, `search_candidates`, `match_records`, `get_run`, `list_results` and
`explain_result` to an agent over MCP on stdin/stdout until the client disconnects. Needs
`pip install 'xwalk[mcp]'`; without it, `xwalk mcp` prints the install command on stderr
and exits `2`. See the [MCP guide](../guide/mcp.md).

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--max-calls-cap` | no | `500` | the largest `max_calls` a `match_records` call may request |
| `--offline-model` | no | — | answer every model call with a fixed scripted reply (demos and tests; the results are meaningless) |

## `export`

Writes one explicit view of a run.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--run` | yes | — | run directory |
| `--view` | no | `raw` | `raw`, `reviewed` or `history` |
| `--out` | yes | — | file to write |

- `raw`: the current view as the model decided it, in `mapping.csv` columns.
- `reviewed`: the current view with the review overlay applied. Columns:
  `source_id, final_matched_id, final_status, review_decision, corrected_target_id,
  reviewer, review_note, reviewed_at, model_matched_id, model_status, model_reason,
  confidence, result_key`. A reviewed row shows the reviewer's answer *and* the model's
  original decision; unreviewed rows have blank review columns; pending rows have
  `final_status` `pending`. The ledger is not changed.
- `history`: every result ever committed (JSONL: `current`, `revision`, `result`),
  including superseded source versions, removed sources and retried failures. When
  reviews exist, `<out stem>.reviews.jsonl` lists every decision with its state
  (`applied`, `superseded`, `stale`).

```console
$ xwalk export --run runs/chebi --view reviewed --out runs/chebi/reviewed.csv
exported 50 reviewed rows to runs/chebi/reviewed.csv
```

## `eval`

Scores a completed run against gold labels. Reads the ledger and calls nothing: no
retriever, no provider, no key required.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--run` | yes | — | run directory; its `manifest.json` supplies the run fingerprint |
| `--gold` | yes | — | CSV with `source_id,gold_ids`; an empty cell means "no match" is correct |
| `--out` | no | none | path **stem**; writes `<out>.json` and `<out>.txt` |

Prints a report that leads with where to spend effort, then the headline metrics, no-match
handling, the three-way failure decomposition, cost per record, and a calibration warning
when confidences do not separate correct from incorrect.

```console
$ xwalk eval --run runs/chebi --gold examples/chebi/sample/gold.csv --out runs/chebi/eval
## Where to spend effort
  56% of failures presented the gold record and the model chose otherwise --
  spend effort on prompts, not retrieval

## Headline
  accepted precision   : 92.7%
  automatic coverage   : 82.0%
  review rate          : 12.0%
  unmatched rate       : 6.0%
  error rate           : 0.0%
  unresolved rate      : 0.0%
  recall at any status : 89.5%

## No-match handling
  no-match precision   : 75.0%
  no-match recall      : 100.0%

## Failure decomposition
  found             : 41
  misjudged         : 5
  truncated         : 1
  never_retrieved   : 3
  gold surfaced by   : bm25 (45)

## Cost
  llm calls / record   : 2.14
  tokens / record      : 1834.00
  seconds / record     : 1.91
  duplicate targets    : 2
```

The recommendation prints on one line; it is wrapped here to fit the page. A metric that
cannot be computed prints `n/a (nothing labelled)` rather than `0`, and a calibration
warning is appended when recorded confidences barely separate correct from incorrect — in
which case tuning `accept_at` is tuning nothing.

The three buckets are separate on purpose. **never retrieved** sends you to the `doc`
template, the retriever set or retrieval depth; **truncated** means the gold record *was*
retrieved and `selector.max_candidates` cut it before the model saw it; **misjudged** means
the model saw it and chose otherwise, which is a prompt problem. Collapsing truncation into
"retrieval failure" sends you to fix the wrong thing.

Always returns `0`. A source id absent from the gold file is unlabelled and excluded from
every metric — it is not a wrong answer.

## `compare`

Puts several completed runs side by side. This is how you actually choose a model.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--gold` | yes | — | the gold CSV, shared by every run |
| `--run` | yes | — | `LABEL=PATH`, repeatable |

```console
$ xwalk compare --gold gold.csv --run 'mini=runs/chebi' --run 'big=runs/chebi-4o'
  label                    precision  coverage   review   errors   calls   tokens    secs    misjudged
  -----------------------------------------------------------------------------------------------------
  mini                     0.927      0.820      0.120    0.000    2.140   1834.000  1.910   5
* big                      0.971      0.900      0.060    0.000    2.020   1901.000  3.440   3
```

The `*` marks the best accepted precision. `misjudged` sits beside precision deliberately:
precision alone cannot tell you whether a worse model judges worse or was handed a worse
candidate list. A `--run` argument without `=` prints
`--run expects LABEL=PATH, got 'runs/chebi'` and returns `2`. Writes nothing.

## `ablate`

Re-runs the job with one component disabled at a time and reports the delta in accepted
precision.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--gold` | yes | — | the gold CSV |
| `--out` | yes | — | output directory; indexes go to `<out>/index`, the report to `<out>/ablation.json` |

The standard set is: drop each retriever (only when more than one is configured — `Matcher`
rejects an empty retriever list, so on a single-retriever job that variant would abort the
whole ablation rather than score badly), `no_verifier` (`verify_band: null`), `no_retries`
(`max_attempts: 1`), and `half_budget` (halve `selector.max_candidates`).

```console
$ xwalk ablate --job examples/chebi/job.yaml --gold examples/chebi/sample/gold.csv \
    --out runs/chebi-ablation
  baseline         all components enabled             delta +0.000
  no_verifier      never buy a second opinion         delta -0.049
  no_retries       one attempt only; no reformulation delta -0.024
  half_budget      halve the selector budget          delta -0.012
```

`half_budget` is in the standard set because the ceiling decomposition separates budget
misses from retrieval misses — an ablation that confirms a diagnosis is worth more than one
that only measures a component.

This runs the whole source set once per variant, against a live provider, with no ledger and
no resume. Point it at a labelled subset. Returns `0`.

## `prompts draft`

Asks a model to write the domain slots for a job. Note the flag order: flags belong to the
leaf subparser, so `--job` comes *after* `draft`.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file; supplies the samples and the LLM |
| `--describe` | yes | — | one sentence describing the matching task |
| `--out` | yes | — | where to write the slots YAML |

The first 8 source records and first 8 target records go into the prompt. If the job's own
`prompts.slots` file already exists it is passed as "current slots, to refine rather than
replace", and the field-level diff is printed before writing.

```console
$ xwalk prompts draft --job examples/chebi/job.yaml \
    --describe 'chemical entity mentions in full-text articles to ChEBI terms' \
    --out examples/chebi/slots.yaml
warning: rubric was not in strictly decreasing score order; reordered
- domain_brief:
    before: "biomedical chemistry nomenclature"
    after:  "biomedical chemistry nomenclature, including salts and anomers"
wrote examples/chebi/slots.yaml
```

The model is asked for slots only, never prompt text, and the result goes through
`PromptSlots` validation and `validate_contract` before anything is written. Prints
`(no changes)` when the draft is identical. Writes only `--out` — pointing the job at the
new file is your job. Returns `0`.

## `prompts optimize`

Label-driven prompt optimisation over the three partitions.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--gold` | yes | — | the gold CSV; only labelled records are partitioned |
| `--out` | yes | — | work directory |
| `--role` | no | `selector` | `selector`, `scorer`, `rewriter` or `doc_template` |
| `--rounds` | no | `4` | maximum rounds |
| `--max-calls` | no | none | hard budget; refuses to start if the estimate exceeds it |

Everything else keeps `OptimizeConfig` defaults: `patience=2`, `failures_per_round=12`,
`objective="accepted_precision"`, `max_tokens=2048`, and the default 50/25/25 hash-based
partitioner. Progress goes to stdout as it happens.

```console
$ xwalk prompts optimize --job examples/chebi/job.yaml \
    --gold examples/chebi/sample/gold.csv --out runs/chebi-opt \
    --role selector --rounds 3 --max-calls 2000
partitions: {'prompt_train': 24, 'validation': 13, 'test': 13}
estimated LLM calls: ~277
baseline accepted_precision: 0.846
round 1: validation accepted_precision 0.923 (kept)
round 2: validation accepted_precision 0.923 (discarded)
round 3: validation accepted_precision 0.846 (discarded)
patience 2 exhausted
evaluating the selected prompt on the test partition (once)
stopped: patience 2 exhausted
test accepted precision: 0.9166666666666666
```

Writes `<out>/index/`, one `round_NN/` directory per usable round holding `slots.yaml` and
`validation.json`, `best/slots.yaml` — the file to copy over your job's slots — and
`report.json`. Returns `0`.

The job's own LLM plays both parts: it matches *and* it revises the slots. Use a strong
model as the optimiser and a cheap one for matching by calling `optimize_prompt` from
Python, which takes `optimiser_llm` separately.

## `review export`

Writes the review bucket to a CSV with identity columns filled and decision columns blank.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--run` | yes | — | run directory |
| `--out` | yes | — | CSV to write |

```console
$ xwalk review export --run runs/chebi --out runs/chebi/review.csv
exported 6 rows to runs/chebi/review.csv
```

Columns: `result_key`, `run_fingerprint`, `source_id`, `source_hash`,
`proposed_target_id`, `decision`, `corrected_target_id`, `reviewer`, `review_note`,
`reviewed_at`. Only `needs_review` results are exported. Fill in `decision` — one of
`accept`, `reject`, `replace`, `no_match`, `defer` — and `reviewer`; `replace` also needs
`corrected_target_id`. Returns `0`.

## `review apply`

Validates every reviewed row against the originating run, then appends it to the review
overlay. The model's own result is never overwritten.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--run` | yes | — | run directory holding the ledger |
| `--reviewed` | yes | — | the filled-in CSV |
| `--job` | yes | — | the same job file, used only to recompute the target-store fingerprint |

```console
$ xwalk review apply --run runs/chebi --reviewed runs/chebi/review.csv \
    --job examples/chebi/job.yaml
applied 6 decisions
```

Validation is all-or-nothing: one row whose `source_hash` or target snapshot no longer
matches fails the whole file with `error: exception: SnapshotMismatch: … re-run before
applying` and exit `3`, because a half-applied review file is worse than an unapplied one.
Rows with a blank `decision` are skipped silently; an unknown decision, a `replace` without
`corrected_target_id`, or a missing `reviewer` is a `ValueError` and exit `3`.

The handler returns `1` when the apply report lists rejected rows, but validation raises on
the first stale row instead of collecting rejections, so in practice this command exits `0`
or `3`.

Applying changes the adjudicated view, not the results table. Export it with
`xwalk export --view reviewed`; the three layers — model result, reviewer decision,
adjudicated view — are never collapsed.

## A complete session

```bash
#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY=sk-…
JOB=examples/chebi/job.yaml
GOLD=examples/chebi/sample/gold.csv

# 1. Preflight: strict job check, files, extras, credential presence. No model call.
xwalk validate --job "$JOB"

# 2. Match, with the index under the run directory (the default). Exit 1 means
#    "completed, and the review bucket is not empty" — a normal outcome, so it must not
#    kill a `set -e` script.
set +e
xwalk match --job "$JOB" --out runs/chebi --max-calls 2000
code=$?
set -e
[ "$code" -le 1 ] || exit "$code"

# 3. Score it. Reads the ledger, calls nothing, needs no credentials.
xwalk eval --run runs/chebi --gold "$GOLD" --out runs/chebi/eval

# 4. Adjudicate the review bucket.
xwalk review export --run runs/chebi --out runs/chebi/review.csv
$EDITOR runs/chebi/review.csv          # fill in decision + reviewer
xwalk review apply --run runs/chebi --reviewed runs/chebi/review.csv --job "$JOB"
xwalk export --run runs/chebi --view reviewed --out runs/chebi/reviewed.csv

# 5. Run a second configuration and compare. Each run needs its own directory (a
#    different fingerprint is refused). Its index directory is its own too.
xwalk match --job examples/chebi/job-4o.yaml --out runs/chebi-4o
xwalk eval  --run runs/chebi-4o --gold "$GOLD"
xwalk compare --gold "$GOLD" --run "mini=runs/chebi" --run "big=runs/chebi-4o"
```

Steps 3 to 5 cost nothing at the provider: evaluation and comparison read the ledger.

## Gotchas

**`match` returns `1` on a healthy run.** A non-empty review bucket is the system working —
uncertain rows routed to a human instead of guessed at — but it is still exit `1`. Under
`set -e` that kills the script between matching and evaluation, which is exactly the wrong
place to stop. Guard it, as the session above does, or use `xwalk match … || [ $? -eq 1 ]`.
Reserve failure handling for `2` and `3`.

**Flags live on the leaf subcommand.** `xwalk prompts --job job.yaml draft` is an
unrecognised-argument error; `xwalk prompts draft --job job.yaml` is right. argparse binds
an option to whichever parser is active when it is seen, so a flag typed before the
subcommand belongs to the group parser, which declares none. The same applies to `review`.

**`prompts` and `review` without a subcommand return `2`,** with
`prompts needs a subcommand: draft or optimize` on stderr.

**An invalid `--role` returns `2`.** It is not an argparse choice, but it is checked
before anything runs (it returned `3` in 0.1).

**`eval --out` is a path stem, and an existing suffix is replaced.** `--out runs/eval`
writes `runs/eval.json` and `runs/eval.txt`; `--out runs/eval.v1` writes `runs/eval.json`
and `runs/eval.txt`, silently dropping the `.v1`.

**Two configurations sharing `--index` refuse each other.** A compatible index is opened,
not rebuilt; one built for other targets, templates or encoder settings is refused rather
than reused or silently replaced. Use one index directory per configuration, or pass
`--rebuild-index` when replacing it is what you want.

**A run directory belongs to one fingerprint.** Any change to templates, slots, retrievers,
thresholds, model or generation parameters produces a different fingerprint, and `match`
refuses to put it in the same `--out`, listing what changed. That is intended: results
produced under different rules never mix, and a "small" job-file tweak is a new run in a
new directory.

**`--limit` takes the first N unfinished records in file order.** It is a smoke test, not a
sample; the metrics it produces are not representative of the collection. Repeating it
continues with the next N.

**`eval`, `compare` and `review` read `<run>/manifest.json` for the run fingerprint.** A run
directory without one — a run killed before it wrote one, or a hand-assembled ledger — fails with
`FileNotFoundError` and exit `3`. One ledger can hold several runs; the fingerprint is what
distinguishes them.

**`ablate` has no ledger and no resume.** It re-runs the whole source set once per variant.
Interrupt it and you have spent the money for nothing.

**`prompts optimize` uses one model for both roles.** There is no flag for a separate
optimiser model, and no way to override `patience`, `failures_per_round` or `objective` from
the CLI. Reach for `optimize_prompt` in Python when you need any of that.

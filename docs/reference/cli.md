# Command line

The SDK is primary. The CLI is a shell over it: every subcommand parses arguments, calls one
library function, prints, and returns an exit code. Nothing decision-shaped lives in
`xwalk/cli/main.py`, which is why anything the CLI does you can do from Python — and why the
things it does not expose (a separate optimiser model, custom gold normalisation, your own
retriever) are not missing features but calls you make yourself.

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
| `0` | success | the command did what it said |
| `1` | attention | the run completed, but something needs a human: a non-empty review bucket, or rejected review rows |
| `2` | usage | bad invocation: no command, an unknown command, a missing subcommand, a malformed `--run` |
| `3` | runtime | a real failure: missing key, unreadable file, stale review snapshot, unsupported platform |

Errors print as `error: <message>` on stderr, or `error: <ExceptionType>: <message>` for
anything unexpected. A CLI must not dump a traceback, so it does not.

## `index`

Builds every retriever index the job declares, and prints each one's fingerprint.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--out` | yes | — | index directory; each retriever gets `<out>/<name or kind>` |

Prints the target count, the retriever count, and one indented line per retriever with its
fingerprint — the value that feeds the run fingerprint, so it is the thing to compare when
two runs disagree about what was retrievable.

```console
$ xwalk index --job examples/chebi/job.yaml --out runs/chebi/index
indexed 200 target records into 1 retriever(s)
  bm25: 9ce1441b96ffef65
```

Writes a Tantivy index plus `xwalk_meta.json` for `bm25`, or `vectors.f32`,
`record_ids.json` and `xwalk_meta.json` for `dense`. Always returns `0` on success.

`match` builds its own indexes regardless, so `index` is not a required first step — it is
how you pay the indexing cost separately, confirm the target loader works, and read the
fingerprint before spending anything on a provider.

## `match`

The run. Loads the job, builds store, retrievers, client and matcher, then matches every
source record into a resumable run directory.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--job` | yes | — | path to the job file |
| `--out` | yes | — | run directory |
| `--index` | no | `<out>/index` | index directory |
| `--resume` | no | on | skip records already completed under this run fingerprint; `failed` records are retried |
| `--no-resume` | no | — | re-run everything; earlier results stay in the ledger's history |
| `--limit` | no | all | process only the first N unfinished source records, in file order; the rest are reported as `pending` |

Prints the number of current source records, the run state (`complete`, `partial`,
`failed`, `aborted`, `interrupted`), a count per status sorted by status name, the
pending count, tokens (with calls of unknown usage shown, never as zero), a
duplicate-target line when any target was chosen by more than one source record, and —
when the review bucket is non-empty — the command that exports it. Errors that stopped
the run go to stderr.

```console
$ export OPENAI_API_KEY=sk-…
$ xwalk match --job examples/chebi/job.yaml --out runs/chebi
matched 50 records into runs/chebi
  run state     : complete
  matched       : 41
  needs_review  : 6
  unmatched     : 3
  tokens        : 61234 in 112 calls
  duplicate targets: 2 (see manifest.json)

6 rows need review: xwalk review export --run runs/chebi
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
history (`Ledger.iter_history`, `xwalk.batch.export_history_jsonl`).

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
matches fails the whole file with `error: SnapshotMismatch: … re-run before applying` and
exit `3`, because a half-applied review file is worse than an unapplied one. Rows with a
blank `decision` are skipped silently; an unknown decision, a `replace` without
`corrected_target_id`, or a missing `reviewer` is a `ValueError` and exit `3`.

The handler returns `1` when the apply report lists rejected rows, but validation raises on
the first stale row instead of collecting rejections, so in practice this command exits `0`
or `3`.

Applying changes the adjudicated view, not the results table. Export it with
`export_mapping_csv(..., use_review=True)` from Python; the three layers — model result,
reviewer decision, adjudicated view — are never collapsed.

## A complete session

```bash
#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY=sk-…
JOB=examples/chebi/job.yaml
GOLD=examples/chebi/sample/gold.csv

# 1. Preflight: confirm the target loader works and read the retriever fingerprint.
#    This index is throwaway — `match` builds its own — so give it its own directory.
xwalk index --job "$JOB" --out runs/preflight-index

# 2. Match, with the index under the run directory (the default). Exit 1 means
#    "completed, and the review bucket is not empty" — a normal outcome, so it must not
#    kill a `set -e` script.
set +e
xwalk match --job "$JOB" --out runs/chebi
code=$?
set -e
[ "$code" -le 1 ] || exit "$code"

# 3. Score it. Reads the ledger, calls nothing, needs no credentials.
xwalk eval --run runs/chebi --gold "$GOLD" --out runs/chebi/eval

# 4. Adjudicate the review bucket.
xwalk review export --run runs/chebi --out runs/chebi/review.csv
$EDITOR runs/chebi/review.csv          # fill in decision + reviewer
xwalk review apply --run runs/chebi --reviewed runs/chebi/review.csv --job "$JOB"

# 5. Run a second configuration and compare. Each run needs its own directory; the index
#    can be shared, since building replaces an index directory rather than adding to it.
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

**An invalid `--role` returns `3`, not `2`.** It is not an argparse choice; the string is
handed to `PromptRole(...)`, and the resulting `ValueError` lands in the generic handler.

**`eval --out` is a path stem, and an existing suffix is replaced.** `--out runs/eval`
writes `runs/eval.json` and `runs/eval.txt`; `--out runs/eval.v1` writes `runs/eval.json`
and `runs/eval.txt`, silently dropping the `.v1`.

**`match` builds indexes every time.** `--index` chooses *where*, not *whether*. Pointing
two runs at one index directory adds a second copy of every document to the existing Tantivy
index while the retriever fingerprint — computed from the records, not the index — stays
identical, so nothing warns you. Use a fresh directory per configuration.

**Resume is keyed on the run fingerprint.** Any change to templates, slots, retrievers,
thresholds, model or generation parameters produces a different fingerprint, so `--resume`
re-runs every record rather than mixing results produced under different rules. That is
intended; it is also why a "small" job-file tweak can cost a full re-run.

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

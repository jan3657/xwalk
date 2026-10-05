# Concepts

What xwalk does, how a record actually flows through it, and why the parts that look
over-engineered are load-bearing. Read this once and the reference pages will make
sense; skip it and you will eventually be surprised by something on this page.

## The problem

You have two collections of records and you need to know which record in the first
corresponds to which record in the second. Mentions in text against an ontology. A
supplier catalogue against your own SKUs. Free-text disease names against MeSH.

Retrieval alone gets you a shortlist and no decision. An LLM alone cannot see a
collection that does not fit in a context window. xwalk is the loop between them:
retrieval proposes, the model decides, and a policy turns the model's confidence into
one of four outcomes you can act on.

## The four outcomes

Every source record ends in exactly one status. The distinctions between them are the
whole point.

| Status | Meaning | What you do |
|---|---|---|
| `matched` | Accepted automatically. | Ship it. |
| `needs_review` | A human should look at this. | Put it in the review queue. |
| `unmatched` | Positive evidence that nothing here corresponds. | Ship it as a non-match. |
| `failed` | The infrastructure broke. | Fix the infrastructure and re-run. |

Exports can also show `pending`: a source in the current collection that this run has
not processed yet (for example beyond `--limit`). It is not a result, and the next
invocation picks it up.

`unmatched` and `failed` are separate because a provider returning 500 is not evidence
about your data. Collapsing them means an outage silently becomes a batch of confident
"no such record" answers, and you will not notice until someone downstream asks why
coverage dropped.

## The loop

One source record, one call to `Matcher.match`:

```
render query + context from the source record
│
└─► for each attempt (up to policy.max_attempts):
    │
    ├─ 1. RETRIEVE   all retrievers concurrently, then fuse by reciprocal rank
    │                (skipped when re-examining a candidate already retrieved)
    │
    ├─ 2. SELECT     show the model a keyed candidate list; it answers C07 or null
    │                → LLM call
    │
    ├─ 3. SCORE      an independent confidence for that choice, judged against the
    │                whole source record, not the query          → LLM call
    │
    ├─ 4. VERIFY     only if the score lands in policy.verify_band → LLM call
    │
    └─ 5. ROUTE      the scorer may propose better candidates or better queries
    │
    ├─ good enough? (score ≥ accept_at, resolved exactly, verifier did not object)
    │     └─► stop
    └─ otherwise: drain the candidate queue, then the query queue, then ask the
          rewriter for a new query. Nothing new? stop.

derive_status(attempts) → one status, one reason, one winning attempt
```

The confident case costs two LLM calls: select and score. Everything else is what
happens when the confident case does not hold.

There is a second loop. A job that declares `decider:` instead of `llm:` runs
`DecisionMatcher` rather than `Matcher`: it retrieves hundreds of candidates instead of
twenty-five, asks a decision model one calibrated yes/no per candidate, chooses among the
survivors, and gates the choice on a rubric and on the identity-bearing properties your
slots declare. There is no retry loop, no verifier and no generated explanation, because
deeper retrieval replaces the first and arithmetic over probabilities replaces the rest.
Everything below this section — the four outcomes, opaque keys, failing toward review,
fingerprints, the ledger, review, evaluation — applies to both loops unchanged. See
[decision models](reference/decide.md).

### Why score separately from select

The selector already reports a confidence. xwalk ignores it for classification and asks
a second time, for two reasons.

The selector is answering "which of these is best", which is a comparative question. It
will happily return the best of five wrong answers with a high confidence, because it
*is* the best of them. The scorer is answering "is this actually correct", against the
full source record. Those are different questions and they get different answers.

The selector's confidence is not wasted: it is used as a well-formedness check. A choice
that arrives without a usable number is demoted to unresolved, because a choice you
cannot score is a choice you cannot classify.

### Why verify only sometimes

Verification is a third call, and paying it on every record roughly doubles your bill
for a second opinion you mostly do not need. `verify_band` buys it exactly where the
first opinion is uncertain — by default when the score lands between 0.6 and 0.8.

Note that `verify_band` is a *cost* control and `accept_at` / `review_floor` are
*classification*. They are separate knobs on purpose: you can widen the band without
changing what counts as a match.

`audit_rate` covers the blind spot. Scores above the band are never verified, so
nothing would catch a model that is confidently and consistently wrong. Setting
`audit_rate` samples a deterministic fraction of those for a second opinion anyway —
deterministic in `(run_fingerprint, source_id)`, so a resumed run audits the same
records rather than re-rolling.

## Opaque candidate keys

Candidates are shown to the model as `[C01]`, `[C02]`, … and the model must answer with
one of those keys or with `null`. Its answer is resolved by exact dictionary lookup
against the keys issued *for that attempt*. Anything else resolves to nothing.

This looks like an unnecessary indirection until you consider what the alternative does
when the model hallucinates.

Suppose target ids are numeric — NCBI gene ids, say — and the model returns `"3"`. A
resolver that accepts raw ids maps that to gene 3, a real record, and you have a
confident, plausible, completely wrong mapping in your output. A resolver that falls
back to interpreting a bare integer as a rank maps it to whatever happened to be third
in the list, which is also a real record and also wrong. Both failures are silent.

With opaque keys, `"3"` is not a key that was issued, so it resolves to nothing, the
attempt is recorded as unresolved, and the record goes to review. The invariant is
worth stating plainly:

> **A malformed model answer must never resolve to a real target record.**

It can fail to resolve. It cannot resolve to the wrong thing. Everything else in the
keying module exists to preserve that.

Opaque keys buy a second thing: `null` is unambiguous. In an id scheme where `-1` or `0`
is a legitimate value, there is no sentinel you can safely reserve for "no match".

Legacy id resolution exists (`MatchPolicy.legacy_id_resolution`) for reproducing prior
work that did accept raw ids. Every legacy resolution is tagged as such, and a result
resolved that way never becomes an automatic match regardless of its score — it routes
to review.

## Failure fails toward review

Every parse failure and every ambiguity in the pipeline resolves in the same direction:

- Selector output that will not parse → unresolved → review.
- Scorer output with no usable number → no score → review.
- Verifier output that will not parse, or an unrecognised verdict → treated as
  disagreement → review, even on a 0.99 score.
- A candidate key the model invented → dropped, with the reason recorded.

None of these can produce an automatic match. That is the one direction where being
wrong is expensive, because an automatic match is the one nobody looks at.

## Cardinality

Matching is many-to-one. Each source record gets at most one target, and several source
records may get the same target; `duplicate_targets()` (and a warning from `xwalk match`)
reports those, and both stay matched. There is no one-to-one assignment and no
one-to-many output. Clustering (experimental) is a different relation: it partitions
one collection into equivalence classes.

## Fingerprints and resume

A run has an identity. `build_run_fingerprint` digests everything whose change should
invalidate previously computed results: the templates, the prompts and slots, the target
collection snapshot, every retriever's index fingerprint, retrieval depth and the fusion
constant, the model and provider identity and generation parameters, the policy
thresholds, the selector budget, and the library version.

Each individual record's identity is `result_key(run_fingerprint, source_id,
hash_record(source))`. Resume skips a record when that key is already in the ledger with
a status other than `failed`; failed records are retried. So:

- Edit a template, change a threshold, or switch models → new fingerprint → everything
  re-runs.
- Fix a typo in one source row → that row's hash changes → that row re-runs, the rest
  do not. The exports keep one row for it: each invocation records the source
  collection as a snapshot, and exports show the result for each source's *current*
  content. The old result stays in the ledger's history.
- Rotate your API key → nothing changes. Credentials are deliberately excluded, along
  with output paths and concurrency, because none of them change what a result *means*.

The one sharp edge: `run_fingerprint` is a string you hand to `Matcher`, and it is never
cross-checked against the matcher you actually built. If you construct it by hand and
then change a component without rebuilding it, resume will happily reuse stale results.
Build it with `build_run_fingerprint` and pass the same objects you pass to `Matcher`.

## The ledger is the source of truth

`mapping.csv` is the deliverable, but `ledger.sqlite` is the run. Each result is
committed to SQLite in its own transaction the moment it finishes. A crash at record
9,000 of 10,000 costs you the records that were in flight. On a fatal provider error or
Ctrl-C, in-flight records are cancelled and awaited before the ledger is closed, and the
manifest records the run as `aborted` or `interrupted`.

The three files in a run directory — `mapping.csv`, `results.jsonl`, `manifest.json` —
are exports written at the end and regenerable from the ledger at any time. If you
delete `mapping.csv`, `export_mapping_csv` puts it back. If you delete the ledger, the
run is gone.

This is also why evaluation is free. `xwalk.evaluate` reads the ledger and calls
nothing, so scoring a run costs no tokens, gives the same answer every time, and works
on a laptop with no credentials.

## Review is an overlay, not an edit

`apply_review` never modifies a result. It appends the reviewer's decision to a separate
table, and the adjudicated view is derived from both. You can always ask what the model
said, what the human decided, and what the final answer is — those three stay distinct.

Applying a review file is all-or-nothing and snapshot-guarded. If a source record or the
target collection changed since the run, the whole file is refused rather than partly
applied, because a half-applied review is worse than an unapplied one: you no longer
know which rows are adjudicated.

## Evaluation tells you where to spend effort

A single accuracy number tells you how you are doing and nothing about what to change.
The report leads with a decomposition of every miss into one of three buckets:

| Bucket | What happened | What to fix |
|---|---|---|
| never retrieved | No retriever surfaced the gold record at all. | The `doc` template, the retriever set, or retrieval depth. |
| truncated | A retriever found it, but the selector budget cut it before the model saw it. | Raise `SelectorPolicy.max_candidates`. |
| misjudged | The gold record was on screen and the model chose otherwise. | The prompts. |

The middle bucket is why there are three and not two. A budget miss looks exactly like a
retrieval miss if you only check whether the gold record made the final candidate list —
and it sends you to rebuild your index when the actual fix is one integer.

Two more things the report will tell you that a headline number will not. `recall_at_any_status`
is the ceiling perfect threshold tuning could reach, so you can see whether your
thresholds are the problem or your pipeline is. And the calibration warning fires when
confidences on correct and incorrect matches barely differ — at which point tuning
`accept_at` is tuning noise, and the fix is the scoring prompt.

## The base install stays small

Matching two CSVs against an API-hosted model should not download PyTorch. Every heavy
dependency lives behind an extra, is imported lazily at the point of use, and raises
`MissingExtra` naming the extra and the install command if it is absent.

The one deliberate exception is Tantivy, which is a base dependency with no fallback.
Substituting a different BM25 implementation would change ranking, and ranking is part
of a run's identity — a silent engine swap would quietly invalidate reproducibility
while appearing to work. If Tantivy will not install on your platform, you choose an
alternative explicitly. See [platforms](platforms.md).

## Where to go next

- [Getting started](guide/getting-started.md) — a first run, end to end.
- [What each component does](components.md) — the module-by-module map.
- [Evaluation](guide/evaluation.md) — measuring a run and acting on the result.

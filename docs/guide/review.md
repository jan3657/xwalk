# Human review

Some records need a person. This guide covers getting them to one, getting the decisions
back, and keeping the result honest afterwards.

## The model of review

xwalk keeps three layers and never collapses them:

1. **What the model said** — the `MatchResult` in the ledger. Review never touches it.
2. **What the reviewer decided** — an append-only overlay in a separate table.
3. **The adjudicated view** — derived from both, on demand.

You can always ask any of those three questions afterwards. That matters more than it
sounds: when someone asks in six months whether a mapping was machine-produced or
human-confirmed, "the file says CHEBI:17234" is not an answer.

## Export the queue

```python
from xwalk.ledger import Ledger
from xwalk.review import export_review

ledger = Ledger.open("run/ledger.sqlite")
count = export_review(ledger, run_fingerprint, "review.csv")
```

```bash
xwalk review export --run run/ --out review.csv
```

By default this exports records with status `needs_review`. To spot-check accepted
matches too — worth doing at least once on a new domain — pass the statuses explicitly:

```python
from xwalk.records import MatchStatus

export_review(
    ledger, run_fingerprint, "audit.csv",
    statuses=(MatchStatus.NEEDS_REVIEW, MatchStatus.MATCHED),
)
```

## The review file

```csv
result_key,run_fingerprint,source_id,source_hash,proposed_target_id,decision,corrected_target_id,reviewer,review_note,reviewed_at
a1b2c3...,f9e8d7...,s7,4a5b6c...,CHEBI:17992,,,,,
```

The first five columns identify the record and are filled in for you. Do not edit them —
they are how the decision is bound to a specific result produced under a specific
configuration. The last five are yours.

| Column | What to put |
|---|---|
| `decision` | One of the five values below. Leave blank for rows you did not get to. |
| `corrected_target_id` | The right target id. Required for `replace`, ignored otherwise. |
| `reviewer` | Who decided. Required whenever `decision` is filled. |
| `review_note` | Free text. Optional, and worth writing for anything non-obvious. |
| `reviewed_at` | A timestamp. Free-form string. |

### The five decisions

| Decision | Final target | Final status | When |
|---|---|---|---|
| `accept` | the model's proposal | `matched` | The model was right. |
| `reject` | none | `unmatched` | The model was wrong and nothing here is correct. |
| `replace` | `corrected_target_id` | `matched` | The model was wrong and you know the right answer. |
| `no_match` | none | `unmatched` | Same effect as `reject`; use it when the record genuinely has no counterpart, as opposed to the model merely picking badly. |
| `defer` | the model's proposal | `needs_review` | You could not decide. Stays in the queue. |

`reject` and `no_match` are identical in effect and distinct in meaning. The distinction
is for you and anyone reading the overlay later; xwalk does not treat them differently.

A blank `decision` is skipped silently — a half-finished review file is a normal thing
to apply.

## Apply the decisions

```python
from xwalk.review import apply_review, read_review

report = apply_review(
    ledger,
    read_review("review.csv"),
    target_store_fingerprint=store.fingerprint,
)
print(f"applied {report.applied} decisions")
```

```bash
xwalk review apply --run run/ --reviewed review.csv --job job.yaml
```

### What gets refused

`read_review` rejects the file before anything is applied if:

- a `decision` value is not one of the five (the error lists the legal ones);
- `replace` has no `corrected_target_id`;
- a filled `decision` has no `reviewer`.

`apply_review` then validates every row against the ledger and raises `SnapshotMismatch`
if:

- the `result_key` is not in this ledger;
- the row's `source_hash` no longer matches the stored result — the source record
  changed since the run, so the decision was made about different data;
- the result is no longer current — the source record was edited or removed and the
  run resumed since the export, or the result was recomputed (`--no-resume`) and now
  proposes a different target;
- the target collection fingerprint no longer matches the one recorded in the manifest.

Validation is all-or-nothing: one stale row refuses the whole file. This is deliberate.
A partly-applied review file leaves you unable to say which rows are adjudicated, which
is strictly worse than having applied none of it. Re-run, re-export, and re-review.

## Read the adjudicated view

```python
from xwalk.review import adjudicated

for row in adjudicated(ledger, run_fingerprint):
    if row.reviewer:
        print(row.source_id, row.model_target_id, "->", row.final_target_id, f"({row.reviewer})")
```

Unreviewed records appear here too, with their final fields equal to the model's and
`reviewer` set to `None`. Reviewed records keep `model_target_id` and `model_status`
alongside `final_target_id` and `final_status` — that is the "never collapsed" part.

The overlay is append-only. If the same record is reviewed twice, the latest decision
wins in this view and both remain in the ledger via `Ledger.iter_reviews`.

A decision is bound to the exact result it was made against. Resuming an unchanged run
keeps it. When the source record is edited or removed, or the result is recomputed, the
decision no longer applies to the adjudicated view: the record needs a new review.
Nothing is deleted; `xwalk.review.review_history(ledger, run_fingerprint)` lists every
decision with its state, `applied`, `superseded` (a later decision replaced it) or
`stale` (its result is no longer current).

## Export the adjudicated mapping

```python
from xwalk.batch import export_mapping_csv

export_mapping_csv(ledger, run_fingerprint, "run/adjudicated.csv", use_review=True)
```

Write it *alongside* `mapping.csv`, not over it. Two files — what the model produced and
what was agreed — is the arrangement that lets you answer questions later. The
adjudicated export leaves the per-run cost columns empty, since a human decision has no
token count.

## A note on the review rate

`review_rate` from the [evaluation report](evaluation.md) is your ongoing human cost. If
it is higher than you can staff, the levers in rough order of effect are:

1. **Fix the prompts** if the ceiling decomposition says *misjudged* — records in review
   because the model was uncertain about a correct answer are the cheapest to eliminate.
2. **Lower `accept_at`**, but only after checking the threshold curve, and only if the
   calibration warning is not firing.
3. **Narrow `verify_band`**, which reduces verifier disagreements routing records to
   review — at the cost of the second opinion that caught them.

Raising `accept_at` to reduce review does not exist as an option; it moves records the
other way.

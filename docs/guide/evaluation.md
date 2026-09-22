# Measuring a run

You have a run directory. This guide turns it into a decision about what to change next.

Nothing here calls an LLM or a retriever. Evaluation reads `ledger.sqlite`, which means
scoring a run costs nothing, gives the same answer every time, and works on a machine
with no credentials.

## Gold labels

A gold file is a CSV with two columns:

```csv
source_id,gold_ids
s1,CHEBI:17234
s2,CHEBI:17234
s3,CHEBI:17992|CHEBI:15903
s4,
```

Multiple acceptable answers go in one cell separated by `|`. Then there is row `s4`,
which is the part people get wrong:

> An **empty** `gold_ids` cell means "the correct answer is no match". A source id
> **absent from the file entirely** is unlabelled and excluded from every metric.

These are different claims and xwalk keeps them different. If you drop your no-match
rows instead of labelling them empty, every correct abstention disappears from the
denominator and your no-match recall is inflated by exactly the cases you got right.

```python
from xwalk.evaluate import load_gold_csv

gold = load_gold_csv("gold.csv")
```

### Normalising and expanding ids

Your gold file and your target collection may not use the same identifier spelling, and
some domains have alias sets where several ids are equally correct. Both are hooks you
supply — the library never learns your identifier scheme:

```python
gold = load_gold_csv(
    "gold.csv",
    normalize=lambda i: f"NCBIGene:{i}" if i.isdigit() else i,
    expand=lambda ids: frozenset().union(*(alias_map.get(i, {i}) for i in ids)),
)
```

`expand` is never called for a no-match label, so alias expansion cannot accidentally
turn a deliberate empty cell into a non-empty one.

## Scoring

```python
from xwalk.evaluate import evaluate, render_report
from xwalk.ledger import Ledger

ledger = Ledger.open("run/ledger.sqlite")
try:
    report = evaluate(ledger, run_fingerprint, gold)
finally:
    ledger.close()

print(render_report(report))
```

The run fingerprint is in `report.run_fingerprint` from the batch run, or in
`run/manifest.json` under `run_fingerprint`. From the shell it is simply:

```bash
xwalk eval --run run/ --gold gold.csv
```

## Reading the report

### Start at the top: where to spend effort

```
## Where to spend effort
  62% of failures presented the gold record and the model chose otherwise --
  spend effort on prompts, not retrieval
```

This is first on purpose. Every miss is classified into one of three buckets, and each
one sends you somewhere different:

| Bucket | What happened | What to change |
|---|---|---|
| `never_retrieved` | No retriever surfaced the gold record on any attempt. | The `doc` template, the retriever set, or retrieval depth (`limit`). |
| `truncated` | A retriever found it, but the selector budget cut it before the model saw it. | `SelectorPolicy.max_candidates`. |
| `misjudged` | The gold record was on screen and the model picked something else. | The prompts — the rubric and hard rules. |

The middle bucket is the reason there are three and not two. A budget miss and a
retrieval miss look identical if you only ask "was the gold record in the final
candidate list" — and mistaking one for the other sends you to rebuild an index when the
fix was raising one integer.

### Then the headline numbers

| Metric | What it means |
|---|---|
| `accepted_precision` | Of the records xwalk matched automatically, the fraction that were right. This is the number that determines whether you can trust the output unreviewed. |
| `automatic_coverage` | The fraction of labelled records matched automatically. Precision without coverage is a matcher that abstains on everything. |
| `review_rate` | The fraction sent to a human. This is your ongoing cost. |
| `recall_at_any_status` | The fraction where the correct target was chosen at *any* status. This is the ceiling perfect threshold tuning could reach. |
| `no_match_precision` / `no_match_recall` | How well abstention works, scored only against rows explicitly labelled as no-match. |
| `unresolved_rate` | Model output that could not be resolved to a record. Reported separately from the review rate because the fix is different. |
| `mean_llm_calls` / `mean_tokens` / `mean_cost_usd` / `mean_seconds` | Cost and latency per record. |
| `duplicate_target_conflicts` | Targets chosen by more than one source record. |

Every ratio with a zero denominator is `None`, never `0.0`, and renders as `n/a` with a
reason. A precision of `n/a (nothing matched)` and a precision of `0.0` are very
different situations, and collapsing them hides the first one.

The gap between `accepted_precision` and `recall_at_any_status` is where threshold
tuning lives. If recall is 0.90 and accepted precision is 0.70, your pipeline is finding
the right answers and your thresholds are misclassifying them. If recall is 0.55, no
threshold will save you.

### Then the calibration warning

```
## Calibration warning
  confidence scores barely separate correct from incorrect matches (mean correct
  0.82 vs mean incorrect 0.78, separation 0.04 < 0.1); threshold tuning will not
  help until the scoring prompt discriminates better
```

If this fires, stop tuning `accept_at`. The confidences do not carry the signal you
would be tuning against, so every threshold you try is sampling noise. The fix is the
scoring rubric: the bands need to be distinguishable in a way the model can actually
apply.

## Choosing thresholds

The report includes a threshold curve re-derived from the confidences already recorded,
so you can see what each `accept_at` would have produced without re-running anything:

```python
for point in report.thresholds:
    print(f"{point.threshold:.2f}  coverage {point.coverage:.3f}  precision {point.precision}")
```

Pick the threshold that meets your precision requirement, then set `accept_at` and
re-run. Note that the re-run changes the fingerprint, so it is a full re-run — the curve
is there so you only have to do that once.

## Comparing runs

This is how you actually choose a model. Run the same job twice with different `llm`
settings, then:

```python
from xwalk.evaluate import compare_runs, summarise_run

summaries = [
    summarise_run(ledger_a, fp_a, gold, label="mini"),
    summarise_run(ledger_b, fp_b, gold, label="big"),
]
print(compare_runs(summaries))
```

```bash
xwalk compare --gold gold.csv --run 'mini=runs/first' --run 'big=runs/second'
```

The table puts precision next to cost next to the misjudged count. That last column
matters: precision alone cannot tell you whether the worse model is worse at *judging*
or was simply handed a worse candidate list, and those have different fixes.

## Ablations

To find out what a component is actually contributing, disable it and measure:

```bash
xwalk ablate --job job.yaml --gold gold.csv --out runs/ablation
```

```
  baseline         all components enabled             delta +0.000
  no_verifier      never buy a second opinion         delta -0.041
  no_retries       one attempt only; no reformulation delta -0.012
  half_budget      halve the selector budget          delta -0.003
```

The standard set drops each retriever (only when more than one is configured — dropping
your only retriever leaves no pipeline to measure), disables the verifier, disables
retries, and halves the selector budget.

`half_budget` is in the standard set specifically so an ablation can confirm what the
ceiling decomposition told you. If your report says truncation is not a problem and
halving the budget costs you nothing, those two agree and you can stop thinking about
it.

Unlike everything else in `xwalk.evaluate`, `ablate` re-runs the matcher and therefore
costs LLM calls. Run it on a subset first.

## Partitions

If you are going to tune anything against these labels — thresholds, prompts, templates
— split them first, or you will tune against your own test set:

```python
from xwalk.evaluate import Partitioner

partitioner = Partitioner()          # 50% prompt-train, 25% validation, 25% test
report = evaluate(ledger, fp, gold, partitioner=partitioner)
print(report.partition_sizes)
```

Assignment is a salted hash of the source id, not a random shuffle. That means it is
reproducible without storing anything, and adding labelled records later never moves the
records already assigned. To reproduce a published split exactly, write the assignment
to a file and pass it as `partition_overrides`.

## Where to go next

- [Prompt optimisation](prompt-optimisation.md) — if the decomposition says *misjudged*.
- [Review](review.md) — for the records that landed in the review bucket.
- [Evaluation reference](../reference/evaluation.md) — every function and every field.

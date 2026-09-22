# Evaluation and review

`xwalk.evaluate` answers "is this run good, and where should I spend effort?";
`xwalk.review` layers human decisions on top of model output without ever overwriting
it. Nothing here calls an LLM or a retriever — everything reads the ledger a completed
run left behind, so evaluation is free, repeatable, and safe on a machine with no
credentials. The one deliberate exception is [`ablate`](#ablation), which re-runs the
matcher on variant configurations and therefore spends real tokens.

## Gold labels

Two things that look alike and are not:

- an **empty** label — the row is present, the id cell is blank — means "the correct
  answer is no match". That is a first-class outcome, counted by every metric.
- an **absent** source id — no row at all — means unlabelled, and is excluded from
  every metric.

Conflating them silently inflates no-match recall: every unlabelled record the matcher
declined to match would count as a success.

### File formats

CSV: a header row naming the two columns, one row per labelled source record, multiple
gold ids joined by the separator. An empty `gold_ids` cell is the no-match label.

```csv
source_id,gold_ids
s1,T1|T3
s2,
```

JSONL: one object per line; the gold field may be a list of ids or a single string.
`{"source_id": "s2", "gold_ids": []}` is the no-match label. Blank lines are skipped;
invalid JSON raises `ValueError` with the line number.

Both loaders raise `ValueError` on a blank source id or a duplicate one — a gold file
that disagrees with itself should fail loudly, not last-row-wins.

```python
load_gold_csv(
    path: str | Path,
    *,
    source_column: str = "source_id",
    gold_column: str = "gold_ids",
    separator: str = "|",
    encoding: str = "utf-8",
    normalize: Normalizer | None = None,
    expand: Expander | None = None,
) -> GoldSet

load_gold_jsonl(
    path: str | Path,
    *,
    source_field: str = "source_id",
    gold_field: str = "gold_ids",
    encoding: str = "utf-8",
    normalize: Normalizer | None = None,
    expand: Expander | None = None,
) -> GoldSet
```

`Normalizer = Callable[[str], str | None]` maps each raw id to a canonical form;
returning `None` (or `""`) drops the id. `Expander = Callable[[frozenset[str]],
AbstractSet[str]]` maps the normalized set to the full alias set. These are
user-supplied callables: the paper repo owned them as named hooks inside the library,
here they live in your code and the library never learns your identifier scheme.
Expansion is skipped for empty sets — a no-match label needs no aliases. Note that a
normalizer which drops every id in a row turns that row into a no-match label.

### `GoldSet`

A frozen wrapper around `labels: Mapping[str, frozenset[str]]`:

| Member | Behaviour |
|---|---|
| `get(source_id)` | The label set, or `None` when unlabelled |
| `source_id in gold` | Labelled at all (including as no-match) |
| `len(gold)`, `iter(gold)` | Count of and iteration over labelled ids |
| `no_match_ids` | Property: the ids whose correct answer is explicitly no match |
| `is_correct(source_id, predicted)` | `None` when unlabelled; a `None` prediction is correct iff the label is empty; otherwise membership of `predicted` in the label set |

## Metrics

Accuracy alone hides the decisions that matter: how much was matched *automatically*,
what precision that automation bought, and how much human time the review bucket
costs. A run at 85% accuracy with a 60% review rate is a worse product than one at 80%
with a 5% review rate, and one accuracy number cannot tell you which you have.

`evaluate_results(results, gold)` returns an `EvalReport`;
`evaluate_run(ledger, run_fingerprint, gold)` reads the results from the ledger first.
Every ratio is computed over **labelled** results only, and an undefined ratio is
`None`, never `0` — "nothing was matched" and "everything matched was wrong" must not
render identically.

| Attribute | Definition | How to read it |
|---|---|---|
| `total` | All results in the run, labelled or not | Run size |
| `labelled` | Results whose source id is in the gold set | The denominator for every rate below |
| `accepted_precision` | Of MATCHED results, the fraction whose `matched_id` is in the gold set | What the automatic output is worth. `None` if nothing matched |
| `automatic_coverage` | MATCHED / labelled | How much needed no human at all |
| `review_rate` | NEEDS_REVIEW / labelled | The human bill |
| `unmatched_rate` | UNMATCHED / labelled | Confident abstentions |
| `error_rate` | FAILED / labelled | Infrastructure failures, not model judgement |
| `unresolved_rate` | Results with reason `UNRESOLVED_OUTPUT` / labelled | Model output that would not parse — a prompt-contract problem |
| `recall_at_any_status` | Of labelled results with a non-empty gold set, the fraction whose `matched_id` is correct at *any* status | The ceiling perfect threshold tuning could reach |
| `no_match_precision` | Of results predicted no-match (`matched_id is None`, not FAILED), the fraction whose gold set is empty | Whether abstention is trustworthy |
| `no_match_recall` | Of results whose gold set is empty, the fraction predicted no-match | Whether true negatives are being found |
| `duplicate_target_conflicts` | Target ids selected by more than one source record (an `int`, over **all** results) | Possible many-to-one collisions worth auditing |
| `mean_llm_calls` | Mean `usage.calls` per completed (non-FAILED) result, over all results | Cost per record |
| `mean_tokens` | Mean `usage.total_tokens` per completed result | Cost per record |
| `mean_cost_usd` | Mean `usage.cost_usd` per completed result, in USD | Cost per record |
| `mean_seconds` | Mean `elapsed_seconds` per completed result | Latency per record |
| `status_counts` | `{status value: count}` over all results | Raw distribution |
| `reason_counts` | `{reason value: count}` over all results | Raw distribution |
| `calibration_warning` | The warning string, or `None` | See below |

`EvalReport.as_dict()` returns the same fields as a JSON-ready dict.

### Threshold curve

```python
def threshold_curve(
    results: Iterable[MatchResult], gold: GoldSet, *, steps: int = 21
) -> list[ThresholdPoint]
```

What precision and coverage would be if `accept_at` were set differently. Re-derived
from recorded confidences, so it costs nothing and needs no re-run. Thresholds are
`i / (steps - 1)` for `i` in `0..steps-1`; a labelled result contributes when it has
both a confidence and a `matched_id`, at any status.

`ThresholdPoint` fields: `threshold`, `coverage` (accepted / all labelled — `0.0` when
nothing is labelled), `precision` (correct / accepted, `None` when nothing clears the
threshold), `accepted`, `correct`.

### Calibration warning

```python
calibration_warning(results, gold) -> str | None
```

Flags confidences that do not separate correct from incorrect matches. It compares the
mean confidence of correct predictions against incorrect ones and warns when the
separation is below **0.10**. It needs at least **3 samples of each class** — two right
and one wrong is not evidence — and returns `None` (silence, not a clean bill) below
that. When the warning fires, every threshold recommendation derived from these
confidences is noise, and a user tuning `accept_at` is tuning nothing; fix the scoring
prompt first.

## Failure decomposition

Three failure buckets, never two. The selector budget creates a genuinely distinct
failure: if the gold record was retrieved but cut before the model saw it, the fix is a
larger budget — not a better retriever and not a better prompt. Collapsing that into
"retrieval failure" sends users to fix the wrong thing.

This is nearly free: the loop already records `evidence` on every candidate and
`issued_keys` on every attempt, so decomposition reads a completed run and calls
nothing.

### `CeilingBucket`

| Member | Meaning | Fix |
|---|---|---|
| `FOUND` | The chosen id is in the gold set | Nothing |
| `MISJUDGED` | A gold record was presented to the model and it chose otherwise | Prompts, not retrieval |
| `TRUNCATED` | A gold record was retrieved but the selector budget cut it before the model saw it | Raise `SelectorPolicy.max_candidates` |
| `NEVER_RETRIEVED` | No gold record ever appeared in any attempt's candidates | The `doc` template, the retriever set, or retrieval depth |
| `NO_GOLD` | The correct answer is no match — not a ceiling question | — |
| `UNLABELLED` | Excluded from every total | — |

### `classify_ceiling(result, gold) -> CeilingBucket`

Precedence matters: presented beats retrieved beats absent. The checks run in order —
unlabelled, then empty gold set, then `matched_id` in the gold set (`FOUND`), then gold
∩ ids actually issued to the model (`MISJUDGED`), then gold ∩ ids retrieved in any
attempt (`TRUNCATED`), else `NEVER_RETRIEVED`. So a record that reached the model is
never reported as a budget miss, and a record the budget cut is never reported as a
retrieval miss.

### `ceiling_report(results, gold) -> CeilingReport`

| Field | Meaning |
|---|---|
| `evaluable` | Results in a ceiling bucket — labelled, non-empty gold set |
| `buckets` | `{CeilingBucket: count}` including `NO_GOLD` and `UNLABELLED` |
| `by_retriever` | For evaluable results, how many times each retriever's evidence surfaced a gold candidate (once per result) |
| `by_attempt` | Index of the first attempt in which a gold candidate appeared, counted |
| `recommendation` | One sentence naming the plurality failure bucket, its share of failures, and the fix — or "no ceiling failures on labelled records" |

## Reports

```python
evaluate(
    ledger: Ledger,
    run_fingerprint: str,
    gold: GoldSet,
    *,
    partitioner: Partitioner | None = None,
    partition_overrides: Mapping[str, Partition] | None = None,
    threshold_steps: int = 21,
) -> FullReport
```

One call, one printable summary. `FullReport` bundles `metrics: EvalReport`,
`ceiling: CeilingReport`, `thresholds: tuple[ThresholdPoint, ...]`, and
`partition_sizes: Mapping[str, int]` (labelled results per partition; empty when no
partitioner is given), plus `as_dict()`.

`render_report(report) -> str` emits sections in this order, with the ceiling
recommendation first — "where should I spend effort?" is the question this phase
exists to answer, and it must not be buried:

1. `## Where to spend effort` — the recommendation
2. `## Headline` — precision, coverage, review/unmatched/error/unresolved rates,
   recall at any status
3. `## No-match handling` — no-match precision and recall
4. `## Failure decomposition` — found / misjudged / truncated / never_retrieved
   counts, and which retrievers surfaced gold
5. `## Cost` — model calls, tokens, cost, seconds per record, duplicate targets
6. `## Partitions` — only when a partitioner was given
7. `## Calibration warning` — only when it fired

`None` metrics render as `n/a` with the reason (`n/a (nothing matched)`), never as 0.

`write_report(report, path_stem)` writes `<stem>.json` (via `as_dict`) and `<stem>.txt`
(via `render_report`), creating parent directories.

## Partitions

Three partitions with distinct roles. Revising a prompt from dev failures *and*
selecting the retained round on dev performance makes dev part of training. Hence:

| `Partition` | Value | Role |
|---|---|---|
| `PROMPT_TRAIN` | `"prompt_train"` | Failures shown to the optimising model |
| `VALIDATION` | `"validation"` | Chooses the retained round and the stopping point |
| `TEST` | `"test"` | Evaluated once, after the final prompt is selected |

```python
Partitioner(
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    salt: str = "xwalk",
)
```

Fractions must sum to 1 (within 1e-9) and all be positive — an empty partition
disables its role. `assign(source_id)` hashes `salt + "\x00" + source_id` with SHA-256
and maps the first 8 bytes onto `[0, 1)`, so assignment is deterministic: adding
labelled records never reshuffles the existing split. A test partition that moves
between runs is not a test partition. The flip side: changing `salt` or `fractions`
reshuffles everything.

`split(source_ids)` returns `{Partition: sorted list of ids}`.

File round-trip, for reproducing a published split:

- `write_partition_file(assignment: Mapping[str, Partition], path)` — a two-column CSV
  (`source_id,partition`), sorted by id.
- `load_partition_file(path) -> dict[str, Partition]` — blank ids are skipped; an
  unknown partition name raises `ValueError` with the row number.
- `partition_of(source_id, partitioner, overrides=None)` — an explicit override always
  wins over hashing.
- `ids_in(source_ids, partition, partitioner, overrides=None)` — the sorted, unique
  subset assigned to one partition.

## Comparing runs

This is how you actually "select an LLM". `summarise_run(ledger, run_fingerprint,
gold, *, label)` reduces one run to a `RunSummary`:

| Field | Source |
|---|---|
| `label` | The label you passed |
| `run_fingerprint` | The run |
| `labelled` | `EvalReport.labelled` |
| `accepted_precision`, `automatic_coverage`, `review_rate`, `error_rate` | `EvalReport` |
| `mean_llm_calls`, `mean_tokens`, `mean_seconds` | `EvalReport` |
| `misjudged` | The `MISJUDGED` count from the ceiling report — a count, not a rate |

`misjudged` sits beside precision on purpose: precision alone cannot tell you whether
a worse model is worse at judging or was simply handed a worse candidate list.

`compare_runs(summaries) -> str` renders a fixed-width table, marking the row with the
highest `accepted_precision` with `*`; an empty sequence renders "no runs to compare".
`compare_runs_dict(summaries)` returns the same rows as dicts for JSON.

## Ablation

Flip one flag, report the delta. The standard set includes halving the selector budget
precisely because the ceiling decomposition separates budget misses from retrieval
misses — an ablation that confirms a diagnosis is worth more than one that only
measures a component.

```python
@dataclass(frozen=True)
class MatcherConfig:
    name: str
    retriever_names: tuple[str, ...]
    policy: MatchPolicy
    selector_policy: SelectorPolicy

@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    apply: Callable[[MatcherConfig], MatcherConfig]
```

`standard_ablations(retriever_names)` produces:

| Variant | What changes |
|---|---|
| `no_<retriever>` (one per retriever) | Drops that retriever from `retriever_names` |
| `no_verifier` | `policy.verify_band = None` — never buy a second opinion |
| `no_retries` | `policy.max_attempts = 1` — one attempt, no reformulation |
| `half_budget` | `selector_policy.max_candidates` halved (floor 1) |

The retriever-drop variants are generated **only when more than one retriever is
configured**. "What did bm25 add?" is not a question a run with no retrieval at all can
answer, and `Matcher` rejects an empty retriever list — so on a single-retriever job
the variant would not report a bad score, it would abort the whole ablation before any
variant reported.

```python
async def ablate(
    matcher_factory: Callable[[MatcherConfig], Matcher],
    base: MatcherConfig,
    records: Sequence[Record],
    gold: GoldSet,
    *,
    ablations: Sequence[Ablation] | None = None,
    objective: str = "accepted_precision",
    out: str | Path | None = None,
) -> AblationReport
```

This is the one function in the package that runs the matcher — every variant matches
every record, so it costs real LLM calls. The baseline runs first, then each variant.
The result is an `AblationReport(rows, objective)` where `rows` is a tuple of
`AblationRow` (`name`, `description`, `accepted_precision`, `automatic_coverage`,
`review_rate`, `mean_llm_calls`, `delta`), baseline first with `delta=0.0`. `delta` is
the variant's objective minus the baseline's; when an ablation destroys the objective
entirely (the variant's value is `None` but the baseline had one), `delta` is the full
`-baseline` — reporting `0.0` would read as "no effect", the opposite of what
happened. `out` writes the report as JSON.

## Failure selection for optimisation

Which failures to show the optimising model depends on which prompt is being
optimised. "Only selection failures feed the optimiser" is correct for the selector
and wrong for everything else: a selector-optimisation prompt shown a case where the
gold record was never retrieved teaches nothing — the model never saw the right
answer. That argument does *not* extend to the scorer. Rejecting a slate that contains
nothing correct is precisely the scorer's job, so a confident accept in that situation
is its canonical failure and must reach the optimiser.

```python
select_failures(
    results: Sequence[MatchResult],
    gold: GoldSet,
    role: PromptRole,
    *,
    limit: int | None = None,
) -> list[FailureCase]
```

Unlabelled and no-match-gold results are skipped for every role. The per-role rules:

| `PromptRole` | A result is kept when |
|---|---|
| `SELECTOR` | Its ceiling bucket is `MISJUDGED` — gold was presented and something else was chosen |
| `SCORER` | Any of: a wrong accept (MATCHED but incorrect); a needless abstention (gold was presented, nothing chosen, reason `SELECTOR_ABSTAINED` or `BELOW_REVIEW_FLOOR`); a suppressed correct answer (correct id chosen but status UNMATCHED or NEEDS_REVIEW) |
| `REWRITER` | The gold record was not among the **first** attempt's candidates — whether a later query recovered it or not, both are its job |
| `DOC_TEMPLATE` | Its ceiling bucket is `NEVER_RETRIEVED` |

Cases are sorted by `source_id` so the selection is deterministic under a `limit`.

`FailureCase` fields: `source_id`, `source_fields` (empty here — populated by the
optimizer, which has the source records), `gold_ids` (sorted tuple), `chosen_id`,
`confidence`, `status`, `reason` (string values), `bucket`, `presented` (record ids
shown to the model, first-seen order), `queries` (one per attempt), `explanation`.

`render_failure(case) -> str` is a compact, prose-safe rendering for inclusion in an
optimisation prompt: the source fields, the correct answer, what the model chose and
why, and which candidates and queries were involved.

## Human review

`xwalk.review` treats review as an immutable overlay. Three layers are preserved and
never collapsed: the model result (in `results`), the reviewer decision (in
`reviews`), and the adjudicated view derived from both. Applying a review appends;
nothing edits a model result, and a later review of the same result simply supersedes
the earlier one in the derived view.

### The review CSV

`export_review` writes one row per result with the identity columns filled and the
decision columns blank:

```text
result_key, run_fingerprint, source_id, source_hash,
proposed_target_id, decision, corrected_target_id,
reviewer, review_note, reviewed_at
```

A reviewer fills in `decision`, `reviewer`, and (for `replace`) `corrected_target_id`.
Rows left with a blank decision are simply skipped on read — a half-finished file is
usable.

### Decisions

| `decision` | Effect on the adjudicated view |
|---|---|
| `accept` | Final target = the model's proposal; final status `MATCHED` |
| `reject` | Final target = none; final status `UNMATCHED` |
| `replace` | Final target = `corrected_target_id`; final status `MATCHED` |
| `no_match` | Final target = none; final status `UNMATCHED` — same effect as `reject`; the stored decision preserves that the reviewer asserted "no correct target exists" rather than "this proposal is wrong" |
| `defer` | Final target = the model's proposal; final status stays `NEEDS_REVIEW` — the row remains in the queue |

### Functions

```python
export_review(
    ledger: Ledger,
    run_fingerprint: str,
    out_path: str | Path,
    *,
    statuses: Sequence[MatchStatus] = (MatchStatus.NEEDS_REVIEW,),
) -> int                       # rows written

read_review(path: str | Path) -> list[ReviewRow]

apply_review(
    ledger: Ledger,
    rows: Sequence[ReviewRow],
    *,
    target_store_fingerprint: str,
) -> ApplyReport               # ApplyReport(applied: int, rejected: list)

adjudicated(ledger: Ledger, run_fingerprint: str) -> Iterator[AdjudicatedResult]
```

`read_review` skips blank decisions and raises `ValueError` (with the row number) on:
an unknown decision value, a `replace` without `corrected_target_id`, or any applied
decision without a `reviewer`.

`apply_review` validates **every** row before writing **any** — all-or-nothing,
because a partially-applied review file is worse than an unapplied one. Each row is
checked against the originating run, and any failure raises `SnapshotMismatch`:

- the `result_key` is unknown to the ledger;
- the result's `source_hash` differs from the row's — the source record changed since
  the run; re-run before applying;
- the run manifest's recorded `target_fingerprint` differs from the
  `target_store_fingerprint` you passed — the target snapshot changed since the run.

Only after every row passes are the rows appended to the review overlay.

`adjudicated` yields one `AdjudicatedResult` per result: `result_key`, `source_id`,
`model_target_id`, `model_status`, `confidence`, `final_target_id`, `final_status`,
`reviewer`, `review_note`, `reviewed_at`. Unreviewed results pass through with the
model's answer as final; reviewed ones apply the **latest** decision for that
`result_key`. Both the model layer and the review layer stay visible in every row.

## Gotchas

- **`ablate` spends tokens; nothing else here does.** Everything else reads the
  ledger. Budget an ablation like a run, because it is several.
- **Some `EvalReport` numbers span all results, not just labelled ones**:
  `duplicate_target_conflicts`, `mean_llm_calls`, `mean_tokens`, `mean_seconds`,
  `status_counts`, and `reason_counts`. The rates are labelled-only.
- **`None` is not zero.** Every undefined ratio is `None` and renders as `n/a` with a
  reason. If you serialise reports yourself, keep the distinction. The one asymmetry:
  `ThresholdPoint.coverage` is `0.0` when nothing is labelled, while its `precision`
  is `None` when nothing clears the threshold.
- **A silent `calibration_warning` is not a clean bill.** Below 3 correct and 3
  incorrect confident predictions it returns `None` for lack of evidence.
- **A normalizer that drops every id converts a labelled row into a no-match label.**
  Intentional design, easy to trigger by accident with an over-strict normalizer.
- **Do not change `Partitioner` salt or fractions mid-project** — assignment is a
  salted hash of the source id, so any change reshuffles all three partitions. To
  publish or freeze a split, write a partition file; explicit files always win.
- **`RunSummary.misjudged` is a count.** When comparing runs over different labelled
  subsets, normalise it yourself before reading it as a rate.
- **`defer` keeps a row in the review queue.** Re-exporting with the default
  `statuses` will include it again; that is the point of the decision.
- **Reviews bind to a snapshot.** `apply_review` refuses a file whose source records
  or target store changed since the run, and one stale row fails the whole file.
  Re-run first, re-export, then review.

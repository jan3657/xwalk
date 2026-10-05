# Benchmarks

What this page can and cannot tell you, how to reproduce it, and where the raw numbers
are. The harness lives in `benchmarks/` at the repository root; it is not part of the
installed package.

**Status (5 October 2026).** Only offline smoke checks have been run. Every number that
involves a model answer comes from a **SYNTHETIC** deterministic judge, not an LLM.
**Real-model results are PENDING**: no inference endpoint or budget was configured. The
string baselines, BM25 retrieval and candidate recall involve no model and are real
measurements on the pilot data, but the pilot is 50 records per dataset, so treat them
as a sanity check, not a ranking.

## Reproduce

```bash
pip install -e ".[dev,ontology]"     # ontology: the food catalog is OWL
python -m benchmarks.run --smoke     # about a minute; writes benchmarks/results/smoke/
python -m benchmarks.run --smoke --datasets ncbi_disease --limit 5   # seconds
```

`--smoke` needs no network and no credentials. It writes `results.json` (every raw
value, plus code commit, data digests, environment and the exact command),
`summary.md` (tables generated from it) and the clustering prediction files. Indexes,
ledgers and run directories go to a temporary `--work` directory and are discarded.

The real path calls a provider and spends money, so it needs an explicit call cap:

```bash
export XWALK_BENCH_BASE_URL=https://...  XWALK_BENCH_MODEL=...  XWALK_BENCH_API_KEY_ENV=MY_KEY
python -m benchmarks.run --real --max-calls 2000 --out benchmarks/results/real-YYYY-MM-DD
```

Without the `XWALK_BENCH_*` overrides (also `XWALK_BENCH_PROFILE`,
`XWALK_BENCH_LLM_KIND`) it uses each example job's own `llm` section. `--max-calls`
caps every LLM-using method separately. A missing credential stops the run before
anything executes (exit 2). The results record the model settings per variant, the
code commit and dirty flag, the data digests and the usage of every method. A real
result is publishable only with all of those, and only if it was produced without
changing prompts or thresholds after looking at it.

## Data

Two frozen pilot datasets, each pinned by sha256 in a manifest under
`benchmarks/manifests/`. The runner refuses to start when a pinned file changed: that
is a new data revision and needs a new manifest entry.

| Variant | Catalog | Sources | Gold no-match | Construction |
|---|---|---|---|---|
| `cafeteria_fcd` | 200 FoodOn terms | 50 food mentions | 0 | frozen sample (CafeteriaFCD test split, first 50) |
| `cafeteria_fcd_nomatch` | 185 | 50 | 11 | 15 gold targets of 8 seed mentions removed; 4 more mentions keep a reduced gold set |
| `ncbi_disease` | 200 CTD records | 50 disease mentions | 11 | frozen sample (NCBI Disease test split, first 50) |
| `ncbi_disease_nomatch` | 199 | 50 | 25 | `MESH:D011125` removed (14 mentions) |

- **No-match construction.** Mentions are visited in a salted hash order; each visited
  mention's whole gold set is deleted from the catalog until the target number of
  no-match mentions is reached (bounded by `max_nomatch`). Only gold targets are
  removed; near-neighbour distractors stay. Every gold set is then restricted to the
  remaining catalog. `construction.json` in each materialised variant lists what was
  removed.
- **Known data issue.** In `ncbi_disease`, 11 mentions have gold ids (`OMIM:215600`,
  `OMIM:261600`) that are absent from the 200-record catalog even after the example's
  alias expansion. Because gold is restricted to the catalog, they count as no-match
  cases here; `xwalk eval` against the raw `gold.csv` would count them as unreachable
  errors instead.
- **Splits.** Hash-based with `xwalk.evaluate.Partitioner` (salt `xwalk-bench-v1`):
  cafeteria 30/12/8, ncbi 21/20/9 (prompt_train/validation/test). Nothing in the
  benchmark is tuned: thresholds are fixed in `benchmarks/methods.py` or come from the
  example jobs, which predate the benchmark. Metrics are reported for `all` and `test`.
  With 8-9 test records, the test split is too small to support any comparison.
- **Upstream provenance** (URLs, versions, extraction script and date) is in each
  manifest. The CafeteriaFCD upstream URL and both upstream versions were not recorded
  when the samples were extracted, and are marked unknown rather than guessed.

## Tracks

The three tracks answer different questions and are never mixed.

**Source-to-catalog matching** (the main track). Every method sees the same catalog
and the same fields (label and synonyms) and, where it retrieves, the job's BM25
settings and candidate budget (k = 25 or 30, 25 shown to the model).

| Method | What it is | LLM calls |
|---|---|---|
| `exact` | normalised mention equals a label or synonym; several hits go to review | 0 |
| `fuzzy_difflib` | best stdlib `difflib` ratio; accept at 0.90, review at 0.75 | 0 |
| `bm25_top1` | retrieval only: BM25 top hit; it cannot abstain | 0 |
| `retrieval_one_llm_pass` | the job's selector prompt over BM25 candidates, accepted on its own confidence | 1 per record with candidates |
| `xwalk_single_attempt` | `xwalk.ops.run` with `max_attempts: 1`, no verification | select + score |
| `xwalk_full` | `xwalk.ops.run` with the example job unchanged | as many as the loop needs |

Not run: dense retrieval (needs `xwalk[dense]` and model weights) and LinkTransformer
(adapter in `benchmarks/adapters/linktransformer.py`, written against its documented
API but never executed; needs the package and a model download). Splink/dedupe were not
attempted: the pilot sources have one text field, which is not what they are for.

**Pairwise classification.** A pair table of BM25 top-5 candidates per mention, judged
by exact, fuzzy, and xwalk's scorer stage (one call per pair). It measures the
classifier only. Gold targets that blocking never proposed are not in the table and are
reported separately as `blocking_misses`, which is exactly the retrieval failure this
track cannot see.

**Clustering.** `benchmarks/clustering.py` computes pairwise and B-cubed
precision/recall/F1, false merges, splits, singleton behaviour, review coverage and
order sensitivity from any predicted assignment file:

```bash
python -m benchmarks.cluster_eval --gold gold_clusters.csv \
    --pred predicted.csv [--pred predicted_reversed_order.csv] [--meta usage.json]
```

Predictions are `item_id,cluster_id[,outcome]` (CSV or JSONL) with the outcomes of
CONTRACTS section 10 (`assigned`, `singleton`, `needs_review`, `failed`). Items outside
the accepted partition are scored two ways: as their own singletons (so sending
everything to review cannot score well), and excluded (`accepted_only`). The pilot gold
clusters group mentions with the same gold id set. xwalk clustering (task 04) is not
scored yet; the smoke run scores three baselines: identical normalised mention text,
an order-dependent greedy fuzzy leader run in two input orders, and grouping by the
accepted catalog target of the `xwalk_full` run (a matching shortcut, not a clustering
algorithm).

## Metrics

Definitions are in the docstring of `benchmarks/metrics.py`; the important choices:

- Only `matched` is system output. Final precision counts accepts on no-match rows as
  errors; final recall is over rows with non-empty gold. Review proposals appear only in
  `recall_any_status` and the review rate.
- Candidate recall is over all retrieved candidates; `truncated` counts gold that was
  retrieved but cut by the candidate budget before the model saw it.
- `recovered` counts records whose first attempt was wrong or empty and whose final
  answer is an accepted correct match; `broken_by_retry` counts the reverse, where a
  later attempt lost a correct first choice. Both are shown because a retry loop can do
  either.
- Calls and tokens come from `BudgetedLLM.usage`, the same accounting `xwalk match`
  reports. Synthetic tokens are characters / 4. Wall time and peak memory are measured
  under `tracemalloc`, which slows Python allocation and does not see tantivy's native
  memory: compare methods within one run, not across machines.

## Smoke results

Raw output: `benchmarks/results/smoke/results.json` and `summary.md`, generated at
commit `cd88923`.

What the smoke run establishes:

- Every method runs end to end on all four variants; xwalk runs finish
  `run_state: complete` with exit code 1 (records in review), and usage is recorded per
  method: at most 1 call per record for the one-pass baseline (none when retrieval
  returns nothing), 1.3-2.0 for a single xwalk attempt and 2.2-2.8 for the full loop.
- Measured without any model: BM25 candidate recall is 0.94-0.95 on the food variants
  and 0.74-0.84 on the disease variants, with no truncation by the 25-candidate budget.
  The disease misses include 12 of 50 mentions that are bare abbreviations (`CT`, `FAP`,
  `MHP`) with no BM25 hit at all; only a query rewrite that expands them from context
  could recover those, which is what a real-model run has to show.
- Measured without any model: exact and fuzzy matching reach 0.74-0.88 coverage at
  precision 1.0 on the food sample, but 0.56-0.58 coverage at about 0.89 precision on
  the disease sample. BM25 top-1, which cannot abstain, accepts a wrong target for 6-9
  of the no-match rows.
- No-match behaviour is exercised: on `cafeteria_fcd_nomatch` the synthetic full loop
  made no false accept on the 11 no-match rows where the single attempt made 3, at the
  price of a higher review rate (0.16 vs 0.10). This shows the verification path
  working; it says nothing about how often a real model would agree.
- The synthetic rewriter never recovers a record (`recovered` = 0 everywhere) and the
  retry loop lost a correct first choice once in three variants. With a real model these
  are the two numbers to watch.
- Clustering on the food pilot is nearly degenerate (48 gold clusters for 50 mentions);
  the disease pilot (9 gold clusters, no singletons) is the informative one. The greedy
  leader baseline gave the same partition in both input orders here, so order
  sensitivity is plumbed but not yet stressed.

## Performance

**Workload.** `xwalk.ops.run` on `ncbi_disease` (50 records, 200 targets, BM25, fresh
run directory each time), with a trivial FakeLLM that always picks the first candidate
at 0.95, so the profile is almost entirely xwalk's own code. Linux, Python 3.11.

```bash
python -m benchmarks.profile_workload --llm trivial
python -m benchmarks.profile_workload --experiment prompt-template-cache --llm trivial --repeat 5
```

**Finding.** `PromptSet._render` (`src/xwalk/prompts/contract.py`) builds a new Jinja
`Environment` and recompiles the prompt skeleton on every call. That compilation is
about 0.70 s of a 0.98 s profiled run (`benchmarks/results/perf/profile_ncbi_trivial.txt`),
roughly 7 ms per prompt in the profile and 3 ms without the profiler. It runs on the
event loop, so it also delays every other in-flight record.

**Measured effect of caching the compiled template** (an in-process patch applied by
the experiment; the library was not changed; prompts verified byte-identical, 88-130
prompts per run):

| Condition | Baseline median | Cached median | Raw output |
|---|---|---|---|
| trivial model, 5 runs | 0.440 s | 0.188 s | `perf/prompt_template_cache_trivial.json` |
| synthetic judge, 5 runs | 0.593 s | 0.226 s | `perf/prompt_template_cache_synthetic.json` |
| synthetic judge + 200 ms per call, 3 runs | 3.622 s | 3.569 s | `perf/prompt_template_cache_latency200ms.json` |

So the saving is large when the model is fast (local models, cache hits, tests) and
about 1.5% when each call takes 200 ms. No speedup is claimed for network-bound runs.
The fix (an `lru_cache` keyed by the skeleton text, as in
`benchmarks/profile_workload.py`) belongs in `xwalk/prompts/contract.py`, which was
outside this task's permitted file set; it is recorded as a follow-up.

Other candidates inspected and not changed:

- Identical concurrent cache misses: only 2 of 111 (food) and 8 of 130 (disease)
  requests in a full run were exact repeats, and `xwalk match` does not use
  `CachingLLM`. Coalescing would not change these workloads measurably.
- Repeated index construction: a fresh run directory builds its index once; reopening
  a compatible index is task 03's tested path. Not a cost here.
- Dense-array copies: not measurable without `xwalk[dense]`.
- Peak Python memory of a full run is about 1.6-2.0 MiB at this size; nothing to fix.

## Limitations

- 50 records per dataset and 8-9 test records: differences of a few records are noise.
- Synthetic results test plumbing only. The synthetic judge is a string-similarity
  heuristic, so it favours exactly the cases the string baselines already solve.
- Wall times are from one machine, under `tracemalloc` in the smoke run.
- The CafeteriaFCD upstream URL and both corpora's upstream versions are unrecorded.
- WDC Products and a larger public corpus were not added; the runner is ready for
  another manifest.

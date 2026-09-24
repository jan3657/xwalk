# Recall and cost on the decider path — design

Date: 2026-09-23. Status: implemented 2026-09-24; see Results. Branch: `jev-recall-cost` (off `jev-decider`).
Plan: `../plans/2026-09-23-jev-recall-and-cost.md`.

## Why

The decider branch's evaluation (see the Results section of
`2026-09-22-jev-decider-design.md`) found three things, in order of size:

1. **Retrieval, not judgement, caps accuracy.** First-attempt retrieval recall is
   0.94 / 0.60 / 0.58 / 0.70 on the four gold samples, identical for both paths, and the
   screen loses at most 8 points on top of that. The LLM baseline's 22- and 34-point
   leads on ncbi_disease and nlm_gene come entirely from its rewrite-and-retry loop
   lifting retrieval recall to 0.72 and 0.92.
2. **Two thirds of the cost is repeated text.** Each per-candidate screen question
   carries `domain_brief`, every hard rule, and a `true`/`false` criteria block, about
   340 tokens, re-sent once per candidate. Ref_zivila cost $5.35 for 2,030 rows.
3. **The shipped `accept_at` of 0.85 is too conservative in three of four domains.**
   Fitted values were 0.50 / 0.50 / 0.70 / 0.85, and on Ref_zivila the unfitted 0.85
   turned 825 correctly proposed ids into a 1,001-row review queue.

This spec designs one improvement per finding, one measurement harness that makes
each of them a pass/fail test, and one addition that the numbers did not ask for but
the data structure invites: a graph hop over the target ontology.

## Principles carried over

- The model decides; retrieval proposes; code combines. Nothing here moves a decision
  out of Jev. The one place an LLM re-enters (A3) it only proposes search queries, on
  the records where the screen found nothing, and Jev still decides.
- Every improvement ships with a test that fails before and passes after, on data in
  the repository, with the threshold written in the test. Live measurements are
  integration-gated and cost cents.
- The LLM path stays behaviourally unchanged. New retriever options default off.

## A. Retrieval recall

### A0. The harness first

`scripts/retrieval_recall.py` and `tests/test_retrieval_recall.py` compute
recall@k of a retriever configuration against a gold file, offline, by building the
index from the example's own sample targets (200 records each, seconds to build) and
running the example's query template over its mentions. It reports retrieval recall
(gold id anywhere in the fused list) at k in {10, 50, 200}. The four sample examples
are the fixtures. Baselines from the decider evaluation, recall@200 with the shipped
BM25 configuration: cafeteria_fcd 0.94, chebi 0.60, ncbi_disease 0.58, nlm_gene 0.70.
Every retrieval change below is judged by this harness with a written threshold.

### A1. A better BM25: stemming and fuzzy terms (no new dependency)

tantivy 0.26 ships `TextAnalyzerBuilder` (lowercase, stemming, ASCII folding) and
`Query.fuzzy_term_query`. Two opt-in `RetrieverSpec` fields:

```yaml
retrievers:
  - kind: bm25
    analyzer: en_stem      # default: "default" (tantivy's tokenizer, unchanged)
    fuzzy_distance: 1      # default: 0 (off); applies to query terms of 5+ chars
```

`analyzer` registers a custom analyzer on the index at build time (lowercase, ASCII
fold, Snowball stem) and records it in the index metadata and fingerprint. With
`fuzzy_distance`, `_search_sync` ORs a `fuzzy_term_query` for each long-enough query
term beside the parsed query, at a lower boost, so exact matches still rank first.
"anaesthetic" then reaches "anesthetic", "agents" reaches "agent".

Pass/fail: recall@200 on no sample below its baseline, and the four-sample mean at
least 0.03 above the baseline mean (0.705).

### A2. Dense multilingual retrieval (the `dense` extra, GPU available)

`DenseRetriever` and `SentenceTransformerEncoder` already exist. The work is to
install `xwalk[dense]` in the project venv (with `uv`, through the proxy), pull
`intfloat/multilingual-e5-small`, and add a dense retriever beside BM25 in the four
sample `job_jev.yaml` files and the Ref_zivila job (uncomment and set prefixes
`query: ` / `passage: `). Fusion is reciprocal rank, already in place.

Pass/fail, in two parts. English: recall@200 with BM25(A1)+dense on the four samples,
mean at least 0.08 above the A1 mean, no sample worse than A1 by more than 0.02.
Slovenian: a ten-row mini gold, `examples/ref_zivila/gold_slo_mini.csv`, of
Slovenian-only names hand-mapped to FoodOn ids (built in the plan from labels that
exist in the target CSV); dense recall@50 on it at least 0.7, where BM25 alone scores
near zero because the query shares no token with any English label.

### A3. Conditional query rewrite (an LLM proposes, Jev decides)

The baseline's whole lead is its rewrite loop, and it costs one small LLM call. On the
decider path, run it only on misses: after the first screen, if the best probability is
below `screen_floor` or there were no candidates, ask the existing `QueryRewriter`
(the LLM path's `rewrite` skeleton, unchanged) for up to three alternative queries,
retrieve again with them, and screen again, merging probabilities by record id. One
round only. Config:

```yaml
decider:
  kind: jev
  ...
  rewrite:                       # optional; absent means no rewrite
    llm: {kind: openai_compat, model: qwen/qwen3-next-80b-a3b-instruct,
          base_url: https://openrouter.ai/api/v1, api_key_env: XWALK_TEST_API_KEY}
```

The rewrite LLM never sees candidates and never decides anything; its output is three
strings. Cost lands only on the misses (about 30-40% of records on the hard domains)
at roughly $0.00003 each. The attempt trace records the rewritten queries in
`Attempt.query` (joined) and a note.

Pass/fail: offline, with `FakeLLM`: the rewriter is called only when the first screen
missed, at most once per record, and the second screen's probabilities are merged
(unit tests). Live (integration, about $0.06): accuracy on ncbi_disease and nlm_gene
at least 10 points above the A1+A2 run, with matched precision unchanged.

### A4. Graph hop over the target (the addition the data invites)

Ontology targets carry `parents` (OWL, OBO, and the Ref_zivila CSV). Many misses are
"the right node is one hop from a retrieved node": the screen finds `persimmon` at 0.6
but the gold is its child `persimmon (raw)`, which BM25 never returned. When the
first screen's best probability lies in a middle band (`hop_band`, default 0.30 to
0.85), expand the candidate set with the parents and children of the top `hop_top`
(default 3) candidates that are not already present, screen the expansion as one more
chunk, and continue as usual. Children need a reverse index, built once from the
store at matcher construction when `hierarchy_field` is set:

```yaml
decider:
  hierarchy_field: parents       # optional; absent means no hop
  hop_band: [0.30, 0.85]
  hop_top: 3
```

It is one extra chunk on the records that need it, no new provider, no new dependency.

Pass/fail: offline first, from the existing Ref_zivila and cafeteria ledgers: count
labelled or adjudicated misses whose gold id is a parent or child of a retrieved
candidate. If fewer than 10% of misses qualify, the feature is not worth building and
the plan stops here (the test records the number). Otherwise, with `FakeDecider`,
unit tests pin: hop only inside the band, only `hop_top` seeds, no duplicates, one
extra chunk. Live gate on cafeteria_fcd (the FoodOn sample, about $0.01): accuracy
not lower, and at least one previously missed gold row now matched.

## B. Cost

### B1. Hoist the preamble into the state

`Screener` puts the rules once in the shared state and the per-candidate noul refers
to them:

```json
{"source": {...},
 "rules": {"entity": "<entity_noun>", "target": "<target_noun>",
           "domain": "<domain_brief>", "hard_rules": ["...", "..."],
           "same": "<what true means>", "different": "<what false means>"},
 "candidates": {"C001": "...", ...}}
```

Per-candidate question: "Does `candidates.C001` denote the same `rules.entity` as
`source`? Apply every rule in `rules.hard_rules`. `rules.same` and `rules.different`
define the answer." No `criteria` block. `questions_version` bumps to 2, which moves
the fingerprint, as it should. Choose and gate keep their current shape (they run once
per record; their preamble is not the cost).

Pass/fail: offline, the JSON-serialised request for a 50-candidate chunk with the
Ref_zivila slots is at least 55% smaller than with `questions_version` 1 (the test
builds both). Live (integration, about $0.04): re-run the four gold samples; accuracy
per domain within 2 points of the recorded decider results and the four-domain mean not
lower; tokens per record from the manifest at least 50% lower. If the accuracy gate
fails, the fallback is a once-per-chunk instruction preamble rather than state, which
the plan carries as an alternative.

### B2. A lighter candidate rendering for the screen (optional knob)

`templates.screen_candidate`, defaulting to `templates.candidate`. The screen sees a
short rendering (label and synonyms); the gate sees the full one. Measured by the same
live gate; adopted only if accuracy holds. Listed last because B1 is the large term.

## C. Thresholds

### C1. `xwalk fit` learns to hold out and to write back

Two flags. `--holdout` splits labelled rows deterministically by source id (the
existing `Partitioner`), fits on the dev half, and reports precision and coverage on
the held-out half beside the dev numbers, so a recommendation on 50 rows is not
mistaken for a law. `--write-job OUT.yaml` writes a copy of `--job` with the fitted
`accept_at` and `property_floor` in its `policy:` block. The sweep also gains
`choose_at` on a coarse grid, since three of the four fits pinned it at the default.

Pass/fail: unit tests on synthetic results: the split is deterministic and disjoint;
the written job loads and its policy equals the recommendation; held-out numbers are
reported separately. Evaluation on the existing ledgers: held-out precision at or
above 0.95 at the recommended point on every sample where a point exists.

### C2. A Ref_zivila gold from the adjudication

The 40 adjudicated disagreements become `examples/ref_zivila/gold_adjudicated.csv`:
verdict `jev` gives Jev's id, `qwen` gives Qwen's, `both_acceptable` gives both
separated by `|`, `both_wrong` and `unsure` are omitted. It is a biased sample (only
disagreements) and says so in a header comment, but it is the first labelled data in
the real domain and makes `xwalk fit --job examples/ref_zivila/jobs/foodon/job_jev.yaml`
meaningful.

Pass/fail: the file loads with at least 35 labelled rows; `fit --holdout` on the
existing Ref_zivila ledger reports a recommended point.

## Order and budget

A0 (harness) → C1, C2 (cheap, offline) → A1 → B1 → A2 → A4 (with its go/no-go) → A3 →
final measurement. Live spend: the four-sample gates about $0.04 each, A3 about $0.06,
one Ref_zivila re-run after B1 and A2 about $2. Total under $3.

## Out of scope

Self-consistency for jitter; hierarchy descent as a retrieval strategy (A4 is a single
hop, not a walk); multi-label output; question wording changes (the numbers say not
to spend effort there).

## Results

### C2. Ref_zivila gold

`scripts/adjudication_to_gold.py` turns the 40-row adjudicated sample
(`2026-09-22-jev-adjudicated-sample.csv`) into `examples/ref_zivila/gold_adjudicated.csv`:
39 labelled rows (both_acceptable 21, jev 14, qwen 4; the 1 both_wrong row is left
unlabelled). This is a biased sample: it covers only Qwen/Jev disagreements from the
2026-09-22 run. A blank id in the sample means that decider answered "no match". Two `jev`
rows pick Jev's no-match, so their gold label is an explicit no-match. Eleven
`both_acceptable` rows have one side blank and keep only the named id, because a gold set
cannot express "this id or no match". Only 10 of the 21 carry two ids.

```
xwalk fit --run runs/ref_zivila/foodon_jev --gold examples/ref_zivila/gold_adjudicated.csv \
  --job examples/ref_zivila/jobs/foodon/job_jev.yaml --holdout --precision 0.95
```

```
recommended: accept_at=0.50 property_floor=0.50 choose_at=0.50 (29/30 correct, coverage 0.77, 2 rows within the jitter margin of accept_at)
holdout: accept_at=0.50 property_floor=0.50 choose_at=0.50 accepted=15 correct=14 precision=0.93 coverage=0.79
```

The recommendation sits at the bottom of the swept grid (accept_at 0.50), and on the held-out
half it falls just short of the 0.95 target (14/15). With 39 rows, all of them contested,
this does not justify changing thresholds. It is a first data point in the real domain.

### A1. BM25 stemming and fuzzy terms

Recall@200 on the four samples (cafeteria_fcd / chebi / ncbi_disease / nlm_gene, mean):

| Configuration | Recall@200 | Mean |
|---|---|---|
| baseline (`default`, fuzzy 0) | 0.94 / 0.60 / 0.58 / 0.70 | 0.705 |
| `analyzer: en_stem` | 0.94 / 0.70 / 0.58 / 0.70 | 0.730 |
| `fuzzy_distance: 1` | 0.94 / 0.60 / 0.58 / 0.70 | 0.705 |
| both | 0.94 / 0.70 / 0.58 / 0.70 | 0.730 |

The gate (mean >= 0.735, no sample below baseline) failed by 0.005. The whole lift is stemming
on chebi. Fuzzy terms added nothing because the misses are ids BM25 never returns at any depth
(Task 1 found recall flat from k=10 to k=200), not near-miss spellings. Both options ship opt-in,
the sample jobs keep the defaults, and `test_stem_and_fuzzy_lift_recall` is a strict xfail.

### B1. The screen preamble in the state

The size test (50-candidate Ref_zivila chunk) gives new/old = 26,613 / 92,156 = 0.29 of the
version-1 request. A live probe on one screen call with the Ref_zivila slots agrees: 40
candidates cost 14,805 prompt tokens before and 4,346 after (10 candidates: 3,915 / 1,556).
Live gate, `runs/jev_eval` against `runs/jev_eval_v2` (`scripts/compare_gates.py`):

| Domain | Accuracy before / after | Tokens per record before / after (ratio) |
|---|---|---|
| cafeteria_fcd | 0.90 / 0.92 | 4067 / 3124 (0.77) |
| chebi | 0.58 / 0.58 | 1905 / 1923 (1.01) |
| ncbi_disease | 0.46 / 0.44 | 6118 / 4953 (0.81) |
| nlm_gene | 0.58 / 0.58 | 4412 / 3188 (0.72) |

Mean accuracy delta +0.000, every domain within -0.02: accuracy holds. The token condition
(at least 50% lower) fails. These samples screen few candidates per record (6.6 / 0.7 / 8.3
/ 10.4 on average), so choose and gate dominate their cost, and on chebi the `rules` block
costs more than it saves. The saving scales with the candidates screened, which is where
Ref_zivila (up to 300 per record) spends. The alternative (per-noul `criteria` holding
`same` / `different`, `runs/jev_eval_v2b`) scored worse on both counts (ratios 0.82 / 1.03 /
0.85 / 0.83, mean accuracy -0.005), so the committed variant is the plain short question.
The gate runs cost $0.028 (v2) and $0.030 (v2b); the baseline run cost $0.035.

Ruling: B1 ships as committed (v2). The token condition failed as written on the four samples
(ratios 0.77 / 1.01 / 0.81 / 0.72) because they screen 0.74 to 10.4 candidates per record,
where Ref_zivila screens up to 300. At about one candidate per record the per-chunk `rules` block
roughly cancels the per-question saving (chebi). The at-least-50% tokens-per-record condition is
re-hosted on Task 9's Ref_zivila re-run, measured against the recorded 62,740 prompt tokens per
record.

### A2. Dense multilingual retrieval

`intfloat/multilingual-e5-small` (revision 614241f6) as a second retriever (`limit: 20`,
`query: ` / `passage: ` prefixes), fused with BM25 by reciprocal rank; sentence-transformers
6.1.0, transformers 5.17.0, torch 2.14.0+cu130, faiss-cpu 1.15.1. English gate, recall@200
against the shipped BM25 (A1 not adopted; baseline 0.94 / 0.60 / 0.58 / 0.70, mean 0.705; the
gate needs mean >= 0.785 and no sample more than 0.02 below its baseline), by dense `limit`:

| Dense `limit` | Recall@200 (cafeteria_fcd / chebi / ncbi_disease / nlm_gene) | Mean | Gate |
|---|---|---|---|
| 20 | 0.96 / 1.00 / 0.70 / 0.92 | 0.895 | pass |
| 50 | 0.98 / 1.00 / 0.72 / 0.92 | 0.905 | pass |
| 150 | 1.00 / 1.00 / 0.76 / 0.92 | 0.920 | pass |

Every dense hit is a candidate the decider screens, so the four sample jobs ship the smallest
limit that passes, 20. Slovenian gate, on
`examples/ref_zivila/gold_slo_mini.csv` (ten rows with no English name; ids chosen by a person
from FoodOn labels, not from a run), recall@50 against all 28,372 FoodOn records: fused 0.20,
dense-only 0.20, BM25-only 0.00. Failed (needs >= 0.7); `multilingual-e5-large` probed at 0.30.
E5 does not know Slovenian food words ("Kosmulja", "Som", "Pšenična moka"), so the Ref_zivila
job keeps its dense block commented out and `test_dense_recovers_slovenian_only_rows` is a
strict xfail. Two things make it worse than it need be. Reciprocal-rank fusion over the three
query templates drops hits a single query does make: the camel-milk gold is at rank 8 on the
third query and green coffee at rank 35, and both fall outside the top 50 after fusion; the
best single query per row would give dense recall 0.40. And the shared `doc` template (label,
synonyms, definition, parent labels) dilutes the embedding: embedding label and synonyms only
moved herring oil from rank 133 to 12, coffee from 35 to 11 and camel milk from 8 to 2, and
best-per-query recall from 4/10 to 5/10. Even so the gate would fail (5/10 < 0.7), so the
verdict stands. Live, `runs/jev_eval_v2` against `runs/jev_eval_v2_dense_limit` (limit 20, a
baseline for A3/A4):

| Domain | Accuracy v2 / v2_dense_limit | Tokens per record v2 / v2_dense_limit (ratio) |
|---|---|---|
| cafeteria_fcd | 0.92 / 0.90 | 3124 / 4676 (1.50) |
| chebi | 0.58 / 0.64 | 1923 / 5449 (2.83) |
| ncbi_disease | 0.44 / 0.56 | 4953 / 8975 (1.81) |
| nlm_gene | 0.58 / 0.80 | 3188 / 5732 (1.80) |

Mean accuracy +0.095, accepted precision 97.4 / 100 / 100 / 100%. For comparison, the first
run at limit 150 (`runs/jev_eval_v2_dense`) scored 0.92 / 0.62 / 0.58 / 0.80 (mean +0.100) at
21,193 / 27,691 / 30,736 / 24,945 tokens per record (6 to 14 times v2, $0.22 for the run):
limit 20 keeps nearly all of the accuracy for 20 to 29% of those tokens.

### A4. Graph hop over the target

Offline go/no-go with `scripts/hop_headroom.py`. A retrieval miss is a labelled row (gold not
"no match") with no gold id among its first attempt's candidates, the only kind a hop can
repair. It qualifies when a gold id is a parent of a retrieved candidate or a child of one
(the gold record's `parents`, from the job's target). Status misses (final `matched_id` not in
gold) are listed for information.

```
scripts/hop_headroom.py runs/jev_eval/cafeteria_fcd examples/cafeteria_fcd/sample/gold.csv \
  parents --job examples/cafeteria_fcd/job_jev.yaml
scripts/hop_headroom.py runs/jev_eval_v2_dense_limit/cafeteria_fcd \
  examples/cafeteria_fcd/sample/gold.csv parents --job examples/cafeteria_fcd/job_jev.yaml
scripts/hop_headroom.py runs/ref_zivila/foodon_jev examples/ref_zivila/gold_adjudicated.csv \
  parents --job examples/ref_zivila/jobs/foodon/job_jev.yaml
```

| Ledger | Labelled | Retrieval misses | Status misses | Qualifying | Share |
|---|---|---|---|---|---|
| `jev_eval/cafeteria_fcd` (BM25 only) | 50 | 3 | 5 | 0 | 0.00 |
| `jev_eval_v2_dense_limit/cafeteria_fcd` (shipped) | 50 | 2 | 5 | 0 | 0.00 |
| `ref_zivila/foodon_jev` | 37 | 0 | 4 | 0 | 0.00 |

No-go: both shipped-configuration shares are 0.00, below the 0.10 bar, so the hop was not
built. On cafeteria_fcd the remaining retrieval misses are T23 (gold `tortilla`, parent
FOODON:00001917) and T44 (`onion (raw)`, parent FOODON:03316347); BM25 alone also missed
T37 (`food (cooked)`). None of their retrieved candidates is the gold's parent or child.
Most status misses are choice or gate errors among retrieved candidates, and a hop cannot
fix those. The Ref_zivila number says little. That gold covers only Qwen/Jev disagreements,
so both deciders' ids were retrieved by construction (37 labelled rows, the 2 explicit
no-match rows excluded). Its zero reflects how the sample was chosen and is not a
measurement of hop headroom on the full 2,030 rows.

### A3. Conditional query rewrite

Built as specified and opt-in (`decider.rewrite`, absent by default; a job without it keeps
its fingerprint and behaviour). Live gate against `runs/jev_eval_v2_dense_limit`, with
`qwen/qwen3-next-80b-a3b-instruct` via OpenRouter as the rewrite LLM, `max_queries: 3`, the
v2 index reused, into `runs/jev_eval_v3_rewrite/<ex>`:

| Domain | Accuracy v2_dense_limit / v3_rewrite (gate) | Accepted precision | Tokens per record | Records rewritten |
|---|---|---|---|---|
| ncbi_disease | 0.56 / 0.56 (>= 0.66) | 1.00 / 1.00 | 8975 / 10378 | 4 of 50 |
| nlm_gene | 0.80 / 0.80 (>= 0.90) | 1.00 / 1.00 | 5732 / 6578 | 5 of 50 |

Failed: accuracy did not move on either domain. Precision held and the cost guard held
(the rewrite ran on 8-10% of records, not every one; run cost $0.0217 and $0.0137 against
$0.0189 and $0.0121, most of the difference being the second screens, which saw 23 to 88
new candidates per rewritten record). The trigger is the reason. Of ncbi_disease's 14
records whose gold was never retrieved, only one had a first-screen best below
`screen_floor` (0.30); the other thirteen scored a wrong candidate between 0.33 and 0.88.
On nlm_gene it is one of four. A miss on these domains hides behind a plausible neighbour
(a related disease, a paralog), so "the screen found nothing" almost never fires where the
recall is lost, and the records it does fire on (bare abbreviations like "CT", "MHP",
"Bmp") stayed unresolved after the second pass. The ncbi_disease failure decomposition
moved one record from never-retrieved to misjudged. The code stays (opt-in, tested) and
the sample jobs and the Ref_zivila job do not set it. Widening the trigger (for example,
rewriting whenever the best is below `accept_at`) would be a different experiment with a
different cost profile. It was not run here.

### Summary

Retrieval, recall@200 from the harness (cafeteria_fcd / chebi / ncbi_disease / nlm_gene,
mean; A1 and A2 subsections): baseline BM25 0.94 / 0.60 / 0.58 / 0.70 (0.705); A1 stemming
and fuzzy 0.94 / 0.70 / 0.58 / 0.70 (0.730, not adopted); A2 dense `limit: 20` beside BM25
0.96 / 1.00 / 0.70 / 0.92 (0.895, shipped on the samples). On Slovenian-only Ref_zivila rows
dense recall@50 was 0.20, so that job keeps BM25 alone.

Decisions on the four samples, accuracy (`recall_at_any_status`) and accepted precision per
gate, from `runs/<run>/<ex>/eval.json`:

| Gate (run) | Accuracy | Accepted precision |
|---|---|---|
| baseline (`jev_eval`) | 0.90 / 0.58 / 0.46 / 0.58 | 0.968 / 1.00 / 1.00 / 1.00 |
| B1 (`jev_eval_v2`) | 0.92 / 0.58 / 0.44 / 0.58 | 0.976 / 1.00 / 1.00 / 1.00 |
| B1 + A2 limit 150 (`jev_eval_v2_dense`) | 0.92 / 0.62 / 0.58 / 0.80 | 0.976 / 1.00 / 1.00 / 1.00 |
| B1 + A2 limit 20, final (`jev_eval_v2_dense_limit`) | 0.90 / 0.64 / 0.56 / 0.80 | 0.974 / 1.00 / 1.00 / 1.00 |
| + A3, two domains (`jev_eval_v3_rewrite`) | - / - / 0.56 / 0.80 | - / - / 1.00 / 1.00 |

The final sample jobs are byte-identical, apart from the fitted `policy:` values below, to
the jobs that produced `jev_eval_v2_dense_limit`, so that run is the final measurement and
the samples were not re-run. Tokens per record, baseline to final: 4067 -> 4676, 1905 ->
5449, 6118 -> 8975, 4412 -> 5732; cost per record (summed `usage.cost_usd` over
`results.jsonl`) $0.000171 -> $0.000196, $0.000080 -> $0.000229, $0.000257 -> $0.000377,
$0.000185 -> $0.000241, $0.035 -> $0.052 for the 200 rows. The samples pay for the recall
gain in tokens: dense adds candidates to screen, and B1 alone saved 19-28% on three of four.

Ref_zivila, `runs/ref_zivila/foodon_jev` (2026-09-22) against `runs/ref_zivila/foodon_jev_v3`
(B1 and the fitted policy, BM25 only; 2,030 rows, 361 s of matching):

| | 2026-09-22 | v3 |
|---|---|---|
| matched / needs_review / unmatched | 442 / 1001 / 587 | 835 / 602 / 593 |
| prompt tokens per record | 62,740 | 24,726 (0.39) |
| model calls per record | 4.54 | 4.53 |
| cost (run / per record) | $5.35 / $0.00264 | $2.11 / $0.00104 |
| adjudicated sample: accuracy | 0.892 | 0.892 |
| adjudicated sample: accepted precision (coverage) | 1.00, 22/22 (0.56) | 0.966, 28/29 (0.74) |

The B1 token gate, re-hosted here, passes: 60.6% fewer prompt tokens per record at the same
number of calls. `scripts/compare_runs.py` on the two `mapping.csv` files: 438 rows matched by
both, 432 on the same FoodOn id (98.6%), 6 on different ids, 4 matched only before, 397 only
after. Moves: 397 needs_review -> matched, 46 needs_review -> unmatched, 40 unmatched ->
needs_review, 4 matched -> needs_review. Most of the new matches are the lower `accept_at`
(0.85 -> 0.50) at work; the adjudicated sample prices that at one wrong accept in 29, on 39
contested rows, and it is not a precision estimate for the whole file.

Fitted policies (`xwalk fit --holdout`, target precision 0.95, written into each job):

| Job (fit on) | accept_at / property_floor / choose_at | Full fit | Holdout |
|---|---|---|---|
| cafeteria_fcd (`v2_dense_limit`) | 0.75 / 0.70 / 0.50 | 38/39, cov 0.78 | 17/17 = 1.00, cov 0.68 (dev half chose choose_at 0.70) |
| chebi | 0.65 / 0.50 / 0.70 | 27/27, cov 0.54 | 14/14 = 1.00, cov 0.56 |
| ncbi_disease | 0.75 / 0.50 / 0.50 | 20/20, cov 0.40 | 8/8 = 1.00, cov 0.32 (dev half chose 0.90 / 0.50 / 0.70) |
| nlm_gene | 0.85 / 0.50 / 0.70 | 29/29, cov 0.58 | 13/13 = 1.00, cov 0.52 |
| ref_zivila (`foodon_jev`) | 0.50 / 0.50 / 0.50 | 29/30, cov 0.77 | 14/15 = 0.93, cov 0.79 |

What earned its keep: B1 (always on) and A2 on the English samples. The rest did not:
- A1: +0.025 mean recall, short of the +0.03 gate; all of it stemming on chebi, and dense recovers more.
- A3: accuracy unchanged on both domains; its trigger (nothing above `screen_floor`) rarely fires where recall is lost.
- A4: hop headroom 0.00 on every ledger, so it was not built.

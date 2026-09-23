# Recall and cost on the decider path — design

Date: 2026-09-23. Status: draft for review. Branch: `jev-recall-cost` (off `jev-decider`).
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

# The Jev decider — design

Date: 2026-09-22. Status: draft for review. Branch: `jev-decider`. Plan: `../plans/2026-09-22-jev-decider.md`.

## Purpose

Add a second decision path to xwalk in which TypeSafe's Jev, a non-generative
"System One" decision model, makes every decision the LLM makes today: which candidate
is the match, how sure we are, and whether to abstain. The path exists because Jev
answers in about a third of a second for a fraction of a cent, which changes what a
matching loop can afford. Where the LLM path shows the model 25 candidates and retries
when they are wrong, the Jev path shows it hundreds and never needs to retry.

The existing LLM path is untouched. Both paths produce the same `MatchResult`, write to
the same ledger, and are measured by the same evaluation code, so a job can be run both
ways and compared on gold.

## What the evidence says

Measured on 2026-09-22 against `~typesafe/jev-latest` through OpenRouter. Full numbers
are in `scratch/jev/` and the scratchpad; the ones that shaped the design:

| Gold sample (50 rows, 200 targets) | Gold reachable | BM25 recall@25 | Jev on BM25 top-25 | Jev on all 200, one call | Jev on all 200, 4 calls of 50 |
|---|---|---|---|---|---|
| cafeteria_fcd (FoodOn) | 1.00 | 0.94 | 0.92 | 0.94 | 0.92 |
| chebi | 1.00 | 0.60 | 0.60 | 0.94 | 0.98 |
| nlm_gene | 0.92 | 0.70 | 0.68 | 0.86 | 0.92 |
| ncbi_disease | 0.78 | 0.58 | 0.56 | 0.66 | 0.64 |

Accuracy is top-1 by per-candidate `noul` probability, against gold.

1. **Retrieval, not the decision, is the ceiling.** Where BM25 misses the gold, Jev
   finds it when allowed to see it. On nlm_gene it found every reachable gold record.
2. **Chunks of about 50 beat one large call** in three of four domains, which matches
   TypeSafe's documented degradation on large states. Chunking is the default.
3. **The per-candidate yes/no (`noul`) is the reliable signal.** The `choice` question
   over the same candidates scored lower everywhere (0.72 versus 0.94 on chebi) because
   its probability mass splits across near-duplicate ontology labels. `choice` is kept
   only as a comparative tie-break over a short list.
4. **Calibration holds.** When the top `noul` was at least 0.8, accuracy was 0.96 to 1.0
   in three domains; the fourth has gold ids missing from its target sample.
5. **Cost and speed.** About 0.4 s and $0.0014 per record for 200 candidates in four
   chunks. The OpenRouter endpoint served 64 concurrent requests at about 50 requests
   per second with no rate-limit responses.
6. **Slovenian-only sources worked** on hand-built candidate sets. Their problem is
   that BM25 over English labels returns nothing for a Slovenian query.
7. **Answers are not deterministic.** Five identical requests returned the same
   `choice` every time, but every probability moved by up to 0.04 between calls
   (0.78 to 0.82 on the same noul). Consequences: responses are cached in the ledger
   by default so a resumed run replays rather than re-rolls, and threshold fitting
   reports a margin rather than a knife edge.

Two constraints from the vendor shape everything below: a `choice` takes at most 255
options; a `score` takes 2 to 10 ordered levels; one request carries at most 32k tokens
of state plus its longest question. Jev is literal, so instructions must state the
exact condition and put boundary cases in the criteria.

## Design principles carried over

- **The model decides; code combines.** Jev returns probabilities. Thresholds and the
  status they imply live in policy, never in a prompt. This is TypeSafe's own
  recommendation and it keeps xwalk's existing rule that nothing downstream trusts a
  capability declaration.
- **A malformed answer never resolves to a real record.** Candidates are still shown
  under opaque keys. An answer naming a key not issued this attempt is unresolved.
- **Same outcomes, same ledger, same evaluation.** `matched`, `needs_review`,
  `unmatched`, `failed` keep their meanings. A provider outage is `failed`, never
  evidence of a non-match.
- **No generated explanation.** The reason column is a deterministic rendering of the
  signals Jev returned. A reviewer reads "same 0.93, chosen over 12 with p 0.81,
  processing state agrees 0.95, species agrees 0.98" instead of a paragraph.

## Architecture

```
render queries from the source record (several, not one)
│
├─ 1. RETRIEVE   every retriever × every query, concurrently; fuse by reciprocal rank;
│                keep up to `max_candidates` (default 300)
│
├─ 2. SCREEN     chunks of `chunk_size` (default 50) candidates, one Jev call each,
│                one `noul` per candidate: "does this candidate denote the same
│                entity as the source?"                              → N/50 calls
│                → per-candidate probability; shortlist = top `shortlist_size`
│                  with probability ≥ `shortlist_floor`
│                → if the best probability < `screen_floor`: UNMATCHED, stop
│
├─ 3. CHOOSE     one Jev call: `choice` over the shortlist plus NONE, instructions
│                "the most specific candidate the source supports"    → 1 call
│
└─ 4. GATE       one Jev call on the chosen candidate alone: a `score` over the
                 rubric levels, one `noul` per declared property
                 (processing state agrees? species agrees? …)        → 1 call

derive_status(signals, policy) → one status, one reason
```

The typical record costs 6 to 8 calls and finishes in about a second of wall time,
almost all of it in parallel screen chunks. There is no retry loop: the LLM path
retries because 25 candidates were the wrong 25; the Jev path retrieves 300 instead.

### Why three stages and not one call

Screen answers an absolute question per candidate and scales to any depth. Choose
answers a comparative question among the survivors, which is where the ontology's
near-duplicates ("persimmon", "persimmon (raw)") get resolved by the specificity rule.
Gate answers the absolute question about the one chosen record with a clean state, and
returns the property-level agreement a reviewer needs. This mirrors why the LLM path
scores separately from selecting: comparative and absolute questions get different
answers. Merging choose and gate into one call is possible but the gate's state would
then carry the whole shortlist, which the evidence says costs accuracy.

### Recall

Deeper retrieval is the recall strategy, and it is retrieval work rather than Jev work:

- **Multi-query.** `templates.queries` is a list of Jinja templates rendered per
  source (English name; Slovenian name; each alias). Every retriever runs every query;
  fusion is reciprocal rank over the whole set. Today's single `templates.query` stays
  valid and becomes the one-element case.
- **Dense multilingual retriever** alongside BM25 for sources whose names are not in
  the target language. `DenseRetriever` and `SentenceTransformerEncoder` already
  exist; the job just declares one with a multilingual model.
- **Depth.** Retriever limits of 100 to 300 per query are cheap now that the screen
  can absorb them.

**Hierarchy descent** (a `choice` at each ontology level; FoodOn's largest sibling set
is 248, under the 255 cap) would give recall independent of lexical overlap. It is
deferred to a later phase and listed under future work, because multi-query plus dense
retrieval is the cheaper experiment and must be measured first.

## Components

### `xwalk.decide` (new package)

`base.py`

```python
@dataclass(frozen=True)
class Noul:   instructions: str | Mapping; criteria: Mapping | None = None
@dataclass(frozen=True)
class Choice: instructions: str | Mapping; criteria: Mapping[str, str | Mapping | None]
@dataclass(frozen=True)
class Score:  instructions: str | Mapping; criteria: Sequence[str | Mapping]

Question = Noul | Choice | Score

@dataclass(frozen=True)
class NoulAnswer:   noul: float
@dataclass(frozen=True)
class ChoiceAnswer: choice: str; probabilities: Mapping[str, float]; confidence: float
@dataclass(frozen=True)
class ScoreAnswer:  score: float; probabilities: Mapping[str, float]; confidence: float; legend: Mapping[str, str]

@dataclass(frozen=True)
class DecisionResponse:
    answers: Mapping[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    model: str            # the versioned id the provider reports
    usage: Usage          # prompt_tokens = input tokens; completion_tokens = 0; calls = 1
    cost_usd: float | None

class DecisionClient(Protocol):
    @property
    def model(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...
    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse: ...
```

Errors mirror the LLM ones so the batch runner's failure handling is shared:
`DecisionError`, `DecisionRetryableError(retry_after)`, `DecisionFatalError`. A response
missing an answer for a question that was asked is a `DecisionFatalError`; a
`ChoiceAnswer` whose `choice` is not one of the criteria keys is too.

`jev.py` — `JevClient(base_url, model, api_key, timeout, max_retries, backoff_base,
backoff_cap, transport)`. Posts `{model, state, questions}` to `base_url` as given (no
path appended, because OpenRouter's path is `/api/alpha/decisions` and TypeSafe's is
`/v1/systemone`). Retries 408, 429, 500, 502, 503, 504, 529 and transport errors with
the same backoff shape as `OpenAICompatClient`, honouring `retry-after`. Fingerprint
is a digest of base_url host, configured model id, and adapter version. The served
model version from each response goes into the attempt's `raw_selection` and into the
run manifest, so a run can be traced to `jev-1.13-20260917` even though the job said
`jev-latest`.

`fake.py` — `FakeDecider` answers from a caller-supplied function of
`(state, questions)`, with a default that scores label overlap, so the whole loop runs
offline in tests.

`cache.py` — `CachingDecider` stores responses in the ledger's existing `llm_cache`
table keyed by a digest of `(model, state, questions)`. The CLI wraps every decider
in it, because Jev's probabilities jitter between identical calls and a resumed run
must see the same answers as the first.

### Stages

`stages/screen.py` — `Screener(decider, questions, templates, chunk_size,
shortlist_size, shortlist_floor)`. Renders each candidate with the job's `candidate`
template, assigns opaque keys `C001…`, builds the state
`{"source": <source fields + context>, "candidates": {key: rendered}}` per chunk, and
one `noul` per key whose instructions name `candidates.<key>` explicitly. Chunks run
concurrently under the policy's concurrency. Returns `ScreenOutcome(probabilities:
Mapping[record_id, float], shortlist: tuple[record_id, ...], usage, chunks: int)`.

`stages/choose.py` — `Chooser(decider, questions, templates)`. State is the source plus
the shortlist under fresh keys; one `choice` whose criteria are the rendered
candidates plus `NONE`. Returns `ChooseOutcome(record_id | None, p_choice, p_none,
confidence, usage)`. A `choice` value that is not an issued key or `NONE` is
`Resolution.UNRESOLVED`, which the policy routes to review, exactly as today.

`stages/property_gate.py` — `PropertyGate(decider, questions, templates)`, its own
module so `gate.py` keeps importing only LLM types. State is the source plus the one
chosen candidate. Questions: a `score` whose criteria are the
rubric levels in ascending order, and one `noul` per declared property. Returns
`GateOutcome(rubric_score, rubric_confidence, properties: Mapping[str, float], usage)`.

### Question set

The existing `slots.yaml` is reused, because its content is exactly what Jev's
instructions and criteria need. `PromptSlots` gains one optional key:

```yaml
properties:                       # optional; used only by the decider path
  - name: processing_state
    question: Do the source and the candidate agree on processing state (raw, cooked, dried, canned, frozen, juice, oil, flour) wherever either states one?
  - name: species
    question: Do the source and the candidate agree on the source species or main ingredient wherever either states one?
```

`decide/questions.py` — `QuestionSet.from_slots(slots)` composes:

- the **screen** instruction from `entity_noun`, `target_noun`, and `hard_rules`, with
  `criteria: {true: "...", false: "..."}` spelled out (Jev is literal);
- the **choose** instruction from `disambiguation_steps` plus a fixed specificity
  clause, and a fixed `NONE` description;
- the **rubric** as `score` criteria: the rubric rows sorted ascending by score, each
  rendered as "<name>: <when>". The 2-to-10 level cap is validated at load time;
- the **properties** as nouls.

Templates are string-formatted, not Jinja, and are covered by tests that assert every
hard rule appears verbatim in the screen instruction.

### Policy

`DecisionPolicy` (frozen dataclass, `decide/policy.py`):

| Field | Default | Meaning |
|---|---|---|
| `screen_floor` | 0.30 | best screen probability below this → `unmatched`, no further calls |
| `shortlist_size` | 15 | at most this many survivors reach choose |
| `shortlist_floor` | 0.20 | survivors must have at least this screen probability |
| `none_at` | 0.70 | `p_none` at or above this → `unmatched` |
| `choose_at` | 0.50 | `p_choice` below this → `needs_review` |
| `accept_at` | 0.85 | screen probability of the chosen record required for `matched` |
| `rubric_floor` | level index of the second-highest rubric row | rubric score below this → `needs_review` |
| `property_floor` | 0.50 | any property below this → `needs_review` |
| `concurrency` | 32 | |
| `retriever_timeout` | 60.0 | |
| `chunk_size` | 50 | |
| `max_candidates` | 300 | fused candidates kept after retrieval |

`derive_status(signals, policy)`:

1. retrieval failed everywhere → `failed` / `retriever_failure`
2. no candidates → `unmatched` / `no_candidates`
3. best screen probability < `screen_floor` → `unmatched` / `below_review_floor`
4. choose said NONE with `p_none` ≥ `none_at` → `unmatched` / `selector_abstained`
5. choose unresolved, or NONE below `none_at` → `needs_review` / `unresolved_output`
6. all of: screen(chosen) ≥ `accept_at`, `p_choice` ≥ `choose_at`, rubric ≥
   `rubric_floor`, every property ≥ `property_floor` → `matched` / `accept_threshold`
7. otherwise → `needs_review` / `below_accept_threshold`

`MatchResult.confidence` is the screen probability of the chosen record, because it is
the calibrated one. There is no `verify_band`, `audit_rate`, or `max_attempts`: there
is no second model to ask and no retry loop. Both are listed under future work.

The defaults are starting points from the probe, not fitted values. Fitting is a
first-class step: `xwalk fit --run … --gold … --precision 0.95` sweeps `accept_at` and
`property_floor` over a completed run's signals and prints the policy that meets the
precision target with the most automatic matches. The existing `threshold_curve` is
the basis.

### Records and serialisation

`Attempt` and `MatchResult` gain `signals: Mapping[str, float]` with default `{}`, and
`Usage` gains `cost_usd: float = 0.0`. Both are additive and default-valued so every
existing ledger, JSONL file, and test keeps working. The decider path fills `signals`
with the flat set above (`screen`, `p_choice`, `p_none`, `choice_confidence`,
`rubric`, `rubric_confidence`, and one `prop_<name>` per property) and renders
`explanation` from them deterministically. `raw_selection` holds the JSON of the
choose and gate answers plus the served model version. `candidates` in the trace keeps
the fused list as today; screen probabilities live in `signals` under `screen_<key>`
for the shortlist only, to keep the blob small. That is enough to measure shortlist
recall from the ledger.

### Matcher

`DecisionMatcher` (`decide/matcher.py`) is a separate class with the same public
surface the batch runner uses: `match(record)`, `run_fingerprint`, `policy`
(exposing `concurrency`), `store_fingerprint`. Retrieval and fusion move out of
`Matcher._retrieve` into `xwalk/retrieve.py` as `retrieve(queries, source,
retrievers, store, timeout, rrf_k)`, and both matchers call it. `run_batch` is
duck-typed already; its type hint widens to a small `Protocol`.

### Configuration

`JobSpec` gains `decider: DeciderSpec | None` and makes `llm` optional; exactly one of
the two must be present, and `policy` is validated against whichever path is chosen
(`PolicySpec` for the LLM path, `DecisionPolicySpec` for the decider path, chosen by
the presence of `decider`).

```yaml
decider:
  kind: jev
  model: "~typesafe/jev-latest"           # pin "typesafe/jev-1.13" for reproducible runs
  base_url: https://openrouter.ai/api/alpha/decisions
  api_key_env: XWALK_TEST_API_KEY

templates:
  queries:                                # new; `query:` still accepted
    - "{{ mention_en }}"
    - "{{ name_slo }}"
    - "{{ aliases | join(' ') }}"

retrievers:
  - kind: bm25
    limit: 150
  - kind: dense
    model: intfloat/multilingual-e5-small
    limit: 150

policy:
  accept_at: 0.85
  screen_floor: 0.30
  chunk_size: 50
  max_candidates: 300
  concurrency: 32
```

`build_run_fingerprint` includes the decider fingerprint and `DecisionPolicy` fields
in place of the LLM ones. The CLI `match`, `eval`, `compare`, and `review` commands
work unchanged; `ablate` and `prompts optimize` are LLM-path features and say so when
given a decider job.

### Error handling

| Situation | Outcome |
|---|---|
| A screen chunk fails after retries | the attempt is `failed` / `provider_failure`; partial screens are not evidence |
| `DecisionFatalError` (auth, unknown model, malformed request) | propagates and stops the batch, as `LLMFatalError` does today |
| An answer key missing from the response | `DecisionFatalError`, because the contract is broken, not the data |
| `choice` names an unissued key | `needs_review` / `unresolved_output` |
| A property noul missing from the job | the property is simply not asked; policy checks only declared properties |
| State over the 32k limit | `Screener` lowers its chunk size for that record and records a note; the choose and gate states are small by construction |

### Testing

- Unit: question composition (every hard rule present; rubric ascending; level cap),
  chunking (exact sizes, key uniqueness across chunks, empty input), shortlist
  selection, `derive_status` over a table of signal vectors, config exclusivity and
  policy selection, serde round-trip with and without `signals`, fingerprint
  sensitivity to model and policy.
- Offline loop: `FakeDecider` with label-overlap scoring runs the four sample jobs end
  to end through `run_batch`, and `evaluate` reads the ledger.
- Integration (gated on `XWALK_TEST_API_KEY`): one real screen chunk, one choose, one
  gate, asserting shapes and that probabilities sum to about 1.
- The determinism check has been done (evidence item 7); the cache is therefore the
  default and a test asserts the CLI wraps the decider in it.

## Evaluation plan

1. Run each of the four gold samples both ways with the same retrievers and compare
   with `xwalk compare`. The LLM baseline is re-run at the same time so the comparison
   is like for like.
2. Run `ref_zivila` FoodOn (2,030 rows) with the decider and compare against the
   finished Qwen and Nex runs with `scripts/compare_runs.py`; adjudicate a sample of
   the three-way disagreements by hand.
3. Measure screen-stage recall separately from final accuracy: on gold, what fraction
   of gold ids reach the shortlist. This is the number that tells us whether to work
   on retrieval or on questions next.
4. Fit thresholds on half of each gold set and report on the other half. Because
   probabilities jitter by about 0.04 between calls, the fitter reports how many
   held-out rows sit within 0.05 of each chosen threshold, so a policy that looks
   good only because of where the jitter landed is visible.

## Out of scope for this spec

- Hierarchy descent as a retrieval strategy.
- A verifier or audit sample. A cheap option is self-consistency: the same or
  rephrased gate questions asked a few times and averaged, which would also smooth
  the jitter. Deferred until the evaluation says review load needs it.
- Multi-label output. Screen probabilities per candidate make it possible; the output
  contract in `docs/ref-zivila-mapping-plan.md` still says one id per cell.
- Generated explanations for reviewers via an LLM on `needs_review` rows only.
- Question optimisation in the style of `prompts optimize`. The knobs that matter for
  Jev are thresholds and criteria wording; threshold fitting is in scope, wording
  search is not.

## Relation to the LLM-first principle

xwalk's rule is that a model decides and classical methods only pre-process. Jev is a
model and it makes every decision here. What changes is that its answers are calibrated
probabilities rather than self-reported confidence, so the policy layer is doing the
job it was always meant to do: turning a model's answer into one of four outcomes.

## Results (2026-09-22)

Every number below comes from a run that completed. Reproduce the four sample runs with
`scripts/run_jev_eval.sh`; the Ref_zivila run is the single `xwalk match` in the
Ref_zivila section. One caveat on that reproduction: the cafeteria_fcd row below was
produced before this commit refit `examples/cafeteria_fcd/job_jev.yaml` to
`accept_at: 0.50` / `property_floor: 0.70`, and both fields are in the run fingerprint,
so re-running the script with the committed job produces a different fingerprint and
about 41 matched rows rather than 31. The other three jobs are unchanged. The decider is `~typesafe/jev-latest` on the alpha decisions
endpoint; the LLM baseline is `qwen/qwen3-next-80b-a3b-instruct` on OpenRouter, run
from generated jobs under `runs/llm_eval/` that differ from `examples/<ex>/job.yaml`
only in the `llm:` block and `policy.concurrency`, and pointed at the same BM25 index
the decider run used.

### Decider and LLM baseline, side by side

50 labelled rows per domain, none of them a gold "no match". "Accuracy" is
`recall_at_any_status`: the fraction of labelled rows whose `matched_id` is a gold id at
any status, which is the accuracy a reviewer would see after clearing the review queue
correctly. "Precision" is the precision of the `matched` bucket alone.

Decider path, at the policy the example jobs shipped with (`accept_at` 0.85,
`screen_floor` 0.30, `property_floor` 0.50 by default, configured depth 200):

| domain | accuracy | matched | needs_review | unmatched | precision of `matched` | calls/rec | cost/rec |
|---|---|---|---|---|---|---|---|
| cafeteria_fcd | 0.90 | 31 | 17 | 2 | 0.968 | 2.90 | $0.00017 |
| chebi | 0.58 | 19 | 10 | 21 | 1.000 | 1.76 | $0.00008 |
| ncbi_disease | 0.46 | 11 | 24 | 15 | 1.000 | 2.10 | $0.00026 |
| nlm_gene | 0.58 | 23 | 6 | 21 | 1.000 | 2.12 | $0.00019 |

LLM baseline, at the policy `examples/<ex>/job.yaml` ships (`accept_at` 0.6,
`review_floor` 0.4, `max_attempts` 3, configured depth 25-30):

| domain | accuracy | matched | needs_review | unmatched | precision of `matched` | calls/rec | tokens/rec |
|---|---|---|---|---|---|---|---|
| cafeteria_fcd | 0.94 | 44 | 4 | 2 | 1.000 | 2.26 | 2,136 |
| chebi | 0.62 | 31 | 0 | 19 | 1.000 | 1.32 | 1,216 |
| ncbi_disease | 0.68 | 47 | 2 | 1 | 0.702 | 2.46 | 6,596 |
| nlm_gene | 0.92 | 46 | 0 | 4 | 1.000 | 2.20 | 2,113 |

Cost per record is not comparable between the two tables: only the decisions endpoint
returns a `cost` field, so `OpenAICompatClient` records tokens and leaves `cost_usd` at
zero. Decider tokens per record were 4,067 / 1,905 / 6,118 / 4,412, so the two paths
are within a factor of two on tokens everywhere except cafeteria_fcd and nlm_gene, where
the decider spends about twice as much. Depth is not the reason: both paths actually
retrieved the same candidates, a mean of 6.56 / 0.74 / 8.28 / 10.42 per record, far
below either configured limit. The gap is the screen's per-candidate preamble —
`QuestionSet.screen_question` inlines `domain_brief` and every hard rule into each noul,
so the same instruction text is re-sent once per candidate. Wall-clock per record went
the other way: 0.6-1.1 s for the decider against 2.3-4.4 s for the LLM.

Two things stand out. The decider is the more trustworthy of the two where they differ
on precision: on ncbi_disease the baseline accepted 47 of 50 rows at 70.2% precision
with a calibration warning (mean confidence 0.99 on both correct and incorrect matches),
while the decider accepted 11 at 100% and put 24 in review. That is the intended trade.
But the baseline is ahead on accuracy in every domain, by 4 points on cafeteria_fcd and
chebi and by 22 and 34 points on ncbi_disease and nlm_gene, and the reason is retrieval,
not judgement — see the next two sections.

### Screen-stage recall

`scripts/screen_recall.py RUN GOLD` reports, over the labelled rows with a gold id,
whether a gold id is among the retrieved candidates, among the shortlist the screen
kept, and equal to `matched_id`.

| domain | retrieval | shortlist | final |
|---|---|---|---|
| cafeteria_fcd | 0.94 | 0.94 | 0.90 |
| chebi | 0.60 | 0.58 | 0.58 |
| ncbi_disease | 0.58 | 0.54 | 0.46 |
| nlm_gene | 0.70 | 0.62 | 0.58 |

The screen loses almost nothing: at most 8 points (nlm_gene), and 0 to 4 points
elsewhere. Retrieval is the ceiling, and the choose-and-gate steps give back a further
0 to 8 points below the shortlist.

The comparison that matters is with the baseline's retrieval. On the *first* attempt
the two paths retrieve identically — 0.94 / 0.60 / 0.58 / 0.70, the same four numbers —
because raising the BM25 limit from 25 to 200 buys nothing: these BM25 indexes return
far fewer than 25 hits per query anyway (a mean of 0.7 candidates per record on chebi,
10.4 on nlm_gene). What the baseline adds is its rewrite-and-retry loop: over its 1.3 to
1.8 attempts per record its retrieval recall rises to 0.94 / 0.68 / 0.72 / 0.92. The
whole of the decider's accuracy deficit is that one missing mechanism.

### Fitted thresholds

`xwalk fit --run R --gold G --job J --precision 0.95` sweeps `accept_at` and
`property_floor` against the run's own policy. `near_threshold` counts labelled rows
whose `screen_chosen` sits within 0.05 of the chosen `accept_at` — the rows the
observed probability jitter could move either way.

| domain | accept_at | property_floor | correct/accepted | coverage | near_threshold |
|---|---|---|---|---|---|
| cafeteria_fcd | 0.50 | 0.70 | 40/41 | 0.82 | 2 |
| chebi | 0.50 | 0.50 | 26/26 | 0.52 | 2 |
| ncbi_disease | 0.70 | 0.50 | 14/14 | 0.28 | 5 |
| nlm_gene | 0.85 | 0.50 | 23/23 | 0.46 | 3 |

The shipped 0.85 is too conservative on three of the four domains: dropping
cafeteria_fcd to 0.50 lifts automatic coverage from 0.62 to 0.82 — 20 points absolute,
about 32% relative — while precision stays at 0.976, and only 2 rows sit in the jitter
band. `examples/cafeteria_fcd/job_jev.yaml`
now ships `accept_at: 0.50` and `property_floor: 0.70`. The fitted `accept_at` spans
0.50 to 0.85 across four domains, so there is no single good default and every new
domain should be fitted on its own gold. ncbi_disease is the one to distrust: 5 of its
50 rows sit in the jitter band at the fitted threshold, and `eval` already warns that
its scores separate correct from incorrect matches by only 0.06.

`examples/ref_zivila/jobs/foodon/job_jev.yaml` is unchanged at `accept_at: 0.85`,
because Ref_zivila has no gold set and nothing was fitted for it. Its shortfall below
is what an unfitted threshold costs.

### Ref_zivila FoodOn, 2,030 rows against 28,372 FoodOn targets

One `xwalk match` at concurrency 16: 2,030 records, 9,215 decision calls, 127,362,919
prompt tokens or about 62,700 per record, 2.4 s in the matcher per record or roughly six
minutes of matching at concurrency 16, **$5.35**. That is 150 times the four sample runs
put together ($0.035); the brief budgeted well under a dollar for the whole task, which
the sample runs fit easily and this one does not.

The cost is not what it looks like. Retrieval returned a mean of 144.8 candidates per
record, not the 300 `max_candidates` cap, and 4.54 calls per record means about three
screen chunks, not six. The dominant driver is inside the screen question:
`QuestionSet.screen_question` in `src/xwalk/decide/questions.py` inlines `domain_brief`
and every hard rule into each per-candidate noul, which for these slots is about 283
tokens of identical preamble repeated ~145 times per record — roughly 41k of the 62.7k
tokens, about two thirds of the $5.35, spent re-sending the same paragraph.

| status | Jev | Qwen | Nex |
|---|---|---|---|
| matched | 442 | 1,408 | 174 |
| needs_review | 1,001 | 287 | 206 |
| unmatched | 587 | 335 | 293 |
| failed | 0 | 0 | 723 |

The Nex run covers only 1,396 of the 2,030 rows and 723 of those failed, so it is not a
comparable third run; it appears below only where the three-way table says so.

Where the two complete runs both accepted a row (427 rows) they chose the same FoodOn id
360 times, 84.3%. The more telling number is that Jev *proposed* Qwen's accepted id on
825 of Qwen's 1,408 accepted rows (58.6%) but accepted only 427 of them: 841 of those
rows are Jev `needs_review` and 140 `unmatched`. 947 of the 1,001 `needs_review` rows
carry a proposed id and the reason on every one of them is `below_accept_threshold`.
The gap between the two runs is mostly a threshold, not a difference of opinion.

Three-way counts are restricted to the 1,396 rows the incomplete Nex run covers, and
that run is worse than incomplete: 723 of its 1,396 rows (52%) are `failed`. A failed
row can never match, so the "one model matched" and "no model matched" rows below are
inflated by provider failures rather than by three models declining to match. Read them
as an upper bound on disagreement, not as a measurement of it:

| outcome | rows | share |
|---|---|---|
| one model matched, the other two did not | 603 | 43.2% |
| no model matched | 460 | 33.0% |
| two matched, same id | 191 | 13.7% |
| all three matched, same id | 85 | 6.1% |
| two matched, different ids | 46 | 3.3% |
| all three matched, not unanimous | 11 | 0.8% |

#### Hand adjudication, 40 Qwen-versus-Jev disagreements

Stratified, because a uniform sample of the 1,063 disagreements would have been 94%
"only one model matched" and would have taught nothing about identity: 20 rows drawn
from the 67 where both accepted different ids, and 20 from the 996 where exactly one
accepted. Saved as `docs/superpowers/specs/2026-09-22-jev-adjudicated-sample.csv`.

| verdict | both accepted (n=20) | one accepted (n=20) | total |
|---|---|---|---|
| both_acceptable | 10 | 11 | 21 |
| jev | 10 | 4 | 14 |
| qwen | 0 | 4 | 4 |
| both_wrong | 0 | 1 | 1 |
| unsure | 0 | 0 | 0 |

Where both models accepted a different id, Jev was never worse and was better on half.
It wins by refusing to add a qualifier the source does not state: "Peanut butter" to
`peanut butter` rather than `peanut butter (hydrogenated)`; "CHICKEN EGG Yolk" to
`chicken egg yolk` rather than `chicken egg yolk (raw)`; "Cauliflower, frozen" to
`cauliflower (frozen)` rather than `cauliflower (quick-frozen)`; "Tomato, dried, in oil"
to `tomato (sun-dried, in oil)` rather than `(sun-dried, in olive oil)`. It also picks
the specific term over a broad one where the source is specific ("wine, cooking" to
`cooking wine` rather than the EuroFIR wine class). The 10 `both_acceptable` rows in
that stratum are almost all the same case: one model chose a native FoodOn class and the
other an imported EFSA FoodEx2, EuroFIR, or GS1 GPC mirror of the same food
(`rye flour` versus `00780 - rye flour (efsa foodex2)`). Nothing in the job says which
to prefer, so both are correct; if the deliverable wants native FoodOn classes, that
belongs in the hard rules, not in the model.

In the one-accepted stratum, 11 of 20 rows are Jev proposing the *same* id Qwen accepted
and holding it at `needs_review` — a status disagreement with no identity content. Of
the rest, Jev's four wins split two and two: two refusals that were right ("Ice cream
streaked with chocolate" is not `chocolate ice cream`; "Soured milk, low fat" is not
`cow milk, skimmed`) and two rows Qwen left unmatched that Jev got (`white grape (raw)`,
`apple (raw, peeled)`). Three of Qwen's four wins are rows where a usable parent-level
mapping existed and Jev declined it ("Kir royale" to `cocktail beverage (alcoholic)`;
"Wheat toast with rye" to the FoodEx2 mixed wheat-and-rye bread group); the fourth is
"Avocado" to `avocado (raw)`, where Qwen added a qualifier the source does not state but
Jev produced nothing at all. The single `both_wrong` is "Kidney bean mature" in the canned-legumes
group, where Qwen accepted `red kidney bean (mature)` (adding "red", missing the canned
state) and Jev shortlisted the better `kidney bean (canned)` but would not accept it.

### What to do next

Retrieval, not questions. The screen costs at most 8 points of recall in any domain,
while retrieval caps three of the four at 0.58-0.70, and first-attempt retrieval is
*identical* between the decider and the LLM baseline — the baseline's 22- and 34-point
accuracy leads on ncbi_disease and nlm_gene come entirely from its rewrite-and-retry
loop lifting retrieval recall to 0.72 and 0.92. Raising `limit` from 25 to 200 changed
nothing, because BM25 over a short `doc` template returns almost no hits for a mention
that shares no token with any label (chebi averages 0.7 candidates per record). So the
work is to give the decider path a second query, not a second opinion: a dense
retriever (already written into `examples/ref_zivila/jobs/foodon/job_jev.yaml` as a
commented-out block; the four sample jobs have no such line and would need one), and a
source-side query expansion cheap enough to run before the decider (the baseline's
rewrite stage is one LLM call and is worth up to 34 points here). Second, and much
cheaper: fit `accept_at` per domain. Three of four domains fit below the shipped 0.85,
and on Ref_zivila the unfitted 0.85 is what turns 825 correctly-proposed ids into a
1,001-row review queue. Third, on cost: hoist `domain_brief` and the hard rules out of
the per-candidate screen noul into the shared state or a once-per-chunk instruction.
That is the first and largest saving — roughly two thirds of the Ref_zivila bill is the
same 283-token preamble re-sent once per candidate. It is expected to change no
decision the model makes, but that has to be re-measured on the four gold samples
before it is adopted (about $0.04 in total): Jev is literal, and moving text out of
the instruction and into the state changes what it attends to. The per-noul
`criteria: {true, false}` block is duplicated per candidate in the same way, about 60
tokens each, and should be hoisted with the preamble. Lowering `max_candidates` is
secondary, and worth doing only once retrieval is good enough that depth is genuinely
surplus. Question wording is the one place the
numbers say not to spend effort next.

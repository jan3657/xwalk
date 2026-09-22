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

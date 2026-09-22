# Prompts

Two things decide what the model is asked, and they are deliberately not the same thing.

The **skeleton** — four Jinja templates shipped inside the package, in
`src/xwalk/prompts/base/` — owns the machine-readable contract: the section headings the
parser and the contract check rely on, the instruction that makes opaque candidate keys
safe, and the JSON shape every answer must take. The **slots file** owns the domain: what
one source record is, what one target record is, how confident to be about what, and which
distinctions a careful annotator in this field would insist on.

The model is only ever asked for slots — never for prompt text. That is the whole point of
the split. A drafting model or the optimiser can produce a poor rubric, a vague brief, or a
hard rule that helps nobody; it cannot produce a prompt whose output will not parse, whose
candidate list went missing, or whose key instruction was edited away, because it never
touches the part of the prompt that carries those. `validate_contract` renders all four
skeletons and asserts the contract survived, and it runs before anything reaches disk.

## The slots file

Seven slots, loaded by `load_slots(path)` into a `PromptSlots` pydantic model. The right-hand
columns say which of the four prompts each slot actually reaches — several templates
deliberately ignore slots you might expect them to use.

| Slot | Type | Required | Default | select | score | verify | rewrite |
|---|---|---|---|---|---|---|---|
| `entity_noun` | `str` | yes | — | yes | yes | no | yes |
| `target_noun` | `str` | yes | — | yes | yes | no | yes |
| `domain_brief` | `str` | yes | — | yes | yes | yes | yes |
| `rubric` | `list[RubricRow]` | yes | — | no | yes | no | no |
| `hard_rules` | `list[str]` | no | `[]` | yes | yes | yes | no |
| `disambiguation_steps` | `str` | no | `""` | yes | no | no | no |
| `properties` | `list[PropertyQuestion]` | no | `[]` | no | no | no | no |

`properties` reaches none of the four skeletons: it is read only by the decider path, where
each `{name, question}` entry becomes one agreement question about the chosen candidate. See
[the `properties` block](decide.md#the-properties-block).

`RubricRow`:

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `score` | `float` | yes | — | the confidence this band assigns, in `[0, 1]` |
| `name` | `str` | yes | — | a short band label, e.g. `Certain` |
| `when` | `str` | yes | — | the condition under which the band applies |
| `example` | `str` | no | `""` | a concrete case; rendered as `(e.g. …)` after `when` |

### What `PromptSlots` enforces

Validation happens on construction, so `load_slots` raises before a bad file can be used.
pydantic wraps each message as `Value error, <message>`; the messages themselves are:

| Condition | Message |
|---|---|
| `rubric` is `[]` | `List should have at least 1 item after validation, not 0` |
| `rubric` has one row | `rubric needs at least 2 rows to be usable` |
| a score outside `[0, 1]` | `rubric score 1.5 must be between 0 and 1` |
| scores not strictly decreasing | `rubric scores must be strictly decreasing as declared` |
| a blank required noun | `entity_noun must be non-empty` |

Notes that matter in practice:

- The empty-list case is caught by the field constraint `min_length=1`, so it reports a
  pydantic `too_short` error rather than the friendlier two-row message. Anything with one
  row or more reaches `_check_rubric` and gets the prose message.
- "Strictly decreasing" means strictly: two rows sharing a score fail. The check compares
  each row with its successor in **declared** order, so a rubric that is merely unsorted
  fails too. `draft_slots` repairs that case before validation; a hand-written file does not
  get that courtesy.
- `entity_noun`, `target_noun` and `domain_brief` are stripped before the emptiness test, so
  a whitespace-only value fails. `hard_rules` and `disambiguation_steps` have no content
  rules at all — an empty list and an empty string are both normal.
- A missing `when` or `name` on a rubric row is an ordinary pydantic `Field required` error
  located at `rubric.0.when`.

### A complete slots.yaml

```yaml
# What one source record is, as a noun phrase. Reaches select, score and rewrite.
entity_noun: gene or protein mention
# What one target record is. Same three prompts.
target_noun: NCBI Gene record
# One clause naming the domain and its naming conventions. This is the only slot that
# reaches all four prompts, including verify and rewrite.
domain_brief: >-
  gene nomenclature across model organisms, where the same symbol names different genes
  in different species and case conventions differ between them

# Confidence bands. At least two rows, scores strictly decreasing, each in [0, 1].
# Rendered as a markdown table — and ONLY into the score prompt.
rubric:
  - score: 1.0
    name: Certain
    when: symbol matches exactly and the organism is confirmed by the context
    example: "human TP53 -> NCBIGene:7157 (Homo sapiens TP53)"
  - score: 0.9
    name: High
    when: an official symbol or alias matches and only one organism is plausible
    example: "Trp53 -> the mouse gene, since Trp53 is mouse-specific nomenclature"
  - score: 0.6
    name: Plausible
    when: the symbol matches but the organism is not stated anywhere in the context
    example: "bare 'p53' with no species mentioned"
  - score: 0.3
    name: Speculative
    when: only a description or partial name matches, not the symbol
    example: "'the tumour suppressor' -> TP53"

# Absolute rules. Reach select, score and verify — never rewrite. Optional.
hard_rules:
  - >-
    A gene symbol in one organism is a DIFFERENT entity from the same symbol in another
    organism. Never match across species without contextual evidence.
  - >-
    If no candidate is the right gene, abstain. A wrong gene mapping is indistinguishable
    from a right one downstream.

# Free text, rendered under "## How to disambiguate" — in the SELECT prompt only.
# Markdown headings are fine here as long as they are not reserved (see Gotchas).
disambiguation_steps: |
  ## Step 1: Determine the organism
  Scan the context for a species name, a cell line, a tissue, or an organism-specific
  nomenclature convention (all-caps for human, initial-caps for mouse).

  ## Step 2: Filter candidates by that organism
  Discard candidates from other organisms before comparing symbols.
```

Shipped examples to copy from: `examples/chemistry/slots.yaml`, `examples/chebi/slots.yaml`,
`examples/nlm_gene/slots.yaml`, `examples/ncbi_disease/slots.yaml`.

## PromptSet

`PromptSet` is a frozen dataclass pairing one `PromptSlots` with the four skeleton texts.

```python
@classmethod
def from_slots(cls, slots: PromptSlots, *, base_dir: Path | None = None) -> PromptSet
```

`base_dir` defaults to the packaged `xwalk/prompts/base/`; from it, `select.j2`, `score.j2`,
`verify.j2` and `rewrite.j2` are read as text. The skeleton *text* is stored on the
instance, so a `PromptSet` is self-contained after construction.

The four renderers are keyword-only and each returns a stripped string:

```python
def render_select(
    self, *, source_fields: Mapping[str, Any], context: str, candidate_block: str
) -> str

def render_score(
    self,
    *,
    source_fields: Mapping[str, Any],
    context: str,
    chosen_block: str,
    other_candidates: str,
    review_floor: float,
) -> str

def render_verify(
    self,
    *,
    source_fields: Mapping[str, Any],
    context: str,
    chosen_block: str,
    other_candidates: str,
) -> str

def render_rewrite(
    self,
    *,
    source_fields: Mapping[str, Any],
    context: str,
    previous_queries: Sequence[str],
    best_candidates: str,
    max_queries: int,
) -> str
```

Rendering uses `StrictUndefined` with `trim_blocks`, `lstrip_blocks` and
`keep_trailing_newline=False`, and the result is `.strip()`ed. Strict undefined means a
skeleton that references a variable nobody passes raises rather than rendering a blank —
which is what you want from a file that decides what a paid API call says.

In the matching loop these are called for you: `Selector` calls `render_select`, `Scorer`
calls `render_score` (passing `MatchPolicy.review_floor`), `Verifier` calls `render_verify`,
and `QueryRewriter` calls `render_rewrite` with its own `max_queries`, default `2`.

```python
@property
def fingerprint(self) -> str
```

A digest of the slots (as JSON) **and** the four skeleton texts, sorted by name. It feeds
`build_run_fingerprint`, so editing one word of a rubric invalidates a resumable run — which
is the intended behaviour, since the results were produced under different instructions.

## The four base templates

Every template receives `slots` plus the keyword arguments of its `render_*` method. Every
`## Context` block is conditional on a non-empty `context`; the other conditionals are named
per template below.

### `select.j2` — pick one candidate or abstain

- **Variables:** `source_fields`, `context`, `candidate_block`.
- **Slots used:** `entity_noun`, `target_noun`, `domain_brief`, `disambiguation_steps`,
  `hard_rules`. **Ignores `rubric` entirely** — the selector is never shown the bands.
- **Sections, in order:** an opening line naming both nouns; `Domain:`; `## Source record`
  (one `- key: value` line per entry of `source_fields`); `## Context` (if non-empty);
  `## Candidates`; `## How to disambiguate` (if `disambiguation_steps`); `## Hard rules` (if
  `hard_rules`); `## Rules`; `## Output`.
- **The key instruction** lives in `## Rules`: *"Never invent a key. Never answer with an
  identifier, a name, or a position number."* Candidate keys are opaque (`C01`, `C02`) and
  resolution is exact-only, so an answer that names a target ID resolves to
  `UNRESOLVED_OUTPUT` and routes to review.

```json
{"chosen_key": "C01", "confidence_score": 0.9, "explanation": "one sentence"}
```

### `score.j2` — an independent confidence for the chosen match

- **Variables:** `source_fields`, `context`, `chosen_block`, `other_candidates`,
  `review_floor`.
- **Slots used:** `entity_noun`, `target_noun`, `domain_brief`, `rubric`, `hard_rules`.
  **Ignores `disambiguation_steps`** — disambiguation already happened.
- **Sections:** opening line; `Domain:`; `## Source record`; `## Context` (if non-empty);
  `## Proposed match`; `## Other candidates that were available` (if `other_candidates`);
  `## Rubric` (a markdown table, one row per `RubricRow`, `example` appended as
  `(e.g. …)`); `## Hard rules` (if `hard_rules`); `## Rules`; `## Output`.
- `review_floor` is interpolated into a rule: *"If the score is below 0.4, you may propose
  better options."* Proposals are constrained — a candidate key only from the listed others,
  a query only for an entity no candidate names.

```json
{"confidence_score": 0.7, "explanation": "one sentence",
 "better_candidate_keys": ["C03"], "better_queries": ["alternative search string"]}
```

### `verify.j2` — a second, independent verdict

- **Variables:** `source_fields`, `context`, `chosen_block`, `other_candidates`.
- **Slots used:** `domain_brief` and `hard_rules`, and nothing else. **Ignores
  `entity_noun`, `target_noun`, `rubric` and `disambiguation_steps`** — the verifier is
  asked for a verdict, not a number to average, and it is deliberately not primed with the
  same framing the selector saw.
- **Sections:** opening line ("Do not assume the proposal is correct"); `Domain:`;
  `## Source record`; `## Context` (if non-empty); `## Proposed match`;
  `## Other candidates` (if `other_candidates`); `## Hard rules` (if `hard_rules`);
  `## Rules`; `## Output`.
- An unreadable verdict is treated as `disagree`, never as support.

```json
{"decision": "support", "preferred_key": null,
 "confidence_score": 0.8, "explanation": "one sentence"}
```

### `rewrite.j2` — propose better search queries

- **Variables:** `source_fields`, `context`, `previous_queries`, `best_candidates`,
  `max_queries`.
- **Slots used:** `entity_noun`, `target_noun`, `domain_brief`. **Ignores `hard_rules`,
  `rubric` and `disambiguation_steps`.**
- **Sections:** opening line; `Domain:`; `## Source record`; `## Context` (if non-empty);
  `## Queries already tried` (one `- query` line each); `## Best candidates those queries
  returned` (if `best_candidates`); `## Rules`; `## Output`.
- `max_queries` is interpolated into the rule text, and `QueryRewriter` also enforces it
  when reading the answer, dropping any query that normalises to one already tried.

```json
{"queries": ["first alternative", "second alternative"], "explanation": "one sentence"}
```

### The four JSON schemas

These are the `SELECT_SCHEMA`, `SCORE_SCHEMA`, `VERIFY_SCHEMA` and `REWRITE_SCHEMA` objects
exported from `xwalk.prompts`. All four set `additionalProperties: false` and are sent as
`response_format` when the adapter's capability profile says the provider supports it.

| Prompt | Property | Type | Required | Notes |
|---|---|---|---|---|
| select | `chosen_key` | `string \| null` | yes | `null` abstains |
| select | `confidence_score` | `number` | yes | `0 ≤ x ≤ 1` |
| select | `explanation` | `string` | yes | |
| score | `confidence_score` | `number` | yes | `0 ≤ x ≤ 1` |
| score | `explanation` | `string` | yes | |
| score | `better_candidate_keys` | `array[string]` | no | keys only, may be empty |
| score | `better_queries` | `array[string]` | no | may be empty |
| verify | `decision` | `string` | yes | `support` / `disagree` / `no_match` |
| verify | `explanation` | `string` | yes | |
| verify | `preferred_key` | `string \| null` | no | must be a key issued this attempt |
| verify | `confidence_score` | `number` | no | `0 ≤ x ≤ 1` |
| rewrite | `queries` | `array[string]` | yes | |
| rewrite | `explanation` | `string` | no | |

## validate_contract

```python
def validate_contract(prompts: PromptSet) -> None
```

Renders all four prompts against fixed fixtures, then asserts the contract survived. Raises
`ContractError` on the first violation; returns `None` on success. Run it before anything is
saved — `draft_slots` and `optimize_prompt` both do.

The fixtures are constants in `xwalk/prompts/contract.py`:

| Fixture | Value |
|---|---|
| `source_fields` | `{"mention": "glucose", "organism": "Homo sapiens"}` |
| `context` | `blood [glucose] levels were elevated` |
| `candidate_block` | `[C01] ID: T1 Label: glucose\n\n[C02] ID: T2 Label: fructose` |
| `chosen_block` | `[C01] ID: T1 Label: glucose` |
| `other_candidates` | `[C02] ID: T2 Label: fructose` |
| `review_floor` | `0.4` |
| `previous_queries` | `("glucose",)` |
| `best_candidates` | `[C01] ID: T1` |
| `max_queries` | `2` |

For each prompt, in this order:

1. **Rendered inputs appear exactly once.** select: `candidate_block`, `context`. score and
   verify: `chosen_block`, `context`. rewrite: `context`. This runs first on purpose — a
   skeleton that dropped a block should report *that*, not a downstream missing heading.
2. **Required headings appear exactly once.** select: `## Source record`, `## Candidates`,
   `## Output`. score: `## Source record`, `## Rubric`, `## Output`. verify:
   `## Source record`, `## Proposed match`, `## Output`. rewrite: `## Source record`,
   `## Output`. A prompt that renders `## Candidates` twice would show the model two lists.
3. **The select prompt still contains `Never answer with an identifier`.** Delete it and the
   model is free to answer with a target ID, which becomes `UNRESOLVED_OUTPUT` every time.
4. **The text after `## Output` parses as a JSON object** via the same `parse_json_object`
   used on real responses.
5. **That example carries every `required` key of the role's schema.**

Every `ContractError` it can raise:

| Trigger | Message |
|---|---|
| an input block missing or duplicated | `select prompt rendered candidate_block 0 times; it must appear exactly once` |
| a heading missing or duplicated | `score prompt contains section '## Rubric' 2 times; it must appear exactly once` |
| the key instruction deleted | `select prompt lost the key instruction ('Never answer with an identifier'); without it the model may answer with an identifier` |
| the output example does not parse | `rewrite output block is not a parseable example: <ParseError>` |
| the example is missing schema keys | `verify output example is missing keys: ['decision']` |

## Drafting slots

```python
async def draft_slots(
    llm: LLMClient,
    *,
    description: str,
    source_samples: Sequence[Record],
    target_samples: Sequence[Record],
    existing: PromptSlots | None = None,
    max_samples: int = 8,
    max_tokens: int = 2048,
) -> DraftResult
```

The model is handed a fixed instruction block, your `description`, up to `max_samples`
rendered source and target records, and — when `existing` is passed — the current slots as
JSON, to refine rather than replace. The request carries `SLOTS_SCHEMA`, whose required keys
are `entity_noun`, `target_noun`, `domain_brief` and `rubric`, with `minItems: 2` on the
rubric.

`DraftResult` is a frozen dataclass:

| Field | Type | Meaning |
|---|---|---|
| `slots` | `PromptSlots` | the validated result |
| `warnings` | `tuple[str, ...]` | what the repair pass had to fix |
| `usage` | `Usage` | prompt/completion tokens and call count for the draft |
| `raw` | `str` | the model's response text, unmodified |

**Repair rules**, applied to the parsed payload before `PromptSlots` sees it. Only what is
mechanically fixable is fixed; everything else is left to fail with a clear message.

| Situation | What happens |
|---|---|
| `rubric` is not a list | left alone; `PromptSlots` rejects it |
| a row has no numeric score (missing, string, or `bool`) | raises `ValueError: rubric row 'High' has no usable numeric score` |
| a score outside `[0, 1]` | clamped, warning `clamped rubric score 1.4 into [0, 1]` |
| rows not in descending score order | re-sorted, warning `rubric was not in strictly decreasing score order; reordered` |
| a non-dict row | passed through untouched, and skipped by the reorder check |

Reordering does not rescue ties: two rows at `0.9` sort fine and then fail the
strictly-decreasing rule. If the response cannot be reduced to a JSON object at all, the
error is `could not read slots from the drafting model: <ParseError>` followed by
`raw response:` and the first 1000 characters. After repair and validation,
`validate_contract(PromptSet.from_slots(slots))` runs, so a draft that would break a prompt
never becomes a `DraftResult`.

```python
def slots_diff(before: PromptSlots, after: PromptSlots) -> str
def write_slots(slots: PromptSlots, path: str | Path) -> None
```

`slots_diff` renders a field-level diff over the union of keys, in sorted order, skipping
fields that are equal; each changed field becomes three lines (`- key:`, `before:`,
`after:`) with JSON values. Identical slots give `""` — the CLI prints `(no changes)` in
that case. Show it before writing anything.

`write_slots` creates parent directories and writes YAML with `sort_keys=False` and
`allow_unicode=True`, so declaration order survives the round trip.

## Optimising slots

```python
async def optimize_prompt(
    *,
    matcher_factory: Callable[[PromptSet], Matcher],
    source_records: Sequence[Record],
    gold: GoldSet,
    initial: PromptSlots,
    optimiser_llm: LLMClient,
    work_dir: str | Path,
    partitioner: Partitioner = Partitioner(),
    partition_overrides: Mapping[str, Partition] | None = None,
    config: OptimizeConfig = OptimizeConfig(),
    progress: Callable[[str], None] | None = None,
) -> OptimizeReport
```

`matcher_factory` **must** thread its `PromptSet` argument into the matcher it builds. A
factory that ignores it re-runs the original slots every round, every metric still looks
plausible, and the optimiser measures nothing.

### OptimizeConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `role` | `PromptRole` | `PromptRole.SELECTOR` | which failures are mined and what the model is told to revise |
| `rounds` | `int` | `4` | maximum optimisation rounds |
| `patience` | `int` | `2` | consecutive non-improving rounds before stopping |
| `failures_per_round` | `int` | `12` | failure cases shown to the optimising model |
| `max_calls` | `int \| None` | `None` | hard budget; checked against the estimate before spending |
| `objective` | `str` | `"accepted_precision"` | attribute of `EvalReport` maximised on validation |
| `max_tokens` | `int` | `2048` | cap on the optimiser's own response |

`role` also selects which failures are worth showing. A selector learns nothing from a case
where the gold record was never retrieved — it never saw the right answer; a scorer learns a
great deal from exactly that case, because abstaining there was its job.

| Role | Failures selected | What the model is told to revise |
|---|---|---|
| `selector` | gold was presented and something else was chosen | `hard_rules`, `disambiguation_steps` |
| `scorer` | confident wrong accepts, needless abstentions, suppressed correct matches | `rubric`, `hard_rules` |
| `rewriter` | the first query did not surface the gold record | `domain_brief`, `disambiguation_steps` |
| `doc_template` | the gold record was never retrieved at all | `domain_brief` |

An unknown objective name scores `-1.0` rather than raising, so a typo in `objective` reads
as "nothing ever improves".

### The three partitions

Labelled records — those whose id appears in `gold` — are split by hashing the source id,
not at random, so adding labels never reshuffles an existing split. `Partitioner` defaults
to fractions `(0.5, 0.25, 0.25)` with salt `"xwalk"`; `partition_overrides` lets a published
split win over the hash.

- **prompt-train** supplies the failures shown to the optimising model.
- **validation** scores the baseline and every candidate, and decides which round is kept
  and when to stop.
- **test** is evaluated **once**, after the final slots are chosen.

Test is never reported per round. Revising a prompt from a partition's failures *and*
selecting the retained round on that same partition would make it training data; reporting
test each round makes test a second validation set — the identical mistake one level up.

### estimate_calls

```python
def estimate_calls(
    config: OptimizeConfig,
    n_prompt_train: int,
    n_validation: int,
    n_test: int,
    *,
    calls_per_record: int = 2,
) -> int
```

```
baseline  = n_validation * calls_per_record
per_round = (n_prompt_train + n_validation) * calls_per_record + 1
total     = baseline + config.rounds * per_round + n_test * calls_per_record
```

The `+ 1` is the optimising model's own call. Test is counted once, not once per round. The
estimate is printed through `progress` before anything is spent, and when `max_calls` is set
and the estimate exceeds it the run aborts before the first call with:

```
estimated 5300 calls exceeds max_calls=5000; reduce rounds, shrink the labelled set,
or raise the budget
```

### The loop

Each round runs prompt-train under the current best slots, selects up to
`failures_per_round` cases for the role, asks the optimiser for complete revised slots, and
puts the answer through `PromptSlots` validation and `validate_contract`. An unusable answer
is recorded as a round with `improved=False` and a warning — `round 2 produced unusable
slots: <error>` — and counts toward patience, but never ends the run by itself. A usable
answer is evaluated on validation; it is kept only if its objective is **strictly greater**
than the incumbent.

`stopped_because` is one of exactly three strings:

| Value | When |
|---|---|
| `completed 4 rounds` | the loop ran to `config.rounds` (the number is `config.rounds`) |
| `no failures for role selector on prompt-train` | nothing left to learn from for that role |
| `patience 2 exhausted` | `config.patience` consecutive rounds without improvement |

### RoundResult and OptimizeReport

| `RoundResult` | Type | Meaning |
|---|---|---|
| `round_index` | `int` | 1-based |
| `slots` | `PromptSlots` | the candidate, or the incumbent if the candidate was unusable |
| `validation` | `EvalReport` | the candidate's validation report, or the baseline if unusable |
| `improved` | `bool` | whether it beat the incumbent objective |
| `warnings` | `tuple[str, ...]` | why a round was discarded, if it was |
| `usage` | `Usage` | the optimiser call only, not the matching runs |

| `OptimizeReport` | Type | Meaning |
|---|---|---|
| `best_slots` | `PromptSlots` | the retained slots |
| `baseline` | `EvalReport` | the initial slots on validation |
| `rounds` | `tuple[RoundResult, ...]` | every round attempted |
| `test_report` | `EvalReport \| None` | the single test evaluation |
| `total_usage` | `Usage` | every call: baseline, per-round matching, optimiser, test |
| `stopped_because` | `str` | one of the three strings above |
| `partition_sizes` | `Mapping[str, int]` | labelled counts per partition |

`as_dict()` on either is JSON-ready; `OptimizeReport.as_dict()` renames `baseline` to
`baseline_validation` and `test_report` to `test`.

### What lands in the work directory

```
<work_dir>/
  round_01/slots.yaml        # the candidate, written even when it was discarded
  round_01/validation.json   # its EvalReport.as_dict()
  round_02/…
  best/slots.yaml            # the retained slots — copy this over your job's slots file
  report.json                # OptimizeReport.as_dict()
```

Rounds that produced unusable slots write no directory: there was nothing valid to write.
`best/slots.yaml` and `report.json` are always written, including when the loop stopped
before round one.

## Gotchas

**A slot's own text can trip the contract checks.** The contract counts literal substrings
in the rendered prompt, and slot text is part of that render. All three of these raise
`ContractError`:

| Slot content | What breaks |
|---|---|
| `hard_rules` containing `## Rubric` | `score prompt contains section '## Rubric' 2 times` |
| `disambiguation_steps` containing `## Candidates` | duplicate heading in select |
| any slot repeating a fixture string, e.g. `blood [glucose] levels were elevated` | `select prompt rendered context 2 times` |

Reserved headings are per prompt: `## Source record` and `## Output` everywhere,
`## Candidates` in select, `## Rubric` in score, `## Proposed match` in verify. Other
headings are not checked — a slot may safely contain `## Step 1` (as
`examples/nlm_gene/slots.yaml` does), and duplicating `## Rules` or `## Context` from a slot
is not caught at all.

**Some roles' slots never reach the prompt they seem to target.** The `rewriter` role brief
asks the optimising model to revise `domain_brief` *and* `disambiguation_steps`, but
`rewrite.j2` renders only `domain_brief` — a disambiguation edit made for the rewriter shows
up in the select prompt instead, and changes retrieval only indirectly. Likewise: the
`rubric` reaches nothing but `score.j2`, so editing bands to fix a selection failure changes
nothing the selector sees; and `verify.j2` reads only `domain_brief` and `hard_rules`, so
neither noun nor the rubric influences a verdict.

**`doc_template` optimisation does not fix the doc template.** That role revises
`domain_brief` so a human can read which target fields carry the searchable names. The
actual fix is in the job file's `templates.doc`, and xwalk will not make it for you.

**A `|` in a rubric cell breaks the rendered table.** `when` and `example` are interpolated
straight into a markdown table row. The contract does not check table shape.

**The fingerprint covers the skeleton text too.** Passing `base_dir` to `from_slots` changes
`PromptSet.fingerprint`, hence the run fingerprint, hence resume: an existing run directory
will re-match every record.

**A custom `base_dir` only replaces skeleton text, not the Jinja search path.** Rendering
always constructs its environment with a loader rooted at the packaged base directory, so an
`{% include %}` inside your own skeleton resolves against the package, not against your
directory.

**`prompts draft` does not update the job file.** It writes wherever `--out` points, and
pointing `prompts.slots` at the new file is your job.

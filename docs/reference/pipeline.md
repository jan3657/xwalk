# The matching pipeline

`Matcher.match` turns one source record into one `MatchResult`. Everything between is a
loop over *attempts*: retrieve candidates, ask a model to pick one, ask a second prompt
to score the pick, sometimes buy a third opinion, and mine the score response for leads
to try next. The matcher itself is orchestration only — every decision lives in a stage
(`xwalk.stages.*`) or in `MatchPolicy`. This page documents the loop, the stages, and
the batch layer (`xwalk.batch`) that runs the loop over a whole source.

Modules covered: `xwalk.matcher`, `xwalk.batch`, `xwalk.stages.keying`,
`xwalk.stages.proposals`, `xwalk.stages.select`, `xwalk.stages.gate`,
`xwalk.stages.rewrite`.

## Control flow

### Setup

`match(source)` renders two strings from the `TemplateSet`: the *context* (shown to
every prompt) and the *first query* (fed to retrieval). It then initialises two queues:

- **query queue** — `(query, proposal)` pairs waiting for a fresh retrieval. Starts
  with the first query. FIFO, except that an infrastructure failure re-inserts its own
  query at the front.
- **candidate queue** — candidate keys (e.g. `C03`) proposed by the scorer, waiting to
  be re-examined. These records are already in hand; consuming one runs **no retrieval
  and no selector call** — the loop goes straight to scoring.

The candidate queue always drains first. Re-examining a record you already retrieved is
strictly cheaper than a new search, and the paper repo's mistake — pushing candidate
labels back into the query queue and re-retrieving records already in hand — is exactly
what the split exists to prevent.

Alongside the queues: `seen_queries` holds normalised forms of every query ever
enqueued (deduplication), `tried_queries` keeps original casing for the rewriter's
prompt, and the last non-empty `KeyedCandidates` is retained so candidate proposals can
be resolved against the keys that were actually issued.

### The attempt loop

Up to `policy.max_attempts` iterations (default 4). Each iteration:

```
match(source)
  render context + first query; queue = [first query]
  for index in range(max_attempts):
    pick work: candidate queue first, else query queue, else STOP (no work)
    attempt:
      1. retrieve   all retrievers concurrently, RRF-fused   (no LLM; skipped
                    when re-examining a candidate)
      2. select     budget -> assign keys -> one key or null  (LLM; skipped when
                    re-examining a candidate — the key is forced)
      3. score      chosen record vs the full source record   (LLM; always runs
                    on a resolved choice)
      4. verify     only if score is in verify_band, or on    (LLM, conditional)
                    an audit draw above it
      5. route      scorer proposals -> candidate keys + new queries
    fatal LLM error          -> record a failed attempt, STOP
    acceptable match         -> STOP (early accept)
    attempts exhausted       -> STOP
    infrastructure failure   -> re-queue the same query at the front, continue
    push candidate keys onto candidate queue, new queries onto query queue
    both queues empty        -> rewrite (LLM); nothing new -> STOP
  derive_status(attempts) -> MatchResult
```

Within an attempt (`Matcher._attempt`):

1. **Retrieval.** Every retriever is searched concurrently, each guarded by
   `policy.retriever_timeout`. Depth comes from each retriever's own `default_limit`;
   the matcher's `retriever_limit` is only the fallback for a backend that does not
   declare one. Results are merged by reciprocal rank fusion with constant `rrf_k`.
   Failed or timed-out retrievers become degradation notes on the attempt. Two empty
   outcomes are distinguished: if *every* retriever failed, the attempt is tagged
   `RETRIEVER_FAILURE` — a dead index is not evidence of a non-match; if retrieval
   worked and found nothing, it is `NO_CANDIDATES`.
2. **Selection** (LLM). Skipped entirely for a candidate re-examination — the forced
   key resolves directly through the retained `KeyedCandidates`, recorded as
   `EXACT_KEY`. Otherwise `Selector.select` trims the fused list, issues opaque keys,
   and asks the model for one key or null. An abstention or unresolvable answer ends
   the attempt without a scorer call — neither is worth one.
3. **Scoring** (LLM). `Scorer.score` judges the chosen record against the **whole
   source record**, never the retrieval query. If the gate produces no usable number,
   the attempt is downgraded to `UNRESOLVED`: falling back to the selector's
   self-reported confidence would let malformed output become an automatic `MATCHED`
   on a score the gate never gave. Malformed output routes to review instead.
4. **Verification** (LLM, conditional). Bought only when `should_verify` says the
   score falls inside `policy.verify_band`, or when `should_audit` samples this
   otherwise-automatic high-confidence result (deterministic in
   `(run_fingerprint, source_id)`, so audits survive resumes). An audit verdict is
   honoured exactly like any other verdict.
5. **Routing.** The scorer's proposals are split by `route_proposals` into candidate
   keys and genuinely new queries; everything else is dropped with a recorded reason
   and surfaces on the attempt as `dropped_proposals`.

Every stage call goes through one helper that books its usage (a call that raised
counts as a call with unknown usage) and classifies provider errors. `Matcher.match`
never raises for a provider error:

| Error | Stage | Outcome |
|---|---|---|
| recoverable (`LLMError` other than fatal, or a timeout) | selector, scorer | attempt `reason=PROVIDER_FAILURE`; the same query is retried |
| recoverable | gating verifier (score in `verify_band`) | the score is kept, `verifier_decision="error"`, `reason=PROVIDER_FAILURE`; retried; an unverified in-band score is never `MATCHED` |
| recoverable | audit verifier | decision unchanged; the error is noted on the attempt |
| recoverable | rewriter | the loop ends; status comes from the attempts so far; error noted on the last attempt |
| `LLMFatalError` (bad key, unknown model, invalid request) | any stage | the record is `FAILED` with `reason=FATAL_PROVIDER_FAILURE`; the loop stops |

Malformed model output is not a provider error: it is handled inside the stage and
routes to review.

### What triggers another attempt

After each attempt the loop stops early if the attempt is *acceptable*: a primary score
`>= policy.accept_at`, resolved as `EXACT_KEY`, from an attempt with no failure
reason, whose verifier decision (if any) is `"support"`. Otherwise, if attempts remain:

- An infrastructure failure (`RETRIEVER_FAILURE`, `PROVIDER_FAILURE`) re-inserts the
  *same* query at the front of the query queue. An outage is not evidence about this
  record, and the rewriter is not asked to invent a new query for one.
- Routed candidate keys extend the candidate queue; routed queries are appended to the
  query queue (and their normalised forms to `seen_queries`).
- If both queues are then empty, `QueryRewriter.rewrite` is called with every query
  tried so far and the last keyed candidate list. Its proposals are deduplicated
  against `seen_queries`; survivors are enqueued.

### Every way the loop terminates

1. **Early accept** — an acceptable attempt (see above).
2. **Fatal LLM error** — `LLMFatalError` in any stage (including the verifier and
   rewriter) marks the attempt `FATAL_PROVIDER_FAILURE` and stops everything.
3. **Attempts exhausted** — `policy.max_attempts` iterations have run.
4. **Rewriter exhausted** — both queues empty and the rewriter proposed nothing new.
5. **No work at the top of an iteration** — both queues empty when picking work
   (a defensive twin of 4).

### After the loop

`derive_status` (in `xwalk.policy`) reduces the attempt list to one
`(status, reason, best_attempt)`. Any `FATAL_PROVIDER_FAILURE` attempt makes the result
`FAILED` with that reason. Only a finite score in `[0, 1]` counts. Highest score wins;
on a tie an attempt that completed beats one whose verifier call failed, then the
earlier attempt wins. A best attempt with `verifier_decision="error"` is
`NEEDS_REVIEW` with reason `PROVIDER_FAILURE`.
`matched_id` is cleared for `UNMATCHED` and `FAILED`. The matched record is looked up
in the store — a missing id is tolerated, leaving `matched_record` as `None`. The
result carries a `result_key` derived from `(run_fingerprint, source.id,
hash_record(source))`, the full attempt trace, and summed `Usage`.

## Matcher

```python
class Matcher:
    def __init__(
        self,
        *,
        templates: TemplateSet,
        retrievers: Sequence[Retriever],
        store: TargetStore,
        selector: Selector,
        scorer: Scorer,
        verifier: Verifier,
        rewriter: QueryRewriter,
        policy: MatchPolicy | None = None,
        run_fingerprint: str = "",
        retriever_limit: int = 20,
        rrf_k: int = 60,
        keep_candidates_in_trace: bool = True,
    ) -> None: ...
```

An empty `retrievers` sequence raises `ValueError`. `policy=None` means a default
`MatchPolicy()`. `retriever_limit` is the fallback depth for retrievers without a
`default_limit`; `rrf_k` is the reciprocal-rank-fusion constant.
`keep_candidates_in_trace=False` drops the candidate tuples from attempts and the
result — the counts survive, the records do not.

| Member | Kind | Returns |
|---|---|---|
| `run_fingerprint` | property | the fingerprint string passed at construction |
| `policy` | property | the effective `MatchPolicy` |
| `store_fingerprint` | property | `store.fingerprint` |
| `match(source: Record)` | async method | `MatchResult` |
| `match_sync(source: Record)` | method | `MatchResult`, via `asyncio.run` |

## Candidate keying

Candidates are shown to the model under temporary keys — `C01`, `C02`, ... — assigned
per attempt, and the model returns a key or null. Resolution is an exact dictionary
lookup against the keys issued *for that attempt*; anything else is `UNRESOLVED`.

The keys are opaque because heuristic ID resolution is unsafe as a default. Its worst
branch — treating a bare integer as a candidate rank — turns a hallucinated ID into a
different, *real* mapping whenever target IDs are numeric. Opaque keys also remove the
sentinel muddle: `null` is unambiguous where `"-1"` and `"0"` are values some ID scheme
legitimately uses.

```python
def assign_keys(candidates: Sequence[Candidate], templates: TemplateSet) -> KeyedCandidates
```

Assigns `C01, C02, ...` in the order given, zero-padded to
`max(2, len(str(len(candidates))))` digits, and renders the candidate block: one
`[Ckk] <candidate template>` block per record, kept separately by key. An empty input
yields an empty `KeyedCandidates`.

`KeyedCandidates` (frozen dataclass; `len()` gives the candidate count):

| Field | Type | Meaning |
|---|---|---|
| `order` | `tuple[str, ...]` | issued keys, in presentation order |
| `by_key` | `Mapping[str, Candidate]` | key -> full candidate |
| `issued` | `Mapping[str, str]` | key -> record id |
| `blocks` | `Mapping[str, str]` | key -> that candidate's rendered block |

`rendered` (property) joins every block in key order with blank lines;
`render_except(key)` joins all blocks but one. The scorer and verifier select the
chosen block by key, never by re-splitting the joined text, so a rendered record that
itself contains blank lines keeps all of its text in its own block.

```python
def resolve_key(
    raw: str | None,
    keyed: KeyedCandidates,
    *,
    legacy: bool = False,
) -> ResolvedChoice
```

`ResolvedChoice` carries `record_id: str | None`, `resolution: Resolution`, and the
untouched `raw` answer. The ladder, in order:

1. `raw is None` -> `ABSTAIN`.
2. Normalise: strip whitespace, then trim quotes, backticks, brackets, parentheses,
   and trailing punctuation (`.,;:`), then strip again — so `[C03]`, `"C03"`, and
   `C03` all normalise alike.
3. Case-insensitive abstain tokens -> `ABSTAIN`: the empty string, `null`, `none`,
   `nil`, `no_match`, `nomatch`, `no match`, `n/a`, `na`.
4. Uppercase and look up in `keyed.issued` -> `EXACT_KEY`. Upper-casing is safe because
   the key space is ours: it cannot merge two distinct records.
5. **Strict mode stops here.** Anything else -> `UNRESOLVED`, `record_id=None`.

`legacy=True` re-enables heuristic ID resolution for reproducing prior work, adding:

6. Case-sensitive match against an issued record id -> `LEGACY_EXACT_ID`.
7. Case-insensitive id match -> `LEGACY_FUZZY`.
8. Match against the id suffix after the last `":"` (CURIE local part) ->
   `LEGACY_FUZZY`.
9. All-digit text in `1..len(candidates)` treated as a 1-based rank -> `LEGACY_RANK`.
10. Otherwise -> `UNRESOLVED`.

A malformed answer must never resolve to a real record. In strict mode it cannot: the
only path to a `record_id` is an exact issued-key hit. In legacy mode every non-exact
path carries a `LEGACY_*` tag, and `derive_status` forces any resolution other than
`EXACT_KEY` to `NEEDS_REVIEW` — a legacy resolution never becomes an automatic match.
`UNRESOLVED` always has `record_id=None`; the malformed text is preserved in `raw` for
the trace, and nothing downstream treats it as a choice.

## The stages

Every stage that parses model output has the same shape: an outcome dataclass with the
raw response, `Usage`, and an `error` field, plus a fail-safe branch for unparseable
output. No stage raises on bad JSON.

### Selector

```python
@dataclass(frozen=True)
class SelectorPolicy:
    max_candidates: int = 30
    max_candidate_tokens: int = 8_000
```

This is not retrieval depth — each retriever owns its own `k`. With N retrievers the
fused list grows without bound; `SelectorPolicy` is the separate constraint that
governs how much of it reaches the model.

```python
def apply_budget(
    candidates: Sequence[Candidate], policy: SelectorPolicy
) -> tuple[list[Candidate], int]
```

Deterministic trim: sort by `(-fused_score, id)`, keep the first `max_candidates`,
then walk a character budget of `max_candidate_tokens * 4` (a deliberately crude
chars-per-token estimate — a guard rail, not a meter) using a cheap size proxy over the
record's field values. The first candidate is always kept, even over budget. Returns
`(kept, dropped_count)`.

```python
class Selector:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        policy: SelectorPolicy | None = None,
        legacy_id_resolution: bool = False,
        system: str = "You return JSON only. No prose, no code fences.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None: ...

    async def select(
        self,
        source: Record,
        context: str,
        candidates: Sequence[Candidate],
    ) -> SelectionOutcome: ...
```

`SelectionOutcome`:

| Field | Type | Meaning |
|---|---|---|
| `choice` | `ResolvedChoice` | resolved record id (or refusal) plus resolution tag |
| `confidence` | `float \| None` | model self-report; `None` unless a finite number in `[0, 1]` |
| `explanation` | `str` | model explanation |
| `raw` | `str` | the raw response text |
| `keyed` | `KeyedCandidates` | the keys actually issued this attempt |
| `truncated` | `int` | candidates cut by `apply_budget` |
| `usage` | `Usage` | tokens and calls |
| `error` | `str \| None` | default `None` |
| `finish_reason` | `str \| None` | default `None` |

Fail-safes: an empty budgeted list returns an `ABSTAIN` outcome without calling the
LLM. A `ParseError` returns `UNRESOLVED` with the raw text as `choice.raw` and the
parse error in `error`. A non-string `chosen_key` is coerced with `str()` before
resolution. An `EXACT_KEY` choice whose `confidence_score` is invalid (see
"Confidence values" below) is *demoted to* `UNRESOLVED` — a choice we cannot score is a
choice we cannot classify.

#### Confidence values

Every stage validates `confidence_score` with `xwalk.llm.parsing.parse_confidence`.
Only a finite JSON number in `[0, 1]` is accepted. Missing/`null`, strings (even
`"0.9"`), booleans, `NaN`, `Infinity`, `-Infinity` and finite numbers outside `[0, 1]`
are invalid, with the reason in `error`; nothing is clamped or coerced. An invalid
scorer confidence routes the attempt to review (`UNRESOLVED_OUTPUT`); it can never
pass `accept_at`.

#### Generation parameters

`temperature` and `max_tokens` on every stage default to `None`, which sends the
client's configured values (from the job's `llm.*` or the client constructor). Pass a
value only to override it for that stage.

### Scorer

```python
class Scorer:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        review_floor: float = 0.4,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None: ...

    async def score(
        self,
        source: Record,
        context: str,
        keyed: KeyedCandidates,
        chosen_key: str,
    ) -> ScoreOutcome: ...
```

`review_floor` is passed into the prompt so the rubric and the policy threshold agree.
`score` takes the chosen candidate's block and the other blocks by key (a
`chosen_key` that was never issued raises `KeyError`) and judges the pick against the
whole source record.

`ScoreOutcome`:

| Field | Type | Meaning |
|---|---|---|
| `score` | `float \| None` | a finite number in `[0, 1]`; `None` when invalid or unusable |
| `explanation` | `str` | model explanation |
| `proposals` | `tuple[RetryProposal, ...]` | candidate-key and query leads |
| `raw` | `str` | raw response text |
| `usage` | `Usage` | tokens and calls |
| `error` | `str \| None` | default `None` |
| `finish_reason` | `str \| None` | default `None` |

Fail-safes: a `ParseError` yields `score=None`, no proposals, and the parse error. A
`confidence_score` that is not a finite number in `[0, 1]` yields `score=None` with
the reason in `error`. Proposal cleaning: non-lists and
non-strings are ignored; values are stripped and deduplicated; a proposed candidate key
that was not issued this attempt is discarded outright — naming a key we never issued
is a hallucination, not a lead. Candidate keys are upper-cased; queries are kept as
written. All proposals carry `source="scorer"`.

### Verifier

```python
class Verifier:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None: ...

    async def verify(
        self,
        source: Record,
        context: str,
        keyed: KeyedCandidates,
        chosen_key: str,
    ) -> VerifierVerdict: ...
```

A second, independent opinion. It returns a verdict, not a number to average.

`VerifierVerdict` (`Decision = Literal["support", "disagree", "no_match"]`):

| Field | Type | Meaning |
|---|---|---|
| `decision` | `Decision` | the verdict |
| `preferred_key` | `str \| None` | an issued key the verifier likes better |
| `confidence` | `float \| None` | `None` unless a finite number in `[0, 1]` (an invalid one is noted in `error`) |
| `explanation` | `str` | model explanation |
| `raw` | `str` | raw response text |
| `usage` | `Usage` | tokens and calls |
| `error` | `str \| None` | default `None` |
| `finish_reason` | `str \| None` | default `None` |

Fail-safes: a `ParseError` returns `decision="disagree"` — an unreadable second
opinion is not a supporting one. An unrecognised decision string also becomes
`"disagree"`, with the original recorded in `error`. `preferred_key` is kept only if,
stripped and upper-cased, it is a key issued this attempt; anything else is dropped.

### QueryRewriter

```python
class QueryRewriter:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        max_queries: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None: ...

    async def rewrite(
        self,
        source: Record,
        context: str,
        previous_queries: Sequence[str],
        keyed: KeyedCandidates,
    ) -> RewriteOutcome: ...
```

Produces query proposals only — never candidate proposals. `RewriteOutcome`:

| Field | Type | Meaning |
|---|---|---|
| `proposals` | `tuple[RetryProposal, ...]` | `kind="query"`, `source="rewriter"` |
| `explanation` | `str` | model explanation |
| `raw` | `str` | raw response text |
| `usage` | `Usage` | tokens and calls |
| `error` | `str \| None` | default `None` |

Fail-safes: a `ParseError` returns no proposals plus the error. Non-string entries are
skipped; blank queries and queries whose normalised form matches a previous query (or
an earlier proposal in the same response) are skipped; output is capped at
`max_queries`.

## Retry proposals

```python
def normalise_query(text: str) -> str
```

The comparison form for query deduplication: strip, lowercase, collapse internal
whitespace to single spaces. Enqueued queries keep their original casing; only the
normalised form is compared.

```python
def route_proposals(
    proposals: Sequence[RetryProposal],
    issued_keys: Sequence[str],
    seen_queries: Set[str],
) -> RoutedProposals
```

`seen_queries` holds *already-normalised* strings. `RoutedProposals`:

| Field | Type | Meaning |
|---|---|---|
| `candidate_keys` | `tuple[str, ...]` | issued keys to re-examine, upper-cased |
| `queries` | `tuple[str, ...]` | genuinely new searches, original casing |
| `dropped` | `tuple[tuple[RetryProposal, str], ...]` | each drop with its reason |

Two objects share the name "retry lead", and conflating them causes real waste: a
candidate proposal means "look again at a record already in hand" (no new retrieval),
a query proposal means "run a genuinely new search". A proposal is dropped when:

- its value is blank after stripping — reason `"blank value"`;
- it is a candidate key that was not issued this attempt — an invented key is
  recorded, never promoted to a query;
- it is a duplicate candidate key within the same routing call;
- it is a query whose normalised form is in `seen_queries` or already appeared earlier
  in the same call — reason `"query already tried"`.

Drops are never silent: they land in the attempt trace as `dropped_proposals`.

## Batch running

```python
async def run_batch(
    matcher: Matcher,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    limit: int | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    fingerprint_components: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport

def run_batch_sync(...same arguments...) -> BatchReport
```

`run_batch_sync` is `asyncio.run(run_batch(...))`. The run directory is created if
missing. One invocation:

1. **Reads the source once, as a stream.** Every record joins the invocation's
   *snapshot*, its ordered `(source_id, source_hash)` list, whether or not it is
   processed. A repeated source id raises `DuplicateSourceIdError` (a `ValueError`).
2. **Schedules unfinished records** on a bounded pool of at most `policy.concurrency`
   record tasks, fed from the source as slots free up. A record is *finished* when the
   ledger already holds a non-`failed` result for its key (see Resume). `limit` caps
   how many unfinished records this invocation processes; the source is still read to
   the end and the rest stay *pending*.
3. **Commits each result in its own transaction** as soon as it is ready, and the
   snapshot in one transaction once the source has been read to the end. `progress` is
   called once per freshly committed result, never for records skipped by resume.
4. **Exports the current view** (below) and the manifest, then closes the ledger.

A result with reason `fatal_provider_failure` is **never committed**: it aborts the run
and resume retries the record. On an abort (fatal result, or an exception in a record
task or the ledger) and on an interruption (task cancellation, Ctrl-C), `run_batch`
stops scheduling, cancels every in-flight record task and waits for each to finish,
discards their uncommitted results, writes the exports and the manifest with the run
state, and only then closes the ledger. Nothing writes to the ledger after it is closed
(`Ledger` raises `LedgerClosedError` if anything tries). A fatal provider failure
returns a report with `run_state=ABORTED` and the error in `errors`; any other exception,
and the cancellation itself, is re-raised after cleanup. When the source was not read to
the end, no snapshot is recorded for that invocation and the exports keep the previous
snapshot (or, for a first run, show what was committed).

### Run states

`BatchReport.run_state` and `manifest.json` `run_state` (a `RunState` string enum),
highest precedence first:

| State | Meaning |
|---|---|
| `interrupted` | cancelled or Ctrl-C; committed results are kept |
| `aborted` | a fatal provider failure or an exception stopped the run |
| `failed` | finished, but some current results are `failed`; resume retries them |
| `partial` | finished, but some current sources are pending (`limit`) |
| `complete` | every current source has a result and none is `failed` |

The CLI maps `aborted`/`failed` to exit code 3 and `partial` to 1.

### Current view and history

The **current view** has exactly one entry per source in the latest recorded snapshot:
the result whose key matches the source's current content, or a pending entry. Every
export, `BatchReport` method, the duplicate-target report, review export and the
adjudicated view read it. **History** is every result ever committed for the run
fingerprint and is never deleted: `Ledger.iter_history(run_fp)` and
`export_history_jsonl(ledger, run_fp, path)` expose it, each entry marked `current` or
not, with its revision.

| Case | Current view | History |
|---|---|---|
| source edited, same id | one row, for the new content | old version kept |
| source removed | absent (counted in manifest `removed_sources`) | kept |
| `limit=N` | processed rows plus `pending` rows for the rest | — |
| `failed` result, resumed | the retried result replaces it | the failure kept as an earlier revision |
| `resume=False` | recomputed results | earlier revisions kept |

A ledger without a snapshot for the run (a 0.1.1 ledger, or results written with
`Ledger.put_result` directly) uses the latest committed result per source id as its
current view.

Files written to the run directory:

| File | Contents |
|---|---|
| `ledger.sqlite` | the source of truth: every committed `MatchResult` and its earlier revisions, source snapshots, invocations, reviews and the run manifest. Written incrementally. |
| `results.jsonl` | one JSON object per current result (`result_to_dict`), full trace included. Pending sources have no line. |
| `mapping.csv` | the deliverable: one row per current source, columns `MAPPING_COLUMNS`; pending sources have status and reason `pending` and blank values. |
| `manifest.json` | the stored manifest (run fingerprint, library version, target fingerprint, `manifest_extra`, `fingerprint_components` when given) plus the latest invocation's `run_state`, `usage`, `errors` and `limit`, current `counts` (with `pending`), `duplicate_targets`, `removed_sources`, `history.results`, `snapshot` and `ledger_schema_version`. |

`usage` in the manifest carries `calls`, `prompt_tokens`, `completion_tokens`,
`unknown_calls`, `cache_hits` and `tokens`, the display form from
`Usage.describe_tokens()` that never shows unknown usage as zero.

### Resume

Each record's identity under this configuration is
`result_key(run_fingerprint, record.id, hash_record(record))` — the parts are hashed
as a list, never concatenated, so keys cannot collide by string arithmetic. With
`resume=True` (the default), a record whose key already has a ledger row with a status
other than `failed` is skipped without any retrieval or LLM call; a `failed` record is
retried. Resume is invalidated — the record re-runs — when any of the three parts
changes:

- the **run fingerprint** (anything fed to `build_run_fingerprint`);
- the **source id**;
- the **source content** — `hash_record` digests the record's id *and* fields.

`resume=False` recomputes every record. Repeating an unchanged complete run makes no
LLM calls and rewrites identical exports.

### Ledger schema

The ledger records `schema_version` (currently 2) in a `ledger_meta` table. A 0.1.1
ledger (version 1, no such table) is upgraded on open: the file is first copied to
`ledger.sqlite.v1-backup`, then the upgrade adds tables and defaulted columns in one
transaction. Nothing existing is rewritten or dropped. All 0.1.1 rows become history,
and the current view is inferred as the latest row per source id. A ledger with a newer
schema version, an SQLite file that is not a ledger, or a file that is not SQLite raises
`UnsupportedLedgerError` and is not modified.

### BatchReport

```python
@dataclass(frozen=True)
class BatchReport:
    run_fingerprint: str
    out_dir: Path
    total: int
    usage: Usage
    _ledger_path: Path
    run_state: RunState = RunState.COMPLETE
    pending: int = 0
    errors: tuple[BatchError, ...] = ()
```

`total` is the number of sources in the current view, `pending` how many of them have
no result yet. `usage` is only what this invocation spent on records it finished
(including a fatal one); resumed records contribute nothing, and calls made by records
cancelled mid-flight are not included. `errors` holds `BatchError(code, message,
source_id)` entries (`fatal_provider_failure`, or `exception`). The private
`_ledger_path` lets the methods reopen the ledger on demand, so the report stays usable
after `run_batch` returns:

| Method | Returns |
|---|---|
| `by_status()` | `dict[MatchStatus, int]` — current result counts per status |
| `needs_review()` | `list[MatchResult]` — every current result with status `NEEDS_REVIEW` |
| `duplicate_targets()` | `dict[str, list[str]]` — target id -> sorted source ids, for every target chosen by more than one current source |

### MAPPING_COLUMNS

```python
MAPPING_COLUMNS = (
    "source_id", "matched_id", "confidence", "status", "reason", "explanation",
    "attempts", "prompt_tokens", "completion_tokens", "llm_calls", "elapsed_seconds",
    "unknown_calls", "cache_hits",
)
```

`unknown_calls` and `cache_hits` were appended in 0.2, after the 0.1 columns.

### Export functions

```python
def export_results_jsonl(ledger: Ledger, run_fingerprint: str, path: str | Path) -> int
def export_history_jsonl(ledger: Ledger, run_fingerprint: str, path: str | Path) -> int
def export_mapping_csv(
    ledger: Ledger, run_fingerprint: str, path: str | Path, *, use_review: bool = False
) -> int
def export_manifest(ledger: Ledger, run_fingerprint: str, path: str | Path) -> None
```

The row-writers return the number of rows written. `export_mapping_csv` with
`use_review=True` writes the adjudicated view instead of raw model output: same
columns, `matched_id` and `status` reflect review decisions, `reason` is `"reviewed"`
for human-adjudicated rows (the model status otherwise), `explanation` carries the
review note, and the cost columns are blank. `export_manifest` regenerates the manifest
from the ledger alone and writes sorted, indented JSON.

## build_run_fingerprint

```python
def build_run_fingerprint(
    *,
    templates: TemplateSet,
    prompts: PromptSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    llm: LLMClient,
    policy: MatchPolicy,
    selector_policy: SelectorPolicy,
    retriever_limit: int = 20,
    rrf_k: int = 60,
) -> str
```

Everything whose change should invalidate prior results, hashed into one digest.
`run_fingerprint_components(...)` (same arguments) returns the unhashed dict, which
`run_batch(fingerprint_components=...)` stores in the manifest:

- the library version;
- `templates.fingerprint` and `prompts.fingerprint`;
- `store.fingerprint` (the target data);
- every retriever's `name:fingerprint`, sorted;
- retrieval shape: per-retriever depths (`default_limit`, falling back to
  `retriever_limit`), `retriever_limit` itself, `rrf_k`, and
  `policy.retriever_timeout`. Depth and the fusion constant change which candidates
  exist at all — raising `k` from 20 to 100 must not silently reuse old results;
- `llm.fingerprint`;
- policy classification and loop shape: `max_attempts`, `accept_at`, `review_floor`,
  `verify_band`, `audit_rate`, `legacy_id_resolution`;
- selector budget: `max_candidates`, `max_candidate_tokens`;
- `encoders` (only when a retriever exposes `encoder_identity`, as `DenseRetriever`
  does): the live encoder's name, dimension and `settings` — for
  `SentenceTransformerEncoder` the revision (`"unknown"` unless pinned), `normalize`,
  `query_prefix`, `doc_prefix` and `max_seq_length`.

Deliberately excluded: credentials (API keys, credential-like LiteLLM keyword
arguments and headers, userinfo or key parameters in a base URL), output paths, and
`policy.concurrency` — none of them change what a result means.
`Matcher.keep_candidates_in_trace` is likewise not an input: it changes trace
verbosity, not the decision. `retriever_limit` and `rrf_k`
must match the values handed to `Matcher`, or a resumed run will reuse results
produced at another retrieval depth.

## Gotchas

- **Non-exact resolutions never auto-match.** `derive_status` sends every resolution
  other than `EXACT_KEY` — including every `LEGACY_*` tag — to `NEEDS_REVIEW`,
  regardless of score.
- **The scorer judges the source record, not the query.** A retrieval query is a lossy
  projection; scoring against it would grade the match on the wrong evidence.
- **A candidate re-examination skips both retrieval and the selector.** The forced key
  is resolved as `EXACT_KEY` against the *retained* `KeyedCandidates`; that object is
  kept precisely because `rendered` and `by_key` cannot be reconstructed from a
  key->id map, and the scorer needs both.
- **Infrastructure failures still consume attempt slots.** A retriever or provider
  outage retries the same query, but each retry is one of `max_attempts`.
- **All-retrievers-failed is not "no candidates".** The former is `RETRIEVER_FAILURE`
  (and yields `FAILED` if every attempt is one); only the latter is evidence of a
  non-match.
- **Audit verdicts count.** Sampling decides *which* high-confidence results get a
  second opinion, never whether that opinion is honoured. The draw is seeded from
  `(run_fingerprint, source_id)`, so audits are reproducible across resumes.
- **`Attempt.finish_reason` is the last provider call's.** Each LLM-facing stage
  overwrites it; the survivor is the call that ended the attempt.
- **`Attempt.error` doubles as degradation notes.** When no stage errored, it carries
  the joined retriever failure notes, if any.
- **Ties in `derive_status` go to the earlier attempt**, and the highest-scoring
  attempt wins even if a later one scored lower.
- **`apply_budget` always keeps at least one candidate**, even when that single
  candidate exceeds the character budget.
- **`matched_record` can be `None` while `matched_id` is set.** The store lookup
  tolerates a missing id rather than failing the result.
- **`BatchReport.total` and `BatchReport.usage` measure different things** — the
  current view's size versus this invocation's spend.

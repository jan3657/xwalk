# Decision models

The second matching path. Everything on this page lives under `xwalk.decide` and
`xwalk.stages`, and none of it replaces the LLM path — a job picks one or the other with
`llm:` or `decider:`, and [the job file](job-file.md) rejects a file that declares both or
neither.

## What a decision model is

A decision model answers typed questions about a state and returns probabilities, never
text. You hand it a JSON-shaped state and a named set of questions — a yes/no `Noul`, a
`Choice` over at most 255 options, or a `Score` over 2 to 10 ordered levels — and it
returns one answer per question, each of them numbers: a `Noul` answers with the
probability that the answer is yes, a `Choice` with an option key plus a probability
distribution over the options and a confidence, and a `Score` with a level index (`0`
worst, `rubric_levels - 1` best) plus a distribution and a confidence. Nothing parses
prose: there is
no JSON to salvage out of a fenced block, no explanation to read, and no way for a
malformed sentence to become a match. What replaces the model's reasoning is arithmetic —
`DecisionPolicy` thresholds those probabilities into the same four statuses the LLM path
produces.

TypeSafe's Jev is the one implementation shipped, but `DecisionClient` is a protocol and
nothing above it knows the vendor.

## The loop

One source record, one call to `DecisionMatcher.match`:

```
render queries from the source record (several, not one)
│
├─ 1. RETRIEVE   every retriever × every query, concurrently; fuse by reciprocal rank;
│                keep up to `max_candidates` (default 300)
│
├─ 2. SCREEN     chunks of `chunk_size` (default 50) candidates, one call each,
│                one `noul` per candidate: "does this candidate denote the same
│                entity as the source?"                              → N/50 calls
│                → per-candidate probability; shortlist = top `shortlist_size`
│                  with probability ≥ `shortlist_floor`
│                → if the best probability < `screen_floor`: UNMATCHED, stop
│
├─ 3. CHOOSE     one call: `choice` over the shortlist plus NONE, instructions
│                "the most specific candidate the source supports"    → 1 call
│
└─ 4. GATE       one call on the chosen candidate alone: a `score` over the
                 rubric levels, one `noul` per declared property
                 (processing state agrees? species agrees? …)        → 1 call

derive_status(signals, policy) → one status, one reason
```

The typical record costs 6 to 8 calls, almost all of them in parallel screen chunks. There
is no retry loop: the LLM path retries because 25 candidates were the wrong 25, and this
path retrieves 300 instead.

Screen answers an absolute question per candidate and scales to any depth. Choose answers a
comparative question among the survivors, which is where near-duplicate ontology labels get
resolved by the specificity rule. Gate answers the absolute question about the one chosen
record with a clean state, and returns the property-level agreement a reviewer needs. That
is the same reason the LLM path scores separately from selecting: comparative and absolute
questions get different answers.

## The client contract

`xwalk.decide.base` defines the protocol, the questions, the answers, and the errors.

```python
@runtime_checkable
class DecisionClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """Digest of endpoint host, model id, and adapter version. Never a secret."""

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse: ...
```

`state` is anything JSON-serialisable; the stages pass dicts. `questions` is a mapping from
a name you choose to a question, and every answer comes back under the name you asked with.

### Questions

```python
Structured = str | Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is the probability that the answer is yes."""

    instructions: Structured
    criteria: Mapping[str, Structured] | None = None


@dataclass(frozen=True)
class Choice:
    """Pick one option from `criteria`. At most 255 options."""

    instructions: Structured
    criteria: Mapping[str, Structured | None]


@dataclass(frozen=True)
class Score:
    """Rate the state against ordered levels. Between 2 and 10 levels."""

    instructions: Structured
    criteria: Sequence[Structured]


Question = Noul | Choice | Score
```

The cardinality limits are enforced in `__post_init__` and raise `ValueError`: a `Choice`
needs 1 to 255 options, a `Score` needs 2 to 10 levels.

### Answers

```python
@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: Mapping[str, float]
    confidence: float
    legend: Mapping[str, str]


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DecisionResponse:
    answers: Mapping[str, Answer]
    model: str
    usage: Usage
```

`DecisionResponse.model` is the **served** model version, which is why a job may pin
`~typesafe/jev-latest` and still trace a run to the exact build that answered it.

### Parsing and errors

```python
def parse_response(
    payload: Mapping[str, Any],
    questions: Mapping[str, Question],
    *,
    configured_model: str,
) -> DecisionResponse
```

Every answer is typed against the question that was asked. A missing answer, an answer of
the wrong kind, a probability outside `[0, 1]`, or a `ChoiceAnswer` naming an option that
was never offered is a `DecisionFatalError` — a broken contract, not data about your
records. `usage` is read from the payload's `usage` object into a `Usage` with
`prompt_tokens`, `calls=1`, and `cost_usd`; there are no completion tokens, because there
is no completion.

| Error | Raised for |
|---|---|
| `DecisionError` | base class for every provider failure |
| `DecisionRetryableError(message, *, retry_after=None)` | transient: 408, 429, 5xx, 529, timeouts, connection resets |
| `DecisionFatalError` | auth failure, unknown model, malformed request, broken contract |

`response_to_dict` and `response_from_dict` are the cache representation; they mirror the
wire shape so `parse_response` reads a cached response back through the same validation.

## JevClient

```python
class JevClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None
```

It posts `{"model", "state", "questions"}` to `base_url` **exactly as given** — no path is
appended, because OpenRouter's path is `/api/alpha/decisions` and TypeSafe's own is
`/v1/systemone`. The job file's default is the OpenRouter one:

```yaml
decider:
  kind: jev
  model: "~typesafe/jev-latest"
  base_url: https://openrouter.ai/api/alpha/decisions
  api_key_env: XWALK_TEST_API_KEY
```

**Retries.** Status codes in `RETRYABLE_STATUSES` — `408, 429, 500, 502, 503, 504, 529` —
and `httpx` timeout and transport errors are retried up to `max_retries` times with
exponential backoff, `min(backoff_cap, backoff_base * 2 ** attempt)`, honouring a numeric
`retry-after` header when the response carries one. Any other status is a
`DecisionFatalError` immediately, with the provider's error message. Exhausting the retries
raises `DecisionRetryableError`.

**Fingerprint.** `hash_value` over the adapter name, `ADAPTER_VERSION`, the endpoint's host
and path, and the configured model id. The API key is not in it, so rotating a key does not
invalidate a resumable run — the same rule the LLM adapters follow. Moving from the
OpenRouter endpoint to TypeSafe's *does* change it, because the host and path are part of
what answered.

`aclose()` closes the underlying `httpx.AsyncClient`.

## FakeDecider

The reason the whole decider loop is testable offline, with no credentials and no network.

```python
Handler = Callable[[Any, Mapping[str, Question]], "Mapping[str, Answer] | BaseException"]
```

```python
class FakeDecider:
    def __init__(
        self,
        handler: Handler | None = None,
        *,
        model: str = "fake-decider",
        prompt_tokens: int = 100,
    ) -> None
```

A handler takes `(state, questions)` and returns a mapping of answers — or returns an
exception instance, which `decide` raises, which is how provider failures are exercised.
Every call is appended to `FakeDecider.calls` as a `(state, questions)` pair, so a test can
assert what was asked as well as what came back.

```python
def overlap_handler(state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]
```

The default handler. It scores Jaccard token overlap between `state["source"]` and the
candidate a question refers to: a `Noul` whose instructions mention `candidates.<key>` is
answered about that candidate, a `Choice` picks the highest-overlap option (or `NONE` when
every option scores zero), and a `Score` answers the middle level with confidence `0.5`. It
is enough to drive the loop end to end and it is never a model of your domain — do not read
accuracy into a run made with it.

## CachingDecider

```python
class CachingDecider:
    def __init__(
        self, inner: DecisionClient, ledger: Ledger, *, read: bool = True, write: bool = True
    ) -> None


def decision_cache_key(
    client: DecisionClient, state: Any, questions: Mapping[str, Question]
) -> str
```

`xwalk match` wraps every decider in this, always, and not as an optimisation. Jev's
probabilities move by up to about 0.04 between two identical calls. A resumed run that
re-asked would get different numbers for records the first run already decided, and a
`accept_at` of 0.85 would then classify the same record differently on Tuesday than it did
on Monday. Caching makes the run reproducible; it does not make it cheaper in any
interesting way.

The key is `hash_value` over the cache version, the client's `fingerprint`, the state, and
the serialised questions — so it is the *inner* client's identity that keys the cache, and
wrapping changes how an answer was obtained and never what it means. `model` and
`fingerprint` both delegate to the inner client for the same reason. The cache lives in the
run's own `ledger.sqlite`, which `xwalk match` opens a second connection to; SQLite in WAL
mode allows both.

**A cache hit reports `Usage.zero()`.** Tokens and cost are what a run *spent*, so a
resumed run does not re-bill work the first one paid for. `hits` and `misses` are counters
on the wrapper if you want to see the split.

## Questions from slots

The decider path reads the same `slots.yaml` the LLM path does. `QuestionSet` turns those
slots into the four kinds of question the stages ask; no skeleton `.j2` file is involved.

```python
NONE_KEY = "NONE"
```

```python
@dataclass(frozen=True)
class QuestionSet:
    slots: PromptSlots

    @classmethod
    def from_slots(cls, slots: PromptSlots) -> QuestionSet

    @property
    def rubric_levels(self) -> int

    @property
    def fingerprint(self) -> str

    def rules_state(self) -> dict[str, Any]
    def screen_question(self, key: str) -> Noul
    def choose_question(self, criteria: Mapping[str, str]) -> Choice
    def rubric_question(self) -> Score
    def property_questions(self) -> dict[str, Noul]
```

| Question | Built from | Shape |
|---|---|---|
| `rules_state()` | `entity_noun`, `target_noun`, `domain_brief`, every `hard_rules` entry verbatim, and fixed "same" / "different" definitions | a dict with keys `entity`, `target`, `domain`, `hard_rules`, `same`, `different`, sent once per screen chunk as the state's `rules` |
| `screen_question(key)` | nothing from the slots | a short `Noul` with no `criteria`, naming `candidates.<key>`, `source`, `rules.entity`, `rules.target`, `rules.domain`, `rules.hard_rules`, `rules.same` and `rules.different` |
| `choose_question(criteria)` | `entity_noun`, `target_noun`, `disambiguation_steps`, `hard_rules` | a `Choice` over the shortlist plus a `NONE` option described as "no candidate denotes the same entity" |
| `rubric_question()` | `rubric`, sorted **ascending** by score, each row rendered `"<name>: <when>"` | a `Score`; raises `ValueError` outside 2 to 10 rows |
| `property_questions()` | `properties` | one `Noul` per entry, named `prop_<name>` |

Jev is literal, which is why every instruction states the exact condition, names the state
fields it refers to in backticks, and says what true and false mean rather than leaving it
to be inferred.

The screen question is the one asked once per candidate, so it carries no preamble of its
own: the domain brief, the hard rules and what true and false mean travel once per chunk
in the state (`rules_state()`), and each question refers to them by path. Repeating them
per candidate was about two thirds of a Ref_zivila run's tokens. Choose, gate, rubric and property
questions run once per record and keep their preamble inline.

`fingerprint` digests the whole slots file plus a `questions_version` (2 since the screen
preamble moved into the state), so editing a hard rule changes the run
fingerprint and a resumed run re-matches rather than mixing results decided under different
rules.

### The `properties` block

`properties` is the one slots key the LLM skeletons ignore entirely. Each entry declares
one identity-bearing property the gate checks on the chosen candidate, and its agreement
probability becomes a `prop_<name>` signal — which is what replaces a generated explanation
for a reviewer.

```yaml
properties:
  - name: processing_state
    question: >-
      Do the source and the candidate agree on processing state (raw, cooked, dried,
      canned, frozen, smoked, fermented, juice, oil, flour, jam) wherever either states one?
  - name: species_or_ingredient
    question: >-
      Do the source and the candidate agree on the source species or main ingredient
      wherever either states one?
  - name: part_or_form
    question: >-
      Do the source and the candidate agree on the anatomical part or product form
      (whole, pieces, fillet, leg, powder, paste) wherever either states one?
```

`name` must match `^[a-z][a-z0-9_]*$` and `question` must be non-empty; both are validated
by `PromptSlots`. The list defaults to empty, so an existing slots file stays valid — a
decider job with no `properties` simply has no property gate, and `derive_status`'s
"every property agrees" clause is vacuously true. Write the question so that "neither
record states this property" is a **yes**: the rendered instruction says so explicitly, and
a property nobody mentions must not block a correct match.

Both food examples ship a populated block: `examples/cafeteria_fcd/slots.yaml` and
`examples/ref_zivila/jobs/foodon/slots.yaml`.

## The stages

### Screener

```python
def build_source_state(source: Record, context: str) -> dict[str, Any]


@dataclass(frozen=True)
class ScreenOutcome:
    probabilities: Mapping[str, float]  # opaque key -> p(same entity)
    shortlist: tuple[str, ...]  # record ids, best first
    issued: Mapping[str, str]  # opaque key -> record id
    usage: Usage
    chunks: int
    model: str
    notes: tuple[str, ...] = ()

    @property
    def best(self) -> float | None


class ScreenFailed(DecisionError):
    cause: DecisionError  # the chunk's own error
    usage: Usage  # what the sibling chunks that finished had already cost


class Screener:
    def __init__(
        self,
        decider: DecisionClient,
        questions: QuestionSet,
        templates: TemplateSet,
        *,
        chunk_size: int = 50,
        shortlist_size: int = 15,
        shortlist_floor: float = 0.2,
        max_state_chars: int = 120_000,
    ) -> None

    async def screen(
        self, source: Record, context: str, candidates: Sequence[Candidate]
    ) -> ScreenOutcome
```

This is where the path gets its recall. Candidates are rendered with the job's `candidate`
template and given opaque keys `C001`, `C002`, … — the same invariant as the LLM path, for
the same reason. Chunks of `chunk_size` go out concurrently, one `Noul` per candidate named
`n_<key>`. Each chunk's state is

```json
{"source": {"fields": {...}, "context": "..."},
 "rules": {"entity": "...", "target": "...", "domain": "...", "hard_rules": ["..."],
           "same": "...", "different": "..."},
 "candidates": {"C001": "...", "C002": "..."}}
```

with `source` and `rules` built once per record and shared by every chunk.

A chunk whose serialised state (source, rules and candidates together) would exceed
`max_state_chars` is halved until it fits, and each split is recorded in `notes` rather
than silently dropped. A single candidate that is over the cap on its own cannot be split
further: it is sent anyway, and a note (`candidate C… alone exceeds the state cap`) is the
only warning.

**Failures.** The first chunk to fail cancels the chunks still in flight, since they are
wasted spend once the record is going to fail. A `DecisionFatalError` propagates as
itself. Any other `DecisionError` is raised as `ScreenFailed`, which carries the original
error as `cause` and, as `usage`, what the chunks that had already finished cost, so the
caller can bill the failed attempt for it. Cancelling `screen()` itself cancels and awaits
every chunk task, so none is left running.

`shortlist` is the top `shortlist_size` record ids by probability that also clear
`shortlist_floor`. `best` is the highest probability seen over **all** candidates, not just
the shortlist, which is what `screen_floor` is tested against. An empty candidate list
returns an empty outcome without calling anything.

Those two floors are separate, so setting `shortlist_floor` above `screen_floor` creates a
gap: a record whose best candidate clears `screen_floor` but not `shortlist_floor` gets an
empty shortlist, choose and gate never run, and the result is `needs_review` /
`unresolved_output` with `Attempt.resolution` recorded as `abstain`. That is a correct
outcome — nothing was decided, so nothing is claimed — but it is not what `screen_floor`
alone would suggest. Keep `shortlist_floor` at or below `screen_floor` unless you want that
gap.

### Chooser

```python
@dataclass(frozen=True)
class ChooseOutcome:
    record_id: str | None
    resolution: Resolution
    p_choice: float
    p_none: float
    confidence: float
    raw: str
    usage: Usage
    model: str


class Chooser:
    def __init__(
        self, decider: DecisionClient, questions: QuestionSet, templates: TemplateSet
    ) -> None

    async def choose(
        self, source: Record, context: str, shortlist: Sequence[Candidate]
    ) -> ChooseOutcome
```

One `Choice` over the shortlist plus `NONE`. Three outcomes:

| Answer | `resolution` | `record_id` |
|---|---|---|
| a shortlist key | `Resolution.EXACT_KEY` | the record id that key was issued for |
| `NONE` | `Resolution.ABSTAIN` | `None`, with `p_choice = p_none` |
| anything else | `Resolution.UNRESOLVED` | `None` — unreachable through `parse_response`, kept because the invariant lives here |

An empty shortlist abstains without calling anything. Note that `p_choice` is *not* the
number the policy accepts on: near-duplicate labels split a `Choice`'s probability mass, so
the screen probability of the chosen record is the calibrated one and `p_choice` is only a
floor (`choose_at`).

### PropertyGate

```python
@dataclass(frozen=True)
class GateOutcome:
    rubric_score: float
    rubric_confidence: float
    rubric_levels: int
    properties: Mapping[str, float]
    raw: str
    usage: Usage
    model: str


class PropertyGate:
    def __init__(
        self, decider: DecisionClient, questions: QuestionSet, templates: TemplateSet
    ) -> None

    async def gate(self, source: Record, context: str, chosen: Candidate) -> GateOutcome
```

One call on the chosen record alone, with a state holding just the source and that one
candidate: the rubric `Score` plus every `prop_<name>` `Noul`. `rubric_score` is a level
index, `0` for the worst level and `rubric_levels - 1` for the best — not a `[0, 1]`
confidence. `properties` is keyed by the bare property name, with the `prop_` prefix
stripped.

## Policy

`DecisionPolicy` is a frozen dataclass in `xwalk.decide.policy`. In a job file it is the
`policy:` block — the same YAML key the LLM path uses, routed to this model when `decider:`
is present.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `screen_floor` | `float` | `0.30` | best screen probability below this → `unmatched`, no further calls |
| `shortlist_size` | `int` | `15` | at most this many survivors reach choose |
| `shortlist_floor` | `float` | `0.20` | survivors must have at least this screen probability |
| `none_at` | `float` | `0.70` | `p_none` at or above this → `unmatched` |
| `choose_at` | `float` | `0.50` | `p_choice` below this → `needs_review` |
| `accept_at` | `float` | `0.85` | screen probability of the chosen record required for `matched` |
| `rubric_floor` | `float \| None` | `None` → `levels - 2` | rubric level below this → `needs_review` |
| `property_floor` | `float` | `0.50` | any property below this → `needs_review` |
| `chunk_size` | `int` | `50` | candidates per screen call |
| `max_candidates` | `int` | `300` | fused candidates kept after retrieval |
| `concurrency` | `int` | `32` | records in flight |
| `retriever_timeout` | `float` | `60.0` | seconds before a retriever is treated as failed |

Validation in `__post_init__`: every probability field must be in `[0, 1]`,
`shortlist_floor` must not exceed `accept_at`, and `shortlist_size`, `chunk_size`,
`max_candidates` and `concurrency` must each be at least 1. Each violation raises
`ValueError`.

```python
def effective_rubric_floor(policy: DecisionPolicy, levels: int) -> float
```

`rubric_floor` defaults to `None`, which resolves to `max(0, levels - 2)` — the level index
of the second-highest rubric row. A four-row rubric therefore demands level 2 or better.
Set it explicitly to override.

These defaults are starting points from the September 2026 probe, not fitted values. Fit
them; see [below](#fitting-thresholds).

### Signals

```python
@dataclass(frozen=True)
class Signals:
    """Everything the decider path learned about one record, as numbers."""

    candidate_count: int
    retrieval_failed: bool
    screen_best: float | None = None
    screen_chosen: float | None = None
    resolution: str | None = None
    p_choice: float | None = None
    p_none: float | None = None
    choice_confidence: float | None = None
    rubric: float | None = None
    rubric_levels: int | None = None
    rubric_confidence: float | None = None
    properties: Mapping[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, float]

    @classmethod
    def from_flat(
        cls,
        flat: Mapping[str, float],
        *,
        candidate_count: int,
        retrieval_failed: bool,
        resolution: str | None,
    ) -> Signals
```

`flat()` drops every `None` and prefixes each property with `prop_`, producing the
`Mapping[str, float]` stored on `Attempt.signals` and `MatchResult.signals`. `from_flat` is
its inverse and is what makes fitting free: the whole classification can be re-derived from
the ledger without calling anything.

### derive_status

```python
def derive_status(signals: Signals, policy: DecisionPolicy) -> tuple[MatchStatus, DecisionReason]
```

Seven rules, in order; the first that matches wins.

1. every retriever failed → `failed` / `retriever_failure`
2. no candidates, or nothing was screened → `unmatched` / `no_candidates`
3. best screen probability < `screen_floor` → `unmatched` / `below_review_floor`
4. choose said `NONE` and `p_none` ≥ `none_at` → `unmatched` / `selector_abstained`
5. choose said `NONE` below `none_at`, or did not resolve to a key → `needs_review` /
   `unresolved_output`
6. all of: screen probability of the chosen record ≥ `accept_at`, `p_choice` ≥
   `choose_at`, rubric ≥ `effective_rubric_floor`, and every property ≥ `property_floor`
   → `matched` / `accept_threshold`
7. otherwise → `needs_review` / `below_accept_threshold`

A provider failure is handled by the matcher rather than here, and produces `failed` /
`provider_failure`.

Rule 6 is a conjunction: one property at `0.40` sends an otherwise perfect match to review.
That is the intended direction — an automatic match is the one nobody looks at.

```python
def render_explanation(signals: Signals) -> str
```

The reviewer-facing line, built only from numbers and therefore identical on every run of
the same configuration. It reads like
`same 0.97; chosen 0.94; p_choice 0.81; p_none 0.04; rubric 3.0/3; part_or_form 0.99; processing_state 0.41`,
which is enough to see *why* a row is in the review bucket without a generated sentence
anyone has to trust.

## DecisionMatcher

```python
class DecisionMatcher:
    def __init__(
        self,
        *,
        templates: TemplateSet,
        retrievers: Sequence[Retriever],
        store: TargetStore,
        screener: Screener,
        chooser: Chooser,
        gate: PropertyGate,
        policy: DecisionPolicy | None = None,
        run_fingerprint: str = "",
        rrf_k: int = 60,
        keep_candidates_in_trace: bool = True,
    ) -> None

    async def match(self, source: Record) -> MatchResult
    def match_sync(self, source: Record) -> MatchResult
```

It satisfies the same `MatcherLike` protocol `run_batch` takes — `run_fingerprint`,
`policy`, `store_fingerprint`, `async match` — so batching, the ledger, resume, review and
every export work unchanged. An empty retriever list raises `ValueError`.

Retrieval is the shared `xwalk.retrieve.retrieve`, which searches every retriever with
every rendered query concurrently and fuses by reciprocal rank. A record surfaced by
several queries accumulates several votes, which is the point of `templates.queries` being
a list.

What lands in the result:

| Field | Value |
|---|---|
| `MatchResult.confidence` | the screen probability of the chosen record (`signals.screen_chosen`) — the calibrated number, not `p_choice` |
| `MatchResult.signals` / `Attempt.signals` | `Signals.flat()`, plus one `screen_<key>` entry per shortlisted candidate so screen-stage recall can be measured from the ledger |
| `MatchResult.explanation` | `render_explanation(signals)` |
| `Attempt.raw_selection` | JSON: the served model, the choose answer, and the gate answer |
| `Attempt.issued_keys` | the screen stage's opaque key → record id map |
| `Attempt.primary_score` | the same screen probability as `confidence` |
| `Attempt.resolution` | a `Resolution` value: `exact_key`, `abstain`, or `unresolved` |
| `Attempt.index` | always `0` — one attempt per record, always |

`Attempt.proposal`, `verifier_decision`, `verifier_score`, `verifier_preferred_id`,
`audited`, `dropped_proposals` and `finish_reason` are all `None`, `False` or empty on this
path; the stages that fill them do not run.

A `DecisionFatalError` propagates and kills the run, because a broken contract will not fix
itself over 10,000 records. A `DecisionRetryableError` that survived the client's own
retries is caught per record and becomes `failed` / `provider_failure` with the message in
`Attempt.error`.

`mapping.csv` carries the same columns on both paths, `cost_usd` included; on the decider
path `completion_tokens` is always `0`, and `llm_calls` counts decision calls.

## Fitting thresholds

The defaults ship unfitted. `xwalk fit` sweeps `accept_at` and `property_floor` over a
completed run's recorded signals and prints the pair that meets a precision target with the
most automatic matches. It calls nothing — it re-derives statuses through `derive_status`
from the ledger — so it is free and repeatable.

```console
$ xwalk fit --run runs/cfcd_jev --gold examples/cafeteria_fcd/sample/gold.csv \
    --job examples/cafeteria_fcd/job_jev.yaml --precision 0.95
labelled rows: 120; target precision: 0.95

accept_at prop_floor accepted correct precision coverage near
     0.75       0.50       96      85      0.89     0.80   31
     0.80       0.50       88      82      0.93     0.73   24
     0.85       0.00       81      76      0.94     0.68   17
     0.85       0.50       74      72      0.97     0.62   17
     0.90       0.50       61      61      1.00     0.51    9

recommended: accept_at=0.85 property_floor=0.50 (72/74 correct, coverage 0.62, 17 rows within the jitter margin of accept_at)
```

Only those two thresholds are swept. Every other gate — `screen_floor`, `choose_at`,
`none_at`, `rubric_floor`, `shortlist_floor` — has to be the one the run actually used, or
the fitted pair is tuned against a policy nobody ran. That is what `--job` is for: it reads
the job's own `policy:` block as the base. **`--job` is optional, and omitting it fits
against `DecisionPolicy()` defaults** and prints `note: --job not given; fitting against
default policy thresholds` on stderr. If your job overrides any unswept gate, pass it.

### Reading `near_threshold`

Jev's probabilities move by up to about 0.04 between identical calls, so a grid point's
precision is only as trustworthy as the number of labelled rows that are not sitting on the
knife edge. `near` counts labelled rows whose screen probability is within `margin`
(default `0.05`) of that row's `accept_at`.

A large `near` next to a precision that just clears the target means the result is an
artefact of where the jitter landed this time, and the same run repeated would give a
different number. Prefer a point with a comparable precision and a smaller `near`, or
gather more labels. The recommended point is chosen by accepted count, not by `near` — the
column is there for you to overrule it.

### The API

```python
def fit_thresholds(
    results: Iterable[MatchResult],
    gold: GoldSet,
    *,
    base: DecisionPolicy,
    target_precision: float = 0.95,
    accept_grid: Sequence[float] = DEFAULT_ACCEPT_GRID,
    property_grid: Sequence[float] = DEFAULT_PROPERTY_GRID,
    margin: float = 0.05,
) -> FitReport


def render_fit(report: FitReport) -> str


@dataclass(frozen=True)
class FitPoint:
    accept_at: float
    property_floor: float
    accepted: int
    correct: int
    precision: float | None
    coverage: float
    near_threshold: int


@dataclass(frozen=True)
class FitReport:
    points: list[FitPoint]
    recommended: FitPoint | None
    target_precision: float
    labelled: int
```

`DEFAULT_ACCEPT_GRID` is `0.50` to `0.95` in steps of `0.05`; `DEFAULT_PROPERTY_GRID` is
`(0.0, 0.3, 0.5, 0.7)`. `recommended` is the eligible point — precision at or above
`target_precision` — with the most accepted rows, and is `None` when no point qualifies, in
which case `xwalk fit` prints `no grid point meets the target precision; lower the target
or improve the questions` and exits `1` (`EXIT_ATTENTION`).

Unlabelled source ids are skipped entirely; `precision` is `None`, printed as `-`, for a
grid point that accepted nothing.

## What the decider path does not do

Every one of these is an absence with a reason, not a gap waiting to be filled.

- **No retries.** One attempt per record, `Attempt.index` always `0`. The LLM path retries
  because 25 candidates were the wrong 25; this path retrieves 300. There is no
  `max_attempts`.
- **No rewriter.** Retry leads have nowhere to go without a retry loop. Recall comes from
  `templates.queries` and retrieval depth instead.
- **No verifier and no audit sampling.** There is no second model to ask. `verify_band` and
  `audit_rate` do not exist on `DecisionPolicy`; the gate's rubric and properties are the
  independent second look, bought on every accepted record rather than on a band.
- **No generated explanation.** `explanation` is rendered from the numbers by
  `render_explanation`, so it is identical across runs of the same configuration and
  nothing in it can be a fabrication. The per-property probabilities are what a reviewer
  reads.
- **`xwalk ablate` refuses a decider job**, printing `error: this command works on the LLM
  path; the job has a decider: block` and exiting `2` (`EXIT_USAGE`). Its standard variants
  are `no_verifier`, `no_retries` and `half_budget` — none of which exists here.
- **`xwalk prompts` refuses a decider job**, both `draft` and `optimize`, with the same
  message and the same exit code. The optimiser measures accepted precision by re-running
  the LLM matcher with mutated slots; there is no decider equivalent yet. Edit
  `slots.yaml` by hand and re-fit.

The LLM path is unchanged and remains the default. Use the decider path when you have
hundreds of plausible candidates per record and you want a calibrated number for each one.

## See also

- [The job file](job-file.md) — the `decider:` block, `templates.queries`, and which
  `policy:` keys apply.
- [Command line](cli.md) — `xwalk fit`, and what `match` does differently here.
- [Prompts](prompts.md) — the slots file both paths share.
- [Core types](core-types.md) — `Usage`, `Attempt.signals`, `MatchResult.signals`.
- [The matching pipeline](pipeline.md) — `run_batch`, the ledger, and the exports, all of
  which this path reuses.
- `examples/cafeteria_fcd/job_jev.yaml` and `examples/ref_zivila/jobs/foodon/job_jev.yaml`
  — runnable jobs.

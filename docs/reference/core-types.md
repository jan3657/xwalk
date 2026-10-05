# Core types

Everything in xwalk is built from a small set of frozen dataclasses, two enums, one
policy object, and the ledger they are persisted in. This page is the reference for
that layer: the value types that flow through a run, the rules that turn attempts into
a status, and the state that makes a run resumable.

All value types live in `xwalk.records` and are immutable (`@dataclass(frozen=True)`).
Construct a replacement rather than mutating in place.

## Record

One row from either collection — source or target. `fields` is whatever the source
produced; xwalk never imposes a schema on it. The templates are where field names
acquire meaning.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | — | stable identity of the row within its collection |
| `fields` | `Mapping[str, Any]` | — | raw field values, as produced by the source |

Validation in `__post_init__`:

- `id` must be a `str` — anything else raises `TypeError`
- `id` must be non-empty after stripping whitespace — raises `ValueError`
- `fields` must be a `Mapping` — raises `TypeError`

No properties or methods beyond the dataclass machinery.

## RetrievalHit

One retriever's opinion about one target record.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `record_id` | `str` | — | id of the target record that was hit |
| `retriever` | `str` | — | name of the retriever that produced the hit |
| `raw_score` | `float \| None` | — | backend-native score; scales differ per backend |
| `rank` | `int` | — | position in that retriever's ranking, **1-based** |

Validation in `__post_init__`: `rank < 1` raises `ValueError`. Rank is 1-based
because that is what rank fusion consumes; a 0-based rank fed into RRF silently
inflates every score.

## Candidate

A fused candidate: one target record plus the evidence for why it surfaced.
`evidence` records every retriever that returned it, so a candidate found by both
the lexical and the dense index is distinguishable from one found by either alone.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `record` | `Record` | — | the target record itself |
| `fused_score` | `float` | — | score after rank fusion across retrievers |
| `evidence` | `tuple[RetrievalHit, ...]` | — | every hit that surfaced this record |

Validation in `__post_init__`: every `hit.record_id` in `evidence` must equal
`record.id`; a mismatch raises `ValueError`. Evidence for one record attached to
another would corrupt every downstream diagnostic.

Properties:

```python
@property
def id(self) -> str  # shorthand for candidate.record.id
```

## Usage

Token and call accounting. Deliberately additive so per-stage usages sum into an
attempt and per-attempt usages sum into a result.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `prompt_tokens` | `int` | `0` | prompt tokens consumed |
| `completion_tokens` | `int` | `0` | completion tokens consumed |
| `calls` | `int` | `0` | upstream requests dispatched, including client-side retries and calls that failed |
| `unknown_calls` | `int` | `0` | calls whose token usage was not reported (errors, interrupted calls, providers that omit usage) |
| `cache_hits` | `int` | `0` | responses served from the LLM or decision cache; not upstream calls |
| `cost_usd` | `float` | `0.0` | cost reported by the provider, in US dollars |

Token fields sum only provider-reported tokens, so `total_tokens` is a lower bound
whenever `unknown_calls > 0`; `describe_tokens()` renders it as
`"N (+k calls with unknown usage)"`. `cost_usd` is whatever the provider stated; a
provider that reports no cost (every LLM adapter today) leaves it `0.0`, so it is a
record of what was billed and never an estimate -- and, like the tokens, it is a lower
bound when `unknown_calls > 0`. Ledgers written before 0.2 read back with
`unknown_calls=0`, `cache_hits=0` and `cost_usd=0.0`. No validation. Methods and
properties:

```python
@classmethod
def zero(cls) -> Usage          # all zero

@classmethod
def unreported(cls, calls=1) -> Usage  # `calls` calls with unknown usage

@property
def total_tokens(self) -> int   # prompt_tokens + completion_tokens

def describe_tokens(self) -> str     # "120" or "120 (+1 calls with unknown usage)"
def __add__(self, other) -> Usage   # fieldwise sum; NotImplemented for non-Usage
def __radd__(self, other) -> Usage  # 0 + Usage == Usage, so sum(usages) works
```

`__radd__` accepts `0` specifically so `sum(a.usage for a in attempts)` works with
the built-in `sum` and its default start value.

## RetryProposal

A lead for the next attempt in the matching loop.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `Literal["candidate", "query"]` | — | what kind of lead this is |
| `value` | `str` | — | a candidate key, or new query text |
| `source` | `Literal["scorer", "rewriter"]` | — | which stage proposed it |

No validation. The `kind` distinction is load-bearing: `"candidate"` means "look
again at a record we already retrieved" — no new search — while `"query"` means "run
a genuinely new search". Conflating the two makes the loop re-retrieve records it
already has in hand.

## MatchStatus

The terminal classification of one source record. Produced by
`derive_status` (see [MatchPolicy](#matchpolicy)).

| Member | Value | Produced when |
|---|---|---|
| `MATCHED` | `"matched"` | best attempt resolved exactly, unchallenged, score ≥ `accept_at` |
| `NEEDS_REVIEW` | `"needs_review"` | a human should look: mid-band score, verifier disagreement, or inexact/unresolved output |
| `UNMATCHED` | `"unmatched"` | evidence points to no match: sub-floor score, abstention, or no candidates |
| `FAILED` | `"failed"` | infrastructure failed; **not** evidence about the record |

`UNMATCHED` is a claim about the data; `FAILED` is a claim about the run. A dead
index or provider outage must never be recorded as "there is no match".

## DecisionReason

Why a result carries the status it does. Every `MatchResult` and (optionally) every
`Attempt` carries one.

| Member | Value | Produced when |
|---|---|---|
| `ACCEPT_THRESHOLD` | `"accept_threshold"` | best exact-key score ≥ `accept_at` → `MATCHED` |
| `BELOW_ACCEPT_THRESHOLD` | `"below_accept_threshold"` | best score in `[review_floor, accept_at)` → `NEEDS_REVIEW` |
| `BELOW_REVIEW_FLOOR` | `"below_review_floor"` | best score < `review_floor` → `UNMATCHED` |
| `SELECTOR_ABSTAINED` | `"selector_abstained"` | candidates were shown, the model declined every one → `UNMATCHED` |
| `NO_CANDIDATES` | `"no_candidates"` | no attempt ever retrieved a candidate → `UNMATCHED` |
| `UNRESOLVED_OUTPUT` | `"unresolved_output"` | model output could not be resolved to an issued key, or the best attempt used an inexact (legacy) resolution → `NEEDS_REVIEW` |
| `RETRIEVER_FAILURE` | `"retriever_failure"` | every retriever in an attempt failed or timed out; all attempts failed → `FAILED` |
| `PROVIDER_FAILURE` | `"provider_failure"` | a recoverable LLM call failure; all attempts failed → `FAILED`; best attempt's in-band score could not be verified → `NEEDS_REVIEW`. Also the fallback reason when there are no attempts at all |
| `FATAL_PROVIDER_FAILURE` | `"fatal_provider_failure"` | an `LLMFatalError` (auth, unknown model, invalid request) in any stage → `FAILED`. Added in 0.2 |
| `VERIFIER_DISAGREEMENT` | `"verifier_disagreement"` | verifier answered `"disagree"` or `"no_match"` on the best attempt → `NEEDS_REVIEW` |

## Attempt

One pass through retrieve → select → gate. This is the complete audit trail: every
decision, every raw model answer, and every degradation note for one pass survives
here. Nothing an attempt learned is discarded when the loop moves on.

| Field | Type | Meaning |
|---|---|---|
| `index` | `int` | 0-based position within the run's attempt sequence |
| `query` | `str` | the retrieval query this attempt ran (or reused) |
| `proposal` | `RetryProposal \| None` | the proposal that triggered this attempt; `None` for the first |
| `candidates` | `tuple[Candidate, ...]` | the fused candidate pool; empty when the matcher is configured not to keep candidates in the trace |
| `candidate_count` | `int` | how many candidates were retrieved, even when `candidates` is empty |
| `candidates_truncated` | `int` | candidates cut by the selector's budget before the model saw them |
| `issued_keys` | `Mapping[str, str]` | opaque key → record id, as issued **for this attempt** |
| `raw_selection` | `str \| None` | the selector's verbatim answer, before resolution |
| `chosen_id` | `str \| None` | resolved record id; `None` on abstention or unresolved output |
| `resolution` | `str` | how the raw answer became an id: a `Resolution` value such as `"exact_key"`, `"abstain"`, `"unresolved"`, or a `legacy_*` variant |
| `primary_score` | `float \| None` | the scorer's confidence; `None` if scoring never ran or produced no usable number |
| `explanation` | `str` | the model's stated reasoning, scorer's preferred over selector's |
| `verifier_decision` | `str \| None` | verifier verdict; `None` when verification did not run |
| `verifier_score` | `float \| None` | verifier confidence |
| `verifier_preferred_id` | `str \| None` | record id the verifier preferred instead, if any |
| `audited` | `bool` | verification was triggered by audit sampling, not the verify band |
| `dropped_proposals` | `tuple[tuple[str, str], ...]` | `(proposal value, why it was dropped)` pairs |
| `reason` | `DecisionReason \| None` | per-attempt failure reason; set on infrastructure failures |
| `error` | `str \| None` | error text or joined retriever degradation notes |
| `usage` | `Usage` | tokens and calls spent in this attempt |
| `elapsed_seconds` | `float` | wall time for this attempt |
| `finish_reason` | `str \| None` | `finish_reason` of the **last** provider response in the attempt |
| `signals` | `Mapping[str, float]` | calibrated numbers from a decision model; `{}` on the LLM path |

`finish_reason` earns its place: `"length"` is the tell for a truncated answer — the
provider stopped mid-JSON, which surfaces as `UNRESOLVED_OUTPUT` and is otherwise
indistinguishable from a model that simply answered badly.

`signals` defaults to `{}` and is filled only by the [decider path](decide.md), with
`Signals.flat()`: `screen_best`, `screen_chosen`, `p_choice`, `p_none`,
`choice_confidence`, `rubric`, `rubric_levels`, `rubric_confidence`, one `prop_<name>` per
declared property, and one `screen_<key>` per shortlisted candidate. Everything the policy
thresholds is in there, which is what makes `xwalk fit` free: a finished run can be
re-classified from the ledger without calling anything.

## MatchResult

The terminal record for one source record under one run configuration.
`attempts` holds **every** attempt in order — the result is a summary, the attempts
are the evidence, and nothing summarised is unrecoverable from them.

| Field | Type | Meaning |
|---|---|---|
| `result_key` | `str` | stable identity: `result_key(run_fingerprint, source_id, source_hash)` |
| `source_id` | `str` | id of the source record |
| `source_hash` | `str` | `hash_record` of the source record at match time |
| `matched_id` | `str \| None` | chosen target id; `None` for `UNMATCHED` and `FAILED` |
| `matched_record` | `Record \| None` | the matched target record, when it could be fetched from the store |
| `confidence` | `float \| None` | the winning attempt's `primary_score` |
| `status` | `MatchStatus` | terminal classification |
| `reason` | `DecisionReason` | why that status |
| `explanation` | `str` | the winning attempt's explanation |
| `candidates` | `tuple[Candidate, ...]` | candidate pool from the most recent attempt that produced one; empty when candidates are not kept in the trace |
| `attempts` | `tuple[Attempt, ...]` | the full audit trail, in order |
| `usage` | `Usage` | sum of all attempts' usage |
| `elapsed_seconds` | `float` | wall time for the whole match |
| `run_fingerprint` | `str` | the run configuration this result belongs to |
| `signals` | `Mapping[str, float]` | the winning attempt's `signals`; `{}` on the LLM path |

## MatchPolicy

Defined in `xwalk.policy`. Cost controls, classification thresholds, and the loop
budget in one frozen dataclass.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_attempts` | `int` | `4` | hard cap on attempts per source record |
| `accept_at` | `float` | `0.6` | scores at or above this become `MATCHED` |
| `review_floor` | `float` | `0.4` | scores at or above this (but below `accept_at`) become `NEEDS_REVIEW`; below it, `UNMATCHED` |
| `verify_band` | `tuple[float, float] \| None` | `(0.6, 0.8)` | inclusive score band that buys a verifier call; `None` disables verification |
| `audit_rate` | `float` | `0.0` | fraction of above-band matches sampled for a verifier call |
| `concurrency` | `int` | `32` | batch-run concurrency |
| `legacy_id_resolution` | `bool` | `False` | re-enable heuristic ID resolution for reproducing prior work |
| `retriever_timeout` | `float` | `60.0` | per-retriever search timeout, seconds |

Validation in `__post_init__` — each violation raises `ValueError`:

- `max_attempts >= 1`
- `0.0 <= review_floor <= accept_at <= 1.0`
- `0.0 <= audit_rate <= 1.0`
- `verify_band`, when set, is an ordered pair with `0.0 <= low <= high <= 1.0`
- `concurrency >= 1`

### How the thresholds interact

`accept_at` and `review_floor` are *classification*: they slice the score axis into
`MATCHED` / `NEEDS_REVIEW` / `UNMATCHED`. `verify_band` is a *cost control*: it buys
a second opinion only where the primary score is genuinely uncertain, and changes no
classification boundary by itself. The two are deliberately separate so each can be
tuned without disturbing the other. `audit_rate` covers the remaining blind spot:
high-confidence errors are by definition invisible, so a fraction of scores **above**
the verify band is sampled for the same verifier call. Sampling decides *which*
results get a second opinion, never whether the opinion counts — an audit verdict is
honoured exactly like any other verdict.

### Functions

```python
def should_verify(score: float | None, policy: MatchPolicy) -> bool
```

`True` iff `score` is not `None`, `policy.verify_band` is not `None`, and
`low <= score <= high`. Both ends inclusive.

```python
def should_audit(
    score: float | None,
    policy: MatchPolicy,
    run_fingerprint: str,
    source_id: str,
) -> bool
```

`False` when `audit_rate <= 0.0`, when `score` is `None`, or when a verify band is
set and `score <= band[1]` — that region is already covered by verification. The
draw is deterministic: SHA-256 of `run_fingerprint + "\x00" + source_id`, first 8
bytes as a fraction of 2^64, compared to `audit_rate`. Seeded this way so audits are
reproducible across resumes rather than re-rolled each time a run restarts.

```python
def derive_status(
    attempts: Sequence[Attempt],
    policy: MatchPolicy,
) -> tuple[MatchStatus, DecisionReason, Attempt | None]
```

Reduces a list of attempts to one status, one reason, and the winning attempt.
Precedence, in order:

1. No attempts at all → `(FAILED, PROVIDER_FAILURE, None)`.
2. Every attempt is an infrastructure failure (`RETRIEVER_FAILURE` or
   `PROVIDER_FAILURE`) → `FAILED` with the last attempt's reason (falling back to
   `PROVIDER_FAILURE`), no winning attempt.
3. Any attempt has both `chosen_id` and `primary_score`: the highest score wins,
   ties go to the earlier attempt. Then, on that best attempt:
   1. `verifier_decision` in `("disagree", "no_match")` →
      `(NEEDS_REVIEW, VERIFIER_DISAGREEMENT)`.
   2. `resolution` is not `"exact_key"` → `(NEEDS_REVIEW, UNRESOLVED_OUTPUT)`.
      Legacy or otherwise inexact resolution never becomes an automatic match.
   3. score ≥ `accept_at` → `(MATCHED, ACCEPT_THRESHOLD)`.
   4. score ≥ `review_floor` → `(NEEDS_REVIEW, BELOW_ACCEPT_THRESHOLD)`.
   5. otherwise → `(UNMATCHED, BELOW_REVIEW_FLOOR)`.
4. No scored attempt, but some attempt's resolution is `"unresolved"` →
   `(NEEDS_REVIEW, UNRESOLVED_OUTPUT)` with the last such attempt.
5. Some attempt had candidates → `(UNMATCHED, SELECTOR_ABSTAINED)` with the last
   attempt that had any.
6. Otherwise → `(UNMATCHED, NO_CANDIDATES)` with the last attempt.

## TemplateSet

Defined in `xwalk.templates`. The four Jinja2 templates that carry the entire domain
mapping — everything xwalk knows about your fields, it learns from these.

| Template | Input | Produces |
|---|---|---|
| `query` | source record | retrieval query string |
| `context` | source record | context block shown to the LLM |
| `doc` | target record | indexed text |
| `candidate` | target record | one entry in the candidate list |

`TemplateSet` holds template *source* strings and compiles all four on construction,
so a typo fails fast at startup rather than 40k rows into a run. A compile failure
raises `TemplateError` (subclass of `Exception`) naming the offending template.

```python
def render_query(self, record: Record) -> str
def render_context(self, record: Record) -> str
def render_doc(self, record: Record) -> str
def render_candidate(self, record: Record) -> str
```

Rendering rules:

- The template sees every entry of `record.fields` as a variable, plus `id`, which
  is **always** `record.id` — record identity wins over a field literally named
  `"id"`.
- Any Jinja runtime error is re-raised as `TemplateError` naming the template and
  the record id.
- Missing fields render as the empty string, not an error. Source collections are
  sparse in practice; aborting a 100k-row run because one row lacks an optional
  column is the wrong trade, and a silently-empty slot is visible in the rendered
  output and in the attempt trace.
- Whitespace is tidied after rendering: runs of spaces and tabs collapse to one
  space, lines are stripped, blank lines are dropped, and the result is stripped.
  Conditionals therefore never leave ragged gaps in what the model sees.

```python
@property
def fingerprint(self) -> str
```

`hash_value` of the four template sources, keyed `query` / `context` / `doc` /
`candidate`. Feeds the run fingerprint, so editing any template invalidates prior
results.

## Fingerprints (`xwalk.fingerprint`)

Deterministic hashing of records and run configuration. Everything that decides
whether a prior result is still valid flows through here.

```python
def canonical_json(value: Any) -> str
```

A byte-stable JSON rendering: sorted keys, `(",", ":")` separators, no NaN,
`ensure_ascii=False`. Values are first normalised: `None`, `bool`, `int`, `str`,
and `float` pass through; `datetime`/`date` become ISO strings; `Decimal` becomes
`str`; mapping keys are stringified and values normalised; sets are normalised then
sorted by their canonical rendering (deterministic despite having no order);
non-string sequences become lists. Anything else raises `TypeError` — an
unserialisable value must fail loudly, not hash to something accidental.

```python
def hash_value(value: Any) -> str
```

SHA-256 of the UTF-8 `canonical_json`, truncated to **16 hex characters** — short
enough to read in a filename, wide enough to be safe.

```python
def hash_record(record: Record) -> str
```

`hash_value` of `{"id": record.id, "fields": record.fields}` — identity *and*
content, so an edited source row is detected on resume.

```python
def run_fingerprint(**components: Any) -> str
```

Digest of everything that would invalidate prior results. Callers pass the target
snapshot fingerprint, retriever fingerprints, prompt and slots hashes, model and
provider identity, generation parameters, the match policy, and the library version.

```python
def result_key(run_fp: str, source_id: str, source_hash: str) -> str
```

Stable identity of one source record under one run configuration. The parts go in
as a list, never concatenated: `"a" + "bc"` and `"ab" + "c"` produce the same
string, and a resume key that can collide is worse than none.

## Serialisation (`xwalk.serde`)

```python
def result_to_dict(result: MatchResult) -> dict[str, Any]
def result_from_dict(data: Mapping[str, Any]) -> MatchResult
```

Lossless round-trip between a `MatchResult` and a plain JSON-safe dict, nested
attempts, candidates, and evidence included. Enums travel as their string values and
come back as enums; tuples travel as lists (JSON has none) and come back as tuples,
including `dropped_proposals` pairs. The ledger stores one JSON blob per result;
keeping the conversion here means the storage layer never has to know the shape of a
`MatchResult`.

## Ledger

Defined in `xwalk.ledger`. Transactional, resumable execution state: every completed
result is committed as it finishes, so a batch run never loses work. JSONL/CSV/
manifest *files* are exports regenerated from the ledger, never the source of truth
— delete them freely; the ledger can rebuild them.

Storage is SQLite in WAL mode. Reads go straight through — WAL allows concurrent
readers. Every write is one explicit transaction with no `await` inside it, so a
cancelled task cannot leave half a transaction behind; a transaction that fails is
rolled back. Tables: `ledger_meta` (`schema_version`), `results` (the latest revision
of each result key), `result_history` (earlier revisions, append-only), `snapshots` and
`snapshot_entries` (each invocation's ordered source list), `invocations` (how each
`run_batch` call ended), `llm_cache` (keyed provider responses), `manifests` (one JSON
blob per run fingerprint), and `reviews` (append-only human adjudications, each bound to
the result revision it was made against).

```python
LEDGER_SCHEMA_VERSION = 2

class UnsupportedLedgerError(Exception): ...
class LedgerClosedError(RuntimeError): ...

class Ledger:
    @classmethod
    def open(cls, path: str | Path) -> Ledger: ...
    schema_version: int          # property
    migrated_from: int | None    # 1 when this open upgraded a 0.1.1 ledger
    backup_path: Path | None     # the copy taken before that upgrade
    closed: bool                 # property

    # results
    async def put_result(self, result: MatchResult) -> None: ...
    def get_result(self, result_key: str) -> MatchResult | None: ...
    def has_result(self, result_key: str) -> bool: ...
    def is_settled(self, result_key: str) -> bool: ...   # committed and not failed
    def result_revision(self, result_key: str) -> int | None: ...

    # current view (one entry per source in the latest snapshot)
    def put_snapshot(self, run_fingerprint: str, entries: Sequence[tuple[str, str]]) -> int: ...
    def iter_current(self, run_fingerprint: str) -> Iterator[CurrentEntry]: ...
    def iter_results(self, run_fingerprint: str) -> Iterator[MatchResult]: ...
    def current_keys(self, run_fingerprint: str) -> dict[str, int]: ...
    def count(self, run_fingerprint: str) -> int: ...
    def pending_count(self, run_fingerprint: str) -> int: ...
    def count_by_status(self, run_fingerprint: str) -> dict[MatchStatus, int]: ...
    def duplicate_targets(self, run_fingerprint: str) -> dict[str, list[str]]: ...
    def removed_sources(self, run_fingerprint: str) -> list[str]: ...

    # history (every result ever committed)
    def iter_history(self, run_fingerprint: str) -> Iterator[HistoryEntry]: ...
    def history_count(self, run_fingerprint: str) -> int: ...

    # invocations
    def begin_invocation(self, run_fingerprint: str, *, library_version: str,
                         resume: bool, limit: int | None) -> int: ...
    def finish_invocation(self, invocation_id: int, *, run_state: str,
                          snapshot_id: int | None, usage: Mapping[str, Any],
                          errors: Sequence[Mapping[str, Any]]) -> None: ...
    def last_invocation(self, run_fingerprint: str) -> dict[str, Any] | None: ...

    # llm cache, manifest, reviews
    def get_cached(self, cache_key: str) -> str | None: ...
    async def put_cached(self, cache_key: str, response: str) -> None: ...
    def put_manifest(self, run_fingerprint: str, manifest: Mapping[str, Any]) -> None: ...
    def get_manifest(self, run_fingerprint: str) -> dict[str, Any] | None: ...
    def put_review(self, row: Mapping[str, Any]) -> None: ...
    def iter_reviews(self, run_fingerprint: str) -> Iterator[dict[str, Any]]: ...

    def close(self) -> None: ...
```

`CurrentEntry(source_id, source_hash, result, revision)` has `result=None` for a pending
source. `HistoryEntry(result, revision, current)`.

- `open(path)` creates parent directories, connects with `check_same_thread=False` and
  autocommit (`isolation_level=None`), sets `journal_mode=WAL` and `synchronous=NORMAL`.
  A new file gets the current schema. A 0.1.1 ledger is copied to
  `<name>.v1-backup` and then upgraded additively in one transaction. A newer schema
  version, an SQLite file that is not a ledger, or a non-SQLite file raises
  `UnsupportedLedgerError` without modifying the file.
- `put_result` on an existing `result_key` moves the previous row to history and
  stores the new one as the next revision; nothing is overwritten in place.
- The current view is the latest snapshot's entries joined to their results. Without a
  snapshot (a 0.1.1 ledger, or results written directly), it is the latest committed
  result per source id.
- `iter_results`, `iter_current` and `iter_history` yield in `source_id` order, so
  exports are stable across runs.
- `duplicate_targets` reports targets that several *current* source records selected,
  as `{matched_id: sorted source ids}`. Reported, never resolved — deduplication is a
  domain decision, not a storage one.
- Every write method raises `LedgerClosedError` after `close()`.
- `journal_mode` reads the live PRAGMA, so a test (or a suspicious operator) can
  assert WAL actually took effect on the filesystem in use.

## Gotchas

- **`FAILED` is not `UNMATCHED`.** Infrastructure failure says nothing about the
  record. `derive_status` only emits `FAILED` when *every* attempt failed, and a
  `FAILED`/`UNMATCHED` result always has `matched_id=None`.
- **Inexact resolution never auto-matches.** Even a score of 1.0 lands in
  `NEEDS_REVIEW` when the best attempt's resolution is anything but `"exact_key"` —
  legacy ID heuristics are tagged precisely so this rule can catch them.
- **The verifier's `"no_match"` counts as disagreement**, same as `"disagree"`;
  either routes the best attempt to review regardless of score.
- **`should_verify` is inclusive at both band ends**; `should_audit` starts strictly
  above the band's high end. With the defaults, a score of exactly `0.8` is
  verified, never audited.
- **Audit sampling is deterministic** per `(run_fingerprint, source_id)`. Resuming a
  run audits the same records; it does not re-roll.
- **Ties in `derive_status` go to the earlier attempt** — a retry that merely
  equals the first attempt's score does not displace it.
- **`RetrievalHit.rank` is 1-based.** Constructing with `rank=0` raises.
- **Template variables: `id` always wins.** A source field literally named `"id"`
  is shadowed by `record.id` in every template.
- **Missing template fields render as empty strings**, by design. Check the
  rendered query/context in the attempt trace before assuming your data has the
  field.
- **`hash_value` digests are 16 hex chars**, not a full SHA-256. Fine for
  fingerprints and resume keys; do not treat them as cryptographic commitments.
- **`result_key` requires all three parts.** Changing the run configuration *or*
  the source record's content produces a new key, which is exactly what makes
  resume skip only genuinely unchanged work.
- **`Usage` sums with plain `sum()`** thanks to `__radd__`; no start value needed.
- **`put_result` overwrites** on a repeated `result_key`. If you want history, it
  lives in the attempt trail of the new blob, not in old rows.
- **Only `put_result` and `put_cached` take the write lock.** `put_manifest` and
  `put_review` are synchronous and intended for single-writer moments (run start,
  review apply), not the hot concurrent path.
- **Exports are disposable.** `mapping.csv`, JSONL, and manifest files are
  regenerated from the ledger; the SQLite file is the only artifact that must
  survive.
- **`Attempt.finish_reason == "length"`** means the provider truncated the answer
  mid-JSON. It surfaces as `UNRESOLVED_OUTPUT` and is otherwise indistinguishable
  from a model that simply answered badly — check it before blaming the prompt.

# xwalk — a reusable LLM-RAG record matching library

**Date:** 2026-07-26
**Status:** Design approved (revised after review), ready for implementation planning
**Distribution and import name:** `xwalk`

`crosswalk` is taken on PyPI by an unrelated geographical package (0.0.0b0, July 2022), so the
name is settled now rather than deferred — renaming after examples, manifests, caches and config
schemas reference it would be disruptive. `xwalk` is the standard data-integration abbreviation
for crosswalk, is verified unclaimed on PyPI, and is not ontology-bound. Also verified free if a
change is wanted before any code exists: `matchbridge`, `linkset`, `recolink`.

## Problem

OntoRAG works, and its core loop is domain-agnostic: hybrid retrieval → rank fusion → LLM
selection → confidence-gated stopping → query reformulation. But the implementation is welded
to biomedical ontology normalization:

- Dataset-specific behaviour lives in `src/experiment/hooks.py` **inside the library**, so
  adding a dataset means editing library code.
- Each dataset config repeats 15 derivable paths.
- `src/pipeline.py:766` wires three near-duplicate provider branches.
- SapBERT is hardcoded as "the second dense model" — meaningless outside biomedicine.
- Ingestion knows only `owl` and `tsv`; `src/ingestion/run_dataset.py:41` checks
  `if dataset_key == "ncbi_gene"`.
- There is no "just match these records" entry point. `run_benchmark` requires gold labels.
- The repo is 11 GB of git history over 33 GB of data. Nobody will `pip install` that.

We want a library that makes the method reusable against **any two record collections**, with
the LLM choice, the prompts, and the retrieval stack all under the user's control.

## Phase 1 contract

> **Independently map each source record to zero or one candidate from a target collection,
> using retrieval and LLM-based disambiguation.**

Every word is load-bearing. *Independently* means no cross-record coordination. *Zero or one*
means abstention is a first-class outcome, not a failure.

### Explicit non-goals

- One source record mapping to **multiple** targets.
- **Composite** mappings (one source → a combination of targets) or **split** mappings.
- **Global one-to-one assignment**, or any solver that optimizes across the whole source set.
- **Deduplication within** a single collection.
- **Schema or column mapping** between the two collections.

Reporting is not solving: when several source records select the same target, that is surfaced
as a duplicate-target conflict for review, because duplicates are usually worth a human look.
The library reports the conflict and does nothing about it. A global assignment solver stays
outside the library.

Also out of scope, from the earlier round: reproducing the paper's exact numbers (this repo
stays frozen as the paper archive; the new library is clean-room), a web UI, error clustering
and t-SNE atlas (paper-only), an out-of-core storage engine (served by plugging in an existing
backend), and a composable stage DAG (flexibility comes from replacing a stage, not rebuilding
the graph).

## Goals

1. Match records from any source collection to any target collection, ontology or not.
2. Python SDK first: import, compose, run in a notebook. CLI is a thin wrapper.
3. Swap the LLM freely; swap or bring your own retrieval backend and target store.
4. Have an LLM draft domain prompts, then improve them against labelled examples.
5. Produce a **mapping table** with honest per-row status and reason, including a review bucket.
6. Base install is small and CPU-only. Heavy dependencies are opt-in extras.

## Decisions taken

| Question | Decision |
|---|---|
| What does a match compare? | Field-mapped documents on both sides; a bare mention is a one-field record |
| Primary interface | Python SDK; config files are serialized constructor arguments |
| Where does it live? | New clean repo. This repo stays frozen as the paper archive and does **not** depend on it |
| Prompt authoring | Contract-safe base skeleton with LLM-filled domain slots, then label-driven optimization |
| Retrieval layer | Built-in BM25 + N dense encoders, plus `Retriever` and `TargetStore` protocols for BYO backends |
| Paper coupling | None. Total refactor freedom; the four datasets port as examples |

## Architecture

### Object model, not registry

The extension contract is a protocol, never a name in a table the library owns. Adding a data
source means writing a function in *your* code:

```python
def my_source() -> Iterable[Record]:
    for row in my_database.query(...):
        yield Record(id=row.pk, fields={"name": row.name, "city": row.city})
```

This is what replaces `hooks.py`. Nothing in the library needs to know your dataset exists.

### Core types

```python
@dataclass(frozen=True)
class Record:
    id: str
    fields: Mapping[str, Any]

@dataclass(frozen=True)
class RetrievalHit:
    record_id: str
    retriever: str
    raw_score: float | None       # None when a backend exposes no comparable score
    rank: int

@dataclass(frozen=True)
class Candidate:
    record: Record
    fused_score: float
    evidence: tuple[RetrievalHit, ...]    # every retriever that surfaced it, with its rank
```

Splitting `RetrievalHit` from `Candidate` removes a real ambiguity: after fusion, a single
`source: str` field cannot express that three retrievers each contributed at different ranks.
`evidence` carries exactly that, and it is what the retrieval-ceiling diagnostic reads.

### Retrieval and storage are separate protocols

```python
@dataclass(frozen=True)
class SearchRequest:
    text: str                              # rendered query
    limit: int
    filters: Mapping[str, Any] | None = None
    source_record: Record | None = None     # enables field-aware retrieval later

class Retriever(Protocol):
    name: str
    fingerprint: str
    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]: ...

class TargetStore(Protocol):
    fingerprint: str
    def get(self, record_id: str) -> Record: ...
    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]: ...
```

Retrievers return IDs and ranks; the store resolves IDs to records. Keeping these apart is what
makes "plug in Elasticsearch for scale" actually work — otherwise `Target` still has to hold
every target record in memory, and the protocol buys nothing.

`SearchRequest` rather than `search(query, k)` means filters and field-aware retrieval can be
added later without a breaking protocol change. `fingerprint` on both protocols feeds run
fingerprinting and cache keys.

### Templates carry the field mapping

Four Jinja2 templates, rather than an invented mini-language — conditionals and truncation are
needed immediately (`{% if synonyms %}`), and Jinja is already familiar.

| Template | Input | Produces |
|---|---|---|
| `query` | source record | retrieval query string |
| `context` | source record | context block shown to the LLM |
| `doc` | target record | indexed text (overridable per retriever) |
| `candidate` | target record | one line of the candidate list |

Ontology use sets `query: "{{ mention }}"`. Row-linkage sets `query: "{{ name }} {{ city }}"`.
Same engine, different templates. Rendering is pure and unit-testable.

### Opaque candidate keys

Candidates are presented to the model under temporary keys assigned per attempt:

```
[C01] ID: CHEBI:1234
      Label: glucose
      Synonyms: dextrose; grape sugar

[C02] ID: CHEBI:5678
      Label: fructose
```

The model must return `C01`, `C02`, … or `null`. Resolution is an exact dictionary lookup
against the keys issued for that attempt; anything else is `UNRESOLVED_OUTPUT`.

This replaces the existing heuristic resolver (`src/pipeline.py:67`), which is well-crafted for
absorbing research-pipeline output variance but is unsafe as the default for general mapping
software:

- Its final branch treats a bare integer as a 1-based candidate rank. With numeric target IDs
  (`NCBIGene:3`), a hallucinated ID that is not in the candidate set falls through to that
  branch and **silently becomes a different, real mapping**.
- Stripping CURIE prefixes can create collisions across namespaces.
- Case-insensitive matching is invalid for identifier schemes that are case-sensitive.

Opaque keys also remove the sentinel muddle: `null` is unambiguous where `"-1"` and `"0"` are
values some identifier scheme might legitimately use.

The heuristic resolver survives as `legacy_id_resolution=True`, off by default, for reproducing
prior work. In that mode **any non-exact resolution forces `NEEDS_REVIEW`** and the resolution
path is recorded on the attempt.

### Selector budget

Retrieval depth `k` belongs to each retriever's own configuration — a BM25 and a dense retriever
have no reason to share a depth. But with N retrievers the fused list can grow arbitrarily, so a
separate constraint governs how much of it reaches the model:

```python
@dataclass(frozen=True)
class SelectorPolicy:
    max_candidates: int = 30
    max_candidate_tokens: int = 8_000
```

This is not a retrieval-depth setting. Truncation is deterministic (fused score, then a stable
tie-break on record ID) and the attempt trace records how many candidates were dropped.

**This interacts with the ceiling diagnostic and must not be conflated with it.** If the gold
record was retrieved but cut by the selector budget, that is a *budget* miss, not a *retrieval*
miss, and the fix is a larger budget rather than a better retriever. `evaluate.ceiling` reports
the three buckets separately: never retrieved / retrieved but truncated / presented and misjudged.

### Status and reason are separate

```python
class MatchStatus(Enum):
    MATCHED
    NEEDS_REVIEW
    UNMATCHED
    FAILED

class DecisionReason(Enum):
    ACCEPT_THRESHOLD
    BELOW_ACCEPT_THRESHOLD
    BELOW_REVIEW_FLOOR
    SELECTOR_ABSTAINED
    NO_CANDIDATES
    UNRESOLVED_OUTPUT
    RETRIEVER_FAILURE
    PROVIDER_FAILURE
    VERIFIER_DISAGREEMENT
```

A single `REJECTED` status cannot distinguish rejected-by-model from rejected-by-policy from
rejected-by-human. Recording the outcome (`reason`) separately from its classification
(`status`) means classification policy can change later — retune `accept_at`, reclassify an
existing run — without having destroyed the original outcome. A weak candidate becomes
`UNMATCHED / BELOW_REVIEW_FLOOR`; an explicit abstention becomes `UNMATCHED / SELECTOR_ABSTAINED`;
the two are no longer indistinguishable in analytics.

```python
@dataclass(frozen=True)
class Attempt:
    index: int
    query: str
    proposal: RetryProposal | None          # what prompted this attempt; None for the first
    candidates: tuple[Candidate, ...]
    candidates_truncated: int               # dropped by the selector budget
    issued_keys: Mapping[str, str]          # opaque key → record ID, for audit
    raw_selection: str | None               # exactly what the model returned
    chosen_id: str | None
    resolution: str                         # exact_key | legacy_* | unresolved
    primary_score: float | None
    verifier_decision: str | None           # support | disagree | no_match
    verifier_score: float | None
    audited: bool
    reason: DecisionReason | None
    error: str | None
    usage: Usage

@dataclass(frozen=True)
class MatchResult:
    result_key: str
    source_id: str
    matched_id: str | None
    matched_record: Record | None
    confidence: float | None
    status: MatchStatus
    reason: DecisionReason
    explanation: str
    candidates: tuple[Candidate, ...]
    attempts: tuple[Attempt, ...]
    usage: Usage
    run_fingerprint: str
```

### Module layout

```
src/xwalk/
  records.py         Record, RetrievalHit, Candidate, MatchResult, Attempt, Usage   ~160
  templates.py       Jinja2 rendering of the four templates                          ~80
  fingerprint.py     run + component fingerprinting, canonical record hashing       ~120
  stores/
    base.py          TargetStore protocol                                            ~40
    memory.py        in-memory store built from a RecordSource                      ~120
  sources/
    __init__.py      RecordSource protocol (= Iterable[Record])
    tabular.py       csv / tsv / jsonl / parquet                                    ~120
    ontology.py      owl / obo via rdflib                            [ontology]     ~200
    sql.py           SQLAlchemy URL → rows                           [sql]           ~80
  retrieval/
    base.py          Retriever protocol, SearchRequest                               ~70
    bm25.py          index build + search                                           ~180
    dense.py         encoder + FAISS                                  [dense]       ~200
    fusion.py        reciprocal rank fusion → Candidate with evidence                ~80
  llm/
    base.py          LLMClient protocol, LLMCapabilities, Usage                      ~90
    openai_compat.py structured output, capability probing, backoff                 ~260
    litellm.py       optional adapter                                 [litellm]      ~60
    parsing.py       thinking-block stripping, JSON extraction/repair               ~120
  stages/
    keying.py        opaque candidate keys, exact resolution, legacy mode           ~110
    select.py        Selector                                                       ~110
    gate.py          scorer + verifier verdicts                                     ~180
    rewrite.py       QueryRewriter                                                  ~100
    proposals.py     RetryProposal validation and routing                           ~110
  matcher.py         the loop — orchestration only                                  ~260
  ledger.py          SQLite run ledger: results, cache, review overlay              ~220
  batch.py           run a source, resume, export mapping table                     ~200
  review.py          export / apply immutable review overlay                        ~150
  prompts/
    base/*.j2        contract-safe skeletons
    contract.py      placeholder + output-schema validation                         ~100
    author.py        draft slots from samples                                       ~150
    optimize.py      label-driven slot optimization                                 ~220
  evaluate/
    gold.py          gold loading + user-supplied normalization                     ~100
    metrics.py       operational metric suite                                       ~200
    ceiling.py       retrieval / budget / judgement decomposition                   ~140
    ablate.py        flag-flipping wrapper over eval                                 ~80
    compare.py       run comparison table                                            ~80
  config.py          job spec → objects (pydantic)                                  ~200
  cli/               index · match · draft · optimize · eval · compare · review     ~280
```

Roughly 5k lines against today's 7.1k, with `matcher.py` at ~260 against `pipeline.py`'s 938.
No file large enough to be awkward to hold in context or to review.

### Dependency tiers

Base: `jinja2`, `httpx`, `pyyaml`, `pydantic`, `tantivy`. No torch.
Extras: `[dense]` (sentence-transformers, torch, faiss-cpu), `[ontology]` (rdflib),
`[sql]` (sqlalchemy), `[litellm]`, `[all]`.

Matching two CSVs with an API-hosted model installs none of the heavy stack.

**BM25 engine — Tantivy, with a defined support matrix rather than a silent fallback.** A silent
switch between Tantivy and Whoosh would change ranking behaviour and quietly undermine
reproducibility, so the spec fixes:

- Minimum Python 3.10 (Tantivy's floor).
- Supported matrix: manylinux x86-64 and aarch64, macOS arm64, Windows x86-64.
- A CI install smoke test per matrix entry that builds a tiny index and asserts a known ranking.
- A *documented, explicitly selected* alternate retriever for unsupported platforms. Never
  automatic.

### LLM providers: compatibility is a capability, not a guarantee

One `LLMClient` protocol; one well-built OpenAI-compatible client as the default. But
interchangeability is genuinely partial — Google offers an OpenAI-compatible Gemini endpoint
while recommending the native API for anyone not already committed to OpenAI libraries, and
Anthropic describes its compatibility layer as primarily for testing rather than production, and
notes strict schema enforcement may be ignored. So adapters declare what they actually do:

```python
@dataclass(frozen=True)
class LLMCapabilities:
    json_schema: bool
    strict_schema: bool
    usage_reporting: bool
    seed: bool
    native_retry_after: bool
```

**Runtime validation always runs regardless of declared support.** A provider claiming
`strict_schema=True` still gets its output parsed and validated; the claim only decides whether
to *ask* for structured output, never whether to *trust* the answer. Native adapters can be
added later without touching stage interfaces.

Three pieces of hard-won behaviour port over from the current code:

- Structured output via `response_format: json_schema`, with graceful fallback when a provider
  rejects it (`src/components/openrouter_client.py:231`).
- Reasoning-block stripping for `<think>` / `<thinking>` / `<reasoning>`, including the case
  where a model emits only a closing tag (`src/utils/response_parsing.py`).
- Exponential backoff on 408/409/429/5xx, honouring `Retry-After` where the adapter declares it.

### Async with a sync facade

The core is async. `Matcher.match_sync()` and `run_batch_sync()` exist so notebook users are not
forced into asyncio.

## The matching loop

`Matcher.match(record) -> MatchResult`. Per attempt:

1. Render `query` and `context` from the source record.
2. Search every configured retriever concurrently with a per-retriever timeout (60 s default).
   Fuse hits by RRF into `Candidate`s carrying their evidence. Resolve records via `TargetStore`.
3. No candidates → record the attempt, take the next queued query proposal.
4. Assign opaque keys, apply the selector budget, render the candidate list.
5. Selector returns a key or `null`, with confidence and explanation. Resolve the key exactly.
6. The gate scores the choice **against the full rendered source record and context**, not the
   retrieval query. The query is deliberately lossy — for a multi-field source it may omit the
   very fields needed to disambiguate — so scoring against it evaluates the wrong thing.
7. If the score falls in `verify_band`, run verification (below).
8. Score ≥ `accept_at` → return immediately. Otherwise collect retry proposals and continue.

On exhaustion, the best attempt by primary score is returned.

### Retry proposals are two different things

The current design funnels scorer alternatives and rewritten queries into one queue. They are
not the same object, and conflating them causes real waste: `prompts/chemical_confidence.tpl`
asks the scorer for alternatives **"from `Other Top Candidates`"** — already-retrieved records —
and `src/pipeline.py:667` then pushes those labels into `queries_to_try` and re-runs full
retrieval on them.

```python
@dataclass(frozen=True)
class RetryProposal:
    kind: Literal["candidate", "query"]
    value: str
    source: Literal["scorer", "rewriter"]
```

- `kind="candidate"` — evaluate that candidate against the source record **without re-running
  retrieval**. It is already in hand.
- `kind="query"` — a genuinely new search string; only these enter the query queue.

A candidate proposal naming something not in the current candidate set is a hallucination, not a
new lead: it is dropped and recorded on the attempt rather than silently promoted to a query.

### Verification is a verdict, not an aggregation

Taking `min(primary, verifier)` of two uncalibrated scores is conservative but arbitrary, and it
cannot distinguish agreement-with-different-numbers from genuine disagreement. When verification
runs, it returns an independent verdict:

```json
{
  "decision": "support | disagree | no_match",
  "preferred_key": "C03",
  "confidence_score": 0.72,
  "explanation": "..."
}
```

`disagree` or `no_match` forces `NEEDS_REVIEW / VERIFIER_DISAGREEMENT` regardless of numerical
aggregation. When the verifier names a different `preferred_key`, that key is **recorded and
flagged, not chased** — re-entering selection on the verifier's preference would make loop
termination depend on two models negotiating. Bounded loop, honest flag, human decides.

**Optional random audit.** Because high-confidence errors exist and are by definition invisible,
`audit_rate` (default 0.0) verifies that fraction of otherwise-automatic high-confidence matches.
Set to 0.02 it costs ~2% extra verification calls and yields an ongoing estimate of silent error
rate, rather than doubling verification cost across the board. Audited attempts set
`audited=True` so the estimate can be computed over the sampled subset.

An audit verdict is **honoured, not merely counted**: if the auditor disagrees, that result goes
to `NEEDS_REVIEW / VERIFIER_DISAGREEMENT` like any other disagreement. Sampling decides *which*
results get a second opinion, never whether the opinion counts. Sampling is seeded from
`(run_fingerprint, source_id)` so audits are reproducible across resumes rather than re-rolled.

### Policy

```python
@dataclass(frozen=True)
class MatchPolicy:
    max_attempts: int = 4
    accept_at: float = 0.6
    review_floor: float = 0.4
    verify_band: tuple[float, float] | None = (0.6, 0.8)
    audit_rate: float = 0.0
    concurrency: int = 32
    legacy_id_resolution: bool = False
```

`verify_band` is a **cost control** — buy a second opinion only when the first is uncertain.
`accept_at` / `review_floor` are **classification**. The current code conflates these; separating
them lets each be tuned without disturbing the other.

Status is derived from the best attempt in this precedence order, so mixed outcomes across
attempts resolve deterministically:

1. Every attempt failed with an infrastructure error → `FAILED` (`RETRIEVER_FAILURE` or
   `PROVIDER_FAILURE`).
2. Verification disagreed → `NEEDS_REVIEW / VERIFIER_DISAGREEMENT`.
3. A resolvable scored choice exists → classify by final score: `≥ accept_at` →
   `MATCHED / ACCEPT_THRESHOLD`; `[review_floor, accept_at)` →
   `NEEDS_REVIEW / BELOW_ACCEPT_THRESHOLD`; `< review_floor` →
   `UNMATCHED / BELOW_REVIEW_FLOOR`.
4. No resolvable choice, selector abstained → `UNMATCHED / SELECTOR_ABSTAINED`.
5. No candidates were ever retrieved → `UNMATCHED / NO_CANDIDATES`.
6. Candidates existed but output never resolved → `NEEDS_REVIEW / UNRESOLVED_OUTPUT`.

A low-confidence pick outranks an earlier explicit abstention, which is correct: the model found
*something* worth surfacing. Unresolvable output routes to review rather than to silence,
because a malformed answer is a signal, not a non-match.

## Batch matching — the mapping platform

### Identity and resume

Resume must not key on `source_id` alone. The same ID may carry changed field values, or have
been processed under a different prompt, target snapshot, retriever set or model. The result key
is:

```
result_key = hash(run_fingerprint, source_id, canonical_source_record_hash)
```

`run_fingerprint` covers target snapshot and index fingerprints, prompt and slots hashes, model
and provider endpoint and generation parameters, retrieval configuration, matching policy, and
library version. Change any of them and prior results correctly stop matching — you get a fresh
run instead of a silently mixed one.

Likewise `(model, rendered_prompt)` is not a sufficient LLM cache key. The key covers the
complete request body, the schema, provider identity, model parameters, and adapter version.

### SQLite as the run ledger

Concurrent append to JSONL invites partial final lines and duplicate records, which complicates
exactly the resume logic that must be trustworthy. Instead:

- SQLite (WAL mode) stores one JSON result blob per `result_key`, plus cache entries and the
  review overlay.
- A single writer coroutine serialises writes; WAL keeps concurrent readers working.
- `results.jsonl`, `mapping.csv` and `manifest.json` become **exports**, regenerated from the
  ledger on demand.

This makes execution state transactional and resumable without turning the library into an
out-of-core matching engine.

```python
report = run_batch(matcher, source, out=Path("run/"), resume=True)
report.needs_review()
report.duplicate_targets()      # several source records chose the same target
report.by_status()
```

### Human review is an immutable overlay

Applying review never overwrites model output. Three layers are preserved: the original model
result, the reviewer decision, and the adjudicated result derived from both.

Exported review rows carry `result_id`, `run_fingerprint`, `source_id`, `source_hash`,
`proposed_target_id`, `decision`, `corrected_target_id`, `reviewer`, `review_note`,
`reviewed_at`. Decisions are `accept`, `reject`, `replace`, `no_match`, `defer`.

**Applying a review fails if its source or target snapshot no longer matches the originating
run.** A decision made against different data is not a decision about this data, and silently
applying it would corrupt the mapping in the least detectable way possible.

```
xwalk review export --run run/ --out review.csv
xwalk review apply  --run run/ --reviewed review.csv
```

Confirmed pairs can be emitted as a gold file for the prompt optimizer, closing the loop from
curation back into prompt improvement.

## Prompt authoring and optimization

### Contract-safe skeletons

`prompts/base/{select,score,rewrite}.j2` own the fixed structure: role line, the four rendered
input blocks, the opaque-key candidate list, rubric table, hard-rules section, and the JSON
output contract. Domain content lives in a separate slots file:

```yaml
# prompts/chemistry/slots.yaml
entity_noun: "chemical entity mention"
target_noun: "ChEBI ontology term"
domain_brief: "biomedical chemistry nomenclature"
rubric:
  - {score: 1.0, name: Certain, when: "exact case-insensitive match to label or synonym",
     example: "garlic → Garlic"}
  - {score: 0.9, name: High, when: "normalized form or common abbreviation", example: "..."}
  - {score: 0.6, name: Plausible, when: "specific instance of the candidate", example: "..."}
  - {score: 0.4, name: Speculative, when: "related by broad category only", example: "..."}
hard_rules:
  - "For regulated substances, any difference in name or number means distinct entities"
disambiguation_steps: |     # optional free-text, e.g. the gene prompt's organism logic
  ## Step 1: Use context for organism disambiguation
```

`contract.py` validates before anything is saved: renders against a synthetic record, asserts
every input block appears exactly once, asserts the output block parses as the expected JSON
schema, asserts the key instruction is intact, and asserts rubric scores are monotone and span
`[0, 1]`.

### Drafting

`xwalk prompts draft --job job.yaml --describe "matching legal citations to court decisions"`

Samples N records from each side, sends them with the description to a strong model, receives
**slots only** — never raw prompt text — validates, shows a diff, writes. A bad draft can produce
a poor rubric but never a broken prompt.

### Optimization uses three partitions

Revising prompts from dev failures *and* selecting the retained round on dev performance makes
dev part of training. Three partitions:

| Partition | Role |
|---|---|
| **prompt-train** | failures shown to the optimisation model |
| **validation** | chooses the retained round and the stopping point |
| **test** | evaluated **once**, after the final prompt is selected |

Test is not reported per round. Repeated observation would turn it into a second validation set,
which is the same mistake one level up.

Failure selection depends on which role is being optimised — "only selection failures" is correct
for the selector and wrong for everything else:

| Role | Useful optimisation cases |
|---|---|
| Selector | gold retrieved and presented, wrong candidate chosen |
| Scorer / gate | incorrect automatic accepts, and unnecessary abstentions |
| Rewriter | gold absent initially but recoverable through reformulation |
| Retrieval template | gold never surfaced despite being present in the target |

This is why the ceiling decomposition belongs in the library rather than in paper-only analysis:
the optimiser depends on it to select the right failures for the role being tuned.

Cost is real, so the command prints estimated call count and spend before starting and accepts
`--max-calls`.

## Evaluation and diagnostics

Gold labels are `source_id, gold_ids` (pipe-separated for multi-gold) in CSV or JSONL. ID
normalization and alias expansion — today's `normalize_ncbi_gene_prediction` and
`expand_ctd_eval_ids` — become a user-supplied callable on the job, not library-owned hooks.

**`evaluate.metrics`** reports operational numbers, not just exact-match accuracy:

- accepted precision and automatic-match coverage
- review rate
- unresolved and error rate
- no-match precision and recall, where gold no-match labels exist
- duplicate-target conflict count
- cost and latency per completed source record
- threshold curves, with a calibration warning when confidence bands are poorly separated

**`evaluate.ceiling`** decomposes failures three ways — never retrieved / retrieved but cut by
the selector budget / presented and misjudged — per retriever and per attempt. It is nearly free
because the loop already records `evidence` on every candidate, it tells a new user whether to
spend effort on retrieval, on budget, or on prompts, and the optimiser consumes it directly.

**`evaluate.ablate`** re-runs eval with each component disabled and reports the delta.

**`evaluate.compare`** tables accuracy, cost and latency across runs or models. This is how you
actually "select an LLM".

## Error handling

- Every stage failure is recorded as an `Attempt` with its reason and message. All attempts
  failing yields `FAILED`, never `UNMATCHED` — a provider 500 is not evidence of a non-match.
- Per-retriever timeout: one retriever timing out degrades to the others rather than failing
  the match, and the degradation is recorded.
- Provider errors retry with backoff on 408/409/429/5xx; non-retryable errors fail the attempt.
- Structured-output rejection falls back to prompt-only JSON with robust parsing, always
  validated regardless of declared capability.
- Malformed LLM output goes through strip-thinking → extract-JSON → repair; if all fail, the
  attempt is `UNRESOLVED_OUTPUT` and routes to review.
- Batch runs never lose completed work: every result is committed to the ledger as it finishes.

## Testing

- **Pure units, no network:** template rendering, key assignment and exact resolution, RRF and
  evidence assembly, selector-budget truncation determinism, status/reason derivation, contract
  validation, fingerprint stability, thinking-strip and JSON repair.
- **`FakeLLM`** implementing `LLMClient` with scripted responses and declared capabilities, so
  the entire matcher loop — retry paths, proposal routing, verify band, disagreement, audit
  sampling, exhaustion, error handling — is testable offline. Today's pipeline cannot be tested
  without a provider; this is the single biggest testability win.
- **Adversarial resolution tests** covering exactly the cases opaque keys exist to prevent: a
  numeric hallucinated ID, an ID differing only by case, an ID valid in another namespace, and a
  key not issued this attempt. Each must resolve to `UNRESOLVED_OUTPUT`, never to a real record.
- **Ledger tests:** resume after simulated crash mid-run, fingerprint change forcing a fresh run,
  concurrent writer safety, review-apply rejection on snapshot mismatch.
- **Tiny fixture target** of ~50 records as CSV, so retrieval tests run in milliseconds with
  BM25 only and no torch.
- **Integration:** one marked test against a real provider, skipped when no API key is present.
- **Examples as regression:** the four datasets port to `examples/*/job.yaml` with small sample
  slices committed.
- CI on the declared platform matrix with base + `[ontology]` extras; ruff and mypy.

## Migration of the four datasets

| Dataset | What it exercises |
|---|---|
| CRAFT ChEBI | OWL loader, the simple one-field source case |
| NCBI Disease | user-supplied gold alias expansion (today's `expand_ctd_eval_ids`) |
| NLM-Gene | TSV loader, row filtering, multi-field disambiguation, numeric IDs — the case that motivates opaque keys |
| CafeteriaFCD | second OWL domain, showing prompt slots are the only domain-specific part |

Sample slices are committed; full data stays in this repo.

## Build order

Too large for a single implementation plan. Three phases, each ending at something usable.

**Phase 1 — the mapping platform** (first implementation plan)

1. Core types, fingerprinting, templates, `FakeLLM`, tests — the spine everything attaches to.
2. Sources (tabular first), in-memory `TargetStore`, BM25 retriever, fusion.
3. `LLMClient` + OpenAI-compatible client with capabilities, keying, stages, `Matcher`. First
   end-to-end match, driven by hand-written slots rendered through the base skeletons. The
   skeletons and `contract.py` exist from this point; only *LLM-assisted authoring* is deferred.
4. Ledger, `batch.py`, exports, review overlay. **A user can point at two CSVs and get a
   reviewed mapping table at the end of this phase.**

**Phase 2 — knowing whether it works**

5. Evaluation: gold loading, operational metrics, three-way ceiling decomposition.
6. LLM-assisted prompt drafting into slots.
7. Prompt optimization with three partitions and role-specific failure selection.

**Phase 3 — breadth**

8. Dense retrieval, ontology and SQL sources, ablate and compare, full CLI surface.
9. Port the four datasets as examples.

Each phase gets its own plan; phase 1's is written first, since phases 2 and 3 will be better
informed once the core types have survived contact with real code.

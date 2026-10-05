# The job file

A job file is serialized constructor arguments — nothing more. Every field maps to
something the SDK takes by hand: `templates` becomes a `TemplateSet`, `policy` becomes a
`MatchPolicy`, `retrievers[]` become retriever objects, `prompts.slots` becomes a
`PromptSet`. No field gates behaviour the SDK cannot express, and none of them is a
shortcut you could not have written in Python.

A job declares **exactly one** decision maker: `llm:` for the LLM path, or `decider:` for
the [decision-model path](decide.md). Declaring both, or neither, is rejected at load. Which
one is present also decides how `policy:` is parsed, and which keys in it mean anything.

`load_job(path)` returns a `JobSpec`, which is a pydantic model with builder methods:
`build_templates`, `build_target_records`, `build_source_records`, `build_store`,
`build_retrievers`, `build_llm`, `build_prompts`, `build_policy`, `build_selector_policy`,
`run_fingerprint` and `build_matcher` — plus `build_decider`, `build_questions`,
`build_decision_policy`, `decision_run_fingerprint` and `build_decision_matcher` on the
decider path. The CLI calls them in that order; so can you. A
`JobSpec` can also be constructed straight from a dict with `JobSpec.model_validate` — the
file is a convenience, not a gate.

Credentials never appear in the file. `api_key_env` names an environment variable, and an
inline `api_key` is rejected at load time.

## Validation is strict

Every section rejects keys it does not define, so a misspelling fails instead of silently
running with a default: `accept_att: 0.9` under `policy` is an error
(`policy.accept_att: unknown field; did you mean 'accept_at'?`), not an `accept_at` of 0.6.
Numbers are range-checked (thresholds within `[0, 1]`, `review_floor <= accept_at`, an
ordered `verify_band`, positive limits, attempts and concurrency), a field that does not
apply to the declared `kind` is rejected rather than ignored (`query_prefix` on a `bm25`
retriever, `url` on a `csv` collection), a key may not appear twice in one mapping (YAML
would keep the last value silently), retriever names must be unique (each owns an
index directory), and `llm.profile` must be a known profile. `load_job` raises
`JobValidationError`; its `issues` carry a code (`unknown_field`, `missing_field`,
`invalid_value`, `job_not_found`, `job_yaml_invalid`), a dotted location and a message,
and every issue is reported, not only the first.

`xwalk validate --job job.yaml` (or `ops.validate`) runs this plus the checks a file alone
cannot answer — referenced files exist, optional extras are installed, the credential
variable is set, both collections load with unique ids — without calling a model. See
[the CLI reference](cli.md#validate-alias-doctor).

## A complete job.yaml

```yaml
# Every relative path below resolves against THIS FILE's directory.
name: nlm_gene                       # appears in the run manifest as "job"

templates:
  # source record -> the retrieval query string. `queries:` (a list) is the general form.
  query: "{{ mention }}"
  # source record -> the context block shown to the model; omit for no context section
  context: "{{ context_left }}[{{ mention }}]{{ context_right }}"
  # target record -> the text that gets indexed. Fix this first when gold records are
  # never retrieved.
  doc: "{{ label }} {{ synonyms | join(' ') }} {{ definition }} {{ organism }}"
  # target record -> one entry in the candidate list the model sees
  candidate: |
    ID: {{ id }}
    Symbol: {{ label }}
    Organism: {{ organism }}
    {% if synonyms %}Synonyms: {{ synonyms | join('; ') }}{% endif %}
    {% if definition %}Description: {{ definition }}{% endif %}

target:
  kind: tsv                          # csv | tsv | jsonl | obo | owl | sql
  path: sample/targets.tsv
  id_column: id
  multivalue_columns: [synonyms]     # split on multivalue_sep into a list

source:
  kind: jsonl
  path: sample/mentions.jsonl
  id_field: mention_id               # jsonl uses id_field; csv/tsv/sql use id_column

retrievers:                          # at least one; results are fused by RRF
  - kind: bm25
    name: bm25                       # also the index subdirectory name
    limit: 30                        # this retriever's own retrieval depth
    exact_fields: [label, synonyms]  # whole-string hits get a large boost

llm:
  kind: openai_compat                # openai_compat | litellm
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY        # the NAME of a variable, never a key
  profile: openai                    # what the endpoint is asked for, not trusted for

prompts:
  slots: slots.yaml                  # see docs/reference/prompts.md

policy:
  max_attempts: 3
  accept_at: 0.6                     # >= this is an automatic match
  review_floor: 0.4                  # below this is unmatched, not review
  verify_band: [0.6, 0.8]            # buy a second opinion only in here; null disables
  concurrency: 8

selector:
  max_candidates: 25                 # how much of the fused list reaches the model
```

Shipped, runnable versions: `examples/chebi/job.yaml`, `examples/cafeteria_fcd/job.yaml`,
`examples/ncbi_disease/job.yaml`, `examples/nlm_gene/job.yaml`. Decider-path twins:
`examples/cafeteria_fcd/job_jev.yaml` and `examples/ref_zivila/jobs/foodon/job_jev.yaml`.

## Top level

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `name` | `str` | yes | — | job name; `xwalk match` writes it into the manifest as `job` |
| `templates` | mapping | yes | — | the four templates, below |
| `target` | mapping | yes | — | the collection being matched **to** |
| `source` | mapping | yes | — | the collection being matched **from** |
| `retrievers` | list | yes | — | at least one entry; fewer is rejected |
| `llm` | mapping | exactly one of these two | — | provider and model, for the LLM path |
| `decider` | mapping | exactly one of these two | — | decision model, for the [decider path](decide.md) |
| `prompts` | mapping | yes | — | `{slots: <path>}` |
| `policy` | mapping | no | all defaults | thresholds and cost controls; parsed by whichever path the job declares |
| `selector` | mapping | no | all defaults | candidate budget; LLM path only (rejected on a decider job) |
| `base_dir` | path | — | — | a `JobSpec` field set by `load_job`; rejected in the file |

## `templates`

Each is Jinja source, compiled on construction so a typo fails immediately with
`TemplateError`. Missing record fields render as the empty string rather than aborting a
100k-row run; `id` always refers to the record's identity, even if a field is also named
`id`.

| Field | Type | Required | Default | Input | Produces |
|---|---|---|---|---|---|
| `query` | `str \| None` | one of `query` / `queries` | `None` | source record | the retrieval query string |
| `queries` | `list[str]` | one of `query` / `queries` | `[]` | source record | several query strings, all of them searched |
| `context` | `str` | no | `""` | source record | the context block; empty omits the section |
| `doc` | `str` | yes | — | target record | the indexed text |
| `candidate` | `str` | yes | — | target record | one entry in the candidate list |

A file with neither `query` nor a non-empty `queries` is rejected at load with `templates
need a query or a non-empty queries list`.

`queries` is how you buy recall without touching the index. Every retriever runs **every**
rendered query and the results are fused by reciprocal rank, so a target surfaced by several
queries accumulates several votes:

```yaml
templates:
  queries:
    - "{{ mention_en }}"
    - "{{ name_slo }}"
    - "{{ aliases | join(' ') }}"
```

`render_queries` drops every query that renders empty and de-duplicates the rest, so listing
a field that is blank on most records costs nothing on those records. Both keys may be given;
`queries` is then what is searched, while `query` remains `TemplateSet.query` for anything
that wants a single string. A job giving only `query` gets `queries = ()` and retrieval falls
back to that one query — the single-query path, byte-identical to what it always was.

## `target` and `source`

Both take the same shape. Several fields apply only to some `kind` values.

| Field | Type | Required | Default | Applies to | Meaning |
|---|---|---|---|---|---|
| `kind` | `csv \| tsv \| jsonl \| obo \| owl \| sql` | yes | — | all | which loader to use |
| `path` | `str` | yes except `sql` | `None` | csv, tsv, jsonl, obo, owl | file path, resolved against the job directory |
| `id_column` | `str` | no | `"id"` | csv, tsv, sql | column holding the record id; removed from `fields` |
| `id_field` | `str \| None` | no | `None` | jsonl | JSON key holding the id; falls back to `id_column` |
| `multivalue_columns` | `list[str]` | no | `[]` | csv, tsv, sql | split into lists; blank parts dropped |
| `multivalue_sep` | `str` | no | `"\|"` | csv, tsv, sql | separator for the above |
| `url` | `str \| None` | required for sql | `None` | sql | SQLAlchemy URL |
| `query` | `str \| None` | required for sql | `None` | sql | the SELECT |
| `id_prefix` | `str \| None` | no | `None` | obo, owl | keep only ids with this prefix |
| `include_obsolete` | `bool` | no | `false` | obo, owl | keep deprecated terms |

`obo` and `owl` both emit the same field names — `label`, `synonyms`, `definition`,
`parents`, `obsolete` — so one `doc` template works against either. `owl` needs
`xwalk[ontology]`; `sql` needs `xwalk[sql]`; `obo` needs nothing.

## `retrievers[]`

| Field | Type | Required | Default | Applies to | Meaning |
|---|---|---|---|---|---|
| `kind` | `bm25 \| dense` | yes | — | both | which engine |
| `name` | `str \| None` | no | `None` | both | retriever name and index subdirectory: a letter or digit, then letters, digits, `.`, `_` or `-` (at most 64 characters), so it cannot leave the index directory |
| `limit` | `int` | no | `20` | both | this retriever's retrieval depth |
| `exact_fields` | `list[str] \| None` | no | `None` → `["label", "synonyms"]` | bm25 | fields whose whole-string value gets a large boost |
| `analyzer` | `default \| en_stem` | no | `default` | bm25 | tokenizer for the text field; `en_stem` adds ASCII folding and English stemming |
| `fuzzy_distance` | `int` | no | `0` | bm25 | edit distance (0 to 2) for extra fuzzy clauses on query terms of 5+ characters; `0` is off |
| `model` | `str \| None` | required for dense | `None` | dense | sentence-transformers model name |
| `device` | `str \| None` | no | `None` | dense | torch device |
| `query_prefix` | `str` | no | `""` | dense | prepended when encoding a query |
| `doc_prefix` | `str` | no | `""` | dense | prepended when encoding a document |
| `revision` | `str \| None` | no | `None` | dense | pinned model revision; unpinned is recorded as `"unknown"` in the index identity |
| `normalize` | `bool` | no | `true` | dense | unit-normalise vectors |

`limit` must be at least 1, and a field marked for one kind is rejected on the other. The
index identity stored with each built index covers the doc template, the target records and
these engine settings (for dense: model name, `revision`, dimension, `normalize`, both
prefixes and the max sequence length); `xwalk match` opens a matching index and refuses a
different one unless `--rebuild-index`.

Indexes are written to `<index_dir>/<name or kind>`. An unnamed `bm25` retriever is built
as `bm25`; an unnamed `dense` retriever names itself `dense:<model>` even though its index
lands in `<index_dir>/dense`.

`limit` reaches the retriever object as `default_limit`, and the matcher reads it per
attempt — so with two retrievers, each searches to its own depth. The matcher's own
`retriever_limit` is set to `max(limit)` across the specs and is only a fallback for a
backend that declares no depth of its own; it does feed the run fingerprint.

## `llm`

| Field | Type | Required | Default | Applies to | Meaning |
|---|---|---|---|---|---|
| `kind` | `openai_compat \| litellm` | no | `openai_compat` | both | which adapter |
| `model` | `str` | yes | — | both | model identifier |
| `base_url` | `str \| None` | required for `openai_compat` | `None` | openai_compat | endpoint root, without `/chat/completions` |
| `api_key_env` | `str \| None` | no | `None` | both | **name** of the environment variable holding the key |
| `api_key` | — | never | `None` | — | present only so it can be rejected |
| `profile` | `str` | no | `"unknown"` | openai_compat | capability profile, below |
| `temperature` | `float` | no | `0.0` | both | sent on every matching request; recorded in the fingerprint |
| `max_tokens` | `int` | no | `1024` | both | sent on every matching request; recorded in the fingerprint |
| `seed` | `int \| None` | no | `None` | both | sent when the adapter declares `seed` support; recorded in the fingerprint when set |

Valid `profile` values are `openai`, `anthropic-compat`, `google-compat`, `vllm` and
`unknown`. The profile decides only what the adapter **asks** for — strict JSON schema,
seeding, `Retry-After` — never whether it trusts the answer; every response is parsed and
validated regardless. `unknown` asks for nothing, which always works.

`kind: litellm` needs `xwalk[litellm]` and ignores `base_url` and `profile`; only `model`,
`temperature`, `max_tokens` and `seed` reach `LiteLLMClient`, and its capabilities default to
all-false.

## `decider`

Present instead of `llm:` when the job runs the [decision-model path](decide.md). Becomes a
`JevClient` via `build_decider()`.

```yaml
decider:
  kind: jev
  model: "~typesafe/jev-latest"      # pin typesafe/jev-1.13 for a reproducible run
  base_url: https://openrouter.ai/api/alpha/decisions
  api_key_env: XWALK_TEST_API_KEY    # the NAME of a variable, never a key
```

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `kind` | `jev` | no | `"jev"` | the only adapter there is; any other value is rejected at load |
| `model` | `str` | yes | — | model identifier as the endpoint spells it |
| `base_url` | `str` | no | `https://openrouter.ai/api/alpha/decisions` | the **full** URL; no path is appended |
| `api_key_env` | `str \| None` | no | `None` | **name** of the environment variable holding the key |
| `api_key` | — | never | `None` | present only so it can be rejected |
| `timeout` | `float` | no | `60.0` | per-request timeout, seconds |
| `max_retries` | `int` | no | `5` | retries on 408, 429, 5xx, 529 and transport errors |

`base_url` is posted to exactly as written, because OpenRouter's path is
`/api/alpha/decisions` and TypeSafe's own is `/v1/systemone`. Pointing the job at a
different host changes the decider fingerprint and therefore the run fingerprint; rotating
the key does not.

`timeout` and `max_retries` are excluded from the run fingerprint — they change how fast an
answer arrives, never what it means — so tuning them does not invalidate a resume.

### `decider.rewrite`

Optional, off by default. After the first screen, a record that found nothing worth
choosing from (no candidates, or a best probability below `policy.screen_floor`) gets one
second round: an LLM proposes alternative queries, they are retrieved and screened, and the
two screens are merged by record id (see [decide.md](decide.md#decisionmatcher)). The LLM
writes query strings only; it never sees the screened candidates and never decides.

```yaml
decider:
  kind: jev
  # ...
  rewrite:                       # absent means no rewrite
    llm: {kind: openai_compat, model: qwen/qwen3-next-80b-a3b-instruct,
          base_url: https://openrouter.ai/api/v1, api_key_env: XWALK_TEST_API_KEY}
    max_queries: 3               # 1 to 5
```

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `llm` | [`llm` block](#llm) | yes | — | the rewrite model; same fields and the same inline-`api_key` rejection as the top-level `llm:` |
| `max_queries` | `int`, 1 to 5 | no | `3` | most alternative queries asked for per miss |

The prompt is the `rewrite` skeleton rendered from the job's `prompts.slots`, as on the LLM
path. `build_rewrite_llm()` builds the client; an unset `api_key_env` there fails with
`change decider.rewrite.llm.api_key_env in the job file`. The block's model, sampling
parameters, `max_queries` and the prompts are part of the run fingerprint (only when the
block is present, so a job without it keeps its fingerprint and its cached ledger).

On the four gold samples the rewrite did not pay: see the spec's A3 result
(`docs/superpowers/specs/2026-09-23-jev-recall-and-cost-design.md`). The shipped jobs do not
set it.

`xwalk match` (`ops.run`) always wraps the built client in a `CachingDecider` over the
run's own ledger.
There is no job-file key for that and it is not an optimisation: the model's probabilities
jitter between identical calls, and a resumed run must see the answers the first run saw.

## `policy`

Applies when the job declares `llm:`. Becomes a `MatchPolicy`. `verify_band` is a *cost
control* — buy a second opinion only when the first is uncertain. `accept_at` and
`review_floor` are *classification*. For a job with `decider:`, see
[the next section](#policy-on-the-decider-path) instead — the same YAML key, a different
model.

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `max_attempts` | `int` | no | `4` | retrieve → select → score passes per record; ≥ 1 |
| `accept_at` | `float` | no | `0.6` | score ≥ this becomes `matched` |
| `review_floor` | `float` | no | `0.4` | score in `[review_floor, accept_at)` becomes `needs_review`; below is `unmatched` |
| `verify_band` | `[float, float] \| null` | no | `[0.6, 0.8]` | verify when the score lands inside; `null` never verifies |
| `audit_rate` | `float` | no | `0.0` | fraction of high-confidence matches sampled for a second opinion |
| `concurrency` | `int` | no | `32` | records in flight; ≥ 1 |
| `legacy_id_resolution` | `bool` | no | `false` | accept non-exact key resolution; such matches never become automatic |
| `retriever_timeout` | `float` | no | `60.0` | seconds before a retriever is treated as failed |

The invariant is `0 <= review_floor <= accept_at <= 1`. Audits are seeded from
`(run_fingerprint, source_id)`, so the same records are audited across a resume rather than
re-rolled.

## `policy` on the decider path

The same `policy:` key, routed to `DecisionPolicySpec` instead of `PolicySpec` whenever the
job declares `decider:`. Becomes a `DecisionPolicy`. None of the LLM path's keys —
`max_attempts`, `review_floor`, `verify_band`, `audit_rate`, `legacy_id_resolution` — exists
here, and as everywhere unknown keys are rejected at load, so a `verify_band` left over from
a copied job file fails with `policy.verify_band: unknown field` instead of doing nothing.

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `screen_floor` | `float` | no | `0.30` | best screen probability below this → `unmatched`, before choose or gate is paid for |
| `shortlist_size` | `int` | no | `15` | at most this many screen survivors reach choose |
| `shortlist_floor` | `float` | no | `0.20` | a survivor needs at least this screen probability |
| `none_at` | `float` | no | `0.70` | `p_none` at or above this → `unmatched`; below it, `needs_review` |
| `choose_at` | `float` | no | `0.50` | `p_choice` below this → `needs_review` |
| `accept_at` | `float` | no | `0.85` | screen probability of the chosen record required for `matched` |
| `rubric_floor` | `float \| null` | no | `null` → `levels - 2` | rubric level below this → `needs_review` |
| `property_floor` | `float` | no | `0.50` | any declared property below this → `needs_review` |
| `chunk_size` | `int` | no | `50` | candidates per screen call |
| `max_candidates` | `int` | no | `300` | fused candidates kept after retrieval |
| `concurrency` | `int` | no | `32` | records in flight; ≥ 1 |
| `retriever_timeout` | `float` | no | `60.0` | seconds before a retriever is treated as failed |

Every probability field must be in `[0, 1]` (`rubric_floor` is a rubric level, so only
`>= 0`), `shortlist_floor` must not exceed `accept_at`, `retriever_timeout` must be
positive, and `shortlist_size`, `chunk_size`, `max_candidates` and `concurrency` must each
be at least 1. As on the LLM path these are checked at load (`xwalk validate` reports
them without a model call).

`selector:` is rejected on this path: the candidate budget is `max_candidates` here, and
there is no rendered candidate list to fit into a context window. The defaults are starting
points, not fitted values — fit `accept_at` and `property_floor` with
[`xwalk fit`](cli.md#fit).

## `selector`

LLM path only. Becomes a `SelectorPolicy`. This is not retrieval depth — each retriever owns its own — but
the separate constraint on how much of the fused list reaches the model.

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `max_candidates` | `int` | no | `30` | candidates kept, best fused score first |
| `max_candidate_tokens` | `int` | no | `8000` | crude character-based budget on the rendered list |

If `eval` reports gold records in the **truncated** bucket, this is the knob: the record was
retrieved and the budget cut it before the model saw it.

## `prompts`

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `slots` | `str` | yes | — | path to a slots YAML file, resolved against the job directory |

See [prompts.md](prompts.md) for the slot reference. Both paths read the same file: the LLM
path renders it into the four skeletons, the decider path composes it into typed questions.

### `properties`, in the slots file

One slots key exists only for the decider path, and the LLM skeletons ignore it. Each entry
declares an identity-bearing property the gate stage checks on the chosen candidate,
producing one agreement probability per entry — which is what `property_floor` thresholds
and what a reviewer reads instead of a generated explanation.

```yaml
properties:
  - name: processing_state
    question: >-
      Do the source and the candidate agree on processing state (raw, cooked, dried,
      canned, frozen, smoked, fermented, juice, oil, flour, jam) wherever either states one?
```

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `name` | `str` | yes | — | must match `^[a-z][a-z0-9_]*$`; becomes the `prop_<name>` signal |
| `question` | `str` | yes | — | non-empty; asked about `source` and `candidate` together |

The list defaults to empty, so every existing slots file stays valid and a decider job
without it simply has no property gate. Phrase each question so that "neither record states
this" is a **yes** — the rendered instruction says so explicitly, and a property nobody
mentions must not block a correct match. See [decision models](decide.md#the-properties-block).

## Clustering jobs (`kind: cluster`)

A file with `kind: cluster` at the top level is a clustering job for `xwalk cluster`
(experimental). It has one collection (`source`), no `target`, `retrievers` or
`prompts`; `templates` takes `query`, `context` and `candidate`; `llm` and `source` read
exactly as here. Its own keys (`relation`, `order`, `pool`, `policy`, `dense`) are listed
in [Clustering one collection](../guide/clustering.md). Validation is just as strict, and
`xwalk match` refuses a clustering job with `wrong_job_kind`.

## Paths

Every path inside a job file resolves against **the job file's own directory**, never the
working directory. `load_job` sets `base_dir` to `path.parent.resolve()` and every builder
joins against it. A job directory is therefore movable as a unit, and

```bash
xwalk match --job examples/chebi/job.yaml --out runs/chebi
```

means the same thing from anywhere in the tree.

Absolute paths pass through untouched — `base_dir / "/data/targets.csv"` is
`/data/targets.csv` — which is how a shared target file lives outside the job directory
while the rest of the job stays relative. Paths given on the command line (`--out`,
`--index`, `--gold`) are *not* job-file paths and resolve against your shell's working
directory as usual.

## Credentials

There is exactly one way to supply a key:

```yaml
llm:
  api_key_env: OPENAI_API_KEY
```

An inline key is rejected while the file loads, before anything else is validated:

```
api_key must not appear in a job file; use api_key_env with the name of an environment
variable
```

The named variable is read at `build_llm()`, not at load, and an unset variable is:

```
environment variable OPENAI_API_KEY is not set; export it or change llm.api_key_env in
the job file
```

`api_key_env` may be omitted entirely — a local vLLM or Ollama-style endpoint that wants no
`Authorization` header simply gets none. The key never reaches the LLM fingerprint, so
rotating a key does not invalidate a resumable run and no secret can leak into
`manifest.json`.

## When errors surface

`load_job` does YAML parsing and pydantic validation, and nothing else. It touches no data
file, no index, no environment variable and no network.

**Fails at `load_job`:**

| Mistake | Message shape |
|---|---|
| a missing required section (`name`, `templates`, `target`, `source`, `retrievers`, `prompts`) | `required field is missing` |
| both `llm:` and `decider:`, or neither | `a job needs exactly one of \`llm:\` or \`decider:\`` (neither: reported at `llm` as a missing field) |
| neither `templates.query` nor a non-empty `templates.queries` | `templates need a query or a non-empty queries list` |
| an unknown `kind` on a source | `Input should be 'csv', 'tsv', 'jsonl', 'obo', 'owl' or 'sql'` |
| an unknown retriever `kind` | `Input should be 'bm25' or 'dense'` |
| `retrievers: []` | `List should have at least 1 item after validation, not 0` |
| a `dense` retriever with no `model` | `a dense retriever needs a model name` |
| a retriever `name` with a path separator or other unsafe character (`../x`) | `retriever name '../x' must start with a letter or digit and use only letters, digits, …` |
| `llm.api_key`, `decider.api_key` or `decider.rewrite.llm.api_key` present | `api_key must not appear in a job file; …` |
| an unknown `decider.kind` | `Input should be 'jev'` |
| a decider `policy` threshold outside `[0, 1]`, `shortlist_floor > accept_at`, `chunk_size: 0` | the field and the bound; an unknown decider `policy` key gets a did-you-mean |
| `selector:` on a decider job | `selector does not apply to a decider job` |
| `templates.queries` with more than one entry on an `llm:` job | `templates.queries with more than one entry applies to decider jobs; …` |
| `kind: openai_compat` with no `base_url` | `openai_compat needs a base_url` |
| a wrongly typed scalar (`accept_at: high`) | pydantic type error |

**Deferred to build time:**

| Mistake | Where it surfaces |
|---|---|
| a path that does not exist | `FileNotFoundError` from the first builder that reads it |
| a csv/tsv/jsonl/obo/owl source with no `path` | `a csv source needs a path` |
| a `sql` source without both `url` and `query` | `a sql source needs url and query` |
| an `id_column` absent from the file | `id_column 'id' not found in …; columns are …` |
| a Jinja syntax error in a template | `build_templates()` → `query template failed to compile: …` |
| a missing or invalid slots file (including a key given twice) | `build_prompts()`; `xwalk validate` reports `prompts_invalid` |
| an unset `api_key_env` variable | `build_llm()`, or `build_decider()` / `build_rewrite_llm()` on the decider path; `xwalk validate` reports `credential_missing` |
| an unknown `profile` | `build_llm()` → `unknown profile 'gpt'; choose from [...]` |
| a `properties` entry whose `name` is not `^[a-z][a-z0-9_]*$` | `build_prompts()` / `build_questions()` |
| a rubric with more than 10 rows, on the decider path | `rubric_question()` → `a decider rubric needs 2 to 10 rows, got 12` |
| a missing optional extra (`owl`, `sql`, `dense`, `litellm`) | the builder that needs it, naming the extra and the install command |

That split is deliberate: loading a job must be free and safe, so `xwalk eval` on a machine
with no credentials and no data mounted still works.

## Gotchas

**`base_dir` (and `decision_policy`) are not job-file keys.** `load_job` sets `base_dir`
to the job file's own resolved parent; a decider job's thresholds go under `policy:`.
Either key in the file is rejected.

**Omitting `exact_fields` is not plain BM25.** The job-file default is
`["label", "synonyms"]`, unlike the SDK's `BM25Retriever.build`, which defaults to no exact
fields. To get unboosted BM25 from a job file, write `exact_fields: []`.

**`llm.temperature`, `llm.max_tokens` and `llm.seed` reach every matching request.**
The matching stages (selection, scoring, verification, rewriting) leave these unset on
their requests, so the client sends the job's values. (Before 0.2 the stages hard-coded
512/256 output tokens and temperature `0.0`, so these fields changed only the
fingerprint.) The prompt drafting and optimising tools still pass their own
`max_tokens=2048`. A per-stage value can be set only from Python, by passing
`temperature=`/`max_tokens=` to a stage constructor.

**Two retrievers of the same kind need distinct names.** The index subdirectory is
`spec.name or spec.kind`, so two unnamed `kind: dense` entries would share `<index>/dense`;
validation refuses that (`invalid_value`) and asks you to name them.

**A persisted index is reused only when it matches.** `xwalk match` and `xwalk index`
compute each index's identity components from the job and the target records (without
encoding anything), then open a matching index, build an absent one, and refuse an
incompatible one (exit 3, naming the differing components) unless `--rebuild-index` is
given. Building replaces the directory's contents rather than adding to them. On a large
collection, build once with `xwalk index` and pass the same `--index` to every `match`.
`xwalk ablate` and `xwalk prompts optimize` still build their own indexes under `--out`.

**Anything that changes what a result *means* changes the run fingerprint**: templates,
slots, target snapshot, retriever set and depths, RRF constant, retriever timeout, LLM
identity and generation parameters, and the classification thresholds. On the decider path
`decision_run_fingerprint` digests the same shape with the decider's fingerprint and the
question set in place of the LLM's fingerprint and prompts, and with the decider thresholds
(`screen_floor` through `chunk_size`) as the policy. It excludes `decider.timeout`,
`decider.max_retries` and `policy.retriever_timeout` as well as `concurrency` — all four
change how fast an answer arrives, never what it means. Credentials, output paths and `concurrency` are excluded on purpose. A changed fingerprint means a resumed run
re-matches everything rather than mixing incomparable results.

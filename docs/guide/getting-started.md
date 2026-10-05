# Getting started

A first matching run, from two CSV files to a mapping table, in about fifteen minutes.
By the end you will have a run directory you can score, resume, and hand to a reviewer.

## Install

```bash
pip install xwalk
```

That is BM25 retrieval plus any OpenAI-compatible endpoint, and it does not install
PyTorch. Add extras only when you need them:

```bash
pip install 'xwalk[dense]'      # sentence-transformers, torch, faiss-cpu
pip install 'xwalk[ontology]'   # rdflib, for OWL sources
pip install 'xwalk[sql]'        # SQLAlchemy, for database sources
pip install 'xwalk[litellm]'    # the LiteLLM adapter
pip install 'xwalk[all]'
```

If an extra is missing you get an error naming the extra and the install command, not a
bare `ImportError` from a library you have never heard of.

## The short route

If a job file is all you need, the operations layer does the wiring below for you:

```bash
xwalk init demo                     # a bundled job: five targets, four mentions
xwalk validate --job demo/job.yaml  # strict and offline: no model call
xwalk match --job demo/job.yaml --out demo/run --max-calls 50
```

The same from Python, offline, with a scripted stand-in for the model
(`examples/quickstart.py` runs it):

```python
import json

from xwalk import ops
from xwalk.llm.fake import FakeLLM


def offline_model(request):  # one answer every stage understands
    return json.dumps({"chosen_key": "C01", "confidence_score": 0.9, "decision": "support"})


job = ops.bundled_example() / "job.yaml"      # or your own job.yaml
assert ops.validate(job, check_credentials=False).ok
result = ops.run(job, "runs/demo", llm=FakeLLM(handler=offline_model), max_calls=100)
print(result.run, result.counts, result.usage["tokens"])
print(ops.explain("runs/demo", "s2").data["decision"])
ops.export("runs/demo", "reviewed", "runs/demo/reviewed.csv")
```

Drop `llm=` to use the model the job names. The rest of this page builds the same thing by
hand, which is what you need once a job file cannot express what you want.

## The two files you need

**Targets** — the collection you are matching *into*. One row per record, one column
holding the id:

```csv
id,label,synonyms,definition
CHEBI:17234,glucose,dextrose|grape sugar,A monosaccharide sugar.
CHEBI:17992,sucrose,table sugar|saccharose,A disaccharide of glucose and fructose.
CHEBI:15903,beta-D-glucose,,The beta-anomer of D-glucose.
```

**Sources** — the records you are matching *from*:

```csv
mention_id,mention,context_left,context_right
s1,glucose,blood ,levels were elevated
s2,dextrose,intravenous ,was administered
s3,table sugar,a spoonful of ,in the coffee
```

Neither layout is prescribed. The column names are yours; the templates in the next step
are what connect them to xwalk.

## Step 1: templates

Four Jinja templates carry your entire domain mapping. This is the only place xwalk
learns what your columns mean.

```python
from xwalk import TemplateSet

templates = TemplateSet(
    # source record → the string handed to retrievers
    query="{{ mention }}",
    # source record → the context block the model sees
    context="{{ context_left }} [{{ mention }}] {{ context_right }}",
    # target record → the text that gets indexed
    doc="{{ label }} {{ synonyms | join(' ') }}",
    # target record → one entry in the candidate list shown to the model
    candidate="ID: {{ id }} Label: {{ label }}",
)
```

Two behaviours worth knowing now. Templates compile immediately, so a syntax error
raises `TemplateError` here rather than on record 40,000. And a field a record does not
have renders as an empty string rather than raising — sparse collections are normal, and
aborting a long run over one missing optional column is the wrong trade.

The `doc` template deserves real attention. It decides what is findable at all: a target
record whose `doc` renders empty can never be retrieved, and no amount of prompt work
will recover it.

## Step 2: load the targets and build an index

```python
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source
from xwalk.stores.memory import MemoryStore

targets = list(csv_source("targets.csv", id_column="id", multivalue_columns=["synonyms"]))
store = MemoryStore.from_source(targets)

retriever = BM25Retriever.build(
    targets, templates, "index/", exact_fields=("label", "synonyms")
)
```

`multivalue_columns` splits `dextrose|grape sugar` into a list, which is what lets the
`doc` template `join` it.

`exact_fields` is worth setting. It adds a heavily boosted whole-string match, so a query
that exactly equals a label or synonym outranks a document that merely repeats the term
more often. Omit it for plain BM25.

Building an index writes to disk. Reopen it later with `BM25Retriever.open("index/")`
instead of rebuilding.

## Step 3: describe your domain in slots

The prompts are assembled from skeletons that ship with xwalk plus a `slots.yaml` you
write. You never write prompt text; you write domain content.

```yaml
entity_noun: chemical entity mention
target_noun: ChEBI ontology term
domain_brief: biomedical chemistry nomenclature

rubric:
  - score: 1.0
    name: Certain
    when: exact case-insensitive match to the candidate label or one of its synonyms
    example: garlic -> Garlic
  - score: 0.6
    name: Plausible
    when: the source names a specific instance of the candidate class
    example: Fuji apple -> apple
  - score: 0.4
    name: Speculative
    when: related only by broad category
    example: sugar -> carbohydrate

hard_rules:
  - A salt, ester, or hydrate is a distinct entity from its parent compound
  - >-
    If no candidate names the same chemical species, abstain. A plausible-looking
    neighbour is worse than no match, because a wrong mapping is silently wrong.

disambiguation_steps: >-
  Read the surrounding context before choosing. "Sugar" in a clinical blood-glucose
  sentence is glucose; in a cooking sentence it is sucrose.
```

Rubric scores must be strictly decreasing, and you need at least two rows. See the
[prompts reference](../reference/prompts.md) for every slot and every rule.

Do not want to write this by hand? `xwalk prompts draft` will propose a first version
from a description and a sample of your records — see [prompt
optimisation](prompt-optimisation.md).

## Step 4: wire up a matcher

```python
import os

from xwalk import Matcher, MatchPolicy
from xwalk.batch import build_run_fingerprint
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

llm = OpenAICompatClient(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    profile="openai",
)

prompts = PromptSet.from_slots(load_slots("slots.yaml"))
policy, selector_policy = MatchPolicy(), SelectorPolicy()

matcher = Matcher(
    templates=templates,
    retrievers=[retriever],
    store=store,
    selector=Selector(llm, prompts, templates, policy=selector_policy),
    scorer=Scorer(llm, prompts, templates, review_floor=policy.review_floor),
    verifier=Verifier(llm, prompts, templates),
    rewriter=QueryRewriter(llm, prompts, templates),
    policy=policy,
    run_fingerprint=build_run_fingerprint(
        templates=templates,
        prompts=prompts,
        store=store,
        retrievers=[retriever],
        llm=llm,
        policy=policy,
        selector_policy=selector_policy,
    ),
)
```

`profile` tells the client what this provider family supports — `"openai"`,
`"anthropic-compat"`, `"google-compat"`, `"vllm"`, or the all-conservative default
`"unknown"`. It only affects what the adapter *asks for*; output is validated either
way.

Pass `build_run_fingerprint` the same objects you pass `Matcher`. That fingerprint is
what makes resume safe, and it is not cross-checked — see [concepts](../concepts.md#fingerprints-and-resume).

## Step 5: run the batch

```python
from xwalk.batch import run_batch_sync
from xwalk.sources.tabular import csv_source

report = run_batch_sync(
    matcher,
    csv_source("sources.csv", id_column="mention_id"),
    out="run/",
)

print(report.by_status())        # {<MatchStatus.MATCHED: 'matched'>: 17, ...}
print(report.duplicate_targets())  # targets chosen by more than one source record
print(len(report.needs_review()))
```

In an async application, `await run_batch(...)` instead — `run_batch_sync` calls
`asyncio.run` and cannot be used inside a running event loop.

You now have:

| File | What it is |
|---|---|
| `run/mapping.csv` | the deliverable: one row per source record |
| `run/results.jsonl` | the full trace of every attempt, one JSON object per record |
| `run/manifest.json` | the run fingerprint, component fingerprints, and counts |
| `run/ledger.sqlite` | the source of truth; everything above is regenerable from it |

Re-running with `resume=True` (the default) skips records already completed under this
fingerprint. Change a template or a threshold and everything re-runs, because the
fingerprint changed. Fix one source row and only that row re-runs.

## Step 6: look at what needs attention

Two things are worth checking on every run before you look at accuracy.

```python
duplicates = report.duplicate_targets()
if duplicates:
    print(f"{len(duplicates)} targets were chosen by more than one source record")
```

xwalk reports many-to-one conflicts and does not resolve them — both records stay
matched. Whether that is a bug or the correct answer depends on your data, and the
library cannot know which.

```python
for result in report.needs_review():
    print(result.source_id, result.matched_id, result.confidence, result.reason.value)
```

The `reason` tells you *why* it landed in review: a score below `accept_at`, a verifier
that disagreed, or output that would not resolve. Those want different responses.

## Where to go next

- **Measure it.** [Evaluation](evaluation.md) — with a gold file you get precision,
  coverage, cost, and a decomposition of every miss into something actionable.
- **Adjudicate it.** [Review](review.md) — export the review bucket, fill in decisions,
  apply them as an overlay.
- **Improve the prompts.** [Prompt optimisation](prompt-optimisation.md).
- **Skip the Python.** The [CLI](../reference/cli.md) runs all of this from a job file.
- **Something looks wrong.** [Troubleshooting](troubleshooting.md).

## A complete working example

`examples/chemistry` is this guide as runnable code, with 52 target terms chosen to be
full of near-misses — anomers, conjugate acid/base pairs, four sugar alcohols — so the
failure decomposition means something:

```bash
export XWALK_TEST_API_KEY=sk-...
export XWALK_TEST_BASE_URL=https://api.openai.com/v1
export XWALK_TEST_MODEL=gpt-4o-mini

python -m examples.chemistry.run --out run/
python -m examples.chemistry.evaluate_run --run run/ --fingerprint <printed above>
```

Four more examples cover OWL sources, numeric target ids, and user-supplied gold alias
expansion. See the table at the end of the [README](../../README.md).

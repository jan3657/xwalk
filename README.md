# xwalk

Match records from any collection to any other, using retrieval plus an LLM.

Retrieval alone gives you a shortlist and no decision. An LLM alone cannot see a
collection that does not fit in a context window. xwalk is the loop between them: a
resumable, auditable pipeline that ends with every source record in one of four states —
matched, needs review, unmatched, or failed — and an evaluation report that tells you
which part to fix next.

**[Full documentation →](docs/README.md)** · [Getting
started](docs/guide/getting-started.md) · [Concepts](docs/concepts.md) · [What each
component does](docs/components.md) · [API reference](docs/README.md#reference) ·
[Troubleshooting](docs/guide/troubleshooting.md)

## Install

```bash
pip install xwalk                 # BM25 + any OpenAI-compatible endpoint. No torch.
pip install 'xwalk[dense]'        # + sentence-transformers, torch, faiss-cpu
pip install 'xwalk[ontology]'     # + rdflib, for OWL sources
pip install 'xwalk[sql]'          # + SQLAlchemy, for database sources
pip install 'xwalk[all]'
```

Matching two CSVs with an API-hosted model installs none of the heavy stack. A missing
extra raises an error naming the extra and the install command, never a bare
`ImportError` from a library you have never heard of.

See [docs/platforms.md](docs/platforms.md) for the supported platform matrix. xwalk
never substitutes a different BM25 engine silently, because that would change ranking
and quietly undermine reproducibility.

```python
import os

from xwalk import Matcher, MatchPolicy, TemplateSet
from xwalk.batch import build_run_fingerprint, run_batch_sync
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.stores.memory import MemoryStore

templates = TemplateSet(
    query="{{ mention }}",
    context="{{ context_left }} [{{ mention }}] {{ context_right }}",
    doc="{{ label }} {{ synonyms | join(' ') }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)

targets = list(csv_source("targets.csv", id_column="id", multivalue_columns=["synonyms"]))
store = MemoryStore.from_source(targets)
# exact_fields makes a whole-string hit on a label or synonym outrank a document that
# merely contains the query term more often. Omit it for plain BM25.
retriever = BM25Retriever.build(
    targets, templates, "index/", exact_fields=("label", "synonyms")
)

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

report = run_batch_sync(matcher, csv_source("sources.csv", id_column="mention_id"), out="run/")
print(report.by_status())
print(report.duplicate_targets())
```

`run/mapping.csv` is the deliverable. `run/ledger.sqlite` makes the run resumable —
re-running with `resume=True` skips completed records and re-runs anything whose source
record or run configuration changed.

## Review

```python
from xwalk.ledger import Ledger
from xwalk.review import apply_review, export_review, read_review

ledger = Ledger.open("run/ledger.sqlite")
export_review(ledger, report.run_fingerprint, "review.csv")
# fill in `decision` (accept / reject / replace / no_match / defer) and `reviewer`
apply_review(ledger, read_review("review.csv"), target_store_fingerprint=store.fingerprint)
```

Review never overwrites model output; `export_mapping_csv(..., use_review=True)` writes
the adjudicated view alongside it.

## Evaluation

```python
from xwalk.evaluate import evaluate, load_gold_csv, render_report
from xwalk.ledger import Ledger

gold = load_gold_csv("gold.csv")            # source_id,gold_ids  (empty = no match)
ledger = Ledger.open("run/ledger.sqlite")
report = evaluate(ledger, run_fingerprint, gold)
print(render_report(report))
```

Nothing here calls an LLM or a retriever — it reads the ledger, so scoring a run is
free, repeatable, and works on a machine with no credentials.

An **empty** `gold_ids` cell means "the correct answer is no match", a first-class
label. A source id **absent** from the file is unlabelled and excluded from every
metric. Conflating those two silently inflates no-match recall.

The report leads with where to spend effort, decomposed three ways:

- **never retrieved** — fix the `doc` template, the retriever set, or retrieval depth
- **truncated** — the gold record was retrieved but the selector budget cut it before
  the model saw it; raise `SelectorPolicy.max_candidates`
- **misjudged** — the gold record was presented and the model chose otherwise; fix prompts

The middle bucket is why there are three and not two. Collapsing a budget miss into
"retrieval failure" sends you to fix the wrong thing.

Alongside it: `accepted_precision`, `automatic_coverage`, `review_rate`,
`recall_at_any_status` (the ceiling perfect thresholds could reach), no-match precision
and recall, cost and latency per record, a threshold curve re-derived from recorded
confidences, and a calibration warning when those confidences do not separate correct
from incorrect — in which case tuning `accept_at` is tuning nothing.

Alias expansion and ID normalization are yours, not the library's:

```python
gold = load_gold_csv(
    "gold.csv",
    normalize=lambda i: f"NCBIGene:{i}" if i.isdigit() else i,
    expand=lambda ids: frozenset().union(*(alias_map.get(i, {i}) for i in ids)),
)
```

## Prompt optimisation

```python
from xwalk.evaluate import PromptRole
from xwalk.prompts.optimize import OptimizeConfig, optimize_prompt

report = await optimize_prompt(
    matcher_factory=build_matcher,          # (PromptSet) -> Matcher
    source_records=records, gold=gold, initial=slots,
    optimiser_llm=strong_model,
    config=OptimizeConfig(role=PromptRole.SELECTOR, rounds=4, max_calls=5_000),
    work_dir="opt/",
)
print(report.stopped_because, report.test_report.accepted_precision)
```

Three partitions with distinct roles: **prompt-train** supplies the failures shown to
the optimising model, **validation** chooses the retained round and the stopping point,
and **test** is evaluated exactly once at the end. Test is never reported per round —
that would make it a second validation set, which is the same mistake one level up.

Which failures the model is shown depends on the role. A selector learns nothing from a
case where the gold record was never retrieved, because it never saw the right answer;
a scorer learns a great deal from exactly that case, because abstaining there was its
job. `estimate_calls` prints the cost before anything is spent, and `max_calls` is a
hard stop.

The model is asked for **slots only**, never prompt text, and every candidate is run
through `validate_contract` before it is written. A bad round produces a poor rubric; it
cannot produce a prompt whose output will not parse.

## Command line

The SDK stays primary; the CLI is a shell over it. A job file is serialized constructor
arguments — every field maps to something the quickstart above passes by hand.

```bash
xwalk index   --job job.yaml --out runs/index
xwalk match   --job job.yaml --out runs/first
xwalk eval    --run runs/first --gold gold.csv
xwalk compare --gold gold.csv --run 'mini=runs/first' --run 'big=runs/second'
xwalk ablate  --job job.yaml --gold gold.csv --out runs/ablation
xwalk review  export --run runs/first --out review.csv
xwalk review  apply  --run runs/first --reviewed review.csv --job job.yaml
```

Exit codes: `0` success, `1` completed but the review bucket is non-empty, `2` usage
error, `3` runtime failure. Credentials are never written in a job file — `api_key_env`
names an environment variable, and an inline `api_key` is rejected at load time.

## Worked examples

| Example | What it exercises |
|---|---|
| [`examples/chemistry`](examples/chemistry) | the smallest end-to-end run, with scripts you can execute |
| [`examples/chebi`](examples/chebi) | the OWL loader; one-field source with long-document context |
| [`examples/ncbi_disease`](examples/ncbi_disease) | user-supplied gold alias expansion, living in your code |
| [`examples/nlm_gene`](examples/nlm_gene) | numeric target IDs — the case opaque candidate keys exist for |
| [`examples/cafeteria_fcd`](examples/cafeteria_fcd) | a second OWL domain; only `slots.yaml` differs from `chebi` |

## Documentation

| | |
|---|---|
| **Guides** | [Getting started](docs/guide/getting-started.md) · [Concepts](docs/concepts.md) · [Components](docs/components.md) · [Evaluation](docs/guide/evaluation.md) · [Review](docs/guide/review.md) · [Prompt optimisation](docs/guide/prompt-optimisation.md) · [Extending](docs/guide/extending.md) · [Troubleshooting](docs/guide/troubleshooting.md) |
| **Reference** | [Core types](docs/reference/core-types.md) · [Pipeline](docs/reference/pipeline.md) · [Retrieval](docs/reference/retrieval.md) · [Sources](docs/reference/sources.md) · [LLM clients](docs/reference/llm.md) · [Prompts](docs/reference/prompts.md) · [Evaluation](docs/reference/evaluation.md) · [Job file](docs/reference/job-file.md) · [CLI](docs/reference/cli.md) |
| **Operations** | [Platforms](docs/platforms.md) · [Contributing](CONTRIBUTING.md) · [Releasing](docs/releasing.md) · [Changelog](CHANGELOG.md) |

## Licence

MIT. See [LICENSE](LICENSE).

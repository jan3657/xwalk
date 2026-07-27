# xwalk

Match records from any collection to any other, using retrieval plus an LLM.

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

# xwalk documentation

Matching records from any collection to any other, using retrieval plus an LLM.

New here? Read [getting started](guide/getting-started.md), then
[concepts](concepts.md). Looking for one specific signature? Go straight to the
reference.

## Guides

Task-oriented, in the order you will need them.

| Guide | Read it when |
|---|---|
| [Getting started](guide/getting-started.md) | You have two CSV files and want a mapping table. |
| [Concepts](concepts.md) | You want to know why the loop is shaped this way before you trust it. |
| [What each component does](components.md) | You need to find the right module without reading forty of them. |
| [Measuring a run](guide/evaluation.md) | You have gold labels and need to know what to change next. |
| [Human review](guide/review.md) | You have a review bucket and need decisions back into the run. |
| [Clustering one collection](guide/clustering.md) | You want one collection's equivalent records grouped (experimental). |
| [Drafting and optimising prompts](guide/prompt-optimisation.md) | Your failures are *misjudged* rather than *never retrieved*. |
| [Extending xwalk](guide/extending.md) | You need your own retriever, source, store, or model client. |
| [Troubleshooting](guide/troubleshooting.md) | Something is behaving oddly. |

## Reference

Complete API documentation. Every signature and default here is taken from the source.

| Page | Covers |
|---|---|
| [Core types](reference/core-types.md) | `Record`, `MatchResult`, `Attempt`, `MatchPolicy`, `TemplateSet`, fingerprints, the `Ledger`. |
| [The matching pipeline](reference/pipeline.md) | `Matcher`, candidate keying, the four stages, `run_batch`, run fingerprints. |
| [Retrieval](reference/retrieval.md) | The `Retriever` protocol, BM25, dense, fusion, target stores. |
| [Loading records](reference/sources.md) | CSV, JSONL, OBO, OWL and SQL loaders; writing your own; optional extras. |
| [LLM clients](reference/llm.md) | The `LLMClient` protocol, the two adapters, `FakeLLM`, caching, output parsing. |
| [Prompts](reference/prompts.md) | The slots contract, `PromptSet`, the four skeletons, drafting and optimisation. |
| [Evaluation and review](reference/evaluation.md) | Gold labels, every metric, the failure decomposition, ablation, the review overlay. |
| [The job file](reference/job-file.md) | Every key in `job.yaml`, with types and defaults. |
| [Command line](reference/cli.md) | Every subcommand, every flag, exit codes. |

## Operations

| Page | Covers |
|---|---|
| [Platforms](platforms.md) | The supported platform matrix and why there is no BM25 fallback. |
| [Releasing](releasing.md) | Cutting a release. Maintainers only. |
| [Benchmarks](benchmarks.md) | The pilot benchmark: data, the three tracks, how to reproduce, what is synthetic and what is pending, measured performance. |
| [Contributing](../CONTRIBUTING.md) | The four gates, test markers, and the invariants the design will not give up. |

## Worked examples

Runnable, in `examples/`. Each is a complete job you can copy and point at your own data.

| Example | What it exercises |
|---|---|
| [`chemistry`](../examples/chemistry) | The smallest end-to-end run, with scripts you can execute. |
| [`chebi`](../examples/chebi) | The OWL loader; a one-field source with long-document context. |
| [`ncbi_disease`](../examples/ncbi_disease) | User-supplied gold alias expansion, living in your code. |
| [`nlm_gene`](../examples/nlm_gene) | Numeric target ids — the case opaque candidate keys exist for. |
| [`cafeteria_fcd`](../examples/cafeteria_fcd) | A second OWL domain; only `slots.yaml` differs from `chebi`. |

## The five things to know

If you read nothing else:

1. **A malformed model answer never resolves to a real record.** Candidates carry opaque
   per-attempt keys and resolution is exact-only, so a hallucinated id fails to resolve
   rather than silently naming the wrong target.
2. **An empty `gold_ids` cell means "no match is correct"; an absent row means
   unlabelled.** Conflating them inflates no-match recall by exactly the cases you got
   right.
3. **The ledger is the source of truth.** `mapping.csv`, `results.jsonl` and
   `manifest.json` are exports regenerable from it.
4. **Evaluation calls nothing.** Scoring a run is free, repeatable, and works with no
   credentials.
5. **Everything fails toward review.** Unparseable output, an unrecognised verdict, an
   invented key — none of them can produce an automatic match.

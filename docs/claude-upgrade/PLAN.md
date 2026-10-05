# xwalk 0.2 implementation plan

Prepared for Jan, 5 October 2026.

## Main decision

Use the credit to target a release that a new user can install, understand, and use to produce an inspectable crosswalk. Repair correctness and persistence first. Improve the short Python route and the existing CLI. Add one flat equivalence-clustering workflow. Finish with reproducible evaluation, documentation, and an optional small MCP adapter.

The proposed product position is: a Python toolkit for reliable crosswalk generation and canonicalization, with explicit matching policies, resumable execution, and evidence for every accepted or unresolved decision.

This is a proposed direction, not a demonstrated superiority claim.

## Audit baseline and evidence

The audited public main commit is f737099f23ce0c6720963759e13e95d21d8bdf89, package version 0.1.1. It contains a substantial SDK and CLI, retrieval and LLM interfaces, SQLite persistence, review overlays, evaluation, prompt optimization, optional dependencies, and CI.

The latest recorded main CI run succeeded. The inspected v0.1.1 release workflow built and verified artifacts successfully but failed in the PyPI publish job. A GitHub release exists. Current PyPI availability was not verified, so publication status must be checked before drawing conclusions.

The audit ran focused probes against AST-loaded repository functions with deterministic replacements for external dependencies and providers. It did not run the complete pytest suite or any paid provider benchmark. The included diagnostic script and observed JSON record that limited verification scope.

### Reproduced issues

| Issue | Observed behavior | Main files |
|---|---|---|
| Trace settings alter decisions | Identical scripted inputs match with candidate traces on and become unmatched with them off | matcher.py |
| Current exports include stale source versions | Edit source s1 and resume, and mapping.csv contains two rows for s1 | ledger.py, batch.py |
| Incomplete usage totals | A two-attempt fixture requires five calls but reports four because rewrite usage is omitted | matcher.py |
| Invalid confidence is accepted | NaN is clamped to 1.0 and can produce an accepted match | stages/gate.py, llm/parsing.py |
| Provider error boundaries differ | Verifier retry exhaustion and fatal rewriter errors escape match | matcher.py |
| Concurrent cleanup is unsafe | A sibling attempts to write after another failure caused ledger closure | batch.py |
| Multiline candidate evidence is misplaced | Detail from the chosen candidate appears under other candidates | stages/gate.py, stages/keying.py |

Source inspection also found CLI success handling that does not check failed records, silently ignored unknown job fields, generation settings that are overridden by stage requests, repeated index construction, and incomplete encoder identity. These need normal package-level regression tests during task 00 or 01.

## Competitors and positioning

| Tool | Documented emphasis | Design lesson |
|---|---|---|
| LinkTransformer | DataFrame linkage, embedding retrieval plus LLM judging, deduplication and clustering | It is the closest direct comparator. Match its ease of first use and measure the value of xwalk's recovery and evidence |
| Splink | Probabilistic linkage, blocking, clustering, and SQL backends | Reuse or accept external candidate generation for large structured problems rather than rebuilding distributed linkage |
| dedupe | Active learning, blocking, deduplication and constrained joins | Explicit matching cardinality and persistent corrections are practical needs |
| Fuzzylink | Embedding-informed statistical linkage with LLM-assisted labelling | A future learned cheap decision stage may reduce repeated inference |
| LOTUS | General semantic operations over data | Keep xwalk focused on entity decisions and their lifecycle |

These descriptions do not establish a performance ranking. Benchmark equivalent tasks, inputs, candidate budgets, and model configurations.

## First release scope

### Required foundation

- Correct matching decisions independent of trace verbosity.
- A defined source snapshot/current-export contract, with history preserved.
- Consistent failure states, bounded retries, and safe cancellation.
- Complete provider usage accounting, including rewrites and explicit unknown usage.
- Strict job validation and documented generation parameter precedence.
- Validated index reuse with semantic fingerprints.
- Short Python execution path and noninteractive CLI.
- Reproducible offline example and matching benchmark runner.
- README, installed-wheel checks, and a release candidate.

### Main feature

Flat equivalence clustering canonicalizes records that refer to the same entity or concept. It must keep different concepts separate even when they are semantically related.

The old clustering design distinguishes equivalence from subsumption. Retain that distinction. Hierarchical roll-up, multiple parents, and taxonomy mutation must not be smuggled into the first release.

### Optional addition

A stdio MCP adapter exposes the same operations as the CLI. It remains optional both in installation and in the credit plan.

## Architecture and code reduction

Keep the existing Record/Candidate types, retriever and LLM protocols, opaque candidate keys, staged reasoning, review overlays, and evaluation decomposition.

Add a small operations layer that owns validation, preparation, execution, inspection, and export. Reuse it from Python, CLI, and MCP. Keep clustering orchestration separate from the fixed-target Matcher.

Refactor only couplings that cause observable problems:

- Runtime candidate state versus serialized traces.
- Stage execution versus error and usage accounting.
- Historical results versus the current mapping.
- Building an index versus opening a validated existing index.
- Core operations versus interface formatting.

Code reduction means removing duplication, dead state, and unnecessary layers. Do not minify Python, replace clear names, remove useful types, or target a percentage reduction in lines. Preserve argparse unless a concrete requirement justifies replacing it.

## Clustering contract

The existing July design is useful but its implementation plan conflicts with it. Reconcile at least these points:

1. Assignment to an immutable taxonomy node is different from merging or modifying that node.
2. Source collection and processing order must be part of clustering run identity.
3. Historical pool representations must be reconstructable, not merely named by snapshot IDs.
4. Exports must use the selected best stable revision when refinement oscillates.
5. An injected incumbent must survive candidate truncation.
6. Record retrieved candidates separately from candidates actually shown.
7. Verifiers must receive actual bounded member evidence, not only exemplar counts.
8. Full pool rebuilding at every mint must not be assumed scalable.

The first vertical slice should have deterministic order, stable run-local IDs, explicit assigned/review/deferred/failed states, candidate and call limits, traceable creation of new clusters, bounded refinement, resumable transactions, and exports.

Every source must be accounted for. Accepted members form a partition. Unresolved records must remain explicitly unresolved or provisional singleton records, not be mislabeled as verified equivalences.

Do not merge A, B, and C solely because A-B and B-C were accepted. Test contradictory chains and use the chosen explicit cluster-merge policy. No verifier score establishes certainty.

## CLI and Python interface

The project already has index, match, eval, compare, ablate, prompts, and review commands. Extend that CLI.

Proposed capabilities:

- init: write a complete small example or job skeleton.
- validate or doctor: check fields, IDs, dependencies, credential presence, paths, and resume compatibility without paid calls.
- match dry-run: show planned work and clearly labelled estimates.
- inspect: show run state, counts, configuration identity, and output artifacts.
- explain: inspect one source decision and its attempts.
- export: select raw or reviewed mappings and output format.
- cluster: execute the first supported clustering workflow.
- JSON output: stable schema version, operation, status, counts, IDs, and structured errors.

These names are proposals. Reconcile them with the existing API during task 00.

All machine-readable output goes to stdout. Progress and logs go to stderr. Preserve or explicitly migrate existing exit-code semantics. A run with unresolved runtime failures must not silently return success.

One high-level Python route should require about 10-15 lines for a common job while retaining the existing low-level constructor interfaces for researchers.

## MCP contract

Use the official supported Python SDK at implementation time, with tested version bounds. Begin with stdio and an optional xwalk[mcp] extra.

A useful first tool set is validate_job, search_candidates, match_records for bounded inputs, get_run, list_results, and explain_result. Reuse core exports. Long-running jobs can remain in the CLI initially unless a durable job/status contract already exists.

Avoid a new distributed scheduler, web hosting, accounts, or authentication product just to add MCP. Do not imply that an in-memory task survives process exit.

Test protocol initialization, tool listing and calls, input/output schemas, pagination, error mapping, and log separation. Do not accept arbitrary executable code or unbounded data dumps as tool parameters.

## Additional feature discovery

Allow each implementation session to propose at most three small additions. Each proposal must identify the affected user, a concrete workflow problem, existing code it reuses, a success check, maintenance cost, and whether it fits the remaining envelope.

Highest-value candidates:

1. Offline preflight or doctor checks.
2. True per-run call limits enforced across concurrency, retries, and rewrites.
3. Clear resume explanations showing which inputs or settings changed.
4. Export of adjudicated mappings through the CLI.
5. Field-level normalization or blocking rules that are explicit in job configuration.

Later candidates include candidate-pair import, one-to-one assignment, calibrated acceptance policies from labelled validation data, review-derived training data, and multilingual fixtures. These need a demonstrated use case. The current many-to-one behavior must remain documented. Independent top-1 choices do not enforce one-to-one assignment.

## Evaluation

Run separate tracks for pairwise decisions, full source-to-catalog matching, and clustering. Do not treat a preblocked pair benchmark as proof of end-to-end retrieval.

Required first benchmark:

- One small pinned public corpus for quick comparisons.
- A frozen food or ontology sample that reflects Jan's actual use, with no-match cases and near-neighbor concepts.

A fixed WDC Products variant is the next expansion for confusable and unseen entities if it is not already the chosen public corpus. Billiger.de Products is an optional bilingual extension after the first release. Synthetic fixtures test implementation behavior. They do not measure real semantic quality.

Begin with exact/fuzzy matching, BM25, embedding nearest neighbor, retrieval plus one LLM pass, full xwalk, and LinkTransformer on a suitable common task. Add Splink or dedupe where multiple structured fields make the comparison meaningful.

For matching, report candidate recall, truncation loss, conditional selector accuracy, final precision/recall, accepted precision versus coverage, no-match performance, review rate, calls/tokens, wall time, and peak memory. Count both recovery gains and new errors.

For clustering, report pairwise and B-cubed precision/recall/F1, false merges, splits, singleton handling, review coverage, order sensitivity, calls, and runtime. Use independent development and test entities. Document target-catalog construction and which distractors each system saw.

First implement the runner and data manifests. Real provider evaluation uses a separately identified endpoint and budget. Keep model-reported confidence distinct from empirical calibration.

## Performance

Profile before changing algorithms. Start with repeated index rebuilding, duplicate target loading, dense embedding memory copies, chunk-wide batch barriers, and concurrent identical cache misses.

A useful optimization must show measured time, memory, throughput, or inference savings on a declared workload, while preserving the intended results. Include encoder-call-count tests for index reuse and concurrency tests for request coalescing. Avoid unmeasured speed claims.

## Documentation and release

The README should lead with an input/output example and a runnable included workflow. Move the long low-level constructor tour into the guide. Include a generic catalog example alongside the current food, chemical, and biomedical examples.

Explain status meanings, no-match versus failure, provider costs, review, reproducibility, supported platforms, limitations, and how to choose xwalk relative to simpler alternatives. Only list features that have shipped and passed their gates.

Execute the documented quickstart from an installed wheel. Retain optional-dependency and packaging checks. Ensure the release gate tests the exact commit and built artifact. Check PyPI availability and trusted-publisher configuration separately. Publication and merging are final user-controlled steps, not prerequisites for preparing a complete release candidate.

## Credit allocation and gates

| Work | Target envelope |
|---|---:|
| Baseline and contracts | $15 |
| Correctness and persistence | $45 |
| Shared operations and CLI | $30 |
| Flat equivalence clustering | $55 |
| Benchmark and measured performance | $25 |
| Optional MCP | $25 |
| Documentation and release | $20 |
| Independent integrated review | $20 |
| Reserve | $15 |
| Total | $250 |

The envelope is not a forecast. Measure actual promotional balance after each task. Use Sonnet for routine implementation and Opus for difficult correctness, design, and review where justified. Leave Fast mode off for unattended work.

Protect the last $55. Reallocate optional MCP first if core work needs more effort. Do not start another feature when consolidation should begin.

Claim deadline: 7 October 2026 at 23:59 Pacific, or 8 October at 08:59 in Slovenia. Expiry: 4 November at 23:59 Pacific, or 5 November at 08:59 in Slovenia.

## Definition of finished

A release candidate is ready when a fresh user can install it, run a documented workflow, inspect evidence and uncertainty, interrupt and resume safely, and export the intended current or reviewed result. Required behavioral regressions, interface tests, packaging checks, and documentation examples pass. Benchmark results identify their data, model, configuration, and verification limits. Optional unfinished features are clearly separated from supported behavior.

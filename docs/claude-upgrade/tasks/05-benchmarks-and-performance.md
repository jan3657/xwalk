# Task 05: Benchmark evidence and measured performance

Target envelope: $25. Dependencies: task 00 contracts. Fixtures/runners can be prepared in parallel with task 04. Final comparisons use integrated fixes.

## Outcome

Provide reproducible evidence about where xwalk helps, what it costs, and which optimization actually improves a declared workload.

## Evaluation work

1. Create data manifests with source URLs, versions, construction steps, identifiers, and train/validation/test boundaries.
2. Keep three tracks distinct: pairwise classification, full source-to-catalog matching, and clustering. A preblocked pair table does not measure retrieval recall over a catalog.
3. Require one small pinned public corpus and a frozen food/ontology sample relevant to Jan. Expand to a WDC Products variant only after the runner works, unless WDC is already the chosen small public corpus. Begin with a small pilot if data preparation would consume the envelope.
4. Prioritize exact/fuzzy, the existing retrieval-only routes, retrieval plus one LLM pass, and full xwalk. Add LinkTransformer on a common supported task within the envelope. Splink/dedupe and extra datasets are optional expansions when structured fields and remaining time make them meaningful.
5. Match fields, candidate budgets, model settings, and evaluation data where the comparison requires them. Record intentional differences.
6. Separate synthetic FakeLLM integration checks from real semantic evaluation. If no actual endpoint or application budget is available, deliver runners and mark real results pending rather than fabricating numbers.
7. Measure candidate recall and truncation, accepted precision/coverage, final precision/recall/F1, no-match behavior, review rate, calls/tokens, wall time, and peak memory. Show both successful recoveries and errors introduced by recovery.
8. For clustering, report pairwise and B-cubed precision/recall/F1, false merges, splits, singleton behavior, review coverage, order sensitivity, calls, and runtime.

## Performance work

Profile one representative workload. Choose at most two fixes justified by measured evidence. Inspect repeated index construction, duplicate target loading, dense-array copies, concurrency barriers, and identical concurrent cache misses.

Potential acceptance checks include no extra embedding call when reopening an unchanged index, one upstream request for concurrent identical cache misses when coalescing is implemented, or reduced peak memory while preserving rankings.

## Acceptance

- A documented command reproduces offline benchmark smoke checks.
- Every published real result records data revision, code revision, model, configuration, and inference accounting.
- No test-set prompt tuning or hidden reuse of labelled test entities.
- Each claimed improvement includes a baseline and measurement conditions.
- Quality and cost are both visible. No unsupported claim of best accuracy, calibrated confidence, or guaranteed speedup.
- Raw benchmark outputs are available beside a short human-readable interpretation.

Do not turn this task into a distributed benchmarking platform. A reliable small comparison is more useful than an unfinished benchmark catalogue.

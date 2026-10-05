# Smoke results (SYNTHETIC)

Produced by `python -m benchmarks.run --smoke` at commit `cd88923`. Every LLM answer in
these files comes from the deterministic string-similarity judge in
`benchmarks/synthetic_llm.py`, not a language model; tokens are characters / 4.

- `results.json`: every raw value, with code commit, data digests and environment.
- `summary.md`: tables generated from `results.json`.
- `clustering/<dataset>/`: gold clusters and the baseline prediction files that
  `python -m benchmarks.cluster_eval` scored.

Interpretation, in short: the runners, accounting and metrics work end to end on all
four pilot variants. The parts that involve no model (exact, fuzzy, BM25, candidate
recall) are real measurements on 50-record samples: BM25 finds the gold target for
0.94-0.95 of food mentions and 0.74-0.84 of disease mentions, where bare abbreviations
get no hit at all. Anything involving the LLM route says only that the code paths run.
Real-model results are pending. See `docs/benchmarks.md` for the method and the full
reading.

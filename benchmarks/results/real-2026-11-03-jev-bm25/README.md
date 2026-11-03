# Real pilot run: Jev decider, BM25 only (3 November 2026)

The decider path (`decider: kind: jev`) on the same four 50-record variants as
`../real-2026-11-03-gpt-4o-mini/`. `benchmarks.run` drives only the LLM path, so these
runs used the CLI: `xwalk index`, `xwalk match --max-calls 1000`, then `xwalk eval
--gold ... --job ...`. Each directory has the job, `eval.json`/`eval.txt` and `mapping.csv`
(the ledger and index are not committed).

- Code: `c791349` (clean tree; same library code as `e3d455b`).
- Model: `~typesafe/jev-latest` through `https://openrouter.ai/api/alpha/decisions`; the
  responses name `jev-1.14`.
- Jobs: the materialised benchmark variants (same targets, sources and gold as the
  gpt-4o-mini run) with the shipped `examples/<dataset>/job_jev.yaml` policy and
  `slots_jev.yaml`, and **BM25 only (limit 200)**. The shipped Jev jobs also fuse a dense
  E5 retriever; it was left out because it needs `xwalk[dense]` and a weights download.
  Their thresholds were fitted with dense retrieval, so these numbers understate the
  shipped configuration's recall.
- Cost: **USD 0.0074** for all four variants plus a 5-record probe, measured as the change
  in the OpenRouter key's usage (0.5899 before, 0.5974 after). The per-row `cost_usd` that
  Jev reports sums to USD 0.0305; the two disagree and the key's figure is what was billed.

| variant | acc. precision | coverage | review | no-match P | no-match R | false accept on no-match | calls | reported cost USD |
|---|---|---|---|---|---|---|---|---|
| cafeteria_fcd | 0.975 | 0.800 | 0.140 | 0.000 | - | 0 | 140 | 0.0063 |
| cafeteria_fcd_nomatch | 0.933 | 0.600 | 0.160 | 0.833 | 0.909 | 1 | 121 | 0.0054 |
| ncbi_disease | 1.000 | 0.280 | 0.340 | 0.316 | 0.545 | 0 | 101 | 0.0102 |
| ncbi_disease_nomatch | 1.000 | 0.260 | 0.260 | 0.833 | 0.800 | 0 | 83 | 0.0086 |

Same variants, gpt-4o-mini `xwalk_full` (from the other directory), for comparison:

| variant | acc. precision | coverage | review | false accept on no-match | calls |
|---|---|---|---|---|---|
| cafeteria_fcd | 1.000 | 0.700 | 0.260 | 0 | 167 |
| cafeteria_fcd_nomatch | 1.000 | 0.540 | 0.360 | 0 | 179 |
| ncbi_disease | 0.767 | 0.860 | 0.120 | 7 | 205 |
| ncbi_disease_nomatch | 0.633 | 0.600 | 0.160 | 7 | 176 |

n = 50 per variant and one run each: a difference of two or three rows is noise.

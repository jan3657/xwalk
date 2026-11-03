# Real pilot run: openai/gpt-4o-mini via OpenRouter (3 November 2026)

First real-model run of the pilot benchmark. Every model number here comes from a real
provider; `summary.md` is generated, `results.json` holds the raw values.

- Code: `e3d455b` (clean tree), Python 3.13.18, Linux x86_64.
- Model: `openai/gpt-4o-mini` through OpenRouter (`https://openrouter.ai/api/v1`), profile
  `openai` (strict structured output). Set with `XWALK_BENCH_BASE_URL`,
  `XWALK_BENCH_MODEL`, `XWALK_BENCH_API_KEY_ENV=OPENROUTER_API_KEY`.
- Command: `python -m benchmarks.run --real --max-calls 2000 --out <this directory>`.
  No call cap was reached.
- Data: the frozen 50-record pilots (cafeteria_fcd, ncbi_disease, and their `nomatch`
  variants), one run each. n = 50 per variant: differences of a few rows are noise.
- Cost: **USD 0.374**, measured as the change in the OpenRouter key's reported usage
  (0.0181 before, 0.3924 after; the earlier 0.018 was a 5-record trial). Wall time
  16 min 11 s.
- Not run: dense retrieval, LinkTransformer (see `summary.md`, "Not run").

This run needed commit `e3d455b`: before it, the scorer's first call failed with HTTP 400
because OpenAI's strict mode rejected the stage schemas.

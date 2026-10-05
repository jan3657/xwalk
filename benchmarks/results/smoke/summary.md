# Benchmark summary (SYNTHETIC)

> SYNTHETIC: every LLM answer here comes from benchmarks/synthetic_llm.py, a deterministic string-similarity heuristic, not a language model. These numbers check that the runners, accounting and metrics work end to end. They are not evidence about xwalk's matching quality with a real model, and token counts are chars/4 estimates.

Generated 2026-10-05T12:55:47+00:00 at commit `cd88923a60c0df033bb2dea7459da1583cd05b3e` (dirty: False). Raw values: `results.json`.

## Matching: cafeteria_fcd (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.740 | 0.140 | 0.851 | 0.000 | - | 0 | - | - | - | - | 0.004 | 0.090 |
| fuzzy_difflib | 1.000 | 0.880 | 0.000 | 0.936 | 0.000 | - | 0 | - | - | - | - | 0.233 | 0.050 |
| bm25_top1 | 0.940 | 1.000 | 0.000 | 0.940 | - | - | 0 | 0.940 | 0 | - | - | 0.206 | 0.460 |
| retrieval_one_llm_pass | 1.000 | 0.900 | 0.060 | 0.947 | 0.000 | - | 0 | 0.940 | 0 | 50 | 35875 | 1.016 | 1.040 |
| xwalk_single_attempt | 1.000 | 0.900 | 0.060 | 0.947 | 0.000 | - | 0 | 0.940 | 0 | 98 | 76650 | 2.127 | 1.420 |
| xwalk_full | 1.000 | 0.880 | 0.080 | 0.936 | 0.000 | - | 0 | 0.940 | 0 | 111 | 84169 | 2.404 | 1.230 |

## Matching: cafeteria_fcd_nomatch (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.560 | 0.140 | 0.836 | 0.733 | 1.000 | 0 | - | - | - | - | 0.003 | 0.080 |
| fuzzy_difflib | 1.000 | 0.700 | 0.000 | 0.946 | 0.733 | 1.000 | 0 | - | - | - | - | 0.274 | 0.030 |
| bm25_top1 | 0.787 | 0.940 | 0.000 | 0.860 | 0.667 | 0.182 | 9 | 0.949 | 0 | - | - | 0.134 | 0.390 |
| retrieval_one_llm_pass | 0.921 | 0.760 | 0.100 | 0.909 | 0.714 | 0.455 | 3 | 0.949 | 0 | 47 | 32602 | 0.768 | 0.880 |
| xwalk_single_attempt | 0.921 | 0.760 | 0.100 | 0.909 | 0.714 | 0.455 | 3 | 0.949 | 0 | 90 | 67584 | 2.087 | 1.120 |
| xwalk_full | 1.000 | 0.700 | 0.160 | 0.946 | 0.714 | 0.455 | 0 | 0.949 | 0 | 123 | 84737 | 2.373 | 1.160 |

## Matching: ncbi_disease (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.893 | 0.560 | 0.000 | 0.746 | 0.364 | 0.727 | 3 | - | - | - | - | 0.010 | 0.360 |
| fuzzy_difflib | 0.897 | 0.580 | 0.160 | 0.765 | 0.615 | 0.727 | 3 | - | - | - | - | 1.971 | 0.160 |
| bm25_top1 | 0.737 | 0.760 | 0.000 | 0.727 | 0.417 | 0.455 | 6 | 0.744 | 0 | - | - | 0.149 | 0.410 |
| retrieval_one_llm_pass | 0.849 | 0.660 | 0.100 | 0.778 | 0.417 | 0.455 | 5 | 0.744 | 0 | 38 | 66274 | 1.347 | 1.550 |
| xwalk_single_attempt | 0.849 | 0.660 | 0.100 | 0.778 | 0.417 | 0.455 | 5 | 0.744 | 0 | 76 | 137414 | 1.872 | 1.880 |
| xwalk_full | 0.903 | 0.620 | 0.140 | 0.800 | 0.417 | 0.455 | 3 | 0.744 | 0 | 130 | 200753 | 2.617 | 2.040 |

## Matching: ncbi_disease_nomatch (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.850 | 0.400 | 0.000 | 0.756 | 0.733 | 0.880 | 3 | - | - | - | - | 0.009 | 0.340 |
| fuzzy_difflib | 0.857 | 0.420 | 0.040 | 0.783 | 0.815 | 0.880 | 3 | - | - | - | - | 1.867 | 0.150 |
| bm25_top1 | 0.625 | 0.640 | 0.000 | 0.702 | 0.944 | 0.680 | 8 | 0.840 | 0 | - | - | 0.152 | 0.400 |
| retrieval_one_llm_pass | 0.741 | 0.540 | 0.100 | 0.769 | 0.944 | 0.680 | 7 | 0.840 | 0 | 32 | 57485 | 0.611 | 0.870 |
| xwalk_single_attempt | 0.741 | 0.540 | 0.100 | 0.769 | 0.944 | 0.680 | 7 | 0.840 | 0 | 64 | 119107 | 1.637 | 1.720 |
| xwalk_full | 0.870 | 0.460 | 0.180 | 0.833 | 0.944 | 0.680 | 3 | 0.840 | 0 | 139 | 213328 | 2.749 | 1.930 |

## Pairwise: cafeteria_fcd (BM25 top-5 pairs; 3 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 184 | 78 | 1.000 | 0.692 | 0.818 |
| fuzzy | 184 | 78 | 1.000 | 0.692 | 0.818 |
| xwalk_scorer | 184 | 78 | 0.783 | 0.833 | 0.807 |

## Pairwise: cafeteria_fcd_nomatch (BM25 top-5 pairs; 2 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 170 | 58 | 1.000 | 0.776 | 0.874 |
| fuzzy | 170 | 58 | 1.000 | 0.776 | 0.874 |
| xwalk_scorer | 170 | 58 | 0.700 | 0.845 | 0.766 |

## Pairwise: ncbi_disease (BM25 top-5 pairs; 10 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 119 | 29 | 0.893 | 0.862 | 0.877 |
| fuzzy | 119 | 29 | 0.897 | 0.897 | 0.897 |
| xwalk_scorer | 119 | 29 | 0.500 | 0.966 | 0.659 |

## Pairwise: ncbi_disease_nomatch (BM25 top-5 pairs; 4 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 113 | 21 | 0.850 | 0.809 | 0.829 |
| fuzzy | 113 | 21 | 0.857 | 0.857 | 0.857 |
| xwalk_scorer | 113 | 21 | 0.417 | 0.952 | 0.580 |

## Clustering: cafeteria_fcd (as_singletons convention)

| prediction | pairwise F1 | B-cubed F1 | false-merge clusters | split gold clusters | pred. singletons | review | order: min F1 between runs |
|---|---|---|---|---|---|---|---|
| exact_mention | 0.000 | 0.970 | 1 | 2 | 48 | 0 | - |
| greedy_fuzzy_leader | 0.000 | 0.970 | 1 | 2 | 48 | 0 | 1.000 |
| via_xwalk_matching | 0.000 | 0.960 | 2 | 2 | 46 | 4 | - |

## Clustering: ncbi_disease (as_singletons convention)

| prediction | pairwise F1 | B-cubed F1 | false-merge clusters | split gold clusters | pred. singletons | review | order: min F1 between runs |
|---|---|---|---|---|---|---|---|
| exact_mention | 0.550 | 0.661 | 0 | 8 | 14 | 0 | - |
| greedy_fuzzy_leader | 0.556 | 0.668 | 0 | 8 | 12 | 0 | 1.000 |
| via_xwalk_matching | 0.529 | 0.706 | 1 | 5 | 21 | 7 | - |

## Not run

- dense_retrieval: skipped: needs xwalk[dense] and sentence-transformer weights (a large download); not run in this environment
- linktransformer: pending: benchmarks/adapters/linktransformer.py documents the adapter; running it needs the linktransformer package and model weights

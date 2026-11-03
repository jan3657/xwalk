# Benchmark summary (REAL)

Generated 2026-11-03T13:57:28+00:00 at commit `e3d455b4870beacae7479c213d8d131c1622ec4a` (dirty: False). Raw values: `results.json`.

## Matching: cafeteria_fcd (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.740 | 0.140 | 0.851 | 0.000 | - | 0 | - | - | - | - | 0.011 | 0.090 |
| fuzzy_difflib | 1.000 | 0.880 | 0.000 | 0.936 | 0.000 | - | 0 | - | - | - | - | 0.357 | 0.040 |
| bm25_top1 | 0.940 | 1.000 | 0.000 | 0.940 | - | - | 0 | 0.940 | 0 | - | - | 0.180 | 0.420 |
| retrieval_one_llm_pass | 0.979 | 0.940 | 0.000 | 0.948 | 0.000 | - | 0 | 0.940 | 0 | 50 | 39642 | 10.410 | 2.810 |
| xwalk_single_attempt | 1.000 | 0.660 | 0.280 | 0.795 | 0.000 | - | 0 | 0.940 | 0 | 98 | 85925 | 22.712 | 1.800 |
| xwalk_full | 1.000 | 0.700 | 0.260 | 0.824 | 0.000 | - | 0 | 0.960 | 0 | 167 | 147574 | 35.046 | 2.040 |

## Matching: cafeteria_fcd_nomatch (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 1.000 | 0.560 | 0.140 | 0.836 | 0.733 | 1.000 | 0 | - | - | - | - | 0.004 | 0.080 |
| fuzzy_difflib | 1.000 | 0.700 | 0.000 | 0.946 | 0.733 | 1.000 | 0 | - | - | - | - | 0.459 | 0.030 |
| bm25_top1 | 0.787 | 0.940 | 0.000 | 0.860 | 0.667 | 0.182 | 9 | 0.949 | 0 | - | - | 0.170 | 0.240 |
| retrieval_one_llm_pass | 0.841 | 0.880 | 0.000 | 0.892 | 0.833 | 0.455 | 6 | 0.949 | 0 | 47 | 35986 | 10.936 | 1.580 |
| xwalk_single_attempt | 1.000 | 0.500 | 0.320 | 0.781 | 0.889 | 0.727 | 0 | 0.949 | 0 | 91 | 77064 | 18.865 | 1.880 |
| xwalk_full | 1.000 | 0.540 | 0.360 | 0.818 | 0.800 | 0.364 | 0 | 0.974 | 0 | 179 | 151827 | 36.043 | 2.010 |

## Matching: ncbi_disease (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.893 | 0.560 | 0.000 | 0.746 | 0.364 | 0.727 | 3 | - | - | - | - | 0.011 | 0.350 |
| fuzzy_difflib | 0.897 | 0.580 | 0.160 | 0.765 | 0.615 | 0.727 | 3 | - | - | - | - | 2.229 | 0.160 |
| bm25_top1 | 0.737 | 0.760 | 0.000 | 0.727 | 0.417 | 0.455 | 6 | 0.744 | 0 | - | - | 0.171 | 0.260 |
| retrieval_one_llm_pass | 0.737 | 0.760 | 0.000 | 0.727 | 0.417 | 0.455 | 6 | 0.744 | 0 | 38 | 78031 | 9.502 | 1.680 |
| xwalk_single_attempt | 0.758 | 0.660 | 0.060 | 0.694 | 0.357 | 0.455 | 5 | 0.744 | 0 | 76 | 162003 | 17.017 | 1.900 |
| xwalk_full | 0.767 | 0.860 | 0.120 | 0.805 | 0.000 | 0.000 | 7 | 0.949 | 0 | 205 | 491058 | 44.897 | 3.550 |

## Matching: ncbi_disease_nomatch (all rows)

| method | acc. precision | coverage | review | final F1 | no-match P | no-match R | false accept on no-match | cand. recall | truncated | calls | tokens | wall s | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.850 | 0.400 | 0.000 | 0.756 | 0.733 | 0.880 | 3 | - | - | - | - | 0.023 | 0.330 |
| fuzzy_difflib | 0.857 | 0.420 | 0.040 | 0.783 | 0.815 | 0.880 | 3 | - | - | - | - | 2.496 | 0.100 |
| bm25_top1 | 0.625 | 0.640 | 0.000 | 0.702 | 0.944 | 0.680 | 8 | 0.840 | 0 | - | - | 0.138 | 0.260 |
| retrieval_one_llm_pass | 0.667 | 0.600 | 0.000 | 0.727 | 0.950 | 0.760 | 6 | 0.840 | 0 | 32 | 67851 | 8.363 | 1.630 |
| xwalk_single_attempt | 0.708 | 0.480 | 0.100 | 0.694 | 0.905 | 0.760 | 3 | 0.840 | 0 | 62 | 135092 | 16.433 | 1.800 |
| xwalk_full | 0.633 | 0.600 | 0.160 | 0.691 | 0.917 | 0.440 | 7 | 0.880 | 0 | 176 | 387909 | 38.050 | 3.390 |

## Pairwise: cafeteria_fcd (BM25 top-5 pairs; 3 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 184 | 78 | 1.000 | 0.692 | 0.818 |
| fuzzy | 184 | 78 | 1.000 | 0.692 | 0.818 |
| xwalk_scorer | 184 | 78 | 1.000 | 0.538 | 0.700 |

## Pairwise: cafeteria_fcd_nomatch (BM25 top-5 pairs; 2 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 170 | 58 | 1.000 | 0.776 | 0.874 |
| fuzzy | 170 | 58 | 1.000 | 0.776 | 0.874 |
| xwalk_scorer | 170 | 58 | 1.000 | 0.621 | 0.766 |

## Pairwise: ncbi_disease (BM25 top-5 pairs; 10 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 119 | 29 | 0.893 | 0.862 | 0.877 |
| fuzzy | 119 | 29 | 0.897 | 0.897 | 0.897 |
| xwalk_scorer | 119 | 29 | 0.722 | 0.897 | 0.800 |

## Pairwise: ncbi_disease_nomatch (BM25 top-5 pairs; 4 gold targets never blocked)

| method | pairs | positives | precision | recall | F1 |
|---|---|---|---|---|---|
| exact | 113 | 21 | 0.850 | 0.809 | 0.829 |
| fuzzy | 113 | 21 | 0.857 | 0.857 | 0.857 |
| xwalk_scorer | 113 | 21 | 0.655 | 0.905 | 0.760 |

## Clustering: cafeteria_fcd (as_singletons convention)

| prediction | pairwise F1 | B-cubed F1 | false-merge clusters | split gold clusters | pred. singletons | review | order: min F1 between runs |
|---|---|---|---|---|---|---|---|
| exact_mention | 0.000 | 0.970 | 1 | 2 | 48 | 0 | - |
| greedy_fuzzy_leader | 0.000 | 0.970 | 1 | 2 | 48 | 0 | 1.000 |
| via_xwalk_matching | 0.000 | 0.970 | 1 | 2 | 48 | 13 | - |
| xwalk_clustering | 0.000 | 0.970 | 1 | 2 | 48 | 9 | - |

## Clustering: ncbi_disease (as_singletons convention)

| prediction | pairwise F1 | B-cubed F1 | false-merge clusters | split gold clusters | pred. singletons | review | order: min F1 between runs |
|---|---|---|---|---|---|---|---|
| exact_mention | 0.550 | 0.661 | 0 | 8 | 14 | 0 | - |
| greedy_fuzzy_leader | 0.556 | 0.668 | 0 | 8 | 12 | 0 | 1.000 |
| via_xwalk_matching | 0.769 | 0.794 | 2 | 3 | 7 | 6 | - |
| xwalk_clustering | 0.506 | 0.635 | 1 | 7 | 19 | 13 | - |

## Not run

- dense_retrieval: skipped: needs xwalk[dense] and sentence-transformer weights (a large download); not run in this environment
- linktransformer: pending: benchmarks/adapters/linktransformer.py documents the adapter; running it needs the linktransformer package and model weights

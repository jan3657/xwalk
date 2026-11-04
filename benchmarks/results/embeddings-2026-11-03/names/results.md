# Embedding bake-off (2026-11-03)

Datasets: chebi, ncbi_disease, nlm_gene, cafeteria_fcd. Endpoint: https://openrouter.ai/api/v1. Dense depth 50. Every number below is from this run; none was edited by hand.

## Mean over datasets

Alone: the dense retriever by itself (for `bm25`, the job's BM25 by itself). Fused: the job's BM25 and the dense retriever by reciprocal rank fusion, as a hybrid job retrieves. A dash is a mean the preset did not finish on every dataset.

| method | model | MRR | R@1 | R@5 | fused MRR | fused R@5 | tokens | cost $ | failed datasets |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b-1024-names | `qwen/qwen3-embedding-8b` | 0.877 | 0.860 | 0.890 | 0.831 | 0.835 | 67804 | 0.000861 |  |
| qwen3-8b-1024 | `qwen/qwen3-embedding-8b` | 0.841 | 0.815 | 0.865 | 0.822 | 0.835 | 38718 | 0.000585 |  |
| bm25 | `tantivy` | 0.693 | 0.690 | 0.700 | — | — | — | — |  |

## chebi

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 8.5 |  |
| qwen3-8b-1024-names | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 33.0 |  |

## ncbi_disease

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.560 | 0.580 | 0.580 | 0.580 | 0.580 | 0.567 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 0.600 | 0.600 | 0.640 | 0.640 | 0.680 | 0.608 | 0.671 | built | 5.7 |  |
| qwen3-8b-1024-names | 1024 | 0.700 | 0.700 | 0.740 | 0.760 | 0.760 | 0.708 | 0.705 | built | 75.0 |  |

## nlm_gene

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.660 | 0.680 | 0.680 | 0.680 | 0.700 | 0.665 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 0.820 | 0.920 | 0.920 | 0.920 | 0.920 | 0.867 | 0.701 | built | 29.0 |  |
| qwen3-8b-1024-names | 1024 | 0.900 | 0.920 | 0.920 | 0.920 | 0.920 | 0.910 | 0.706 | built | 22.5 |  |

## cafeteria_fcd

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.940 | 0.940 | 0.940 | 0.940 | 0.940 | 0.940 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 0.840 | 0.940 | 0.980 | 0.980 | 1.000 | 0.891 | 0.916 | built | 7.2 |  |
| qwen3-8b-1024-names | 1024 | 0.840 | 0.940 | 0.960 | 0.980 | 1.000 | 0.890 | 0.914 | built | 24.1 |  |


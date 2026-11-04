# Embedding bake-off (2026-11-03)

Datasets: ncbi_disease, nlm_gene. Endpoint: https://openrouter.ai/api/v1. Dense depth 50. Every number below is from this run; none was edited by hand.

## Mean over datasets

Alone: the dense retriever by itself (for `bm25`, the job's BM25 by itself). Fused: the job's BM25 and the dense retriever by reciprocal rank fusion, as a hybrid job retrieves. A dash is a mean the preset did not finish on every dataset.

| method | model | MRR | R@1 | R@5 | fused MRR | fused R@5 | tokens | cost $ | failed datasets |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | `tantivy` | 0.616 | 0.610 | 0.630 | — | — | — | — |  |
| nemotron-3.5-8b | `nvidia/nemotron-3.5-embed-8b:free` | — | — | — | — | — | 15624 | 0.000000 | 1 |

## ncbi_disease

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.560 | 0.580 | 0.580 | 0.580 | 0.580 | 0.567 | — | — | — |  |
| nemotron-3.5-8b | 4096 | 0.580 | 0.700 | 0.760 | 0.780 | 0.780 | 0.622 | 0.707 | built | 905.0 |  |

## nlm_gene

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.660 | 0.680 | 0.680 | 0.680 | 0.700 | 0.665 | — | — | — |  |
| nemotron-3.5-8b | 4096 | — | — | — | — | — | — | — | — | 276.1 | EmbeddingAPIError: embeddings endpoint returned HTTP 404: 0 endpoints out of 1 requested are available matching your guardrail restrictions and data policy. We removed them for the following reasons ( |


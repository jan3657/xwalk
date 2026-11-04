# Embedding bake-off (2026-11-03)

Datasets: chebi, ncbi_disease, nlm_gene, cafeteria_fcd. Endpoint: https://openrouter.ai/api/v1. Dense depth 50. Every number below is from this run; none was edited by hand.

## Mean over datasets

Alone: the dense retriever by itself (for `bm25`, the job's BM25 by itself). Fused: the job's BM25 and the dense retriever by reciprocal rank fusion, as a hybrid job retrieves. A dash is a mean the preset did not finish on every dataset.

| method | model | MRR | R@1 | R@5 | fused MRR | fused R@5 | tokens | cost $ | failed datasets |
|---|---|---|---|---|---|---|---|---|---|
| gemini-2-it | `google/gemini-embedding-2` | 0.868 | 0.850 | 0.890 | 0.833 | 0.840 | 34531 | 0.006906 |  |
| codestral-embed | `mistralai/codestral-embed-2505` | 0.867 | 0.845 | 0.890 | 0.822 | 0.835 | 36110 | 0.005417 |  |
| qwen3-8b-1024 | `qwen/qwen3-embedding-8b` | 0.854 | 0.840 | 0.875 | 0.821 | 0.835 | 38718 | 0.000646 |  |
| titan-v2 | `amazon/titan-embed-text-v2` | 0.853 | 0.830 | 0.875 | 0.818 | 0.830 | 35203 | 0.000704 |  |
| gemini-001 | `google/gemini-embedding-001` | 0.835 | 0.780 | 0.905 | 0.821 | 0.845 | 32037 | 0.004806 |  |
| mistral-embed | `mistralai/mistral-embed-2312` | 0.807 | 0.755 | 0.870 | 0.808 | 0.835 | 43670 | 0.004367 |  |
| openai-ada-002 | `openai/text-embedding-ada-002` | 0.762 | 0.690 | 0.840 | 0.768 | 0.830 | 34277 | 0.003428 |  |
| gte-base | `thenlper/gte-base` | 0.754 | 0.700 | 0.815 | 0.792 | 0.830 | 37435 | 0.000187 |  |
| bm25 | `tantivy` | 0.693 | 0.690 | 0.700 | — | — | — | — |  |
| e5-large-v2 | `intfloat/e5-large-v2` | 0.690 | 0.635 | 0.740 | 0.779 | 0.805 | 39283 | 0.000393 |  |
| e5-base-v2 | `intfloat/e5-base-v2` | 0.684 | 0.625 | 0.780 | 0.770 | 0.835 | 39283 | 0.000196 |  |
| titan-v1 | `amazon/titan-embed-text-v1` | 0.683 | 0.630 | 0.775 | 0.780 | 0.805 | 34579 | 0.003458 |  |
| mpnet-base | `sentence-transformers/all-mpnet-base-v2` | 0.606 | 0.535 | 0.685 | 0.756 | 0.800 | 36895 | 0.000184 |  |
| paraphrase-minilm | `sentence-transformers/paraphrase-minilm-l6-v2` | 0.604 | 0.535 | 0.690 | 0.700 | 0.750 | 33870 | 0.000169 |  |
| minilm-l12 | `sentence-transformers/all-minilm-l12-v2` | 0.593 | 0.540 | 0.685 | 0.701 | 0.745 | 33870 | 0.000169 |  |
| minilm-l6 | `sentence-transformers/all-minilm-l6-v2` | 0.590 | 0.515 | 0.710 | 0.699 | 0.795 | 35959 | 0.000180 |  |
| multi-qa-mpnet | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 0.533 | 0.460 | 0.640 | 0.696 | 0.750 | 37435 | 0.000187 |  |
| qwen3-8b-noinst | `qwen/qwen3-embedding-8b` | 0.420 | 0.345 | 0.490 | 0.587 | 0.725 | 36735 | 0.000367 |  |
| bge-large-en | `baai/bge-large-en-v1.5` | — | — | — | — | — | 34557 | 0.000346 | 1 |
| bge-base-en | `baai/bge-base-en-v1.5` | — | — | — | — | — | 34557 | 0.000173 | 1 |
| gte-large | `thenlper/gte-large` | — | — | — | — | — | 34557 | 0.000346 | 1 |

## chebi

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 13.1 |  |
| qwen3-8b-noinst | 4096 | 0.100 | 0.340 | 0.380 | 0.580 | 0.660 | 0.238 | 0.630 | built | 85.0 |  |
| gemini-2-it | 3072 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 19.6 |  |
| gemini-001 | 3072 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 13.8 |  |
| mistral-embed | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 7.3 |  |
| codestral-embed | 1536 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 9.4 |  |
| titan-v2 | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 7.6 |  |
| titan-v1 | 1536 | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 | 0.984 | 1.000 | built | 6.0 |  |
| openai-ada-002 | 1536 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 7.1 |  |
| bge-large-en | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 5.8 |  |
| bge-base-en | 768 | 0.960 | 0.960 | 0.960 | 0.960 | 1.000 | 0.962 | 0.962 | built | 5.2 |  |
| gte-large | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 4.9 |  |
| gte-base | 768 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 6.3 |  |
| e5-large-v2 | 1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 6.0 |  |
| e5-base-v2 | 768 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | built | 7.3 |  |
| mpnet-base | 768 | 0.940 | 0.940 | 0.940 | 0.980 | 0.980 | 0.943 | 0.980 | built | 7.5 |  |
| multi-qa-mpnet | 768 | 0.780 | 0.780 | 0.780 | 0.780 | 0.800 | 0.781 | 0.800 | built | 7.7 |  |
| minilm-l6 | 384 | 0.780 | 0.980 | 1.000 | 1.000 | 1.000 | 0.849 | 0.849 | built | 6.0 |  |
| minilm-l12 | 384 | 0.780 | 0.780 | 0.800 | 0.800 | 0.800 | 0.783 | 0.783 | built | 5.4 |  |
| paraphrase-minilm | 384 | 0.780 | 0.800 | 0.800 | 0.800 | 1.000 | 0.793 | 0.793 | built | 5.4 |  |

## ncbi_disease

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.560 | 0.580 | 0.580 | 0.580 | 0.580 | 0.567 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 0.600 | 0.640 | 0.640 | 0.640 | 0.700 | 0.610 | 0.672 | built | 26.3 |  |
| qwen3-8b-noinst | 4096 | 0.160 | 0.180 | 0.180 | 0.180 | 0.280 | 0.168 | 0.380 | built | 45.5 |  |
| gemini-2-it | 3072 | 0.580 | 0.680 | 0.700 | 0.760 | 0.760 | 0.627 | 0.695 | built | 13.8 |  |
| gemini-001 | 3072 | 0.660 | 0.760 | 0.760 | 0.760 | 0.760 | 0.690 | 0.724 | built | 12.8 |  |
| mistral-embed | 1024 | 0.420 | 0.600 | 0.700 | 0.700 | 0.720 | 0.515 | 0.640 | built | 10.9 |  |
| codestral-embed | 1536 | 0.640 | 0.700 | 0.740 | 0.740 | 0.740 | 0.664 | 0.693 | built | 9.6 |  |
| titan-v2 | 1024 | 0.620 | 0.660 | 0.700 | 0.700 | 0.720 | 0.640 | 0.701 | built | 8.4 |  |
| titan-v1 | 1536 | 0.320 | 0.480 | 0.560 | 0.680 | 0.760 | 0.387 | 0.592 | built | 6.1 |  |
| openai-ada-002 | 1536 | 0.380 | 0.560 | 0.560 | 0.580 | 0.680 | 0.463 | 0.598 | built | 7.2 |  |
| bge-large-en | 1024 | — | — | — | — | — | — | — | — | 3.9 | EmbeddingAPIError: embeddings endpoint returned HTTP 400: HTTP 400: {"error":{"message":"You passed 513 input tokens and requested 0 output tokens. However, the model's context length is only 512 toke |
| bge-base-en | 768 | — | — | — | — | — | — | — | — | 3.5 | EmbeddingAPIError: embeddings endpoint returned HTTP 400: HTTP 400: {"error":{"message":"You passed 513 input tokens and requested 0 output tokens. However, the model's context length is only 512 toke |
| gte-large | 1024 | — | — | — | — | — | — | — | — | 3.8 | EmbeddingAPIError: embeddings endpoint returned HTTP 400: HTTP 400: {"error":{"message":"You passed 513 input tokens and requested 0 output tokens. However, the model's context length is only 512 toke |
| gte-base | 768 | 0.480 | 0.640 | 0.660 | 0.700 | 0.720 | 0.556 | 0.618 | built | 6.3 |  |
| e5-large-v2 | 1024 | 0.140 | 0.360 | 0.360 | 0.540 | 0.760 | 0.259 | 0.593 | built | 5.9 |  |
| e5-base-v2 | 768 | 0.200 | 0.520 | 0.520 | 0.660 | 0.760 | 0.317 | 0.590 | built | 6.6 |  |
| mpnet-base | 768 | 0.120 | 0.340 | 0.340 | 0.520 | 0.580 | 0.235 | 0.532 | built | 6.9 |  |
| multi-qa-mpnet | 768 | 0.040 | 0.340 | 0.340 | 0.340 | 0.540 | 0.159 | 0.547 | built | 6.7 |  |
| minilm-l6 | 384 | 0.040 | 0.280 | 0.340 | 0.500 | 0.560 | 0.132 | 0.474 | built | 5.6 |  |
| minilm-l12 | 384 | 0.220 | 0.440 | 0.440 | 0.480 | 0.680 | 0.298 | 0.526 | built | 3.7 |  |
| paraphrase-minilm | 384 | 0.180 | 0.400 | 0.560 | 0.600 | 0.680 | 0.291 | 0.530 | built | 4.8 |  |

## nlm_gene

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.660 | 0.680 | 0.680 | 0.680 | 0.700 | 0.665 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 0.900 | 0.920 | 0.920 | 0.920 | 0.920 | 0.905 | 0.701 | built | 31.9 |  |
| qwen3-8b-noinst | 4096 | 0.780 | 0.840 | 0.840 | 0.840 | 0.840 | 0.807 | 0.664 | built | 20.0 |  |
| gemini-2-it | 3072 | 0.920 | 0.920 | 0.920 | 0.920 | 0.920 | 0.920 | 0.698 | built | 14.0 |  |
| gemini-001 | 3072 | 0.700 | 0.920 | 0.920 | 0.920 | 0.920 | 0.810 | 0.687 | built | 13.2 |  |
| mistral-embed | 1024 | 0.820 | 0.920 | 0.920 | 0.920 | 0.920 | 0.870 | 0.699 | built | 7.0 |  |
| codestral-embed | 1536 | 0.920 | 0.920 | 0.920 | 0.920 | 0.920 | 0.920 | 0.694 | built | 12.5 |  |
| titan-v2 | 1024 | 0.900 | 0.920 | 0.920 | 0.920 | 0.920 | 0.910 | 0.681 | built | 5.8 |  |
| titan-v1 | 1536 | 0.580 | 0.680 | 0.680 | 0.700 | 0.920 | 0.626 | 0.694 | built | 5.9 |  |
| openai-ada-002 | 1536 | 0.820 | 0.920 | 0.920 | 0.920 | 0.920 | 0.870 | 0.694 | built | 6.9 |  |
| bge-large-en | 1024 | 0.620 | 0.700 | 0.920 | 0.920 | 0.920 | 0.674 | 0.693 | built | 5.5 |  |
| bge-base-en | 768 | 0.620 | 0.700 | 0.700 | 0.920 | 0.920 | 0.657 | 0.690 | built | 5.0 |  |
| gte-large | 1024 | 0.600 | 0.920 | 0.920 | 0.920 | 0.920 | 0.705 | 0.696 | built | 4.6 |  |
| gte-base | 768 | 0.620 | 0.700 | 0.700 | 0.920 | 0.920 | 0.658 | 0.690 | built | 5.7 |  |
| e5-large-v2 | 1024 | 0.700 | 0.700 | 0.700 | 0.700 | 0.920 | 0.706 | 0.670 | built | 5.6 |  |
| e5-base-v2 | 768 | 0.700 | 0.700 | 0.700 | 0.700 | 0.700 | 0.700 | 0.672 | built | 6.0 |  |
| mpnet-base | 768 | 0.480 | 0.520 | 0.560 | 0.660 | 0.700 | 0.505 | 0.685 | built | 15.6 |  |
| multi-qa-mpnet | 768 | 0.500 | 0.580 | 0.600 | 0.680 | 0.700 | 0.535 | 0.681 | built | 5.7 |  |
| minilm-l6 | 384 | 0.640 | 0.680 | 0.680 | 0.680 | 0.680 | 0.652 | 0.674 | built | 6.9 |  |
| minilm-l12 | 384 | 0.520 | 0.620 | 0.640 | 0.680 | 0.680 | 0.551 | 0.672 | built | 3.9 |  |
| paraphrase-minilm | 384 | 0.560 | 0.660 | 0.680 | 0.680 | 0.680 | 0.592 | 0.672 | built | 4.7 |  |

## cafeteria_fcd

| method | dim | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | fused MRR | index | seconds | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | — | 0.940 | 0.940 | 0.940 | 0.940 | 0.940 | 0.940 | — | — | — |  |
| qwen3-8b-1024 | 1024 | 0.860 | 0.940 | 0.960 | 1.000 | 1.000 | 0.899 | 0.912 | built | 38.5 |  |
| qwen3-8b-noinst | 4096 | 0.340 | 0.600 | 0.620 | 0.700 | 0.760 | 0.466 | 0.675 | built | 41.7 |  |
| gemini-2-it | 3072 | 0.900 | 0.960 | 0.960 | 1.000 | 1.000 | 0.927 | 0.938 | built | 12.6 |  |
| gemini-001 | 3072 | 0.760 | 0.940 | 0.960 | 0.980 | 0.980 | 0.839 | 0.875 | built | 14.9 |  |
| mistral-embed | 1024 | 0.780 | 0.960 | 0.960 | 0.960 | 1.000 | 0.845 | 0.892 | built | 7.8 |  |
| codestral-embed | 1536 | 0.820 | 0.940 | 0.940 | 0.980 | 0.980 | 0.883 | 0.903 | built | 13.8 |  |
| titan-v2 | 1024 | 0.800 | 0.920 | 0.940 | 0.940 | 0.960 | 0.861 | 0.891 | built | 8.8 |  |
| titan-v1 | 1536 | 0.640 | 0.940 | 0.940 | 0.960 | 0.960 | 0.734 | 0.832 | built | 6.9 |  |
| openai-ada-002 | 1536 | 0.560 | 0.880 | 0.940 | 0.980 | 0.980 | 0.714 | 0.782 | built | 8.0 |  |
| bge-large-en | 1024 | 0.440 | 0.720 | 0.840 | 0.940 | 0.960 | 0.570 | 0.717 | built | 5.4 |  |
| bge-base-en | 768 | 0.660 | 0.900 | 0.960 | 0.960 | 0.980 | 0.776 | 0.825 | built | 5.1 |  |
| gte-large | 1024 | 0.780 | 0.940 | 0.940 | 0.960 | 1.000 | 0.853 | 0.869 | built | 4.8 |  |
| gte-base | 768 | 0.700 | 0.920 | 0.940 | 0.960 | 0.980 | 0.802 | 0.858 | built | 6.2 |  |
| e5-large-v2 | 1024 | 0.700 | 0.900 | 0.940 | 0.940 | 0.980 | 0.796 | 0.854 | built | 5.7 |  |
| e5-base-v2 | 768 | 0.600 | 0.900 | 0.940 | 0.940 | 0.960 | 0.720 | 0.817 | built | 6.3 |  |
| mpnet-base | 768 | 0.600 | 0.940 | 0.940 | 0.940 | 0.960 | 0.741 | 0.827 | built | 13.1 |  |
| multi-qa-mpnet | 768 | 0.520 | 0.860 | 0.940 | 0.960 | 0.960 | 0.656 | 0.756 | built | 5.8 |  |
| minilm-l6 | 384 | 0.600 | 0.900 | 0.940 | 0.960 | 0.960 | 0.728 | 0.798 | built | 6.0 |  |
| minilm-l12 | 384 | 0.640 | 0.900 | 0.940 | 0.940 | 0.980 | 0.741 | 0.824 | built | 5.4 |  |
| paraphrase-minilm | 384 | 0.620 | 0.900 | 0.940 | 0.940 | 0.940 | 0.741 | 0.807 | built | 5.3 |  |


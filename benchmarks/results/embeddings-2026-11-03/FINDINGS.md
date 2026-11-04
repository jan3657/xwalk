# Embedding bake-off and retrieval experiments, 3 November 2026

Everything here was measured on the four in-repo pilots (CHEBI, NCBI Disease, NLM Gene,
CafeteriaFCD: 50 labelled mentions, about 200 targets each) through OpenRouter, with no
LLM in the loop. Raw numbers: `paid/`, `more/`, `nemotron-3.5-8b/` (`results.json`,
`results.md`, produced by `python -m benchmarks.embeddings`) and `analysis/` (the two
experiment scripts and their JSON output). Reported embedding cost for all of it: about
$0.08.

Read the numbers with the sample size in mind: one mention is 0.02 of a dataset's
recall, so differences under about 0.05 MRR are noise. CHEBI saturates (most models
score 1.0) and carries no signal.

## Models (mean MRR over four datasets, dense retrieval alone)

| preset | MRR | $/M tokens | note |
|---|---|---|---|
| gemini-2 | 0.873 | 0.20 | `input_type` makes no difference (gemini-2-it 0.868) |
| codestral-embed | 0.867 | 0.15 | a code model, second overall |
| qwen3-8b-1024 | 0.854 | 0.01 | same as 4096-d, a quarter of the index |
| titan-v2 | 0.853 | 0.02 | 1024-d |
| qwen3-8b | 0.852 | 0.01 | 4096-d |
| openai-3-large | 0.848 | 0.13 | |
| voyage-4-lite-it | 0.841 | 0.02 | 0.669 without `input_type` |
| gemini-001 | 0.835 | 0.15 | |
| pplx-4b | 0.817 | 0.03 | |
| mistral-embed | 0.807 | 0.10 | |
| openai-3-small | 0.767 | 0.02 | |
| bm25 | 0.693 | — | the jobs' own BM25 |
| minilm-l6 | 0.590 | 0.005 | the explorer's local default model |
| qwen3-8b-noinst | 0.420 | 0.01 | Qwen3 without its query instruction |

Nemotron 3.5 Embed 8B (free) was measured on NCBI Disease only: MRR 0.622 (Qwen3 8B
0.613, Gemini 2 0.646). Its free endpoint takes one text per request and about 20
requests a minute, and was then excluded by the account's zero-data-retention setting,
as were the Voyage models, the other free NVIDIA models and Voyage code/multimodal from
the second run on. BGE large/base, GTE large and LFM 2.5 fail on NCBI Disease: one
document exceeds their 512-token input limit and OpenRouter returns HTTP 400 rather than
truncating.

## Retrieval experiments

**Per-name vectors** (`analysis/multivec.py`). Embedding the label and each synonym as its
own vector, scoring a record by its best name and keeping the whole-document vector as one
more name, raises NCBI Disease MRR from 0.618 to 0.713 (Qwen3 8B) and from 0.646 to 0.712
(Gemini 2), and recall@20 from 0.68 to 0.76; elsewhere the change is within noise. Without
the whole-document vector, Gemini 2 loses on CHEBI (0.963).

**Mention context in the query** (`analysis/multivec.py`). Adding 100 characters either side
of the mention helps Gemini 2 on NCBI Disease (0.646 to 0.717) and CafeteriaFCD, but costs
both models 0.1 to 0.2 MRR on CHEBI. Not a safe default.

**Fusion with BM25** (`analysis/fusion.py`). Reciprocal rank fusion with the jobs' BM25
helps where BM25 and dense miss different records (NCBI Disease 0.613 to 0.703 MRR,
recall@20 0.68 to 0.74) and hurts where BM25 confidently returns wrong records: on NLM Gene
dense alone reaches 0.90 to 0.92 MRR and fusion 0.70, and recall@10 drops from 0.92 to
0.70. At recall@20 the fused list recovers on every dataset. Weighting dense higher, cutting
BM25 to its top few, or min-max score fusion (CombSUM) recover part of the NLM Gene loss
(best 0.80) without beating dense alone there; CombSUM is the most even across datasets.

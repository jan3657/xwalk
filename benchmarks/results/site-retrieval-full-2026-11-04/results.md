# Site retrieval on the pilot gold sets (2026-11-04, catalog full, depth 20)

| backend | dataset | retriever | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|---|
| neon | chebi | bm25 | 0.52 | 0.56 | 0.60 | 0.60 | 0.544 |
| neon | chebi | dense | 0.70 | 0.96 | 0.96 | 0.96 | 0.757 |
| neon | chebi | fused | 0.72 | 0.72 | 0.96 | 0.96 | 0.749 |
| neon | ncbi_disease | bm25 | 0.50 | 0.52 | 0.54 | 0.54 | 0.509 |
| neon | ncbi_disease | dense | 0.58 | 0.70 | 0.70 | 0.70 | 0.637 |
| neon | ncbi_disease | fused | 0.48 | 0.70 | 0.70 | 0.70 | 0.572 |
| neon | cafeteria_fcd | bm25 | 0.86 | 0.92 | 0.92 | 0.94 | 0.885 |
| neon | cafeteria_fcd | dense | 0.32 | 0.66 | 0.80 | 0.88 | 0.475 |
| neon | cafeteria_fcd | fused | 0.64 | 0.94 | 0.94 | 0.94 | 0.766 |

| backend | retriever | mean MRR | datasets |
|---|---|---|---|
| neon | bm25 | 0.646 | 3 |
| neon | dense | 0.623 | 3 |
| neon | fused | 0.696 | 3 |

Median search latency (seconds, from the machine that ran this):

- neon chebi: bm25 0.125, dense 0.967
- neon ncbi_disease: bm25 0.127, dense 1.548
- neon cafeteria_fcd: bm25 0.126, dense 0.457

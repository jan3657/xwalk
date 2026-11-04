# Site retrieval on the pilot gold sets (2026-11-04, catalog samples, depth 20)

| backend | dataset | retriever | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|---|
| bundled | chebi | bm25 | 0.60 | 0.60 | 0.60 | 0.60 | 0.600 |
| bundled | chebi | dense | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| bundled | chebi | fused | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| bundled | ncbi_disease | bm25 | 0.56 | 0.58 | 0.58 | 0.58 | 0.567 |
| bundled | ncbi_disease | dense | 0.70 | 0.70 | 0.74 | 0.76 | 0.708 |
| bundled | ncbi_disease | fused | 0.70 | 0.70 | 0.74 | 0.76 | 0.705 |
| bundled | nlm_gene | bm25 | 0.66 | 0.68 | 0.68 | 0.68 | 0.664 |
| bundled | nlm_gene | dense | 0.90 | 0.92 | 0.92 | 0.92 | 0.910 |
| bundled | nlm_gene | fused | 0.70 | 0.70 | 0.92 | 0.92 | 0.737 |
| bundled | cafeteria_fcd | bm25 | 0.94 | 0.94 | 0.94 | 0.94 | 0.940 |
| bundled | cafeteria_fcd | dense | 0.84 | 0.94 | 0.96 | 0.98 | 0.892 |
| bundled | cafeteria_fcd | fused | 0.88 | 0.94 | 0.96 | 0.98 | 0.914 |
| neon | chebi | bm25 | 0.60 | 0.60 | 0.60 | 0.60 | 0.600 |
| neon | chebi | dense | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| neon | chebi | fused | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| neon | ncbi_disease | bm25 | 0.56 | 0.58 | 0.58 | 0.58 | 0.567 |
| neon | ncbi_disease | dense | 0.70 | 0.74 | 0.74 | 0.74 | 0.708 |
| neon | ncbi_disease | fused | 0.70 | 0.70 | 0.74 | 0.74 | 0.704 |
| neon | nlm_gene | bm25 | 0.66 | 0.66 | 0.66 | 0.66 | 0.660 |
| neon | nlm_gene | dense | 0.90 | 0.92 | 0.92 | 0.92 | 0.910 |
| neon | nlm_gene | fused | 0.68 | 0.92 | 0.92 | 0.92 | 0.745 |
| neon | cafeteria_fcd | bm25 | 0.94 | 0.94 | 0.94 | 0.94 | 0.940 |
| neon | cafeteria_fcd | dense | 0.84 | 0.94 | 0.96 | 0.98 | 0.889 |
| neon | cafeteria_fcd | fused | 0.88 | 0.94 | 0.96 | 0.98 | 0.914 |

| backend | retriever | mean MRR (4 datasets) |
|---|---|---|
| bundled | bm25 | 0.693 |
| bundled | dense | 0.877 |
| bundled | fused | 0.839 |
| neon | bm25 | 0.692 |
| neon | dense | 0.877 |
| neon | fused | 0.841 |

Median search latency (seconds, from the machine that ran this):

- bundled chebi: bm25 0.001, dense 0.200
- bundled ncbi_disease: bm25 0.001, dense 0.180
- bundled nlm_gene: bm25 0.001, dense 0.167
- bundled cafeteria_fcd: bm25 0.001, dense 0.173
- neon chebi: bm25 0.128, dense 0.323
- neon ncbi_disease: bm25 0.125, dense 0.333
- neon nlm_gene: bm25 0.128, dense 0.466
- neon cafeteria_fcd: bm25 0.125, dense 0.780

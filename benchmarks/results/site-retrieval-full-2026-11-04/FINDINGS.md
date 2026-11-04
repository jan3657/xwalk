# Site retrieval on the full catalogs, 4 November 2026

`python -m benchmarks.site_retrieval --backend neon --catalog full --schema xwalk_full
--dimensions 768 --out benchmarks/results/site-retrieval-full-2026-11-04` (raw numbers:
`results.json`, `results.md`), against the 200-term samples on the same backend
(`../site-retrieval-2026-11-04/`, Neon rows). Same pilot gold sets (50 mentions each),
same query template, depth 20, reciprocal-rank fusion; the full catalogs are ChEBI
(219,011 terms, 852,943 vectors), CTD diseases (13,323; 104,630) and FoodOn (28,684;
65,301) at `halfvec(768)` under HNSW. NLM Gene is not evaluated: NCBI Gene was not loaded
(the operator's choice) and genes are not routed with the full catalogs.

| dataset | retriever | MRR, samples | MRR, full | R@20, samples | R@20, full |
|---|---|---|---|---|---|
| chebi | BM25 | 0.600 | 0.544 | 0.60 | 0.60 |
| chebi | dense | 1.000 | 0.757 | 1.00 | 0.96 |
| chebi | fused | 1.000 | 0.749 | 1.00 | 0.96 |
| ncbi_disease | BM25 | 0.567 | 0.509 | 0.58 | 0.54 |
| ncbi_disease | dense | 0.708 | 0.637 | 0.74 | 0.70 |
| ncbi_disease | fused | 0.704 | 0.572 | 0.74 | 0.70 |
| cafeteria_fcd | BM25 | 0.940 | 0.885 | 0.94 | 0.94 |
| cafeteria_fcd | dense | 0.889 | 0.475 | 0.98 | 0.88 |
| cafeteria_fcd | fused | 0.914 | 0.766 | 0.98 | 0.94 |

- **Ranking suffers, recall does not.** Against 200,000 distractors instead of 200, the
  right term's rank drops (fused MRR down 0.13 to 0.25), but it is still in the top 20 for
  96%, 70% and 94% of mentions. Picking it out of the candidates is the selector's job
  (PLAN.md section 13), not retrieval's.
- **Dense collapses on FoodOn** (0.889 to 0.475): the full ontology has many near-duplicate
  food terms (EFSA FoodEx2 entries, "... food product" variants). Exact search at 768-d
  scores the same (0.470, `neon_truncation`), so the HNSW index is not the cause; BM25
  holds (0.885), and fusion recovers to 0.766.
- **HNSW costs about one CTD mention**: dense MRR 0.637 here against 0.633 exact; the
  per-dataset differences between exact and HNSW are within one mention.
- **Latency** (medians from this container to Frankfurt): BM25 125 ms; dense 0.46 to
  1.5 s, almost all of it the OpenRouter query embedding.

Embedding cost of the full load, as OpenRouter reported it: CTD $0.0190, FoodOn $0.0134,
ChEBI $0.4406 (two runs of the CTD load: the test load at 1024-d was dropped and reloaded
at 768-d, so CTD was paid twice). Query embeddings for these evaluations were not
captured; at the measured rate they round to $0.0000.

# Site retrieval, bundled vs Neon, 4 November 2026

`python -m benchmarks.site_retrieval --backend bundled --backend neon --out
benchmarks/results/site-retrieval-2026-11-04` (raw numbers: `results.json`, `results.md`).
The site's four sample ontologies are the pilots' target catalogs, so the pilot gold sets
(50 labelled mentions each) measure exactly what the deployed site retrieves: the site's
query template over the bare mention, BM25 and dense at depth 20, fused by reciprocal rank
as `ner_site.mapping` does. No LLM was called; each backend made one query embedding per
mention (200 per backend, Qwen3-Embedding 8B through OpenRouter; usage was not captured
by the script, so the cost is unknown here, though at the measured $0.01 a million tokens
it rounds to $0.0000).

**The full catalogs were not evaluated.** The Neon org is still on the free plan, which
cannot hold the full load (about 12.5 GB at 1024 dimensions), so `--catalog full` was not
run. The comparison asked for (full catalogs against the 200-term samples) remains open;
the same command with `--backend neon --catalog full` produces it once the load exists.

| retriever | bundled mean MRR | Neon mean MRR |
|---|---|---|
| BM25 | 0.693 | 0.692 |
| dense | 0.877 | 0.877 |
| fused (what the site returns) | 0.839 | 0.841 |

- **Parity.** Neon reproduces the bundled backend within noise (one mention is 0.02 of a
  dataset's recall). Dense differs only by tie order and the HNSW/exact-scan choice
  (ncbi_disease recall@20 0.76 bundled, 0.74 Neon; nlm_gene recall@5 dense identical).
  BM25 (`lakebase_text` against Tantivy) differs on one NLM Gene mention below rank 1.
- **Fusion costs NLM Gene.** As in `embeddings-2026-11-03/FINDINGS.md`: dense alone 0.910
  MRR, fused 0.737 (bundled) and 0.745 (Neon), because BM25 confidently ranks the wrong
  gene first; recall@10 recovers to 0.92. CafeteriaFCD is the only dataset where BM25
  beats dense (0.940 against 0.89).
- **Latency** (medians, from this container in Europe to Frankfurt): Neon BM25 about
  125 ms a search (one HTTPS round trip), Neon dense 0.32 to 0.78 s (the query embedding
  plus a round trip; 8 searches in flight), bundled dense 0.17 to 0.20 s (embedding only),
  bundled BM25 1 ms. Every search succeeded on both backends (no errors).

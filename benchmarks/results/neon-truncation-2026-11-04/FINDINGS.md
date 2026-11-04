# Vector format for the full Neon load, 4 November 2026

Two measurements on the Neon project (PostgreSQL 18.6, pgvector, Launch plan) chose
`halfvec(768)` for the full catalogs.

## Storage per vector

The 3,132 sample vectors (real texts, 1024-d Qwen3) copied server side into scratch
tables, truncated with `l2_normalize(subvector(embedding, 1, D))`, then an HNSW index
(`m = 16, ef_construction = 64`); bytes per vector from `pg_relation_size`,
`pg_total_relation_size` and `pg_indexes_size` (the primary key included):

| D | Storage | Heap | TOAST | HNSW | Total |
|---|---|---|---|---|---|
| 1024 | default | 146 | 2,791 | 2,472 | 5,495 |
| 1024 | `set storage plain` | 2,731 | 97 | 2,472 | 5,385 |
| 1016 | default | 146 | 2,791 | 2,472 | 5,495 |
| 1000 | default | 141 | 2,791 | 2,472 | 5,490 |
| 960 | default | 1,514 | 813 | 2,472 | 4,886 |
| 896 | default | 1,972 | 110 | 2,469 | 4,637 |
| 768 | default | 2,004 | 21 | 1,854 | 3,965 |
| 512 | default | 1,214 | 8 | 1,237 | 2,545 |

A slightly smaller dimension does not help: at 1000 or 1016 the row (vector plus the
text, ids and headers) is still over the 2 KB TOAST threshold, so the vector still moves
out of line. `STORAGE PLAIN` keeps it inline but saves only 2%, because an 8 KB page then
holds three rows either way. The HNSW index costs the same from 896 to 1024 for the same
reason (three elements a page). The steps come at about 960 (rows start to fit inline),
768 (four HNSW elements a page) and 512.

## Quality at full scale

`python -m benchmarks.neon_truncation --dataset ncbi_disease --ontology ctd-disease
--dims 1024 768 512` (`ncbi_disease--ctd-disease.json`): the full CTD disease vocabulary
(13,323 records, 104,630 vectors) loaded at 1024-d, the 50 NCBI Disease mentions embedded
once (the site's query template and encoder), exact search at each D, records scored by
their best vector, depth 20:

| D | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| 1024 | 0.58 | 0.70 | 0.70 | 0.70 | 0.633 |
| 768 | 0.58 | 0.70 | 0.70 | 0.70 | 0.633 |
| 512 | 0.44 | 0.68 | 0.70 | 0.70 | 0.560 |

On the 200-term pilot catalog the same mentions scored 0.708 dense MRR; the full
vocabulary's distractors cost about 0.08. Truncating to 512 costs a further 0.07 (seven
mentions lose rank 1), which the pilots (0.831 against 0.836 mean) could not show. 768
matches 1024 on every cut-off. One dataset of 50 mentions, so a small loss at 768 cannot
be ruled out elsewhere; NCBI Disease is the pilot where dense retrieval is weakest.

## Decision

The full catalogs load as `halfvec(768)` into schema `xwalk_full` (about 28% less vector
storage than 1024: roughly 7.7 GB against 10.6 GB for 1.93 M vectors, at the measured
bytes per vector), with `maintenance_work_mem = 1GB` set on the database for the HNSW
builds. The samples stay at 1024 in schema `xwalk`.

Reported cost of the CTD test load: 104,630 texts, 1,897,340 tokens, $0.019 (the dry run
estimated 3.07 M tokens and $0.031). The 50 query embeddings were not reported separately.

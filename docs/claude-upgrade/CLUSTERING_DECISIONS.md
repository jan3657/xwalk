# Flat clustering: decision record (task 04)

Written 5 October 2026 before the first clustering code. Inputs: the July design
(`docs/superpowers/specs/2026-07-28-hierarchical-clustering-design.md`), its M1 plan
(`docs/superpowers/plans/2026-07-28-hierarchical-clustering-m1.md`, tasks 5-9 read), and
CONTRACTS.md section 10, which already settles most points. This page records how the
eight conflicts listed in PLAN.md "Clustering contract" are resolved in code, and the
choices the contract left open. Code: `src/xwalk/cluster/`. Status: **experimental**
(see "Gates" at the end).

## The eight conflicts

| # | Conflict | Resolution in 0.2 |
|---|---|---|
| 1 | Assignment to an immutable taxonomy node vs merging/modifying it | No taxonomy in 0.2 (CONTRACTS 10.1). Every cluster is minted in-run; the plan's taxonomy-wins merge rule and `Provenance.TAXONOMY` are not built. Assigning to an external node is a later, separate operation; nothing here can mutate an external record. |
| 2 | Source collection and order must be part of run identity | The run fingerprint hashes the *ordered* snapshot `[(id, hash)]` after the ordering rule is applied, the ordering rule itself, templates, prompts, LLM, pool settings and policy. Cluster ID = `hash(run_fp, seed_member_id)`. Default order: stable sort on case-folded, whitespace-collapsed query text, then id (input permutation does not change the run); `order: input` keeps file order and makes order part of identity. |
| 3 | Historical pool representations must be reconstructable | `cluster_revisions(cluster_id, revision, members, representation)` is append-only. Each decision row stores the `(cluster_id, revision)` pairs **retrieved** and **shown**, the exact system and user prompt, and the raw response. `tests/test_cluster_engine.py` re-renders every stored prompt from the stored revisions and compares byte for byte. |
| 4 | Exports must use the best stable revision when refinement oscillates | Every iteration end is stored as a state revision (full member -> cluster/outcome map plus cluster revision pointers). `converged`: the final revision is a fixpoint and is exported. `oscillation` / `max_iterations`: the revision with the fewest unresolved sources, ties earliest (CONTRACTS 10.7), is exported; the manifest names both the selected and the last revision. |
| 5 | An injected incumbent must survive truncation | In reassignment the incumbent is placed in the shown list; when the list is full it replaces the lowest-ranked challenger. Its representation excludes the member being reconsidered. A test truncates to one challenger slot and checks the incumbent is shown. |
| 6 | Record retrieved separately from shown | Every select/novelty/merge/comparative decision row has `retrieved` (all retrieved candidates with rank and fused score) and `shown` (the keyed, rendered subset). Export `decisions.jsonl` keeps both. |
| 7 | Verifiers need actual bounded member evidence | Every cluster block shown to the model lists up to `pool.member_evidence` (default 5) member renders (`candidate` template, each capped at `pool.evidence_chars`): seed first, then join order. No exemplar counts. A member joins only if the verifier judges it equivalent to the cluster as shown (all listed members), not to one member. |
| 8 | Full pool rebuild per mint is not scalable | `PoolIndex` is an in-memory incremental BM25 inverted index (plus an optional dense head through the existing `Encoder` protocol). A mint appends one document; a join re-indexes only that cluster's document; scores are computed from running counters. No Tantivy rebuild. Measured below. |

## Choices the contract left open

- **Outcomes.** Stored per source: `assigned`, `seed` (minted the cluster), `deferred`,
  `failed`. Exported: `assigned`; `singleton` for a seed whose cluster still has one
  member; a seed whose cluster gained verified members is exported `assigned` (its
  equivalence was verified by the joining or merge decision). A source that seeds a
  second cluster (moved out of its first, later minted again) gets
  `hash(run_fp, seed, n)`. `deferred` becomes
  `needs_review`. Sources not reached by an aborted run are `pending`. Each source
  appears exactly once in `members.csv`; `needs_review`, `failed` and `pending` are in
  no cluster and are repeated in `unresolved.csv` with their best proposal.
- **Empty pool.** The first source (or any source when no live cluster exists) is
  minted without a novelty call; provenance `empty_pool`. A novelty verdict against no
  candidates carries no information.
- **Expansion before novelty.** Ordinary retrieval shows the top `shown_limit`
  candidates. On abstention, up to `max_expansion_pages` further pages are shown from
  deeper retrieval (`retrieve_limit` doubled per page up to `expand_limit`) and, when the
  live pool has at most `scan_below` clusters, from the whole pool. Mint provenance is
  `pool_exhausted` (every live cluster was shown), `retrieval_exhausted` (no further
  retrievable candidate), or `bounded` (the configured pages were shown and more
  candidates existed). `mint_requires` sets the weakest provenance that may mint; weaker
  ends as `deferred` with reason `expansion_incomplete:<provenance>`. **Default
  `bounded`**, i.e. "after the configured expansion" (CONTRACTS 10.5). The first draft
  defaulted to `retrieval_exhausted`; a test with 30 concepts sharing one common token
  showed that every late novel record was then deferred for ever, because the common
  token keeps retrieval from ever being exhausted. The stricter modes stay available and
  the provenance is exported per cluster, so a reader can tell a bounded search from an
  exhaustive one. A lexical miss is never proof of novelty by itself: novelty also needs
  the novelty verdict.
- **Merge policy.** One select call per cluster proposes a neighbour, one merge
  verification decides it against both clusters' current evidence at `merge_accept_at`
  (default 0.8, stricter than assignment). A cluster takes part in at most one merge per
  iteration, so a merged representation is only judged again in the next iteration by a
  fresh decision. No union-find, no transitive closure. Winner: larger cluster, ties to
  the earlier-minted one. Proposals in `[merge_review_floor, merge_accept_at)` are
  recorded as `merge_review`; members stay where they are.
- **Contradictory evidence.** A-B accepted and B-C proposed: C is verified against the
  cluster {A, B}; a "not equivalent" verdict defers it, and a later merge of {A, B} with
  {C} needs its own verified decision.
- **Refinement.** Each iteration (at most `max_refine_iterations`, default 2): retry
  deferred and failed sources (may mint), consolidate, reassign members of clusters
  with at least two members (never mints; switch only on a comparative verdict at
  `comparative_accept_at`; a null selection re-verifies the incumbent and a failed
  re-verification removes the member to `deferred`). Stop: `converged` (iteration changed
  nothing), `oscillation` (state hash seen before), `max_iterations`.
- **Transactions and resume.** Work is a deterministic sequence of steps
  (`stream/<source>`, `retry/<i>/<source>`, `consolidate/<i>/<cluster>`,
  `reassign/<i>/<source>`, `end/<i>`). Each step's decisions, cluster revisions,
  assignment changes and completion marker commit in one SQLite transaction. A phase's
  work list is persisted when the phase starts. Resume rebuilds state from the database
  (the pool index is built once on open, not per mint) and skips completed steps; an
  interrupted step's calls are repeated. Resuming a *finished* run with `failed` sources
  in its exported state runs a resume round `r` (last revision + 1, review fix in
  0.2.0rc1): `resume/<r>/restore` appends rows that make the exported state current,
  `resume/<r>/<source>` re-decides each failed source, and `resume/<r>/end` stores state
  revision `r` as the selected one. This mirrors CONTRACTS.md section 2 (failed records
  are retried on resume); earlier rows stay as history.
- **Processing** is sequential (`batch_size=1` reference mode): deterministic, slow.
  Optimistic batching is not in 0.2.
- **Call bounds.** Every stage receives the same `BudgetedLLM`. Per decision at most
  `2 + max_expansion_pages` calls (stream/retry), 2 (consolidate, reassign). A positive
  `max_calls` below `2 + max_expansion_pages` is refused up front (`usage`, exit 2): a
  step commits only when it finishes, so a smaller cap could never get past it.
- **Run directory.** `cluster.sqlite`, `manifest.json`, `members.csv`, `clusters.csv`,
  `unresolved.csv`, `decisions.jsonl`. Same collision rules as `match` (CONTRACTS 9):
  another fingerprint is refused, the same fingerprint resumes, a complete run makes
  zero calls and rewrites identical exports. Because the snapshot is part of identity,
  an edited source collection is a different run.

## Not in this slice

Taxonomy seeding, broader/narrower and multi-level hierarchy, fixed cluster counts,
LLM naming/glosses and name verification (cluster representation is its members), a
review overlay with clustering verbs (`unresolved.csv` is the worksheet; applying
decisions is a later task), `inspect`/`explain` for cluster runs, concurrency.

## Scaling check

Measured 5 October 2026 on the session container (4 vCPU x86-64, Python 3.11), pure
Python, synthetic documents of 2-6 words from a 5,000-word vocabulary; each step is one
query (the new source's search) followed by one mint. Script: the snippet below.

| Pool built to N clusters | Incremental `PoolIndex` (query + append) | Rebuild per mint, same index in Python | Rebuild per mint, Tantivy on disk (`BM25Retriever.build`) |
|---|---|---|---|
| 100 | — | — | 1.83 s |
| 250 | 0.004 s | 0.086 s | 5.84 s |
| 1,000 | 0.022 s | 1.43 s | not run |
| 4,000 | 0.19 s | not run | not run |
| 16,000 | 1.71 s | not run | not run |

The incremental index is not linear either: on a fixed vocabulary each posting list
grows with N, so a query costs O(N / vocabulary); 16,000 mints still take under two
seconds. Two quadratic costs were found by this measurement and removed (copying the
id set, and intersecting the exclusion set with every key, on each search). End to end
with the scripted judge, 1,200 sources in 400 three-member concepts gave 400 correct
clusters in 4.0 s for 4,101 scripted calls (about 1 ms of engine and SQLite overhead per
call); 300 sources: 0.9 s. A real model's latency dominates either way. The optional
dense head is brute-force cosine in Python and was not measured at scale; a vector
index is the next step if it is used on large pools.

```python
import random, time
from xwalk.cluster.pool import PoolIndex

rng = random.Random(0)
vocab = [f"w{i}" for i in range(5000)]
docs = [" ".join(rng.choice(vocab) for _ in range(rng.randint(2, 6))) for _ in range(16000)]
index, start = PoolIndex(), time.perf_counter()
for i, doc in enumerate(docs):
    index.search(doc, 20)
    index.upsert(f"K{i}", doc)
print(time.perf_counter() - start)
```

## Gates (task 04 acceptance)

| Gate | State | Evidence |
|---|---|---|
| Every source exactly once; accepted membership a partition; singletons not called verified | met | `test_every_source_is_accounted_for_once...`; `write_exports` refuses to write a non-partition |
| Synonyms co-cluster, related-but-distinct stay apart, novel records defined | met with a scripted judge only | `test_synonyms_cocluster...` |
| A-B, B-C cannot force A-C | met | `test_a_contradictory_chain...`, `test_a_cluster_takes_at_most_one_merge_per_iteration` |
| Deterministic ids and results | met | `test_fixed_inputs_...`, `test_label_order_makes_input_permutation_irrelevant` |
| Order change documented | met | `test_input_order_is_part_of_identity_and_can_change_results`, guide "Limits" |
| Interruption and resume reproduce a clean run | met | call limit at 5 points across stream and refinement, and KeyboardInterrupt: every deterministic table identical |
| History reconstructable; selected-revision export correct | met | every stored prompt rebuilt from stored revisions; `test_exports_use_the_selected_revision...` |
| Candidate and call bounds under refinement | met | per-step call bound, shown and member-evidence bounds, `--max-calls` exact |
| No full index rebuild per mint; scaling measured | met | `test_the_pool_index_is_updated_per_change...`; table above |
| Small labelled real sample evaluated | **not met** | no inference endpoint was configured; nothing was run against a real model |

Because the last gate is not met, and the thresholds are uncalibrated, clustering ships
marked **experimental** (CLI warning `experimental`, manifest `experimental: true`,
`docs/guide/clustering.md`).

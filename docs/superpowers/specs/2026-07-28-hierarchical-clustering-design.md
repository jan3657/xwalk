# Hierarchical LLM-adjudicated clustering — design

Date: 2026-07-28. Status: approved (conceptual design frozen after three review rounds).

## Purpose

Extend xwalk from record-to-record mapping to hierarchical clustering: given ~17k
label-only concepts, canonicalize trivial variants ("Dark Chocolate" ≡ "chocolate
Dark"), then roll the canonical concepts up through configurable levels (≈1k, then
≈100), where at every level the candidate parents are existing taxonomy nodes plus
clusters minted during the run. The LLM makes every decision; classical techniques
(embeddings, string similarity) serve only as retrieval and ordering heuristics, never
as the decision-maker. All of xwalk's guarantees carry over: ledger as source of truth,
resume, fingerprints, opaque candidate keys, review overlay, token-free structural
evaluation.

## One engine, distinct level semantics

Canonicalization and roll-up share the orchestration engine — stream, decide, mint,
refine — but are not the same operation:

- **Equivalence** levels (level 0): symmetric relation, members become aliases of the
  cluster, a partition is required.
- **Subsumption** levels (level 1..N): asymmetric is-a, members remain distinct
  entities under a parent; a DAG may be permitted via `Cardinality.MANY`.

Each `LevelSpec` declares its semantics explicitly:

```python
LevelSpec(
    name="canonical",
    relation=Relation.EQUIVALENCE,        # or SUBSUMPTION; selects prompt slots + merge semantics
    cardinality=Cardinality.ONE,          # ONE (partition/tree) or MANY (DAG)
    multi_parent=MultiParentPolicy(...),  # only when cardinality=MANY; see below
    eligibility=CandidateEligibility(taxonomy=True, minted=True, provisional=True),
    taxonomy=<Source | None>,             # existing nodes seeding the pool
    taxonomy_constraints=TaxonomyConstraints(restrict_to=...),  # nodes valid at this level
    representation=ClusterRepresentation(exemplars=5, include_gloss=True),
    minting=MintingPolicy(allowed=True, ...),
    policy=ClusterPolicy(...),            # all thresholds; see below
)
```

Granularity is prompt-defined, not count-targeted: the level's relation prompt states
what a parent at this level means, and the LLM owns the boundary. Diagnostics (below)
make a mis-calibrated granularity prompt visible as evidence.

Existing taxonomy nodes are **immutable**: never renamed, never merged into, never
retired, by the engine or by reviewers.

## Terminology: verifier, not independent scorer

Every second opinion in this design is a **separate verifier call**, not an
"independent judgment": unless a different model is explicitly configured
(`verifier_llm`), the selector and verifier share a backend and are not statistically
independent. Verifier confidence is a **routing signal** for thresholds, never proof of
correctness. The verifier calls are: assignment verifier, novelty verifier, merge
verifier, comparative verifier (reassignment), and name verifier (finalization).

## Thresholds

Assignment, novelty, merge, comparative switching, and final-name verification are
different tasks and get **separately configurable thresholds**:

```python
ClusterPolicy(
    assign_accept_at, assign_review_floor,
    novelty_accept_at, novelty_review_floor,
    merge_accept_at, merge_review_floor,
    comparative_accept_at,
    name_verify_accept_at,
    audit_rate,
)
```

All shipped defaults are **provisional** and must be calibrated on a small adjudicated
development set (~100–300 records with human decisions) before a production run; the
diagnostics module supports a threshold sweep for this. No threshold is inherited from
`MatchPolicy`.

## The streaming pass

Per member, in recorded deterministic order (default: stable sort on normalized label;
an optional embedding-similarity ordering is a pluggable heuristic only):

```
1. RETRIEVE   top-k from the pool snapshot (eligible taxonomy nodes + minted clusters)
2. SELECT     opaque keyed candidates → C07 or null                     → LLM call
   ├─ candidate selected
   │   3a. ASSIGNMENT VERIFIER: is this assignment correct, member vs   → LLM call
   │       full parent representation? Confidence routes:
   │       ≥ assign_accept_at → ASSIGN · ≥ review_floor → review · below → DEFER
   └─ null selected
       3b. RETRIEVAL EXHAUSTION before any mint may be considered:
           broaden k (doubling up to exhaust_limit), query all configured
           retrieval heads, optionally reformulate the query once (LLM).  → LLM call (opt)
           New candidates → back to SELECT against the expanded list.
       3c. NOVELTY VERIFIER, against the exhausted candidate set:        → LLM call
           (i) is each retrieved candidate genuinely not a valid parent?
           (ii) does this member warrant a NEW cluster at this granularity?
           ≥ novelty_accept_at → MINT (provisional)                      → LLM call (name+gloss)
           in [novelty_review_floor, accept) → DEFER
           below floor (a candidate was valid after all) → DEFER with re-select hint
```

A retrieval miss is never described as global proof that no parent exists: every mint
records whether it followed ordinary or exhausted retrieval (`mint_provenance`), and
minting is only reachable through the exhaustion step.

DEFER is a first-class outcome distinct from review: early nulls often mean "the right
cluster doesn't exist yet"; deferred members are retried mechanically in phase R1.

**Minted clusters are provisional.** A mint persists: stable cluster id, seed member
id, LLM-generated name and gloss, aliases (grows as equivalence members attach),
provenance (`minted` vs `taxonomy`), creation snapshot id, mint provenance, and
deterministic exemplars (top-K members by verifier confidence, ties by member id).
Provisional representations refresh only at phase boundaries, never mid-batch.

**Stable cluster ids** are deterministic hashes of `(run_fingerprint, level,
seed_member_id)` — independent of generated names and glosses. Renaming, at
finalization or by a reviewer, never changes cluster identity. Taxonomy nodes keep
their source ids.

## Refinement phases (mandatory, per level)

1. **R1 Retry** — deferred members re-run select/verify against the completed pool.
   Still unresolved → `needs_review`.
2. **R2 Consolidate** — duplicate/overlapping provisional clusters merge via
   **deterministic sequential agglomeration**, not blind union-find: provisional
   clusters are processed in deterministic order (membership size desc, then id); each
   proposed merge is a merge-verifier decision against the counterpart's *current*
   representation; an applied merge immediately refreshes the survivor's representation
   (union of aliases, refreshed exemplars), and the merged representation is
   **reverified** before it participates in further merges — if reverification fails,
   the merge routes to review and the survivor takes no further merges this round.
   Union-find records *committed* merges for edge normalization only; it never creates
   unverified transitive merges (A≈B and B≈C never implies A≈C without its own
   verified decision against the merged representation).
   Merge routing: verifier confidence ≥ `merge_accept_at` → apply;
   in `[merge_review_floor, merge_accept_at)` → review; below the floor → rejected and
   recorded, with **no** review work created.
3. **R3 Global comparative reassignment** — every member re-retrieves against the
   completed, consolidated pool, with its **incumbent parent always injected** into the
   candidate list. If selection returns the incumbent → keep (recorded, no further
   calls). If it returns a challenger → a **comparative verifier** call sees incumbent
   and challenger side by side and issues an explicit comparative verdict; the member
   switches only when the verdict prefers the challenger with confidence ≥
   `comparative_accept_at` — otherwise the incumbent is kept and the disagreement
   recorded. This margin requirement exists to prevent arbitrary churn from
   re-rolling near-ties. If selection returns null → the incumbent is re-verified by
   the assignment verifier; pass → keep, fail → `needs_review`. **Reassignment never
   mints** — minting happens only in streaming and R1.
4. **R4 Iterate** — repeat R2–R3 until a fixpoint (no merges, no switches) or
   `max_refine_iterations`. Additionally, an assignment-state hash (sorted member →
   parent pairs plus the live cluster set) is computed each iteration; if a prior hash
   repeats, refinement stops immediately and the level is flagged **unstable**. The
   engine persists the best state (fewest `needs_review` assignments; ties → earlier
   iteration) and records `stop_reason ∈ {converged, max_iterations, oscillation}`.
5. **R5 Finalize** — names and glosses are regenerated by the LLM from final
   membership, then checked by a **name verifier** against the final membership
   (confidence ≥ `name_verify_accept_at`); a failed verification keeps the prior
   name and routes the rename to review. Provisional → `final`; empty clusters retired
   with provenance; only finalized clusters feed the next level.

## Cardinality.MANY mechanics (defined now, built in milestone 2)

Multi-parent assignment uses **repeated masked selection**: after a parent is accepted,
the selector re-runs with all accepted parents' keys removed from the candidate list.
Each accepted parent is verified individually by the assignment verifier. Selection
stops when the selector returns null, `max_parents` is reached, or a verification
fails (the failed proposal is dropped and recorded; it does not create review work).
During consolidation, parent edges pointing at merged clusters are rewritten to the
surviving cluster and deduplicated — two edges collapsing onto one survivor become one
edge. Equivalence levels are always `Cardinality.ONE`.

## Persistence, resume, review

New ledger tables (additive `CREATE TABLE IF NOT EXISTS`, matching the existing
schema style): `cluster_runs`, `cluster_levels`, `pool_snapshots`, `clusters`,
`cluster_decisions`, `cluster_assignments` (append-only; superseded rows keep
`superseded_by`), `cluster_merges`, `cluster_reviews`. Every LLM interaction is
recorded with: order index, batch id and boundaries, pool snapshot id, model and
version, generation parameters, full rendered prompt, raw response, resolved outcome,
and confidence. Order and batch size are **evaluated sources of variance** (see
stability probes), not merely fingerprint fields.

Resume replays: order, batch boundaries, snapshot ids, and mints in the ledger
reconstruct pool state at any interruption point exactly; completed decisions are
skipped by decision key `(fingerprint, level, phase, iteration, subject_id,
subject_hash)`.

Review remains an immutable overlay with clustering verbs: *assign to X*, *confirm
mint*, *reject mint*, *approve/reject merge*, *rename*. Application is all-or-nothing
and snapshot-guarded against both the source concepts and the pool state. Reviewer
renames apply to minted clusters only.

Exports, regenerable from the ledger: `hierarchy.csv` (member → parent per level),
`clusters.csv` (id, name, gloss, provenance, status, level), per-level review CSVs.

## Diagnostics and evaluation

Token-free, from the ledger:

- singleton rate; cluster-size distribution;
- **late-parent rate**: final parent did not yet exist when the member streamed;
- **streaming retrieval recall**: final parent existed at streaming time and appeared
  in that member's streaming top-k;
- **completed-pool retrieval recall**: final parent appears in top-k during global
  reassignment — separating ordering effects from actual retrieval failures;
- duplicate-mint rate (minted clusters later merged ÷ minted);
- reassignment rate per refinement iteration (convergence curve); stop reasons;
- review/defer rates per phase.

LLM-judged, sampled deterministically and explicitly **heuristic** — especially when
the same backend built and judged the clusters: cohesion via intruder tests, taxonomy
consistency via sampled is-a checks. A distinct evaluator model is supported and
recommended; human/gold evaluation takes precedence where available.

Gold-based, when an adjudicated set exists: for equivalence levels, **B-cubed
precision/recall/F1 alongside pairwise metrics**; for hierarchy levels,
ancestor-based or path-based accuracy.

Stability probes: re-run a deterministic sample under N permuted orders and ≥2 batch
sizes; report pairwise assignment agreement, attributed to the level.

## Invariants

1. A malformed model answer never resolves to a real cluster (opaque keys, exact
   lookup — carried over unchanged).
2. No automatic state change without a passed verifier call at its task-specific
   threshold; everything murky routes to review or defer, in that documented order.
3. Taxonomy nodes are immutable to engine and reviewers alike.
4. Minting is reachable only through retrieval exhaustion, and mint provenance is
   recorded.
5. No unverified transitive merges.
6. Cluster identity is independent of cluster name.
7. Reassignment requires a comparative verdict at margin; it never mints.
8. Refinement always terminates with a recorded stop reason and a persisted best/last
   stable state.
9. The ledger reconstructs everything: pool state, decision provenance, exports.

## Milestone scoping

**Milestone 1 (reference implementation, correctness first):** equivalence clustering;
single-parent roll-up; deterministic `batch_size=1` reference mode; BM25 plus one
generic dense head through the existing `Retriever` protocol; retry, consolidation,
comparative reassignment, oscillation handling; ledger/resume; review overlay;
structural diagnostics.

**Milestone 2 (after M1 passes its correctness suite):** optimistic batching with
frozen pool snapshots; `Cardinality.MANY` DAG levels; LLM-judged and stability
diagnostics; calibration tooling; incremental performance work.

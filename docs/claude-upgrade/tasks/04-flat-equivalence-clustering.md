# Task 04: Flat equivalence clustering

Target envelope: $55. Dependencies: task 00 clustering contract and stable persistence/accounting from tasks 01-03.

## Outcome

Ship one end-to-end canonicalization workflow with explicit uncertainty, evidence, bounded execution, and resumability.

## Before implementing

Read the July clustering specification and the relevant implementation proposal. Resolve their conflicts in a short decision record. The first implementation is flat equivalence clustering. Broader/narrower relations, fixed target cluster counts, multiple hierarchy levels, and taxonomy mutation belong to a later milestone.

Do not implement clustering by calling the fixed-target Matcher on the source collection as both inputs. A changing cluster pool requires its own state and identity.

## First vertical slice

1. Reuse Record, Candidate, Usage, retrieval/LLM protocols, opaque candidate keys, and FakeLLM where suitable.
2. Define stable run-local cluster IDs, source identity and order, member ownership, canonical representation, and explicit assigned/review/deferred/failed outcomes.
3. Retrieve candidate clusters and adjudicate under bounded context and call limits. Distinguish candidates retrieved from those actually shown.
4. Consider a new cluster only after the configured retrieval expansion or exhaustion policy. A missed neighbor is not proof of novelty.
5. Preserve incumbent assignments during refinement candidate selection. Define a conservative explicit merge policy and handle contradictory evidence.
6. Persist enough versioned evidence to reconstruct the cluster state used for a decision. Snapshot IDs without reconstructable representations are insufficient.
7. Support bounded refinement and an explicit stopping criterion. If selecting a best stable revision after oscillation, export that revision rather than the last transient state.
8. Add CLI and Python access through the shared operations layer. Export members, clusters, unresolved records, and decision evidence.

## Acceptance

- Every source is accounted for exactly once in an accepted assignment or an explicit unresolved state.
- Accepted membership is a partition. Unresolved singletons are not described as verified equivalences.
- Synonyms co-cluster, related-but-distinct concepts remain separate, and no-match/novel records have a defined outcome.
- A-B and B-C evidence cannot silently force A-C equivalence contrary to the chosen policy.
- Fixed inputs, model responses, and processing order yield deterministic IDs and results.
- Interruption and resume reproduce a clean scripted run at representative transaction boundaries.
- Historical state is reconstructable and selected-revision exports are correct.
- Candidate and upstream call bounds hold under refinement.
- Large-pool handling does not assume a full index rebuild on every new cluster is acceptable. Measure the smallest relevant scaling check.
- A small labelled real sample is evaluated separately from implementation tests when an endpoint is available.

## Scope control

Finish the tested vertical slice before extending it. If it cannot pass the core gates within available credit, keep it experimental or on a separate branch and protect the stable matching release. Do not trade away error handling, provenance, or final review to claim a complete hierarchy.


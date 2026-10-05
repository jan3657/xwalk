# Historical design plans (superseded)

The files in `plans/` and `specs/` are the July 2026 design documents that produced
xwalk 0.1. They are kept as a record, **not as instructions**: about 19,000 lines, written
before 0.2, and partly contradicted by the current code and by
[CONTRACTS.md](../claude-upgrade/CONTRACTS.md). Do not load them to learn how xwalk works
today; read [architecture](../architecture.md), [concepts](../concepts.md) and the
reference pages instead. Where a plan and CONTRACTS.md disagree, CONTRACTS.md wins.

| Document | What it planned | Status today |
|---|---|---|
| `specs/2026-07-26-xwalk-generalized-matching-library-design.md` (724 lines) | The library design: generalising the OntoRAG loop (retrieve, fuse, select, gate, rewrite) to any two collections; templates and slots instead of dataset hooks; ledger, resume, review overlay, evaluation. | Implemented in 0.1. The core ideas (opaque keys, failure toward review, ledger as truth, light base install) are the invariants in [architecture](../architecture.md). Error, resume and export details were redefined in 0.2 (CONTRACTS §2-5). |
| `plans/2026-07-26-phase-1-mapping-platform.md` (9,658 lines) | Task-by-task code for the core types, fingerprints, templates, sources, BM25, fusion, LLM clients, keying, prompts, stages, matcher, ledger and batch. | Done in 0.1. Much of the code shown was later changed (0.2 tasks 01-03); the plan's code is not the current code. |
| `plans/2026-07-26-phase-2-knowing-whether-it-works.md` (3,335 lines) | Gold labels, metrics, the three-way failure decomposition, partitioning, prompt drafting and the three-partition optimiser. | Done in 0.1. Its rule "every spending command prints an estimated cost first" holds only for `prompts optimize` (an estimated call count); `match` instead takes a hard `--max-calls` limit, and xwalk never prints money. |
| `plans/2026-07-26-phase-3-breadth.md` (3,818 lines) | Optional extras, dense retrieval, OWL/OBO/SQL sources, LiteLLM, ablation, comparison, the job file, the CLI, the worked examples, CI. | Done in 0.1. Its CLI built every index on every command; since 0.2 `match` and `index` open a compatible index and refuse an incompatible one (`ablate` and `prompts optimize` still build their own). The CLI now goes through `xwalk.ops`, with strict job validation and the 0.2 exit codes. |
| `plans/2026-07-28-phase-4-release.md` (146 lines) | Licence, changelog, single-source version, tag-triggered Trusted Publishing release. | Done, except the upload: the v0.1.1 publish failed with `invalid-publisher` and xwalk is not on PyPI. See [releasing](../releasing.md). |
| `specs/2026-07-28-hierarchical-clustering-design.md` (252 lines) | Hierarchical LLM-adjudicated clustering: equivalence canonicalisation plus subsumption roll-up into taxonomy levels, multi-parent support. | **Superseded** by CONTRACTS §10: 0.2 ships only flat equivalence clustering, experimental. No hierarchy, taxonomy seeding or multi-parent. |
| `plans/2026-07-28-hierarchical-clustering-m1.md` (1,301 lines) | Milestone 1 implementation of that spec. | **Not executed as written.** Task 04 built a different, flat engine; the reconciled decisions are in [CLUSTERING_DECISIONS.md](../claude-upgrade/CLUSTERING_DECISIONS.md). |

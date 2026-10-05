# Task 02: Persistence, resume, and concurrent lifecycle

Target envelope: approximately $20 of the combined $45 correctness/persistence allocation. Dependency: task 01 contracts integrated.

## Outcome

A run can fail, stop, resume, and export a current mapping without stale records or background writes after cleanup.

## Primary ownership

ledger.py, batch.py, serialization and run identity, related batch/ledger tests. Coordinate CLI error reporting with task 03.

## Work

1. Implement the agreed source snapshot/current-result view while preserving historical attempts and versions. A changed source with the same ID must have one current exported result. Decide what happens when a source is removed, a limit is used, or a prior version failed.
2. Make run reports and duplicate-target reports use the intended current view. Provide a distinct history view when needed.
3. Replace unsafe cancellation/cleanup behavior. On fatal failure or cancellation, stop scheduling, cancel or drain sibling tasks according to policy, await their completion, then close resources.
4. Make transaction boundaries explicit. A committed result and its resumability metadata must agree after interruption.
5. Include all semantically relevant source, target, configuration, and encoder identity in resume decisions. Treat sensitive credentials as secrets, not serialized identity fields.
6. Distinguish complete, partial, failed, and interrupted runs. Task 03 will expose these consistently in CLI and JSON.

## Acceptance

- Change source s1 and resume: exactly one current s1 row, with history still inspectable.
- Remove a source and resume: outputs follow the documented snapshot contract.
- Interrupt at representative transaction boundaries, resume, and compare the completed current mapping with a clean run.
- One task fails while another is active: no write occurs after ledger closure and no orphan task continues silently.
- No successful exit or complete-run flag conceals unresolved execution failures.
- Existing review overlays retain their identity and provenance across the supported resume cases.
- A fixture using the v0.1.1 ledger format remains readable, including results and review overlays. If a migration is needed, test it on a preserved copy and make schema/version handling explicit. Define a clear refusal path for unsupported formats.

Keep persistence migration explicit and tested if the schema changes. Preserve existing user ledgers where feasible. Do not silently overwrite an incompatible ledger or migrate the only copy destructively.

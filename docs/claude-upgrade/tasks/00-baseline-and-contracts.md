# Task 00: Baseline and contracts

Target envelope: $15. Dependency: none. This is the calibration session.

## Outcome

Establish what works on the current branch, confirm which audit findings still apply, and agree the smallest implementable release contracts. Do not begin a broad redesign.

## Work

1. Read the repository's active instructions, pyproject.toml, CI, relevant public types, matcher, batch, ledger, configuration, and CLI. Compare current HEAD with the audited commit. Read the clustering specification and only the relevant portions of its historical implementation proposal.
2. Prepare the normal development environment using declared dependencies. Run the applicable existing test, lint, type, and packaging baseline. Record actual command output and any unavailable dependencies. Do not claim a full green baseline when only a subset ran.
3. Read audit/README.md. Run the diagnostic if the current source is still compatible. Reproduce the important findings through ordinary package-level tests where feasible. Label each finding confirmed, already fixed, changed, or not reproduced. Keep the AST diagnostic separate from normal tests.
4. Write a short contract note covering runtime state versus traces, current source snapshot versus historical results, recoverable versus fatal provider errors, cancellation, complete usage accounting, generation defaults, semantic index identity, and common operation outputs. Specify defaults for an incompatible existing index, a colliding output directory/run ID, and repeated invocation of the same operation. Prefer explicit rebuild/resume choices to silent replacement.
5. Decide the flat clustering relation and minimum supported outcomes. Reconcile conflicts in the historical design instead of executing the old implementation plan verbatim.
6. Check current PyPI availability and the actual release workflow. A recorded failed publishing job is not evidence that no package is available now.
7. Select a small frozen real-data pilot and identify whether it can run on an existing endpoint. Do not consume paid inference merely to establish the offline baseline.
8. Propose no more than three additional features. For each, state the user problem, existing code reused, acceptance check, and maintenance cost. Rank them against the existing scope.

## Acceptance

- A baseline report identifies the exact commit, commands, outcomes, and any differences from the original audit.
- Each high-priority issue has an explicit follow-up and a proposed behavioral test.
- Contracts are short enough for subsequent sessions to read without ingesting historical plans.
- File ownership and task order are clear. Two agents must not independently redesign the same matcher or ledger.
- STATUS.md records actual balance if supplied by the user, otherwise marks it unreported. Never invent a balance.

## Stop condition

Return the baseline and decisions. Only make a small prerequisite fix if needed to obtain the baseline, and report it separately. The result should make task 01 concrete.

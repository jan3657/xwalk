# xwalk Claude cloud handoff

Prepared for Jan on 5 October 2026.

This package turns the repository audit and competitor research into bounded implementation tasks. The target is a coherent xwalk 0.2 release. The budgets are allocation preferences, not guaranteed costs or automatically enforced Claude limits.

## First use

1. Claim the bonus if it is not already active. Use direct Claude Code cloud sessions. This promotion excludes Projects and Routines.
2. Put this directory in your checkout as `docs/claude-upgrade/`. Keep your existing work. Read `CLAUDE_GUIDANCE.md` and merge useful guidance into a root `CLAUDE.md` only if it does not conflict with existing instructions.
3. Give Claude's GitHub integration access to xwalk. Commit and push the task documents on a planning branch. The usual GitHub cloud workflow sees pushed repository state.
4. Start task 00 with the command below.
5. Record the actual promotional balance before and after the session in `STATUS.md`. Read the output and adjust scope before starting the next task.

```bash
claude --cloud "Read START_HERE.md, PLAN.md, and CLAUDE_GUIDANCE.md under docs/claude-upgrade/. Execute only docs/claude-upgrade/tasks/00-baseline-and-contracts.md. Use the actual current repository as the source of truth. Return a bounded implementation plan and the requested baseline report."
```

The audited main commit was `f737099f23ce0c6720963759e13e95d21d8bdf89`. If your branch has changed, reconcile the findings with current code. Do not overwrite later fixes.

## Sequence

- Task 00 establishes the baseline and contracts.
- Tasks 01 and 02 repair core behavior. Execute them sequentially unless their file ownership is explicitly separated.
- Task 03 exposes a small shared operations layer and a useful CLI.
- Task 04 adds the first flat equivalence-clustering workflow.
- Task 05 establishes quality and performance evidence. Its fixtures and runners can be prepared in parallel with task 04 after interfaces are fixed.
- Task 06 is an optional MCP adapter. Start it only when the shared operations are stable.
- Task 07 produces the README, documentation, and release candidate.
- Task 08 reviews the integrated release independently.

At most two direct cloud sessions should be working at once. A second session should own separate files or review a finished branch. Avoid two sessions editing the matching engine or ledger concurrently.

Start each dependent session from a pushed branch containing the accepted prerequisite commits. In the CLI, switch to that branch before using --cloud. On the web, select that repository and branch. A new session on the old main branch will not automatically contain fixes from an unmerged session. Have task 00 record the intended integration branch, then keep it current through reviewed merges or cherry-picks.

## Stop and consolidate

Protect the last $55 for documentation, integrated review, and contingency. When that balance is reached, stop starting optional features. Reallocate the MCP envelope first if correctness or clustering needs more work.

Clustering may remain an explicitly experimental feature or separate branch if it cannot pass its gates. Preserve a working matching release.

## Two different costs

The promotional balance pays for Claude's coding work in the cloud. Running xwalk against a paid LLM endpoint has separate application inference charges. Use FakeLLM and deterministic encoders for implementation tests. Real quality evaluation should use an explicitly configured model endpoint and a separately tracked budget, or Jan's cluster.

## Deliverable from every session

Return the commit or branch, changed behavior, commands actually run, test outcomes, unresolved limitations, and proposed next task. Update STATUS.md. Distinguish implementation tests, synthetic benchmarks, and real model evaluation. Never invent any of them.

## Sources

- [Official bonus terms](https://support.claude.com/en/articles/17152539-cloud-sessions-bonus-credit-promotion)
- [Cloud session reference](https://code.claude.com/docs/en/claude-code-on-the-web)

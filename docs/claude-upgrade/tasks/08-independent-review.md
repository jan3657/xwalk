# Task 08: Independent integrated review

Protected target envelope: $20, with a separate $15 contingency available. Dependency: release candidate from task 07.

## Outcome

Assess the integrated release against its stated contracts, fix concrete blockers, and produce a candid readiness report.

Use a fresh session that did not write the main implementation where practical.

## Review focus

1. Verify the initial correctness regressions through ordinary tests: trace invariance, confidence validation, multiline evidence, stage errors, and complete usage.
2. Exercise changed-source resume, current and historical exports, review overlays, interruption, cancellation, and one failure while other work is active.
3. Check strict configuration, actual generation setting precedence, index identity, and reuse.
4. Check call-bound enforcement under concurrency, retries, and rewrites. Distinguish estimates from actual provider usage.
5. Validate CLI output and exit codes, the short Python example, and MCP protocol behavior if shipped.
6. Challenge clustering with synonyms, related-but-distinct concepts, contradictory chains, unseen concepts, order changes, and interrupted refinement.
7. Verify clean-wheel installation and README commands. Inspect packaging/release evidence for the exact candidate.
8. Read benchmark claims against their raw results. No fixture-only test should be presented as semantic quality evidence.

## Work discipline

Run broad gates once on the integrated candidate, then repeat only tests needed to resolve concrete remaining risk. Do not add a new feature during review. Use contingency for identified blockers. Record any check that cannot run.

## Acceptance

The final report lists:
- Candidate commit and proposed release scope.
- Commands actually run and results.
- Confirmed fixes and any compatibility changes.
- Real benchmark evidence versus pending evaluation.
- Remaining issues, severity, and whether they block release.
- Supported versus experimental features.
- Actual recorded promotional spending if available.
- Exact next user action for merge or publication.

A useful outcome can be a ready matching release with clustering or MCP explicitly deferred. Do not conceal a failed gate to make the roadmap appear complete.


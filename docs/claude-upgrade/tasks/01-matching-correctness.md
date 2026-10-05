# Task 01: Matching correctness and accounting

Target envelope: approximately $25 of the combined $45 correctness/persistence allocation. Dependency: task 00.

## Outcome

Matching decisions, stage failures, and reported usage behave consistently.

## Primary ownership

matcher.py, stages/gate.py, stages/keying.py, llm/parsing.py, closely related stage interfaces, and their tests. Coordinate public type changes with task 02.

## Work and behavioral checks

1. Separate live candidate state from optional serialized trace data. The same scripted inputs must produce the same decisions, attempts, and upstream calls with trace verbosity on or off.
2. Reject non-finite confidence values, including NaN and positive/negative infinity. Define handling of missing, malformed, and out-of-range finite values. Invalid model output must not become an accepted certainty.
3. Keep candidate blocks structured until rendering. A chosen record with multiline text and blank lines must retain its evidence in the chosen block. Other-candidate text must not absorb it.
4. Make all stage boundaries follow the agreed provider error policy. Cover selector, scorer, verifier, and rewriter failures. Distinguish a source-level recoverable failure from a fatal configuration/provider failure that should stop the run.
5. Account for rewrites and every stage invocation. Keep provider-reported usage separate from estimates. Unknown usage after an interrupted call remains unknown. If transport retry usage is not observable, expose that limitation.
6. Verify generation parameter precedence against the actual outgoing request. A configured temperature or token setting must not silently change only a fingerprint. Keep documented per-stage overrides explicit.

## Acceptance

- Normal package tests reproduce the audit cases and pass after the fixes.
- Trace settings do not change matching results.
- Invalid confidence never passes an acceptance gate.
- Each provider failure type has a documented and tested outcome.
- Usage totals equal the sum of all observable calls in a scripted multi-attempt run.
- The candidate evidence regression is tested with realistic multiline text.
- Relevant existing tests pass. Report any expected compatibility change.

Do not rewrite every stage abstraction to solve these issues. Prefer a small common execution/accounting helper when it removes actual duplication.


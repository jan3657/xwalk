# Task 03: Shared operations and a useful CLI

Target envelope: $30. Dependencies: tasks 01 and 02.

## Outcome

Python users, shell users, and later MCP clients can run and inspect xwalk through one small, coherent operations layer.

## Work

1. Identify existing reusable functions before adding an operations module. Keep business behavior outside CLI formatting and MCP transport code.
2. Provide a short high-level Python route for a common matching job while preserving the low-level research API.
3. Make job validation strict. Unknown fields, invalid selector values, duplicate IDs, missing required fields, incompatible resume state, and unavailable optional dependencies should produce specific errors.
4. Implement an offline preflight command. Check credentials only for presence and configuration, never print their values. Do not make paid calls by default.
5. Extend the existing CLI with the highest-value missing operations: example/job initialization, validation, run inspection, one-record explanation, and explicit raw/reviewed export. Reuse existing commands and output formats where possible.
6. Define one versioned machine-readable output contract. Keep data on stdout and progress/logs on stderr. Preserve or explicitly migrate exit codes. Support noninteractive execution.
7. Add per-run call limits to the common execution boundary if not already present. Reserve budget before concurrent dispatch. Include retries, rewrites, and other observable upstream calls. Explain how aborted or unknown-use calls are counted.
8. Separate index preparation from opening a compatible persisted index. Avoid reading the same target collection twice. Include prefixes, normalization, model revision where available, and other semantic encoder settings in validation.

## Acceptance

- A common Python example fits approximately 10-15 readable lines and runs against the included fixture.
- Misspelled configuration fails preflight instead of silently using a default.
- Preflight performs zero paid inference calls.
- CLI JSON parses without log contamination on success and failure.
- Exit codes distinguish execution failure from successful completion and documented review-required outcomes.
- Inspect and explain identify the relevant source, decision, attempts, and run version.
- Reviewed export visibly applies the review overlay while preserving the original decision record.
- An unchanged compatible index is opened without invoking the encoder again.
- Index mismatch, output-directory collisions, and repeated invocation follow documented behavior. A different run must not overwrite prior results implicitly.
- A bounded concurrent fixture cannot dispatch more than the specified call allowance.

Keep argparse unless a demonstrated requirement justifies changing it. If scope must shrink, prioritize strict validation, JSON output, inspection, and reviewed export. Initialization can copy a bundled example. Do not spend this envelope building a TUI or web application.

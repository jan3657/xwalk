# Task 06: Optional MCP adapter

Target envelope: $25. Dependency: stable shared operations from task 03. Optional. Reallocate this envelope first if core fixes or clustering overrun.

## Outcome

An agent can invoke bounded xwalk operations through a thin stdio MCP server.

## Work

1. Check the official Python MCP SDK documentation and supported version at implementation time. Use tested dependency bounds and an optional xwalk[mcp] extra.
2. Reuse the same validation, execution, inspection, and export functions as the CLI.
3. Start with approximately six tools: validate_job, search_candidates, bounded match_records, get_run, list_results, and explain_result. Reconcile names with the shared contract.
4. Give every tool a clear input schema and structured output schema. Support pagination and bounded payloads where needed.
5. Keep stdout exclusively for protocol traffic. Send logs to stderr.
6. Map core validation, execution, and partial-result states to clear tool responses.
7. Leave large background runs in the CLI unless durable job execution already exists. An in-memory task is not a durable job. Do not invent a scheduler just to return a job ID.
8. Document the exact launch and client configuration with a small offline example.

## Acceptance

- A protocol client can initialize, list tools, validate a job, run a bounded fixture, and inspect its result.
- Structured output conforms to the declared schema.
- Invalid input and missing optional dependencies produce useful errors.
- The process emits no non-protocol text on stdout.
- Paging and call limits work, and one tool cannot accidentally return an entire unbounded ledger.
- CLI and MCP return consistent results for the same operation.
- Base installation works without the MCP dependency.

Keep transport code small. Do not add hosted accounts, a public API service, or multi-user deployment in this milestone.


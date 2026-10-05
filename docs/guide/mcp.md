# Driving xwalk from an agent (MCP)

`xwalk mcp` is a small [Model Context Protocol](https://modelcontextprotocol.io) server
over stdio. An agent host (Claude Code, Claude Desktop, any MCP client) launches it as a
subprocess and gets six tools that call the same operations as the CLI, return the same
JSON envelope as `--json`, and are bounded so one call can neither run up an open-ended
bill nor dump a whole ledger into the agent's context.

It is a thin adapter, not a service: no accounts, no HTTP endpoint, no background jobs. A
tool call returns when its work is done. Large runs belong in the CLI (`xwalk match`);
the agent can then inspect them with the read tools.

## Install and launch

```
pip install 'xwalk[mcp]'      # the official MCP Python SDK, mcp>=2.3,<3
xwalk mcp                      # serves on stdin/stdout until the client disconnects
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-calls-cap` | `500` | the largest `max_calls` a `match_records` call may ask for |
| `--offline-model` | off | answer every model call with a fixed scripted reply instead of the job's endpoint; for demos and tests, the results are meaningless |

Without the extra, `xwalk mcp` exits `2` and prints `pip install 'xwalk[mcp]'` on stderr.
The base install never imports `mcp`.

stdout carries protocol messages only. While serving, the SDK's stdio transport points
file descriptor 1 at stderr, so a stray print from any library cannot corrupt the
stream; logs go to stderr. Relative paths in tool arguments resolve against the server's
working directory, so launch it in your project directory (or pass absolute paths). The
server acts with your file permissions: it reads the job's files and writes run and index
directories where the agent asks.

## Client configuration

Every host takes the same three facts: the command, its arguments, and (where supported)
the working directory.

Claude Code, from your project directory:

```
claude mcp add xwalk -- xwalk mcp
```

or a project `.mcp.json`:

```json
{
  "mcpServers": {
    "xwalk": {"command": "xwalk", "args": ["mcp"]}
  }
}
```

Claude Desktop (`claude_desktop_config.json`), which does not set a working directory, so
use absolute paths in tool arguments and the full path of the `xwalk` executable of the
environment where you installed `xwalk[mcp]`:

```json
{
  "mcpServers": {
    "xwalk": {"command": "/path/to/venv/bin/xwalk", "args": ["mcp", "--max-calls-cap", "200"]}
  }
}
```

The test suite drives the server with the MCP Python SDK's own client, both in process
and as a `python -m xwalk.cli mcp` subprocess over stdio (`tests/test_mcp.py`). The host
configurations above follow those hosts' documented stdio format; they are not part of
the automated tests.

## A small offline example

No endpoint, no key, no cost: the bundled quickstart job with the scripted model.

```
xwalk init demo && cd demo
```

```python
import asyncio

from mcp import Client
from mcp.client.stdio import StdioServerParameters


async def main() -> None:
    server = StdioServerParameters(command="xwalk", args=["mcp", "--offline-model"])
    async with Client(server) as client:
        print([tool.name for tool in (await client.list_tools()).tools])
        checked = await client.call_tool(
            "validate_job", {"job": "job.yaml", "check_credentials": False}
        )
        print(checked.structured_content["status"])  # ok
        ran = await client.call_tool(
            "match_records", {"job": "job.yaml", "out": "run", "max_calls": 40, "limit": 10}
        )
        print(ran.structured_content["run"], ran.structured_content["usage"])
        page = await client.call_tool("list_results", {"run": "run", "limit": 2})
        print(page.structured_content["data"]["rows"], page.structured_content["data"]["next_offset"])
        why = await client.call_tool("explain_result", {"run": "run", "source_id": "s2"})
        print(why.structured_content["data"]["decision"])


asyncio.run(main())
```

Run it from `demo/`. Drop `--offline-model` to use the endpoint the job names; its
credential variable must then be set in the server's environment.

## Tools

Every tool returns the [`--json` envelope](../reference/cli.md#machine-readable-output---json)
(schema 1) as structured content, with a declared output schema, and the same object as
JSON text for clients that read only text.

| Tool | Operation | Bounds | Notes |
|---|---|---|---|
| `validate_job(job, check_credentials=true, scan_records=true)` | `ops.validate` | — | offline, never calls a model; accepts clustering jobs too |
| `search_candidates(job, query, index_dir, limit=10)` | `ops.search` | `limit` 1-100 | no model call; builds the indexes in `index_dir` when absent, refuses an incompatible one |
| `match_records(job, out, max_calls, limit=20)` | `ops.run_async` | `max_calls` required, 1 to the server cap; `limit` 1-100 records | resumes the run in `out`; enforced by the same `CallBudget` as `xwalk match --max-calls` |
| `get_run(run)` | `ops.inspect` | — | `status`/`exit_code` are what the run itself reports |
| `list_results(run, offset=0, limit=50, status=null)` | `ops.list_results` | `limit` 1-200 rows | continue with `offset = data.next_offset` until it is `null` |
| `explain_result(run, source_id)` | `ops.explain` | one source, attempts bounded by `policy.max_attempts` | the full stored result (`--full`) is not exposed |

Clustering is not exposed: it is experimental and its runs have no `inspect`/`explain`
yet. Use `xwalk cluster`.

### Reading results and errors

- A performed operation is a normal result; read `status` (`ok`, `attention`, `error`,
  `interrupted`), `exit_code` and `errors`, exactly as for the CLI. An invalid job from
  `validate_job` is `status: "error"` with one entry per problem. A `match_records` call
  that reaches `max_calls` returns `status: "error"`, `run_state: "aborted"` and error
  `call_limit_reached`; a call that processed `limit` records with more left returns
  `status: "attention"`, `run_state: "partial"` and `pending` in `counts`. Call it again
  with the same arguments to continue.
- `isError: true` means the operation could not be performed: arguments outside the
  input schema (`limit: 1000`), a `max_calls` above the server cap, a path that is not a
  run directory (`run_not_found`), a run directory holding a different run
  (`run_fingerprint_mismatch`), a missing credential. The structured content is then the
  error envelope when the operation got far enough to produce one, and the text says
  what to fix.
- With `--offline-model`, every `match_records` result carries an `offline_model`
  warning.

The CLI exposes the same read operations: `xwalk search` and `xwalk results` (see the
[command line reference](../reference/cli.md)), so an agent's findings can be reproduced
in a shell.

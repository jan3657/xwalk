"""The MCP adapter (task 06), driven through a real MCP client.

In-process tests connect `mcp.Client` to the server object; the stdio tests launch
`python -m xwalk.cli mcp --offline-model` as a subprocess, exactly as an agent host
would. Every model answer comes from the scripted offline model: no endpoint, no cost.
Tests marked `mcp` skip without the extra; the missing-extra checks at the end run in
every environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from xwalk import ops
from xwalk.cli.main import main

TOOLS = {
    "validate_job",
    "search_candidates",
    "match_records",
    "get_run",
    "list_results",
    "explain_result",
}


def _mcp() -> Any:
    return pytest.importorskip("mcp", reason="needs xwalk[mcp]")


@pytest.fixture
def job(tmp_path: Path) -> Path:
    ops.init(tmp_path / "quickstart")
    return tmp_path / "quickstart" / "job.yaml"


@pytest.fixture
def server() -> Any:
    _mcp()
    from xwalk.mcp_server import build_server

    return build_server(offline=True, max_calls_cap=100)


async def _call(client: Any, tool: str, **arguments: Any) -> Any:
    return await client.call_tool(tool, arguments)


def _conforms(result: Any, tools: dict[str, Any], name: str) -> dict[str, Any]:
    """The structured content, checked against the tool's declared output schema."""
    import jsonschema

    content = result.structured_content
    jsonschema.validate(content, tools[name].output_schema)
    assert json.loads(result.content[0].text) == content  # text and structure agree
    return dict(content)


@pytest.mark.mcp
async def test_tools_declare_input_and_output_schemas(server):
    from mcp import Client

    async with Client(server) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(listed) == TOOLS
    for tool in listed.values():
        assert tool.description and tool.input_schema["type"] == "object"
        assert {"schema_version", "status", "exit_code", "errors", "data"} <= set(
            tool.output_schema["properties"]
        )
    match_input = listed["match_records"].input_schema
    assert "max_calls" in match_input["required"]
    assert match_input["properties"]["limit"]["maximum"] == 100
    assert listed["list_results"].input_schema["properties"]["limit"]["maximum"] == (
        ops.MAX_PAGE_SIZE
    )
    assert listed["get_run"].annotations.read_only_hint is True


@pytest.mark.mcp
async def test_validate_run_inspect_page_and_explain(server, job, tmp_path):
    from mcp import Client

    out = str(tmp_path / "run")
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        checked = await _call(client, "validate_job", job=str(job), check_credentials=False)
        assert not checked.is_error
        assert _conforms(checked, tools, "validate_job")["status"] == "ok"

        found = await _call(
            client, "search_candidates", job=str(job), query="dextrose", index_dir=f"{out}/index"
        )
        candidates = _conforms(found, tools, "search_candidates")["data"]["candidates"]
        assert candidates[0]["id"] == "CHEBI:17234"

        first = await _call(client, "match_records", job=str(job), out=out, max_calls=50, limit=3)
        envelope = _conforms(first, tools, "match_records")
        assert envelope["run"]["run_state"] == "partial" and envelope["status"] == "attention"
        assert envelope["counts"]["pending"] == 1
        assert envelope["usage"]["calls"] <= 50 and envelope["usage"]["limit"] == 50
        assert "offline_model" in [w["code"] for w in envelope["warnings"]]

        rest = await _call(client, "match_records", job=str(job), out=out, max_calls=50)
        assert _conforms(rest, tools, "match_records")["run"]["run_state"] == "complete"

        run = _conforms(await _call(client, "get_run", run=out), tools, "get_run")
        assert run["operation"] == "inspect" and run["counts"]["total"] == 4

        rows: list[dict[str, Any]] = []
        offset: int | None = 0
        while offset is not None:
            page = await _call(client, "list_results", run=out, offset=offset, limit=3)
            data = _conforms(page, tools, "list_results")["data"]
            assert len(data["rows"]) <= 3
            rows += data["rows"]
            offset = data["next_offset"]
        assert [row["source_id"] for row in rows] == ["s1", "s2", "s3", "s4"]

        why = await _call(client, "explain_result", run=out, source_id="s2")
        decision = _conforms(why, tools, "explain_result")["data"]["decision"]
        assert decision["matched_id"] == rows[1]["matched_id"]


@pytest.mark.mcp
async def test_the_call_limit_is_enforced_and_reported(server, job, tmp_path):
    from mcp import Client

    async with Client(server) as client:
        result = await _call(
            client, "match_records", job=str(job), out=str(tmp_path / "run"), max_calls=1
        )
    envelope = result.structured_content
    assert not result.is_error  # the run was performed; it stopped at its limit
    assert envelope["status"] == "error" and envelope["run"]["run_state"] == "aborted"
    assert envelope["errors"][0]["code"] == "call_limit_reached"
    assert envelope["usage"]["calls"] <= 1


@pytest.mark.mcp
async def test_invalid_input_gives_useful_tool_errors(server, job, tmp_path):
    from mcp import Client

    async with Client(server) as client:
        over_cap = await _call(
            client, "match_records", job=str(job), out=str(tmp_path / "r"), max_calls=101
        )
        too_many_rows = await _call(client, "list_results", run=str(tmp_path), limit=10_000)
        no_run = await _call(client, "get_run", run=str(tmp_path / "nowhere"))
        bad_job = await _call(client, "validate_job", job=str(tmp_path / "missing.yaml"))

    assert over_cap.is_error and "cap of 100" in over_cap.structured_content["errors"][0]["message"]
    assert not (tmp_path / "r").exists()
    assert too_many_rows.is_error and "less than or equal to 200" in too_many_rows.content[0].text
    assert no_run.is_error and no_run.structured_content["errors"][0]["code"] == "run_not_found"
    # validation was performed: an invalid job is the answer, not a tool failure
    assert not bad_job.is_error and bad_job.structured_content["status"] == "error"
    assert bad_job.structured_content["exit_code"] == 2


@pytest.mark.mcp
async def test_cli_and_mcp_return_the_same_envelopes(server, job, tmp_path, capsys):
    from mcp import Client

    out = str(tmp_path / "run")
    async with Client(server) as client:
        await _call(client, "match_records", job=str(job), out=out, max_calls=50, limit=3)
        via_mcp = {
            "inspect": (await _call(client, "get_run", run=out)).structured_content,
            "results": (await _call(client, "list_results", run=out, limit=2)).structured_content,
            "validate": (
                await _call(client, "validate_job", job=str(job), check_credentials=False)
            ).structured_content,
        }
    commands = {
        "inspect": ["inspect", "--run", out],
        "results": ["results", "--run", out, "--limit", "2"],
        "validate": ["validate", "--job", str(job), "--no-credentials"],
    }
    for name, argv in commands.items():
        capsys.readouterr()
        code = main([*argv, "--json"])
        via_cli = json.loads(capsys.readouterr().out)
        assert via_cli == via_mcp[name], name
        assert code == via_mcp[name]["exit_code"]


# --- over stdio, in a subprocess --------------------------------------------------------


def _server_command(*extra: str) -> list[str]:
    return [sys.executable, "-m", "xwalk.cli", "mcp", "--offline-model", *extra]


@pytest.mark.mcp
async def test_a_stdio_client_can_initialize_validate_run_and_inspect(job, tmp_path):
    _mcp()
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=_server_command()[1:], cwd=str(job.parent)
    )
    async with Client(params) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert names == TOOLS
        checked = await _call(client, "validate_job", job="job.yaml", check_credentials=False)
        assert checked.structured_content["status"] == "ok"
        ran = await _call(client, "match_records", job="job.yaml", out="run", max_calls=40)
        assert ran.structured_content["run"]["run_state"] == "complete"
        run = await _call(client, "get_run", run="run")
        assert run.structured_content["counts"]["total"] == 4
    assert (job.parent / "run" / "mapping.csv").is_file()


@pytest.mark.mcp
def test_stdout_carries_only_protocol_messages(job):
    mcp = _mcp()
    version = getattr(mcp.types, "LATEST_PROTOCOL_VERSION", "2025-06-18")
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": {"name": "raw-test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "match_records",
                "arguments": {"job": "job.yaml", "out": "run", "max_calls": 40},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "get_run", "arguments": {"run": "nowhere"}},
        },
    ]
    process = subprocess.Popen(
        _server_command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # logs; unread, so it must not fill a pipe
        text=True,
        cwd=job.parent,
    )
    assert process.stdin is not None and process.stdout is not None
    watchdog = threading.Timer(120, process.kill)  # a hung server fails, not hangs, the test
    watchdog.start()
    lines: list[str] = []
    try:
        for request in requests:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            # Wait for the answer before sending more: a client that closes stdin while
            # a call is in flight cancels it.
            while "id" in request:
                line = process.stdout.readline()
                assert line, "the server closed stdout early"
                lines.append(line)
                if json.loads(line).get("id") == request["id"]:
                    break
        process.stdin.close()
        lines += process.stdout.readlines()  # anything written after the last answer
        process.wait()
    finally:
        watchdog.cancel()
    lines = [line for line in lines if line.strip()]
    messages = [json.loads(line) for line in lines]  # every stdout line is JSON
    assert all(m.get("jsonrpc") == "2.0" for m in messages)
    by_id = {m["id"]: m for m in messages if "id" in m}
    assert set(by_id) == {1, 2, 3, 4}
    assert by_id[3]["result"]["structuredContent"]["run"]["run_state"] == "complete"
    assert by_id[4]["result"]["isError"] is True
    assert process.returncode == 0


# --- without the extra ------------------------------------------------------------------


@pytest.fixture
def without_mcp(monkeypatch):
    """Make `import mcp` fail as it does in a base install."""
    for name in list(sys.modules):
        if name == "mcp" or name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mcp", None)


def test_the_base_install_imports_xwalk_without_mcp(without_mcp):
    from xwalk._extras import MissingExtra
    from xwalk.mcp_server import build_server

    with pytest.raises(MissingExtra, match=r"pip install 'xwalk\[mcp\]'"):
        build_server()


def test_xwalk_mcp_without_the_extra_says_what_to_install(without_mcp, capsys):
    assert main(["mcp"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout belongs to the protocol, even for this error
    assert "xwalk[mcp]" in captured.err

"""A thin MCP server over `xwalk.ops` (stdio only). Needs ``pip install 'xwalk[mcp]'``.

    xwalk mcp                      # serve on stdin/stdout until the client disconnects
    xwalk mcp --offline-model      # scripted model answers: demos and tests, no endpoint

Every tool calls one operation and returns its `OpResult.envelope()` -- the same schema 1
object `--json` prints -- as structured content (and as JSON text for clients that only
read text). A tool result is marked `isError` only when the operation could not be
performed (it raised `OpError` or failed unexpectedly); a performed operation reports
its outcome in `status`/`exit_code`/`errors`, exactly as the CLI does, so an invalid job
from `validate_job` or a run aborted at its call limit is a normal result.

Bounds: `match_records` processes at most `MAX_MATCH_RECORDS` records per call and
requires `max_calls` (at most the server's `--max-calls-cap`), enforced by the same
`CallBudget` the CLI uses; `search_candidates` returns at most `ops.MAX_SEARCH_LIMIT`
candidates; `list_results` pages at most `ops.MAX_PAGE_SIZE` rows. Nothing returns a
whole ledger, and there is no background execution: a call returns when its work is
done, and a larger run continues with another call (it resumes) or in the CLI.

stdout carries protocol messages only: the SDK's stdio transport points file descriptor
1 at stderr while serving, so a stray print cannot corrupt the stream. Logs go to stderr.
"""

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

import anyio
from pydantic import BaseModel, Field

from xwalk import __version__, ops
from xwalk._extras import require
from xwalk.llm.base import LLMClient, LLMRequest
from xwalk.llm.fake import FakeLLM

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer

DEFAULT_MAX_CALLS_CAP = 500
MAX_MATCH_RECORDS = 100

_INSTRUCTIONS = """\
xwalk matches records from a source collection to a target collection described by a
job file (job.yaml). Typical flow: validate_job -> search_candidates (optional, no model
calls) -> match_records (bounded, writes a run directory; call again to continue a
partial run) -> get_run -> list_results (paged) -> explain_result for one source.
Every result is the xwalk JSON envelope: read `status` (ok, attention, error,
interrupted), `exit_code`, `errors` and `data`. Paths are resolved against the server's
working directory."""


# --- output schemas (the `--json` envelope, schema 1) ----------------------------------


class Message(BaseModel):
    code: str
    message: str
    source_id: str | None = None


class RunRef(BaseModel):
    dir: str
    run_fingerprint: str
    run_state: str


class Envelope(BaseModel):
    """The `--json` result object, schema 1 (docs/reference/cli.md)."""

    schema_version: Literal[1]
    operation: str
    status: Literal["ok", "attention", "error", "interrupted"]
    exit_code: int
    run: RunRef | None
    counts: dict[str, int]
    usage: dict[str, Any] | None
    artifacts: dict[str, str]
    warnings: list[Message]
    errors: list[Message]
    data: dict[str, Any]


class SearchCandidate(BaseModel):
    rank: int
    id: str
    fused_score: float
    retrievers: dict[str, int] = Field(description="retriever name -> 1-based rank")
    text: str = Field(description="the candidate as the job's candidate template renders it")


class SearchData(BaseModel):
    query: str
    limit: int
    candidates: list[SearchCandidate]
    indexes: dict[str, str] = Field(description="index name -> opened | built")


class SearchEnvelope(Envelope):
    data: SearchData  # type: ignore[assignment]


class ResultRow(BaseModel):
    source_id: str
    status: str
    reason: str | None
    matched_id: str | None
    confidence: float | None
    revision: int | None
    result_key: str | None


class ResultsPage(BaseModel):
    rows: list[ResultRow]
    offset: int
    limit: int
    status: str | None
    total: int = Field(description="rows matching the status filter")
    next_offset: int | None = Field(description="offset of the next page; null on the last")


class ResultsEnvelope(Envelope):
    data: ResultsPage  # type: ignore[assignment]


# --- the offline model -----------------------------------------------------------------


def offline_model(request: LLMRequest) -> str:
    """One fixed answer every stage understands: pick the first candidate, score 0.9.

    The same scripted reply as `examples/quickstart.py`. It exercises the whole pipeline
    without an endpoint; its "decisions" say nothing about the data.
    """
    return json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.9, "decision": "support", "queries": []}
    )


# --- the server ------------------------------------------------------------------------


def build_server(
    *,
    max_calls_cap: int | None = None,
    offline: bool = False,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> "MCPServer[Any]":
    """The xwalk MCP server with its six tools. Raises `MissingExtra` without `mcp`.

    `max_calls_cap` bounds the `max_calls` a `match_records` call may request.
    `offline=True` gives `match_records` the scripted `offline_model` instead of the
    job's endpoint. `llm_factory` (Python only, for tests) supplies the client directly.
    """
    require("mcp", "mcp", purpose="the xwalk MCP server")
    from mcp.server.mcpserver import MCPServer
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    cap = DEFAULT_MAX_CALLS_CAP if max_calls_cap is None else max_calls_cap
    if cap < 1:
        raise ValueError(f"--max-calls-cap must be >= 1, got {cap}")
    make_llm = llm_factory
    if offline and make_llm is None:

        def make_llm() -> LLMClient:
            return FakeLLM(handler=offline_model, model="xwalk-offline")

    server: MCPServer[Any] = MCPServer(
        "xwalk",
        version=__version__,
        instructions=_INSTRUCTIONS,
        log_level="WARNING",
    )

    def respond(result: ops.OpResult, *, is_error: bool = False) -> CallToolResult:
        envelope = result.envelope()
        text = json.dumps(envelope, ensure_ascii=False, default=str)
        # Round-trip through JSON so the structured content is exactly what the text says.
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content=json.loads(text),
            is_error=is_error,
        )

    async def perform(
        operation: str, work: Callable[[], Awaitable[ops.OpResult]]
    ) -> CallToolResult:
        try:
            result = await work()
        except Exception as exc:  # reported to the agent, never a protocol error
            return respond(ops.failure_result(operation, exc), is_error=True)
        return respond(result)

    def in_thread(fn: Callable[[], ops.OpResult]) -> Callable[[], Awaitable[ops.OpResult]]:
        async def work() -> ops.OpResult:
            return await anyio.to_thread.run_sync(fn)

        return work

    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    async def validate_job(
        job: Annotated[str, Field(description="path to a job.yaml (match or cluster job)")],
        check_credentials: Annotated[
            bool, Field(description="require the llm.api_key_env variable to be set")
        ] = True,
        scan_records: Annotated[
            bool, Field(description="read both collections (duplicate ids, empty texts)")
        ] = True,
    ) -> Annotated[CallToolResult, Envelope]:
        """Strict offline preflight of a job file. Never calls a model.

        An invalid job is a normal result with status "error" and one entry in `errors`
        per problem."""
        return await perform(
            "validate",
            in_thread(
                lambda: ops.validate(
                    job, check_credentials=check_credentials, scan_records=scan_records
                )
            ),
        )

    async def search_candidates(
        job: Annotated[str, Field(description="path to a job.yaml")],
        query: Annotated[str, Field(min_length=1, description="free text to retrieve for")],
        index_dir: Annotated[
            str, Field(description="index directory; built when absent (e.g. <run dir>/index)")
        ],
        limit: Annotated[int, Field(ge=1, le=ops.MAX_SEARCH_LIMIT)] = 10,
    ) -> Annotated[CallToolResult, SearchEnvelope]:
        """Retrieve the fused target candidates for a query, as the matcher would.

        No model calls. Builds the job's indexes in `index_dir` if they are absent and
        refuses an incompatible one."""
        return await perform(
            "search",
            lambda: ops.search_async(job, query, index_dir=index_dir, limit=limit),
        )

    async def match_records(
        job: Annotated[str, Field(description="path to a job.yaml")],
        out: Annotated[
            str, Field(description="run directory: new, or the same run to continue it")
        ],
        max_calls: Annotated[
            int,
            Field(
                ge=1,
                description="upper bound on upstream model requests in this call, retries "
                f"and rewrites included (server cap {cap})",
            ),
        ],
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_MATCH_RECORDS,
                description="process at most this many unfinished source records",
            ),
        ] = 20,
    ) -> Annotated[CallToolResult, Envelope]:
        """Match up to `limit` unfinished source records into the run directory `out`.

        Resumes the run already in `out` (same job); refuses a directory holding a
        different run. Reaching `max_calls` aborts with error `call_limit_reached`
        (status "error"); records left over are `pending` and the run is `partial`
        (status "attention"). Call again with the same arguments to continue."""
        if max_calls > cap:
            return respond(
                ops.OpError(
                    "match",
                    "usage",
                    f"max_calls {max_calls} exceeds this server's cap of {cap}",
                    exit_code=ops.EXIT_USAGE,
                ).result,
                is_error=True,
            )

        async def work() -> ops.OpResult:
            llm = make_llm() if make_llm is not None else None
            # A decider job offline gets the deterministic overlap FakeDecider, never Jev.
            decider = None
            if offline:
                from xwalk.decide.fake import FakeDecider

                decider = FakeDecider(model="xwalk-offline")
            result = await ops.run_async(
                job, out, limit=limit, max_calls=max_calls, llm=llm, decider=decider
            )
            if offline:
                result.warnings.append(
                    ops.OpMessage(
                        "offline_model",
                        "the server runs with --offline-model: every answer is a fixed "
                        "scripted reply, not a model judgement",
                    )
                )
            return result

        return await perform("match", work)

    async def get_run(
        run: Annotated[str, Field(description="run directory written by match_records")],
    ) -> Annotated[CallToolResult, Envelope]:
        """A run's identity, state, current counts, last usage and review overlay.

        `exit_code`/`status` are what the run itself would report (as `xwalk inspect`)."""
        return await perform("inspect", in_thread(lambda: ops.inspect(run)))

    async def list_results(
        run: Annotated[str, Field(description="run directory")],
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=ops.MAX_PAGE_SIZE)] = 50,
        status: Annotated[
            Literal["matched", "needs_review", "unmatched", "failed", "pending"] | None,
            Field(description="only rows with this status"),
        ] = None,
    ) -> Annotated[CallToolResult, ResultsEnvelope]:
        """One page of the run's current results (model decisions), ordered by source id.

        Continue with `offset = data.next_offset` until it is null."""
        return await perform(
            "results",
            in_thread(lambda: ops.list_results(run, offset=offset, limit=limit, status=status)),
        )

    async def explain_result(
        run: Annotated[str, Field(description="run directory")],
        source_id: Annotated[str, Field(min_length=1)],
    ) -> Annotated[CallToolResult, Envelope]:
        """Why one source got its answer: the decision, every attempt (top candidates,
        scores, verifier), the applied review and the source's history."""
        return await perform("explain", in_thread(lambda: ops.explain(run, source_id)))

    server.add_tool(validate_job, annotations=read_only)
    server.add_tool(
        search_candidates,
        annotations=ToolAnnotations(destructive_hint=False, open_world_hint=False),
    )
    server.add_tool(
        match_records, annotations=ToolAnnotations(destructive_hint=False, open_world_hint=True)
    )
    server.add_tool(get_run, annotations=read_only)
    server.add_tool(list_results, annotations=read_only)
    server.add_tool(explain_result, annotations=read_only)
    return server


def serve(*, max_calls_cap: int | None = None, offline_model: bool = False) -> None:
    """Serve on stdin/stdout until the client disconnects (`xwalk mcp`)."""
    build_server(max_calls_cap=max_calls_cap, offline=offline_model).run("stdio")

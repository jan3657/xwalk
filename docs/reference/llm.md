# LLM clients

Everything the matcher knows about a language model goes through one protocol,
`LLMClient`. Two production adapters implement it (`OpenAICompatClient` for any
OpenAI-compatible endpoint, `LiteLLMClient` for everything LiteLLM proxies), one test
double (`FakeLLM`) makes the whole loop runnable offline, and one wrapper (`CachingLLM`)
serves repeated requests from the ledger. All of them are importable from `xwalk.llm`.

The design premise, stated in `base.py`: interchangeability between OpenAI-compatible
endpoints is genuinely partial. Google offers a compatible Gemini endpoint while
recommending the native API; Anthropic describes its compatibility layer as primarily
for testing and notes that strict schema enforcement may be ignored. So adapters
*declare* what they do — and the declaration only decides what to **ask** for, never
whether to **trust** the answer.

## The LLMClient protocol

`LLMClient` is a `@runtime_checkable` `Protocol`. A custom client must implement four
members:

```python
class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """Digest of provider identity, model, generation params, and adapter version."""

    @property
    def capabilities(self) -> LLMCapabilities: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
```

`fingerprint` feeds the run fingerprint, so it must change whenever the client would
produce different answers — model, endpoint, generation parameters, adapter version —
and must never contain a secret, because it ends up in the run manifest on disk.

## Request and response types

All three are frozen dataclasses.

### `LLMRequest`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `system` | `str` | required | System message |
| `user` | `str` | required | User message |
| `schema` | `Mapping[str, Any] \| None` | `None` | JSON schema for structured output, if the adapter can use one |
| `schema_name` | `str` | `"response"` | Name sent alongside the schema in `response_format` |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `1024` | Completion budget; adapters fall back to their own default when this is falsy |
| `seed` | `int \| None` | `None` | Sampling seed; sent only if the adapter declares `seed` support |
| `extra` | `Mapping[str, Any]` | `{}` | Provider-specific body fields, merged into the request body last |

### `LLMResponse`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `text` | `str` | required | Raw completion text |
| `usage` | `Usage` | required | Token and call accounting (`prompt_tokens`, `completion_tokens`, `calls`) |
| `model` | `str` | required | The model string the provider reported, falling back to the configured one |
| `structured` | `bool` | `False` | Whether a schema was **sent** with this request — not whether it was enforced |
| `finish_reason` | `str \| None` | `None` | Provider finish reason (`"stop"`, `"length"`, …) |

### `LLMCapabilities`

What an adapter claims it can do. Every default is `False`: assume nothing.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `json_schema` | `bool` | `False` | The endpoint accepts a `json_schema` response format |
| `strict_schema` | `bool` | `False` | The endpoint enforces the schema, not merely accepts it |
| `usage_reporting` | `bool` | `False` | The endpoint reports token usage |
| `seed` | `bool` | `False` | The endpoint honours a sampling seed |
| `native_retry_after` | `bool` | `False` | The endpoint sends a meaningful `retry-after` header |

Capabilities decide what is *asked for*, never whether the answer is trusted. A client
with `json_schema=True` gets `response_format` in its request body; a client with
`seed=True` gets the seed forwarded. Nothing downstream skips validation because a
capability was declared — every response still goes through `parse_json_object` and
schema-shaped field checks. Declaring a capability the provider lacks costs you a
rejected request or a malformed answer; declaring nothing costs you only prompt-side
JSON discipline.

## Errors

```python
class LLMError(Exception): ...

class LLMRetryableError(LLMError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None: ...

class LLMFatalError(LLMError): ...

class ParseError(LLMError):
    def __init__(self, message: str, *, raw: str) -> None: ...
```

| Error | Raised when |
|---|---|
| `LLMError` | Never directly — the base class for catching all provider failures |
| `LLMRetryableError` | Transient failure: HTTP 408, 409, 429, 5xx, timeouts, connection resets. Carries `retry_after` (seconds) when the provider said how long to wait |
| `LLMFatalError` | Non-retryable failure: auth failure, unknown model, malformed request, a 200 with no choices |
| `ParseError` | The response text could not be reduced to a JSON object (see [Parsing model output](#parsing-model-output)). Carries the original text as `raw` |

`OpenAICompatClient` raises `LLMRetryableError` only after exhausting its own retries;
`LiteLLMClient` raises it immediately and retries nothing.

## OpenAICompatClient

One well-built Chat Completions client over `httpx`, with backoff and a structured-output
fallback. Part of the base install.

```python
OpenAICompatClient(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    capabilities: LLMCapabilities | None = None,
    profile: str = "unknown",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 120.0,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
    extra_headers: Mapping[str, str] | None = None,
)
```

Requests go to `{base_url}/chat/completions` (a trailing `/` on `base_url` is
stripped). `api_key`, when set, becomes an `authorization: Bearer …` header.
An unknown `profile` raises `ValueError` naming the valid ones. An explicit
`capabilities=` argument overrides the profile's. Call `await client.aclose()` when
done with the underlying HTTP client.

### Capability profiles

`CAPABILITY_PROFILES` ships five entries. Each flag changes one concrete thing:
`json_schema` gates whether `response_format` is sent at all, `strict_schema` sets the
`"strict"` field inside it, `seed` gates forwarding `request.seed`, and
`native_retry_after` gates honouring the `retry-after` response header.
`usage_reporting` is a declaration only — usage is always read defensively.

| Profile | `json_schema` | `strict_schema` | `usage_reporting` | `seed` | `native_retry_after` |
|---|---|---|---|---|---|
| `openai` | yes | yes | yes | yes | yes |
| `anthropic-compat` | no | no | yes | no | yes |
| `google-compat` | yes | no | yes | no | no |
| `vllm` | yes | yes | yes | yes | no |
| `unknown` (default) | no | no | no | no | no |

`anthropic-compat` deliberately asks for no schema: Anthropic documents its
compatibility layer as primarily for testing and strict enforcement may be ignored.
`unknown` is the safe default — ask for nothing, validate everything.

### Structured output and the sticky fallback

When `request.schema` is set, `capabilities.json_schema` is true, and fallback has not
been triggered, the body carries:

```python
body["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": request.schema_name,
        "schema": dict(request.schema),
        "strict": capabilities.strict_schema,
    },
}
```

If the provider then answers 400, 404, or 422 with `"response_format"` anywhere in the
error message, the client concludes the capability claim was wrong, disables structured
requests **for the lifetime of the client instance**, and immediately re-sends the same
request as prompt-only JSON — without consuming a retry attempt. The point of the
stickiness: ask once, not once per call. Responses after the fallback report
`structured=False`.

### Retries and backoff

`RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}`. Timeouts and
transport errors (`httpx.TimeoutException`, `httpx.TransportError`) are also retried.
Any other non-200 status raises `LLMFatalError` immediately, with the provider's error
message truncated to 500 characters.

With `max_retries=5` the client makes up to 6 requests. Between attempts it sleeps for
the provider's `retry-after` value if `capabilities.native_retry_after` is set and the
header parses as a float; otherwise `min(backoff_cap, backoff_base * 2**attempt)` —
with the defaults, 1, 2, 4, 8, 16 seconds. When all attempts are spent it raises
`LLMRetryableError("exhausted {max_retries} retries: …", retry_after=…)` carrying the
last `retry-after` seen.

A 200 whose payload has no choices raises `LLMFatalError("provider returned no
choices")`. Missing usage fields become zeros.

### Fingerprint

The fingerprint hashes exactly: `adapter="openai_compat"`, `adapter_version` (currently
`1`), `base_url`, `model`, `profile`, `temperature`, `max_tokens`. The API key is
deliberately absent: rotating a key must not invalidate a resumable run, and a secret
must never reach a manifest on disk. Note that an explicit `capabilities=` override is
*not* hashed — only the profile name is (see [Gotchas](#gotchas)).

## LiteLLMClient

The adapter for everything LiteLLM proxies. Requires the `litellm` extra
(`pip install 'xwalk[litellm]'`); importable from `xwalk.llm.litellm`.

```python
LiteLLMClient(
    model: str,
    *,
    capabilities: LLMCapabilities | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
)
```

`kwargs` are passed through to `litellm.acompletion` on every call, and are hashed
(string-coerced, sorted) into the fingerprint alongside `adapter="litellm"`,
`adapter_version=1`, `model`, `temperature`, and `max_tokens`.

The `litellm` module is imported lazily, on the **first `complete()` call**, via the
`MissingExtra` machinery — constructing the client without the extra installed
succeeds; the first request raises `MissingExtra` naming the install command.

Capabilities default to all-false on purpose: LiteLLM proxies a hundred providers with
wildly different structured-output support, and claiming support that cannot be
verified is exactly what the capability system exists to prevent. Set them explicitly
per model.

Differences from `OpenAICompatClient`, all of them deliberate:

- **No retries.** Errors are classified and raised on the first failure. If you want
  backoff, add it in the caller.
- **Error classification by string matching.** LiteLLM raises many provider-specific
  exception types, so the type name and message are lowercased, underscores removed,
  and scanned for markers. Fatal markers (`authentication`, `permissiondenied`,
  `notfound`, `badrequest`, `invalidrequest`) are checked first, then retryable ones
  (`ratelimit`, `timeout`, `serviceunavailable`, `internalserver`, `apiconnection`,
  `overloaded`).
- **Unknown errors are fatal.** An unrecognised exception becomes `LLMFatalError`:
  retrying a permanent failure burns the budget without ever succeeding.
- **No sticky structured fallback** and no `retry-after` handling. A provider that
  rejects `response_format` will surface as a fatal `badrequest`.

## FakeLLM

A scripted client — the reason the whole matcher loop is testable offline. Exported
from `xwalk.llm`.

```python
FakeLLM(
    script: Sequence[str | BaseException] | None = None,
    *,
    handler: Callable[[LLMRequest], str | BaseException] | None = None,
    capabilities: LLMCapabilities | None = None,
    model: str = "fake",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    finish_reason: str | None = "stop",
)
```

Provide exactly one of `script` or `handler` — both or neither raises `ValueError`.

- **Script mode**: items are consumed in order, one per `complete()` call. A `str`
  becomes the response text; a `BaseException` instance is raised instead (so a script
  can inject an `LLMRetryableError` mid-run). Exhausting the script raises
  `AssertionError` — not a default reply, not a loop — quoting the unexpected
  request's user prompt: a test that makes an extra call has found a real bug.
- **Handler mode**: `handler(request)` decides per call and never exhausts. Use it
  when the reply must depend on which prompt arrived.

Every request is appended to `fake.requests`, and `fake.total_usage` accumulates the
fixed per-call usage — so a test can assert on both what was asked and what it cost.
The response's `structured` flag is `bool(request.schema and capabilities.json_schema)`,
mirroring the real adapters.

A complete offline run of the matcher — no network, no credentials, base install only:

```python
import asyncio
import json
import tempfile

from xwalk import Matcher, MatchPolicy, TemplateSet
from xwalk.llm import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import MatchStatus, Record
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector
from xwalk.stores.memory import MemoryStore

templates = TemplateSet(
    query="{{ mention }}",
    context="{{ mention }}",
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)
prompts = PromptSet.from_slots(
    PromptSlots(
        entity_noun="mention",
        target_noun="term",
        domain_brief="sugars",
        rubric=[
            {"score": 1.0, "name": "Certain", "when": "exact label match"},
            {"score": 0.4, "name": "Weak", "when": "vague resemblance"},
        ],
    )
)

targets = [
    Record(id="T1", fields={"label": "glucose"}),
    Record(id="T2", fields={"label": "fructose"}),
]
store = MemoryStore.from_source(targets)
retriever = BM25Retriever.build(targets, templates, tempfile.mkdtemp())

# Candidates are shown under opaque keys C01, C02, ... in retrieval order.
# A confident source record costs two calls: one selector, one scorer.
# 0.95 clears the default verify_band (0.6, 0.8), so no verifier call follows.
llm = FakeLLM(
    [
        json.dumps(
            {"chosen_key": "C01", "confidence_score": 0.9, "explanation": "exact"}
        ),
        json.dumps({"confidence_score": 0.95, "explanation": "label matches"}),
    ]
)

policy = MatchPolicy()
matcher = Matcher(
    templates=templates,
    retrievers=[retriever],
    store=store,
    selector=Selector(llm, prompts, templates),
    scorer=Scorer(llm, prompts, templates, review_floor=policy.review_floor),
    verifier=Verifier(llm, prompts, templates),
    rewriter=QueryRewriter(llm, prompts, templates),
    policy=policy,
    run_fingerprint="offline-demo",
)

result = asyncio.run(matcher.match(Record(id="s1", fields={"mention": "glucose"})))
assert result.status is MatchStatus.MATCHED
assert result.matched_id == "T1"
assert len(llm.requests) == 2
```

## Caching

`CachingLLM` wraps any `LLMClient` and serves repeats of an identical request from the
ledger.

```python
CachingLLM(
    inner: LLMClient,
    ledger: Ledger,
    *,
    read: bool = True,
    write: bool = True,
)
```

Identity (`model`, `capabilities`, `fingerprint`) delegates to the inner client, so
wrapping a client never changes a run fingerprint — caching changes how an answer was
obtained, never what it means. `read=False` makes it write-through only (refresh the
cache); `write=False` makes it read-only. The wrapper counts `hits` and `misses`.

### The key

```python
llm_cache_key(client: LLMClient, request: LLMRequest) -> str
```

The key hashes everything the provider actually sees, plus who it was sent to:
`CACHE_VERSION` (currently `1`, bumped when the stored representation changes so old
entries are invalidated rather than misread), the client `fingerprint` (which covers
adapter version, endpoint, model, and default generation params), and the request's
`system`, `user`, `schema`, `schema_name`, `temperature`, `max_tokens`, `seed`, and
`extra`. The parts go in as a mapping rather than a concatenated string:
`system="ab"` with `user="c"` must not key the same as `system="a"` with `user="bc"`.

Entries live in the ledger's `llm_cache` SQLite table, keyed by that digest, storing
only the response **text**.

### When caching is and is not safe

Safe: deterministic use — `temperature=0.0` and prompts you intend to be repeatable,
which is the matcher's default posture. Any change to the prompt, schema, generation
parameters, model, endpoint, or adapter version changes the key, so a hit is a genuine
repeat of the same question to the same model.

Not safe, or at least not what you meant:

- **Sampling for diversity.** The key includes `temperature` but not the randomness
  itself; with `temperature > 0` a cache hit replays the first draw forever.
- **Provider-side model drift.** If `model` is an alias the provider re-points (a
  `-latest` tag), the key cannot see the change and stale answers persist.
- **Capability overrides.** An `OpenAICompatClient` given explicit `capabilities=`
  keeps the profile's fingerprint, so two clients that send different request bodies
  can share cache keys.

A cache hit also loses metadata: `usage` is `Usage.zero()` (the honest number — a hit
spends no tokens, so a resumed run's reported cost stays a cost, not a replayed
estimate), `structured` is always `False`, `finish_reason` is the literal `"cached"`
(so truncation detection keyed on `"length"` never fires on a hit), and `model` is the
inner client's configured string, not what the provider reported.

## Parsing model output

`xwalk.llm.parsing` turns whatever the model said into a JSON object. Three behaviours
were learned the hard way in the paper repo: thinking-block stripping (including the
dangling-closing-tag case), fenced-block extraction, and modest repair. Anything these
cannot salvage becomes `UNRESOLVED_OUTPUT` and routes to human review — a malformed
answer is a signal, not a non-match.

### `strip_thinking(text: str) -> str`

Removes reasoning blocks for the tags `think`, `thinking`, and `reasoning`
(case-insensitive, attributes on the opener allowed). Three shapes are handled:

- a properly paired `<think>…</think>` block, removed wherever it appears;
- a **dangling closing tag** with no opener — everything up to and including it was
  reasoning, so the whole prefix is removed;
- an **unclosed opener** — truncated output, so everything from the opener to the end
  is removed.

The result is whitespace-stripped. Plain prose without these tags passes through.

### `extract_json_object(text: str) -> str | None`

Finds the first balanced `{…}`, preferring the contents of a ` ``` ` or ` ```json `
fence when one exists. The scanner tracks string literals and escapes, so braces inside
strings do not unbalance it. Returns `None` when no balanced object exists — it never
raises.

### `repair_json(text: str) -> str`

Only repairs that cannot change meaning: trailing commas before `}` or `]` are removed,
and nothing more. No quote fixing, no bracket balancing.

### `parse_json_object(raw: str) -> dict[str, Any]`

The pipeline: strip thinking, then try in order the cleaned text, the extracted object,
and the repaired cleaned text; finally repair applied to the extracted object. The
first attempt that parses wins.

Tolerated, in combination: thinking blocks in any of the three shapes, prose before and
after the JSON, code fences, and trailing commas — including trailing commas inside a
fenced object surrounded by prose.

Raises `ParseError` (with the original text on `.raw`) when:

- a parse succeeds but the value is not an object — a top-level array or scalar is
  `"expected a JSON object, got list"`, not silently wrapped;
- nothing parses at all — truncated objects, single-quoted keys, and other damage
  beyond a trailing comma are not repaired.

## Gotchas

- **Capabilities are requests, not guarantees.** `structured=True` on a response means
  a schema was sent, not that the provider enforced it. Validate regardless — the rest
  of xwalk does.
- **Explicit `capabilities=` is invisible to the fingerprint** on both adapters
  (`OpenAICompatClient` hashes the profile name; `LiteLLMClient` hashes no capability
  state). Two clients differing only in capability overrides share a fingerprint and
  therefore cache keys and run identity. Prefer a profile when one fits.
- **`request.extra` is merged into the body last**, so it can override `model`,
  `messages`, or `response_format`. It is part of the cache key, so at least a
  divergence never aliases in the cache.
- **`request.max_tokens=0` means "use the client default"** — both adapters compute
  `request.max_tokens or self._max_tokens`.
- **The structured fallback is per-instance state.** A fresh process re-discovers the
  rejection with one wasted request. It also never triggers on providers that accept
  `response_format` and then ignore it — that is what `strict_schema=False` profiles
  are for.
- **`LiteLLMClient` never retries.** If your provider rate-limits, either wrap the
  client with your own backoff or use `OpenAICompatClient` against the provider's
  compatible endpoint.
- **A cache hit changes response metadata.** Zero usage, `structured=False`,
  `finish_reason="cached"`. Do not build logic on those fields being provider-truthful
  under `CachingLLM`.
- **`FakeLLM` exhaustion raises `AssertionError`**, not an `LLMError` — deliberately,
  so retry logic in the code under test cannot swallow it.
- **`usage_reporting` is declared but not branched on** by either adapter; missing
  provider usage simply becomes zeros. Treat zero usage from a provider that should
  report as a sign the declaration is wrong.

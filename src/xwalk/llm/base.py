"""The LLM contract.

Interchangeability between OpenAI-compatible endpoints is genuinely partial: Google
offers an OpenAI-compatible Gemini endpoint while recommending the native API, and
Anthropic describes its compatibility layer as primarily for testing and notes that
strict schema enforcement may be ignored. So adapters *declare* what they do — and the
declaration only decides what to **ask** for, never whether to **trust** the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from xwalk.records import Usage


class LLMError(Exception):
    """Base class for provider failures.

    `usage` is what the client observed before failing: how many requests it dispatched
    (including its own internal retries) and any tokens the provider reported. `None`
    means the client could not say; callers then count one call with unknown usage
    (see `failure_usage`).
    """

    def __init__(self, *args: object, usage: Usage | None = None) -> None:
        super().__init__(*args)
        self.usage = usage


class LLMRetryableError(LLMError):
    """Transient: 408, 409, 429, 5xx, timeouts, connection resets."""

    def __init__(
        self, message: str, *, retry_after: float | None = None, usage: Usage | None = None
    ) -> None:
        super().__init__(message, usage=usage)
        self.retry_after = retry_after


class LLMFatalError(LLMError):
    """Non-retryable: auth failure, unknown model, malformed request."""


class ParseError(LLMError):
    """The response could not be reduced to a JSON object."""

    def __init__(self, message: str, *, raw: str, usage: Usage | None = None) -> None:
        super().__init__(message, usage=usage)
        self.raw = raw


def failure_usage(exc: BaseException) -> Usage:
    """The usage to book for a stage call that raised `exc`.

    A client that knows (it counted its own retries) attaches it to the error. Anything
    else -- a scripted test double, a timeout outside the client -- counts as one
    dispatched call whose tokens are unknown, never as a free call.
    """
    usage = getattr(exc, "usage", None)
    return usage if isinstance(usage, Usage) else Usage.unreported()


def token_count(value: object) -> int | None:
    """A provider-reported token count, or None when it is not a usable count.

    Usable means a non-negative integer (an integral float is accepted). A missing value
    (`None`), booleans, strings, NaN, infinities and negatives are not counts.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def reported_usage(prompt: object, completion: object) -> Usage:
    """Usage of one call that returned. Its tokens count only if reported and valid.

    Both counts must be reported and valid. A missing count (even one of the two), or a
    count that is not a count, makes the call's usage unknown, never a known zero; it
    never raises, so bad accounting cannot discard an otherwise valid answer.
    """
    prompt_tokens = token_count(prompt)
    completion_tokens = token_count(completion)
    if prompt_tokens is None or completion_tokens is None:
        return Usage.unreported()
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, calls=1)


@dataclass(frozen=True)
class LLMCapabilities:
    """What an adapter claims it can do. Defaults are all-false: assume nothing."""

    json_schema: bool = False
    strict_schema: bool = False
    usage_reporting: bool = False
    seed: bool = False
    native_retry_after: bool = False


def request_capabilities(capabilities: LLMCapabilities) -> dict[str, bool]:
    """The capability flags that change what a client sends, for identity.

    `strict_schema` is sent only inside a schema, so it counts only with `json_schema`.
    `usage_reporting` and `native_retry_after` change how an answer is accounted or
    retried, never the request, so they are not identity.
    """
    return {
        "json_schema": capabilities.json_schema,
        "strict_schema": capabilities.json_schema and capabilities.strict_schema,
        "seed": capabilities.seed,
    }


def client_generation_defaults(client: object) -> Mapping[str, Any]:
    """The `temperature`/`max_tokens` a client sends when a request leaves them unset.

    Optional for a client: one that does not report them (`generation_defaults`) yields
    an empty mapping, and its stages' explicit overrides then count as they are given.
    """
    defaults = getattr(client, "generation_defaults", None)
    return defaults if isinstance(defaults, Mapping) else {}


def effective_generation(
    client: object, *, temperature: float | None, max_tokens: int | None
) -> dict[str, Any]:
    """The generation settings a stage's requests actually carry (CONTRACTS.md section 6).

    An explicit per-stage value wins; `None` falls back to the client's configured value.
    So `Scorer(temperature=0.0)` on a client configured with 0.0 is the same request, and
    the same identity, as no override at all.
    """
    defaults = client_generation_defaults(client)
    return {
        "temperature": defaults.get("temperature") if temperature is None else temperature,
        "max_tokens": defaults.get("max_tokens") if max_tokens is None else max_tokens,
    }


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: Mapping[str, Any] | None = None
    schema_name: str = "response"
    # None means "use the client's configured value". A stage sets these only when its
    # caller passed an explicit per-stage override (CONTRACTS.md section 6).
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage
    model: str
    structured: bool = False
    finish_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """Digest of provider identity, model, generation params, and adapter version."""

    @property
    def capabilities(self) -> LLMCapabilities: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


__all__ = [
    "LLMCapabilities",
    "LLMClient",
    "LLMError",
    "LLMFatalError",
    "LLMRequest",
    "LLMResponse",
    "LLMRetryableError",
    "ParseError",
    "client_generation_defaults",
    "effective_generation",
    "failure_usage",
    "request_capabilities",
]

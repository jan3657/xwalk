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
    """Base class for provider failures."""


class LLMRetryableError(LLMError):
    """Transient: 408, 409, 429, 5xx, timeouts, connection resets."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMFatalError(LLMError):
    """Non-retryable: auth failure, unknown model, malformed request."""


class ParseError(LLMError):
    """The response could not be reduced to a JSON object."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class LLMCapabilities:
    """What an adapter claims it can do. Defaults are all-false: assume nothing."""

    json_schema: bool = False
    strict_schema: bool = False
    usage_reporting: bool = False
    seed: bool = False
    native_retry_after: bool = False


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: Mapping[str, Any] | None = None
    schema_name: str = "response"
    temperature: float = 0.0
    max_tokens: int = 1024
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
]

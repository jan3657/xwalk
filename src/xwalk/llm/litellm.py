"""LiteLLM adapter. Requires xwalk[litellm].

Capabilities default to all-false: LiteLLM proxies a hundred providers with wildly
different structured-output support, and claiming support we cannot verify is exactly
what the capability system exists to prevent. Set them explicitly per model.

Usage limitation: retries performed inside LiteLLM (its `num_retries` and router
fallbacks) are not observable from here. Each `complete` call is counted as one
upstream call; a call that raises is counted as one call with unknown usage. Pass
`num_retries=0` if call counts must be exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xwalk._extras import require
from xwalk.fingerprint import hash_value, redact_url, without_secrets
from xwalk.llm.base import (
    LLMCapabilities,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
    reported_usage,
)
from xwalk.records import Usage

ADAPTER_VERSION = 1

# Endpoint kwargs: identity, but hashed with credentials removed (`redact_url`).
_URL_KWARGS = frozenset({"api_base", "base_url"})

_RETRYABLE_MARKERS = (
    "ratelimit",
    "timeout",
    "serviceunavailable",
    "internalserver",
    "apiconnection",
    "overloaded",
)
_FATAL_MARKERS = (
    "authentication",
    "permissiondenied",
    "notfound",
    "badrequest",
    "invalidrequest",
)


def _field(obj: object, name: str) -> Any:
    """`obj.name` or `obj[name]`: LiteLLM objects support both, plain dicts only one."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _reported_usage(raw: object) -> Usage:
    """Tokens count only if reported and valid; otherwise the call's usage is unknown."""
    return reported_usage(_field(raw, "prompt_tokens"), _field(raw, "completion_tokens"))


class LiteLLMClient:
    def __init__(
        self,
        model: str,
        *,
        capabilities: LLMCapabilities | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._capabilities = capabilities or LLMCapabilities()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._kwargs = kwargs
        self._module: Any = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "adapter": "litellm",
                "adapter_version": ADAPTER_VERSION,
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                **({"seed": self._seed} if self._seed is not None else {}),
                # Credentials passed as kwargs (api_key, extra_headers...) are not identity,
                # and neither is userinfo or a key in an endpoint URL's query string.
                "kwargs": {
                    k: redact_url(str(v)) if k in _URL_KWARGS else str(v)
                    for k, v in sorted(without_secrets(self._kwargs).items())
                },
            }
        )

    async def _acompletion(self, **kwargs: Any) -> Any:
        if self._module is None:
            self._module = require("litellm", "litellm", purpose="LiteLLMClient")
        return await self._module.acompletion(**kwargs)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            # A request value is an explicit per-stage override; None means "ours".
            "temperature": self._temperature
            if request.temperature is None
            else request.temperature,
            "max_tokens": self._max_tokens if request.max_tokens is None else request.max_tokens,
            **self._kwargs,
        }
        seed = self._seed if request.seed is None else request.seed
        if seed is not None and self._capabilities.seed:
            body["seed"] = seed
        structured = request.schema is not None and self._capabilities.json_schema
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": dict(request.schema or {}),
                    "strict": self._capabilities.strict_schema,
                },
            }

        try:
            completion = await self._acompletion(**body)
        except Exception as exc:  # LiteLLM raises many provider-specific types
            marker = f"{type(exc).__name__}{exc}".lower().replace("_", "")
            unknown = Usage.unreported()
            if any(m in marker for m in _FATAL_MARKERS):
                raise LLMFatalError(str(exc), usage=unknown) from exc
            if any(m in marker for m in _RETRYABLE_MARKERS):
                raise LLMRetryableError(str(exc), usage=unknown) from exc
            # An error we do not recognise is treated as fatal: retrying a permanent
            # failure burns the budget without ever succeeding.
            raise LLMFatalError(str(exc), usage=unknown) from exc

        return self._parse(completion, structured=structured)

    def _parse(self, completion: object, *, structured: bool) -> LLMResponse:
        """Reduce a completion to a response. Anything malformed is a retryable error.

        A malformed envelope says nothing about credentials or the request, so it is
        recoverable (CONTRACTS.md section 3), booked as one call with unknown usage.
        """

        def malformed(what: str) -> LLMRetryableError:
            return LLMRetryableError(f"malformed response: {what}", usage=Usage.unreported())

        choices = _field(completion, "choices")
        if not choices:
            raise malformed("provider returned no choices")
        if not isinstance(choices, (list, tuple)):
            raise malformed("choices is not a list")
        message = _field(choices[0], "message")
        if message is None or isinstance(message, (str, bytes)):
            raise malformed("choice has no message object")
        text = _field(message, "content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise malformed(f"content is a {type(text).__name__}, not a string")
        finish_reason = _field(choices[0], "finish_reason")
        model = _field(completion, "model")
        return LLMResponse(
            text=text,
            usage=_reported_usage(_field(completion, "usage")),
            model=str(model) if model is not None else self._model,
            structured=structured,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )

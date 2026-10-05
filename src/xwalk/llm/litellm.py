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

from typing import Any

from xwalk._extras import require
from xwalk.fingerprint import hash_value
from xwalk.llm.base import (
    LLMCapabilities,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
)
from xwalk.records import Usage

ADAPTER_VERSION = 1

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


def _reported_usage(raw: Any) -> Usage:
    prompt = getattr(raw, "prompt_tokens", None)
    completion = getattr(raw, "completion_tokens", None)
    if prompt is None and completion is None:
        return Usage.unreported()
    return Usage(prompt_tokens=int(prompt or 0), completion_tokens=int(completion or 0), calls=1)


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
                "kwargs": {k: str(v) for k, v in sorted(self._kwargs.items())},
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

        usage = _reported_usage(getattr(completion, "usage", None))
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMFatalError("provider returned no choices", usage=usage)
        return LLMResponse(
            text=getattr(choices[0].message, "content", "") or "",
            usage=usage,
            model=str(getattr(completion, "model", self._model)),
            structured=structured,
            finish_reason=getattr(choices[0], "finish_reason", None),
        )

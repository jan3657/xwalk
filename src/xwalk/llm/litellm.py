"""LiteLLM adapter. Requires xwalk[litellm].

Capabilities default to all-false: LiteLLM proxies a hundred providers with wildly
different structured-output support, and claiming support we cannot verify is exactly
what the capability system exists to prevent. Set them explicitly per model.
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


class LiteLLMClient:
    def __init__(
        self,
        model: str,
        *,
        capabilities: LLMCapabilities | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._capabilities = capabilities or LLMCapabilities()
        self._temperature = temperature
        self._max_tokens = max_tokens
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
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self._max_tokens,
            **self._kwargs,
        }
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
            if any(m in marker for m in _FATAL_MARKERS):
                raise LLMFatalError(str(exc)) from exc
            if any(m in marker for m in _RETRYABLE_MARKERS):
                raise LLMRetryableError(str(exc)) from exc
            # An error we do not recognise is treated as fatal: retrying a permanent
            # failure burns the budget without ever succeeding.
            raise LLMFatalError(str(exc)) from exc

        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMFatalError("provider returned no choices")
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=getattr(choices[0].message, "content", "") or "",
            usage=Usage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                calls=1,
            ),
            model=str(getattr(completion, "model", self._model)),
            structured=structured,
            finish_reason=getattr(choices[0], "finish_reason", None),
        )

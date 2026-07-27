"""One well-built OpenAI-compatible client, with declared capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

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

RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

CAPABILITY_PROFILES: Mapping[str, LLMCapabilities] = {
    # Full OpenAI Chat Completions.
    "openai": LLMCapabilities(
        json_schema=True,
        strict_schema=True,
        usage_reporting=True,
        seed=True,
        native_retry_after=True,
    ),
    # Anthropic's OpenAI compatibility layer is documented as primarily for testing, and
    # strict schema enforcement may be ignored — so we do not ask for it.
    "anthropic-compat": LLMCapabilities(
        json_schema=False,
        strict_schema=False,
        usage_reporting=True,
        seed=False,
        native_retry_after=True,
    ),
    # Gemini's OpenAI-compatible endpoint accepts json_schema but is not strict.
    "google-compat": LLMCapabilities(
        json_schema=True,
        strict_schema=False,
        usage_reporting=True,
        seed=False,
        native_retry_after=False,
    ),
    # vLLM / SGLang with guided decoding.
    "vllm": LLMCapabilities(
        json_schema=True,
        strict_schema=True,
        usage_reporting=True,
        seed=True,
        native_retry_after=False,
    ),
    # The safe default: ask for nothing, validate everything.
    "unknown": LLMCapabilities(),
}


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        return str(error.get("message", error))[:500]
    return str(error)[:500]


class OpenAICompatClient:
    """Chat Completions over httpx, with backoff and structured-output fallback."""

    def __init__(
        self,
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
    ) -> None:
        if profile not in CAPABILITY_PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; choose from {sorted(CAPABILITY_PROFILES)}"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._profile = profile
        self._capabilities = capabilities or CAPABILITY_PROFILES[profile]
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._extra_headers = dict(extra_headers or {})
        self._structured_disabled = False
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    @property
    def fingerprint(self) -> str:
        # The API key is deliberately absent: rotating a key must not invalidate a
        # resumable run, and a secret must never reach a manifest on disk.
        return hash_value(
            {
                "adapter": "openai_compat",
                "adapter_version": ADAPTER_VERSION,
                "base_url": self._base_url,
                "model": self._model,
                "profile": self._profile,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        )

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self._max_tokens,
        }
        if structured and request.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": dict(request.schema),
                    "strict": self._capabilities.strict_schema,
                },
            }
        if request.seed is not None and self._capabilities.seed:
            body["seed"] = request.seed
        body.update(request.extra)
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self._extra_headers}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        return await self._client.post(
            f"{self._base_url}/chat/completions", json=body, headers=self._headers()
        )

    def _retry_after(self, response: httpx.Response) -> float | None:
        if not self._capabilities.native_retry_after:
            return None
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        structured = (
            request.schema is not None
            and self._capabilities.json_schema
            and not self._structured_disabled
        )

        last_error = "unknown"
        last_retry_after: float | None = None
        for attempt in range(self._max_retries + 1):
            body = self._body(request, structured=structured)
            try:
                response = await self._post(body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, structured=structured)

                message = _error_message(response)
                if (
                    structured
                    and response.status_code in (400, 404, 422)
                    and "response_format" in message.lower()
                ):
                    # The provider claimed json_schema support and rejected it anyway.
                    # Drop to prompt-only JSON and remember, so we ask once, not per call.
                    self._structured_disabled = True
                    structured = False
                    continue
                if response.status_code not in RETRYABLE_STATUSES:
                    raise LLMFatalError(f"HTTP {response.status_code}: {message}")
                last_error = f"HTTP {response.status_code}: {message}"
                last_retry_after = self._retry_after(response)

            if attempt < self._max_retries:
                delay = last_retry_after or min(
                    self._backoff_cap, self._backoff_base * (2**attempt)
                )
                if delay:
                    await asyncio.sleep(delay)

        raise LLMRetryableError(
            f"exhausted {self._max_retries} retries: {last_error}",
            retry_after=last_retry_after,
        )

    def _parse(self, response: httpx.Response, *, structured: bool) -> LLMResponse:
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise LLMFatalError("provider returned no choices")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        raw_usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                completion_tokens=int(raw_usage.get("completion_tokens", 0)),
                calls=1,
            ),
            model=str(payload.get("model", self._model)),
            structured=structured,
            finish_reason=choices[0].get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

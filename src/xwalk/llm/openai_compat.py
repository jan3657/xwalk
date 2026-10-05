"""One well-built OpenAI-compatible client, with declared capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

from xwalk.fingerprint import hash_value, redact_url
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


def _reported_usage(raw: object) -> Usage:
    """One successful call. Its tokens count only if the provider actually reported them."""
    if not isinstance(raw, Mapping) or not ("prompt_tokens" in raw or "completion_tokens" in raw):
        return Usage.unreported()
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        calls=1,
    )


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
        seed: int | None = None,
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
        self._seed = seed
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
                # Userinfo or a key in the query string is a credential, not identity.
                "base_url": redact_url(self._base_url),
                "model": self._model,
                "profile": self._profile,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                # Only when set, so fingerprints of runs without a seed are unchanged.
                **({"seed": self._seed} if self._seed is not None else {}),
            }
        )

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        # Precedence (CONTRACTS.md section 6): a value the request carries is an explicit
        # per-stage override; None falls back to this client's configured value.
        temperature = self._temperature if request.temperature is None else request.temperature
        max_tokens = self._max_tokens if request.max_tokens is None else request.max_tokens
        seed = self._seed if request.seed is None else request.seed
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
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
        if seed is not None and self._capabilities.seed:
            body["seed"] = seed
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
        """Send one request, retrying transient failures.

        Every POST is counted in the returned (or raised) usage: retries and the
        structured-output fallback are real upstream calls. Only the successful response
        reports tokens, so each earlier POST is counted as a call with unknown usage.
        """
        structured = (
            request.schema is not None
            and self._capabilities.json_schema
            and not self._structured_disabled
        )

        dispatched = 0
        last_error = "unknown"
        last_retry_after: float | None = None
        for attempt in range(self._max_retries + 1):
            body = self._body(request, structured=structured)
            dispatched += 1
            try:
                response = await self._post(body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(
                        response, structured=structured, failed_before=dispatched - 1
                    )

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
                    raise LLMFatalError(
                        f"HTTP {response.status_code}: {message}",
                        usage=Usage.unreported(dispatched),
                    )
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
            usage=Usage.unreported(dispatched),
        )

    def _parse(
        self, response: httpx.Response, *, structured: bool, failed_before: int
    ) -> LLMResponse:
        earlier = Usage.unreported(failed_before)
        payload = response.json()
        usage = earlier + _reported_usage(payload.get("usage"))
        choices = payload.get("choices") or []
        if not choices:
            raise LLMFatalError("provider returned no choices", usage=usage)
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        return LLMResponse(
            text=text,
            usage=usage,
            model=str(payload.get("model", self._model)),
            structured=structured,
            finish_reason=choices[0].get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

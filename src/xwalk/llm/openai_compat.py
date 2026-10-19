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
    reported_usage,
    request_capabilities,
)
from xwalk.llm.budget import CallBudget, CallLimitExceeded
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
    """One successful call. Its tokens count only if the provider actually reported them.

    Malformed counts make the call's usage unknown; they never discard the answer and
    never raise (a bad accounting block is not a reason to spend another call).
    """
    if not isinstance(raw, Mapping):
        return Usage.unreported()
    return reported_usage(raw.get("prompt_tokens"), raw.get("completion_tokens"))


# Error codes in an HTTP 200 error body that mean the request itself is unusable: auth,
# billing, unknown model, invalid request. Anything else in an error body -- a
# retryable status, an unknown code, no code at all -- is treated as transient.
FATAL_BODY_CODES = frozenset({400, 401, 402, 403, 404, 413, 422})


def _body_error(payload: object) -> tuple[int | None, str] | None:
    """The (code, message) of an error object in a 200 body, or None if there is none.

    Some gateways (OpenRouter among them) report upstream failures as HTTP 200 with an
    `error` object instead of choices.
    """
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not error:
        return None
    if not isinstance(error, Mapping):
        return None, str(error)[:500]
    raw_code = error.get("code")
    code: int | None = None
    if isinstance(raw_code, int) and not isinstance(raw_code, bool):
        code = raw_code
    elif isinstance(raw_code, str) and raw_code.isdigit():
        code = int(raw_code)
    return code, str(error.get("message", error))[:500]


def _json_payload(response: httpx.Response, dispatched: int) -> object:
    """The decoded body of a 200, or a retryable error if it is not JSON (an HTML page)."""
    try:
        return response.json()
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError both subclass it
        raise LLMRetryableError(
            f"malformed response: body is not JSON ({response.text[:200]!r})",
            usage=Usage.unreported(dispatched),
        ) from exc


def openai_compat_fingerprint(
    *,
    base_url: str,
    model: str,
    profile: str = "unknown",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    capabilities: LLMCapabilities | None = None,
) -> str:
    """The fingerprint `OpenAICompatClient` would have, computed from its settings alone.

    Pure: it builds no client, so identity can be computed where an HTTP client cannot
    (or need not) be constructed. The API key is deliberately absent: rotating a key
    must not invalidate a resumable run, and a secret must never reach a manifest.
    """
    profile_capabilities = CAPABILITY_PROFILES[profile]
    sent = request_capabilities(capabilities or profile_capabilities)
    return hash_value(
        {
            "adapter": "openai_compat",
            "adapter_version": ADAPTER_VERSION,
            # Userinfo or a key in the query string is a credential, not identity.
            "base_url": redact_url(base_url.rstrip("/")),
            "model": model,
            "profile": profile,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Only when set, so fingerprints of runs without a seed are unchanged.
            **({"seed": seed} if seed is not None else {}),
            # An explicit `capabilities=` that changes what is sent (a schema, a strict
            # flag, a seed) is identity; the profile name covers the rest.
            **(
                {"capabilities": sent} if sent != request_capabilities(profile_capabilities) else {}
            ),
        }
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
        trust_env: bool = True,
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
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=trust_env)
        self._call_budget: CallBudget | None = None

    def attach_call_budget(self, budget: CallBudget) -> None:
        """Reserve from `budget` before every HTTP request, retries included."""
        self._call_budget = budget

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    @property
    def generation_defaults(self) -> Mapping[str, Any]:
        """What a request that leaves `temperature`/`max_tokens` unset is sent with."""
        return {"temperature": self._temperature, "max_tokens": self._max_tokens}

    @property
    def fingerprint(self) -> str:
        return openai_compat_fingerprint(
            base_url=self._base_url,
            model=self._model,
            profile=self._profile,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            seed=self._seed,
            capabilities=self._capabilities,
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
            if self._call_budget is not None:
                try:
                    self._call_budget.acquire()
                except CallLimitExceeded as exc:
                    # The requests already sent (and failed) are still real calls.
                    raise CallLimitExceeded(str(exc), usage=Usage.unreported(dispatched)) from None
            dispatched += 1
            try:
                response = await self._post(body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    payload = _json_payload(response, dispatched)
                    body_error = _body_error(payload)
                    if body_error is None:
                        return self._parse(
                            payload, structured=structured, failed_before=dispatched - 1
                        )
                    code, message = body_error
                    if code in FATAL_BODY_CODES:
                        raise LLMFatalError(
                            f"HTTP 200 with error {code}: {message}",
                            usage=Usage.unreported(dispatched),
                        )
                    last_error = f"HTTP 200 with error {code}: {message}"
                    last_retry_after = None
                    if attempt < self._max_retries:
                        delay = min(self._backoff_cap, self._backoff_base * (2**attempt))
                        if delay:
                            await asyncio.sleep(delay)
                    continue

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

    def _parse(self, payload: object, *, structured: bool, failed_before: int) -> LLMResponse:
        """Reduce a 200 body to a response. Anything malformed is a retryable error.

        A malformed envelope says nothing about credentials or the request, so it is
        recoverable (CONTRACTS.md section 3), and its call is counted with unknown usage.
        """
        dispatched = failed_before + 1

        def malformed(what: str) -> LLMRetryableError:
            return LLMRetryableError(
                f"malformed response: {what}", usage=Usage.unreported(dispatched)
            )

        if not isinstance(payload, Mapping):
            raise malformed(f"body is a JSON {type(payload).__name__}, not an object")
        choices = payload.get("choices")
        if not choices:
            raise malformed("provider returned no choices")
        if not isinstance(choices, list):
            raise malformed("choices is not a list")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise malformed("choice is not an object")
        choice_error = choice.get("error")
        if choice_error:
            # OpenRouter's partial-failure shape: the upstream error sits in the choice
            # (finish_reason "error", empty content). Keep its message; it is transient
            # as far as we can tell, and content beside it is not trustworthy.
            detail = (
                choice_error.get("message", choice_error)
                if isinstance(choice_error, Mapping)
                else choice_error
            )
            raise malformed(f"the choice carries an error: {str(detail)[:500]}")
        message = choice.get("message")
        if message is None:
            message = {}
        if not isinstance(message, Mapping):
            raise malformed("message is not an object")
        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise malformed(f"content is a {type(text).__name__}, not a string")
        finish_reason = choice.get("finish_reason")
        usage = Usage.unreported(failed_before) + _reported_usage(payload.get("usage"))
        return LLMResponse(
            text=text,
            usage=usage,
            model=str(payload.get("model", self._model)),
            structured=structured,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

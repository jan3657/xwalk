"""TypeSafe's Jev over HTTP, through OpenRouter or the vendor endpoint.

The request shape is the same on both; only the URL differs, so the caller passes the
full URL and this adapter appends nothing.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from typing import Any

import httpx

from xwalk.decide.base import (
    DecisionFatalError,
    DecisionResponse,
    DecisionResponseError,
    DecisionRetryableError,
    Question,
    parse_response,
    question_to_dict,
)
from xwalk.decide.budget import reserve
from xwalk.fingerprint import hash_value, redact_url
from xwalk.llm.budget import CallBudget
from xwalk.records import Usage

ADAPTER_VERSION = 2
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping) and "message" in error:
        return str(error["message"])[:500]
    if isinstance(error, str):
        return error[:500]
    return response.text[:500]


class JevClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._url = base_url
        self._model = model
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        headers = {"content-type": "application/json", **dict(extra_headers or {})}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport, headers=headers)
        self._budget: CallBudget | None = None

    def attach_call_budget(self, budget: CallBudget) -> None:
        """Reserve one unit of `budget` before every HTTP request, retries included
        (CONTRACTS.md section 5: `calls` counts every dispatched request), and report
        how each ended (`CallBudget.settle`)."""
        self._budget = budget

    def _settle(self, index: int | None, error: str | None) -> None:
        if index is not None and self._budget is not None:
            self._budget.settle(index, error)

    @property
    def model(self) -> str:
        return self._model

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "adapter": "jev",
                "adapter_version": ADAPTER_VERSION,
                "endpoint": redact_url(self._url),
                "model": self._model,
            }
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        body = {
            "model": self._model,
            "state": state,
            "questions": {name: question_to_dict(q) for name, q in questions.items()},
        }
        last_error = "no attempt made"
        retry_after: float | None = None
        posts = 0
        for attempt in range(self._max_retries + 1):
            index: int | None = None
            if self._budget is not None:
                try:
                    index = reserve(self._budget)
                except DecisionFatalError as exc:
                    exc.usage = Usage.unreported(posts)
                    raise
            posts += 1
            dispatch_error: str | None = None
            try:
                response = await self._http.post(self._url, json=body)
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise DecisionResponseError(
                            f"response was not JSON: {exc}", usage=Usage.unreported(posts)
                        ) from exc
                    if not isinstance(payload, Mapping):
                        raise DecisionResponseError(
                            "response was not a JSON object", usage=Usage.unreported(posts)
                        )
                    try:
                        parsed = parse_response(payload, questions, configured_model=self._model)
                    except DecisionResponseError as exc:
                        exc.usage = Usage.unreported(posts)
                        raise
                    # Earlier posts were dispatched but their usage is unknown.
                    return dataclasses.replace(
                        parsed,
                        usage=parsed.usage
                        + (Usage.unreported(posts - 1) if posts > 1 else Usage.zero()),
                    )
                last_error = f"HTTP {response.status_code}: {_error_message(response)}"
                dispatch_error = last_error
                if response.status_code not in RETRYABLE_STATUSES:
                    raise DecisionFatalError(last_error, usage=Usage.unreported(posts))
                header = response.headers.get("retry-after")
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = dispatch_error = f"{type(exc).__name__}: {exc}"
            except BaseException as exc:
                dispatch_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                # HTTP 200 is not a success until the answer has been interpreted.
                self._settle(index, dispatch_error)
            if attempt < self._max_retries:
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(self._backoff_cap, self._backoff_base * 2**attempt)
                )
                await asyncio.sleep(delay)
        raise DecisionRetryableError(
            f"exhausted {self._max_retries} retries: {last_error}",
            retry_after=retry_after,
            usage=Usage.unreported(posts),
        )

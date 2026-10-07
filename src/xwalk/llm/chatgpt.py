"""Subscription-backed Responses API adapter for official ChatGPT OAuth tokens."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
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
from xwalk.llm.budget import CallBudget
from xwalk.records import Usage
from xwalk.ui.chatgpt import RESOURCE, ChatGPTError


def _diagnostic(payload: Any, request_id: str | None) -> str:
    """Keep stable error codes and request IDs without echoing an arbitrary body."""
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    parts = []
    if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,150}", code):
        parts.append(code)
    if request_id and re.fullmatch(r"[a-zA-Z0-9_-]{1,150}", request_id):
        parts.append(f"request {request_id}")
    return ": " + ", ".join(parts) if parts else ""


class ChatGPTClient:
    def __init__(
        self,
        model: str,
        token: Callable[[], str],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._token = token
        self._transport = transport
        self._call_budget: CallBudget | None = None

    def attach_call_budget(self, budget: CallBudget) -> None:
        self._call_budget = budget

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        # Prompt JSON is validated by the existing selector. Do not advertise strict
        # schemas until the plan route's support is established for the chosen model.
        return LLMCapabilities(usage_reporting=True)

    @property
    def fingerprint(self) -> str:
        return hash_value({"adapter": "chatgpt_responses", "version": 1, "model": self.model})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            token = await asyncio.to_thread(self._token)
        except ChatGPTError as exc:
            raise LLMFatalError(str(exc), usage=Usage.zero()) from None
        # This route rejects temperature, max_output_tokens, seed, and system-role
        # input messages. Never merge request.extra into its constrained contract.
        body = {
            "model": self.model,
            "instructions": request.system,
            "input": [{"role": "user", "content": request.user}],
            "store": False,
            "stream": True,
        }
        if self._call_budget is not None:
            self._call_budget.acquire()
        try:
            async with httpx.AsyncClient(timeout=120, transport=self._transport) as client:  # noqa: SIM117
                async with client.stream(
                    "POST",
                    RESOURCE + "/responses",
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        try:
                            error_body = response.json()
                        except ValueError:
                            error_body = None
                        # Do not echo arbitrary upstream bodies (or any token) into a ledger.
                        error = (
                            LLMRetryableError
                            if response.status_code in (408, 429, 500, 502, 503, 504)
                            else LLMFatalError
                        )
                        raise error(
                            f"ChatGPT request failed (HTTP {response.status_code})"
                            + _diagnostic(error_body, response.headers.get("x-request-id")),
                            usage=Usage.unreported(),
                        )
                    data: list[str] = []
                    deltas: list[str] = []
                    completed_text: dict[int, dict[int, str]] = {}
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip())
                        elif not line and data:
                            raw = "\n".join(data)
                            data.clear()
                            if raw == "[DONE]":
                                continue
                            event: dict[str, Any] = json.loads(raw)
                            kind = event.get("type")
                            if kind == "response.output_text.delta":
                                deltas.append(str(event.get("delta", "")))
                            elif kind == "response.output_text.done":
                                completed_text.setdefault(int(event.get("output_index", 0)), {})[
                                    int(event.get("content_index", 0))
                                ] = str(event.get("text", ""))
                            if kind == "response.completed":
                                result = event["response"]
                                text = "".join(
                                    c.get("text", "")
                                    for item in result.get("output", [])
                                    if item.get("type") == "message"
                                    for c in item.get("content", [])
                                    if c.get("type") == "output_text"
                                )
                                # Some plan streams omit message content from the terminal
                                # envelope. Its streamed text is still the completed answer.
                                if not text:
                                    text = "".join(
                                        completed_text[i][j]
                                        for i in sorted(completed_text)
                                        for j in sorted(completed_text[i])
                                    ) or "".join(deltas)
                                usage_raw = result.get("usage")
                                usage = Usage.unreported()
                                if (
                                    isinstance(usage_raw, dict)
                                    and "input_tokens" in usage_raw
                                    and "output_tokens" in usage_raw
                                ):
                                    usage = Usage(
                                        prompt_tokens=int(usage_raw["input_tokens"]),
                                        completion_tokens=int(usage_raw["output_tokens"]),
                                        calls=1,
                                    )
                                if not text.strip():
                                    raise LLMFatalError(
                                        "ChatGPT completed without any answer text; "
                                        "try another available model and run again.",
                                        usage=usage,
                                    )
                                return LLMResponse(
                                    text=text,
                                    usage=usage,
                                    model=result.get("model", self.model),
                                    finish_reason="stop",
                                )
                            if kind in ("response.failed", "response.incomplete", "error"):
                                failed = event.get("response", event)
                                raise LLMFatalError(
                                    f"ChatGPT inference ended with {kind}"
                                    + _diagnostic(failed, response.headers.get("x-request-id"))
                                    + "; check plan access and usage in ChatGPT Settings.",
                                    usage=Usage.unreported(),
                                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMRetryableError(
                "ChatGPT request was interrupted.", usage=Usage.unreported()
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMFatalError(
                "ChatGPT returned an invalid response stream.", usage=Usage.unreported()
            ) from exc
        raise LLMFatalError(
            "ChatGPT stream ended without response.completed.", usage=Usage.unreported()
        )

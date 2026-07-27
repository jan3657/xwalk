"""A scripted LLM client. The reason the whole matcher loop is testable offline."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from xwalk.fingerprint import hash_value
from xwalk.llm.base import LLMCapabilities, LLMRequest, LLMResponse
from xwalk.records import Usage

Scripted = str | BaseException
Handler = Callable[[LLMRequest], Scripted]


class FakeLLM:
    """Either a fixed script (consumed in order) or a handler that inspects the request.

    Exhausting the script raises AssertionError rather than looping or returning a
    default: a test that makes an unexpected extra call has found a real bug.
    """

    def __init__(
        self,
        script: Sequence[Scripted] | None = None,
        *,
        handler: Handler | None = None,
        capabilities: LLMCapabilities | None = None,
        model: str = "fake",
        prompt_tokens: int = 100,
        completion_tokens: int = 20,
        finish_reason: str | None = "stop",
    ) -> None:
        if (script is None) == (handler is None):
            raise ValueError("provide exactly one of script or handler")
        self._script = list(script or [])
        self._handler = handler
        self._capabilities = capabilities or LLMCapabilities()
        self._model = model
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._finish_reason = finish_reason
        self._index = 0
        self.requests: list[LLMRequest] = []
        self.total_usage = Usage.zero()

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
                "adapter": "fake",
                "model": self._model,
                "script": [s if isinstance(s, str) else type(s).__name__ for s in self._script],
                "handler": self._handler.__name__ if self._handler else None,
            }
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._handler is not None:
            item: Scripted = self._handler(request)
        else:
            assert self._index < len(self._script), (
                f"FakeLLM script exhausted after {self._index} call(s); "
                f"unexpected request:\n{request.user[:400]}"
            )
            item = self._script[self._index]
            self._index += 1

        if isinstance(item, BaseException):
            raise item

        usage = Usage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            calls=1,
        )
        self.total_usage = self.total_usage + usage
        return LLMResponse(
            text=item,
            usage=usage,
            model=self._model,
            structured=bool(request.schema and self._capabilities.json_schema),
            finish_reason=self._finish_reason,
        )

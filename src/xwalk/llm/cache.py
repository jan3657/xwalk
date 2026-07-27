"""Ledger-backed LLM response caching.

Wraps any `LLMClient`. Identity (`model`, `capabilities`, `fingerprint`) delegates to
the inner client, so wrapping a client never changes a run fingerprint — caching changes
how an answer was obtained, never what it means.
"""

from __future__ import annotations

from xwalk.fingerprint import hash_value
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMCapabilities, LLMClient, LLMRequest, LLMResponse
from xwalk.records import Usage

# Bump when the stored representation changes, to invalidate old entries rather than
# misread them.
CACHE_VERSION = 1


def llm_cache_key(client: LLMClient, request: LLMRequest) -> str:
    """Everything the provider actually sees, plus who it was sent to.

    The parts go in as a mapping rather than a concatenated string: `system="ab"` with
    `user="c"` must not key the same as `system="a"` with `user="bc"`.
    """
    return hash_value(
        {
            "cache_version": CACHE_VERSION,
            # Covers adapter version, endpoint, model, and default generation params.
            "client": client.fingerprint,
            "system": request.system,
            "user": request.user,
            "schema": dict(request.schema) if request.schema is not None else None,
            "schema_name": request.schema_name,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "extra": dict(request.extra),
        }
    )


class CachingLLM:
    """An `LLMClient` that serves repeats of an identical request from the ledger."""

    def __init__(
        self,
        inner: LLMClient,
        ledger: Ledger,
        *,
        read: bool = True,
        write: bool = True,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._read = read
        self._write = write
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._inner.capabilities

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = llm_cache_key(self._inner, request)

        if self._read:
            cached = self._ledger.get_cached(key)
            if cached is not None:
                self.hits += 1
                # Zero usage is the honest number: a cache hit spends no tokens, so a
                # resumed run's reported cost stays a cost, not a replayed estimate.
                return LLMResponse(
                    text=cached,
                    usage=Usage.zero(),
                    model=self._inner.model,
                    structured=False,
                    finish_reason="cached",
                )

        response = await self._inner.complete(request)
        self.misses += 1
        if self._write:
            await self._ledger.put_cached(key, response.text)
        return response

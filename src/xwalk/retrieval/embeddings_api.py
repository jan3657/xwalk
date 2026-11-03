"""A dense encoder backed by an OpenAI-compatible `/embeddings` endpoint.

One client for every service that speaks the OpenAI embeddings shape: OpenRouter, OpenAI,
Voyage, DeepInfra, a local vLLM or Ollama. It needs only httpx, so a dense retriever no
longer requires torch or a downloaded model.

What changes a vector is identity (`settings`): the base URL (redacted), the model, the
dimension, whether `dimensions` is sent, truncation, normalisation, both prefixes, both
input types and the provider preferences. The API key, batch size, timeout and retry
settings are not: they change cost and speed, never vectors. The remote model cannot be
pinned to a revision, so its revision is recorded as "unknown".

The dimension is declared, never probed: computing an index's identity must not spend a
call, and a provider that ignores a requested `dimensions` is caught by the length check
instead of silently producing an index of another shape.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from xwalk.fingerprint import redact_url
from xwalk.llm.openai_compat import (
    FATAL_BODY_CODES,
    RETRYABLE_STATUSES,
    _body_error,
    _error_message,
)
from xwalk.retrieval.base import RetrieverError

ADAPTER = "openai_embeddings"
ADAPTER_VERSION = 1
_MAX_RETRY_AFTER = 300.0  # seconds; a longer Retry-After is waited for this long, then retried


class EmbeddingAPIError(RetrieverError):
    """The embeddings endpoint failed. `retryable` is False for a request that can never
    succeed as sent (auth, billing, unknown model, a vector of the wrong shape)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass(frozen=True)
class EmbeddingUsage:
    """What the encoder has spent so far. `None` means unknown, never zero: one response
    without a count makes the total unknown."""

    requests: int
    texts: int
    prompt_tokens: int | None
    cost: float | None


class _Payload413(Exception):
    """Internal: the batch was too large for the endpoint."""


class OpenAIEmbeddingsEncoder:
    """`Encoder` over `POST {base_url}/embeddings`. Thread-safe."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        dimensions: int,
        api_key: str | None = None,
        send_dimensions: bool = True,
        truncate: bool = False,
        normalize: bool = True,
        query_prefix: str = "",
        doc_prefix: str = "",
        query_input_type: str | None = None,
        doc_input_type: str | None = None,
        provider: Mapping[str, Any] | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        extra_headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        trust_env: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("an embeddings encoder needs a base_url")
        if not model or not model.strip():
            raise ValueError("an embeddings encoder needs a model")
        if dimensions < 1:
            raise ValueError(f"dimensions must be at least 1, got {dimensions}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if max_retries < 0:
            raise ValueError(f"max_retries must not be negative, got {max_retries}")
        self._base_url = base_url.strip().rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key
        self._send_dimensions = send_dimensions
        self._truncate = truncate
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._query_input_type = query_input_type
        self._doc_input_type = doc_input_type
        self._provider = dict(provider) if provider is not None else None
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._sleep = sleep
        headers = {"content-type": "application/json", **dict(extra_headers or {})}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=timeout, transport=transport, trust_env=trust_env, headers=headers
        )
        self._lock = threading.Lock()
        self._requests = 0
        self._texts = 0
        self._tokens = 0
        self._tokens_known = True
        self._cost = 0.0
        self._cost_known = True

    # --- Encoder protocol ------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimensions

    @property
    def settings(self) -> dict[str, Any]:
        """Everything besides name and dimension that changes what a vector means."""
        return {
            "adapter": ADAPTER,
            "adapter_version": ADAPTER_VERSION,
            "base_url": redact_url(self._base_url),
            "revision": "unknown",
            "send_dimensions": self._send_dimensions,
            "truncate": self._truncate,
            "normalize": self._normalize,
            "query_prefix": self._query_prefix,
            "doc_prefix": self._doc_prefix,
            "query_input_type": self._query_input_type,
            "doc_input_type": self._doc_input_type,
            "provider": self._provider,
        }

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        """One vector per text, in order. A blank text is a zero vector and is not sent:
        gateways reject empty strings, and one would fail its whole batch."""
        prefix = self._query_prefix if is_query else self._doc_prefix
        out: list[list[float]] = [[0.0] * self._dimensions for _ in texts]
        pending = [(i, f"{prefix}{t}") for i, t in enumerate(texts) if t.strip()]
        for start in range(0, len(pending), self._batch_size):
            chunk = pending[start : start + self._batch_size]
            vectors = self._embed([text for _, text in chunk], is_query)
            for (index, _), vector in zip(chunk, vectors, strict=True):
                out[index] = vector
        return out

    # --- usage -----------------------------------------------------------------------

    @property
    def usage(self) -> EmbeddingUsage:
        with self._lock:
            return EmbeddingUsage(
                requests=self._requests,
                texts=self._texts,
                prompt_tokens=self._tokens if self._tokens_known else None,
                cost=self._cost if self._cost_known else None,
            )

    def probe_dimension(self, text: str = "dimension probe") -> int:
        """The length of the vector the endpoint returns for one document text, as sent
        with this encoder's settings. Spends one call; nothing else here probes."""
        return len(self._embed_raw([f"{self._doc_prefix}{text}"], False)[0])

    def close(self) -> None:
        self._client.close()

    # --- one batch -------------------------------------------------------------------

    def _embed(self, texts: list[str], is_query: bool) -> list[list[float]]:
        return [self._shape(vector) for vector in self._embed_raw(texts, is_query)]

    def _embed_raw(self, texts: list[str], is_query: bool) -> list[list[float]]:
        try:
            return self._embed_once(texts, is_query)
        except _Payload413 as exc:
            if len(texts) == 1:
                raise EmbeddingAPIError(str(exc), status=413) from None
            middle = len(texts) // 2
            return self._embed_raw(texts[:middle], is_query) + self._embed_raw(
                texts[middle:], is_query
            )

    def _body(self, texts: list[str], is_query: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._model, "input": texts, "encoding_format": "float"}
        if self._send_dimensions:
            body["dimensions"] = self._dimensions
        input_type = self._query_input_type if is_query else self._doc_input_type
        if input_type:
            body["input_type"] = input_type
        if self._provider is not None:
            body["provider"] = self._provider
        return body

    def _embed_once(self, texts: list[str], is_query: bool) -> list[list[float]]:
        body = self._body(texts, is_query)
        url = f"{self._base_url}/embeddings"
        last: EmbeddingAPIError | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(self._delay(attempt, last))
            with self._lock:
                self._requests += 1
            try:
                response = self._client.post(url, content=json.dumps(body))
            except httpx.TransportError as exc:  # timeouts included
                last = EmbeddingAPIError(f"embeddings request failed: {exc}", retryable=True)
                continue
            try:
                return self._parse(response, len(texts))
            except EmbeddingAPIError as exc:
                if not exc.retryable:
                    raise
                last = exc
        assert last is not None
        raise EmbeddingAPIError(
            f"{last} (gave up after {self._max_retries + 1} attempts)",
            status=last.status,
            retryable=True,
        )

    def _delay(self, attempt: int, last: EmbeddingAPIError | None) -> float:
        if last is not None and last.retry_after is not None:
            return min(last.retry_after, _MAX_RETRY_AFTER)
        ceiling = min(self._backoff_cap, self._backoff_base * 2 ** (attempt - 1))
        return float(random.uniform(0.0, ceiling))  # noqa: S311 -- jitter, not security

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        status = response.status_code
        if status == 413:
            raise _Payload413(f"embeddings endpoint: payload too large: {_error_message(response)}")
        if status != 200:
            raise EmbeddingAPIError(
                f"embeddings endpoint returned HTTP {status}: {_error_message(response)}",
                status=status,
                retryable=status in RETRYABLE_STATUSES,
                retry_after=_retry_after(response),
            )
        try:
            payload = response.json()
        except ValueError:
            raise EmbeddingAPIError(
                f"embeddings endpoint returned a body that is not JSON: {response.text[:200]!r}",
                status=status,
                retryable=True,
            ) from None
        body_error = _body_error(payload)
        if body_error is not None:
            code, message = body_error
            raise EmbeddingAPIError(
                f"embeddings endpoint reported an error: {message}",
                status=code,
                retryable=code not in FATAL_BODY_CODES,
            )
        vectors = self._vectors(payload, expected)
        self._book(payload, expected)
        return vectors

    def _vectors(self, payload: object, expected: int) -> list[list[float]]:
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list) or len(data) != expected:
            got = len(data) if isinstance(data, list) else "no"
            raise EmbeddingAPIError(
                f"embeddings endpoint returned {got} vectors for {expected} texts",
                retryable=True,
            )
        # Restore input order from `index` when it is a permutation; otherwise trust order.
        positions: dict[int, Any] = {}
        for item in data:
            index = item.get("index") if isinstance(item, Mapping) else None
            if isinstance(index, int) and not isinstance(index, bool):
                positions[index] = item
        if sorted(positions) == list(range(expected)):
            data = [positions[i] for i in range(expected)]
        vectors: list[list[float]] = []
        for item in data:
            raw = item.get("embedding") if isinstance(item, Mapping) else None
            if not isinstance(raw, list) or not all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in raw
            ):
                raise EmbeddingAPIError(
                    "embeddings endpoint returned an embedding that is not a list of numbers "
                    "(xwalk requests encoding_format=float)"
                )
            vector = [float(v) for v in raw]
            if not all(math.isfinite(v) for v in vector):
                raise EmbeddingAPIError("embeddings endpoint returned a non-finite value")
            vectors.append(vector)
        return vectors

    def _shape(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dimensions:
            if not (self._truncate and len(vector) > self._dimensions):
                hint = (
                    "set dimensions to match"
                    if len(vector) < self._dimensions
                    else "set dimensions to match, or truncate: true for a model trained "
                    "for truncation (Matryoshka)"
                )
                raise EmbeddingAPIError(
                    f"model {self._model!r} returned {len(vector)}-dimensional vectors but "
                    f"dimensions is {self._dimensions}; {hint}"
                )
            vector = vector[: self._dimensions]
            return _unit(vector)
        return _unit(vector) if self._normalize else vector

    def _book(self, payload: object, texts: int) -> None:
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        cost = usage.get("cost") if isinstance(usage, Mapping) else None
        with self._lock:
            self._texts += texts
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                self._tokens += tokens
            else:
                self._tokens_known = False
            if isinstance(cost, int | float) and not isinstance(cost, bool) and cost >= 0:
                self._cost += float(cost)
            else:
                self._cost_known = False


def _unit(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vector))
    return [v / length for v in vector] if length else vector


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


__all__ = ["EmbeddingAPIError", "EmbeddingUsage", "OpenAIEmbeddingsEncoder"]

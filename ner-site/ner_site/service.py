"""Settings from the environment, the LLM factory, and one request end to end.

Binding: docs/ner-site/PLAN.md sections 3.2 (environment), 3.3 (signatures) and 7
(security). ``run`` validates the request, makes the one extraction call, maps the
entities and assembles a :class:`NerResponse`; every failure it knows about leaves as a
:class:`ServiceError` with a code from section 3.1. ``ner_site.http`` turns those into
responses.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ner_site.extract import MAX_TOKENS, TEMPERATURE, extract, render_default_prompt
from ner_site.indexes import OntologyIndex, summarise_status
from ner_site.mapping import MapStats, map_entities
from ner_site.schema import (
    Limits,
    Mode,
    NerRequest,
    NerResponse,
    RetrieverStatus,
    ServiceError,
    Usage,
)
from xwalk.fingerprint import redact_url
from xwalk.llm.base import LLMClient, LLMError
from xwalk.llm.budget import BudgetedLLM, CallBudget, CallLimitExceeded
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.records import Usage as XwalkUsage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = (
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3-next-80b-a3b-instruct",
)
DEFAULT_CACHE_DIR = Path("/tmp/xwalk-ner-site")
OFFLINE_MODEL = "offline"

# Sent to OpenRouter for its app attribution (https://openrouter.ai/docs/app-attribution).
SITE_URL = "https://github.com/jan3657/xwalk"
SITE_TITLE = "xwalk NER demo"
LLM_TIMEOUT_S = 60.0  # one HTTP request
LLM_MAX_RETRIES = 2  # so at most 3 requests, the same as the call budget

# Wall-clock bounds inside the function's 120 s maxDuration (vercel.json): the whole
# extraction step, retries included; then the whole mapping step gets what is left of
# REQUEST_DEADLINE_S (at least MIN_MAP_BUDGET_S), and each retriever call within it at
# most MAP_RETRIEVER_TIMEOUT_S. Worst case: 90 s extracting, 10 s mapping, plus
# mapping.LOCAL_GRACE_S for BM25 only, so a request ends by about 105 s.
EXTRACT_TIMEOUT_S = 90.0
MAP_RETRIEVER_TIMEOUT_S = 30.0
REQUEST_DEADLINE_S = 100.0
MIN_MAP_BUDGET_S = 5.0

MAX_ERROR_DETAIL = 300
MIN_SCRUBBED_CHARS = 8

# (env name, Limits field, lowest, highest). A value outside the range is a settings
# error, not clamped: an operator who typed 500000 should learn that it was refused.
# The highest values keep a request comfortably inside the 64 KB body cap.
_LIMIT_ENV: tuple[tuple[str, str, int, int], ...] = (
    ("NER_SITE_MAX_CHARS", "max_chars", 1, 20000),
    ("NER_SITE_MAX_ENTITIES", "max_entities", 1, 100),
    ("NER_SITE_MAX_PROMPT_CHARS", "max_prompt_chars", 1, 16000),
)
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "NER_SITE_BASE_URL",
    "NER_SITE_MODELS",
    "NER_SITE_PASSCODE",
    "NER_SITE_OFFLINE",
    "NER_SITE_DENSE_DIR",
    "NER_SITE_CACHE_DIR",
    *(name for name, _, _, _ in _LIMIT_ENV),
)


class SettingsError(ValueError):
    """A bad environment value. The message names the variable, never a secret."""


@dataclass(frozen=True)
class IndexView:
    """What ``indexes.get_indexes`` reads (its ``IndexSettings`` protocol)."""

    cache_dir: Path
    dense_dir: Path | None
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Settings:
    """The function's configuration (section 3.2). Secrets are left out of ``repr``."""

    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_BASE_URL
    models: tuple[str, ...] = DEFAULT_MODELS
    passcode: str | None = field(default=None, repr=False)
    limits: Limits = field(default_factory=Limits)
    offline: bool = False
    dense_dir: Path | None = PROJECT_ROOT / "indexes"
    cache_dir: Path = DEFAULT_CACHE_DIR

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Read section 3.2. Empty values count as unset. Raises :class:`SettingsError`."""

        def get(name: str) -> str:
            return (env.get(name) or "").strip()

        base_url = get("NER_SITE_BASE_URL") or DEFAULT_BASE_URL
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise SettingsError(
                "NER_SITE_BASE_URL must be an http(s) URL such as https://openrouter.ai/api/v1"
            )

        raw_models = get("NER_SITE_MODELS")
        if raw_models:
            models = tuple(dict.fromkeys(m.strip() for m in raw_models.split(",") if m.strip()))
            if not models:
                raise SettingsError("NER_SITE_MODELS must list at least one model id")
        else:
            models = DEFAULT_MODELS

        passcode = get("NER_SITE_PASSCODE") or None
        if passcode is not None and not passcode.isascii():
            # http.server decodes header values as Latin-1 and browsers refuse other
            # characters in a header, so a non-ASCII passcode could never match.
            raise SettingsError("NER_SITE_PASSCODE must be ASCII (printable, no accents)")

        offline_raw = get("NER_SITE_OFFLINE").lower()
        if offline_raw not in _TRUE | _FALSE:
            raise SettingsError("NER_SITE_OFFLINE must be 1 or 0 (or true/false, yes/no, on/off)")

        numbers: dict[str, int] = {}
        for name, attr, low, high in _LIMIT_ENV:
            raw = get(name)
            if not raw:
                continue
            if not re.fullmatch(r"[0-9]+", raw) or not low <= int(raw) <= high:
                raise SettingsError(f"{name} must be a whole number from {low} to {high}")
            numbers[attr] = int(raw)

        dense = get("NER_SITE_DENSE_DIR")
        cache = get("NER_SITE_CACHE_DIR")
        return cls(
            api_key=get("OPENROUTER_API_KEY") or None,
            base_url=base_url.rstrip("/"),
            models=models,
            passcode=passcode,
            limits=Limits(**numbers),
            offline=offline_raw in _TRUE,
            dense_dir=Path(dense) if dense else PROJECT_ROOT / "indexes",
            cache_dir=Path(cache) if cache else DEFAULT_CACHE_DIR,
        )

    @property
    def mode(self) -> Mode:
        return "offline" if self.offline else "live"

    @property
    def offered_models(self) -> tuple[str, ...]:
        """What ``/api/health`` lists and a request may name: the stand-in when offline."""
        return (OFFLINE_MODEL,) if self.offline else self.models

    @property
    def default_model(self) -> str:
        return self.offered_models[0]

    @property
    def default_prompt(self) -> str:
        """The extraction prompt used when a request sends none, stating this
        deployment's entity limit; ``/api/health`` serves it."""
        return render_default_prompt(self.limits.max_entities)

    @property
    def configured(self) -> bool:
        """Whether ``/api/ner`` can run: offline, or a key to call the provider with."""
        return self.offline or self.api_key is not None

    def index_settings(self) -> IndexView:
        """The view the indexes are opened with. Offline mode never spends the key, so
        dense retrieval there reports the key as missing."""
        return IndexView(
            cache_dir=self.cache_dir,
            dense_dir=self.dense_dir,
            api_key=None if self.offline else self.api_key,
        )

    def secrets(self) -> tuple[str, ...]:
        """Values scrubbed from every response and log line as a last line of defence
        (nothing echoes them in the first place). Values shorter than
        ``MIN_SCRUBBED_CHARS`` are left alone: replacing a passcode like ``ok`` would
        corrupt ordinary JSON, and so short a value cannot be hidden anyway."""
        return tuple(s for s in (self.api_key, self.passcode) if s and len(s) >= MIN_SCRUBBED_CHARS)


def scrub(message: str, settings: Settings) -> str:
    """``message`` with the key and passcode removed and every URL redacted."""
    for secret in settings.secrets():
        message = message.replace(secret, "[redacted]")
    return re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+", lambda m: redact_url(m[0]), message)


def make_llm(
    settings: Settings,
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    backoff_base: float = 1.0,
) -> BudgetedLLM:
    """``BudgetedLLM(OpenAICompatClient(...), CallBudget(limit=llm_call_cap))``.

    ``transport`` and ``backoff_base`` exist for tests (MockTransport, no sleeping).
    The caller closes it (``await llm.aclose()``)."""
    client = OpenAICompatClient(
        settings.base_url,
        model,
        api_key=settings.api_key,
        profile="openai",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
        backoff_base=backoff_base,
        transport=transport,
        extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_TITLE},
    )
    return BudgetedLLM(client, CallBudget(limit=settings.limits.llm_call_cap))


def _bad(code: str, message: str) -> ServiceError:
    return ServiceError(code, message, status=400)


def _shown(value: str, limit: int = 80) -> str:
    return repr(value if len(value) <= limit else value[:limit] + "...")


def validate_request(request: NerRequest, settings: Settings) -> NerRequest:
    """The request with defaults resolved, or a 400 :class:`ServiceError` (section 3.1).

    Lengths are Python ``len``: Unicode code points, the unit of every offset."""
    limits = settings.limits
    if not request.text.strip():
        raise _bad("missing_text", "text is empty; paste some text to annotate")
    if len(request.text) > limits.max_chars:
        raise _bad(
            "text_too_long",
            f"text is {len(request.text)} characters; the limit is {limits.max_chars}",
        )
    prompt = (
        request.prompt if request.prompt and request.prompt.strip() else settings.default_prompt
    )
    if len(prompt) > limits.max_prompt_chars:
        raise _bad(
            "prompt_too_long",
            f"prompt is {len(prompt)} characters; the limit is {limits.max_prompt_chars}",
        )
    if not 1 <= request.top_k <= limits.top_k_max:
        raise _bad("bad_top_k", f"top_k must be from 1 to {limits.top_k_max}")
    model = request.model if request.model is not None else settings.default_model
    if model not in settings.offered_models:
        raise _bad(
            "unknown_model",
            f"model {_shown(model)} is not offered; choose one of "
            f"{', '.join(settings.offered_models)}",
        )
    return NerRequest(text=request.text, prompt=prompt, model=model, top_k=request.top_k)


def wire_usage(usage: XwalkUsage, embedding_requests: int) -> Usage:
    """Tokens are reported only when every call reported them; otherwise null, never 0.
    No call at all (the offline stand-in) reports no tokens either."""
    known = usage.calls > 0 and usage.unknown_calls == 0
    return Usage(
        llm_calls=usage.calls,
        prompt_tokens=usage.prompt_tokens if known else None,
        completion_tokens=usage.completion_tokens if known else None,
        embedding_requests=embedding_requests,
    )


def retrieval_status(
    indexes: Mapping[str, OntologyIndex], settings: Settings
) -> dict[str, RetrieverStatus]:
    """``summarise_status``; when offline mode is what keeps a configured key from dense
    retrieval, the reason says so instead of claiming the key is missing."""
    summary = summarise_status(indexes)
    for kind, status in summary.items():
        reasons = {
            index.status[kind].reason
            for index in indexes.values()
            if kind in index.status and index.status[kind].status != "ok"
        }
        failing = sum(
            1 for i in indexes.values() if kind not in i.status or i.status[kind].status != "ok"
        )
        # One reason shared by every ontology: say it once, not four times.
        if status.status != "ok" and len(reasons) == 1 and failing == len(indexes) > 1:
            (reason,) = reasons
            if reason:
                summary[kind] = RetrieverStatus(
                    status="unavailable", reason=f"{reason} (all {len(indexes)} ontologies)"
                )
    dense = summary.get("dense")
    if settings.offline and settings.api_key and dense is not None and dense.status != "ok":
        summary["dense"] = RetrieverStatus(
            status="unavailable",
            reason="offline mode (NER_SITE_OFFLINE=1) requests no query embeddings; BM25 only",
        )
    return summary


def _clean(note: str, settings: Settings) -> str:
    """A retriever note fit for a response: secrets and URL userinfo removed, capped."""
    note = scrub(note, settings)
    return note if len(note) <= MAX_ERROR_DETAIL else note[: MAX_ERROR_DETAIL - 3] + "..."


def _dense_outcome(
    retrieval: dict[str, RetrieverStatus], stats: MapStats, notes: list[str]
) -> None:
    """What dense retrieval did in this request, beyond its open-time status: failing
    for every entity that tried it makes it unavailable; failing for some is a warning
    naming them."""
    dense = retrieval.get("dense")
    tried, failed = stats.dense_entities, stats.dense_failed
    if dense is None or dense.status != "ok" or not tried or not failed:
        return
    if all(eid in failed for eid in tried):
        retrieval["dense"] = RetrieverStatus(
            status="unavailable",
            reason=f"every dense search failed in this request: {failed[tried[0]]}",
        )
        return
    ids = [eid for eid in tried if eid in failed]
    notes.append(
        f"dense search failed for {len(ids)} of {len(tried)} entities ({', '.join(ids)}); "
        "their matches use BM25 only"
    )


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


async def run(
    request: NerRequest,
    *,
    settings: Settings,
    llm: LLMClient,
    indexes: Mapping[str, OntologyIndex],
    started_at: float | None = None,
) -> NerResponse:
    """One request: validate, extract with ``llm`` (one call), map, report.

    ``started_at`` (``time.monotonic()``) is when the request began, if earlier than
    this call (opening the indexes on a cold start); the mapping deadline counts from it.

    Raises :class:`ServiceError`: 400 codes from :func:`validate_request`, 422 from the
    extraction, 429 ``call_budget_exhausted``, 502 ``provider_error``."""
    req = validate_request(request, settings)
    assert req.prompt is not None and req.model is not None
    started = time.monotonic()
    try:
        entities, warnings, usage = await asyncio.wait_for(
            extract(llm, req.text, req.prompt, max_entities=settings.limits.max_entities),
            timeout=EXTRACT_TIMEOUT_S,
        )
    except ServiceError:
        raise
    except CallLimitExceeded:
        raise ServiceError(
            "call_budget_exhausted",
            f"the limit of {settings.limits.llm_call_cap} model calls per request was "
            "reached before the model answered; try again later",
            status=429,
        ) from None
    except LLMError as exc:
        detail = scrub(f"{type(exc).__name__}: {exc}", settings)[:MAX_ERROR_DETAIL]
        raise ServiceError(
            "provider_error", f"the model provider failed ({detail})", status=502
        ) from None
    except asyncio.TimeoutError:  # builtins.TimeoutError from 3.11; distinct on 3.10
        raise ServiceError(
            "provider_error",
            f"the model did not answer within {EXTRACT_TIMEOUT_S:g} seconds",
            status=502,
        ) from None
    extracted = time.monotonic()

    stats = MapStats()
    began = started if started_at is None else min(started_at, started)
    budget = max(MIN_MAP_BUDGET_S, REQUEST_DEADLINE_S - (extracted - began))
    results = await map_entities(
        entities,
        indexes,
        top_k=req.top_k,
        timeout=min(MAP_RETRIEVER_TIMEOUT_S, budget),
        deadline=budget,
        stats=stats,
        clean=lambda note: _clean(note, settings),
    )
    mapped = time.monotonic()

    retrieval = retrieval_status(indexes, settings)
    notes: list[str] = list(warnings) + stats.warnings
    _dense_outcome(retrieval, stats, notes)
    routed = any(r.ontology is not None for r in results)
    for kind in ("bm25", "dense"):
        status = retrieval.get(kind)
        if routed and status is not None and status.status != "ok":
            fallback = "BM25 only" if kind == "dense" else "dense only"
            notes.append(f"{kind} retrieval unavailable; matches use {fallback}")
    return NerResponse(
        model=req.model,
        mode=settings.mode,
        entities=results,
        retrieval=retrieval,
        warnings=[scrub(n, settings) for n in notes],
        usage=wire_usage(usage, stats.embedding_requests),
        timing_ms={
            "extract": _ms(extracted - started),
            "map": _ms(mapped - extracted),
            "total": _ms(mapped - started),
        },
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODELS",
    "ENV_NAMES",
    "OFFLINE_MODEL",
    "IndexView",
    "Settings",
    "SettingsError",
    "make_llm",
    "retrieval_status",
    "run",
    "scrub",
    "validate_request",
    "wire_usage",
]

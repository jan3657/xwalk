"""Settings from the environment, the LLM factory, and one request end to end.

Binding: docs/ner-site/PLAN.md sections 3.2 (environment), 3.3 (signatures) and 7
(security). ``run`` validates the request, makes the one extraction call, maps the
entities and assembles a :class:`NerResponse`; every failure it knows about leaves as a
:class:`ServiceError` with a code from section 3.1. ``ner_site.http`` turns those into
responses.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from ner_site.extract import MAX_TOKENS, TEMPERATURE, extract, render_default_prompt
from ner_site.indexes import (
    NEON_DIMENSIONS,
    NEON_SCHEMA,
    Backend,
    Catalog,
    OntologyIndex,
    summarise_status,
)
from ner_site.mapping import MapStats, map_entities
from ner_site.ontologies import FULL_ROUTES, ROUTES
from ner_site.schema import (
    Limits,
    Mode,
    NerRequest,
    NerResponse,
    RetrieverStatus,
    ServiceError,
    Usage,
)
from ner_site.select import (
    JEV,
    JEV_MODEL,
    JEV_RETRIES,
    JEV_TIMEOUT_S,
    JEV_URL,
    RETRIEVAL_ONLY,
    SelectStats,
    select_entities,
)
from xwalk.decide.base import DecisionClient
from xwalk.decide.budget import BudgetedDecider
from xwalk.decide.fake import FakeDecider
from xwalk.decide.jev import JevClient
from xwalk.fingerprint import redact_url
from xwalk.llm.base import LLMClient, LLMError, LLMRequest
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
# What may pick the match among an entity's retrieved candidates (ner_site.select), after
# "retrieval" (the fused ranking alone), which is always offered. Each one answered live
# requests on 4 November 2026 (cheap, recent, structured output); DeepSeek V4.1 Flash
# returned non-JSON once and Qwen3.7 Flash had no endpoint under the account's data
# policy, so they are left out.
DEFAULT_SELECTORS = (
    JEV,
    "openai/gpt-6-luna",
    "anthropic/claude-haiku-5.5",
    "google/gemini-3.1-flash-lite",
)
SELECTOR_CALLS_PER_ENTITY = 4  # Jev: one screen chunk, choose, gate, and one spare
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
# The selection step after mapping gets what is left of REQUEST_DEADLINE_S, at least
# MIN_SELECT_BUDGET_S; worst case the request then ends by about 110 s.
MIN_SELECT_BUDGET_S = 5.0

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
    "NER_SITE_BACKEND",
    "DATABASE_URL",
    "NER_SITE_NEON_SCHEMA",
    "NER_SITE_NEON_DIMENSIONS",
    "NER_SITE_NEON_CATALOG",
    "NER_SITE_SELECTORS",
    "NER_SITE_SELECTOR",
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
    backend: Backend = "bundled"
    database_url: str | None = field(default=None, repr=False)
    neon_schema: str = NEON_SCHEMA
    neon_dimensions: int = NEON_DIMENSIONS
    catalog: Catalog = "samples"


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
    backend: Backend = "bundled"
    database_url: str | None = field(default=None, repr=False)
    neon_schema: str = NEON_SCHEMA
    neon_dimensions: int = NEON_DIMENSIONS
    catalog: Catalog = "samples"
    selectors: tuple[str, ...] = DEFAULT_SELECTORS
    selector: str = RETRIEVAL_ONLY

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

        backend = get("NER_SITE_BACKEND").lower() or "bundled"
        if backend not in {"bundled", "neon"}:
            raise SettingsError("NER_SITE_BACKEND must be bundled or neon")
        schema = get("NER_SITE_NEON_SCHEMA") or NEON_SCHEMA
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", schema):
            raise SettingsError(
                "NER_SITE_NEON_SCHEMA must be a lower-case identifier (letters, digits, _)"
            )
        raw_dims = get("NER_SITE_NEON_DIMENSIONS")
        if raw_dims and (
            not re.fullmatch(r"[0-9]{1,4}", raw_dims) or not 1 <= int(raw_dims) <= 4000
        ):
            raise SettingsError("NER_SITE_NEON_DIMENSIONS must be a whole number from 1 to 4000")
        catalog = get("NER_SITE_NEON_CATALOG").lower() or "samples"
        if catalog not in {"samples", "full"}:
            raise SettingsError("NER_SITE_NEON_CATALOG must be samples or full")
        if catalog == "full" and backend != "neon":
            raise SettingsError("NER_SITE_NEON_CATALOG=full needs NER_SITE_BACKEND=neon")
        # Read only for the Neon backend: a bundled deployment ignores whatever a
        # platform integration may have put in DATABASE_URL.
        database_url = (get("DATABASE_URL") or None) if backend == "neon" else None
        if database_url is not None and urlsplit(database_url).scheme not in {
            "postgres",
            "postgresql",
        }:
            raise SettingsError("DATABASE_URL must be a postgres:// or postgresql:// URL")

        raw_selectors = get("NER_SITE_SELECTORS")
        selectors = (
            tuple(dict.fromkeys(m.strip() for m in raw_selectors.split(",") if m.strip()))
            if raw_selectors
            else DEFAULT_SELECTORS
        )
        selectors = tuple(m for m in selectors if m != RETRIEVAL_ONLY)
        selector = get("NER_SITE_SELECTOR") or RETRIEVAL_ONLY
        offered = (RETRIEVAL_ONLY, JEV) if offline_raw in _TRUE else (RETRIEVAL_ONLY, *selectors)
        if selector not in offered:
            raise SettingsError(
                f"NER_SITE_SELECTOR must be one of {', '.join(offered)} (NER_SITE_SELECTORS)"
            )

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
            backend="neon" if backend == "neon" else "bundled",
            database_url=database_url,
            neon_schema=schema,
            neon_dimensions=int(raw_dims) if raw_dims else NEON_DIMENSIONS,
            catalog="full" if catalog == "full" else "samples",
            selectors=selectors,
            selector=selector,
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
    def offered_selectors(self) -> tuple[str, ...]:
        """What may pick the match: ``retrieval`` (no selector) first, then Jev and the
        LLMs. Offline, Jev is a token-overlap stand-in and no LLM is offered."""
        if self.offline:
            return (RETRIEVAL_ONLY, JEV)
        return (RETRIEVAL_ONLY, *self.selectors)

    @property
    def default_selector(self) -> str:
        return self.selector

    @property
    def default_prompt(self) -> str:
        """The extraction prompt used when a request sends none, stating this
        deployment's entity limit; ``/api/health`` serves it."""
        return render_default_prompt(self.limits.max_entities, types=tuple(self.routes))

    @property
    def routes(self) -> dict[str, str]:
        """Entity type -> ontology slug for this deployment's catalog."""
        return FULL_ROUTES if self.catalog == "full" else ROUTES

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
            backend=self.backend,
            database_url=self.database_url,
            neon_schema=self.neon_schema,
            neon_dimensions=self.neon_dimensions,
            catalog=self.catalog,
        )

    def secrets(self) -> tuple[str, ...]:
        """Values scrubbed from every response and log line as a last line of defence
        (nothing echoes them in the first place). Values shorter than
        ``MIN_SCRUBBED_CHARS`` are left alone: replacing a passcode like ``ok`` would
        corrupt ordinary JSON, and so short a value cannot be hidden anyway."""
        values: list[str | None] = [self.api_key, self.passcode, self.database_url]
        if self.database_url:
            password = urlsplit(self.database_url).password
            if password:
                values += [password, unquote(password)]
        return tuple(dict.fromkeys(s for s in values if s and len(s) >= MIN_SCRUBBED_CHARS))


def scrub(message: str, settings: Settings) -> str:
    """``message`` with the key and passcode removed and every URL redacted."""
    for secret in settings.secrets():
        message = message.replace(secret, "[redacted]")
    return re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+", lambda m: redact_url(m[0]), message)


class OpenRouterClient(OpenAICompatClient):
    """`OpenAICompatClient` that switches off OpenRouter's reasoning on every request.

    On real abstracts (4 November 2026) GPT-6 Luna and Claude Haiku 5.5 spent the whole
    2048-token output budget on hidden reasoning and returned no answer at all
    (``finish_reason: length``, ``reasoning_tokens: 2048``); with ``reasoning:
    {enabled: false}`` they answer in a few hundred tokens. Only sent to OpenRouter:
    other OpenAI-compatible endpoints do not know the field."""

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        body = super()._body(request, structured=structured)
        body.setdefault("reasoning", {"enabled": False})
        return body


def provider_profile(model: str) -> str:
    """The capability profile for `model`. Anthropic models are asked for JSON in the
    prompt, not with a strict json_schema: with reasoning off, Claude Haiku 5.5 under
    the schema returned one entity with zero offsets; in the prompt, all of them."""
    return "anthropic-compat" if model.startswith("anthropic/") else "openai"


def _client(settings: Settings, model: str, **kw: Any) -> OpenAICompatClient:
    host = urlsplit(settings.base_url).hostname or ""
    openrouter = host == "openrouter.ai" or host.endswith(".openrouter.ai")
    kind = OpenRouterClient if openrouter else OpenAICompatClient
    return kind(
        settings.base_url,
        model,
        api_key=settings.api_key,
        profile=provider_profile(model),
        extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_TITLE},
        **kw,
    )


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
    client = _client(
        settings,
        model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
        backoff_base=backoff_base,
        transport=transport,
    )
    return BudgetedLLM(client, CallBudget(limit=settings.limits.llm_call_cap))


def make_select_llm(
    settings: Settings, model: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> BudgetedLLM:
    """The LLM selector's client: one call per routed entity, capped at one per entity
    allowed (no retries beyond the client's own). The caller closes it."""
    client = _client(
        settings,
        model,
        temperature=0.0,
        max_tokens=600,
        timeout=LLM_TIMEOUT_S,
        max_retries=1,
        transport=transport,
    )
    return BudgetedLLM(client, CallBudget(limit=settings.limits.max_entities * 2))


def make_decider(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> DecisionClient:
    """Jev through OpenRouter with the site's key, capped per request; offline, the
    token-overlap stand-in. The caller closes it when it has ``aclose``."""
    if settings.offline:
        return FakeDecider(model="offline-jev")
    client = JevClient(
        JEV_URL,
        JEV_MODEL,
        api_key=settings.api_key,
        timeout=JEV_TIMEOUT_S,
        max_retries=JEV_RETRIES,
        transport=transport,
    )
    budget = CallBudget(limit=settings.limits.max_entities * SELECTOR_CALLS_PER_ENTITY)
    return BudgetedDecider(client, budget)


# /api/health asks OpenRouter whether the key still works (GET /api/v1/key: free, no
# model call), remembering the answer; an expired key otherwise shows "llm ok" while every
# POST /api/ner fails with 502 (4 November 2026).
KEY_CHECK_URL = "https://openrouter.ai/api/v1/key"
KEY_CHECK_TIMEOUT_S = 5.0
KEY_OK_TTL_S = 300.0
KEY_REFUSED_TTL_S = 60.0
_KEY_CACHE: dict[str, tuple[float, RetrieverStatus]] = {}
_KEY_LOCK = threading.Lock()


def reset_key_check() -> None:
    with _KEY_LOCK:
        _KEY_CACHE.clear()


def _key_status(settings: Settings, transport: httpx.BaseTransport | None) -> RetrieverStatus:
    try:
        with httpx.Client(timeout=KEY_CHECK_TIMEOUT_S, transport=transport) as client:
            response = client.get(
                KEY_CHECK_URL, headers={"Authorization": f"Bearer {settings.api_key}"}
            )
    except httpx.HTTPError as exc:
        return RetrieverStatus(
            status="ok", reason=f"key not checked (OpenRouter unreachable: {type(exc).__name__})"
        )
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    if response.status_code in (401, 403):
        error = payload.get("error") if isinstance(payload, dict) else None
        detail = error.get("message") if isinstance(error, dict) else None
        reason = (
            f"OPENROUTER_API_KEY was refused by OpenRouter (HTTP {response.status_code}"
            + (f": {str(detail)[:120]}" if detail else "")
            + "); POST /api/ner answers 502 until the key is replaced"
        )
        return RetrieverStatus(status="unavailable", reason=scrub(reason, settings))
    if response.status_code != 200:
        return RetrieverStatus(
            status="ok", reason=f"key not checked (OpenRouter answered HTTP {response.status_code})"
        )
    data = payload.get("data") if isinstance(payload, dict) else None
    remaining = data.get("limit_remaining") if isinstance(data, dict) else None
    if isinstance(remaining, (int, float)) and remaining <= 0:
        return RetrieverStatus(
            status="unavailable",
            reason="the OpenRouter key's spending limit is used up; raise it or replace the key",
        )
    return RetrieverStatus(status="ok")


def check_key(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> RetrieverStatus | None:
    """Whether OpenRouter accepts ``OPENROUTER_API_KEY`` (None without a key). A refused
    or spent key is ``unavailable`` with OpenRouter's reason; a check that could not reach
    OpenRouter is ``ok`` with a "not checked" reason, never an outage. Remembered for
    ``KEY_OK_TTL_S`` (``KEY_REFUSED_TTL_S`` when refused), per key."""
    if not settings.api_key:
        return None
    slot = hashlib.sha256(settings.api_key.encode()).hexdigest()
    now = time.monotonic()
    with _KEY_LOCK:
        cached = _KEY_CACHE.get(slot)
        if cached is not None and cached[0] > now:
            return cached[1]
    status = _key_status(settings, transport)
    ttl = KEY_OK_TTL_S if status.status == "ok" else KEY_REFUSED_TTL_S
    with _KEY_LOCK:
        _KEY_CACHE[slot] = (now + ttl, status)
    return status


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
    selector = request.selector if request.selector is not None else settings.default_selector
    if selector not in settings.offered_selectors:
        raise _bad(
            "unknown_selector",
            f"selector {_shown(selector)} is not offered; choose one of "
            f"{', '.join(settings.offered_selectors)}",
        )
    return NerRequest(
        text=request.text, prompt=prompt, model=model, top_k=request.top_k, selector=selector
    )


def wire_usage(
    usage: XwalkUsage, embedding_requests: int, selected: SelectStats | None = None
) -> Usage:
    """Tokens are reported only when every call reported them; otherwise null, never 0.
    No call at all (the offline stand-in) reports no tokens either. The same holds for
    the selector's tokens and the cost its provider reported."""
    known = usage.calls > 0 and usage.unknown_calls == 0
    picked = selected if selected is not None and selected.calls > 0 else None
    return Usage(
        llm_calls=usage.calls,
        prompt_tokens=usage.prompt_tokens if known else None,
        completion_tokens=usage.completion_tokens if known else None,
        embedding_requests=embedding_requests,
        selector_calls=0 if selected is None else selected.calls,
        selector_prompt_tokens=None if picked is None else picked.prompt_tokens,
        selector_completion_tokens=None if picked is None else picked.completion_tokens,
        selector_cost_usd=None if picked is None else picked.reported_cost,
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
    decider: DecisionClient | None = None,
    select_llm: LLMClient | None = None,
) -> NerResponse:
    """One request: validate, extract with ``llm`` (one call), map, select, report.

    The selection step runs when the request's selector is not ``retrieval``: Jev with
    ``decider``, any other selector with ``select_llm`` (``ner_site.http`` builds them
    with :func:`make_decider` and :func:`make_select_llm`).

    ``started_at`` (``time.monotonic()``) is when the request began, if earlier than
    this call (opening the indexes on a cold start); the mapping deadline counts from it.

    Raises :class:`ServiceError`: 400 codes from :func:`validate_request`, 422 from the
    extraction, 429 ``call_budget_exhausted``, 502 ``provider_error``."""
    req = validate_request(request, settings)
    assert req.prompt is not None and req.model is not None
    started = time.monotonic()
    try:
        entities, warnings, usage = await asyncio.wait_for(
            extract(
                llm,
                req.text,
                req.prompt,
                max_entities=settings.limits.max_entities,
                types=(*settings.routes, "other"),
            ),
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
        routes=settings.routes,
    )
    mapped = time.monotonic()

    selector = req.selector or RETRIEVAL_ONLY
    picked: SelectStats | None = None
    if selector != RETRIEVAL_ONLY:
        if (selector == JEV and decider is None) or (selector != JEV and select_llm is None):
            raise ServiceError("not_configured", f"selector {selector!r} has no client", status=503)
        picked = SelectStats()
        results = await select_entities(
            results,
            stats.candidates,
            indexes,
            selector=selector,
            text=req.text,
            stats=picked,
            decider=decider,
            llm=select_llm,
            deadline=max(MIN_SELECT_BUDGET_S, REQUEST_DEADLINE_S - (mapped - began)),
        )
        results = [
            r.model_copy(
                update={
                    "pick": r.pick.model_copy(
                        update={"explanation": _clean(r.pick.explanation, settings)}
                    )
                }
            )
            if r.pick is not None
            else r
            for r in results
        ]
    selected = time.monotonic()

    retrieval = retrieval_status(indexes, settings)
    notes: list[str] = list(warnings) + stats.warnings
    _dense_outcome(retrieval, stats, notes)
    routed = any(r.ontology is not None for r in results)
    for kind in ("bm25", "dense"):
        status = retrieval.get(kind)
        if routed and status is not None and status.status != "ok":
            fallback = "BM25 only" if kind == "dense" else "dense only"
            notes.append(f"{kind} retrieval unavailable; matches use {fallback}")
    timing = {"extract": _ms(extracted - started), "map": _ms(mapped - extracted)}
    if picked is not None:
        timing["select"] = _ms(selected - mapped)
    timing["total"] = _ms(selected - started)
    return NerResponse(
        model=req.model,
        mode=settings.mode,
        selector=selector,
        entities=results,
        retrieval=retrieval,
        warnings=[scrub(n, settings) for n in notes],
        usage=wire_usage(usage, stats.embedding_requests, picked),
        timing_ms=timing,
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODELS",
    "ENV_NAMES",
    "OFFLINE_MODEL",
    "IndexView",
    "Settings",
    "SettingsError",
    "check_key",
    "make_decider",
    "make_llm",
    "make_select_llm",
    "reset_key_check",
    "retrieval_status",
    "run",
    "scrub",
    "validate_request",
    "wire_usage",
]

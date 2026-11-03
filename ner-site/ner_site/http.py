"""Framework-free request handling: routing, the header and passcode rules, the error
envelope, and one log line per request (docs/ner-site/PLAN.md sections 3.1 and 7).

:func:`handle` is what ``api/*.py`` (through :func:`serve`), ``tools/serve_local.py``
and the tests call. It never raises: every outcome is ``(status, headers, body)`` with a
JSON body, ``Cache-Control: no-store`` and no ``Access-Control-*`` header.

Order of checks for ``POST /api/ner``: path (404), method (405; ``OPTIONS`` too, on
purpose, so no preflight ever succeeds), body size (413), ``X-Requested-With:
xwalk-ner`` (400 ``bad_header``), passcode (401), configuration (503), JSON (400), field
limits (400), then the run (422/429/502). An unexpected exception is a 500 ``internal``
whose log line names only the exception type and where it was raised: its message could
carry the visitor's text.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from ner_site import fake
from ner_site.extract import DEFAULT_PROMPT
from ner_site.indexes import get_indexes
from ner_site.ontologies import catalog
from ner_site.schema import (
    HealthResponse,
    NerRequest,
    OntologySummary,
    RetrieverStatus,
    ServiceError,
)
from ner_site.service import (
    ENV_NAMES,
    Settings,
    SettingsError,
    make_llm,
    retrieval_status,
    run,
    validate_request,
)
from xwalk.llm.base import LLMClient

Response = tuple[int, dict[str, str], bytes]
LLMFactory = Callable[[Settings, str], LLMClient]

HEALTH_PATH = "/api/health"
NER_PATH = "/api/ner"
ALLOWED = {HEALTH_PATH: ("GET", "HEAD"), NER_PATH: ("POST",)}
REQUESTED_WITH = "xwalk-ner"
MAX_LOGGED_PATH = 80


def _headers(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        **(extra or {}),
    }


def _json_body(data: object, settings: Settings | None) -> bytes:
    text = json.dumps(data, ensure_ascii=False)
    if settings is not None:  # last line of defence: no secret ever leaves
        for secret in settings.secrets():
            text = text.replace(secret, "[redacted]")
    return text.encode("utf-8")


def _error(
    exc: ServiceError, settings: Settings | None, extra: Mapping[str, str] | None = None
) -> Response:
    return exc.status, _headers(extra), _json_body(exc.envelope(), settings)


def _log(record: dict[str, Any], settings: Settings | None) -> None:
    line = json.dumps({"event": "ner_site.request", **record}, ensure_ascii=True)
    if settings is not None:
        for secret in settings.secrets():
            line = line.replace(secret, "[redacted]")
    print(line, file=sys.stderr, flush=True)


def default_llm_factory(settings: Settings, model: str) -> LLMClient:
    """The offline stand-in in offline mode, otherwise the budgeted OpenRouter client."""
    if settings.offline:
        return fake.offline_llm()
    return make_llm(settings, model)


def _passcode_error(settings: Settings, headers: Mapping[str, str]) -> ServiceError | None:
    if settings.passcode is None:
        return None
    given = headers.get("x-passcode", "")
    if not given:
        return ServiceError("passcode_required", "this demo needs a passcode", status=401)
    if not hmac.compare_digest(given.encode("utf-8"), settings.passcode.encode("utf-8")):
        return ServiceError("passcode_invalid", "the passcode is not correct", status=401)
    return None


def _parse(body: bytes) -> NerRequest:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ServiceError(
            "bad_json", "the request body is not valid UTF-8 JSON", status=400
        ) from None
    if not isinstance(data, dict):
        raise ServiceError("bad_json", "the request body must be a JSON object", status=400)
    try:
        return NerRequest.model_validate(data, strict=True)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = str(first["loc"][0]) if first["loc"] else ""
        if field == "text":
            raise ServiceError(
                "missing_text", "'text' is required and must be a string", status=400
            ) from None
        if field == "top_k":
            raise ServiceError("bad_top_k", "'top_k' must be an integer", status=400) from None
        if first["type"] == "extra_forbidden":
            message = f"unknown field {field[:40]!r}; send text, prompt, model and top_k"
        else:
            message = f"'{field}' must be a string or null"
        raise ServiceError("bad_json", message, status=400) from None


def _health(settings: Settings) -> dict[str, Any]:
    indexes = get_indexes(settings.index_settings())
    health = HealthResponse(
        mode=settings.mode,
        default_model=settings.default_model,
        models=list(settings.offered_models),
        limits=settings.limits.public(),
        passcode_required=settings.passcode is not None,
        default_prompt=DEFAULT_PROMPT,
        ontologies=[OntologySummary.model_validate(o) for o in catalog()],
        retrieval=retrieval_status(indexes, settings),
    )
    data: dict[str, Any] = health.model_dump(mode="json")
    # Not in HealthResponse: what an operator needs to see when /api/ner would be 503.
    if settings.offline:
        llm = RetrieverStatus(status="ok", reason="offline stand-in extractor (NER_SITE_OFFLINE=1)")
    elif settings.api_key is None:
        llm = RetrieverStatus(
            status="unavailable",
            reason="OPENROUTER_API_KEY is not set; POST /api/ner answers 503 not_configured",
        )
    else:
        llm = RetrieverStatus(status="ok")
    data["llm"] = llm.model_dump(mode="json")
    return data


async def _ner(
    settings: Settings, request: NerRequest, factory: LLMFactory, log: dict[str, Any]
) -> dict[str, Any]:
    valid = validate_request(request, settings)
    assert valid.model is not None
    log["model"] = valid.model
    log["top_k"] = valid.top_k
    indexes = get_indexes(settings.index_settings())
    llm = factory(settings, valid.model)
    try:
        response = await run(valid, settings=settings, llm=llm, indexes=indexes)
    finally:
        close = getattr(llm, "aclose", None)
        if callable(close):
            await close()
    log.update(
        entities=len(response.entities),
        matches=sum(len(e.matches) for e in response.entities),
        warnings=len(response.warnings),
        llm_calls=response.usage.llm_calls,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        embedding_requests=response.usage.embedding_requests,
        dense=response.retrieval["dense"].status if "dense" in response.retrieval else None,
        timing_ms=response.timing_ms,
    )
    return response.model_dump(mode="json")


def handle(
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    *,
    settings: Settings,
    llm_factory: LLMFactory | None = None,
) -> Response:
    """One request. ``headers`` keys in any case; ``path`` with or without a query."""
    started = time.monotonic()
    method = method.upper()
    path = urlsplit(path).path or "/"
    lowered = {k.lower(): v for k, v in headers.items()}
    log: dict[str, Any] = {
        "method": method,
        "path": path[:MAX_LOGGED_PATH],
        "mode": settings.mode,
        "body_bytes": len(body),
    }
    try:
        result = _dispatch(method, path, lowered, body, settings, llm_factory, log)
    except ServiceError as exc:
        log["code"] = exc.code
        allow = ALLOWED.get(path) if exc.code == "method_not_allowed" else None
        result = _error(exc, settings, {"Allow": ", ".join(allow)} if allow else None)
    except Exception as exc:  # noqa: BLE001 - a 500 with no detail; type and place to the log
        frames = traceback.extract_tb(exc.__traceback__)
        log["code"] = "internal"
        log["exception"] = type(exc).__name__
        log["where"] = [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames]
        result = _error(
            ServiceError("internal", "internal error; see the function log", status=500),
            settings,
        )
    log["status"] = result[0]
    log.setdefault("timing_ms", {})["handle"] = round((time.monotonic() - started) * 1000)
    _log(log, settings)
    return result


def _dispatch(
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    settings: Settings,
    llm_factory: LLMFactory | None,
    log: dict[str, Any],
) -> Response:
    allowed = ALLOWED.get(path)
    if allowed is None:
        raise ServiceError("not_found", f"no such endpoint: {path[:MAX_LOGGED_PATH]}", status=404)
    if method not in allowed:
        raise ServiceError(
            "method_not_allowed",
            f"{method[:16]} is not allowed on {path}; use {' or '.join(allowed)}",
            status=405,
        )
    if path == HEALTH_PATH:
        return 200, _headers(), _json_body(_health(settings), settings)

    if len(body) > settings.limits.max_body_bytes:
        raise ServiceError(
            "body_too_large",
            f"the request body is larger than {settings.limits.max_body_bytes} bytes",
            status=413,
        )
    if headers.get("x-requested-with") != REQUESTED_WITH:
        raise ServiceError(
            "bad_header", f"missing header X-Requested-With: {REQUESTED_WITH}", status=400
        )
    passcode = _passcode_error(settings, headers)
    if passcode is not None:
        raise passcode
    if not settings.configured:
        raise ServiceError(
            "not_configured",
            "the service has no OPENROUTER_API_KEY; the operator must set it "
            "(or NER_SITE_OFFLINE=1 for the stand-in extractor)",
            status=503,
        )
    request = _parse(body)
    log["text_chars"] = len(request.text)
    log["custom_prompt"] = bool(request.prompt and request.prompt.strip())
    data = asyncio.run(_ner(settings, request, llm_factory or default_llm_factory, log))
    return 200, _headers(), _json_body(data, settings)


# --- Vercel: settings from the process environment, and the shim's I/O -----------

_SETTINGS_LOCK = threading.Lock()
_SETTINGS: dict[tuple[str | None, ...], Settings | SettingsError] = {}


def env_settings(env: Mapping[str, str] | None = None) -> Settings | SettingsError:
    """``Settings.from_env`` for the process environment, read once per distinct set of
    values (a warm instance reuses it); a bad value is returned, not raised."""
    source = os.environ if env is None else env
    key = tuple(source.get(name) for name in ENV_NAMES)
    with _SETTINGS_LOCK:
        found = _SETTINGS.get(key)
        if found is None:
            try:
                found = Settings.from_env(source)
            except SettingsError as exc:
                found = exc
            _SETTINGS[key] = found
        return found


def reset() -> None:
    with _SETTINGS_LOCK:
        _SETTINGS.clear()


def _shim_response(request: BaseHTTPRequestHandler, settings: Settings) -> Response:
    """Read the body (at most Content-Length bytes, nothing when it is over the cap)
    and handle the request; refusals before reading are logged like any other."""
    method = request.command.upper()
    cap = settings.limits.max_body_bytes
    raw_length = request.headers.get("Content-Length")
    early: ServiceError | None = None
    length = 0
    if raw_length is None:
        if method == "POST":
            early = ServiceError(
                "length_required", "a POST needs a Content-Length header", status=411
            )
    elif not raw_length.strip().isdigit():
        early = ServiceError("bad_header", "Content-Length is not a valid length", status=400)
    else:
        length = int(raw_length)
        if length > cap:
            early = ServiceError(
                "body_too_large", f"the request body is larger than {cap} bytes", status=413
            )
    if early is not None:
        request.close_connection = True
        _log(
            {
                "method": method,
                "path": urlsplit(request.path).path[:MAX_LOGGED_PATH],
                "status": early.status,
                "code": early.code,
                "body_bytes": length,
            },
            settings,
        )
        return _error(early, settings)
    body = request.rfile.read(length) if length else b""
    headers = {k.lower(): v for k, v in request.headers.items()}
    return handle(method, request.path, headers, body, settings=settings)


def serve(request: BaseHTTPRequestHandler) -> None:
    """Answer one request on a ``BaseHTTPRequestHandler`` (``api/*.py``)."""
    settings = env_settings()
    if isinstance(settings, SettingsError):
        error = ServiceError(
            "not_configured", f"the service is misconfigured: {settings}", status=503
        )
        _log(
            {
                "method": request.command,
                "path": urlsplit(request.path).path[:MAX_LOGGED_PATH],
                "status": 503,
                "code": "not_configured",
            },
            None,
        )
        status, headers, body = _error(error, None)
    else:
        status, headers, body = _shim_response(request, settings)
    request.send_response(status)
    for key, value in headers.items():
        request.send_header(key, value)
    request.send_header("Content-Length", str(len(body)))
    if request.close_connection:
        request.send_header("Connection", "close")
    request.end_headers()
    if request.command.upper() != "HEAD":
        request.wfile.write(body)


__all__ = ["default_llm_factory", "env_settings", "handle", "reset", "serve"]

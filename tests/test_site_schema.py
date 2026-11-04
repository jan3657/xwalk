"""The NER demo site's wire contract (docs/ner-site/PLAN.md section 3).

The JSON examples below are the ones in the plan, verbatim (hence the long lines). A
response model that renames, drops, adds or reorders a field fails the round-trip.
"""

# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from ner_site.schema import (
    Entity,
    EntityResult,
    HealthResponse,
    Limits,
    Match,
    NerRequest,
    NerResponse,
    RetrieverStatus,
    ServiceError,
    Usage,
)
from pydantic import ValidationError

NER_RESPONSE = """
{"ok": true, "model": "openai/gpt-4o-mini", "mode": "live",
 "entities": [
   {"id": "e1", "text": "Aspirin", "start": 0, "end": 7, "type": "chemical", "ontology": "chebi-sample",
    "matches": [{"id": "CHEBI:15365", "label": "acetylsalicylic acid", "score": 0.0325,
                 "retrievers": {"bm25": 1, "dense-qwen3": 2},
                 "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365"}],
    "notes": []},
   {"id": "e2", "text": "fever", "start": 16, "end": 21, "type": "other", "ontology": null,
    "matches": [], "notes": ["no ontology for type 'other'"]}],
 "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
 "warnings": ["entity 'feverr' dropped: not found in text"],
 "usage": {"llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 88, "embedding_requests": 2},
 "timing_ms": {"extract": 1800, "map": 900, "total": 2800}}
"""

HEALTH_RESPONSE = """
{"ok": true, "mode": "live", "default_model": "openai/gpt-4o-mini",
 "models": ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"],
 "limits": {"max_chars": 5000, "max_prompt_chars": 4000, "max_entities": 25, "top_k_max": 10},
 "passcode_required": false, "default_prompt": "...",
 "ontologies": [{"slug": "chebi-sample", "name": "ChEBI (sample)", "type": "chemical", "count": 200}],
 "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "unavailable", "reason": "no index at indexes/chebi-sample/dense; run ner-site/tools/build_indexes.py"}}}
"""


def _round_trip(model: Any, raw: str) -> None:
    data = json.loads(raw)
    dumped = model.model_validate(data).model_dump(mode="json", exclude_none=False)
    assert json.dumps(dumped) == json.dumps(data)


def test_the_plan_s_ner_response_round_trips_unchanged() -> None:
    _round_trip(NerResponse, NER_RESPONSE)


def test_the_plan_s_health_response_round_trips_unchanged() -> None:
    _round_trip(HealthResponse, HEALTH_RESPONSE)


def test_the_response_types_are_the_nested_models() -> None:
    response = NerResponse.model_validate(json.loads(NER_RESPONSE))
    assert isinstance(response.entities[0], EntityResult)
    assert isinstance(response.entities[0], Entity)
    assert isinstance(response.entities[0].matches[0], Match)
    assert isinstance(response.retrieval["dense"], RetrieverStatus)
    assert isinstance(response.usage, Usage)


def test_unknown_token_counts_are_none_not_zero() -> None:
    usage = Usage(llm_calls=1, embedding_requests=0)
    assert usage.prompt_tokens is None and usage.completion_tokens is None
    assert usage.model_dump(mode="json")["prompt_tokens"] is None


def test_an_unavailable_retriever_without_a_reason_dumps_only_its_status() -> None:
    """3.1 writes `{"status": "ok"}` with no `reason` key."""
    assert RetrieverStatus(status="ok").model_dump(mode="json") == {"status": "ok"}
    assert RetrieverStatus(status="unavailable", reason="r").model_dump(mode="json") == {
        "status": "unavailable",
        "reason": "r",
    }


def test_ner_request_defaults() -> None:
    request = NerRequest.model_validate({"text": "Aspirin"})
    assert (request.prompt, request.model, request.top_k) == (None, None, 5)


def test_the_plan_s_ner_request_validates() -> None:
    raw = '{"text": "Aspirin reduced fever in rheumatoid arthritis.", "prompt": null, "model": null, "top_k": 5}'
    assert NerRequest.model_validate_json(raw).text.startswith("Aspirin")


def test_an_unknown_request_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NerRequest.model_validate({"text": "x", "temperature": 2})


def test_an_unknown_entity_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Entity(id="e1", text="x", start=0, end=1, type="protein")  # type: ignore[arg-type]


def test_limits_defaults() -> None:
    limits = Limits()
    assert (
        limits.max_chars,
        limits.max_prompt_chars,
        limits.max_entities,
        limits.top_k_max,
        limits.max_body_bytes,
        limits.llm_call_cap,
    ) == (5000, 4000, 25, 10, 65536, 3)


def test_limits_are_frozen() -> None:
    limits = Limits()
    with pytest.raises((ValidationError, dataclasses.FrozenInstanceError, TypeError)):
        limits.max_chars = 1  # type: ignore[misc]


def test_service_error_carries_code_message_and_status() -> None:
    error = ServiceError("text_too_long", "text is 7000 characters; the limit is 5000", status=400)
    assert isinstance(error, Exception)
    assert (error.code, error.message, error.status) == (
        "text_too_long",
        "text is 7000 characters; the limit is 5000",
        400,
    )
    assert str(error) == "text is 7000 characters; the limit is 5000"


def test_service_error_envelope_matches_the_plan() -> None:
    error = ServiceError("text_too_long", "text is 7000 characters; the limit is 5000", status=400)
    expected = '{"ok": false, "error": {"code": "text_too_long", "message": "text is 7000 characters; the limit is 5000"}}'
    assert error.envelope() == json.loads(expected)


def test_limits_public_is_the_health_subset() -> None:
    assert Limits().public() == json.loads(HEALTH_RESPONSE)["limits"]

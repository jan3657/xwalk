"""Wire contract of the NER demo site: request, response and error shapes.

Binding: docs/ner-site/PLAN.md section 3. Every response model serialises with
``model_dump(mode="json")`` to exactly the shapes below.

``GET /api/health`` -> 200::

    {"ok": true, "mode": "live", "default_model": "openai/gpt-4o-mini",
     "models": ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"],
     "limits": {"max_chars": 5000, "max_prompt_chars": 4000, "max_entities": 25, "top_k_max": 10},
     "passcode_required": false, "default_prompt": "...",
     "ontologies": [{"slug": "chebi-sample", "name": "ChEBI (sample)", "type": "chemical",
                     "count": 200}],
     "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "unavailable",
       "reason": "no index at indexes/chebi-sample/dense; run ner-site/tools/build_indexes.py"}},
     "selectors": ["retrieval", "jev", "openai/gpt-6-luna"], "default_selector": "jev"}

``POST /api/ner`` request::

    {"text": "Aspirin reduced fever in rheumatoid arthritis.", "prompt": null, "model": null,
     "top_k": 5}

200 response::

    {"ok": true, "model": "openai/gpt-4o-mini", "mode": "live", "selector": "jev",
     "entities": [
       {"id": "e1", "text": "Aspirin", "start": 0, "end": 7, "type": "chemical",
        "ontology": "chebi-sample",
        "matches": [{"id": "CHEBI:15365", "label": "acetylsalicylic acid", "score": 0.0325,
                     "retrievers": {"bm25": 1, "dense-qwen3": 2},
                     "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365"}],
        "notes": [],
        "pick": {"selector": "jev", "status": "match", "id": "CHEBI:15365",
                 "label": "acetylsalicylic acid", "url": "https://www.ebi.ac.uk/...",
                 "confidence": 0.93, "explanation": "rubric 0.95, properties 0.97"},
        "more": [{"id": "CHEBI:46195", "label": "paracetamol", "score": 0.0161, ...}]},
       {"id": "e2", "text": "fever", "start": 16, "end": 21, "type": "other", "ontology": null,
        "matches": [], "notes": ["no ontology for type 'other'"], "pick": null,
        "more": []}],
     "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
     "warnings": ["entity 'feverr' dropped: not found in text"],
     "usage": {"llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 88,
               "embedding_requests": 2},
     "timing_ms": {"extract": 1800, "map": 900, "select": 1400, "total": 4200}}

    ``usage`` also carries ``selector_calls``, ``selector_prompt_tokens``,
    ``selector_completion_tokens`` and ``selector_cost_usd`` (PLAN.md section 13).

Token counts are ``null`` when the provider did not report them, never 0.

Error envelope (see :meth:`ServiceError.envelope`)::

    {"ok": false, "error": {"code": "text_too_long",
                            "message": "text is 7000 characters; the limit is 5000"}}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

EntityType = Literal["chemical", "disease", "gene", "food", "other"]
ENTITY_TYPES: tuple[EntityType, ...] = ("chemical", "disease", "gene", "food", "other")

Mode = Literal["live", "offline"]


class Entity(BaseModel):
    """One mention located in the visitor's text; ``text[start:end] == text`` field."""

    id: str
    text: str
    start: int
    end: int
    type: EntityType


class Match(BaseModel):
    """One ontology term for an entity. ``id`` always comes from the store."""

    id: str
    label: str
    score: float
    retrievers: dict[str, int]
    url: str | None


PickStatus = Literal["match", "review", "no_match", "failed"]


class Pick(BaseModel):
    """The selector's verdict on one entity's retrieved candidates.

    ``id`` is None unless ``status`` is "match" or "review", and is always one of the
    candidates retrieved for the entity (an issued key resolved by exact lookup, or
    Jev's choice among them); a model's raw text never becomes an id."""

    selector: str
    status: PickStatus
    id: str | None = None
    label: str | None = None
    url: str | None = None
    confidence: float | None = None
    explanation: str = ""


class EntityResult(Entity):
    """``matches`` is the visitor's top_k of the fused retrieval ranking; ``more`` the
    rest of it down to ``mapping.MAX_LISTED`` (30), the depth a selector chooses from."""

    ontology: str | None
    matches: list[Match]
    notes: list[str]
    pick: Pick | None = None
    more: list[Match] = []


class RetrieverStatus(BaseModel):
    """``reason`` is omitted from the JSON when it is None (``{"status": "ok"}``)."""

    status: Literal["ok", "unavailable"]
    reason: str | None = None

    @model_serializer(mode="wrap")
    def _drop_missing_reason(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("reason") is None:
            data.pop("reason", None)
        return data


class Usage(BaseModel):
    llm_calls: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    embedding_requests: int
    # The selection step (Jev or an LLM): its calls, tokens and the cost the provider
    # reported (null when any call did not report it).
    selector_calls: int = 0
    selector_prompt_tokens: int | None = None
    selector_completion_tokens: int | None = None
    selector_cost_usd: float | None = None


class NerRequest(BaseModel):
    """``prompt``/``model`` None mean the defaults that ``/api/health`` reports.

    Limits (text length, prompt length, top_k range, model allowlist) are checked by the
    service against :class:`Limits`, so each failure gets its own error code.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    prompt: str | None = None
    model: str | None = None
    top_k: int = 5
    selector: str | None = None


class NerResponse(BaseModel):
    ok: Literal[True] = True
    model: str
    mode: Mode
    selector: str = "retrieval"
    entities: list[EntityResult]
    retrieval: dict[str, RetrieverStatus]
    warnings: list[str]
    usage: Usage
    timing_ms: dict[str, int]


class OntologySummary(BaseModel):
    slug: str
    name: str
    type: EntityType
    count: int


class HealthResponse(BaseModel):
    ok: Literal[True] = True
    mode: Mode
    default_model: str
    models: list[str]
    limits: dict[str, int]  # Limits.public()
    passcode_required: bool
    default_prompt: str
    ontologies: list[OntologySummary]
    retrieval: dict[str, RetrieverStatus]
    selectors: list[str] = ["retrieval"]
    default_selector: str = "retrieval"


@dataclass(frozen=True)
class Limits:
    max_chars: int = 5000
    max_prompt_chars: int = 4000
    max_entities: int = 25
    top_k_max: int = 10
    max_body_bytes: int = 65536
    llm_call_cap: int = 3

    def public(self) -> dict[str, int]:
        """The subset ``/api/health`` reports (plan section 3.1)."""
        return {
            "max_chars": self.max_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_entities": self.max_entities,
            "top_k_max": self.top_k_max,
        }


class ServiceError(Exception):
    """A failure with an HTTP status and a code from the table in plan section 3.1.

    ``message`` is shown to the visitor: never put a key, a URL's userinfo or the
    visitor's text in it.
    """

    def __init__(self, code: str, message: str, *, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def envelope(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message}}

"""The offline stand-in for the extraction model (``NER_SITE_OFFLINE=1``).

A deterministic dictionary tagger, not a model: it finds the labels and synonyms of the
four bundled samples in the TEXT section of the request (typed by the ontology they come
from), plus a few everyday words from the page's example (``aspirin``, which the ChEBI
sample lacks, and ``fever``, typed ``other``). It answers with the same JSON object a real
model returns, so the answer goes through ``extract.parse_entities`` (offset checks,
relocation, overlap rules) exactly like a live one.

Matching: whole words only. Terms of four or more characters match case-insensitively;
shorter ones (gene symbols such as ``ADA``) only with their exact case, so ``was`` or
``her`` never become genes. Longest term first at each position, no overlaps.

It reports no token usage and zero provider calls: nothing was sent anywhere.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from ner_site.ontologies import ROUTES, load_terms
from ner_site.schema import EntityType
from xwalk.fingerprint import hash_value
from xwalk.llm.base import LLMCapabilities, LLMRequest, LLMResponse
from xwalk.records import Usage

MODEL = "offline"
TEXT_MARKER = "\n\nTEXT:\n"  # extract.build_request: "INSTRUCTIONS:\n{prompt}\n\nTEXT:\n{text}"
MIN_CASEFOLD_CHARS = 4
EXTRA_TERMS: dict[str, EntityType] = {
    "aspirin": "chemical",
    "paracetamol": "chemical",
    "ibuprofen": "chemical",
    "fever": "other",
    "headache": "other",
}


@lru_cache(maxsize=1)
def _lexicon() -> tuple[dict[str, EntityType], dict[str, EntityType]]:
    """(case-insensitive terms keyed casefolded, exact-case short terms)."""
    folded: dict[str, EntityType] = {}
    exact: dict[str, EntityType] = {}
    for entity_type, slug in ROUTES.items():
        for record in load_terms(slug):
            synonyms = record.fields.get("synonyms") or []
            names = [record.fields.get("label"), *(synonyms if isinstance(synonyms, list) else [])]
            for name in names:
                if not isinstance(name, str) or not re.search(r"[A-Za-z]", name):
                    continue
                name = name.strip()
                kind: EntityType = entity_type  # type: ignore[assignment]
                if len(name) >= MIN_CASEFOLD_CHARS:
                    folded.setdefault(name.casefold(), kind)
                elif len(name) >= 2:
                    exact.setdefault(name, kind)
    for name, kind in EXTRA_TERMS.items():
        folded.setdefault(name.casefold(), kind)
    return folded, exact


def _alternation(terms: list[str]) -> str:
    # Longest first, so at any position the longest term wins.
    ordered = sorted(terms, key=lambda t: (-len(t), t))
    return "|".join(re.escape(t) for t in ordered)


@lru_cache(maxsize=1)
def _pattern() -> re.Pattern[str]:
    folded, exact = _lexicon()
    return re.compile(
        rf"(?<!\w)(?:(?P<folded>(?i:{_alternation(list(folded))}))"
        rf"|(?P<exact>{_alternation(list(exact))}))(?!\w)"
    )


def tag(text: str) -> list[dict[str, object]]:
    """The entities the stand-in finds, as a model would list them."""
    folded, exact = _lexicon()
    found: list[dict[str, object]] = []
    for match in _pattern().finditer(text):
        mention = match.group(0)
        if match.group("folded") is not None:
            entity_type = folded[mention.casefold()]
        else:
            entity_type = exact[mention]
        found.append(
            {"text": mention, "type": entity_type, "start": match.start(), "end": match.end()}
        )
    return found


class OfflineLLM:
    """An xwalk ``LLMClient`` that tags the request's TEXT section from the lexicon."""

    @property
    def model(self) -> str:
        return MODEL

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities()

    @property
    def fingerprint(self) -> str:
        return hash_value({"adapter": "ner-site-offline", "version": 1})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        _, marker, text = request.user.partition(TEXT_MARKER)
        entities = tag(text) if marker else []
        return LLMResponse(
            text=json.dumps({"entities": entities}, ensure_ascii=False),
            usage=Usage.zero(),
            model=MODEL,
            finish_reason="stop",
        )


def offline_llm() -> OfflineLLM:
    return OfflineLLM()


__all__ = ["MODEL", "OfflineLLM", "offline_llm", "tag"]

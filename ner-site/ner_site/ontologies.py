"""The four bundled sample ontologies: catalog, terms, routing, term URLs, templates.

The JSONL comes from the installed ``xwalk.ui`` package (``xwalk/ui/ontologies/``), so
the site ships no copy of its own. Plan: docs/ner-site/PLAN.md section 5.
"""

from __future__ import annotations

import json
import re
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from xwalk.records import Record
from xwalk.sources.tabular import jsonl_source
from xwalk.templates import TemplateSet

# Entity type -> ontology slug. "other" has no ontology on purpose.
ROUTES: dict[str, str] = {
    "chemical": "chebi-sample",
    "disease": "ctd-disease-sample",
    "gene": "ncbi-gene-sample",
    "food": "foodon-sample",
}

# Every sample record has `label`, `synonyms` (a list), `definition` (often ""), and the
# gene records add `organism` (human, mouse and rat genes are mixed, so it carries
# meaning) and `tax_id`. The doc template guards anyway: a missing or null field renders
# as nothing (xwalk's templates drop nulls), and a synonym given as one string is not
# joined character by character.
DOC_TEMPLATE = (
    "{{ label }} "
    "{% if synonyms is string %}{{ synonyms }}{% else %}{{ synonyms | join(' ') }}{% endif %} "
    "{{ definition }} {{ organism }}"
)
TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ mention }}",
    doc=DOC_TEMPLATE,
    candidate="{{ id }} {{ label }}",
)

# Best-effort links to each term's own page. The id must match exactly; anything else
# (another prefix, a stray character) gets no link rather than a guessed one.
_URL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"CHEBI:(\d+)"), "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{0}"),
    (re.compile(r"MESH:([A-Z]\d+)"), "https://meshb.nlm.nih.gov/record/ui?ui={0}"),
    (re.compile(r"OMIM:(\d+)"), "https://omim.org/entry/{0}"),
    (re.compile(r"NCBIGene:(\d+)"), "https://www.ncbi.nlm.nih.gov/gene/{0}"),
    (re.compile(r"FOODON:(\d+)"), "http://purl.obolibrary.org/obo/FOODON_{0}"),
)


def _ontology_dir() -> Path:
    return Path(str(resources.files("xwalk.ui") / "ontologies"))


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, Any], ...]:
    entries = json.loads((_ontology_dir() / "index.json").read_text(encoding="utf-8"))
    return tuple(dict(e) for e in entries)


def catalog() -> list[dict[str, Any]]:
    """The routed ontologies as ``{"slug", "name", "type", "count"}`` (``OntologySummary``),
    in ``ROUTES`` order."""
    by_slug = {str(e["slug"]): e for e in _index()}
    return [
        {
            "slug": slug,
            "name": str(by_slug[slug]["name"]),
            "type": entity_type,
            "count": int(by_slug[slug]["count"]),
        }
        for entity_type, slug in ROUTES.items()
    ]


@cache
def _terms(slug: str) -> tuple[Record, ...]:
    if slug not in ROUTES.values():
        raise KeyError(f"unknown ontology {slug!r}; expected one of {sorted(ROUTES.values())}")
    return tuple(jsonl_source(_ontology_dir() / f"{slug}.jsonl", id_field="id"))


def load_terms(slug: str) -> list[Record]:
    """Every term of one routed sample. Raises KeyError for any other slug."""
    return list(_terms(slug))


def route(entity_type: str) -> str | None:
    """The ontology slug an entity type maps onto; None for "other" or anything unknown."""
    return ROUTES.get(entity_type)


def term_url(record_id: str) -> str | None:
    for pattern, template in _URL_RULES:
        found = pattern.fullmatch(record_id)
        if found:
            return template.format(found.group(1))
    return None

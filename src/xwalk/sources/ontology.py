"""Ontology record sources.

OBO is parsed here directly -- the format is line-oriented stanzas, and sixty lines of
parsing beats pulling rdflib in for it. OWL uses rdflib behind xwalk[ontology].

Both emit the same field names -- `label`, `synonyms`, `definition`, `parents`,
`obsolete` -- so a single `doc` template works against either.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from xwalk._extras import require
from xwalk.records import Record

_OBO_PURL = re.compile(r"^http://purl\.obolibrary\.org/obo/([A-Za-z0-9]+)_(.+)$")
_QUOTED = re.compile(r'^"((?:[^"\\]|\\.)*)"')

DEFAULT_LABEL_PREDICATES = ("http://www.w3.org/2000/01/rdf-schema#label",)
DEFAULT_SYNONYM_PREDICATES = (
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
)
DEFAULT_DEFINITION_PREDICATES = (
    "http://purl.obolibrary.org/obo/IAO_0000115",
    "http://www.w3.org/2004/02/skos/core#definition",
)


def curie(uri: str) -> str:
    """Compact an OBO PURL or fragment URI. Anything else passes through unchanged."""
    match = _OBO_PURL.match(uri)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri


def _obo_value(raw: str) -> str:
    """Strip a trailing ` ! comment` and surrounding whitespace."""
    return raw.split(" ! ", 1)[0].strip()


def _obo_quoted(raw: str) -> str:
    """`def:` and `synonym:` values start with a quoted string, then metadata."""
    match = _QUOTED.match(raw.strip())
    return match.group(1).replace('\\"', '"') if match else _obo_value(raw)


def obo_source(
    path: str | Path,
    *,
    id_prefix: str | None = None,
    include_obsolete: bool = False,
    encoding: str = "utf-8",
) -> Iterator[Record]:
    """Yield one Record per `[Term]` stanza. No dependencies."""
    path = Path(path)
    state: dict[str, Any] = {"stanza": None, "id": None, "fields": {}}

    def finish() -> Record | None:
        term_id = state["id"]
        fields = state["fields"]
        if state["stanza"] != "Term" or not term_id:
            return None
        if id_prefix and not term_id.startswith(id_prefix):
            return None
        if fields.get("obsolete") and not include_obsolete:
            return None
        return Record(
            id=term_id,
            fields={
                "label": fields.get("label", ""),
                "synonyms": fields.get("synonyms", []),
                "definition": fields.get("definition", ""),
                "parents": fields.get("parents", []),
                "obsolete": bool(fields.get("obsolete", False)),
            },
        )

    with path.open("r", encoding=encoding) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                record = finish()
                if record is not None:
                    yield record
                state = {"stanza": stripped[1:-1], "id": None, "fields": {}}
                continue
            if not stripped or ":" not in stripped or state["stanza"] is None:
                continue

            key, _, raw = stripped.partition(":")
            raw = raw.strip()
            fields = state["fields"]
            if key == "id":
                state["id"] = _obo_value(raw)
            elif key == "name":
                fields["label"] = _obo_value(raw)
            elif key == "def":
                fields["definition"] = _obo_quoted(raw)
            elif key == "synonym":
                fields.setdefault("synonyms", []).append(_obo_quoted(raw))
            elif key == "is_a":
                fields.setdefault("parents", []).append(_obo_value(raw))
            elif key == "is_obsolete":
                fields["obsolete"] = _obo_value(raw).lower() == "true"

    record = finish()
    if record is not None:
        yield record


def owl_source(
    path: str | Path,
    *,
    format: str | None = None,
    id_prefix: str | None = None,
    label_predicates: Sequence[str] = DEFAULT_LABEL_PREDICATES,
    synonym_predicates: Sequence[str] = DEFAULT_SYNONYM_PREDICATES,
    definition_predicates: Sequence[str] = DEFAULT_DEFINITION_PREDICATES,
    include_obsolete: bool = False,
) -> Iterator[Record]:
    """Yield one Record per owl:Class. Requires xwalk[ontology]."""
    rdflib = require("ontology", "rdflib", purpose="owl_source")

    graph = rdflib.Graph()
    graph.parse(str(path), format=format)

    rdf_type = rdflib.RDF.type
    owl_class = rdflib.OWL.Class
    subclass_of = rdflib.RDFS.subClassOf
    deprecated = rdflib.OWL.deprecated

    labels = [rdflib.URIRef(p) for p in label_predicates]
    synonyms = [rdflib.URIRef(p) for p in synonym_predicates]
    definitions = [rdflib.URIRef(p) for p in definition_predicates]

    for subject in sorted(set(graph.subjects(rdf_type, owl_class)), key=str):
        if not isinstance(subject, rdflib.URIRef):
            continue  # skip blank-node class expressions
        record_id = curie(str(subject))
        if id_prefix and not record_id.startswith(id_prefix):
            continue

        is_obsolete = any(str(o).lower() == "true" for o in graph.objects(subject, deprecated))
        if is_obsolete and not include_obsolete:
            continue

        label = next((str(o) for p in labels for o in graph.objects(subject, p)), "")
        definition = next((str(o) for p in definitions for o in graph.objects(subject, p)), "")
        synonym_values = sorted({str(o) for p in synonyms for o in graph.objects(subject, p)})
        parents = sorted(
            curie(str(o))
            for o in graph.objects(subject, subclass_of)
            if isinstance(o, rdflib.URIRef)
        )

        yield Record(
            id=record_id,
            fields={
                "label": label,
                "synonyms": synonym_values,
                "definition": definition,
                "parents": parents,
                "obsolete": is_obsolete,
            },
        )

"""Shared helpers for the one-off sample extractions under `examples/*/extract.py`.

These scripts are run by hand, once, against a local checkout of the OntoRAG paper
repository; their output is committed and the scripts are kept only so the provenance of
every committed row is checkable. They are not part of the library and nothing imports
them at runtime.

Sample slices only: at most ~200 target records and ~50 mentions per dataset, no file
over 1 MB. The full data stays in the paper repo.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

PAPER_REPO = Path("/ceph/grid/home/jd3099/projects/onto_rag_paper_version")

_OBO_BASE = "http://purl.obolibrary.org/obo/"


def read_mentions(dataset: str, limit: int) -> list[dict[str, Any]]:
    """First `limit` rows of the paper repo's test split for `dataset`."""
    path = PAPER_REPO / "data/datasets" / dataset / "test.jsonl.gz"
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def read_ontology(key: str) -> dict[str, dict[str, Any]]:
    """The paper repo's normalised ontology dump: id -> {label, synonyms, ...}."""
    path = PAPER_REPO / "data" / key / "ontology_dump.json"
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, dict[str, Any]] = json.load(handle)
    return data


def choose_targets(
    ontology: Mapping[str, dict[str, Any]],
    gold_ids: Sequence[str],
    *,
    total: int,
) -> dict[str, dict[str, Any]]:
    """Every gold target, plus deterministic distractors up to `total`.

    Distractors are drawn from the gold terms' siblings first -- a sample whose only
    wrong answers are obviously wrong makes retrieval look better than it is.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for gid in gold_ids:
        if gid in ontology:
            chosen[gid] = ontology[gid]

    parents = {p for gid in chosen for p in ontology[gid].get("parents", [])}
    siblings = sorted(
        tid
        for tid, term in ontology.items()
        if tid not in chosen and parents.intersection(term.get("parents", []))
    )
    for tid in siblings:
        if len(chosen) >= total:
            break
        chosen[tid] = ontology[tid]

    for tid in sorted(ontology):
        if len(chosen) >= total:
            break
        if tid not in chosen:
            chosen[tid] = ontology[tid]
    return dict(sorted(chosen.items()))


def write_mentions_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "mention_id": row["mention_id"],
                        "mention": row["mention"],
                        "context_left": (row.get("context_left") or "")[-200:],
                        "context_right": (row.get("context_right") or "")[:200],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_gold_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "gold_ids"])
        for row in rows:
            writer.writerow([row["mention_id"], "|".join(row.get("gold_ids") or [])])


def write_targets_table(
    targets: Mapping[str, Mapping[str, Any]],
    path: Path,
    *,
    delimiter: str = "\t",
    extra_columns: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "label", "synonyms", "definition", *extra_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        for tid, term in targets.items():
            writer.writerow(
                [
                    tid,
                    term.get("label", ""),
                    "|".join(term.get("synonyms") or []),
                    (term.get("definition") or "").replace("\n", " ").replace(delimiter, " "),
                    *[str(term.get(c, "")) for c in extra_columns],
                ]
            )


def _uri(curie: str) -> str:
    prefix, _, local = curie.partition(":")
    return f"{_OBO_BASE}{prefix}_{local}"


def _owl_class(curie: str, term: Mapping[str, Any]) -> Iterator[str]:
    yield f'  <owl:Class rdf:about="{_uri(curie)}">'
    yield f"    <rdfs:label>{escape(str(term.get('label', '')))}</rdfs:label>"
    definition = str(term.get("definition") or "").replace("\n", " ")
    if definition:
        yield f"    <obo:IAO_0000115>{escape(definition)}</obo:IAO_0000115>"
    for synonym in term.get("synonyms") or []:
        yield f"    <oboInOwl:hasExactSynonym>{escape(str(synonym))}</oboInOwl:hasExactSynonym>"
    for parent in term.get("parents") or []:
        yield f'    <rdfs:subClassOf rdf:resource="{_uri(str(parent))}"/>'
    yield "  </owl:Class>"


def write_targets_owl(targets: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    """Emit RDF/XML so the example genuinely exercises the OWL loader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
        '         xmlns:owl="http://www.w3.org/2002/07/owl#"',
        '         xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#"',
        '         xmlns:obo="http://purl.obolibrary.org/obo/">',
    ]
    for curie, term in targets.items():
        lines.extend(_owl_class(curie, term))
    lines.append("</rdf:RDF>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(name: str, targets: Mapping[str, Any], mentions: Sequence[Any], out: Path) -> None:
    print(f"{name}: {len(targets)} targets, {len(mentions)} mentions -> {out}")
    for file in sorted(out.rglob("*")):
        if file.is_file():
            print(f"  {file.name:<28} {file.stat().st_size / 1024:8.1f} KiB")

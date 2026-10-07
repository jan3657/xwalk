#!/usr/bin/env python
"""Parse the example target collections into the UI's built-in ontology samples.

Writes `src/xwalk/ui/ontologies/<slug>.jsonl` (one term per line: `id`, `label`,
`synonyms`, `definition`, plus any other column the source had) and `index.json`. The
UI ships these files, so `xwalk ui` offers parsed ontologies without rdflib. Each is the
small slice an example already carries, not the full ontology.

Needs `xwalk[ontology]` (two samples are OWL). Run from the repository root:

    python scripts/build_ui_ontologies.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xwalk.config import load_job

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "xwalk" / "ui" / "ontologies"

SAMPLES: list[dict[str, str]] = [
    {
        "slug": "chebi-sample",
        "job": "examples/chebi/job.yaml",
        "name": "ChEBI (sample)",
        "description": "Chemical Entities of Biological Interest: chemical compounds, as "
        "used by the CRAFT ChEBI example.",
        "homepage": "https://www.ebi.ac.uk/chebi/",
        "licence": "CC BY 4.0",
    },
    {
        "slug": "foodon-sample",
        "job": "examples/cafeteria_fcd/job.yaml",
        "name": "FoodOn (sample)",
        "description": "Food ontology terms: foods, ingredients and food products, as used "
        "by the cafeteria example.",
        "homepage": "https://foodon.org/",
        "licence": "CC BY 4.0",
    },
    {
        "slug": "ctd-disease-sample",
        "job": "examples/ncbi_disease/job.yaml",
        "name": "CTD diseases / MeSH (sample)",
        "description": "Disease records (MeSH and OMIM identifiers) from the CTD disease "
        "vocabulary, as used by the NCBI Disease example.",
        "homepage": "https://ctdbase.org/voc.go?type=disease",
        "licence": "see the CTD and MeSH terms of use",
    },
    {
        "slug": "ncbi-gene-sample",
        "job": "examples/nlm_gene/job.yaml",
        "name": "NCBI Gene (sample)",
        "description": "Gene records with symbols, aliases and organism, as used by the "
        "NLM-Gene example.",
        "homepage": "https://www.ncbi.nlm.nih.gov/gene",
        "licence": "public domain (NCBI)",
    },
]


def _term(record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    term: dict[str, Any] = {
        "id": record_id,
        "label": str(fields.get("label") or ""),
        "synonyms": [str(s) for s in fields.get("synonyms") or [] if str(s).strip()],
        "definition": str(fields.get("definition") or ""),
    }
    for key, value in fields.items():
        if key not in term and key not in ("parents", "obsolete") and value not in ("", None):
            term[key] = value
    return term


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for sample in SAMPLES:
        job = load_job(ROOT / sample["job"])
        terms = [_term(r.id, dict(r.fields)) for r in job.build_target_records()]
        path = OUT / f"{sample['slug']}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for term in terms:
                handle.write(json.dumps(term, ensure_ascii=False, sort_keys=True) + "\n")
        entry: dict[str, Any] = {key: value for key, value in sample.items() if key != "job"}
        entry["count"] = len(terms)
        entry["source"] = f"{sample['job']} (target collection)"
        index.append(entry)
        print(f"{sample['slug']}: {len(terms)} terms")
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

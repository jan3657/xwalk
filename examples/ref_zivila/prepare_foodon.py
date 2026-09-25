#!/usr/bin/env python
"""Flatten a pinned FoodOn OWL snapshot into an enriched xwalk CSV target."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from xwalk.records import Record
from xwalk.sources.ontology import owl_source

OUTPUT_COLUMNS = (
    "id",
    "label",
    "synonyms",
    "definition",
    "parents",
    "parent_labels",
)
SOURCE_URL = "http://purl.obolibrary.org/obo/foodon.owl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        prefix = handle.read(64_000)
    match = re.search(r"<owl:versionInfo>([^<]+)</owl:versionInfo>", prefix)
    return match.group(1).strip() if match else "unknown"


def _safe_multivalue(values: list[str]) -> str:
    # xwalk uses | as the configured separator. Preserve the text without allowing a
    # literal pipe in an ontology annotation to manufacture an extra value.
    return "|".join(value.replace("|", " / ").strip() for value in values if value.strip())


def flatten(records: list[Record]) -> list[dict[str, str]]:
    labels = {record.id: str(record.fields.get("label") or "") for record in records}
    rows: list[dict[str, str]] = []
    for record in records:
        parents = [str(value) for value in record.fields.get("parents") or []]
        parent_labels = sorted({labels[parent] for parent in parents if labels.get(parent)})
        rows.append(
            {
                "id": record.id,
                "label": labels[record.id],
                "synonyms": _safe_multivalue(
                    [str(value) for value in record.fields.get("synonyms") or []]
                ),
                "definition": str(record.fields.get("definition") or "").replace("|", " / "),
                "parents": _safe_multivalue(parents),
                "parent_labels": _safe_multivalue(parent_labels),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ref_zivila/snapshots/foodon/foodon.owl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/ref_zivila/targets/foodon.csv")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ref_zivila/manifests/foodon.json"),
    )
    args = parser.parse_args()

    records = list(owl_source(args.input, id_prefix="FOODON:", include_obsolete=False))
    rows = flatten(records)
    if not rows or any(not row["id"] or not row["label"] for row in rows):
        raise ValueError("FoodOn extraction produced blank IDs/labels or no records")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("FoodOn extraction produced duplicate IDs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "vocabulary": "FoodOn",
        "release": _release(args.input),
        "source_url": SOURCE_URL,
        "snapshot_path": str(args.input),
        "snapshot_sha256": _sha256(args.input),
        "snapshot_bytes": args.input.stat().st_size,
        "record_count": len(rows),
        "records_with_synonyms": sum(bool(row["synonyms"]) for row in rows),
        "records_with_definitions": sum(bool(row["definition"]) for row in rows),
        "records_with_parent_labels": sum(bool(row["parent_labels"]) for row in rows),
        "filters": {"id_prefix": "FOODON:", "include_obsolete": False},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} FoodOn targets to {args.out}; manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

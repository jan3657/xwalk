#!/usr/bin/env python
"""Merge xwalk mapping results back into Ref_zivila.csv preserving row order and schema."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path

# Mapping of vocabulary identifier to column name in Ref_zivila.csv
COLUMN_BY_VOCABULARY = {
    "foodon": "FOODON",
    "snomedct": "SNOMEDCT",
    "foodex": "FOODEX",
    "hansard": "HANSARD",
    "chembl": "Chembl (Maybe)",
    "umls": "UMLS (maybe)",
}

LONG_HEADER = (
    "source_id",
    "vocabulary",
    "matched_id",
    "confidence",
    "status",
    "reason",
    "explanation",
)


def load_mapping(mapping_path: Path) -> dict[str, dict[str, str]]:
    """Load xwalk mapping.csv keyed by source_id."""
    mappings: dict[str, dict[str, str]] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = (row.get("source_id") or "").strip()
            if source_id:
                mappings[source_id] = row
    return mappings


def merge(
    input_csv: Path,
    mapping_csv: Path,
    out_csv: Path,
    vocabulary: str = "foodon",
    long_out: Path | None = None,
    allow_unreviewed: bool = False,
) -> tuple[int, int]:
    """Merge mapping results into input CSV.

    Returns (total_rows, matched_rows).
    """
    vocab_key = vocabulary.lower().replace("_", "").replace("-", "")
    target_column = COLUMN_BY_VOCABULARY.get(vocab_key)
    if not target_column:
        raise ValueError(
            f"Unknown vocabulary {vocabulary!r}; expected one of {list(COLUMN_BY_VOCABULARY)}"
        )

    mappings = load_mapping(mapping_csv)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = reader.fieldnames
        if not fieldnames or target_column not in fieldnames:
            raise ValueError(f"{input_csv}: missing target column {target_column!r}")
        rows = list(reader)

    matched_count = 0
    long_rows: list[dict[str, str]] = []

    for row in rows:
        source_id = (row.get("ID") or "").strip()
        match_info = mappings.get(source_id)
        if match_info:
            status = (match_info.get("status") or "").lower()
            matched_id = (match_info.get("matched_id") or "").strip()
            # An accepted/matched result populates the cell. If allow_unreviewed is set,
            # any non-empty matched_id is populated.
            if (status in ("matched", "accepted") and matched_id) or (allow_unreviewed and matched_id):
                row[target_column] = matched_id
                matched_count += 1
            else:
                row[target_column] = ""

            if long_out:
                long_rows.append(
                    {
                        "source_id": source_id,
                        "vocabulary": vocabulary,
                        "matched_id": matched_id if row[target_column] else "",
                        "confidence": match_info.get("confidence", ""),
                        "status": status,
                        "reason": match_info.get("reason", ""),
                        "explanation": match_info.get("explanation", ""),
                    }
                )
        else:
            row[target_column] = ""
            if long_out:
                long_rows.append(
                    {
                        "source_id": source_id,
                        "vocabulary": vocabulary,
                        "matched_id": "",
                        "confidence": "",
                        "status": "unmapped",
                        "reason": "not_in_run",
                        "explanation": "",
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if long_out:
        long_out.parent.mkdir(parents=True, exist_ok=True)
        with long_out.open("w", encoding="utf-8", newline="") as long_handle:
            writer = csv.DictWriter(long_handle, fieldnames=list(LONG_HEADER))
            writer.writeheader()
            writer.writerows(long_rows)

    return len(rows), matched_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge xwalk mapping into Ref_zivila CSV")
    parser.add_argument("--input", type=Path, default=Path("data/Ref_zivila.csv"))
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/Ref_zivila_mapped.csv"))
    parser.add_argument("--long-out", type=Path, default=Path("data/ref_zivila/mapping_long.csv"))
    parser.add_argument("--vocabulary", type=str, default="foodon")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Write matched_id even if status is not 'accepted'",
    )
    args = parser.parse_args()

    total, matched = merge(
        input_csv=args.input,
        mapping_csv=args.mapping,
        out_csv=args.out,
        vocabulary=args.vocabulary,
        long_out=args.long_out,
        allow_unreviewed=args.allow_unreviewed,
    )
    print(
        f"Merged {matched}/{total} {args.vocabulary} matches into {args.out} "
        f"(long audit: {args.long_out})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

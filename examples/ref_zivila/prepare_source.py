#!/usr/bin/env python
"""Build a stable xwalk source view and a deliberately difficult FoodOn pilot."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

REQUIRED_COLUMNS = (
    "ID",
    "SHORT_NAME_SLO",
    "NAME_SLO",
    "TAG_NAME",
    "NAME_ENG",
    "FGNM",
)
OUTPUT_COLUMNS = (
    "ID",
    "mention_en",
    "short_name_slo",
    "name_slo",
    "tag_name_slo",
    "fgnm",
    "aliases",
    "curation_note",
)

# A fixed diagnostic slice, not a random accuracy sample. It covers exact/common foods,
# processing and concentration, a dish, an additive, missing English, editorial warnings,
# an animal organ, and one visibly conflicting English/Slovenian record.
PILOT_IDS = (
    "3460a7a9-b80e-4b0b-9472-866ae554c41d",  # processed cheese, fat percentage
    "96ec6499-6d4d-4d03-b142-03c347786a5b",  # shea butter
    "8c0b9771-d64c-45a6-ae1c-177798647455",  # pineapple
    "0af3efa6-3d0b-4793-8b12-84540b18ed72",  # sour cherry jam
    "53b2ad7b-2fa4-437c-935e-5166fe89c91a",  # canned/stewed, missing English
    "84327028-8be9-4c39-8f2f-ba96ab5d8c99",  # pasteurised low-fat milk
    "4813fb99-4159-42da-820f-5cae2b2a14b6",  # composite dish
    "ec953d6e-7de8-4b47-8638-da150eeb2374",  # conflicting oyster/mussel names
    "de263d49-97d9-4748-8534-0b84fdf48caa",  # citric acid
    "3f8435f5-6f96-43fc-8bd5-75ea45cc4511",  # explicit translation uncertainty
    "2189ef46-f67d-4e9f-8205-9e092698e8b0",  # explicit NE POTRDI warning
    "cc776b92-1d10-41ce-aed3-7baf9c6422c8",  # raw sheep kidney, missing English
)

_WARNING = re.compile(r"\b(?:NE\s+POTRDI|NE\s+VEM)\b|\borig\.", re.IGNORECASE)
_NE_POTRDI_PREFIX = re.compile(r"^\s*NE\s+POTRDI\s+", re.IGNORECASE)


def _text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def _retrieval_text(value: object) -> str:
    """Remove a known editorial prefix while preserving the raw field separately."""
    return _NE_POTRDI_PREFIX.sub("", _text(value)).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def prepare_row(raw: Mapping[str, object]) -> dict[str, str]:
    names = [_text(raw.get(column)) for column in REQUIRED_COLUMNS[1:5]]
    short_slo, name_slo, tag_slo, name_eng = names
    aliases = _dedupe(
        _retrieval_text(raw.get(column))
        for column in ("NAME_ENG", "TAG_NAME", "NAME_SLO", "SHORT_NAME_SLO")
    )

    notes: list[str] = []
    for column, value in zip(REQUIRED_COLUMNS[1:5], names, strict=True):
        if value and _WARNING.search(value):
            notes.append(f"{column}: {value}")
    if not name_eng:
        notes.append("NAME_ENG missing; retrieval relies on Slovenian aliases")

    return {
        "ID": _text(raw.get("ID")),
        "mention_en": name_eng,
        "short_name_slo": short_slo,
        "name_slo": name_slo,
        "tag_name_slo": tag_slo,
        "fgnm": _text(raw.get("FGNM")),
        "aliases": "|".join(aliases),
        "curation_note": " || ".join(notes),
    }


def read_and_validate(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        rows = [prepare_row(row) for row in reader]

    ids = [row["ID"] for row in rows]
    if any(not source_id for source_id in ids):
        raise ValueError(f"{path}: blank ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ID")
    if any(not row["aliases"] for row in rows):
        raise ValueError(f"{path}: at least one row has no usable name")
    if any(not row["fgnm"] for row in rows):
        raise ValueError(f"{path}: at least one row has no FGNM")
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def select_pilot(
    rows: Sequence[dict[str, str]], pilot_ids: Sequence[str] = PILOT_IDS
) -> list[dict[str, str]]:
    wanted = set(pilot_ids)
    selected = [row for row in rows if row["ID"] in wanted]
    found = {row["ID"] for row in selected}
    missing = [source_id for source_id in pilot_ids if source_id not in found]
    if missing:
        raise ValueError(f"pilot IDs absent from source: {missing}")
    if len(selected) != len(pilot_ids):
        raise ValueError("pilot selection did not preserve one row per requested ID")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/Ref_zivila.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/ref_zivila/source.csv"))
    parser.add_argument(
        "--pilot-out", type=Path, default=Path("data/ref_zivila/source_pilot.csv")
    )
    args = parser.parse_args()

    rows = read_and_validate(args.input)
    pilot = select_pilot(rows)
    write_rows(args.out, rows)
    write_rows(args.pilot_out, pilot)
    print(
        f"wrote {len(rows)} source rows to {args.out}; "
        f"wrote {len(pilot)} diagnostic pilot rows to {args.pilot_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

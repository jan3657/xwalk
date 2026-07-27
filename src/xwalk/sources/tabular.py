"""CSV / TSV / JSONL record sources. Lazy by construction."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from xwalk.records import Record

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def csv_source(
    path: str | Path,
    *,
    id_column: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    multivalue_columns: Sequence[str] = (),
    multivalue_sep: str = "|",
) -> Iterator[Record]:
    """Yield one Record per row. The id column is removed from `fields`.

    `multivalue_columns` are split on `multivalue_sep` into lists, with blank parts
    dropped — a blank cell becomes `[]`, never `[""]`.
    """
    multivalue = set(multivalue_columns)
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or id_column not in reader.fieldnames:
            raise ValueError(
                f"id_column {id_column!r} not found in {path}; columns are {reader.fieldnames!r}"
            )
        for line_no, row in enumerate(reader, start=2):  # header is line 1
            raw_id = (row.pop(id_column) or "").strip()
            if not raw_id:
                raise ValueError(f"{path}: row {line_no} has a blank {id_column!r}")
            fields: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue  # extra columns beyond the header
                text = value or ""
                if key in multivalue:
                    fields[key] = [p.strip() for p in text.split(multivalue_sep) if p.strip()]
                else:
                    fields[key] = text.strip()
            yield Record(id=raw_id, fields=fields)


def jsonl_source(
    path: str | Path,
    *,
    id_field: str,
    encoding: str = "utf-8",
) -> Iterator[Record]:
    """Yield one Record per non-blank line. The id field is removed from `fields`."""
    path = Path(path)
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no} is not valid JSON: {exc}") from exc
            if id_field not in obj:
                raise ValueError(f"{path}: line {line_no} has no {id_field!r}")
            raw_id = str(obj.pop(id_field)).strip()
            if not raw_id:
                raise ValueError(f"{path}: line {line_no} has a blank {id_field!r}")
            yield Record(id=raw_id, fields=obj)

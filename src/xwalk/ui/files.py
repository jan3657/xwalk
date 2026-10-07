"""Files people bring to the UI: storing uploads, inspecting them, reading them as records.

An uploaded file is read once into normalized records, so a generated job never depends
on the upload's column names (which may hold spaces or punctuation a template cannot
name):

- a **source** record has `text` (what to match), `context` (optional surrounding text)
  and the file's other columns under safe names;
- a **target** record has `label`, `synonyms` (a list) and `definition`, plus the other
  columns, the same shape as the OBO and OWL loaders produce.

Formats: CSV (delimiter sniffed), TSV, JSONL, plain text (one term per line), OBO and
OWL/RDF (OWL needs `xwalk[ontology]`). Spreadsheets (.xlsx) are refused with a hint to
save them as CSV, since reading them would need another dependency.
"""

from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from xwalk.records import Record

UPLOAD_DIR = "uploads"
SAMPLE_ROWS = 8
COUNT_LIMIT = 2_000_000
MAX_FIELD = 64 * 1024 * 1024

FORMATS = ("csv", "tsv", "jsonl", "txt", "obo", "owl")
TABULAR = ("csv", "tsv", "jsonl")
_SUFFIX_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".txt": "txt",
    ".obo": "obo",
    ".owl": "owl",
    ".rdf": "owl",
    ".xml": "owl",
}
_SPREADSHEET = (".xlsx", ".xls", ".ods")
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_FIELD_CHARS = re.compile(r"[^0-9a-z]+")
_SEPARATORS = ("|", ";")

_ID_HINTS = ("id", "identifier", "code", "key", "accession", "curie", "iri", "uri")
_TEXT_HINTS = ("text", "mention", "term", "name", "label", "title", "description", "value")
_LABEL_HINTS = ("label", "name", "preferred_label", "pref_label", "term", "title")
_SYNONYM_HINTS = ("synonym", "synonyms", "alias", "aliases", "alt_label", "altlabel", "other")
_DEFINITION_HINTS = ("definition", "description", "desc", "comment", "note", "scope_note")
_CONTEXT_HINTS = ("context", "sentence", "passage", "snippet")


class FileProblem(ValueError):
    """A file cannot be read as asked. The message is for the person who chose it."""


csv.field_size_limit(MAX_FIELD)


# --- storing -----------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    """A file name without directories or unusual characters (never empty, never hidden)."""
    base = Path(name.replace("\\", "/")).name
    cleaned = _SAFE_CHARS.sub("_", base).strip("._") or "upload"
    return cleaned[:120]


def unique_path(directory: Path, filename: str) -> Path:
    """`directory/filename`, or `name-2.ext`, `name-3.ext`, ... when it already exists."""
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    if candidate.name.endswith(".gz"):
        inner = Path(candidate.stem)
        stem, suffix = inner.stem, inner.suffix + ".gz"
    for n in itertools.count(2):
        if not candidate.exists():
            return candidate
        candidate = directory / f"{stem}-{n}{suffix}"
    raise AssertionError("unreachable")  # pragma: no cover


def detect_format(path: Path) -> str:
    """The format named by the file's extension (`.gz` looked through)."""
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    suffix = Path(name).suffix
    if suffix in _SPREADSHEET:
        raise FileProblem(
            f"{path.name}: spreadsheets are not read directly; save the sheet as CSV "
            "(File > Save as > CSV UTF-8) and upload that"
        )
    fmt = _SUFFIX_FORMATS.get(suffix)
    if fmt is None:
        raise FileProblem(
            f"{path.name}: unknown file type {suffix or '(none)'}; use .csv, .tsv, .jsonl, "
            ".txt, .obo or .owl"
        )
    return fmt


def _open_text(path: Path) -> IO[str]:
    if path.name.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _delimiter(path: Path, fmt: str) -> str:
    if fmt == "tsv":
        return "\t"
    with _open_text(path) as handle:
        sample = handle.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


# --- reading rows ------------------------------------------------------------------------


def iter_rows(path: Path, fmt: str) -> Iterator[dict[str, Any]]:
    """The rows of a tabular or text file as dicts (text: one `term` per line)."""
    try:
        if fmt in ("csv", "tsv"):
            delimiter = _delimiter(path, fmt)
            with _open_text(path) as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if not reader.fieldnames:
                    return
                for row in reader:
                    yield {
                        str(k): (v or "").strip()
                        for k, v in row.items()
                        if k is not None and k.strip()
                    }
        elif fmt == "jsonl":
            with _open_text(path) as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        raise FileProblem(
                            f"{path.name}: line {line_no} is not valid JSON"
                        ) from None
                    if not isinstance(data, dict):
                        raise FileProblem(f"{path.name}: line {line_no} is not a JSON object")
                    yield data
        elif fmt == "txt":
            with _open_text(path) as handle:
                for line in handle:
                    if line.strip():
                        yield {"term": line.strip()}
        else:
            raise FileProblem(f"{path.name}: {fmt} files are read as ontologies, not rows")
    except UnicodeDecodeError:
        raise FileProblem(
            f"{path.name}: not UTF-8 text; save it with UTF-8 encoding and upload again"
        ) from None


def _norm(name: str) -> str:
    return _FIELD_CHARS.sub("_", name.lower()).strip("_")


def _guess(columns: list[str], hints: Iterable[str], *, exclude: Iterable[str] = ()) -> str | None:
    skip = set(exclude)
    normalized = {c: _norm(c) for c in columns if c not in skip}
    for hint in hints:
        for column, norm in normalized.items():
            if norm == hint:
                return column
    for hint in hints:
        for column, norm in normalized.items():
            if hint in norm.split("_") or norm.endswith(hint):
                return column
    return None


def _separator(values: Iterable[Any]) -> str | None:
    counts = dict.fromkeys(_SEPARATORS, 0)
    for value in values:
        if isinstance(value, str):
            for sep in _SEPARATORS:
                counts[sep] += value.count(sep)
    best = max(counts, key=lambda s: counts[s])
    return best if counts[best] else None


@dataclass
class Inspection:
    """What a file holds and a guess at how to read it."""

    path: str
    format: str
    columns: list[str] = field(default_factory=list)
    sample: list[dict[str, Any]] = field(default_factory=list)
    rows: int | None = None
    rows_capped: bool = False
    suggest: dict[str, Any] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "columns": self.columns,
            "sample": self.sample,
            "rows": self.rows,
            "rows_capped": self.rows_capped,
            "suggest": self.suggest,
            "note": self.note,
        }


def inspect_file(path: Path, shown: str) -> Inspection:
    """Columns, a few rows, a bounded row count and suggested column roles."""
    fmt = detect_format(path)
    result = Inspection(path=shown, format=fmt)
    if fmt in ("obo", "owl"):
        terms = read_ontology(path, fmt)
        first = list(itertools.islice(terms, SAMPLE_ROWS))
        result.sample = [{"id": r.id, **_jsonable(r.fields)} for r in first]
        result.columns = ["id", "label", "synonyms", "definition"]
        count = len(first) + sum(1 for _ in itertools.islice(terms, COUNT_LIMIT))
        result.rows, result.rows_capped = count, count >= COUNT_LIMIT
        result.note = "an ontology: terms are read with their labels, synonyms and definitions"
        result.suggest = {"id_column": "id", "label_column": "label"}
        return result
    rows = iter_rows(path, fmt)
    sample = list(itertools.islice(rows, SAMPLE_ROWS))
    count = len(sample) + sum(1 for _ in itertools.islice(rows, COUNT_LIMIT))
    result.rows, result.rows_capped = count, count >= COUNT_LIMIT
    result.sample = [_jsonable(r) for r in sample]
    columns: list[str] = []
    for row in sample:
        columns += [c for c in row if c not in columns]
    result.columns = columns
    id_column = _guess(columns, _ID_HINTS)
    label = _guess(columns, _LABEL_HINTS, exclude=[id_column] if id_column else [])
    text = _guess(columns, _TEXT_HINTS, exclude=[id_column] if id_column else []) or label
    if text is None:
        text = next((c for c in columns if c != id_column), None)
    synonyms = _guess(columns, _SYNONYM_HINTS, exclude=[c for c in (id_column, label) if c])
    definition = _guess(
        columns, _DEFINITION_HINTS, exclude=[c for c in (id_column, label, synonyms) if c]
    )
    context = _guess(columns, _CONTEXT_HINTS, exclude=[c for c in (id_column, text) if c])
    if id_column is not None and len({r.get(id_column) for r in sample}) < len(sample):
        id_column = None  # repeated values in the sample: not an identifier
    result.suggest = {
        "id_column": id_column,
        "text_column": text,
        "context_columns": [context] if context else [],
        "label_column": label or text,
        "synonyms_column": synonyms,
        "synonyms_sep": _separator(r.get(synonyms) for r in sample) if synonyms else None,
        "definition_column": definition,
    }
    return result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# --- normalizing -----------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(_text(v) for v in value)
    return str(value).strip()


def _split(value: Any, sep: str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_text(v) for v in value if _text(v)]
    text = _text(value)
    if not text:
        return []
    parts = text.split(sep) if sep else [text]
    return [p.strip() for p in parts if p.strip()]


def _extra_fields(row: Mapping[str, Any], used: Iterable[str | None]) -> dict[str, Any]:
    """The row's other columns under safe field names (never overriding a fixed field)."""
    skip = {u for u in used if u}
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in skip or value in ("", None):
            continue
        name = _norm(key) or "field"
        if name[0].isdigit():
            name = f"f_{name}"
        if name in ("id", "text", "context", "label", "synonyms", "definition", "ontology"):
            name = f"col_{name}"
        out.setdefault(name, value)
    return out


def _check_columns(columns: Iterable[str | None], sample: Mapping[str, Any], path: Path) -> None:
    for column in columns:
        if column and column not in sample:
            raise FileProblem(f"{path.name}: there is no column {column!r}")


def source_records(
    path: Path,
    fmt: str,
    *,
    id_column: str | None,
    text_column: str | None,
    context_columns: Iterable[str] = (),
) -> Iterator[Record]:
    """Records to match: `text`, optional `context`, and the other columns.

    Without `id_column` the row number is the id (`r1`, `r2`, ...). Rows with an empty
    text are skipped; a repeated id is an error, since a source id names one record.
    """
    if fmt in ("obo", "owl"):
        for record in read_ontology(path, fmt):
            yield Record(
                record.id,
                {
                    "text": _text(record.fields.get("label")),
                    "context": _text(record.fields.get("definition")),
                    "synonyms": record.fields.get("synonyms", []),
                },
            )
        return
    contexts = [c for c in context_columns if c]
    text_key = "term" if fmt == "txt" else text_column
    if not text_key:
        raise FileProblem("choose the column that holds the text to match")
    seen: set[str] = set()
    checked = False
    for number, row in enumerate(iter_rows(path, fmt), start=1):
        if not checked:
            _check_columns([id_column, text_key, *contexts], row, path)
            checked = True
        text = _text(row.get(text_key))
        if not text:
            continue
        record_id = _text(row.get(id_column)) if id_column else f"r{number}"
        if not record_id:
            raise FileProblem(f"{path.name}: row {number} has an empty {id_column!r}")
        if record_id in seen:
            raise FileProblem(
                f"{path.name}: id {record_id!r} appears twice in {id_column!r}; choose a "
                "column with unique values, or none to number the rows"
            )
        seen.add(record_id)
        fields = _extra_fields(row, [id_column, text_key, *contexts])
        fields["text"] = text
        fields["context"] = " … ".join(_text(row.get(c)) for c in contexts if _text(row.get(c)))
        yield Record(record_id, fields)


def target_records(
    path: Path,
    fmt: str,
    *,
    id_column: str | None,
    label_column: str | None,
    synonyms_column: str | None = None,
    synonyms_sep: str | None = "|",
    definition_column: str | None = None,
) -> Iterator[Record]:
    """Records to match onto: `label`, `synonyms`, `definition`, and the other columns."""
    if fmt in ("obo", "owl"):
        for record in read_ontology(path, fmt):
            yield Record(record.id, _ontology_fields(record.fields))
        return
    label_key = "term" if fmt == "txt" else label_column
    if not label_key:
        raise FileProblem("choose the column that holds each target's label")
    seen: set[str] = set()
    checked = False
    for number, row in enumerate(iter_rows(path, fmt), start=1):
        if not checked:
            _check_columns([id_column, label_key, synonyms_column, definition_column], row, path)
            checked = True
        label = _text(row.get(label_key))
        if not label:
            continue
        record_id = _text(row.get(id_column)) if id_column else f"t{number}"
        if not record_id:
            raise FileProblem(f"{path.name}: row {number} has an empty {id_column!r}")
        if record_id in seen:
            raise FileProblem(f"{path.name}: id {record_id!r} appears twice in {id_column!r}")
        seen.add(record_id)
        fields = _extra_fields(row, [id_column, label_key, synonyms_column, definition_column])
        fields["label"] = label
        fields["synonyms"] = (
            _split(row.get(synonyms_column), synonyms_sep) if synonyms_column else []
        )
        fields["definition"] = _text(row.get(definition_column)) if definition_column else ""
        yield Record(record_id, fields)


def _ontology_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": _text(fields.get("label")),
        "synonyms": [s for s in (_text(v) for v in fields.get("synonyms") or []) if s],
        "definition": _text(fields.get("definition")),
        "parents": list(fields.get("parents") or []),
    }


def read_ontology(path: Path, fmt: str) -> Iterator[Record]:
    """Terms of an OBO or OWL file (gzip allowed), obsolete terms left out."""
    from xwalk.sources.ontology import obo_source, owl_source

    if path.name.lower().endswith(".gz"):
        raise FileProblem(f"{path.name}: decompress ontology files before reading them")
    if fmt == "obo":
        return obo_source(path)
    if fmt == "owl":
        return owl_source(path)
    raise FileProblem(f"{path.name}: not an ontology file")


def write_jsonl(records: Iterable[Record], path: Path) -> int:
    """Write records as JSONL (`id` plus fields) atomically; returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            line = {"id": record.id, **_jsonable(dict(record.fields))}
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            count += 1
    partial.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[Record]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                record_id = str(data.pop("id"))
                yield Record(record_id, data)


def terms_from_text(text: str) -> list[Record]:
    """One source record per non-empty line of pasted text (`q1`, `q2`, ...).

    A line `text<TAB>context` keeps the part after the first tab as context. Repeated
    lines are kept once.
    """
    records: list[Record] = []
    seen: set[str] = set()
    for line in text.splitlines():
        term, _, context = line.partition("\t")
        term = term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        records.append(Record(f"q{len(records) + 1}", {"text": term, "context": context.strip()}))
    return records


__all__ = [
    "FORMATS",
    "UPLOAD_DIR",
    "FileProblem",
    "Inspection",
    "detect_format",
    "inspect_file",
    "iter_rows",
    "read_jsonl",
    "read_ontology",
    "safe_filename",
    "source_records",
    "target_records",
    "terms_from_text",
    "unique_path",
    "write_jsonl",
]

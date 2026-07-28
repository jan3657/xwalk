"""Gold labels.

Two things that look alike and are not: an **empty** label means "the correct answer is
no match", a first-class outcome; an **absent** source id means unlabelled, and is
excluded from every metric. Conflating them inflates no-match recall silently.

ID normalization and alias expansion are user-supplied callables. The paper repo owned
these as named hooks inside the library (`normalize_ncbi_gene_prediction`,
`expand_ctd_eval_ids`); here they live in your code and the library never learns your
identifier scheme.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

Normalizer = Callable[[str], str | None]
Expander = Callable[[frozenset[str]], AbstractSet[str]]


@dataclass(frozen=True)
class GoldSet:
    labels: Mapping[str, frozenset[str]]

    def get(self, source_id: str) -> frozenset[str] | None:
        return self.labels.get(source_id)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[str]:
        return iter(self.labels)

    @property
    def no_match_ids(self) -> frozenset[str]:
        """Source ids whose correct answer is explicitly 'no match'."""
        return frozenset(sid for sid, ids in self.labels.items() if not ids)

    def is_correct(self, source_id: str, predicted: str | None) -> bool | None:
        """None when unlabelled. Otherwise: does the prediction match the label?"""
        expected = self.get(source_id)
        if expected is None:
            return None
        if predicted is None:
            return not expected
        return predicted in expected


def _finalise(
    raw: Iterable[str],
    normalize: Normalizer | None,
    expand: Expander | None,
) -> frozenset[str]:
    ids = {i.strip() for i in raw if i.strip()}
    if normalize is not None:
        ids = {n for n in (normalize(i) for i in ids) if n}
    if not ids:
        return frozenset()  # a no-match label needs no alias expansion
    if expand is not None:
        ids = set(expand(frozenset(ids)))
    return frozenset(ids)


def load_gold_csv(
    path: str | Path,
    *,
    source_column: str = "source_id",
    gold_column: str = "gold_ids",
    separator: str = "|",
    encoding: str = "utf-8",
    normalize: Normalizer | None = None,
    expand: Expander | None = None,
) -> GoldSet:
    labels: dict[str, frozenset[str]] = {}
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for column in (source_column, gold_column):
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(
                    f"{column!r} not found in {path}; columns are {reader.fieldnames!r}"
                )
        for line_no, row in enumerate(reader, start=2):
            source_id = (row[source_column] or "").strip()
            if not source_id:
                raise ValueError(f"{path}: row {line_no} has a blank {source_column!r}")
            if source_id in labels:
                raise ValueError(f"{path}: row {line_no} is a duplicate source id {source_id!r}")
            cell = row[gold_column] or ""
            labels[source_id] = _finalise(cell.split(separator), normalize, expand)
    return GoldSet(labels)


def load_gold_jsonl(
    path: str | Path,
    *,
    source_field: str = "source_id",
    gold_field: str = "gold_ids",
    encoding: str = "utf-8",
    normalize: Normalizer | None = None,
    expand: Expander | None = None,
) -> GoldSet:
    labels: dict[str, frozenset[str]] = {}
    path = Path(path)
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no} is not valid JSON: {exc}") from exc
            source_id = str(obj.get(source_field, "")).strip()
            if not source_id:
                raise ValueError(f"{path}: line {line_no} has a blank {source_field!r}")
            if source_id in labels:
                raise ValueError(f"{path}: line {line_no} duplicates source id {source_id!r}")
            raw = obj.get(gold_field, [])
            values = [raw] if isinstance(raw, str) else [str(v) for v in raw]
            labels[source_id] = _finalise(values, normalize, expand)
    return GoldSet(labels)

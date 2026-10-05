"""Three partitions with distinct roles.

Revising a prompt from dev failures *and* selecting the retained round on dev
performance makes dev part of training. Hence:

  prompt-train : failures shown to the optimising model
  validation   : chooses the retained round and the stopping point
  test         : evaluated once, after the final prompt is selected

Assignment is hash-based rather than random so that adding labelled records never
reshuffles the existing split -- a test partition that moves between runs is not a test
partition.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Partition(Enum):
    PROMPT_TRAIN = "prompt_train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class Partitioner:
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25)
    salt: str = "xwalk"

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {sum(self.fractions)}")
        if any(f <= 0 for f in self.fractions):
            raise ValueError(
                "every fraction must be positive; an empty partition disables its role"
            )

    def draw(self, source_id: str) -> float:
        """The id's deterministic position in [0, 1), salted; what `assign` buckets."""
        digest = hashlib.sha256(f"{self.salt}\x00{source_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def assign(self, source_id: str) -> Partition:
        draw = self.draw(source_id)
        train, validation, _ = self.fractions
        if draw < train:
            return Partition.PROMPT_TRAIN
        if draw < train + validation:
            return Partition.VALIDATION
        return Partition.TEST

    def split(self, source_ids: Iterable[str]) -> dict[Partition, list[str]]:
        out: dict[Partition, set[str]] = {p: set() for p in Partition}
        for source_id in source_ids:
            out[self.assign(source_id)].add(source_id)
        return {partition: sorted(ids) for partition, ids in out.items()}


def write_partition_file(assignment: Mapping[str, Partition], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "partition"])
        for source_id in sorted(assignment):
            writer.writerow([source_id, assignment[source_id].value])


def load_partition_file(path: str | Path) -> dict[str, Partition]:
    """An explicit file always wins over hashing -- for reproducing a published split."""
    out: dict[str, Partition] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            source_id = (row.get("source_id") or "").strip()
            name = (row.get("partition") or "").strip().lower()
            if not source_id:
                continue
            try:
                out[source_id] = Partition(name)
            except ValueError as exc:
                raise ValueError(
                    f"row {line_no}: unknown partition {name!r}; "
                    f"expected one of {[p.value for p in Partition]}"
                ) from exc
    return out


def partition_of(
    source_id: str,
    partitioner: Partitioner,
    overrides: Mapping[str, Partition] | None = None,
) -> Partition:
    if overrides and source_id in overrides:
        return overrides[source_id]
    return partitioner.assign(source_id)


def ids_in(
    source_ids: Sequence[str],
    partition: Partition,
    partitioner: Partitioner,
    overrides: Mapping[str, Partition] | None = None,
) -> list[str]:
    return sorted(
        {sid for sid in source_ids if partition_of(sid, partitioner, overrides) is partition}
    )

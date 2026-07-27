"""In-memory target store. The default for collections that fit in RAM."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record


class MemoryStore:
    """Dict-backed TargetStore built eagerly from any iterable of Records."""

    def __init__(self, records: dict[str, Record], fingerprint: str) -> None:
        self._records = records
        self._fingerprint = fingerprint

    @classmethod
    def from_source(cls, source: Iterable[Record]) -> MemoryStore:
        records: dict[str, Record] = {}
        digests: list[str] = []
        for record in source:
            if record.id in records:
                raise ValueError(f"duplicate target record id: {record.id!r}")
            records[record.id] = record
            digests.append(hash_record(record))
        return cls(records, hash_value(sorted(digests)))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def get(self, record_id: str) -> Record:
        return self._records[record_id]

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        # Unknown IDs are skipped rather than raising: a retriever backed by a stale
        # index can legitimately name a record that has since been removed.
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records.values())

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._records

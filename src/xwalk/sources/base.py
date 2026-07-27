"""The entire extension contract for input data."""

from __future__ import annotations

from collections.abc import Iterable

from xwalk.records import Record

# A RecordSource is anything iterable that yields Records. A list works. A generator
# reading 10M rows works. A function querying your warehouse works. Nothing in xwalk
# needs to know your dataset exists.
RecordSource = Iterable[Record]

__all__ = ["RecordSource"]

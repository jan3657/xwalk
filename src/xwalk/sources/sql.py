"""SQL record source. Requires xwalk[sql].

Streamed in chunks rather than materialised: "point at your warehouse" is exactly the
case where loading the whole result set defeats the purpose.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from xwalk._extras import require
from xwalk.records import Record


def sql_source(
    url: str,
    query: str,
    *,
    id_column: str,
    params: Mapping[str, Any] | None = None,
    multivalue_columns: Sequence[str] = (),
    multivalue_sep: str = "|",
    chunk_size: int = 1000,
) -> Iterator[Record]:
    sa = require("sql", "sqlalchemy", purpose="sql_source")

    multivalue = set(multivalue_columns)
    engine = sa.create_engine(url)
    try:
        with engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(
                sa.text(query), dict(params or {})
            )
            columns = list(result.keys())
            if id_column not in columns:
                raise ValueError(
                    f"id_column {id_column!r} not in the query result; columns are {columns!r}"
                )

            while True:
                rows = result.fetchmany(chunk_size)
                if not rows:
                    break
                for row in rows:
                    mapping = dict(zip(columns, row, strict=True))
                    raw_id = mapping.pop(id_column)
                    if raw_id is None or not str(raw_id).strip():
                        raise ValueError(f"row with a blank {id_column!r}: {mapping!r}")
                    fields: dict[str, Any] = {}
                    for key, value in mapping.items():
                        if key in multivalue:
                            text = "" if value is None else str(value)
                            fields[key] = [
                                p.strip() for p in text.split(multivalue_sep) if p.strip()
                            ]
                        else:
                            fields[key] = "" if value is None else value
                    yield Record(id=str(raw_id).strip(), fields=fields)
    finally:
        # Abandoning the generator must not strand a warehouse connection in the pool.
        engine.dispose()

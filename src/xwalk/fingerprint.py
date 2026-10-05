"""Deterministic hashing of records and run configuration.

Everything that decides whether a prior result is still valid flows through here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from xwalk.records import Record

_DIGEST_CHARS = 16


def _normalise(value: Any) -> Any:
    """Reduce a value to JSON-safe primitives, deterministically."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        # Sets have no order, so the elements are normalised and then sorted by their
        # canonical rendering: deterministic, and it keeps the elements themselves
        # rather than replacing them with their JSON text.
        normalised = [_normalise(v) for v in value]
        return sorted(normalised, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalise(v) for v in value]
    raise TypeError(f"value of type {type(value).__name__} is not serialisable for hashing")


def canonical_json(value: Any) -> str:
    """A byte-stable JSON rendering: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_value(value: Any) -> str:
    """A 16-hex-char digest. Short enough to read in a filename, wide enough to be safe."""
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_DIGEST_CHARS]


def hash_record(record: Record) -> str:
    """Digest of a record's identity *and* content."""
    return hash_value({"id": record.id, "fields": record.fields})


def run_fingerprint(**components: Any) -> str:
    """Digest of everything that would invalidate prior results.

    Callers pass the target snapshot fingerprint, retriever fingerprints, prompt and
    slots hashes, model and provider identity, generation parameters, the match policy,
    and the library version.
    """
    return hash_value(components)


def result_key(run_fp: str, source_id: str, source_hash: str) -> str:
    """Stable identity of one source record under one run configuration.

    The parts go in as a list, never concatenated: ``"a" + "bc"`` and ``"ab" + "c"``
    produce the same string, and a resume key that can collide is worse than none.
    """
    return hash_value([run_fp, source_id, source_hash])


# Names whose values are credentials. They never enter a fingerprint or a manifest:
# rotating a key must not invalidate a resumable run, and a secret must not reach disk.
# `token` only as a whole word: `max_tokens` is a generation setting, not a secret.
_SECRET_NAME = re.compile(
    r"api[-_]?key|(^|[-_])key$|secret|passw|credential|authori[sz]ation|bearer|header|cookie"
    r"|(^|[-_])token$",
    re.I,
)


def is_secret_name(name: str) -> bool:
    """True for a parameter name that may carry a credential (``api_key``, headers...)."""
    return bool(_SECRET_NAME.search(name))


def without_secrets(params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if not is_secret_name(str(k))}


def redact_url(url: str) -> str:
    """The URL without userinfo or credential-like query parameters."""
    parts = urlsplit(url)
    if not parts.scheme and not parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not is_secret_name(k)])
    return urlunsplit((parts.scheme, host, parts.path, query, parts.fragment))

"""Opaque candidate keys.

Candidates are shown to the model under temporary keys assigned per attempt. The model
returns a key or null. Resolution is an exact dictionary lookup against the keys issued
*for that attempt*; anything else is UNRESOLVED.

This exists because heuristic ID resolution is unsafe as a default. Its worst branch —
treating a bare integer as a candidate rank — turns a hallucinated ID into a different,
real mapping whenever target IDs are numeric. Opaque keys also remove the sentinel
muddle: `null` is unambiguous where "-1" and "0" are values some scheme legitimately uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from xwalk.records import Candidate
from xwalk.templates import TemplateSet

_ABSTAIN_TOKENS = frozenset(
    {"", "null", "none", "nil", "no_match", "nomatch", "no match", "n/a", "na"}
)
_TRIM = " \t\r\n\"'`[]().,;:"


class Resolution(Enum):
    """How a raw model answer became (or failed to become) a record id."""

    EXACT_KEY = "exact_key"
    ABSTAIN = "abstain"
    UNRESOLVED = "unresolved"
    LEGACY_EXACT_ID = "legacy_exact_id"
    LEGACY_FUZZY = "legacy_fuzzy"
    LEGACY_RANK = "legacy_rank"


@dataclass(frozen=True)
class ResolvedChoice:
    record_id: str | None
    resolution: Resolution
    raw: str | None


@dataclass(frozen=True)
class KeyedCandidates:
    order: tuple[str, ...]
    by_key: Mapping[str, Candidate]
    issued: Mapping[str, str]
    rendered: str

    def __len__(self) -> int:
        return len(self.order)


def assign_keys(candidates: Sequence[Candidate], templates: TemplateSet) -> KeyedCandidates:
    """Assign C01, C02, ... in the order given, and render the candidate block."""
    if not candidates:
        return KeyedCandidates(order=(), by_key={}, issued={}, rendered="")

    width = max(2, len(str(len(candidates))))
    order: list[str] = []
    by_key: dict[str, Candidate] = {}
    issued: dict[str, str] = {}
    blocks: list[str] = []

    for index, candidate in enumerate(candidates, start=1):
        key = f"C{index:0{width}d}"
        order.append(key)
        by_key[key] = candidate
        issued[key] = candidate.id
        blocks.append(f"[{key}] {templates.render_candidate(candidate.record)}")

    return KeyedCandidates(
        order=tuple(order), by_key=by_key, issued=issued, rendered="\n\n".join(blocks)
    )


def _normalise(raw: str) -> str:
    return raw.strip().strip(_TRIM).strip()


def resolve_key(
    raw: str | None,
    keyed: KeyedCandidates,
    *,
    legacy: bool = False,
) -> ResolvedChoice:
    """Turn the model's answer into a record id, or refuse.

    `legacy=True` re-enables heuristic ID resolution for reproducing prior work. Every
    non-exact path is tagged so Task 14 can force NEEDS_REVIEW on it.
    """
    if raw is None:
        return ResolvedChoice(None, Resolution.ABSTAIN, raw)

    text = _normalise(raw)
    if text.lower() in _ABSTAIN_TOKENS:
        return ResolvedChoice(None, Resolution.ABSTAIN, raw)

    # The key space is ours, so upper-casing cannot merge two distinct records.
    upper = text.upper()
    if upper in keyed.issued:
        return ResolvedChoice(keyed.issued[upper], Resolution.EXACT_KEY, raw)

    if not legacy:
        return ResolvedChoice(None, Resolution.UNRESOLVED, raw)

    ids = [keyed.issued[key] for key in keyed.order]
    if text in ids:
        return ResolvedChoice(text, Resolution.LEGACY_EXACT_ID, raw)

    lowered = {rid.lower(): rid for rid in ids}
    if text.lower() in lowered:
        return ResolvedChoice(lowered[text.lower()], Resolution.LEGACY_FUZZY, raw)

    suffixes = {rid.rsplit(":", 1)[-1]: rid for rid in ids}
    if text in suffixes:
        return ResolvedChoice(suffixes[text], Resolution.LEGACY_FUZZY, raw)

    if text.isdigit():
        rank = int(text)
        if 1 <= rank <= len(ids):
            return ResolvedChoice(ids[rank - 1], Resolution.LEGACY_RANK, raw)

    return ResolvedChoice(None, Resolution.UNRESOLVED, raw)

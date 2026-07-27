"""Routing retry proposals.

Two different objects share the name "retry lead", and conflating them causes real
waste: the paper repo asks the scorer for alternatives drawn from the candidates it was
just shown, then pushes those labels into the query queue and re-runs full retrieval on
records it already has in hand.
"""

from __future__ import annotations

import re
from collections.abc import Sequence, Set
from dataclasses import dataclass

from xwalk.records import RetryProposal

_WS = re.compile(r"\s+")


def normalise_query(text: str) -> str:
    """The comparison form for query deduplication."""
    return _WS.sub(" ", text.strip().lower())


@dataclass(frozen=True)
class RoutedProposals:
    candidate_keys: tuple[str, ...]
    queries: tuple[str, ...]
    dropped: tuple[tuple[RetryProposal, str], ...]


def route_proposals(
    proposals: Sequence[RetryProposal],
    issued_keys: Sequence[str],
    seen_queries: Set[str],
) -> RoutedProposals:
    """Split proposals into re-examinations and new searches, dropping the rest.

    `seen_queries` holds already-normalised query strings.
    """
    issued = {key.upper() for key in issued_keys}
    keys: list[str] = []
    queries: list[str] = []
    seen_normalised = set(seen_queries)
    dropped: list[tuple[RetryProposal, str]] = []

    for proposal in proposals:
        value = proposal.value.strip()
        if not value:
            dropped.append((proposal, "blank value"))
            continue

        if proposal.kind == "candidate":
            key = value.upper()
            if key not in issued:
                # Not a lead — an invented key. Recorded, never promoted to a query.
                dropped.append((proposal, f"candidate key {value!r} was not issued this attempt"))
            elif key in keys:
                dropped.append((proposal, "duplicate candidate key"))
            else:
                keys.append(key)
            continue

        normalised = normalise_query(value)
        if normalised in seen_normalised:
            dropped.append((proposal, "query already tried"))
            continue
        seen_normalised.add(normalised)
        queries.append(value)

    return RoutedProposals(
        candidate_keys=tuple(keys), queries=tuple(queries), dropped=tuple(dropped)
    )

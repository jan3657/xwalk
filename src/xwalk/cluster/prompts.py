"""The clustering prompts. Built in, versioned, and part of the run fingerprint.

Each prompt starts with a `Task:` line naming the decision, so a stored prompt says what
it was for and a scripted test double can route on it. Cluster evidence is always the
members themselves (bounded), never counts or a single exemplar (CONTRACTS.md 10.6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.fingerprint import hash_value

PROMPT_VERSION = 1
SYSTEM = "You return JSON only. No prose, no code fences."

DEFAULT_RELATION = (
    "Two records are EQUIVALENT when they denote the same entity or concept and their "
    "labels could be used interchangeably. A broader concept, a narrower concept, a part, "
    "or a merely related concept is NOT equivalent."
)

_SELECT = """Task: select-cluster

{relation}

## {subject_heading}
{subject}

## Candidate clusters
{candidates}

Choose the one candidate cluster whose members are all equivalent to the {subject_noun}, \
or null when none is. A cluster that is broader, narrower or only related is not a match.
Return JSON: {{"chosen_key": "<key such as C01, or null>", "confidence_score": <number 0-1>, \
"explanation": "<one sentence>"}}"""

_VERIFY = """Task: verify-assignment

{relation}

## Record
{subject}

## Proposed cluster
{cluster}

Is the record equivalent to every member listed in the proposed cluster?
Return JSON: {{"decision": "equivalent" or "not_equivalent", \
"confidence_score": <number 0-1>, "explanation": "<one sentence>"}}"""

_NOVELTY = """Task: verify-novelty

{relation}

## Record
{subject}

## Nearest existing clusters
{candidates}

Is the record a concept that none of these clusters denote, so that it should start a \
new cluster?
Return JSON: {{"novel": true or false, "confidence_score": <number 0-1>, \
"explanation": "<one sentence>"}}"""

_MERGE = """Task: verify-merge

{relation}

## Cluster A
{left}

## Cluster B
{right}

Do all members of cluster A and cluster B denote the same concept, so that the two \
clusters should become one?
Return JSON: {{"decision": "same" or "different", "confidence_score": <number 0-1>, \
"explanation": "<one sentence>"}}"""

_COMPARE = """Task: compare-clusters

{relation}

## Record
{subject}

## Current cluster
{incumbent}

## Alternative cluster
{challenger}

Which cluster is the record equivalent to? Prefer the current cluster unless the \
alternative is clearly the better fit.
Return JSON: {{"preferred": "current" or "alternative", "confidence_score": <number 0-1>, \
"explanation": "<one sentence>"}}"""

SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chosen_key": {"type": ["string", "null"]},
        "confidence_score": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["chosen_key", "confidence_score", "explanation"],
}
VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["equivalent", "not_equivalent"]},
        "confidence_score": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["decision", "confidence_score", "explanation"],
}
NOVELTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "novel": {"type": "boolean"},
        "confidence_score": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["novel", "confidence_score", "explanation"],
}
MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["same", "different"]},
        "confidence_score": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["decision", "confidence_score", "explanation"],
}
COMPARE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "preferred": {"type": "string", "enum": ["current", "alternative"]},
        "confidence_score": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["preferred", "confidence_score", "explanation"],
}


@dataclass(frozen=True)
class ClusterPrompts:
    """The prompt texts and the relation brief. `relation` is the only domain hook."""

    relation: str = DEFAULT_RELATION

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "version": PROMPT_VERSION,
                "system": SYSTEM,
                "relation": self.relation,
                "templates": [_SELECT, _VERIFY, _NOVELTY, _MERGE, _COMPARE],
            }
        )

    def select(self, subject: str, candidates: str, *, subject_is_cluster: bool = False) -> str:
        return _SELECT.format(
            relation=self.relation,
            subject_heading="Cluster" if subject_is_cluster else "Record",
            subject_noun="members of this cluster" if subject_is_cluster else "record",
            subject=subject,
            candidates=candidates,
        )

    def verify(self, subject: str, cluster: str) -> str:
        return _VERIFY.format(relation=self.relation, subject=subject, cluster=cluster)

    def novelty(self, subject: str, candidates: str) -> str:
        return _NOVELTY.format(
            relation=self.relation, subject=subject, candidates=candidates or "(none)"
        )

    def merge(self, left: str, right: str) -> str:
        return _MERGE.format(relation=self.relation, left=left, right=right)

    def compare(self, subject: str, incumbent: str, challenger: str) -> str:
        return _COMPARE.format(
            relation=self.relation, subject=subject, incumbent=incumbent, challenger=challenger
        )


def cluster_block(member_lines: Sequence[str], total: int, *, key: str | None = None) -> str:
    """A cluster as the model sees it: its first members, and how many are not shown."""
    head = f"[{key}] " if key else ""
    noun = "member" if total == 1 else "members"
    lines = [f"{head}cluster with {total} {noun}:"]
    lines += [f"  - {line}" for line in member_lines]
    if total > len(member_lines):
        lines.append(f"  (+{total - len(member_lines)} more not shown)")
    return "\n".join(lines)


def schema_for(kind: str) -> Mapping[str, Any]:
    return {
        "select": SELECT_SCHEMA,
        "merge_select": SELECT_SCHEMA,
        "verify": VERIFY_SCHEMA,
        "reverify": VERIFY_SCHEMA,
        "novelty": NOVELTY_SCHEMA,
        "merge": MERGE_SCHEMA,
        "compare": COMPARE_SCHEMA,
    }[kind]


__all__ = [
    "DEFAULT_RELATION",
    "PROMPT_VERSION",
    "SYSTEM",
    "ClusterPrompts",
    "cluster_block",
    "schema_for",
]

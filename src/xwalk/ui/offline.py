"""Offline stand-ins for a model, so the UI can be explored with no endpoint and no spend.

None of these is a judgement about the data. The UI labels every run made with them.

- Matching (LLM path): `scripted_match_model`, the fixed reply of `examples/quickstart.py`
  and `xwalk mcp --offline-model` -- pick the first candidate with confidence 0.9.
- Matching (decider path): `xwalk.decide.fake.FakeDecider`, deterministic token overlap.
- Clustering: `lexical_cluster_model`, which reads the built-in clustering prompts
  (`xwalk.cluster.prompts`) and decides by word overlap between labels, so a demo run
  produces clusters of near-identical spellings instead of one cluster or all singletons.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from xwalk.llm.base import LLMClient, LLMRequest
from xwalk.llm.fake import FakeLLM

OFFLINE_MODEL_NAME = "xwalk-offline"

_SECTION = re.compile(r"^## (.+)$", re.M)
_KEYED = re.compile(r"^\[(C\d+)\] cluster with \d+ members?:$")
_TOKEN = re.compile(r"[a-z0-9]+")

# Two labels are "equivalent" to the lexical stand-in when their word sets overlap at
# least this much (Jaccard). A demo setting, not a calibrated threshold.
SAME_AT = 0.6
SURE_MARGIN = 0.2


def scripted_match_model(request: LLMRequest) -> str:
    """One fixed answer every matching stage understands: first candidate, score 0.9."""
    return json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.9, "decision": "support", "queries": []}
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' lower-case word sets (0 when either is empty)."""
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _sections(prompt: str) -> dict[str, str]:
    parts = _SECTION.split(prompt)
    return {parts[i].strip(): parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def _members(block: str) -> list[str]:
    return [line[4:] for line in block.splitlines() if line.startswith("  - ")]


def _keyed_clusters(block: str) -> list[tuple[str, list[str]]]:
    """`[C01] cluster with n members:` blocks with their member lines."""
    found: list[tuple[str, list[str]]] = []
    for line in block.splitlines():
        head = _KEYED.match(line)
        if head:
            found.append((head.group(1), []))
        elif line.startswith("  - ") and found:
            found[-1][1].append(line[4:])
    return found


def _subject(sections: dict[str, str]) -> list[str]:
    """The record (one line) or the cluster's member lines being decided about."""
    if "Record" in sections:
        return [sections["Record"].strip()]
    return _members(sections.get("Cluster", ""))


def _affinity(left: Sequence[str], right: Sequence[str]) -> float:
    """Mean pairwise overlap between two groups of labels."""
    pairs = [overlap(a, b) for a in left for b in right]
    return sum(pairs) / len(pairs) if pairs else 0.0


def _weakest(left: Sequence[str], right: Sequence[str]) -> float:
    pairs = [overlap(a, b) for a in left for b in right]
    return min(pairs) if pairs else 0.0


def _sure(score: float) -> float:
    """Confidence in a yes/no call made from `score`: 0.5 at `SAME_AT`, rising to 1 at
    `SURE_MARGIN` away from it, so only near-threshold calls fall below an accept level."""
    return 0.5 + 0.5 * min(1.0, abs(score - SAME_AT) / SURE_MARGIN)


def _reply(confidence: float, explanation: str, **fields: object) -> str:
    body = {**fields, "confidence_score": round(confidence, 3), "explanation": explanation}
    return json.dumps(body)


def lexical_cluster_model(request: LLMRequest) -> str:
    """Answer the five clustering tasks by word overlap. See the module docstring."""
    prompt = request.user
    task = prompt.split("\n", 1)[0].removeprefix("Task: ").strip()
    sections = _sections(prompt)
    subject = _subject(sections)
    note = "offline lexical stand-in: word overlap, not a model judgement"

    if task == "select-cluster":
        chosen: str | None = None
        top = 0.0
        for key, members in _keyed_clusters(sections.get("Candidate clusters", "")):
            score = _affinity(subject, members)
            if score > top:
                chosen, top = key, score
        if chosen is None or top < SAME_AT:
            return _reply(_sure(top), note, chosen_key=None)
        return _reply(_sure(top), note, chosen_key=chosen)
    if task == "verify-assignment":
        score = _weakest(subject, _members(sections.get("Proposed cluster", "")))
        decision = "equivalent" if score >= SAME_AT else "not_equivalent"
        return _reply(_sure(score), note, decision=decision)
    if task == "verify-novelty":
        clusters = _keyed_clusters(sections.get("Nearest existing clusters", ""))
        nearest = max((_affinity(subject, members) for _, members in clusters), default=0.0)
        return _reply(_sure(nearest), note, novel=nearest < SAME_AT)
    if task == "verify-merge":
        score = _weakest(
            _members(sections.get("Cluster A", "")), _members(sections.get("Cluster B", ""))
        )
        decision = "same" if score >= SAME_AT else "different"
        return _reply(_sure(score), note, decision=decision)
    if task == "compare-clusters":
        current = _affinity(subject, _members(sections.get("Current cluster", "")))
        alternative = _affinity(subject, _members(sections.get("Alternative cluster", "")))
        preferred = "alternative" if alternative > current + 0.1 else "current"
        return _reply(0.8, note, preferred=preferred)
    return _reply(0.0, f"{note}; unknown task {task!r}")


def match_llm() -> LLMClient:
    return FakeLLM(handler=scripted_match_model, model=OFFLINE_MODEL_NAME)


def cluster_llm() -> LLMClient:
    return FakeLLM(handler=lexical_cluster_model, model=OFFLINE_MODEL_NAME)


__all__ = [
    "OFFLINE_MODEL_NAME",
    "SAME_AT",
    "cluster_llm",
    "lexical_cluster_model",
    "match_llm",
    "overlap",
    "scripted_match_model",
]

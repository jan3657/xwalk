"""A scripted judge for clustering tests.

`Oracle` answers every clustering prompt from a fixed ground truth instead of a model:
it parses the prompt the engine actually sent (subject, keyed cluster blocks, member
lines) and replies with JSON. So a test exercises the real prompts, keys and parsing,
and its expected clusters follow from the ground truth, not from a call script.

`same(a, b)` decides equivalence of two labels. By default two labels are equivalent
when they map to the same concept; `pairs` adds explicit (possibly non-transitive)
equivalences for contradictory-chain tests.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from xwalk.cluster import ClusterSettings, run_clustering
from xwalk.cluster.settings import ClusterPolicy, PoolSettings
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.records import Record
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ label }}", context="{{ label }}", doc="{{ label }}", candidate="{{ label }}"
)

_SECTION = re.compile(r"^## (.+)$", re.M)
_BLOCK = re.compile(r"(?:\[(C\d+)\] )?cluster with \d+ members?:\n((?:  - .*(?:\n|$))*)")

CHOCOLATE = {
    "c1": ("chocolate", "chocolate"),
    "c2": ("chocolate, dark", "dark"),
    "c3": ("dark chocolate", "dark"),
    "c4": ("plain chocolate", "dark"),
    "c5": ("milk chocolate", "milk"),
    "c6": ("heart attack", "mi"),
    "c7": ("myocardial infarction", "mi"),
    "c8": ("quinoa", "quinoa"),
}


def records(table: Mapping[str, tuple[str, str]]) -> list[Record]:
    return [Record(id=rid, fields={"label": label}) for rid, (label, _) in table.items()]


def concepts(table: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    return {label: concept for label, concept in table.values()}


def sections(prompt: str) -> dict[str, str]:
    parts = _SECTION.split(prompt)
    return {parts[i].strip(): parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def blocks(text: str) -> list[tuple[str | None, list[str]]]:
    found = []
    for match in _BLOCK.finditer(text):
        labels = [line[4:] for line in match.group(2).splitlines() if line.startswith("  - ")]
        found.append((match.group(1), labels))
    return found


def task(prompt: str) -> str:
    return prompt.split("\n", 1)[0].removeprefix("Task: ").strip()


class Oracle:
    def __init__(
        self,
        concept_of: Mapping[str, str],
        *,
        pairs: Iterable[tuple[str, str]] = (),
        naive_select: bool = False,
        confidence: float = 0.9,
        override: Callable[[str, LLMRequest], str | None] | None = None,
    ) -> None:
        self.concept_of = dict(concept_of)
        self.pairs = {frozenset(p) for p in pairs}
        self.naive_select = naive_select
        self.confidence = confidence
        self.override = override
        self.calls: Counter[str] = Counter()
        self.llm = FakeLLM(handler=self.answer)

    def same(self, a: str, b: str) -> bool:
        if a == b:
            return True
        if self.pairs:
            return frozenset((a, b)) in self.pairs
        return self.concept_of[a] == self.concept_of[b]

    def all_same(self, left: list[str], right: list[str]) -> bool:
        return all(self.same(a, b) for a in left for b in right)

    def _reply(self, **payload: object) -> str:
        payload.setdefault("confidence_score", self.confidence)
        payload.setdefault("explanation", "scripted")
        return json.dumps(payload)

    def answer(self, request: LLMRequest) -> str:
        name = task(request.user)
        self.calls[name] += 1
        if self.override is not None:
            custom = self.override(name, request)
            if custom is not None:
                return custom
        parts = sections(request.user)
        if name == "select-cluster":
            if "Cluster" in parts:
                subject = blocks(parts["Cluster"])[0][1]
            else:
                subject = [parts["Record"].strip()]
            for key, members in blocks(parts["Candidate clusters"]):
                hit = (
                    any(self.same(s, m) for s in subject for m in members)
                    if self.naive_select
                    else self.all_same(subject, members)
                )
                if hit:
                    return self._reply(chosen_key=key)
            return self._reply(chosen_key=None)
        if name == "verify-assignment":
            subject = [parts["Record"].strip()]
            members = blocks(parts["Proposed cluster"])[0][1]
            same = self.all_same(subject, members)
            return self._reply(decision="equivalent" if same else "not_equivalent")
        if name == "verify-novelty":
            subject = [parts["Record"].strip()]
            known = any(
                self.all_same(subject, members)
                for _, members in blocks(parts["Nearest existing clusters"])
            )
            return self._reply(novel=not known)
        if name == "verify-merge":
            left = blocks(parts["Cluster A"])[0][1]
            right = blocks(parts["Cluster B"])[0][1]
            same = self.all_same(left, right) and self.all_same(left, left + right)
            return self._reply(decision="same" if same else "different")
        if name == "compare-clusters":
            subject = [parts["Record"].strip()]
            current = blocks(parts["Current cluster"])[0][1]
            alternative = blocks(parts["Alternative cluster"])[0][1]
            better = self.all_same(subject, alternative) and not self.all_same(subject, current)
            return self._reply(preferred="alternative" if better else "current")
        raise AssertionError(f"unexpected prompt: {request.user[:200]}")


def settings(**changes: object) -> ClusterSettings:
    pool = {k: v for k, v in changes.items() if k in PoolSettings.__dataclass_fields__}
    policy = {k: v for k, v in changes.items() if k in ClusterPolicy.__dataclass_fields__}
    order = changes.get("order", "label")
    return ClusterSettings(order=order, pool=PoolSettings(**pool), policy=ClusterPolicy(**policy))  # type: ignore[arg-type]


async def cluster(sources: list[Record], out: Path, oracle: Oracle, **kwargs: object) -> object:
    config = kwargs.pop("settings", None) or settings()
    return await run_clustering(
        sources,
        out=out,
        llm=oracle.llm,
        templates=TEMPLATES,
        settings=config,
        **kwargs,  # type: ignore[arg-type]
    )


def partition(report_dir: Path) -> set[frozenset[str]]:
    import csv

    with (report_dir / "clusters.csv").open(encoding="utf-8") as handle:
        return {frozenset(row["member_ids"].split("|")) for row in csv.DictReader(handle)}


def member_rows(report_dir: Path) -> dict[str, dict[str, str]]:
    import csv

    with (report_dir / "members.csv").open(encoding="utf-8") as handle:
        return {row["source_id"]: row for row in csv.DictReader(handle)}

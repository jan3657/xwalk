"""A deterministic, offline stand-in for a model: the SYNTHETIC judge.

It answers xwalk's four prompts (select, score, verify, rewrite) with a string-similarity
heuristic over the candidate blocks it is shown. It never sees gold labels. Its purpose
is to drive every code path of the pipeline offline -- retrieval, truncation, keying,
retries, verification, accounting, exports -- so the benchmark runners can be checked
end to end. Its "quality" is that of a lexical matcher; numbers produced with it say
nothing about how an LLM would do and are always labelled SYNTHETIC.

Token counts are an estimate (characters / 4) so that cost columns are exercised; they
are not provider-reported usage.
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from xwalk.fingerprint import hash_value
from xwalk.llm.base import LLMCapabilities, LLMRequest, LLMResponse
from xwalk.records import Usage

_WORD = re.compile(r"[a-z0-9]+")
_KEY_LINE = re.compile(r"^\[(C\d+)\] ID: (.*)$")
_SOURCE_LINE = re.compile(r"^- ([A-Za-z_][\w]*): (.*)$")

SELECT_FLOOR = 0.45  # below this best similarity the judge abstains
VERIFY_SUPPORT = 0.75  # at or above this the verifier supports the proposal


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@functools.lru_cache(maxsize=100_000)
def _prepared(text: str) -> str:
    return " ".join(_singular(w) for w in normalise(text).split())


@functools.lru_cache(maxsize=200_000)
def similarity(a: str, b: str) -> float:
    """max(difflib ratio, token Jaccard) over normalised, singularised text.

    Pure and cached: the judge sees the same (mention, name) pairs in every stage.
    """
    left, right = _prepared(a), _prepared(b)
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    lt, rt = set(left.split()), set(right.split())
    jaccard = len(lt & rt) / len(lt | rt)
    return max(ratio, jaccard)


@dataclass(frozen=True)
class ShownCandidate:
    key: str
    record_id: str
    names: tuple[str, ...]


def parse_candidates(text: str) -> list[ShownCandidate]:
    """Read `[Cnn] ID: ...` blocks rendered by the example jobs' candidate template."""
    out: list[ShownCandidate] = []
    key = rid = None
    names: list[str] = []
    for line in text.splitlines():
        match = _KEY_LINE.match(line)
        if match:
            if key is not None and rid is not None:
                out.append(ShownCandidate(key, rid, tuple(names)))
            key, rid, names = match.group(1), match.group(2).strip(), []
        elif key is not None and line.startswith("Label: "):
            names.append(line[len("Label: ") :].strip())
        elif key is not None and line.startswith("Synonyms: "):
            names.extend(s.strip() for s in line[len("Synonyms: ") :].split(";") if s.strip())
    if key is not None and rid is not None:
        out.append(ShownCandidate(key, rid, tuple(names)))
    return out


def _section(text: str, title: str) -> str:
    start = text.find(f"## {title}\n")
    if start < 0:
        return ""
    body = text[start + len(title) + 4 :]
    end = body.find("\n## ")
    return body if end < 0 else body[:end]


def _source_value(text: str, field: str) -> str:
    for line in _section(text, "Source record").splitlines():
        match = _SOURCE_LINE.match(line)
        if match and match.group(1) == field:
            return match.group(2)
    return ""


def _score(mention: str, candidate: ShownCandidate) -> float:
    return max((similarity(mention, name) for name in candidate.names), default=0.0)


class SyntheticJudgeLLM:
    """An `LLMClient` answering from string similarity. Deterministic and offline."""

    def __init__(
        self,
        *,
        query_field: str = "mention",
        latency_s: float = 0.0,
        model: str = "synthetic-lexical-judge-v1",
    ) -> None:
        self._field = query_field
        self._latency = latency_s
        self._model = model
        self.calls_by_role: dict[str, int] = {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities()

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "adapter": "synthetic_judge",
                "model": self._model,
                "select_floor": SELECT_FLOOR,
                "verify_support": VERIFY_SUPPORT,
                "query_field": self._field,
            }
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._latency:
            await asyncio.sleep(self._latency)
        role, payload = self.answer(request.user)
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1
        text = json.dumps(payload)
        usage = Usage(
            prompt_tokens=math.ceil(len(request.system + request.user) / 4),
            completion_tokens=math.ceil(len(text) / 4),
            calls=1,
        )
        return LLMResponse(text=text, usage=usage, model=self._model, finish_reason="stop")

    # --- the heuristic ---------------------------------------------------------------

    def answer(self, prompt: str) -> tuple[str, dict[str, object]]:
        mention = _source_value(prompt, self._field)
        if prompt.startswith("You are improving a search query"):
            return "rewrite", self._rewrite(mention, prompt)
        if prompt.startswith("You are matching"):
            shown = parse_candidates(_section(prompt, "Candidates"))
            ranked = sorted(shown, key=lambda c: (-_score(mention, c), c.key))
            if not ranked or _score(mention, ranked[0]) < SELECT_FLOOR:
                return "select", {"chosen_key": None, "confidence_score": 0.0, "explanation": ""}
            best = ranked[0]
            return "select", {
                "chosen_key": best.key,
                "confidence_score": round(_score(mention, best), 3),
                "explanation": "synthetic",
            }
        chosen = parse_candidates(_section(prompt, "Proposed match"))
        others = parse_candidates(
            _section(prompt, "Other candidates that were available")
            or _section(prompt, "Other candidates")
        )
        chosen_score = _score(mention, chosen[0]) if chosen else 0.0
        better = sorted(
            (c for c in others if _score(mention, c) > chosen_score),
            key=lambda c: (-_score(mention, c), c.key),
        )
        if prompt.startswith("You are judging"):
            return "score", {
                "confidence_score": round(chosen_score, 3),
                "explanation": "synthetic",
                "better_candidate_keys": [better[0].key] if better else [],
                "better_queries": [],
            }
        # verify
        if chosen_score >= VERIFY_SUPPORT and not better:
            decision, preferred = "support", None
        elif better:
            decision, preferred = "disagree", better[0].key
        else:
            decision, preferred = "no_match", None
        return "verify", {
            "decision": decision,
            "preferred_key": preferred,
            "confidence_score": round(chosen_score, 3),
            "explanation": "synthetic",
        }

    @staticmethod
    def _rewrite(mention: str, prompt: str) -> dict[str, object]:
        tried = {normalise(q[2:]) for q in _section(prompt, "Queries already tried").splitlines()}
        words = normalise(mention).split()
        options = [
            " ".join(_singular(w) for w in words),
            " ".join(words[1:]),
            " ".join(words[:-1]),
        ]
        queries: list[str] = []
        for option in options:
            if option and option not in tried and option not in queries:
                queries.append(option)
        return {"queries": queries[:2], "explanation": "synthetic"}

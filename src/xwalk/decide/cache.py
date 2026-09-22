"""Ledger-backed decision caching.

Jev's probabilities move by a few hundredths between identical calls. A resumed run must
see the answers the first run saw, so every response is stored and replayed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from xwalk.decide.base import (
    DecisionClient,
    DecisionResponse,
    Question,
    question_to_dict,
    response_from_dict,
    response_to_dict,
)
from xwalk.fingerprint import hash_value
from xwalk.ledger import Ledger
from xwalk.records import Usage

CACHE_VERSION = 1


def decision_cache_key(
    client: DecisionClient, state: Any, questions: Mapping[str, Question]
) -> str:
    return hash_value(
        {
            "cache_version": CACHE_VERSION,
            "kind": "decision",
            "client": client.fingerprint,
            "state": state,
            "questions": {name: question_to_dict(q) for name, q in questions.items()},
        }
    )


class CachingDecider:
    def __init__(
        self, inner: DecisionClient, ledger: Ledger, *, read: bool = True, write: bool = True
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._read = read
        self._write = write
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        key = decision_cache_key(self._inner, state, questions)
        if self._read:
            cached = self._ledger.get_cached(key)
            if cached is not None:
                self.hits += 1
                stored = response_from_dict(json.loads(cached), questions)
                return DecisionResponse(
                    answers=stored.answers, model=stored.model, usage=Usage.zero()
                )
        response = await self._inner.decide(state, questions)
        self.misses += 1
        if self._write:
            await self._ledger.put_cached(
                key, json.dumps(response_to_dict(response), ensure_ascii=False)
            )
        return response

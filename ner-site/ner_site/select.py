"""Selection: one model picks the matching term among an entity's retrieved candidates.

Retrieval ranks; a selector decides. The visitor chooses the selector per request:

- ``retrieval`` (no selector): the fused ranking is the answer, as before;
- ``jev``: TypeSafe's Jev through ``xwalk.decide.DecisionMatcher`` (screen every
  candidate, choose among the survivors or none, gate the choice on the slots'
  properties; thresholds fitted on the pilots, ``POLICIES``);
- an LLM id (``openai/gpt-6-luna`` ...): ``xwalk.stages.select.Selector``, one call that
  returns an opaque candidate key or null.

Both read the question sets in ``ner_site/slots/`` (copied from the examples). Both only
ever resolve to a candidate retrieved for the entity: Jev chooses among keys, and an LLM's
key is resolved by exact lookup, so a malformed or invented answer is a failed pick, never
an id. Every failure (provider error, timeout, the request deadline) becomes a ``failed``
pick with its reason, never an exception.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ner_site.indexes import OntologyIndex
from ner_site.mapping import MAX_LISTED
from ner_site.ontologies import DOC_TEMPLATE, term_url
from ner_site.schema import EntityResult, Pick, PickStatus
from xwalk.decide.base import DecisionClient
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.llm.base import LLMClient
from xwalk.prompts.contract import PromptSet, PromptSlots, load_slots
from xwalk.records import Candidate, MatchStatus, Record, RetrievalHit
from xwalk.records import Usage as XwalkUsage
from xwalk.retrieval.base import SearchRequest
from xwalk.stages.choose import Chooser
from xwalk.stages.keying import Resolution
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.screen import Screener
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.templates import TemplateSet

RETRIEVAL_ONLY = "retrieval"
JEV = "jev"
JEV_MODEL = "~typesafe/jev-latest"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_TIMEOUT_S = 20.0
JEV_RETRIES = 1

SLOTS_DIR = Path(__file__).resolve().parent / "slots"
CONTEXT_CHARS = 150  # either side of the mention, for the selector's context
# Of the fused list (two retrievers at depth 20): one screen chunk, and exactly the
# candidates the page lists (matches plus more), so every pick is visible there.
MAX_CANDIDATES = MAX_LISTED
CONCURRENCY = 8  # entities selected at once
DEADLINE_NOTE = "cut off by the request deadline"

SELECT_TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ context_left }}[{{ mention }}]{{ context_right }}",
    doc=DOC_TEMPLATE,
    candidate=(
        "ID: {{ id }}\nLabel: {{ label }}\n"
        "{% if synonyms %}Synonyms: "
        "{% if synonyms is string %}{{ synonyms }}{% else %}{{ synonyms | join('; ') }}"
        "{% endif %}\n{% endif %}"
        "{% if definition %}Definition: {{ definition }}\n{% endif %}"
        "{% if organism %}Organism: {{ organism }}{% endif %}"
    ),
)

# The thresholds the examples' job_jev.yaml files fitted on the pilots (`xwalk fit`), per
# entity type; depth and chunking sized for the site (one screen call per entity).
_FITTED: dict[str, tuple[float, float, float]] = {  # accept_at, property_floor, choose_at
    "chemical": (0.65, 0.50, 0.70),
    "disease": (0.75, 0.50, 0.50),
    "food": (0.75, 0.70, 0.50),
    "gene": (0.85, 0.50, 0.70),
}
POLICIES: dict[str, DecisionPolicy] = {
    entity_type: DecisionPolicy(
        accept_at=accept,
        property_floor=floor,
        choose_at=choose,
        screen_floor=0.30,
        chunk_size=50,
        max_candidates=MAX_CANDIDATES,
        concurrency=CONCURRENCY,
    )
    for entity_type, (accept, floor, choose) in _FITTED.items()
}

_JEV_STATUS: dict[MatchStatus, PickStatus] = {
    MatchStatus.MATCHED: "match",
    MatchStatus.NEEDS_REVIEW: "review",
    MatchStatus.UNMATCHED: "no_match",
    MatchStatus.FAILED: "failed",
}


@dataclass
class SelectStats:
    """Filled in by `select_entities`: the selector's calls, tokens and reported cost.
    Tokens and cost are None once any call did not report them ("unknown", not 0)."""

    calls: int = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    cost_usd: float | None = 0.0
    cost_reported: bool = True  # False for selectors whose provider reports no cost

    def add(self, usage: XwalkUsage) -> None:
        self.calls += usage.calls
        if usage.unknown_calls:
            self.prompt_tokens = self.completion_tokens = self.cost_usd = None
            return
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
        if self.cost_usd is not None:
            self.cost_usd += usage.cost_usd

    @property
    def reported_cost(self) -> float | None:
        if self.calls == 0:
            return None
        return self.cost_usd if self.cost_reported else None


@cache
def slots(entity_type: str) -> PromptSlots:
    return load_slots(SLOTS_DIR / f"{entity_type}.yaml")


@cache
def _questions(entity_type: str) -> QuestionSet:
    return QuestionSet.from_slots(slots(entity_type))


@cache
def _prompts(entity_type: str) -> PromptSet:
    return PromptSet.from_slots(slots(entity_type))


class _Retrieved:
    """A retriever that answers with the candidates the site already retrieved, in the
    fused order, so the decision matcher re-ranks nothing and searches nothing."""

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._hits = [
            RetrievalHit(c.id, "retrieved", c.fused_score, rank)
            for rank, c in enumerate(candidates, start=1)
        ]

    @property
    def name(self) -> str:
        return "retrieved"

    @property
    def fingerprint(self) -> str:
        return "retrieved"

    @property
    def default_limit(self) -> int:
        return max(1, len(self._hits))

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return self._hits


def _source(entity: EntityResult, text: str) -> Record:
    return Record(
        entity.id,
        {
            "mention": entity.text,
            "context_left": text[max(0, entity.start - CONTEXT_CHARS) : entity.start],
            "context_right": text[entity.end : entity.end + CONTEXT_CHARS],
        },
    )


def _pick(
    selector: str,
    status: PickStatus,
    candidates: Sequence[Candidate],
    record_id: str | None = None,
    *,
    confidence: float | None = None,
    explanation: str = "",
) -> Pick:
    record = next((c.record for c in candidates if c.id == record_id), None)
    if record_id is not None and record is None:  # cannot happen; refuse rather than invent
        return Pick(selector=selector, status="failed", explanation="chose an unknown term")
    return Pick(
        selector=selector,
        # A match names a term. A review may not: Jev leaning to "none of these" below
        # the policy's none_at is needs_review with no id, an honest "unsure".
        status=status if record is not None or status != "match" else "failed",
        id=None if record is None else record.id,
        label=None if record is None else str(record.fields.get("label", "")),
        url=None if record is None else term_url(record.id),
        confidence=confidence,
        explanation=explanation,
    )


async def _with_jev(
    entity: EntityResult,
    index: OntologyIndex,
    candidates: Sequence[Candidate],
    text: str,
    decider: DecisionClient,
    stats: SelectStats,
) -> Pick:
    entity_type = entity.type
    questions = _questions(entity_type)
    policy = POLICIES[entity_type]
    matcher = DecisionMatcher(
        templates=SELECT_TEMPLATES,
        retrievers=[_Retrieved(candidates)],
        store=index.store,
        screener=Screener(
            decider,
            questions,
            SELECT_TEMPLATES,
            chunk_size=policy.chunk_size,
            shortlist_size=policy.shortlist_size,
            shortlist_floor=policy.shortlist_floor,
        ),
        chooser=Chooser(decider, questions, SELECT_TEMPLATES),
        gate=PropertyGate(decider, questions, SELECT_TEMPLATES),
        policy=policy,
        keep_candidates_in_trace=False,
    )
    result = await matcher.match(_source(entity, text))
    stats.add(result.usage)
    status = _JEV_STATUS[result.status]
    explanation = result.explanation
    error = result.attempts[0].error if result.attempts else None
    if status == "failed" and error:
        # A Jev call failed (after the screen, say): say which, not the partial signals.
        explanation = f"{error} (signals so far: {explanation})" if explanation else error
    return _pick(
        JEV,
        status,
        candidates,
        result.matched_id,
        confidence=result.confidence,
        explanation=explanation,
    )


async def _with_llm(
    entity: EntityResult,
    candidates: Sequence[Candidate],
    text: str,
    llm: LLMClient,
    selector: str,
    stats: SelectStats,
) -> Pick:
    source = _source(entity, text)
    chooser = Selector(
        llm,
        _prompts(entity.type),
        SELECT_TEMPLATES,
        policy=SelectorPolicy(max_candidates=MAX_CANDIDATES),
    )
    outcome = await chooser.select(source, SELECT_TEMPLATES.render_context(source), candidates)
    stats.add(outcome.usage)
    choice = outcome.choice
    if outcome.error is not None:
        return _pick(
            selector, "failed", candidates, explanation=f"unreadable answer: {outcome.error}"
        )
    if choice.resolution is Resolution.ABSTAIN:
        return _pick(
            selector,
            "no_match",
            candidates,
            confidence=outcome.confidence,
            explanation=outcome.explanation,
        )
    if choice.resolution is not Resolution.EXACT_KEY or choice.record_id is None:
        return _pick(
            selector,
            "failed",
            candidates,
            explanation="the model named no candidate it was shown",
        )
    return _pick(
        selector,
        "match",
        candidates,
        choice.record_id,
        confidence=outcome.confidence,
        explanation=outcome.explanation,
    )


async def select_entities(
    results: Sequence[EntityResult],
    candidates: Mapping[str, Sequence[Candidate]],
    indexes: Mapping[str, OntologyIndex],
    *,
    selector: str,
    text: str,
    stats: SelectStats,
    decider: DecisionClient | None = None,
    llm: LLMClient | None = None,
    deadline: float | None = None,
) -> list[EntityResult]:
    """`results` with ``pick`` set for every entity that was routed to an ontology and
    has candidates (None otherwise, and for ``retrieval``). `candidates` maps entity ids
    to their fused candidates; `deadline` (seconds from now) bounds the whole step."""
    if selector == RETRIEVAL_ONLY:
        return list(results)
    if selector == JEV and decider is None:
        raise ValueError("selector 'jev' needs a decider")
    if selector != JEV and llm is None:
        raise ValueError(f"selector {selector!r} needs an LLM client")
    stats.cost_reported = selector == JEV
    deadline_at = None if deadline is None else time.monotonic() + deadline
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(entity: EntityResult) -> EntityResult:
        found = list(candidates.get(entity.id, ()))[:MAX_CANDIDATES]
        index = indexes.get(entity.ontology) if entity.ontology else None
        if index is None or not found or entity.type not in POLICIES:
            return entity
        async with semaphore:
            remaining = None if deadline_at is None else deadline_at - time.monotonic()
            if remaining is not None and remaining <= 0:
                return entity.model_copy(
                    update={"pick": _pick(selector, "failed", found, explanation=DEADLINE_NOTE)}
                )
            try:
                if selector == JEV:
                    assert decider is not None
                    work = _with_jev(entity, index, found, text, decider, stats)
                else:
                    assert llm is not None
                    work = _with_llm(entity, found, text, llm, selector, stats)
                pick = await asyncio.wait_for(work, timeout=remaining)
            except asyncio.TimeoutError:
                pick = _pick(selector, "failed", found, explanation=DEADLINE_NOTE)
            except Exception as exc:  # noqa: BLE001 - a failed pick, never a 5xx
                pick = _pick(selector, "failed", found, explanation=f"{type(exc).__name__}: {exc}")
        return entity.model_copy(update={"pick": pick})

    return list(await asyncio.gather(*(one(e) for e in results)))


__all__ = [
    "JEV",
    "JEV_MODEL",
    "JEV_URL",
    "POLICIES",
    "RETRIEVAL_ONLY",
    "SELECT_TEMPLATES",
    "SelectStats",
    "select_entities",
]

"""One LLM call per clustering decision, reduced to a `Reply`.

Every call goes through `ask`: the request is built from a stored prompt, recoverable
provider errors and unparseable output become a `Reply` with `error` set (a per-source
outcome, never a crash), and a fatal error -- including a refused call budget --
propagates so the run stops at a transaction boundary.

Candidate selection reuses the matcher's opaque keys: the model answers a key issued for
this call or null, and anything else is unresolved (`stages.keying.resolve_key`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xwalk.cluster.prompts import SYSTEM, schema_for
from xwalk.llm.base import LLMClient, LLMError, LLMFatalError, LLMRequest, failure_usage
from xwalk.llm.parsing import parse_confidence, parse_json_object
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.keying import KeyedCandidates, Resolution, resolve_key


@dataclass(frozen=True)
class Reply:
    kind: str
    user: str
    raw: str | None
    payload: Mapping[str, Any] | None
    confidence: float | None
    explanation: str
    usage: Usage
    error: str | None = None
    finish_reason: str | None = None


async def ask(llm: LLMClient, kind: str, user: str) -> Reply:
    """Send one prompt. Fatal provider errors propagate; everything else is a Reply."""
    request = LLMRequest(system=SYSTEM, user=user, schema=schema_for(kind), schema_name=kind)
    try:
        response = await llm.complete(request)
    except LLMFatalError:
        raise
    except (LLMError, asyncio.TimeoutError) as exc:
        return Reply(
            kind,
            user,
            None,
            None,
            None,
            "",
            failure_usage(exc),
            error=f"provider_failure: {type(exc).__name__}: {exc}",
        )
    try:
        payload = parse_json_object(response.text)
    except LLMError as exc:
        return Reply(
            kind,
            user,
            response.text,
            None,
            None,
            "",
            response.usage,
            error=f"unparseable_output: {exc}",
            finish_reason=response.finish_reason,
        )
    confidence, invalid = parse_confidence(payload.get("confidence_score"))
    return Reply(
        kind,
        user,
        response.text,
        payload,
        confidence,
        str(payload.get("explanation", "")),
        response.usage,
        error=None if confidence is not None else f"invalid_output: {invalid}",
        finish_reason=response.finish_reason,
    )


def keyed_clusters(cluster_ids: Sequence[str], blocks: Sequence[str]) -> KeyedCandidates:
    """Opaque keys C01.. for the shown clusters. `blocks` are already rendered with keys."""
    order = tuple(issue_keys(len(cluster_ids)))
    by_key = {
        key: Candidate(record=Record(id=cid, fields={}), fused_score=0.0, evidence=())
        for key, cid in zip(order, cluster_ids, strict=True)
    }
    issued = {key: cid for key, cid in zip(order, cluster_ids, strict=True)}
    return KeyedCandidates(
        order=order, by_key=by_key, issued=issued, blocks=dict(zip(order, blocks, strict=True))
    )


def issue_keys(count: int) -> list[str]:
    """The keys `keyed_clusters` issues for `count` clusters, in order."""
    width = max(2, len(str(count)))
    return [f"C{i:0{width}d}" for i in range(1, count + 1)]


@dataclass(frozen=True)
class Choice:
    cluster_id: str | None
    resolution: str
    error: str | None


def read_choice(reply: Reply, keyed: KeyedCandidates) -> Choice:
    """The chosen cluster, an abstention, or an unresolved answer."""
    if reply.error and reply.payload is None:
        return Choice(None, "error", reply.error)
    assert reply.payload is not None
    raw_key = reply.payload.get("chosen_key")
    if raw_key is not None and not isinstance(raw_key, str):
        raw_key = str(raw_key)
    choice = resolve_key(raw_key, keyed)
    if choice.resolution is Resolution.UNRESOLVED:
        return Choice(None, "unresolved", f"unresolved_output: {raw_key!r} was not issued")
    if choice.resolution is Resolution.EXACT_KEY and reply.confidence is None:
        return Choice(None, "unresolved", reply.error)
    return Choice(choice.record_id, choice.resolution.value, None)


def read_label(reply: Reply, field: str, allowed: Sequence[Any]) -> Any | None:
    """A categorical field of the reply, or None when missing, invalid or unscored."""
    if reply.payload is None or reply.confidence is None:
        return None
    value = reply.payload.get(field)
    # exact type too: JSON 1 is not `true`, and "true" is not a boolean
    for option in allowed:
        if type(value) is type(option) and value == option:
            return option
    return None


__all__ = ["Choice", "Reply", "ask", "issue_keys", "keyed_clusters", "read_choice", "read_label"]

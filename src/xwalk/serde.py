"""Converting results to and from plain JSON-safe dicts.

The ledger stores one JSON blob per result. Keeping the conversion here means the
storage layer never has to know the shape of a MatchResult.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)


def _record_to_dict(record: Record) -> dict[str, Any]:
    return {"id": record.id, "fields": dict(record.fields)}


def _record_from_dict(data: Mapping[str, Any]) -> Record:
    return Record(id=data["id"], fields=data["fields"])


def _usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "calls": usage.calls,
    }


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "record": _record_to_dict(candidate.record),
        "fused_score": candidate.fused_score,
        "evidence": [
            {
                "record_id": hit.record_id,
                "retriever": hit.retriever,
                "raw_score": hit.raw_score,
                "rank": hit.rank,
            }
            for hit in candidate.evidence
        ],
    }


def _candidate_from_dict(data: Mapping[str, Any]) -> Candidate:
    return Candidate(
        record=_record_from_dict(data["record"]),
        fused_score=data["fused_score"],
        evidence=tuple(RetrievalHit(**hit) for hit in data["evidence"]),
    )


def _attempt_to_dict(attempt: Attempt) -> dict[str, Any]:
    return {
        "index": attempt.index,
        "query": attempt.query,
        "proposal": (
            None
            if attempt.proposal is None
            else {
                "kind": attempt.proposal.kind,
                "value": attempt.proposal.value,
                "source": attempt.proposal.source,
            }
        ),
        "candidates": [_candidate_to_dict(c) for c in attempt.candidates],
        "candidate_count": attempt.candidate_count,
        "candidates_truncated": attempt.candidates_truncated,
        "issued_keys": dict(attempt.issued_keys),
        "raw_selection": attempt.raw_selection,
        "chosen_id": attempt.chosen_id,
        "resolution": attempt.resolution,
        "primary_score": attempt.primary_score,
        "explanation": attempt.explanation,
        "verifier_decision": attempt.verifier_decision,
        "verifier_score": attempt.verifier_score,
        "verifier_preferred_id": attempt.verifier_preferred_id,
        "audited": attempt.audited,
        # JSON has no tuples; restored as tuple-of-tuples on the way back in.
        "dropped_proposals": [list(pair) for pair in attempt.dropped_proposals],
        "reason": None if attempt.reason is None else attempt.reason.value,
        "error": attempt.error,
        "usage": _usage_to_dict(attempt.usage),
        "elapsed_seconds": attempt.elapsed_seconds,
        "finish_reason": attempt.finish_reason,
    }


def _attempt_from_dict(data: Mapping[str, Any]) -> Attempt:
    return Attempt(
        index=data["index"],
        query=data["query"],
        proposal=None if data["proposal"] is None else RetryProposal(**data["proposal"]),
        candidates=tuple(_candidate_from_dict(c) for c in data["candidates"]),
        candidate_count=data["candidate_count"],
        candidates_truncated=data["candidates_truncated"],
        issued_keys=data["issued_keys"],
        raw_selection=data["raw_selection"],
        chosen_id=data["chosen_id"],
        resolution=data["resolution"],
        primary_score=data["primary_score"],
        explanation=data["explanation"],
        verifier_decision=data["verifier_decision"],
        verifier_score=data["verifier_score"],
        verifier_preferred_id=data["verifier_preferred_id"],
        audited=data["audited"],
        dropped_proposals=tuple((str(pair[0]), str(pair[1])) for pair in data["dropped_proposals"]),
        reason=None if data["reason"] is None else DecisionReason(data["reason"]),
        error=data["error"],
        usage=Usage(**data["usage"]),
        elapsed_seconds=data["elapsed_seconds"],
        finish_reason=data["finish_reason"],
    )


def result_to_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "result_key": result.result_key,
        "source_id": result.source_id,
        "source_hash": result.source_hash,
        "matched_id": result.matched_id,
        "matched_record": (
            None if result.matched_record is None else _record_to_dict(result.matched_record)
        ),
        "confidence": result.confidence,
        "status": result.status.value,
        "reason": result.reason.value,
        "explanation": result.explanation,
        "candidates": [_candidate_to_dict(c) for c in result.candidates],
        "attempts": [_attempt_to_dict(a) for a in result.attempts],
        "usage": _usage_to_dict(result.usage),
        "elapsed_seconds": result.elapsed_seconds,
        "run_fingerprint": result.run_fingerprint,
    }


def result_from_dict(data: Mapping[str, Any]) -> MatchResult:
    return MatchResult(
        result_key=data["result_key"],
        source_id=data["source_id"],
        source_hash=data["source_hash"],
        matched_id=data["matched_id"],
        matched_record=(
            None if data["matched_record"] is None else _record_from_dict(data["matched_record"])
        ),
        confidence=data["confidence"],
        status=MatchStatus(data["status"]),
        reason=DecisionReason(data["reason"]),
        explanation=data["explanation"],
        candidates=tuple(_candidate_from_dict(c) for c in data["candidates"]),
        attempts=tuple(_attempt_from_dict(a) for a in data["attempts"]),
        usage=Usage(**data["usage"]),
        elapsed_seconds=data["elapsed_seconds"],
        run_fingerprint=data["run_fingerprint"],
    )

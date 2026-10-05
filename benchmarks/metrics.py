"""Metrics for the pairwise and source-to-catalog tracks.

Every method -- a string baseline or a full xwalk run -- is reduced to one `Prediction`
per source record, and every metric is computed from those by the same code, so methods
are compared on identical definitions.

Definitions (gold is restricted to the catalog; an empty gold set means "no match is
correct"):

- accepted: status ``matched``. Only accepted predictions count as system output for
  final precision/recall/F1. ``needs_review`` is human work, reported as review rate,
  and its proposals appear only in ``recall_any_status``.
- final_precision = accepted and correct / accepted (includes accepts on no-match rows)
- final_recall    = accepted and correct / rows with non-empty gold
- recall_any_status: correct proposed id at any status / rows with non-empty gold
- no_match_precision / no_match_recall: predicted ``unmatched`` vs empty gold
- false_accept_on_no_match: accepted rows whose gold is empty
- candidate_recall: gold id among the retrieved candidates / rows with non-empty gold
- truncated: gold retrieved but cut by the candidate budget before the model saw it
- recovered / broken_by_retry: first-attempt choice wrong (or none) and final accepted
  correct, and the reverse -- a retry that turned a correct first choice into a wrong or
  unaccepted final answer
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from xwalk.evaluate import GoldSet

STATUSES = ("matched", "needs_review", "unmatched", "failed")


@dataclass(frozen=True)
class Prediction:
    source_id: str
    status: str
    predicted_id: str | None
    confidence: float | None = None
    # None: the method has no retrieval stage (exact, fuzzy)
    retrieved_ids: tuple[str, ...] | None = None
    shown_ids: tuple[str, ...] | None = None
    # The first attempt's choice, for methods that retry; None when not applicable
    first_choice_id: str | None = None
    has_first_choice: bool = False
    attempts: int = 0


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None if precision is None or recall is None else 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def matching_metrics(
    predictions: Iterable[Prediction], gold: GoldSet, ids: Sequence[str] | None = None
) -> dict[str, Any]:
    keep = set(ids) if ids is not None else None
    rows = [p for p in predictions if p.source_id in gold and (keep is None or p.source_id in keep)]

    def correct(p: Prediction, rid: str | None) -> bool:
        return rid is not None and rid in (gold.get(p.source_id) or frozenset())

    with_gold = [p for p in rows if gold.get(p.source_id)]
    no_gold = [p for p in rows if not gold.get(p.source_id)]
    accepted = [p for p in rows if p.status == "matched"]
    accepted_correct = [p for p in accepted if correct(p, p.predicted_id)]
    predicted_nomatch = [p for p in rows if p.status == "unmatched"]

    precision = _ratio(len(accepted_correct), len(accepted))
    recall = _ratio(len(accepted_correct), len(with_gold))
    out: dict[str, Any] = {
        "n": len(rows),
        "with_gold": len(with_gold),
        "gold_no_match": len(no_gold),
        "status_counts": {s: sum(1 for p in rows if p.status == s) for s in STATUSES},
        "accepted": len(accepted),
        "accepted_correct": len(accepted_correct),
        "accepted_precision": precision,
        "coverage": _ratio(len(accepted), len(rows)),
        "review_rate": _ratio(sum(1 for p in rows if p.status == "needs_review"), len(rows)),
        "final_precision": precision,
        "final_recall": recall,
        "final_f1": f1(precision, recall),
        "recall_any_status": _ratio(
            sum(1 for p in with_gold if correct(p, p.predicted_id)), len(with_gold)
        ),
        "no_match_precision": _ratio(
            sum(1 for p in predicted_nomatch if not gold.get(p.source_id)), len(predicted_nomatch)
        ),
        "no_match_recall": _ratio(sum(1 for p in no_gold if p.status == "unmatched"), len(no_gold)),
        "false_accept_on_no_match": sum(1 for p in no_gold if p.status == "matched"),
    }

    retrieving = [p for p in with_gold if p.retrieved_ids is not None]
    if retrieving:
        golds = {p.source_id: gold.get(p.source_id) or frozenset() for p in retrieving}
        in_retrieved = [p for p in retrieving if golds[p.source_id] & set(p.retrieved_ids or ())]
        shown = [
            p
            for p in retrieving
            if p.shown_ids is not None and golds[p.source_id] & set(p.shown_ids)
        ]
        truncated = [p for p in in_retrieved if p.shown_ids is not None and p not in shown]
        out["candidate_recall"] = _ratio(len(in_retrieved), len(retrieving))
        out["candidate_recall_shown"] = (
            _ratio(len(shown), len(retrieving))
            if any(p.shown_ids is not None for p in retrieving)
            else None
        )
        out["truncated"] = len(truncated)
        out["mean_candidates_retrieved"] = round(
            mean(len(p.retrieved_ids or ()) for p in retrieving), 2
        )
    else:
        out["candidate_recall"] = None
        out["candidate_recall_shown"] = None
        out["truncated"] = None
        out["mean_candidates_retrieved"] = None

    retrying = [p for p in rows if p.has_first_choice]
    if retrying:
        final_ok = {p.source_id for p in accepted_correct}
        out["recovered"] = sum(
            1 for p in retrying if not correct(p, p.first_choice_id) and p.source_id in final_ok
        )
        out["broken_by_retry"] = sum(
            1
            for p in retrying
            if correct(p, p.first_choice_id) and p.source_id not in final_ok and p.attempts > 1
        )
        out["mean_attempts"] = round(mean(p.attempts for p in retrying), 3)
    else:
        out["recovered"] = out["broken_by_retry"] = out["mean_attempts"] = None
    return out


def pairwise_metrics(labels: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    """Binary classification over a preblocked pair table."""
    if len(labels) != len(predicted):
        raise ValueError("labels and predictions differ in length")
    tp = sum(1 for y, p in zip(labels, predicted, strict=True) if y and p)
    fp = sum(1 for y, p in zip(labels, predicted, strict=True) if not y and p)
    fn = sum(1 for y, p in zip(labels, predicted, strict=True) if y and not p)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "pairs": len(labels),
        "positives": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
    }

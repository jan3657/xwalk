"""Reading a completed run and answering "is this good, and where should I spend effort?".

Nothing in this package calls an LLM or a retriever. It reads the ledger, so evaluation
is free, repeatable, and safe to run on a machine with no credentials.
"""

from xwalk.evaluate.gold import GoldSet, load_gold_csv, load_gold_jsonl
from xwalk.evaluate.metrics import (
    EvalReport,
    ThresholdPoint,
    calibration_warning,
    evaluate_results,
    evaluate_run,
    threshold_curve,
)

__all__ = [
    "EvalReport",
    "GoldSet",
    "ThresholdPoint",
    "calibration_warning",
    "evaluate_results",
    "evaluate_run",
    "load_gold_csv",
    "load_gold_jsonl",
    "threshold_curve",
]

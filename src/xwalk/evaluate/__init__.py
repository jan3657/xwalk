"""Reading a completed run and answering "is this good, and where should I spend effort?".

Nothing in this package calls an LLM or a retriever. It reads the ledger, so evaluation
is free, repeatable, and safe to run on a machine with no credentials.
"""

from xwalk.evaluate.ceiling import (
    CeilingBucket,
    CeilingReport,
    ceiling_report,
    classify_ceiling,
)
from xwalk.evaluate.failures import FailureCase, PromptRole, render_failure, select_failures
from xwalk.evaluate.gold import GoldSet, load_gold_csv, load_gold_jsonl
from xwalk.evaluate.metrics import (
    EvalReport,
    ThresholdPoint,
    calibration_warning,
    evaluate_results,
    evaluate_run,
    threshold_curve,
)
from xwalk.evaluate.partition import (
    Partition,
    Partitioner,
    ids_in,
    load_partition_file,
    partition_of,
    write_partition_file,
)
from xwalk.evaluate.report import FullReport, evaluate, render_report, write_report

__all__ = [
    "CeilingBucket",
    "CeilingReport",
    "EvalReport",
    "FailureCase",
    "FullReport",
    "GoldSet",
    "Partition",
    "Partitioner",
    "PromptRole",
    "ThresholdPoint",
    "calibration_warning",
    "ceiling_report",
    "classify_ceiling",
    "evaluate",
    "evaluate_results",
    "evaluate_run",
    "ids_in",
    "load_gold_csv",
    "load_gold_jsonl",
    "load_partition_file",
    "partition_of",
    "render_failure",
    "render_report",
    "select_failures",
    "threshold_curve",
    "write_partition_file",
    "write_report",
]

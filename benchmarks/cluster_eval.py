"""Score predicted cluster assignments against gold clusters.

    python -m benchmarks.cluster_eval --gold gold_clusters.csv \\
        --pred run_a.csv [--pred run_b_reversed.csv ...] [--meta usage.json] [--out result.json]

Files: CSV with a header ``item_id,cluster_id[,outcome]`` or JSONL objects with those
keys (see benchmarks/clustering.py). The first ``--pred`` is scored; every ``--pred``
together feeds order sensitivity (pass the same items run in different input orders).
``--meta`` is any JSON object (calls, tokens, seconds, model, ...) copied into the output
verbatim, so cost travels with quality. Calls nothing; reads files only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.clustering import evaluate_clustering, order_sensitivity, read_assignments


def evaluate_files(
    gold_path: Path, pred_paths: Sequence[Path], meta_path: Path | None = None
) -> dict[str, Any]:
    gold, _ = read_assignments(gold_path)
    predictions = [read_assignments(p) for p in pred_paths]
    clusters, outcomes = predictions[0]
    result = {
        "gold": str(gold_path),
        "predictions": [str(p) for p in pred_paths],
        "scores": evaluate_clustering(clusters, outcomes, gold),
        "order_sensitivity": order_sensitivity([c for c, _ in predictions]),
        "meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path else None,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.cluster_eval")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, action="append", required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_files(args.gold, args.pred, args.meta)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

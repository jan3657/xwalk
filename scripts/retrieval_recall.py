#!/usr/bin/env python
"""Offline retrieval recall@k of a job's retrievers against a gold file.

Builds the job's indexes into a temporary directory, retrieves for every labelled source
record exactly as the matchers do, and prints the fraction of rows (gold "no match" rows
excluded) with a gold id in the top k fused candidates. No LLM, no decider, no API key.

Usage: scripts/retrieval_recall.py JOB GOLD_CSV [--k 200] [--k 50 ...]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from xwalk.config import load_job
from xwalk.evaluate.gold import load_gold_csv
from xwalk.evaluate.recall import retrieval_recall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("job", type=Path, help="job YAML")
    parser.add_argument("gold", type=Path, help="gold CSV (source_id,gold_ids)")
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        help="cut-off to report; repeatable (default: 10, 50, 200)",
    )
    args = parser.parse_args(argv)

    job = load_job(args.job)
    gold = load_gold_csv(args.gold)
    ks = tuple(args.k) if args.k else (10, 50, 200)
    with tempfile.TemporaryDirectory(prefix="xwalk-recall-") as index_dir:
        recall = retrieval_recall(job, gold, index_dir=Path(index_dir), ks=ks)
    for k in ks:
        print(f"recall@{k}: {recall[k]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

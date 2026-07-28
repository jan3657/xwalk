"""Score a completed run against the gold labels.

    python -m examples.chemistry.evaluate_run --run run/ --fingerprint <fp>

Calls no LLM and no retriever -- it reads `run/ledger.sqlite`. Free, repeatable, and
runnable on a machine with no credentials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xwalk.evaluate import Partitioner, evaluate, load_gold_csv, render_report, write_report
from xwalk.ledger import Ledger

HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run/", help="the run directory")
    parser.add_argument("--fingerprint", required=True, help="run fingerprint to score")
    parser.add_argument("--gold", default=str(HERE / "gold.csv"))
    parser.add_argument(
        "--partitions", action="store_true", help="also report the three-way split sizes"
    )
    args = parser.parse_args(argv)

    gold = load_gold_csv(args.gold)
    ledger = Ledger.open(Path(args.run) / "ledger.sqlite")
    try:
        report = evaluate(
            ledger,
            args.fingerprint,
            gold,
            partitioner=Partitioner() if args.partitions else None,
        )
    finally:
        ledger.close()

    print(render_report(report))
    write_report(report, Path(args.run) / "eval")
    print()
    print(f"written: {Path(args.run) / 'eval.json'} and {Path(args.run) / 'eval.txt'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

#!/usr/bin/env python
"""Compare a candidate decider run against the recorded baseline, domain by domain.

Reads `<dir>/<example>/eval.json` (accuracy is `metrics.recall_at_any_status`, the number
recorded as each domain's decider result) and the `tokens / record` line of
`<dir>/<example>/eval.txt` on both sides. The gate passes when

- every domain's accuracy is at most 0.02 below its baseline,
- the mean accuracy over the domains is not lower, and
- tokens per record are at least 50% lower in every domain.

Exit 0 on pass, 1 on fail.

Usage: scripts/compare_gates.py --baseline runs/jev_eval --candidate runs/jev_eval_v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXAMPLES = ("cafeteria_fcd", "chebi", "ncbi_disease", "nlm_gene")
ACCURACY_TOLERANCE = 0.02
MAX_TOKEN_RATIO = 0.5
_TOKENS = re.compile(r"tokens / record\s*:\s*([0-9.]+)")


def accuracy(run: Path) -> float:
    metrics = json.loads((run / "eval.json").read_text(encoding="utf-8"))["metrics"]
    return float(metrics["recall_at_any_status"])


def tokens_per_record(run: Path) -> float:
    found = _TOKENS.search((run / "eval.txt").read_text(encoding="utf-8"))
    if found is None:
        raise ValueError(f"{run / 'eval.txt'} has no 'tokens / record' line")
    return float(found.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--examples", nargs="+", default=list(EXAMPLES))
    args = parser.parse_args(argv)

    failures: list[str] = []
    deltas: list[float] = []
    print(
        f"{'domain':<15} {'acc base':>8} {'acc cand':>8} {'delta':>7}   "
        f"{'tok base':>9} {'tok cand':>9} {'ratio':>6}"
    )
    for ex in args.examples:
        base, cand = args.baseline / ex, args.candidate / ex
        acc_base, acc_cand = accuracy(base), accuracy(cand)
        tok_base, tok_cand = tokens_per_record(base), tokens_per_record(cand)
        delta, ratio = acc_cand - acc_base, tok_cand / tok_base
        deltas.append(delta)
        print(
            f"{ex:<15} {acc_base:>8.2f} {acc_cand:>8.2f} {delta:>+7.2f}   "
            f"{tok_base:>9.1f} {tok_cand:>9.1f} {ratio:>6.2f}"
        )
        # Rounded so a delta of exactly -0.02 on 50 rows is not failed by float noise.
        if round(delta, 6) < -ACCURACY_TOLERANCE:
            failures.append(f"{ex}: accuracy {delta:+.2f} is below -{ACCURACY_TOLERANCE}")
        if ratio > MAX_TOKEN_RATIO:
            failures.append(f"{ex}: tokens per record ratio {ratio:.2f} > {MAX_TOKEN_RATIO}")

    mean_delta = sum(deltas) / len(deltas)
    print(f"mean accuracy delta: {mean_delta:+.3f}")
    if round(mean_delta, 6) < 0:
        failures.append(f"mean accuracy is lower by {-mean_delta:.3f}")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        f"PASS: every domain within -{ACCURACY_TOLERANCE}, mean delta {mean_delta:+.3f} >= 0, "
        f"every token ratio <= {MAX_TOKEN_RATIO}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

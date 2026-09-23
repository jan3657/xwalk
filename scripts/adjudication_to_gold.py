#!/usr/bin/env python
"""Turn the adjudicated Qwen/Jev disagreement sample into a gold CSV.

Biased sample: only Qwen/Jev disagreements from the 2026-09-22 run. Rows where the two
deciders agreed are absent, so metrics against this file describe the contested rows, not
the dataset as a whole.

Verdicts map to gold ids as: ``jev`` -> jev_id, ``qwen`` -> qwen_id, ``both_acceptable``
-> ``qwen_id|jev_id``. ``both_wrong`` and ``unsure`` rows are left out (unlabelled); any
other verdict is an error. Output is sorted by source_id so re-running is byte-identical.

A blank id means that decider answered "no match". A winning blank side becomes an empty
gold cell, which `load_gold_csv` reads as an explicit no-match label. In a
``both_acceptable`` row the blank side is dropped, because a gold set cannot hold "this id
or no match". Those rows keep only the one named id.

Usage: scripts/adjudication_to_gold.py [IN.csv] [OUT.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_IN = Path("docs/superpowers/specs/2026-09-22-jev-adjudicated-sample.csv")
DEFAULT_OUT = Path("examples/ref_zivila/gold_adjudicated.csv")
SKIPPED = frozenset({"both_wrong", "unsure"})


def gold_ids(row: dict[str, str]) -> str | None:
    """The gold cell for one adjudicated row, or None when the row stays unlabelled."""
    verdict = row["verdict"].strip()
    qwen_id, jev_id = row["qwen_id"].strip(), row["jev_id"].strip()
    if verdict == "jev":
        return jev_id
    if verdict == "qwen":
        return qwen_id
    if verdict == "both_acceptable":
        return "|".join(i for i in (qwen_id, jev_id) if i)
    if verdict in SKIPPED:
        return None
    raise ValueError(f"source {row['source_id']!r}: unknown verdict {verdict!r}")


def convert(source: Path, out: Path) -> int:
    """Write the gold CSV for `source` to `out`; returns the number of labelled rows."""
    with source.open(encoding="utf-8", newline="") as handle:
        labelled = {
            row["source_id"]: cell
            for row in csv.DictReader(handle)
            if (cell := gold_ids(row)) is not None
        }
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source_id", "gold_ids"])
        writer.writerows(sorted(labelled.items()))
    return len(labelled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_IN, help="adjudicated CSV")
    parser.add_argument("out", type=Path, nargs="?", default=DEFAULT_OUT, help="gold CSV to write")
    args = parser.parse_args(argv)
    count = convert(args.source, args.out)
    print(f"wrote {count} labelled rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

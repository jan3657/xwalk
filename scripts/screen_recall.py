#!/usr/bin/env python
"""Retrieval, shortlist, and final recall of a decider run against gold.

Three numbers, measured on the labelled rows of one finished run:

- retrieval : a gold id is among the candidates the retrievers returned
- shortlist : a gold id survived the screen, i.e. it is one of the records whose
              per-candidate probability the decider matcher kept as `screen_<key>`
- final     : `matched_id` is a gold id

A large retrieval-to-shortlist drop means the questions are wrong; a low retrieval
recall means no amount of question work will help, and the effort belongs upstream.

Usage: scripts/screen_recall.py RUN_DIR GOLD_CSV [LABEL]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from xwalk.evaluate.gold import load_gold_csv
from xwalk.ledger import Ledger

_PREFIX = "screen_C"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    run, gold_path = Path(argv[1]), argv[2]
    label = argv[3] if len(argv) > 3 else run.name

    gold = load_gold_csv(gold_path)
    fingerprint = str(json.loads((run / "manifest.json").read_text())["run_fingerprint"])
    ledger = Ledger.open(run / "ledger.sqlite")
    retrieved = shortlisted = final = labelled = 0
    try:
        for result in ledger.iter_results(fingerprint):
            want = gold.get(result.source_id)
            if not want:  # unlabelled, or labelled "no match": no id to find
                continue
            labelled += 1
            attempt = result.attempts[0]
            candidates = {c.id for c in attempt.candidates} | set(attempt.issued_keys.values())
            short = {
                attempt.issued_keys[key[len("screen_") :]]
                for key in attempt.signals
                if key.startswith(_PREFIX)
            }
            retrieved += bool(want & candidates)
            shortlisted += bool(want & short)
            final += result.matched_id in want
    finally:
        ledger.close()

    if not labelled:
        print(f"{label}: no labelled rows with a gold id")
        return 1
    print(
        f"{label}: labelled {labelled} | retrieval {retrieved / labelled:.2f} "
        f"shortlist {shortlisted / labelled:.2f} final {final / labelled:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

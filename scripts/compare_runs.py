#!/usr/bin/env python
"""Compare and ensemble two xwalk mapping runs for Ref_zivila."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row.get("source_id") or "").strip(): row
            for row in csv.DictReader(handle)
            if (row.get("source_id") or "").strip()
        }


def load_source(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("ID") or "").strip(): row
            for row in csv.DictReader(handle)
            if (row.get("ID") or "").strip()
        }


def analyze_comparison(
    run_a_path: Path,
    run_b_path: Path,
    source_path: Path,
    name_a: str = "Model A",
    name_b: str = "Model B",
    ensemble_out: Path | None = None,
) -> dict[str, Any]:
    map_a = load_mapping(run_a_path)
    map_b = load_mapping(run_b_path)
    source = load_source(source_path)

    all_ids = list(source.keys())
    total = len(all_ids)

    status_a = Counter(m.get("status", "missing") for m in map_a.values())
    status_b = Counter(m.get("status", "missing") for m in map_b.values())

    both_matched = 0
    both_agreed = 0
    disagreed = []
    only_a = []
    only_b = []
    both_unmatched = 0

    ensemble_rows = []

    for sid in all_ids:
        src_row = source[sid]
        rec_a = map_a.get(sid, {})
        rec_b = map_b.get(sid, {})

        stat_a = rec_a.get("status", "")
        stat_b = rec_b.get("status", "")

        id_a = (rec_a.get("matched_id") or "").strip()
        id_b = (rec_b.get("matched_id") or "").strip()

        matched_a = stat_a in ("matched", "accepted") and bool(id_a)
        matched_b = stat_b in ("matched", "accepted") and bool(id_b)

        final_id = ""
        final_status = "unmatched"
        ensemble_method = "none"

        if matched_a and matched_b:
            both_matched += 1
            if id_a == id_b:
                both_agreed += 1
                final_id = id_a
                final_status = "matched"
                ensemble_method = "consensus"
            else:
                disagreed.append(
                    {
                        "id": sid,
                        "food": src_row.get("NAME_ENG") or src_row.get("NAME_SLO"),
                        "id_a": id_a,
                        "id_b": id_b,
                        "conf_a": rec_a.get("confidence"),
                        "conf_b": rec_b.get("confidence"),
                    }
                )
                # Pick the higher confidence or flag for review
                c_a = float(rec_a.get("confidence") or 0.0)
                c_b = float(rec_b.get("confidence") or 0.0)
                if abs(c_a - c_b) >= 0.2:
                    final_id = id_a if c_a > c_b else id_b
                    final_status = "matched"
                    ensemble_method = f"higher_confidence_{name_a if c_a > c_b else name_b}"
                else:
                    final_id = id_a
                    final_status = "needs_review"
                    ensemble_method = "divergent_review"
        elif matched_a and not matched_b:
            only_a.append(sid)
            final_id = id_a
            final_status = (
                "matched" if float(rec_a.get("confidence") or 0.0) >= 0.85 else "needs_review"
            )
            ensemble_method = f"single_{name_a}"
        elif matched_b and not matched_a:
            only_b.append(sid)
            final_id = id_b
            final_status = (
                "matched" if float(rec_b.get("confidence") or 0.0) >= 0.85 else "needs_review"
            )
            ensemble_method = f"single_{name_b}"
        else:
            both_unmatched += 1
            final_status = "unmatched"
            ensemble_method = "both_unmatched"

        ensemble_rows.append(
            {
                "source_id": sid,
                "matched_id": final_id,
                "status": final_status,
                "method": ensemble_method,
                "id_a": id_a,
                "id_b": id_b,
            }
        )

    if ensemble_out:
        ensemble_out.parent.mkdir(parents=True, exist_ok=True)
        with ensemble_out.open("w", encoding="utf-8", newline="") as h:
            writer = csv.DictWriter(
                h, fieldnames=["source_id", "matched_id", "status", "method", "id_a", "id_b"]
            )
            writer.writeheader()
            writer.writerows(ensemble_rows)

    return {
        "total": total,
        "status_a": dict(status_a),
        "status_b": dict(status_b),
        "both_matched": both_matched,
        "both_agreed": both_agreed,
        "agreement_rate_when_matched": (both_agreed / both_matched) if both_matched else 0.0,
        "disagreed_count": len(disagreed),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "both_unmatched_count": both_unmatched,
        "sample_disagreements": disagreed[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two mapping runs")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/Ref_zivila.csv"))
    parser.add_argument("--name-a", type=str, default="Qwen")
    parser.add_argument("--name-b", type=str, default="Nex")
    parser.add_argument("--ensemble-out", type=Path, default=None)
    args = parser.parse_args()

    res = analyze_comparison(
        run_a_path=args.run_a,
        run_b_path=args.run_b,
        source_path=args.source,
        name_a=args.name_a,
        name_b=args.name_b,
        ensemble_out=args.ensemble_out,
    )

    print("\n" + "=" * 60)
    print(f"COMPARISON REPORT: {args.name_a} vs {args.name_b}")
    print("=" * 60)
    print(f"Total source rows: {res['total']}")
    print(f"{args.name_a} statuses: {res['status_a']}")
    print(f"{args.name_b} statuses: {res['status_b']}")
    print(f"Rows matched by BOTH models: {res['both_matched']}")
    print(
        f"Rows with EXACT SAME FoodOn concept: {res['both_agreed']} "
        f"({res['agreement_rate_when_matched'] * 100:.1f}%)"
    )
    print(f"Disagreed concept matches: {res['disagreed_count']}")
    print(f"Matched ONLY by {args.name_a}: {res['only_a_count']}")
    print(f"Matched ONLY by {args.name_b}: {res['only_b_count']}")
    print(f"Both unmatched: {res['both_unmatched_count']}")

    if res["sample_disagreements"]:
        print("\nSample Disagreements:")
        for d in res["sample_disagreements"]:
            print(
                f"  * {d['food']}: {args.name_a}={d['id_a']} (conf {d['conf_a']}) "
                f"vs {args.name_b}={d['id_b']} (conf {d['conf_b']})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

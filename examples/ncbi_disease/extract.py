"""Extract the NCBI Disease sample slice. Run once, by hand.

Sources (OntoRAG paper repo):
  data/datasets/ncbi_disease/test.jsonl.gz  -> the first 50 mentions
  data/ctd_diseases/ontology_dump.json      -> 200 CTD disease terms: every gold target
                                               of those mentions, then sibling distractors

TSV rather than OWL, because the point of this dataset here is user-supplied gold alias
expansion (MESH/OMIM), not the loader.
"""

from __future__ import annotations

from pathlib import Path

from examples._paper_extract import (
    choose_targets,
    read_mentions,
    read_ontology,
    report,
    write_gold_csv,
    write_mentions_jsonl,
    write_targets_table,
)

HERE = Path(__file__).parent


def main() -> int:
    mentions = read_mentions("ncbi_disease", 50)
    ontology = read_ontology("ctd_diseases")
    gold = [g for row in mentions for g in row.get("gold_ids") or []]
    targets = choose_targets(ontology, gold, total=200)

    write_targets_table(targets, HERE / "sample/targets.tsv")
    write_mentions_jsonl(mentions, HERE / "sample/mentions.jsonl")
    write_gold_csv(mentions, HERE / "sample/gold.csv")
    report("ncbi_disease", targets, mentions, HERE / "sample")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

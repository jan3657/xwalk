"""Extract the NLM-Gene sample slice. Run once, by hand.

Sources (OntoRAG paper repo):
  data/datasets/nlm_gene/test.jsonl.gz  -> the first 50 mentions
  data/ncbi_gene/ontology_dump.json     -> 200 NCBI Gene records: every gold target of
                                           those mentions, then sibling distractors

Keeps `organism`, because a bare gene symbol is ambiguous across species and that
ambiguity is what this example exists to exercise.
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
    mentions = read_mentions("nlm_gene", 50)
    ontology = read_ontology("ncbi_gene")
    gold = [g for row in mentions for g in row.get("gold_ids") or []]
    targets = choose_targets(ontology, gold, total=200)

    write_targets_table(targets, HERE / "sample/targets.tsv", extra_columns=("organism", "tax_id"))
    write_mentions_jsonl(mentions, HERE / "sample/mentions.jsonl")
    write_gold_csv(mentions, HERE / "sample/gold.csv")
    report("nlm_gene", targets, mentions, HERE / "sample")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

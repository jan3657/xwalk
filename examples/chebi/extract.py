"""Extract the CRAFT ChEBI sample slice. Run once, by hand.

Sources (OntoRAG paper repo):
  data/datasets/craft_chebi/test.jsonl.gz   -> the first 50 mentions
  data/craft_chebi/ontology_dump.json       -> 200 ChEBI terms: every gold target of
                                               those mentions, then sibling distractors

Emits RDF/XML so this example exercises the OWL loader rather than the CSV one.
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
    write_targets_owl,
)

HERE = Path(__file__).parent


def main() -> int:
    mentions = read_mentions("craft_chebi", 50)
    ontology = read_ontology("craft_chebi")
    gold = [g for row in mentions for g in row.get("gold_ids") or []]
    targets = choose_targets(ontology, gold, total=200)

    write_targets_owl(targets, HERE / "sample/targets.owl")
    write_mentions_jsonl(mentions, HERE / "sample/mentions.jsonl")
    write_gold_csv(mentions, HERE / "sample/gold.csv")
    report("chebi", targets, mentions, HERE / "sample")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

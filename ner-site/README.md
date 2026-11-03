# ner-site: NER-to-ontology demo on Vercel

A minimal web page where a visitor pastes text and an extraction prompt; one LLM call
extracts named entities (chemical, disease, gene, food, other) with character spans, and
xwalk retrieval maps each entity onto the matching bundled sample ontology (ChEBI, CTD
disease, NCBI Gene, FoodOn).

This directory is a separate Vercel project (Root Directory `ner-site`). It is not part
of the `xwalk` package and is not installed with it. The plan, contracts and workstreams
are in [docs/ner-site/PLAN.md](../docs/ner-site/PLAN.md).

- `ner_site/`: all logic, importable and tested offline from `tests/test_site_*.py`
  (pytest puts `ner-site/` on `sys.path`).

## Deploy

Deploy instructions: see section 11 of the plan; WS-C fills this in.

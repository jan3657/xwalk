# CRAFT ChEBI example

Matching chemical entity mentions in full-text articles to ChEBI ontology terms. The
simplest source shape — one mention field plus sentence context — against an
ontology-shaped target.

What it exercises: the **OWL loader** (`target.kind: owl`, parsed with rdflib behind
`xwalk[ontology]`), and long-document context, where the surrounding paragraph decides
which of several chemically-related terms is meant.

## Files

| File | What it is |
|---|---|
| `job.yaml` | the whole run, as serialized constructor arguments |
| `slots.yaml` | the domain vocabulary — anomers, conjugate bases, salts |
| `sample/targets.owl` | 200 ChEBI terms in RDF/XML |
| `sample/mentions.jsonl` | 50 mentions with sentence context |
| `sample/gold.csv` | `source_id,gold_ids` |
| `extract.py` | how the slice was taken from the paper repo; run once, by hand |

## Run

```bash
pip install 'xwalk[ontology]'
export OPENAI_API_KEY=...
xwalk index --job examples/chebi/job.yaml --out runs/chebi/index
xwalk match --job examples/chebi/job.yaml --out runs/chebi
xwalk eval  --run runs/chebi --gold examples/chebi/sample/gold.csv
```

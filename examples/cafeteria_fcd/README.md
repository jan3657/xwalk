# CafeteriaFCD example

Matching food and ingredient mentions to FoodOn ontology terms.

This example exists to make one claim checkable: **`slots.yaml` is the only
domain-specific part.** Compare it with `examples/chebi/` — same `job.yaml` shape, same
`target.kind: owl`, same templates, same retriever, same policy. Chemistry and food
composition share no vocabulary, and the library code is identical for both.

Many mentions here carry **more than one** gold ID (a foodstuff often has both a
specific and a general FoodOn term), which is why `gold_ids` is a `|`-separated list and
why every metric counts any listed ID as correct.

## Files

| File | What it is |
|---|---|
| `job.yaml` | identical in shape to `examples/chebi/job.yaml` |
| `slots.yaml` | **the only file that differs in kind** — processing state, dish vs ingredient |
| `sample/targets.owl` | 200 FoodOn terms in RDF/XML |
| `sample/mentions.jsonl` | 50 mentions |
| `sample/gold.csv` | `source_id,gold_ids`, frequently multi-valued |
| `extract.py` | how the slice was taken from the paper repo; run once, by hand |

## Run

```bash
pip install 'xwalk[ontology]'
export OPENAI_API_KEY=...
xwalk index --job examples/cafeteria_fcd/job.yaml --out runs/cafeteria/index
xwalk match --job examples/cafeteria_fcd/job.yaml --out runs/cafeteria
xwalk eval  --run runs/cafeteria --gold examples/cafeteria_fcd/sample/gold.csv
```

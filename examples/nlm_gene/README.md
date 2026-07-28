# NLM-Gene example

The dataset that motivates opaque candidate keys. NCBI gene IDs are **numeric**
(`NCBIGene:7157`), and the paper repo's ID resolver ends in a branch that treats a bare
integer as a 1-based candidate rank. A hallucinated ID not in the candidate set falls
through that branch and silently becomes a different, real mapping.

In xwalk the model answers with `C01`, never with an identifier, and any answer that is
not a key issued for that attempt resolves to `UNRESOLVED_OUTPUT` and routes to review.
A malformed answer cannot become a real target record.

Also exercises: TSV loading with multivalue columns, multi-field disambiguation
(organism plus symbol), and a `context` template that matters — a bare symbol is
ambiguous across species, so scoring against the retrieval query alone would evaluate
the wrong thing.

## Files

| File | What it is |
|---|---|
| `job.yaml` | the whole run, as serialized constructor arguments |
| `slots.yaml` | the domain vocabulary, including the three-step organism-first procedure |
| `sample/targets.tsv` | 200 NCBI Gene records with `organism` and `tax_id` |
| `sample/mentions.jsonl` | 50 mentions with sentence context |
| `sample/gold.csv` | `source_id,gold_ids` |
| `extract.py` | how the slice was taken from the paper repo; run once, by hand |

Three of the 26 distinct gold IDs are absent from the 200-record target slice — they
are retired identifiers with no record in the dump. That is deliberate: it gives the
ceiling decomposition real `never_retrieved` cases to find.

## Run

```bash
export OPENAI_API_KEY=...
xwalk index --job examples/nlm_gene/job.yaml --out runs/nlm_gene/index
xwalk match --job examples/nlm_gene/job.yaml --out runs/nlm_gene
xwalk eval  --run runs/nlm_gene --gold examples/nlm_gene/sample/gold.csv
```

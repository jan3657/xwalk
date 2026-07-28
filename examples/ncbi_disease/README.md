# NCBI Disease example

Matching disease mentions in PubMed abstracts to CTD disease records (MeSH and OMIM).

What this one exercises is **user-supplied gold alias expansion**. A CTD concept can be
referenced by more than one accession, so a prediction that is right can look wrong to a
naive string comparison. The paper repo owned that knowledge as named library hooks —
`normalize_ctd_prediction` and `expand_ctd_eval_ids` — which meant the library knew
about MeSH. In xwalk that knowledge lives in `gold_normalize.py`, **here**, and reaches
the evaluator as two ordinary callables:

```python
from examples.ncbi_disease.gold_normalize import make_expander, normalize
from xwalk.evaluate import evaluate, load_gold_csv, render_report

gold = load_gold_csv(
    "examples/ncbi_disease/sample/gold.csv",
    normalize=normalize,
    expand=make_expander("examples/ncbi_disease/sample/targets.tsv"),
)
```

The library never learns your identifier scheme; `normalize` and `expand` are the whole
extension point.

## Files

| File | What it is |
|---|---|
| `job.yaml` | the whole run, as serialized constructor arguments |
| `slots.yaml` | the domain vocabulary — eponyms, acronyms, subtype vs parent |
| `gold_normalize.py` | **the point of this example**: your ID knowledge, in your code |
| `sample/targets.tsv` | 200 CTD disease records |
| `sample/mentions.jsonl` | 50 mentions with sentence context |
| `sample/gold.csv` | `source_id,gold_ids` |
| `extract.py` | how the slice was taken from the paper repo; run once, by hand |

## Run

```bash
export OPENAI_API_KEY=...
xwalk index --job examples/ncbi_disease/job.yaml --out runs/ncbi_disease/index
xwalk match --job examples/ncbi_disease/job.yaml --out runs/ncbi_disease
```

Scoring needs the expander, so use the SDK rather than `xwalk eval` for this one — the
CLI deliberately has no flag for "import this Python callable", because a config file
that can name arbitrary importable code is a config file that can execute anything.

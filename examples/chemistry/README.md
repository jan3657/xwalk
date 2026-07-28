# Chemistry example

Matching chemical entity mentions to ChEBI-style ontology terms. The simplest useful
case: a one-field source with sentence context, an ontology-shaped target, and every
piece of domain knowledge living in two YAML files.

The target set is deliberately full of near-misses — anomers (`alpha-D-glucose` vs
`glucose`), conjugate acid/base pairs (`lactate` vs `lactic acid`), the methylxanthine
family, and four sugar alcohols. A matcher that only does string similarity scores badly
here, which is the point: it makes the failure decomposition mean something.

## Files

| File | What it is |
|---|---|
| `slots.yaml` | the domain vocabulary the model sees — nouns, confidence rubric, hard rules |
| `templates.yaml` | the four templates (query, context, doc, candidate) |
| `targets.csv` | 52 ChEBI-style terms: `id,label,synonyms,definition` |
| `sources.csv` | 20 mentions with left/right sentence context |
| `gold.csv` | `source_id,gold_ids`; **an empty cell means the answer is no match** |
| `run.py` | builds the matcher and runs the batch |
| `evaluate_run.py` | scores a completed run against `gold.csv` |

Two of the twenty mentions (`unobtainium`, `phlogiston`) have no correct target. They
are labelled with an empty cell, so abstaining on them counts as *right* rather than as
a missing label.

## Run it

`run.py` calls a real provider. Any OpenAI-compatible endpoint works:

```bash
export XWALK_TEST_API_KEY=sk-...
export XWALK_TEST_BASE_URL=https://openrouter.ai/api/v1
export XWALK_TEST_MODEL=openai/gpt-4o-mini

python -m examples.chemistry.run --out run/
```

That writes `run/mapping.csv` (the deliverable) and `run/ledger.sqlite` (which makes the
run resumable — `--resume` skips completed records and re-runs anything whose source
record or configuration changed). It prints the run fingerprint you need next.

## Score it

```bash
python -m examples.chemistry.evaluate_run --run run/ --fingerprint <fingerprint>
```

This calls nothing — it reads the ledger, so it is free, repeatable, and works on a
machine with no credentials. It prints a report that leads with where to spend effort:

```
## Where to spend effort
  62% of failures presented the gold record and the model chose otherwise --
  spend effort on prompts, not retrieval
```

The three failure buckets are distinct on purpose:

- **never retrieved** — fix the `doc` template, the retriever set, or retrieval depth
- **truncated** — retrieved, but the selector budget cut it before the model saw it;
  raise `SelectorPolicy.max_candidates`
- **misjudged** — presented, and the model chose otherwise; fix prompts

Collapsing truncation into "retrieval failure" would send you to fix the wrong thing.

## Change domain

Edit `slots.yaml` and `templates.yaml`, swap `targets.csv` and `sources.csv`. No library
code changes.

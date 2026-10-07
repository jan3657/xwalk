# xwalk

[![CI](https://github.com/jan3657/xwalk/actions/workflows/ci.yml/badge.svg)](https://github.com/jan3657/xwalk/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

**Map messy records onto a reference collection, with an LLM making each decision and a
ledger recording why.**

You have records in one collection — free-text mentions, supplier product names, legacy
codes — and need, for each one, the matching record in another: an ontology, a
vocabulary, your own catalogue. Retrieval alone gives a shortlist and no decision. An LLM
alone cannot see a collection that does not fit in its context window. xwalk joins
them: retrieval proposes candidates, the model chooses among them, a policy turns its
confidence into a status, and every decision is stored so the run can be resumed,
inspected, reviewed and scored.

| Source (`sources.csv`) | | Target (`targets.csv`) | Status |
|---|---|---|---|
| `glucose` — "blood glucose levels were elevated" | → | `CHEBI:17234` glucose | `matched` |
| `dextrose` — "intravenous dextrose was administered" | → | `CHEBI:17234` glucose (synonym) | `matched` |
| `table sugar` — "a spoonful of table sugar in the coffee" | → | `CHEBI:17992` sucrose | `matched` |
| `unobtainium` — "a sample of unobtainium was requested" | → | — | `unmatched` |

That table is the bundled quickstart: five target terms, four mentions, one with no
correct answer. The result is a `mapping.csv` with one row per source record.

## Quickstart (offline, no credentials, no inference spending)

Requires Python 3.10-3.12. xwalk 0.2 is a release candidate and is not on PyPI yet
(see [releasing](docs/releasing.md)); install it from the repository or a built wheel.
Once published, `pip install --pre xwalk` installs the candidate.

```bash
pip install "xwalk @ git+https://github.com/jan3657/xwalk"
```

Then copy the bundled example and check it:

<!-- quickstart:begin -->
```bash
xwalk init demo
xwalk validate --job demo/job.yaml --no-credentials
```

Save this as `quickstart.py`. It runs the job with `FakeLLM`, a scripted stand-in that
always picks the first candidate with confidence 0.9. It exercises the whole pipeline
(retrieval, the four stages, the ledger, the exports) but makes no judgement: use it to
learn the mechanics, not to assess quality.

```python
import json

from xwalk import ops
from xwalk.llm.fake import FakeLLM


def scripted_model(request):  # one answer every stage understands
    return json.dumps({"chosen_key": "C01", "confidence_score": 0.9, "decision": "support"})


result = ops.run("demo/job.yaml", "demo/run", llm=FakeLLM(handler=scripted_model), max_calls=50)
print(result.run["run_state"], result.counts, "model calls:", result.usage["calls"])
```

Run it, then look at what it produced:

```bash
python quickstart.py
xwalk inspect --run demo/run
xwalk explain --run demo/run s4
xwalk export --run demo/run --view reviewed --out demo/reviewed.csv
```
<!-- quickstart:end -->

`quickstart.py` prints `complete {'matched': 3, 'unmatched': 1, 'total': 4} model calls:
7`. `demo/run/mapping.csv` holds the four rows; `inspect` warns that one target was
chosen by two sources (glucose and dextrose, which is correct here — see cardinality
below); `explain` shows that `s4` had zero retrieval candidates. These commands are
checked against a clean install of the built wheel by `scripts/check_readme_quickstart.py`.

## With a real model

This route needs an endpoint and a credential, and every call is billed by your
provider. The bundled job names `gpt-4o-mini` on the OpenAI API with the key read from
`OPENAI_API_KEY`; edit the `llm` block of `demo/job.yaml` for any OpenAI-compatible
endpoint (vLLM, Ollama, a gateway). Credentials are never written in a job file:
`api_key_env` names an environment variable and an inline key is rejected.

```bash
export OPENAI_API_KEY=...
xwalk validate --job demo/job.yaml                          # offline; checks the key is set
xwalk match --job demo/job.yaml --out demo/real --max-calls 50
```

A confident record costs two calls (select, then score); uncertain ones add a verifier
call or a rewritten query, up to `policy.max_attempts`. `--max-calls` is a hard limit on
upstream requests for that invocation, retries included; reaching it stops the run with
exit 3, and repeating the command resumes. xwalk reports calls and provider-reported
tokens (calls without reported usage are counted as unknown, never as zero); it does not
convert them to money. Use a new `--out` directory: a run directory belongs to one
configuration and a different one is refused, never mixed in.

From Python the short route is the same `ops.run` without `llm=` — see the
[getting started guide](docs/guide/getting-started.md). Building every component by
hand (your own retriever, store or client) is in [the Python SDK
guide](docs/guide/python-sdk.md).

## What a run produces

A run directory holds `mapping.csv` (the deliverable), `results.jsonl` (every attempt,
with candidates and model output), `manifest.json` (configuration fingerprint, counts,
usage, run state) and `ledger.sqlite`, the source of truth that every export is
regenerated from.

Every source record ends in exactly one status:

| Status | Meaning | What to do |
|---|---|---|
| `matched` | Accepted automatically: score at or above `accept_at`, resolved exactly, verifier did not object. | Use it; spot-check with `explain`. |
| `needs_review` | Plausible but uncertain, the verifier disagreed, or the model's answer did not resolve. | Review it (below). |
| `unmatched` | No candidate was retrieved, the model chose none, or the best score was below `review_floor`. | Treat as no match (or check retrieval if you expected one). |
| `failed` | Infrastructure, not data: the provider or retriever failed. | Fix it and re-run; failed records are retried on resume. |
| `pending` | Export only: in the source collection but not processed yet (`--limit N` processes N records per invocation). | Run the command again; it continues. |

A malformed model answer can never become a real target: candidates are shown under
opaque keys (`C01`, `C02`, ...) and only an exact key resolves, so anything else routes
to review. The `reason` column says why a record got its status. The run as a whole is
`complete`, `partial`, `failed`, `aborted` or `interrupted`, and the exit code follows
it: `0` complete with nothing to review, `1` needs attention (review rows, a partial
run), `2` usage or configuration error, `3` runtime failure (aborted run, failed rows,
incompatible run directory), `130` interrupted.

**Raw and reviewed.** `xwalk review export` writes the records that need a decision;
a reviewer fills in `accept`, `reject`, `replace`, `no_match` or `defer`; `xwalk review
apply` validates the whole file and records the decisions as an overlay. The model's
output is never edited. `xwalk export --view raw` is the model's answer;
`--view reviewed` is the final answer per row with the model's original decision beside
it. See [human review](docs/guide/review.md).

**Current and history.** Re-running after editing a source row re-matches only that row.
Exports always show the *current* view: one row per source record in the latest source
collection. Superseded results, removed sources and retried failures stay in the ledger;
`xwalk export --view history` writes all of them.

**Cardinality.** Each source record gets at most one target. Many sources may map to
the same target (many-to-one): xwalk reports these as duplicate targets and leaves both
matched, because "glucose" and "dextrose" sharing one term is correct and two suppliers
sharing one SKU may not be. There is no one-to-one assignment mode and no
one-to-many output.

**Uncertainty.** Confidence is the scorer model's own number, not a calibrated
probability. Before trusting `accept_at`, label a sample and run `xwalk eval`: it
reports accepted precision, review rate, a threshold curve and a warning when the
confidences do not separate right from wrong. Evaluation reads the ledger and makes no
model calls. See [measuring a run](docs/guide/evaluation.md).

## When to use something simpler

- Identifiers or normalised strings already line up: use a join or exact lookup.
- Spelling variants of the same strings: fuzzy string matching (for example RapidFuzz)
  is faster, free and deterministic.
- Deduplicating structured person or company records at scale with field-level
  evidence: a probabilistic record-linkage tool such as Splink or dedupe.
- You want a ranked shortlist, not a decision: run retrieval alone (`xwalk search`
  needs no model).

xwalk earns its cost when the match needs reading — synonyms, context, domain rules —
and when you need a per-record audit trail and a review queue rather than a score.
Practical limits: every record needs one to a few model calls, so throughput and cost
follow your provider; targets are loaded in memory and indexed with BM25 (or dense
retrieval with the `dense` extra); the run ledger is a local SQLite file.

## Install options

```bash
pip install xwalk                 # BM25 + any OpenAI-compatible endpoint. No torch.
pip install 'xwalk[dense]'        # + sentence-transformers, torch, faiss-cpu
pip install 'xwalk[ontology]'     # + rdflib, for OWL sources
pip install 'xwalk[sql]'          # + SQLAlchemy, for database sources
pip install 'xwalk[litellm]'      # + the LiteLLM adapter
pip install 'xwalk[mcp]'          # + the MCP SDK, for `xwalk mcp`
pip install 'xwalk[all]'
```

Extras stay optional: a missing one raises an error naming the extra and the install
command. BM25 runs on Tantivy, a base dependency with no silent fallback; see
[platforms](docs/platforms.md).

## Also included

- **Evaluation and prompt tools**: `xwalk eval`, `compare`, `ablate`, `prompts draft` and
  `prompts optimize` against your gold labels ([evaluation](docs/guide/evaluation.md),
  [prompt optimisation](docs/guide/prompt-optimisation.md)).
- **Decision-model path**: a job with a `decider:` block instead of `llm:` matches with
  TypeSafe's Jev, which answers typed questions with probabilities (screen every
  candidate, choose, gate on declared properties); `xwalk fit` fits its thresholds on
  gold. Same run directories, exit codes and `--max-calls` as the LLM path
  ([decision models](docs/reference/decide.md)).
- **A browser UI**: `xwalk ui` serves a local web app over the same operations. Upload a
  file and map it onto ontologies or onto another file, find duplicates, quick-map
  pasted terms, keep a library of parsed ontologies (built-in samples, imports, OBO
  Foundry downloads), then click any row to see why, review, evaluate and export
  ([UI guide](docs/guide/ui.md)). Standard library only.
- **Agent access (optional extra)**: `xwalk mcp` serves bounded validate, search, match
  and read operations over the Model Context Protocol ([MCP guide](docs/guide/mcp.md)).
- **Clustering (experimental)**: `xwalk cluster` groups one collection into clusters of
  equivalent records. It is tested with a scripted model only and has not been
  evaluated on a real model; treat its output as a draft
  ([clustering guide](docs/guide/clustering.md)).
- **Benchmarks**: a pilot harness in `benchmarks/` (repository only). Only offline
  smoke checks and non-LLM baselines have been run; real-model results are pending
  ([benchmarks](docs/benchmarks.md)).

## Documentation

**[Full documentation →](docs/README.md)** ·
[Getting started](docs/guide/getting-started.md) · [Concepts](docs/concepts.md) ·
[Architecture](docs/architecture.md) · [CLI](docs/reference/cli.md) ·
[Job file](docs/reference/job-file.md) · [Troubleshooting](docs/guide/troubleshooting.md) ·
[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

Worked examples with real data slices live in [`examples/`](examples) (chemistry, ChEBI,
NCBI disease, NLM gene, food); they need a real endpoint.

## Licence

MIT. See [LICENSE](LICENSE).

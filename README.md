# xwalk

[![CI](https://github.com/jan3657/xwalk/actions/workflows/ci.yml/badge.svg)](https://github.com/jan3657/xwalk/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jan3657/xwalk/blob/main/LICENSE)

**Match records to a reference collection, with retrieval, model decisions, and an audit trail.**

xwalk turns free-text mentions, product descriptions, or legacy labels into identifiers
from your ontology, vocabulary, or catalog. Retrieval finds candidates; an LLM or typed
decision model chooses among them; a policy accepts, rejects, or routes the result for
human review. A SQLite ledger records each attempt so you can inspect and resume runs.

| Source | Reference match |
|---|---|
| glucose | `CHEBI:17234` — glucose |
| dextrose | `CHEBI:17234` — glucose |
| table sugar | `CHEBI:17992` — sucrose |
| unobtainium | No match |

These are illustrative rows from the bundled five-term chemistry example, not an
accuracy benchmark.

This repository contains the **Python SDK and CLI**. Browser applications live in
[xwalk-ui](https://github.com/jan3657/xwalk-ui); the hosted NER demo is at
[www.xwalk.si](https://www.xwalk.si) and requires a passcode.

## Install

Tested on Python 3.10–3.12. The current development version is `0.2.0rc1`.
**Do not install this project with `pip install xwalk`: that PyPI name belongs to an
unrelated project.** The distribution is named **`xwalk-match`**; imports and the command remain `xwalk`.
Until the first `xwalk-match` release is published, install this repository
in a fresh virtual environment (Git is required):

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "xwalk-match @ git+https://github.com/jan3657/xwalk.git"
```

The default install includes BM25 retrieval and an OpenAI-compatible HTTP client.
Torch and local embedding models are optional. For reproducible work, append a full
commit SHA to the Git URL with `@<commit>`.

## Try it offline

No credentials or inference spending. Copy the bundled example, then run a scripted
model to learn the pipeline. The script always picks the first candidate: it verifies
the mechanics, not model quality.

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

Expected output:

```text
complete {'matched': 3, 'unmatched': 1, 'total': 4} model calls: 7
```

`demo/run/mapping.csv` contains the mapping. `results.jsonl` records attempts;
`manifest.json` records configuration, usage, and run state; `ledger.sqlite` is the
source of truth. CI executes this quickstart against the built wheel in a clean environment.

## Use your data and a real model

Edit the source, target, templates, and provider settings in `demo/job.yaml`.
The bundled job uses the OpenAI API; other OpenAI-compatible endpoints are supported.
Credentials belong in environment variables, not YAML. Provider calls may incur charges.

```bash
export OPENAI_API_KEY=...
xwalk validate --job demo/job.yaml
xwalk match --job demo/job.yaml --out demo/real --max-calls 50
```

`--max-calls` bounds upstream requests, including retries, for that invocation.
Repeat the command to resume; use a new output directory for a different job configuration.
xwalk reports calls and provider-reported usage; missing usage remains unknown.
For an end-to-end walkthrough, see [getting started](https://github.com/jan3657/xwalk/blob/main/docs/guide/getting-started.md).

## What you can do

- **Match and inspect:** BM25 or optional dense retrieval, exact candidate-key resolution,
  per-record evidence, resumable runs, and CSV/JSONL exports.
- **Review:** export uncertain rows, apply human decisions as an overlay, and retain the
  original model output. See [human review](https://github.com/jan3657/xwalk/blob/main/docs/guide/review.md).
- **Evaluate:** measure precision, review rate, candidate recall, and threshold behavior
  against gold labels without making model calls. See [evaluation](https://github.com/jan3657/xwalk/blob/main/docs/guide/evaluation.md).
- **Choose a model path:** conventional LLM stages or a typed decision model through
  [Jev](https://github.com/jan3657/xwalk/blob/main/docs/reference/decide.md).
- **Integrate:** use [Python](https://github.com/jan3657/xwalk/blob/main/docs/guide/python-sdk.md), the [CLI](https://github.com/jan3657/xwalk/blob/main/docs/reference/cli.md),
  or the optional [MCP server](https://github.com/jan3657/xwalk/blob/main/docs/guide/mcp.md).

Results distinguish `matched`, `needs_review`, `unmatched`, and infrastructure `failed`
rows. Exports also show `pending` records in incomplete runs. See
[concepts](https://github.com/jan3657/xwalk/blob/main/docs/concepts.md) for status, resume, and cardinality semantics.

## Limits and evidence

Each source maps to at most one target; multiple sources may share a target. There is
no one-to-one assignment mode. Model confidence is not a calibrated probability:
evaluate on your own labeled sample before choosing an automatic acceptance threshold.
Targets are loaded into memory, the ledger is local SQLite, and model latency and cost
limit throughput. Exact joins or fuzzy matching are often sufficient when identifiers
or spellings already line up.

[Recorded pilot benchmarks](https://github.com/jan3657/xwalk/blob/main/docs/benchmarks.md) include real-model runs on small
50-record datasets and separate synthetic smoke checks. They do not establish accuracy
on your data or on the current revision. [Clustering](https://github.com/jan3657/xwalk/blob/main/docs/guide/clustering.md) remains
experimental; review its evidence-bound and historical-export limitations before use.

## Optional dependencies

Install extras from this repository before the first PyPI release:

```bash
python -m pip install "xwalk-match[dense] @ git+https://github.com/jan3657/xwalk.git"
```

| Extra | Adds |
|---|---|
| `dense` | Local sentence-transformers, Torch, and FAISS |
| `ontology` | OWL loading through rdflib |
| `sql` | SQLAlchemy sources |
| `litellm` | LiteLLM provider adapter |
| `mcp` | MCP server SDK |
| `chatgpt` | Subscription OAuth support used by the separate UI |
| `all` | All runtime extras |

See [platform support](https://github.com/jan3657/xwalk/blob/main/docs/platforms.md) before installing optional native dependencies.

## Documentation and contributing

Start with the [documentation index](https://github.com/jan3657/xwalk/blob/main/docs/README.md), [job-file reference](https://github.com/jan3657/xwalk/blob/main/docs/reference/job-file.md),
and [worked examples](https://github.com/jan3657/xwalk/tree/main/examples). For development, read [CONTRIBUTING](https://github.com/jan3657/xwalk/blob/main/CONTRIBUTING.md)
and the [architecture](https://github.com/jan3657/xwalk/blob/main/docs/architecture.md). Bugs and reproducible examples are welcome
in [GitHub issues](https://github.com/jan3657/xwalk/issues).

[Release checklist](https://github.com/jan3657/xwalk/blob/main/docs/releasing.md) · [Changelog](https://github.com/jan3657/xwalk/blob/main/CHANGELOG.md) ·
[Publication readiness](https://github.com/jan3657/xwalk/blob/main/docs/publication-readiness.md) · [MIT license](https://github.com/jan3657/xwalk/blob/main/LICENSE)

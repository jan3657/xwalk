# Exploring xwalk in the browser (`xwalk ui`)

`xwalk ui` serves a local web app over the same operations the CLI uses (`xwalk.ops`). It
is for learning what xwalk does, checking a job before spending on it, and reading a
finished run without writing commands: validate a job, look at what its templates render,
search its index, run it, open each decision, review uncertain rows, score the run against
gold labels and download the exports.

It needs nothing beyond the base install. The server is the Python standard library's
`http.server`; the page is plain HTML, CSS and JavaScript shipped in the wheel, with no
build step and nothing loaded from the internet.

```bash
xwalk ui                                  # serve the current directory, open a browser tab
xwalk ui --root ~/mappings --port 9000    # another workspace and port
xwalk ui --offline-only --no-browser      # never call a job's endpoint; print the URL only
```

| Flag | Default | Meaning |
|---|---|---|
| `--root` | `.` | the workspace: the only directory the app reads and writes |
| `--host` | `127.0.0.1` | address to listen on (see [Security](#security)) |
| `--port` | `8765` | port; `0` picks a free one |
| `--no-browser` | — | do not open a browser tab |
| `--max-calls-cap` | `500` | the largest call limit a run against the job's endpoint may request |
| `--offline-only` | — | refuse runs against the job's endpoint; only the offline stand-in |
| `--max-upload-mb` | `512` | the largest file the page may upload |
| `--verbose` | — | log every request to stderr |

Ctrl-C stops the server and exits `0`. A root that is not a directory, or a cap below 1,
exits `2`.

## Bring your own data

The start page offers what most people come for. Each one writes an ordinary job
directory under `<root>/projects/<name>/` (a `job.yaml` you can also run with `xwalk
match`, its `slots.yaml`, and the records it read), then runs it like any other job.

| Task | Where | What happens |
|---|---|---|
| Map a file to an ontology | **Map a file** | upload a CSV, TSV, JSONL or text file; choose the column with the text to match, an optional id column and context columns; pick one or more ontologies from the library |
| Map one file onto another | **Map a file → My own file** | the target is a second upload: choose its label, id, synonyms (and their separator) and definition columns |
| Find duplicates in one file | **Find duplicates** | the same upload step, written as a `kind: cluster` job (experimental) |
| Map a few terms | **Quick map** | paste terms (one per line; `term<TAB>context` adds context), choose ontologies or a file, then either *candidates only* or a run |
| Browse and add ontologies | **Ontologies** | search and filter each entry; import a file; download from a URL |
| Review in a spreadsheet | **Run → Review** | download the worksheet, fill it in, upload it back |
| Use a real model | **Settings** | paste an API key for this server process only |

**Files.** Uploads go to `<root>/uploads/` (at most `--max-upload-mb`, default 512 MB).
Each file is read once into normalized records, so a column name with spaces or
punctuation never reaches a template: a source record has `text`, `context` and its other
columns; a target record has `label`, `synonyms` (a list), `definition` and its other
columns. The page suggests which column is which from the names and the first rows. A
repeated id or a missing column is reported with the row or column to fix. Spreadsheets
(`.xlsx`) are not read directly: save them as CSV (UTF-8).

**Candidates only** is retrieval with no model: for each term, the top candidates from
the job's own BM25 index and fusion, with a flag where the term equals a label or synonym.
It is free and instant, and a good first look at whether the right answers are
retrievable at all. It is not a decision; the first candidate is the best text match.

**The ontology library** holds parsed term collections:

- *Built in*: 200-term samples of ChEBI, FoodOn, the CTD disease vocabulary (MeSH/OMIM)
  and NCBI Gene, the slices the examples use (`scripts/build_ui_ontologies.py` makes
  them). They are marked *sample*: they are not the full ontologies.
- *Imported*: an OBO, OWL (needs `xwalk[ontology]` at import time only), CSV, TSV or
  JSONL file, parsed once into `<root>/.xwalk-ui/library/<name>/terms.jsonl`.
- *Downloaded*: any OBO Foundry ontology from the catalog (HPO, Mondo, DOID, GO, ChEBI,
  Uberon, CL, PATO, UO, ENVO by their permanent `purl.obolibrary.org` URLs), or any
  `http(s)` URL ending in `.obo`, `.owl`, `.csv`, `.tsv` or `.jsonl` (gzip is
  unpacked). The download runs as a background task with progress.

Mapping onto several entries at once combines their terms, each tagged with its
`ontology`, which the model sees in the candidate text. An id two entries share keeps
the first entry's term, and the overlap is reported.

**Session API keys.** *Settings* sets a key such as `OPENAI_API_KEY` in the memory of the
`xwalk ui` process, where a job's `api_key_env` reads it. It is never written to disk or
sent back to the page, and it is gone when the server stops. Only names ending in
`_API_KEY`, `_KEY` or `_TOKEN` are accepted, and only keys set from the page can be
cleared from it. Keys already in the environment work as before.

**Which model answered.** A run's manifest names the job's model even when the offline
stand-in answered, so the page records which one a run started here used
(`.xwalk-ui/run-models.json`) and labels offline runs as such.

## A first session

1. **Workspace.** The home page lists every job file (a YAML file with `templates`,
   `target` and `source`, or `kind: cluster`), every run directory (a directory with a
   `manifest.json`) and every CSV with `gold` in its name under the root. With an empty
   workspace, **Create demo job** copies the bundled quickstart, exactly as `xwalk init`.
2. **Job → Overview** runs `validate` (offline, zero model calls) and shows the record
   counts, retrievers and model. Tick *check credentials* to also check that the
   credential variable is set; its value is never read into the page.
3. **Job → Configuration** shows the job file. Edit it in your editor and reload.
4. **Job → Data** shows the first source and target records as the job's templates
   render them: the query retrieval searches for, the context the model sees, the
   document the index stores, and each target as a candidate. Click a source to search
   for its query.
5. **Job → Search** returns the fused candidates for any text with no model call: the
   shortlist a record with that query would be shown. The index is built on first use
   under `<root>/.xwalk-ui/index/`; an incompatible one is refused (tick *rebuild index*).
6. **Job → Run** starts a match (or clustering) in the background and streams each
   record's status as it settles. **Open run** when it is done.
7. **Run → Summary / Results.** Counts by status, usage, and the result table. Click a
   row to see *why*: every attempt's query, the candidates under their opaque keys
   (`C01`, …), which one the model chose, its score, the verifier's verdict, the raw model
   output and the record's history.
8. **Run → Review** lists the `needs_review` rows with a decision for each (`accept`,
   `reject`, `replace` with a corrected target id, `no_match`, `defer`). **Apply** records
   them through `ops.review_apply`: all-or-nothing, as an overlay that keeps the model's
   answer ([human review](review.md)).
9. **Run → Evaluate** scores the run against a gold CSV (`source_id,gold_ids`), as
   `xwalk eval` does: accuracy, precision and coverage at each confidence threshold, and
   the retrieval ceiling ([measuring a run](evaluation.md)).
10. **Run → Export** downloads the raw, reviewed and history views.

A run started from the page is recorded in `<root>/.xwalk-ui/runs.json` (run directory →
job file), so the run page can offer *Continue / re-run* and pre-fill the job for review.
Running the same job into the same directory resumes it, as on the command line; a
directory that belongs to another configuration is refused and nothing is overwritten.

Clustering jobs (experimental) get a cluster view instead: the clusters with their
members, the unresolved sources, and every decision recorded about one record, with its
prompt and raw model output.

## Which model answers

The **Run** tab offers two choices.

**Offline stand-in** (the default) needs no endpoint and costs nothing. It exercises the
whole pipeline (retrieval, keying, every stage, the policy, the ledger, the exports) but
makes **no judgement about your data**, and every run made with it carries an
`offline_model` warning:

| Job | Stand-in |
|---|---|
| matching, LLM path | the scripted reply of the quickstart and `xwalk mcp --offline-model`: the first candidate, confidence 0.9 |
| matching, decider path | `FakeDecider`: word overlap between the source and each candidate |
| clustering | word overlap between labels: two labels are equivalent when their word sets overlap by at least 0.6 (Jaccard); near-threshold calls get low confidence and go to review |

**Job endpoint** uses the model the job names, with the credential from the environment
of the process that started `xwalk ui`. A call limit is then required (at most
`--max-calls-cap`), enforced by the same budget as `xwalk match --max-calls`: reaching it
stops the run, and starting it again resumes. Every call is billed by your provider.
`--offline-only` removes this choice.

## Security

`xwalk ui` is a local, single-user tool. Anyone who can open the page can read the
workspace, write run directories in it and, unless `--offline-only` is set, spend on the
jobs' endpoints. The defences are aimed at other web pages open in the same browser:

- It listens on `127.0.0.1` by default. `--host 0.0.0.0` exposes it to the network,
  without authentication beyond the page token; it prints a warning when you do.
- Every API request must carry a random per-process token that is embedded in the page
  (`X-Xwalk-Token`; downloads carry it in the link). A page from another origin cannot
  read it, and the custom header forces a CORS preflight the server never grants.
- On a loopback address the `Host` header must name it (`localhost`, `127.0.0.1`, `::1`),
  which defeats DNS rebinding.
- Every path a request names is resolved, symlinks included, and refused (`403`) when it
  lies outside the root.
- The page sets a Content Security Policy that allows only its own scripts and styles,
  and renders record text and model output as text, never as HTML.

## The API

The page talks to a small JSON API in `xwalk.ui.api`, which you can also drive from
Python without HTTP:

```python
from xwalk.ui.api import Workspace

ws = Workspace("workspace")
ws.init({"dest": "demo"})
task = ws.start_task({"job": "demo/job.yaml", "out": "demo/run"})   # offline stand-in
print(ws.tasks.wait(task["id"]).result["counts"])
print(ws.explain({"dir": "demo/run", "source_id": "s1"})["data"]["decision"])
```

Handlers that perform an operation return its `--json` envelope (schema 1), also when it
failed; misuse of the API raises `ApiError` with an HTTP status. Routes:

| Method | Path | Does |
|---|---|---|
| GET | `/api/info`, `/api/workspace` | server settings; jobs, runs and gold files under the root |
| POST | `/api/init` | `ops.init` |
| GET | `/api/job`, `/api/preview` | a job file; its first records as the templates render them |
| POST | `/api/validate`, `/api/search` | `ops.validate`; `ops.search` (and `ops.index` to rebuild) |
| GET, POST | `/api/tasks` | list tasks; start a match or cluster task |
| GET, POST | `/api/tasks/<id>`, `/api/tasks/<id>/cancel` | poll a task (`since` = events already seen); cancel it |
| GET | `/api/run`, `/api/results`, `/api/explain` | `ops.inspect`; `ops.list_results`; `ops.explain(full=True)` |
| GET, POST | `/api/review` | the review worksheet; apply decisions via `ops.review_apply` |
| POST | `/api/eval` | evaluate against a gold CSV |
| GET | `/api/export` | download a view (`raw`, `reviewed`, `history`; cluster runs: `members`, `clusters`, `unresolved`, `decisions`) |
| GET | `/api/clusters`, `/api/cluster-member` | a cluster run's clusters; one record's membership and decisions |
| POST | `/api/upload?name=` | store the request body as `uploads/<name>` (streamed; `--max-upload-mb`) |
| GET, POST | `/api/files`, `/api/files/delete`, `/api/inspect` | list uploads; delete one; columns, sample rows and suggested roles of a file |
| GET | `/api/library`, `/api/library/catalog`, `/api/library/terms` | library entries; the download catalog; a page of an entry's terms (`filter`) |
| POST | `/api/library/import`, `/api/library/download`, `/api/library/delete`, `/api/library/search` | parse a file into the library; download a URL as a task; delete an import; ranked search |
| POST | `/api/projects` | write a matching or clustering project from a file or pasted text |
| POST | `/api/quickmap` | pasted terms: `candidates` (retrieval only) or a run (`offline`, `endpoint`) |
| GET | `/api/mapping`, `/api/mapping.csv` | a run's results with source texts and target labels; the same as CSV |
| GET | `/api/job-file`, `/api/review-sheet` | download a job file; download the review worksheet |
| POST | `/api/review-upload` | apply a filled-in worksheet from the workspace |
| GET, POST | `/api/presets`, `/api/credentials` | model presets; session API keys (state only, never values) |

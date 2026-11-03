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
- `api/ner.py`, `api/health.py`: the two Vercel Python functions, thin shims over
  `ner_site.http.handle`.
- `public/`: the page (`index.html`, `app.js`, `app.css`; no build step, no
  dependencies). It builds every element with `textContent`, so model output is only
  ever rendered as text.
- `vercel.json`: function duration and bundled files, and the security headers.
- `tools/serve_local.py`: a standard-library server for `public/` and `/api/*`.
- `tools/screenshot.mjs`: Playwright smoke check that loads the page, runs the example
  and writes a PNG.
- `tools/fixtures/`: canned `/api` responses (`health*.json`, `ner*.json`,
  `error.json`) for working on the page without the service.

## Deploy

### Run it locally first

From the repository root, with the venv active (`. .venv/bin/activate`):

```bash
# Page only, canned API answers (no ner_site import, no key, no network)
python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures
# ... the same with dense retrieval unavailable and offline mode, to see the banner
python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures --scenario degraded

# The real service with the stand-in extractor (NER_SITE_OFFLINE=1; no key needed)
python ner-site/tools/serve_local.py --offline

# The real service against OpenRouter (spends money on your key)
OPENROUTER_API_KEY=... python ner-site/tools/serve_local.py
```

The server binds `127.0.0.1`, port 8765 by default (`--port`), and adds the headers
from `vercel.json` to every response, so the page runs under the production
Content-Security-Policy. With `--canned`, posting text that contains `ERROR` returns
`fixtures/error.json` with HTTP 400 to show the error box; `--delay SECONDS` slows the
POST to show the busy state.

Smoke check (Playwright is resolved from `NODE_PATH` or `npm root -g`):

```bash
node ner-site/tools/screenshot.mjs http://127.0.0.1:8765 /tmp/ner.png --width 1280
node ner-site/tools/screenshot.mjs http://127.0.0.1:8765 /tmp/ner-360.png --width 360
```

It clicks **Load example** and **Run**, waits for the result cards and exits non-zero
on a page error, a console error, a failed run, or horizontal scrolling. `--expect
error --text "...ERROR..."` checks the error state instead; `--dark` uses the dark
colour scheme.

### Deploy on Vercel (operator)

1. Import the repository into Vercel. Set **Root Directory** to `ner-site` and
   **Framework Preset** to "Other" (no build command). Vercel serves `public/` as the
   static site and `api/*.py` as Python functions (`.python-version` selects 3.12).
2. Under Settings, Environment Variables, add `OPENROUTER_API_KEY` (mark it
   sensitive). Optionally add `NER_SITE_PASSCODE` and `NER_SITE_MODELS`; see the table
   below. Redeploy after changing variables.
3. Open `https://<deployment>/api/health` and check `"mode": "live"`,
   `"retrieval": {"dense": {"status": "ok"}}` and the model list. `dense: unavailable`
   with a reason means the committed indexes under `indexes/` are missing or stale, or
   the key is not set; mapping then runs on BM25 alone and the page shows a banner.
4. Load the page, run the example text, and confirm the entities, their highlighted
   spans and the matches. Check the function log for the cold-start time. If a
   passcode is set, confirm that a request without it gets 401
   (`curl -si -X POST -H 'Content-Type: application/json' -H 'X-Requested-With: xwalk-ner' -d '{"text":"aspirin"}' https://<deployment>/api/ner`).
5. Set a spend limit on the OpenRouter key (see the security notes), then record what
   was verified in a "Verified" section of this file.

### Environment variables (function runtime)

| Name | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | unset | The operator's key, used for the chat call and query embeddings. Without it (and not offline) `/api/ner` answers 503 `not_configured`. |
| `NER_SITE_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. |
| `NER_SITE_MODELS` | `openai/gpt-4o-mini,google/gemini-2.5-flash-lite,qwen/qwen3-next-80b-a3b-instruct` | Comma-separated model allowlist; the first is the default. |
| `NER_SITE_PASSCODE` | unset | When set, every `/api/ner` request must send it as `X-Passcode`; the page asks for it and keeps it in `sessionStorage`. |
| `NER_SITE_MAX_CHARS` | 5000 | Maximum text length (characters). |
| `NER_SITE_MAX_ENTITIES` | 25 | Maximum entities kept per request. |
| `NER_SITE_MAX_PROMPT_CHARS` | 4000 | Maximum prompt length (characters). |
| `NER_SITE_OFFLINE` | unset | `1` replaces the LLM with the stand-in extractor; `/api/health` reports `"mode": "offline"`. |
| `NER_SITE_DENSE_DIR` | `<project root>/indexes` | Where the committed dense indexes live. |
| `NER_SITE_CACHE_DIR` | `/tmp/xwalk-ner-site` | Where BM25 indexes are built at cold start. |

### Security notes for the operator

- **Set a spend limit on the OpenRouter key.** It is the only hard cap that holds
  across all function instances. Per request the service allows at most 3 LLM calls
  (retries included) and 25 entities (one embedding request each), and a function run
  is cut off at 120 s, but nothing stops many requests in a row.
- **Set `NER_SITE_PASSCODE` for a public deployment.** Without it anyone who finds the
  URL can spend your key. Share the passcode out of band; the service compares it
  in constant time. For more, Vercel's firewall rate limiting is an option.
- The key lives only in the function environment. `/api/health` never returns it, and
  error messages pass URLs through redaction. Do not put the key in `NER_SITE_BASE_URL`.
- `vercel.json` sends `Content-Security-Policy: default-src 'self'; connect-src
  'self'; img-src 'self' data:; style-src 'self'`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, and `Cache-Control: no-store` on `/api/*`. No CORS
  headers are sent, and the API requires `X-Requested-With: xwalk-ner`, so other
  sites cannot call it from a browser.
- Visitors' text is sent to the model provider (OpenRouter and the model's host). The
  function log records sizes, counts, timings and status only, never the text, the
  prompt or the key.

## Verified

### Locally, 3 November 2026 (WS-D; Python 3.13, the repository venv)

What was run, and nothing more:

- `python -m pytest -q tests/test_site_service.py tests/test_site_http.py`: 111 passed.
  Offline only (FakeLLM, the offline stand-in, `httpx.MockTransport`; BM25 built in
  `tmp_path`). They cover every error code in plan section 3.1 through `handle`, the
  passcode, `X-Requested-With`, OPTIONS 405, the health shape (validated with
  `HealthResponse`), a full offline POST, a full live POST against a mocked chat
  endpoint, a full POST with dense indexes built in `tmp_path` by the default Qwen3
  encoder against a mocked 26-letter histogram embeddings endpoint, both `api/*.py`
  shims on a real `ThreadingHTTPServer` socket (including 413 before reading, 411, HEAD,
  and a bad setting giving 503), and that a sentinel key, the passcode, the text and the
  prompt appear in no response and no log line.
- `python ner-site/tools/serve_local.py --offline`, started with `OPENROUTER_API_KEY`
  removed from its environment, then `curl /api/health` (200, `mode: offline`,
  `default_model: offline`, BM25 ok, dense unavailable "OPENROUTER_API_KEY is not set
  ... (all 4 ontologies)"), one `curl -X POST /api/ner` with the page's example text (200,
  11 entities whose offsets equal `tools/fixtures/ner.json`, `llm_calls: 0`, null
  tokens), and `curl -X OPTIONS /api/ner` (405, `Allow: POST`, no CORS headers).
- `node tools/screenshot.mjs` at 1280 px and at `--width 360` against that server: both
  exit 0 with "11 entities found, 9 with candidate terms", no horizontal scroll.
- The four gates from `CONTRIBUTING.md`, all passing.

Not run: anything against OpenRouter (no chat call, no embeddings call, no cost), any
Vercel build or deployment, Python 3.12, and the committed dense indexes opened with a
real key.

### Python 3.12 and live, 3 November 2026 (the orchestrating session)

- A fresh Python 3.12.3 venv with only `pip install -r ner-site/requirements.txt` (xwalk
  from GitHub at the pinned commit, as Vercel installs it): installs cleanly, 125 MB of
  site-packages, and the wheel ships the four ontology samples. All 201
  `tests/test_site_*.py` pass in it against that installed xwalk (checked that `xwalk`
  imported from site-packages, not the repository's `src/`).
- The committed dense indexes, built with the operator's key by
  `tools/build_indexes.py`: 3,132 vectors, 13 MB, reported cost $0.00065; all four open
  as `dense: ok`.
- `tools/serve_local.py` (live, not `--offline`) in that 3.12 venv with the key, then
  `tools/screenshot.mjs` at 1280 and 360 px with the page's example text:
  `/api/health` answered `mode: live`, `llm: ok`, BM25 ok, dense ok; the run found 10
  entities, all with candidate terms, with one `openai/gpt-4o-mini` call (641 + 186
  tokens), 10 embedding requests, 3.9 to 5.2 s in total. Sample terms present in the text
  ranked first in both retrievers (for example haloperidol `CHEBI:5613`, NAT2
  `NCBIGene:10`, Alkaptonuria `MESH:D000474`, cumin seed `FOODON:00003396`); aspirin,
  which is not in the 200-term ChEBI sample, got only loose dense neighbours. The
  request log line held no text, prompt or key.

Still not run: a Vercel build or deployment.

### For the operator after deploying

1. `GET /api/health`: `mode: live`, `llm: {"status": "ok"}`, `dense: ok`. Without
   `OPENROUTER_API_KEY` health still answers 200 with `llm` unavailable and says why;
   `POST /api/ner` then answers 503 `not_configured`.
2. That Vercel passes `self.path` as `/api/ner` and `/api/health` (with any query
   string): anything else gets a 404 `not_found` from `handle`. Also that a POST from the
   page carries `Content-Length` (fetch does; a missing one gets 411 `length_required`).
3. The example run on the deployed page: entities, highlighted spans, matches naming
   both `bm25` and `dense-qwen3`, and `usage` with real token counts.
4. The function log: one JSON line per request (`"event": "ner_site.request"`) with
   sizes, counts, timings, status and model, and no text, prompt or key. Also note the
   cold-start time (the first request builds four BM25 indexes in `/tmp`).
5. With `NER_SITE_PASSCODE` set: 401 `passcode_required` without it, 401
   `passcode_invalid` with a wrong one. Use an ASCII passcode (browsers reject other
   characters in a header) of 8 characters or more.
6. Query embeddings always go to OpenRouter's default embeddings URL with
   `OPENROUTER_API_KEY`; `NER_SITE_BASE_URL` changes only the chat endpoint.

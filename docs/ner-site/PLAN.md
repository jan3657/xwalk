# NER-to-ontology demo site on Vercel: implementation plan

Status: plan, 2026-11-03. Drafted by a planning agent, reviewed and amended by the
orchestrating session (amendments marked **[amended]**). The executing agents have no
`vercel` CLI, no Vercel token and no OpenRouter key: they produce a deploy-ready
`ner-site/` directory verified locally, and the operator deploys and verifies (section
11).

The request: a minimal page where a visitor pastes text and an extraction prompt, an LLM
extracts named entities (chemical, disease, gene, food, other) with character spans, and
each entity is mapped with xwalk onto the matching bundled sample ontology (ChEBI, CTD
disease, NCBI Gene, FoodOn; 200 terms each in `src/xwalk/ui/ontologies/`). Hosted on
Vercel.

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| 1 | New top-level `ner-site/` with its own `vercel.json`, `requirements.txt`, `.python-version`; Vercel project Root Directory = `ner-site`. Nothing added to the `xwalk` package. **[amended]** The directory is not called `site/`: that is the name of a standard-library module and the plan puts the directory on `sys.path`. | Keeps the library light and the Vercel build from scanning the repo (the `web/` package.json would otherwise be picked up as a framework). |
| 2 | Static page in `ner-site/public/` (plain HTML/CSS/JS, no build step) plus two file-based Python functions `ner-site/api/ner.py` and `ner-site/api/health.py`, each defining `handler(BaseHTTPRequestHandler)`. No web framework. | The `/api` form is supported and needs no extra dependency. |
| 3 | Python 3.12 (`ner-site/.python-version`). | Vercel's default; tantivy 0.27.0 publishes a `cp311-abi3-manylinux_2_17_x86_64` wheel, so it installs without a compiler. |
| 4 | xwalk installed from git in `ner-site/requirements.txt`: `xwalk @ git+https://github.com/jan3657/xwalk.git@<full sha>`, plus `numpy`. **[amended]** The repository is public (checked: github.com/jan3657/xwalk answers 200 without credentials), so this works on Vercel without a token. | Vercel installs only what the root directory declares; a pinned SHA is reproducible. |
| 5 | NER = one chat completion through `xwalk.llm.openai_compat.OpenAICompatClient` against OpenRouter, wrapped in `BudgetedLLM`. | Reuses xwalk's retry/parse/usage code and its call cap. |
| 6 | Mapping = retrieval only: BM25 + dense fused by reciprocal rank through `xwalk.retrieve.retrieve`, top-k shown with scores. No LLM in the mapping step in v1. | One LLM call per run keeps latency and cost low, and the user's prompt cannot influence which IDs are returned. An LLM-matcher mode is phase 2. |
| 7 | BM25 indexes built at cold start into `/tmp` from the bundled JSONL. Dense indexes precomputed by `ner-site/tools/build_indexes.py` and committed under `ner-site/indexes/<slug>/dense/`. **[amended]** The orchestrating session holds the user's key and builds and commits them after WS-B; no subagent receives the key. | The function filesystem is read-only except `/tmp`; tantivy writes on build. A committed index is verifiable offline. |
| 8 | If a dense index is missing, stale, or the key is unset, the response and `/api/health` report `dense: unavailable` with the reason and mapping runs on BM25 alone. Never silently. | xwalk's no-silent-substitution invariant. |
| 9 | All logic in `ner-site/ner_site/` (importable, tested); `ner-site/api/*.py` are short shims. | Testable without Vercel. |

### Vercel facts relied on (read 2026-11-03 by the planning agent)

- Python runtime, entrypoints, versions, dependencies, 500 MB bundle, `excludeFiles`, cwd is the project root: https://vercel.com/docs/functions/runtimes/python
- File-based `/api` functions, `handler(BaseHTTPRequestHandler)`, `api/ner.py` -> `/api/ner`: https://vercel.com/docs/functions/runtimes/python/api-directory
- Version selection (`.python-version`, 3.12 default): https://vercel.com/docs/functions/runtimes/python/python-version
- Limits: max duration default 300 s (Hobby max 300 s, Pro 800 s); 2 GB / 1 vCPU; request and response body 4.5 MB: https://vercel.com/docs/functions/limitations
- Read-only filesystem with writable `/tmp` up to 500 MB: https://vercel.com/docs/functions/runtimes
- `functions` (`maxDuration`, `includeFiles`, `excludeFiles`), `headers`, `rewrites`: https://vercel.com/docs/project-configuration/vercel-json
- Environment variables: https://vercel.com/docs/environment-variables
- Root Directory for a subdirectory project: https://vercel.com/docs/monorepos
- Unverified: cold-start time with tantivy + pydantic + httpx (measure after deploy); whether `includeFiles` is needed for `ner_site/**` and `indexes/**` (set it anyway).

## 2. Layout

```text
ner-site/
  .python-version          3.12
  requirements.txt         xwalk @ git+...@<sha>, numpy
  vercel.json              functions.maxDuration=120, includeFiles, security headers
  README.md                deploy and local-run instructions
  api/ner.py               POST /api/ner  -> ner_site.http.handle
  api/health.py            GET  /api/health -> ner_site.http.handle
  ner_site/__init__.py
  ner_site/schema.py       pydantic request/response models, Limits, ServiceError
  ner_site/extract.py      prompt, JSON schema, LLM call, span relocation
  ner_site/ontologies.py   catalog, term loading, type->ontology routing, term URLs, templates
  ner_site/indexes.py      BM25 cold-start build in /tmp, dense open, status reporting
  ner_site/mapping.py      fused retrieval per entity
  ner_site/service.py      Settings from env, LLM factory, run()
  ner_site/http.py         framework-free request handling, error envelope, passcode
  ner_site/fake.py         offline stand-in LLM (NER_SITE_OFFLINE=1) for demos and smoke tests
  indexes/<slug>/dense/    vectors.f32, record_ids.json, xwalk_meta.json (committed once built)
  public/index.html, app.js, app.css
  tools/build_indexes.py   builds ner-site/indexes with OPENROUTER_API_KEY
  tools/serve_local.py     stdlib server: public/ + /api/* through ner_site.http (--offline)
  tools/screenshot.mjs     Playwright: load page, run offline sample, screenshot
  tools/fixtures/*.json    canned /api responses for frontend work before WS-D lands
tests/test_site_*.py       offline tests (root tests dir; pytest `pythonpath = ["ner-site"]`)
```

`api/*.py` import `ner_site` lazily inside the handler method after
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`. The bundled ontology JSONL
comes from the installed package via `importlib.resources.files("xwalk.ui") / "ontologies"`.

## 3. Contracts (binding for every workstream)

### 3.1 HTTP

`GET /api/health` -> 200:

```json
{"ok": true, "mode": "live", "default_model": "openai/gpt-4o-mini",
 "models": ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"],
 "limits": {"max_chars": 5000, "max_prompt_chars": 4000, "max_entities": 25, "top_k_max": 10},
 "passcode_required": false, "default_prompt": "...",
 "ontologies": [{"slug": "chebi-sample", "name": "ChEBI (sample)", "type": "chemical", "count": 200}],
 "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "unavailable", "reason": "no index at indexes/chebi-sample/dense; run ner-site/tools/build_indexes.py"}},
 "selectors": ["retrieval", "jev", "openai/gpt-6-luna"], "default_selector": "jev"}
```

`POST /api/ner` with `Content-Type: application/json` and `X-Requested-With: xwalk-ner`
(and `X-Passcode` when required):

```json
{"text": "Aspirin reduced fever in rheumatoid arthritis.", "prompt": null, "model": null, "top_k": 5,
 "selector": null}
```

`prompt`/`model`/`selector` null mean the defaults from `/api/health` (section 13 for
`selector`, `pick` and the `selector_*` usage fields). 200 response:

```json
{"ok": true, "model": "openai/gpt-4o-mini", "mode": "live", "selector": "jev",
 "entities": [
   {"id": "e1", "text": "Aspirin", "start": 0, "end": 7, "type": "chemical", "ontology": "chebi-sample",
    "matches": [{"id": "CHEBI:15365", "label": "acetylsalicylic acid", "score": 0.0325,
                 "retrievers": {"bm25": 1, "dense-qwen3": 2},
                 "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365"}],
    "notes": [],
    "pick": {"selector": "jev", "status": "match", "id": "CHEBI:15365", "label": "acetylsalicylic acid",
             "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365", "confidence": 0.93,
             "explanation": "rubric 0.95, properties 0.97"},
    "more": [{"id": "CHEBI:46195", "label": "paracetamol", "score": 0.0161, "retrievers": {"dense-qwen3": 2},
              "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:46195"}]},
   {"id": "e2", "text": "fever", "start": 16, "end": 21, "type": "other", "ontology": null,
    "matches": [], "notes": ["no ontology for type 'other'"], "pick": null,
    "more": []}],
 "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
 "warnings": ["entity 'feverr' dropped: not found in text"],
 "usage": {"llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 88, "embedding_requests": 2,
           "selector_calls": 3, "selector_prompt_tokens": 2900, "selector_completion_tokens": 0,
           "selector_cost_usd": 0.0001},
 "timing_ms": {"extract": 1800, "map": 900, "select": 1400, "total": 4200}}
```

Token counts are `null` when the provider did not report them, never 0. Errors:

```json
{"ok": false, "error": {"code": "text_too_long", "message": "text is 7000 characters; the limit is 5000"}}
```

| HTTP | code |
|---|---|
| 400 | `bad_json`, `missing_text`, `text_too_long`, `prompt_too_long`, `bad_top_k`, `unknown_model`, `bad_header` (missing `X-Requested-With`) |
| 401 | `passcode_required`, `passcode_invalid` |
| 404 / 405 | unknown path / method (OPTIONS is 405 on purpose) |
| 413 | `body_too_large` (> 64 KB) |
| 422 | `extraction_unparseable` (model answer not a JSON object after xwalk's parsing), `extraction_schema` (object lacks `entities`) |
| 429 | `call_budget_exhausted` |
| 502 | `provider_error` (LLM or embeddings failure; message without the key or URL userinfo) |
| 503 | `not_configured` (no `OPENROUTER_API_KEY` and not offline mode) |
| 500 | `internal` (generic message; details to the function log only) |

Every response carries `Cache-Control: no-store`. No `Access-Control-*` headers are ever
sent.

**[amended by WS-D]** Gaps settled while implementing `ner_site/http.py`:

- 404 is `not_found`; 405 is `method_not_allowed` and carries `Allow` (`POST` on
  `/api/ner`, `GET, HEAD` on `/api/health`). `OPTIONS` is 405 on both, never a preflight
  answer.
- The shim reads at most `Content-Length` bytes and refuses before reading: no
  `Content-Length` on a POST is **411 `length_required`**, a non-numeric or negative one
  is 400 `bad_header`, one over 64 KB is 413 `body_too_large`. `handle` also checks
  the length of the body it is given (413).
- Order of checks on `POST /api/ner`: path, method, body size, `X-Requested-With`,
  passcode, configuration (503), JSON, then field limits. `Content-Type` is not checked:
  `X-Requested-With` already forces a CORS preflight that always fails.
- `/api/health` needs neither `X-Requested-With` nor the passcode (it returns nothing
  secret, and the operator opens it in a browser). It always answers 200, also without a
  key: `mode` is `"live"` and an extra block `"llm": {"status": "unavailable", "reason":
  "OPENROUTER_API_KEY is not set; POST /api/ner answers 503 not_configured"}` says what
  is missing (`{"status": "ok"}` with a key; `ok` with reason "offline stand-in
  extractor" offline). `llm` is not a `HealthResponse` field; the page ignores it.
- A bad environment value (section 3.2) makes both endpoints answer 503 `not_configured`
  with a message naming the variable (never its value), not a traceback.
- Offline mode: `mode` is `"offline"`, the response `model` and the health
  `default_model` are `"offline"`, `models` is `["offline"]`, and a request may name only
  `"offline"` (or null). Offline mode never uses the key, even when one is set: dense is
  unavailable (reason "OPENROUTER_API_KEY is not set", or "offline mode ... requests no
  query embeddings" when a key is set) and `usage` is `llm_calls: 0` with null tokens.
- `start`/`end` are Python `str` indices: Unicode code points, not bytes or UTF-16
  units. `text_too_long` and `prompt_too_long` count code points too.
- `usage.llm_calls` counts dispatched HTTP requests, retries included. The token counts
  are null unless every request reported them. `embedding_requests` is a lower bound
  (`MapStats`).
- Embedding failures at request time do not fail the request: the entity gets a note,
  `warnings` summarises it, and BM25 matches are still returned. `provider_error` (502)
  is for the extraction call only, and also covers an extraction that takes longer than
  90 s (retries included). Each retriever call then gets `min(30 s, what is left of
  115 s)`, at least 5 s. **[superseded after review, below]**
- When the retrieval block says dense (or BM25) is unavailable and some entity was routed,
  `warnings` also carries `"dense retrieval unavailable; matches use BM25 only"`. A reason
  shared by all four ontologies is reported once, with "(all 4 ontologies)".
- The key, and a passcode of 8 or more characters, are also string-replaced out of every
  response body and log line as a last line of defence. Provider messages pass through
  `redact_url` for every URL and are cut to 300 characters.
- **[amended after review]** Time: the whole mapping step gets one wall-clock deadline,
  what is left of 100 s since the request began (opening the indexes and the
  extraction included; at least 5 s); each retriever call within
  it at most 30 s. A dense search still running at the deadline is cut off and one not
  yet started is skipped (BM25 gets 5 s of grace, so every entity keeps its BM25
  matches); the entity gets the note `"dense-qwen3: cut off by the request deadline"`
  and `warnings` the usual `"<note> (<entity ids>)"` summary. Worst case about 105 s,
  inside `maxDuration` 120 s. Each request runs on its own event loop and thread pool,
  shut down without waiting (`http.run_request`): an abandoned embeddings thread
  finishes in the background and cannot touch a later response. Query embeddings use
  `max_retries=0` (timeout 15 s), so a 429 with a long `Retry-After` fails at once
  instead of sleeping.
- **[amended after review]** Dense at request time: when every entity that reached a
  dense search had it fail, time out or be cut off, the response's `retrieval.dense` is
  `unavailable` with reason `"every dense search failed in this request: <first note>"`
  (plus the `"dense retrieval unavailable; matches use BM25 only"` warning), even though
  `/api/health` says `ok`. When only some failed, dense stays `ok` and `warnings` adds
  `"dense search failed for N of M entities (e2, e5); their matches use BM25 only"`.
- **[amended after review]** Entity `notes` from retrievers pass through the same scrub
  as provider messages (key and passcode removed, every URL through `redact_url`) and
  are cut to 300 characters.
- **[amended after review]** `bad_json` also covers a `text`, `prompt` or `model` that
  is not encodable as UTF-8 (a lone surrogate escape such as `"\ud800"`), which used to
  be a 500. `Content-Length` must be 1 to 18 ASCII digits (`"²"` passes `str.isdigit()`
  but not `int()`); anything else is 400 `bad_header`.
- **[amended after review]** `default_prompt` (health, and the prompt used when a
  request sends none) states the configured `NER_SITE_MAX_ENTITIES`
  (`extract.render_default_prompt(limit)`; `extract.DEFAULT_PROMPT` is the one for 25).
- Vercel (https://vercel.com/docs/functions/runtimes/python/api-directory, read
  2026-11-03): a file-based function defines a top-level `handler` class that inherits
  from `BaseHTTPRequestHandler` (or `app`/`application` for ASGI/WSGI); `api/ner.py`
  serves `/api/ner`. The request is read the standard way (`self.command`, `self.path`,
  `self.headers`, `self.rfile`). Both shims are identical: `do_*` call
  `ner_site.http.serve(self)`, which routes by path, so a request reaching the wrong file
  still gets the right 404/405. `log_message` is silenced so each request logs one line.

### 3.2 Environment variables (function runtime)

| Name | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | unset | the operator's key; used for the chat call and query embeddings |
| `NER_SITE_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. **[amended after review]** For the chat call only: query embeddings always go to OpenRouter (`https://openrouter.ai/api/v1/embeddings`) with `OPENROUTER_API_KEY`. The committed dense indexes record that embeddings base URL in their identity, so moving embeddings elsewhere means rebuilding them (`tools/build_indexes.py`) and changing the code |
| `NER_SITE_MODELS` | `openai/gpt-4o-mini,google/gemini-2.5-flash-lite,qwen/qwen3-next-80b-a3b-instruct` | allowlist, first is default. **[amended]** All three ids exist on OpenRouter and list `structured_outputs` (checked 2026-11-03) |
| `NER_SITE_PASSCODE` | unset | when set, `X-Passcode` must equal it (`hmac.compare_digest`). **[amended after review]** ASCII only: `http.server` reads header values as Latin-1, so another value could never match; it is refused (503 `not_configured` naming the variable) |
| `NER_SITE_MAX_CHARS` / `NER_SITE_MAX_ENTITIES` / `NER_SITE_MAX_PROMPT_CHARS` | 5000 / 25 / 4000 | limits. **[amended by WS-D]** Whole numbers in 1..20000 / 1..100 / 1..16000; anything else is refused (503 `not_configured` naming the variable), not clamped. Empty means unset. `NER_SITE_OFFLINE` accepts 1/0, true/false, yes/no, on/off |
| `NER_SITE_OFFLINE` | unset | `1` swaps the LLM for `ner_site.fake.offline_llm`; `/api/health` reports `"mode": "offline"` |
| `NER_SITE_DENSE_DIR` | `<project root>/indexes` | where committed dense indexes live |
| `NER_SITE_CACHE_DIR` | `/tmp/xwalk-ner-site` | BM25 build target |

### 3.3 Module signatures

`ner-site/ner_site/schema.py` (WS-0; pydantic v2):

```python
from typing import Literal

EntityType = Literal["chemical", "disease", "gene", "food", "other"]


class Entity:  # pydantic BaseModel in the real file
    id: str
    text: str
    start: int
    end: int
    type: EntityType


class Match:
    id: str
    label: str
    score: float
    retrievers: dict[str, int]
    url: str | None


class EntityResult(Entity):
    ontology: str | None
    matches: list[Match]
    notes: list[str]


class RetrieverStatus:
    status: Literal["ok", "unavailable"]
    reason: str | None = None


class NerRequest:
    text: str
    prompt: str | None = None
    model: str | None = None
    top_k: int = 5


class ServiceError(Exception):
    def __init__(self, code: str, message: str, *, status: int) -> None: ...
```

Also `NerResponse`, `HealthResponse`, `Usage` (llm_calls, prompt_tokens | None,
completion_tokens | None, embedding_requests) and `Limits(max_chars=5000,
max_prompt_chars=4000, max_entities=25, top_k_max=10, max_body_bytes=65536,
llm_call_cap=3)`.

`ner-site/ner_site/extract.py` (WS-A):

```python
from xwalk.llm.base import LLMClient, LLMRequest

DEFAULT_PROMPT: str  # [amended after review] = render_default_prompt(25)
def render_default_prompt(max_entities: int) -> str: ...  # Settings.default_prompt uses it
SYSTEM_PROMPT: str  # fixed, not user-editable
ENTITY_SCHEMA: dict  # JSON schema passed as LLMRequest.schema


def build_request(text: str, prompt: str) -> LLMRequest: ...
def locate(text: str, mention: str, hint: int | None) -> tuple[int, int] | None: ...
def parse_entities(raw: str, text: str, *, max_entities: int) -> tuple[list, list[str]]: ...
async def extract(llm: LLMClient, text: str, prompt: str, *, max_entities: int) -> tuple[list, list[str], object]: ...
```

(`extract` returns entities, warnings and the xwalk `Usage` of the call.)

`ner-site/ner_site/ontologies.py` (WS-B):

```python
from xwalk.records import Record
from xwalk.templates import TemplateSet

ROUTES: dict[str, str]  # chemical->chebi-sample, disease->ctd-disease-sample, gene->ncbi-gene-sample, food->foodon-sample
TEMPLATES: TemplateSet  # query "{{ mention }}", doc "{{ label }} {{ synonyms | join(' ') }} {{ definition }}"


def catalog() -> list[dict]: ...
def load_terms(slug: str) -> list[Record]: ...
def route(entity_type: str) -> str | None: ...
def term_url(record_id: str) -> str | None: ...
```

`ner-site/ner_site/indexes.py` (WS-B):

```python
from collections.abc import Callable, Mapping
from pathlib import Path


class OntologyIndex:  # dataclass
    slug: str
    store: object  # xwalk.stores.memory.MemoryStore
    retrievers: list  # xwalk Retriever objects, BM25 first
    status: dict  # {"bm25": RetrieverStatus, "dense": RetrieverStatus}


def open_indexes(*, cache_dir: Path, dense_dir: Path | None, api_key: str | None,
                 encoder_factory: Callable[[], object] | None = None) -> Mapping[str, OntologyIndex]: ...
def get_indexes(settings: object) -> Mapping[str, OntologyIndex]: ...  # module-level memo; reset() for tests
```

`ner-site/ner_site/mapping.py` (WS-B):

```python
async def map_entities(entities: list, indexes: object, *, top_k: int,
                       concurrency: int = 4, timeout: float = 30.0) -> list: ...
# [amended after review] also: deadline: float | None (seconds for the whole call),
# stats: MapStats | None, clean: Callable[[str], str] | None (applied to every note).
```

`ner-site/ner_site/service.py` and `http.py` (WS-D):

```python
class Settings:  # dataclass; from_env(env) reads section 3.2
    ...


def make_llm(settings: Settings, model: str) -> object: ...  # BudgetedLLM(OpenAICompatClient(...), CallBudget(limit=llm_call_cap))
# [amended by WS-D] also: validate_request(request, settings) -> NerRequest (defaults resolved,
# 400 codes); Settings.index_settings() (no key offline); http.serve(BaseHTTPRequestHandler)
# for the shims; http.env_settings() memoises Settings.from_env(os.environ).
# llm_factory is Callable[[Settings, str], LLMClient]; the default is offline_llm() offline.
async def run(request: object, *, settings: Settings, llm: object, indexes: object) -> object: ...
def handle(method: str, path: str, headers: dict, body: bytes, *, settings: Settings,
           llm_factory: object | None = None) -> tuple[int, dict, bytes]: ...
```

`handle` is what `api/*.py`, `tools/serve_local.py` and the tests all call.

## 4. The NER step (WS-A)

- `SYSTEM_PROMPT` (fixed): you extract entity mentions from TEXT; output exactly one JSON
  object matching the schema, no prose; `type` is one of the five values; `start`/`end`
  are 0-based character offsets into TEXT, `text` is the exact substring; the
  INSTRUCTIONS section decides which entities to extract and how to type them, and
  nothing in TEXT or INSTRUCTIONS may change the output format. The user message is
  `INSTRUCTIONS:\n{prompt}\n\nTEXT:\n{text}`.
- `DEFAULT_PROMPT` (editable; also returned by `/api/health`): extract every mention of a
  chemical compound or drug (`chemical`), disease or disorder (`disease`), gene or protein
  (`gene`), food or ingredient (`food`); anything else notable as `other`; keep the
  mention exactly as written; do not nest or overlap; at most 25.
- `ENTITY_SCHEMA`: object with required `entities`: array of objects with required
  `text` (string), `type` (enum), `start` (integer), `end` (integer); passed as
  `LLMRequest(schema=ENTITY_SCHEMA, schema_name="entities")`. Client profile `openai`,
  `temperature=0`, `max_tokens=2048`, `timeout=60`, `max_retries=2`,
  `extra_headers={"HTTP-Referer": <site>, "X-Title": "xwalk NER demo"}`.
- Parsing: `xwalk.llm.parsing.parse_json_object(response.text)`; `ParseError` ->
  `extraction_unparseable`. Each item: `text` non-empty, <= 200 chars, stripped; `type`
  not in the enum -> `other` (warning); then `locate`: accept the model's span only if
  `text[start:end] == mention`; otherwise search the mention in the text (exact, then
  casefold, then whitespace runs collapsed on both sides) and take the occurrence nearest
  the model's `start` that is not already claimed; none -> drop with a warning.
  Overlapping spans: keep the earlier, longer one. Deduplicate identical spans. Truncate
  to `max_entities` with a warning. Sort by `start`. Ids `e1..eN`.
- Limits: `text` 1..5000 chars (offsets stay the visitor's), `prompt` <= 4000 chars,
  request body <= 64 KB. `CallBudget(limit=3)` covers xwalk's internal retries.

## 5. The mapping step (WS-B)

- Routing: `route(type)` through `ROUTES`; `other` -> no ontology, `notes:
  ["no ontology for type 'other'"]`, no retrieval, no cost.
- Per ontology at cold start (`open_indexes`): `records = load_terms(slug)`, `store =
  MemoryStore.from_source(records)`, `BM25Retriever.build(records, TEMPLATES,
  cache_dir / slug / "bm25", name="bm25", exact_fields=("label", "synonyms"),
  default_limit=20)` guarded by a `threading.Lock` and reused when already built (warm
  instance). Record the measured build time for the four samples in `ner-site/README.md`.
- Dense: when `dense_dir / slug / "dense" / "xwalk_meta.json"` exists and `api_key` is
  set, `encoder = default_embeddings_encoder(api_key=api_key)` and
  `DenseRetriever.open(dir, encoder, expected=index_components(..., name_fields=("label",
  "synonyms")))`; `FileNotFoundError`, `IndexMismatchError`, `ValueError` or a missing
  key -> `status["dense"] = unavailable(reason)` and BM25 only.
- `ner-site/tools/build_indexes.py --out ner-site/indexes [--only slug]`:
  `DenseRetriever.build(records, TEMPLATES, out / slug / "dense", encoder,
  name="dense-qwen3", default_limit=20, name_fields=("label", "synonyms"))`, prints
  `encoder.usage` (cost as reported, unknown when not reported).
- Per entity: `xwalk.retrieve.retrieve([mention], Record(entity.id, {"mention": ...}),
  index.retrievers, index.store, timeout=timeout)` -> fused `Candidate`s; take `top_k`;
  `Match(id=c.id, label=c.record.fields.get("label", ""), score=c.fused_score,
  retrievers={h.retriever: h.rank for h in c.evidence}, url=term_url(c.id))`. Degradation
  notes go into the entity's `notes` and are summarised in `warnings`. IDs only ever come
  from the store; no model text becomes an ID.
- Concurrency `asyncio.Semaphore(4)`; each dense search is one embeddings request.
- Term URLs (best effort): `CHEBI:n` -> `https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:n`;
  `MESH:x` -> `https://meshb.nlm.nih.gov/record/ui?ui=x`; `OMIM:n` -> `https://omim.org/entry/n`;
  `NCBIGene:n` -> `https://www.ncbi.nlm.nih.gov/gene/n`; `FOODON:n` ->
  `http://purl.obolibrary.org/obo/FOODON_n`; otherwise `null`.
- Latency budget: extract <= 60 s, each retriever call <= 30 s, `maxDuration` 120 s.
  **[amended after review]** Extraction <= 90 s, the whole mapping step bounded by one
  deadline (section 3.1), so a request ends by about 105 s.

## 6. The page (WS-C)

Single `public/index.html` with `app.css` and `app.js` (ES2020, no dependencies, no
build). On load it fetches `/api/health` and fills: model `<select>` (hidden when one
model), the prompt `<textarea>` inside a collapsed `<details>` with a "Reset to default"
button, the limits shown as hints, a passcode `<input type="password">` only when
`passcode_required` (kept in `sessionStorage`), and a "Load example" button whose text
uses mentions that exist in the samples (pick them from the JSONL). Form: text
`<textarea>` with a live character counter against `max_chars`, `top_k` select (1..10),
Run button (disabled while running, `aria-busy`). Results: the text re-rendered with one
`<mark class="type-chemical">` per entity (colour plus a type badge in `title` /
`aria-label`, so colour is not the only signal), then one card per entity: mention, type,
ontology, a table of matches (id as a link when `url`, label, score to 4 decimals,
retriever ranks), notes. A status line shows `retrieval` and `usage`/`timing`. Errors
render in a `role="alert"` box with the envelope's message. Layout: one column,
`max-width: 60rem`, 16 px base font, `textarea { width: 100% }`, buttons >= 44 px tall,
works at 360 px wide. Build DOM with `textContent`, never `innerHTML` with data. CSP in
`vercel.json`: `default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'`.

`tools/serve_local.py [--offline] [--canned tools/fixtures] --port 8765` serves `public/`
and routes `/api/*` through `ner_site.http.handle` (or canned JSON until WS-D lands).
`tools/screenshot.mjs` (Playwright, preinstalled) loads the page, clicks "Load example"
and "Run", waits for results, and writes a PNG to a path given on the command line.

## 7. Security and abuse (WS-D, with C for headers)

- The key lives only in the function environment; `/api/health` never echoes it or the
  base URL's userinfo; URLs in error messages pass through `xwalk.fingerprint.redact_url`.
- Body cap 64 KB before JSON parsing. Text/prompt/top_k/model validated per section 3.
- Cost cap per request: `BudgetedLLM` with `CallBudget(limit=3)`; entities <= 25 bound
  the embedding requests; `maxDuration` 120 s bounds everything else.
- Optional passcode `NER_SITE_PASSCODE` (`hmac.compare_digest`); 401 otherwise.
  Recommended with an OpenRouter per-key spend limit (the only hard cap across function
  instances).
- Cross-origin: require `X-Requested-With: xwalk-ner`; OPTIONS -> 405 with no CORS headers.
- Prompt injection: the visitor's prompt only reaches the extraction call, inside a fixed
  system prompt; output validated against the schema and the text; mapping has no LLM;
  the page renders model text as text only.
- Headers via `vercel.json`: CSP above, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on `/api/*`.
- Logging: one line per request with sizes, counts, timings, status, model; never the
  text, prompt or key.

## 8. Testing and gates

Offline only; no test needs a key or the network.

| File | Covers | Owner |
|---|---|---|
| `tests/test_site_schema.py` | response round-trip through `model_validate`/`model_dump`; `Limits` defaults | WS-0 |
| `tests/test_site_extract.py` | `build_request`; `locate` (exact, casefold, whitespace, nearest-to-hint, unclaimed); `parse_entities` (fenced JSON, wrong offsets relocated, missing mention dropped with warning, unknown type -> other, overlap, truncation, non-object -> `extraction_unparseable`); `extract` with `FakeLLM` | WS-A |
| `tests/test_site_mapping.py` | `load_terms` (200 each), `route`, `term_url`; `open_indexes` in `tmp_path` builds BM25 and reports dense unavailable without a dir or key; a fake dense index from a 26-d letter-histogram encoder gives dense `ok` and fused ranks naming both retrievers; stale index -> unavailable with reason; `other` makes no retriever call; a raising retriever yields notes, not a 5xx; top_k honoured; `build_indexes.py` against the histogram encoder into `tmp_path` | WS-B |
| `tests/test_site_service.py` | `run` end to end with `FakeLLM`; `Settings.from_env`; `make_llm` wraps `BudgetedLLM`; `offline_llm` | WS-D |
| `tests/test_site_http.py` | every error code in 3.1 via `handle`; passcode; header requirement; OPTIONS 405; health shape; LLM via `httpx.MockTransport`; embeddings via monkeypatched `httpx.Client` (pattern in `tests/test_ui_dense.py`) | WS-D |

Write each test first and see it fail for the stated reason; FakeLLM and MockTransport
only; no network; no paid call.

Gate changes (WS-0): `pyproject.toml` `[tool.pytest.ini_options] pythonpath =
["ner-site"]` (append to any existing value); `[tool.mypy]` `files` gains
`ner-site/ner_site`, `ner-site/api`, `ner-site/tools` and `mypy_path = "ner-site"` (**[amended by WS-D]** the
last two were added by WS-D once they existed; `mypy` with no arguments checks all 13
ner-site files, as top-level modules `ner`, `health`, `serve_local`, `build_indexes`; module
names must stay unique: no `__init__.py` under `ner-site/api` or `ner-site/tools`, and
file names must not repeat ones in `scripts/` or `examples/`); the ruff gate commands
gain `ner-site`; `.github/workflows/ci.yml` lint job adds `ner-site` to both ruff lines
(`tests/test_packaging.py::test_ci_checks_the_same_paths_as_the_local_gate` must still
pass); `CLAUDE.md` and `CONTRIBUTING.md` list the new path; `docs/README.md` links
`ner-site/PLAN.md`. The site runs on 3.12 on Vercel and 3.13 locally: nothing 3.13-only.

Smoke (WS-D): `python ner-site/tools/serve_local.py --offline` then `node
ner-site/tools/screenshot.mjs http://127.0.0.1:8765 <scratch>/ner.png`; inspect the PNG;
`curl` the health and one POST. Say exactly what was run.

## 9. Workstreams

No two workstreams edit the same file. Order: WS-0, then A, B, C in parallel, then D,
then the orchestrator builds the dense indexes with the user's key and runs a live check.

**WS-0 Scaffold.** `ner-site/ner_site/__init__.py`, `ner-site/ner_site/schema.py`,
`tests/test_site_schema.py`, `pyproject.toml`, `.github/workflows/ci.yml`, `CLAUDE.md`,
`CONTRIBUTING.md`, `docs/README.md`, `ner-site/README.md` stub. Gates green.

**WS-A Extraction.** `ner-site/ner_site/extract.py`, `tests/test_site_extract.py`.
Section 4.

**WS-B Ontologies, indexes, mapping.** `ner-site/ner_site/{ontologies,indexes,mapping}.py`,
`ner-site/tools/build_indexes.py`, `ner-site/requirements.txt`,
`ner-site/.python-version`, `tests/test_site_mapping.py`. Section 5.

**WS-C Frontend and local tooling.** `ner-site/public/{index.html,app.js,app.css}`,
`ner-site/vercel.json`, `ner-site/tools/serve_local.py`, `ner-site/tools/screenshot.mjs`,
`ner-site/tools/fixtures/*.json`; WS-C may append to `ner-site/README.md` only after WS-0
created it (WS-0 owns the stub, WS-C the deploy section). Sections 6 and 7 (headers).

**WS-D Service, HTTP, integration (after A, B, C).** `ner-site/ner_site/{service,http,fake}.py`,
`ner-site/api/{ner,health}.py`, `tests/test_site_service.py`, `tests/test_site_http.py`.
Pin the xwalk SHA in `ner-site/requirements.txt` to a pushed commit, run the offline
smoke and all four gates, review the diff against sections 3 and 7, update
`docs/claude-upgrade/STATUS.md`.

## 10. Open questions (defaults chosen)

1. Dense index: committed (default; the orchestrator builds it).
2. NER model allowlist: verified ids above; the operator may change `NER_SITE_MODELS`.
3. Passcode: off by default, recommended on for a public deployment.
4. Phase 2, LLM candidate selection ("match" mode) with `spec.build_matcher(...)`: out of v1.
5. Shared per-IP rate limiting needs storage; v1 relies on passcode, caps and the
   OpenRouter spend limit (Vercel firewall rate limiting is an operator option).

## 11. After deploying (operator)

1. Vercel project: import the repo, Root Directory `ner-site`, Framework Preset "Other".
2. Environment variables: `OPENROUTER_API_KEY` (sensitive), optionally `NER_SITE_PASSCODE`,
   `NER_SITE_MODELS`.
3. Check `GET /api/health`: `mode: live`, `dense: ok`, the model list.
4. Run the example text; confirm entities, spans and matches; check the function log for
   cold-start time; confirm a 401 without the passcode if set.
5. Set a spend limit on the OpenRouter key. Record what was verified in `ner-site/README.md`.

## 12. Scaling with Neon **[amendment, 2026-11-04]**

The bundled backend (sections 5 and 7: BM25 in `/tmp`, dense vectors committed) holds the
four 200-term samples and cannot hold the full catalogs (629,558 records, 1,934,527
per-name vectors). Decision: keep records, BM25 and vectors in Neon Postgres, behind a
backend switch, with the bundled backend unchanged as the default. Details, measurements
and the full-load command: [NEON.md](NEON.md).

| # | Decision | Why |
|---|---|---|
| 12.1 | `NER_SITE_BACKEND=bundled\|neon` (default `bundled`); `neon` needs `DATABASE_URL`. `NER_SITE_NEON_SCHEMA` (`xwalk`), `NER_SITE_NEON_DIMENSIONS` (1024), `NER_SITE_NEON_CATALOG=samples\|full`. | The current deployment is untouched; a misconfigured Neon backend reports why instead of failing. |
| 12.2 | Library code in `xwalk.retrieval.postgres` (`PostgresBM25Retriever`, `PgVectorRetriever`, `PostgresStore`, `PostgresIndexWriter`), not in the site. | Reusable by any xwalk job; the site stays thin. |
| 12.3 | All SQL through an executor protocol; `NeonHTTPExecutor` over HTTPS with httpx. | Works where TCP 5432 is blocked and on Vercel; no new base dependency. |
| 12.4 | BM25 is Neon's `lakebase_text` (`lakebase_bm25` index, `<@>`), not `pg_search`. | Neon refuses new `pg_search` installs (deprecated). |
| 12.5 | Vectors are `halfvec(1024)` under HNSW (inner product), one row per doc and per name, max-pooled per record. | Same quality as the committed float32 index (NEON.md, "Vector format"). |
| 12.6 | One partial BM25 and one partial HNSW index per ontology. | Per-ontology statistics and graphs, as the per-ontology Tantivy indexes. |
| 12.7 | Identity in `index_meta`, written by `ner-site/tools/load_neon.py` (`loading` before, `ready` after); the site opens with `expected` and reports missing, incomplete, stale or unreachable as the status reason. | No silent substitution; an interrupted load is never served. |
| 12.8 | The full catalogs are not loaded: they need about 12.5 GB at 1024 dimensions (6.6 GB at 512), the free plan has 0.5 GB. The loader refuses a full load over `--max-bytes` (400 MB). | Measured: 5.7 KB a stored 1024-d vector (TOAST plus HNSW). |
| 12.9 | **Amended 2026-11-04 (Launch plan).** The full catalogs are loaded into schema `xwalk_full` at `halfvec(768)`, NCBI Gene excepted; the samples stay at 1024 in `xwalk`. Production: `NER_SITE_NEON_SCHEMA=xwalk_full`, `NER_SITE_NEON_DIMENSIONS=768`, `NER_SITE_NEON_CATALOG=full`. | 768 matched 1024 on the full CTD vocabulary and 512 lost 0.07 MRR; 1000 or 1016 saves nothing (`benchmarks/results/neon-truncation-2026-11-04/FINDINGS.md`). Genes were skipped by the operator, so `FULL_ROUTES` leaves them out and the default prompt does not ask for them. |

Response shapes are unchanged; `/api/health` gains `"backend"` and, with Neon, a `"neon"`
block. Tests: `tests/test_postgres_retrieval.py`, `tests/test_site_neon.py` (offline), and
`tests/test_postgres_integration.py` plus one site test (`-m integration`, need
`DATABASE_URL`). The Vercel build needs `requirements.txt` re-pinned to a commit that
contains `xwalk.retrieval.postgres` before `NER_SITE_BACKEND=neon` can work there.



## 13. Selection: Jev or an LLM picks the match **[amendment, 2026-11-04]**

Retrieval ranks; a selector decides. Each request names one (`selector`, default
`NER_SITE_SELECTOR`, `retrieval` unless set), from the list `/api/health` serves
(`selectors`, `default_selector`):

| # | Decision | Why |
|---|---|---|
| 13.1 | `retrieval`: no selector; the fused ranking is the answer, as before. Always offered, first. | The selector is optional and costs calls. |
| 13.2 | `jev`: TypeSafe's Jev through `xwalk.decide.DecisionMatcher` (screen, choose or none, gate on the slots' properties) over the entity's first 30 fused candidates, with the thresholds the examples' `job_jev.yaml` fitted on the pilots, per entity type. Jev through OpenRouter's decisions endpoint with the site's key; at most 4 calls per entity (`BudgetedDecider`). | Probabilities, not text: Jev chooses among keys, so nothing it returns can become an id outside the candidates. Measured: 3 calls an entity, about $0.0004 an entity as reported, under a second for five entities. |
| 13.3 | Any other selector is an LLM (`xwalk.stages.select.Selector`): one call an entity, an opaque key or null, exact-only resolution. `NER_SITE_SELECTORS` lists them; the default is Jev, `openai/gpt-6-luna`, `anthropic/claude-haiku-5.5`, `google/gemini-3.1-flash-lite` (each answered live requests on 4 November 2026). | The library's own selection stage and invariants; a malformed answer is a failed pick, never an id. |
| 13.4 | The question sets are `ner_site/slots/<type>.yaml`, copied from the examples (`chebi/slots.yaml`, `ncbi_disease/slots_jev.yaml`, `cafeteria_fcd/slots_jev.yaml`, `nlm_gene/slots.yaml`). The selector sees 150 characters either side of the mention. | Reviewed slots with properties already exist; context disambiguates acronyms. |
| 13.5 | Each routed entity with candidates gets `pick`: `selector`, `status` (`match`, `review`, `no_match`, `failed`), `id`, `label`, `url`, `confidence`, `explanation`. `matches` (the retrieval ranking) is unchanged. A provider error, a timeout or the request deadline is a `failed` pick with its reason, never a 5xx. | The visitor sees both the ranking and the decision. |
| 13.6 | Time: the selection gets what is left of the 100 s request deadline, at least 5 s; usage adds `selector_calls`, `selector_prompt_tokens`, `selector_completion_tokens` and `selector_cost_usd` (Jev reports cost; an LLM selector's cost is null, unknown). `timing_ms` adds `select` when a selector ran. | Same accounting rules as the extraction call: unknown is null, not 0. |
| 13.7 | Offline (`NER_SITE_OFFLINE=1`): `retrieval` and `jev`, where Jev is the library's token-overlap stand-in. | Demonstrates the flow without a key. |
| 13.8 | `matches` stays the visitor's top_k; `more` continues the same fused ranking down to 30 (`mapping.MAX_LISTED`), the depth the selector chooses from. The page lists `matches` under "Retrieved candidates" and expands to `more`, so a pick is always on the page with its rank. | A pick ranked below the visible rows looked like it came from nowhere (Jev picked rank 20 for "eggs"). |
| 13.9 | **Real-world fixes (4 November 2026).** Every site LLM request to OpenRouter carries `reasoning: {enabled: false}`; `anthropic/` models use the `anthropic-compat` profile (JSON in the prompt); the extraction prompt and schema list only the routed types plus "other"; mapping embeds all entities' queries in one request (`PrefetchEncoder`); `/api/health` checks the key with OpenRouter. Claude Haiku 5.5 is a selector only. | Reproduced on PubMed abstracts and Wikipedia food texts: `benchmarks/results/site-realworld-2026-11-04/FINDINGS.md`. |

Tests: `tests/test_site_select.py` (FakeDecider with a scripted Jev, FakeLLM), and the
selection cases in `tests/test_site_service.py`.

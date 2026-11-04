# OpenRouter dense retrieval: findings and implementation brief

> **Update (2026-11-03, later the same day).** Implemented as
> `xwalk.retrieval.embeddings_api.OpenAIEmbeddingsEncoder` / `encoder: openai_embeddings`
> (see `docs/reference/job-file.md`), close to the sketch below with these differences:
> the encoder lives in its own module; `dimensions` is required and `send_dimensions`
> (default true) controls whether it is sent; `truncate` cuts and renormalises a longer
> vector; `probe_dimension()` exists for tooling only. The default model is **not
> chosen**: `python -m benchmarks.embeddings` measures the candidates on the pilot gold
> sets and has not been run against a real endpoint yet. Newer than the table below:
> `nvidia/nemotron-3.5-embed-8b:free` and `-1b:free` (released 2026-10-21; 4,096-d for
> the 8B, truncatable with renormalisation; `query: ` / `passage: ` roles; benchmarked by
> NVIDIA only on visual-document retrieval, ViDoRe v3.1 first at 61.17 nDCG@10), and
> `liquid/lfm-2.5-embedding-350m:free`.

Research only (2026-11-03). No code changed, no paid calls made. Prices and model
metadata were read from `GET https://openrouter.ai/api/v1/embeddings/models` and
`GET https://openrouter.ai/api/v1/models/<id>/endpoints` on that date.

## What xwalk has today

- `xwalk.retrieval.dense.Encoder` is a three-member protocol (`name`, `dimension`,
  `encode(texts, *, is_query)`), and the module docstring already promises that "an API
  embedding service can be plugged in with no torch installed at all". Only
  `SentenceTransformerEncoder` implements it.
- `RetrieverSpec(kind="dense")` in `config.py` has sentence-transformers-only fields
  (`device`, `revision`, `normalize`, `query_prefix`, `doc_prefix`) and
  `required_extra` hardwires `("dense", "sentence_transformers")`.
  `JobSpec.build_encoder` always constructs a `SentenceTransformerEncoder`.
- Index identity (`index_components` -> `encoder_identity`) records `name`, `dimension`
  and the encoder's `settings` mapping; `DenseRetriever.open` refuses an index whose
  components differ. Any new encoder must expose a `settings` mapping covering
  everything that changes a vector.
- The LLM side already talks to OpenRouter through `OpenAICompatClient`
  (`llm/openai_compat.py`): httpx, retry on `RETRYABLE_STATUSES`, OpenRouter's
  "HTTP 200 with an `error` body" quirk, `metadata.raw` upstream-error unwrapping,
  `api_key_env` credentials, `redact_url` in fingerprints. All of that is reusable.

## The OpenRouter embeddings API

- `POST https://openrouter.ai/api/v1/embeddings`, bearer `OPENROUTER_API_KEY`. The body is
  OpenAI-shaped: `model`, `input` (string or list of non-empty strings), optional
  `dimensions` (int), `encoding_format` (`float` | `base64`), `input_type`
  (`search_query` | `search_document`), `provider` (routing preferences: `order`,
  `only`, `allow_fallbacks`, `data_collection`, `zdr`, `max_price`, ...).
- Response: `data[i].embedding` in input order, `model`, `usage.prompt_tokens`,
  `usage.total_tokens`, `usage.cost` (credits). No streaming.
- Errors use the same `{error: {code, message, metadata}}` shape as chat, with the same
  status set (400/401/402/404/413/429/5xx/529). Expect the 200-with-error-body quirk too.
- No documented batch or payload limit; 413 exists. Plan for a configurable batch size
  (default 64-128 strings) and treat 413 as "halve the batch", not as fatal.
- Per-model `supported_parameters` is empty for every embedding model, so whether
  `dimensions` / `input_type` are honoured is not machine-readable. OpenRouter's own blog
  verified `dimensions` on text-embedding-3-small (256), voyage-4-large (512) and
  gemini-embedding-2 (768), and says "some models" use `input_type`. Pass-through for
  other models is unverified: the first integration test should assert the returned
  vector length equals the configured `dimensions`, and refuse the index otherwise.

## Models available (text, retrieval-relevant), price per 1M input tokens

| Model id | $/1M | Dims (default) | Context | Notes |
|---|---|---|---|---|
| `perplexity/pplx-embed-v1-0.6b` | 0.004 | 1,024 | 32,000 | Cheapest paid; single provider (Perplexity); Mar 2026; web-scale retrieval tuned, open weights |
| `qwen/qwen3-embedding-8b` | 0.01 | 4,096 | 32,768-40,960 | Open weights, 100+ languages, 3 providers (Nebius, DeepInfra, SiliconFlow); most-used embedding model on OpenRouter; wants an `Instruct: ...\nQuery:` prefix on queries, none on docs |
| `baai/bge-m3` | 0.01 | 1,024 | 8,194 | Multilingual, mature, open weights |
| `intfloat/multilingual-e5-large` | 0.01 | 1,024 | 512 | Same family as the `multilingual-e5-small` used in every example job (`query: ` / `passage: ` prefixes) |
| `qwen/qwen3-embedding-4b` | 0.02 | 2,560 | 32,768 | |
| `openai/text-embedding-3-small` | 0.02 | 1,536 | 8,192 | `dimensions` verified; two providers (OpenAI, Azure); English-first |
| `voyageai/voyage-4-lite` | 0.02 | 1,024 (256-2,048 Matryoshka) | 32,000 | Multilingual, shares a vector space with `voyage-4` / `voyage-4-large`; single provider |
| `voyageai/voyage-4` | 0.06 | 1,024 | 32,000 | Stronger tier, same space |
| `voyageai/voyage-4-large` | 0.12 | 1,024 (256-2,048) | 32,000 | `dimensions` verified |
| `openai/text-embedding-3-large` | 0.13 | 3,072 | 8,192 | |
| `google/gemini-embedding-2` | 0.20 | 3,072 | 8,192 | Multimodal; priciest text option |
| `nvidia/nemotron-3.5-embed-1b:free` | 0 | 2,048 | 32,768 | Free, but requires the account's free-model data-sharing setting or requests 404; not for reproducible benchmarks |

Also listed but older or weaker: `thenlper/gte-*`, `intfloat/e5-*-v2`, `baai/bge-*-en`,
`sentence-transformers/all-*` (all 0.005-0.01, 512 ctx), `mistralai/mistral-embed-2312`
(0.10), `openai/text-embedding-ada-002` (0.10, legacy), `amazon/titan-embed-text-v2`
(0.02).

Cost is negligible for xwalk's workloads either way: an ontology document rendered by the
example `doc` template is ~30-80 tokens, so embedding a 30k-concept target with a $0.02
model costs ~$0.04, and a 2,000-row source ~$0.002. Quality and operability matter more
than the per-token price.

Quality benchmarks: no independent leaderboard numbers for voyage-4-lite vs
text-embedding-3-small vs qwen3-embedding-8b were found that I would trust; third-party
sites disagree. OpenRouter's own guidance ("test at least two candidates on labelled
queries from your own data") matches xwalk's existing benchmark harness and the
`ref_zivila` gold set.

## Recommendation

Implement one generic `OpenAIEmbeddingsEncoder` (not an OpenRouter-specific class): a
`base_url` + `model` + `api_key_env` encoder that works against OpenRouter, OpenAI,
Voyage, DeepInfra, vLLM, Ollama or any other `/v1/embeddings` endpoint. This mirrors how
`LLMSpec(kind="openai_compat")` already treats OpenRouter as "an OpenAI-compatible base
URL", keeps the no-silent-substitution invariant (the model id and base URL are identity),
and avoids a second vendor-specific client.

Default and documented model choices (job-file examples):

1. **`qwen/qwen3-embedding-8b` ($0.01)** as the recommended default. Multilingual (the
   ref_zivila source is Slovenian/English), open weights so a local
   `SentenceTransformerEncoder("Qwen/Qwen3-Embedding-8B")` can reproduce the same vectors
   later, three providers (no single point of failure), and already the most-used
   embedding model on OpenRouter. 4,096 dims makes the on-disk index 4x a 1,024-dim
   model (30k records x 4,096 x 4 B = ~0.5 GB in `vectors.f32`), which the pure-Python
   `_read_vectors` will load slowly; see "index storage" below.
2. **`voyageai/voyage-4-lite` ($0.02)** as the quality-per-dollar alternative:
   multilingual, 1,024 dims, 32k context, `dimensions` down to 256, and the index can be
   queried later with `voyage-4` or `voyage-4-large` without rebuilding (same space).
   Closed weights, single provider.
3. **`openai/text-embedding-3-small` ($0.02)** for English-only targets and users who
   already have the OpenAI key; `dimensions` verified.
4. `perplexity/pplx-embed-v1-0.6b` ($0.004) only as a "cheapest possible" option; too
   new and single-provider to be a default.

Pin `provider.order` / `provider.only` in the job file for reproducibility when a model
has several providers (Qwen3-8B via Nebius vs DeepInfra may not be bit-identical); record
the chosen provider list in the index identity.

## Implementation sketch for the next session

Files: `retrieval/dense.py` (new encoder), `config.py` (`RetrieverSpec` fields +
`build_encoder` dispatch), `ops.py` (`_extra_errors` must not demand
sentence-transformers for an API encoder), `docs/reference/job-file.md`,
`docs/reference/retrieval.md`, `examples/*/job_*.yaml` (one example), `CHANGELOG.md`,
`tests/test_dense.py` + a new `tests/test_dense_api.py`.

### `OpenAIEmbeddingsEncoder` (in `retrieval/dense.py`, httpx only, no extra needed)

```python
OpenAIEmbeddingsEncoder(
    base_url="https://openrouter.ai/api/v1", model="qwen/qwen3-embedding-8b",
    api_key=..., dimensions=None, query_input_type=None, doc_input_type=None,
    query_prefix="", doc_prefix="", normalize=True, batch_size=64, timeout=60.0,
    max_retries=5, provider=None, extra_headers=None, transport=None)
```

- `encode()` is synchronous (the protocol is sync; `DenseRetriever.search` already runs
  it in a thread). Use `httpx.Client`, not the async client, and a lock like the
  sentence-transformers encoder.
- `dimension`: from `dimensions` when set; otherwise probe once with a single short
  string on first access and cache it. `encoder_identity` reads `dimension` ("may load
  the model; it never encodes anything" currently holds; the probe is one tiny paid
  call, document it, or require `dimensions` for API encoders to keep identity free of
  network. Recommend requiring an explicit `dimensions` in the job file, which also makes
  the index refuse a provider that ignores the parameter).
- `settings` (identity): `{"adapter": "openai_embeddings", "adapter_version": 1,
  "base_url": redact_url(base_url), "dimensions", "normalize", "query_prefix",
  "doc_prefix", "query_input_type", "doc_input_type", "provider": provider or None}`.
  Never the key. Revision is unavailable; record `"revision": "unknown"` as the
  sentence-transformers encoder does for unpinned models.
- Request: `{"model", "input": batch, **{"dimensions"} if set, **{"input_type"} if set}`;
  `encoding_format` left default (`float`). Verify `len(data) == len(batch)` and sort by
  `data[i].index`. Normalise client-side when `normalize=True` (several providers return
  unnormalised vectors; `IndexFlatIP` assumes unit vectors).
- Errors: reuse `_error_message`, `_body_error`, `FATAL_BODY_CODES`, `RETRYABLE_STATUSES`
  from `llm/openai_compat.py` (move them to a small shared module or import them; they
  are private names today). Retry with the same backoff; raise `RetrieverError`-compatible
  exceptions so the matcher degrades to the remaining retrievers as CONTRACTS section 3
  requires. On 413 halve the batch and retry.
- Usage: `usage.prompt_tokens` and `usage.cost` are reported by OpenRouter; expose a
  counter on the encoder (`tokens_used`, `cost_credits`) so `xwalk index` can print what
  the build cost. Unknown counts stay "unknown", never zero.

### `RetrieverSpec` changes

- Add `encoder: Literal["sentence_transformers", "openai_embeddings"] = "sentence_transformers"`
  (backwards compatible: every existing job file keeps working).
- New dense fields, rejected unless `encoder: openai_embeddings`: `base_url`
  (required), `api_key_env` (required, same `CredentialMissingError` path as
  `LLMSpec`), `dimensions` (required for this encoder), `query_input_type`,
  `doc_input_type`, `batch_size`, `provider` (list of slugs). `api_key` present only to be
  rejected, as in `LLMSpec`.
- `device` and `revision` rejected for the API encoder; `required_extra` returns `None`
  for it.
- `build_encoder` dispatches on `encoder`.

### Tests (FakeLLM-style, `httpx.MockTransport`, no network)

- Request body shape incl. `dimensions`, `input_type` per side, prefixes per side.
- Response order restored from `index`; length mismatch and dimension mismatch refused.
- Client-side normalisation; `normalize=False` leaves vectors as returned.
- Retry on 429/5xx, fatal on 401/402/404, the 200-with-error-body shape, 413 halving.
- Identity: changing `dimensions`, `base_url`, an input type or `provider` changes
  `index_components`; changing the key or `batch_size` does not.
- `open()` of an index built with the API encoder, zero encode calls.
- Job validation: fields rejected for the wrong encoder, `api_key` rejected,
  missing env var surfaced, `_extra_errors` does not require sentence-transformers.
- One `integration`-marked smoke test gated on `XWALK_TEST_API_KEY` that embeds three
  strings with `qwen/qwen3-embedding-8b` and `dimensions`-checked `voyage-4-lite`,
  skipped by default.

### Index storage note

`_read_vectors` unpacks the whole file through `struct` into Python lists, then
`DenseRetriever.__init__` copies them again. At 4,096 dims x 30k records that is ~120M
Python floats. Not in scope for this task, but when numpy is present `np.fromfile(...)
.reshape(-1, dim)` would make high-dimensional API models practical; recommend a
follow-up, or defaulting the Qwen example to `dimensions: 1024` if OpenRouter honours it
for that model (unverified).

## Sources

- OpenRouter embeddings reference: https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings.md
- OpenRouter "Best Embedding Models in 2026": https://openrouter.ai/blog/insights/best-embedding-models-2026/
- Model list: https://openrouter.ai/api/v1/embeddings/models (2026-11-03)
- Voyage 4 dimensions and shared space: https://docs.voyageai.com/embeddings/
- Qwen3-Embedding query instruction: https://github.com/QwenLM/Qwen3-Embedding

# Supported platforms

xwalk's BM25 retriever is built on Tantivy. Ranking behaviour is part of a run's
identity — the retriever fingerprint feeds the run fingerprint — so xwalk does **not**
substitute a different BM25 engine when Tantivy is unavailable. A silent switch would
change ranking and quietly undermine reproducibility. Selecting an alternate is a
deliberate act by the user, **never automatic**.

## Supported

| Platform | Python |
|---|---|
| Linux x86-64 (manylinux) | 3.10, 3.11, 3.12 |
| Linux aarch64 (manylinux) | 3.10, 3.11, 3.12 |
| macOS arm64 | 3.10, 3.11, 3.12 |
| Windows x86-64 | 3.10, 3.11, 3.12 |

CI runs an install smoke test on every row: build a tiny index, search it, and assert a
known ranking. A row is supported only while that test passes.

## Unsupported platforms

If `pip install tantivy` fails, you have two explicit options.

**1. Bring your own retriever.** Implement the `Retriever` protocol against whatever
search backend you already run (Elasticsearch, OpenSearch, Postgres full-text, Vespa).
It is four members:

```python
class MyRetriever:
    name: str            # appears in RetrievalHit.retriever and in diagnostics
    fingerprint: str     # digest of index content and configuration
    default_limit: int   # this retriever's own retrieval depth

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]: ...
```

`TargetStore` stays separate, so the backend never needs your records in memory.

**2. Use dense retrieval only.** Install `xwalk[dense]` and configure a single dense
retriever. Expect different behaviour on exact-string matches — a lexical index and an
embedding index disagree most sharply on precisely the cases where a label matches
verbatim — which is why this is a decision you make rather than one xwalk makes for you.

Whichever you choose, record it: the retriever's `fingerprint` appears in the run
manifest, so results produced on different retrieval stacks are never silently mixed.

## Optional extras and what they weigh

| Extra | Pulls | Needed for |
|---|---|---|
| *(base)* | pydantic, jinja2, httpx, pyyaml, tantivy | BM25 + any OpenAI-compatible endpoint |
| `dense` | sentence-transformers, torch, faiss-cpu | `DenseRetriever` with a local encoder |
| `ontology` | rdflib | `owl_source`; `obo_source` needs nothing |
| `sql` | SQLAlchemy | `sql_source` |
| `litellm` | litellm | `LiteLLMClient` |

Matching two CSVs with an API-hosted model installs none of the heavy stack. A missing
extra raises `MissingExtra` naming the extra and the install command, not a bare
`ImportError` from a library you have never heard of.

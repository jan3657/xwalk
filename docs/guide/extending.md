# Extending xwalk

Every pluggable part of xwalk is a `typing.Protocol`, not a base class. You implement the
members and pass your object in; there is nothing to inherit from and nothing to
register. All four protocols are `@runtime_checkable`, so `isinstance(x, Retriever)`
works — though it only checks that the members exist, not that they behave.

## A custom record source

The easiest one. A record source is `Iterable[Record]` and nothing else:

```python
from collections.abc import Iterator

from xwalk.records import Record


def warehouse_source(client, table: str) -> Iterator[Record]:
    for row in client.stream(f"SELECT * FROM {table}"):
        yield Record(id=str(row.pop("uid")), fields=dict(row))
```

Pass it anywhere a source is expected. Two rules the built-in loaders follow and yours
should too: pop the id out of `fields` rather than leaving it in both places, and refuse
a blank id rather than inventing one — `Record` will raise if you try.

Prefer a generator. Sources are consumed lazily, which is what lets xwalk stream a
collection that does not fit in memory. If you return a list, `MemoryStore.from_source`
and `BM25Retriever.build` will both hold it, and you have given up that property.

## A custom retriever

Four members:

```python
from collections.abc import Sequence

from xwalk.records import RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest


class ElasticRetriever:
    def __init__(self, client, index: str, *, limit: int = 20):
        self._client, self._index, self._limit = client, index, limit

    @property
    def name(self) -> str:
        return "elastic"

    @property
    def fingerprint(self) -> str:
        # Must change when the index content or the query configuration changes.
        return f"elastic:{self._index}:{self._client.index_version(self._index)}"

    @property
    def default_limit(self) -> int:
        return self._limit

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        try:
            hits = await self._client.search(self._index, request.text, size=request.limit)
        except ConnectionError as exc:
            raise RetrieverError(f"elastic unavailable: {exc}") from exc
        return [
            RetrievalHit(record_id=h["_id"], retriever=self.name, raw_score=h["_score"], rank=i)
            for i, h in enumerate(hits, start=1)
        ]
```

Things that will bite you if you get them wrong:

**Ranks are 1-based and contiguous.** `RetrievalHit` raises on `rank < 1`, and fusion
scores are `1 / (k + rank)` — an off-by-one silently reweights everything.

**Raise `RetrieverError`, do not return an empty list, when the backend is down.** The
matcher degrades to the surviving retrievers on `RetrieverError` and records the
degradation. An empty list is indistinguishable from "no matching documents", which is
evidence about your data rather than about your infrastructure.

**`fingerprint` must change when ranking would change.** It feeds the run fingerprint,
which is what decides whether previously computed results still apply. A fingerprint that
does not move when your index is rebuilt means resume will reuse results computed against
different data.

**`search` must be genuinely async.** If your client is synchronous, wrap it in
`asyncio.to_thread`, as the built-in BM25 and dense retrievers do. A blocking call here
serialises every retriever and every concurrent record.

Also: `SearchRequest.source_record` carries the whole source record, so a field-aware
backend can use it. Ignoring it is correct and normal.

## A custom target store

Three members. Worth implementing when your target collection does not fit in memory:

```python
from collections.abc import Sequence

from xwalk.records import Record


class RedisStore:
    @property
    def fingerprint(self) -> str:
        return self._snapshot_digest

    def get(self, record_id: str) -> Record:
        raw = self._client.get(record_id)
        if raw is None:
            raise KeyError(record_id)
        return Record(id=record_id, fields=json.loads(raw))

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        # Request order preserved; unknown ids skipped, not raised.
        found = self._client.mget(record_ids)
        return [
            Record(id=rid, fields=json.loads(raw))
            for rid, raw in zip(record_ids, found)
            if raw is not None
        ]
```

`get` raises `KeyError` on a miss; `get_many` silently skips. That asymmetry is
deliberate — `get_many` is called with ids that came from a retriever, and a stale index
may name a record that has since been deleted. Fusion drops those candidates.

## A custom LLM client

Four members. Implement this to reach a provider neither adapter covers, or to add
behaviour — rate limiting, logging, a router — around one that does:

```python
from xwalk.llm.base import LLMCapabilities, LLMRequest, LLMResponse, LLMRetryableError
from xwalk.records import Usage


class BedrockClient:
    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        # Defaults are all-false. Only claim what you have verified.
        return LLMCapabilities(json_schema=True, usage_reporting=True)

    @property
    def fingerprint(self) -> str:
        # Provider identity, model, and generation params. Never the credential.
        return hash_value({"adapter": "bedrock", "model": self._model, "temperature": self._temp})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...
        return LLMResponse(text=text, usage=Usage(p, c, calls=1), model=self._model)
```

**Capabilities decide what you ask for, never whether you trust the answer.** Declaring
`json_schema=True` means the adapter may send a schema; xwalk still parses and validates
whatever comes back. Claiming a capability the provider does not really honour degrades
output quality; it cannot corrupt a result.

**Classify errors correctly.** Raise `LLMRetryableError` for 429s, 5xx, and timeouts —
the matcher retries the attempt. Raise `LLMFatalError` for auth failures and unknown
models — it aborts the record immediately rather than burning the retry budget on
something that will never succeed.

**Keep credentials out of `fingerprint`.** Rotating a key must not invalidate a resumable
run, and the fingerprint reaches `manifest.json` on disk.

### Wrapping instead of replacing

Composition works, because identity is delegated. `CachingLLM` is exactly this pattern:

```python
from xwalk.llm.cache import CachingLLM

llm = CachingLLM(OpenAICompatClient(...), ledger)
```

It forwards `model`, `capabilities` and `fingerprint` to the wrapped client, so wrapping
does not change the run fingerprint. Your own wrapper — a rate limiter, a logger — should
do the same, unless it genuinely changes what an answer means.

## Testing without a provider

`FakeLLM` is why xwalk's own test suite runs in seconds with no credentials, and it is
available to you for the same purpose.

Script mode returns responses in order and raises `AssertionError` if you run off the
end — an unexpected extra LLM call is treated as a bug rather than quietly absorbed:

```python
import json

from xwalk.llm.fake import FakeLLM

llm = FakeLLM([
    json.dumps({"chosen_key": "C01", "confidence_score": 0.9, "explanation": "exact"}),
    json.dumps({"confidence_score": 0.95, "explanation": ""}),
])
```

Handler mode inspects each request, which is what you want for anything that loops:

```python
def handler(request):
    if "## Candidates" in request.user:
        return json.dumps({"chosen_key": "C01", "confidence_score": 0.9, "explanation": ""})
    return json.dumps({"confidence_score": 0.95, "explanation": ""})

llm = FakeLLM(handler=handler)
```

Both modes accept a `BaseException` in place of a string, so you can script a 429 or a
malformed response and check that your pipeline degrades the way you expect:

```python
from xwalk.llm.base import LLMRetryableError

llm = FakeLLM([LLMRetryableError("429"), "{...}"])
```

Afterwards, `llm.requests` holds every `LLMRequest` seen and `llm.total_usage` the
accumulated tokens.

For retrievers, a scripted stand-in is a dozen lines — see `ScriptedRetriever` in
`tests/test_matcher.py`.

## Custom prompt skeletons

`PromptSet.from_slots(slots, base_dir=...)` loads `select.j2`, `score.j2`, `verify.j2`
and `rewrite.j2` from a directory of your own. Do this only with a reason: the skeletons
own the machine-readable contract, and `validate_contract` exists because breaking it is
easy and the symptom is a run where every record resolves to nothing.

Run `validate_contract` on the result before you use it:

```python
from xwalk.prompts.contract import PromptSet, validate_contract

prompts = PromptSet.from_slots(slots, base_dir=Path("my_prompts/"))
validate_contract(prompts)   # raises ContractError if the contract did not survive
```

One sharp edge: `{% include %}` and `{% extends %}` inside a custom skeleton resolve
against xwalk's packaged template directory, not against your own. Keep custom skeletons
self-contained.

## What is deliberately not extensible

There is no plugin registry, no entry-point discovery, and no configuration key that
loads an arbitrary class by dotted path. A job file cannot name your custom retriever;
you construct it in Python and pass it to `Matcher`.

That is a choice, not an omission. A job file that can load arbitrary code is a job file
you cannot safely accept from someone else, and the run fingerprint could no longer
account for what actually ran. The SDK is the extension point; the CLI is a shell over
the subset that is expressible as data.

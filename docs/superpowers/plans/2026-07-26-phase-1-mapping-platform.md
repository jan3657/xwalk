# xwalk Phase 1 — The Mapping Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a library where a user points at two record collections (CSV/JSONL to start), supplies four Jinja2 templates and a prompt slots file, picks an LLM, and gets back a resumable, reviewable mapping table with honest per-row status and reason.

**Architecture:** A frozen-dataclass core (`Record`, `RetrievalHit`, `Candidate`, `Attempt`, `MatchResult`) sits under four independent protocols — `RecordSource` (= `Iterable[Record]`), `Retriever`, `TargetStore`, `LLMClient`. Domain knowledge lives entirely in user-supplied Jinja2 templates and a YAML slots file; nothing in the library knows any dataset exists. `Matcher.match()` runs an async attempt loop (retrieve → fuse → key → select → gate → verify → retry) and returns one `MatchResult`. `batch.py` drives a `RecordSource` through the matcher with bounded concurrency, committing every result to a SQLite WAL ledger through a single writer coroutine so runs resume exactly.

**Tech Stack:** Python 3.10+, pydantic v2 (config only), jinja2, httpx, tantivy (BM25), pyyaml, stdlib sqlite3/asyncio. Tests: pytest + pytest-asyncio. Lint/type: ruff, mypy --strict. No torch, no sentence-transformers, no faiss, no rdflib in Phase 1.

**Spec:** `docs/superpowers/specs/2026-07-26-xwalk-generalized-matching-library-design.md`. Read the spec's "Phase 1 contract" and "Explicit non-goals" sections before starting. When this plan and the spec disagree, the spec wins — stop and report the discrepancy.

## Global Constraints

Every task's requirements implicitly include this section.

- **The package is `xwalk`.** Never `crosswalk` — that name is taken on PyPI. Distribution name, import name, CLI name, and config keys are all `xwalk`.
- **`requires-python = ">=3.10"`** — Tantivy's floor. Do not use 3.11+ syntax (`Self`, `except*`, `tomllib` at import time).
- **`src/` layout.** All library code under `src/xwalk/`. All tests under `tests/`.
- **Base install must not pull `torch`, `sentence-transformers`, `faiss`, `rdflib`, or `sqlalchemy`.** If a task tempts you to import one, it belongs in Phase 3 behind an extra.
- **All core data types are `@dataclass(frozen=True)`.** Mutation happens by constructing a new value.
- **The core is `async`; every public entry point also gets a `_sync` facade** (`Matcher.match_sync`, `run_batch_sync`) implemented with `asyncio.run`.
- **No network in tests.** The only exception is a single test marked `@pytest.mark.integration`, which must `pytest.skip` when the relevant API-key env var is absent.
- **A provider's declared capability is never trusted.** `LLMCapabilities` decides what to *ask* for; output is parsed and validated regardless of what was claimed.
- **A malformed model answer must never resolve to a real target record.** This is the single hardest invariant in the library. Any resolution path that is not an exact lookup against keys issued for *that attempt* yields `UNRESOLVED_OUTPUT`.
- **`mypy --strict` must pass** on `src/xwalk/` at the end of every task. `ruff check` and `ruff format --check` likewise.
- **Import ABCs from `collections.abc`, not `typing`.** `Mapping`, `Sequence`, `Iterable`, `Iterator`, `Callable`, and `Set` all live there; `typing` keeps only `Any`, `Protocol`, `Literal`, `TYPE_CHECKING`, and `runtime_checkable`. Ruff's `UP035` fails the lint gate otherwise, and this plan's `select = ["E", "F", "I", "UP", "B", "SIM"]` turns that into a hard error at the end of *every* task.
- **Put `TMPDIR` on local disk or tmpfs before running the suite.** Several tests build real Tantivy indexes under `tmp_path`, which follows `TMPDIR`. On a network-mounted `TMPDIR` (NFS, Ceph, Lustre) the BM25 tests take minutes; on tmpfs they take under a second. `export TMPDIR=/dev/shm/xwalk-tmp && mkdir -p "$TMPDIR"` is enough. This is environmental, not a code problem — but without it you will misread slow tests as a bug.
- **Commit at the end of every task**, with the test and implementation in the same commit.

---

## File Structure

Files created in Phase 1. Line estimates are guidance, not targets.

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, extras, ruff/mypy/pytest config |
| `src/xwalk/__init__.py` | Public surface re-exports, `__version__` |
| `src/xwalk/records.py` | `Record`, `RetrievalHit`, `Candidate`, `Usage`, **`RetryProposal`**, `Attempt`, `MatchResult`, `MatchStatus`, `DecisionReason` |
| `src/xwalk/fingerprint.py` | `canonical_json`, `hash_value`, `hash_record`, `run_fingerprint`, `result_key` |
| `src/xwalk/templates.py` | `TemplateSet` — compiles and renders the four Jinja2 templates |
| `src/xwalk/sources/base.py` | `RecordSource` protocol |
| `src/xwalk/sources/tabular.py` | `csv_source`, `jsonl_source` |
| `src/xwalk/stores/base.py` | `TargetStore` protocol |
| `src/xwalk/stores/memory.py` | `MemoryStore` |
| `src/xwalk/retrieval/base.py` | `SearchRequest`, `Retriever` protocol, `RetrieverError` |
| `src/xwalk/llm/cache.py` | `CachingLLM` — ledger-backed response cache |
| `src/xwalk/retrieval/bm25.py` | `BM25Retriever` — Tantivy index build + search |
| `src/xwalk/retrieval/fusion.py` | `reciprocal_rank_fusion` |
| `src/xwalk/llm/base.py` | `LLMClient` protocol, `LLMCapabilities`, `LLMRequest`, `LLMResponse`, error types |
| `src/xwalk/llm/parsing.py` | thinking-block stripping, JSON extraction, JSON repair |
| `src/xwalk/llm/fake.py` | `FakeLLM` — scripted offline client |
| `src/xwalk/llm/openai_compat.py` | `OpenAICompatClient` — structured output, backoff, capability profiles |
| `src/xwalk/stages/keying.py` | `assign_keys`, `resolve_key`, `Resolution` |
| `src/xwalk/stages/proposals.py` | `route_proposals`, `normalise_query` (`RetryProposal` itself lives in `records.py`, so stages and serde share one definition) |
| `src/xwalk/stages/select.py` | `Selector`, `SelectorPolicy`, `SelectionOutcome` |
| `src/xwalk/stages/gate.py` | `Scorer`, `Verifier`, `ScoreOutcome`, `VerifierVerdict` |
| `src/xwalk/stages/rewrite.py` | `QueryRewriter` |
| `src/xwalk/policy.py` | `MatchPolicy`, `derive_status`, `should_verify`, `should_audit` |
| `src/xwalk/serde.py` | `result_to_dict`, `result_from_dict` |
| `src/xwalk/matcher.py` | `Matcher` — the attempt loop |
| `src/xwalk/ledger.py` | SQLite WAL ledger: results, LLM cache, review overlay |
| `src/xwalk/batch.py` | `run_batch`, `BatchReport`, exports |
| `src/xwalk/review.py` | `export_review`, `apply_review` |
| `src/xwalk/prompts/base/{select,score,verify,rewrite}.j2` | Contract-safe skeletons |
| `src/xwalk/prompts/contract.py` | Slots model + skeleton contract validation |
| `tests/__init__.py` | Makes `tests` a package so modules can share fixtures |
| `tests/fixtures/targets_tiny.csv` | 5-row target collection |
| `tests/fixtures/sources_tiny.csv` | 4-row source collection |

---

## Task 1: Project scaffolding and core types

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: `src/xwalk/__init__.py`, `src/xwalk/py.typed`, `src/xwalk/records.py`
- Test: `tests/test_records.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Record(id: str, fields: Mapping[str, Any])`; `RetrievalHit(record_id: str, retriever: str, raw_score: float | None, rank: int)`; `Candidate(record: Record, fused_score: float, evidence: tuple[RetrievalHit, ...])` with property `id -> str`; `Usage(prompt_tokens: int, completion_tokens: int, calls: int)` with `Usage.zero()`, `total_tokens` property, and `__add__`. Every later task imports from `xwalk.records`.

- [x] **Step 1: Create the repository skeleton**

```bash
mkdir -p src/xwalk tests/fixtures docs/superpowers/plans docs/superpowers/specs
touch src/xwalk/py.typed
```

- [x] **Step 2: Write `pyproject.toml`**

`rank` is 1-based throughout the library. Note `asyncio_mode = "auto"` — tests do not need `@pytest.mark.asyncio`.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "xwalk"
version = "0.1.0.dev0"
description = "LLM-RAG record matching between two collections"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "pydantic>=2.0",
    "jinja2>=3.1",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "tantivy>=0.22",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.6",
    "mypy>=1.11",
    "types-PyYAML",
]

[tool.hatch.build.targets.wheel]
packages = ["src/xwalk"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "integration: hits a real LLM provider; skipped without an API key",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
strict = true
files = ["src/xwalk"]
```

- [x] **Step 3: Write `.gitignore`**

The paper repo grew to 11 GB of history over 33 GB of data. That must not happen here — no run artefacts, no indexes, no data ever enters git.

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.env
# never commit run artefacts, indexes, or data
run/
runs/
*.sqlite
*.sqlite-wal
*.sqlite-shm
index/
*.index/
data/
```

- [x] **Step 4: Write the failing test**

Create `tests/test_records.py`:

```python
import pytest

from xwalk.records import Candidate, Record, RetrievalHit, Usage


def test_record_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        Record(id="", fields={"name": "x"})


def test_record_rejects_non_string_id():
    with pytest.raises(TypeError, match="str"):
        Record(id=42, fields={})  # type: ignore[arg-type]


def test_record_is_frozen():
    record = Record(id="A1", fields={"name": "glucose"})
    with pytest.raises(AttributeError):
        record.id = "A2"  # type: ignore[misc]


def test_retrieval_hit_rejects_rank_below_one():
    with pytest.raises(ValueError, match="1-based"):
        RetrievalHit(record_id="A1", retriever="bm25", raw_score=1.0, rank=0)


def test_retrieval_hit_allows_missing_score():
    hit = RetrievalHit(record_id="A1", retriever="external", raw_score=None, rank=1)
    assert hit.raw_score is None


def test_candidate_id_delegates_to_record():
    record = Record(id="CHEBI:17234", fields={"label": "glucose"})
    hit = RetrievalHit(record_id="CHEBI:17234", retriever="bm25", raw_score=3.2, rank=1)
    candidate = Candidate(record=record, fused_score=0.016, evidence=(hit,))
    assert candidate.id == "CHEBI:17234"


def test_candidate_rejects_evidence_for_a_different_record():
    record = Record(id="A1", fields={})
    hit = RetrievalHit(record_id="B2", retriever="bm25", raw_score=1.0, rank=1)
    with pytest.raises(ValueError, match="evidence"):
        Candidate(record=record, fused_score=0.5, evidence=(hit,))


def test_usage_zero_is_all_zeroes():
    usage = Usage.zero()
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (0, 0, 0)
    assert usage.total_tokens == 0


def test_usage_adds_componentwise():
    total = Usage(prompt_tokens=10, completion_tokens=5, calls=1) + Usage(
        prompt_tokens=3, completion_tokens=2, calls=1
    )
    assert total == Usage(prompt_tokens=13, completion_tokens=7, calls=2)


def test_usage_sum_starts_from_zero():
    parts = [Usage(prompt_tokens=1, completion_tokens=1, calls=1) for _ in range(3)]
    assert sum(parts, Usage.zero()) == Usage(prompt_tokens=3, completion_tokens=3, calls=3)
```

- [x] **Step 5: Run the test to verify it fails**

Run: `python -m pytest tests/test_records.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk'`.

- [x] **Step 6: Write `src/xwalk/records.py`**

Only the four types the tests exercise. `Attempt` and `MatchResult` arrive in Task 14, once every field they reference exists.

```python
"""Core value types. Everything in xwalk is built from these."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Record:
    """One row from either collection. `fields` is whatever the source produced."""

    id: str
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError(f"Record.id must be str, got {type(self.id).__name__}")
        if not self.id.strip():
            raise ValueError("Record.id must be non-empty")
        if not isinstance(self.fields, Mapping):
            raise TypeError("Record.fields must be a Mapping")


@dataclass(frozen=True)
class RetrievalHit:
    """One retriever's opinion about one target record. `rank` is 1-based."""

    record_id: str
    retriever: str
    raw_score: float | None
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"RetrievalHit.rank is 1-based, got {self.rank}")


@dataclass(frozen=True)
class Candidate:
    """A fused candidate. `evidence` records every retriever that surfaced it."""

    record: Record
    fused_score: float
    evidence: tuple[RetrievalHit, ...]

    def __post_init__(self) -> None:
        for hit in self.evidence:
            if hit.record_id != self.record.id:
                raise ValueError(
                    f"Candidate evidence names {hit.record_id!r} "
                    f"but the record is {self.record.id!r}"
                )

    @property
    def id(self) -> str:
        return self.record.id


@dataclass(frozen=True)
class Usage:
    """Token and call accounting. Additive so attempts can be summed into a result."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @classmethod
    def zero(cls) -> Usage:
        return cls()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Any) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: Any) -> Usage:
        if other == 0:
            return self
        return self.__add__(other)
```

- [x] **Step 7: Write `src/xwalk/__init__.py`**

```python
"""xwalk — LLM-RAG record matching between two collections."""

# Defined before any submodule import and kept that way: `batch.py` (Task 18) does
# `from xwalk import __version__`, which fails if a submodule import above this line
# ever pulls `xwalk.batch` back in while __version__ is still unbound.
__version__ = "0.1.0.dev0"

from xwalk.records import Candidate, Record, RetrievalHit, Usage

__all__ = ["Candidate", "Record", "RetrievalHit", "Usage", "__version__"]
```

- [x] **Step 8: Install and run the tests**

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_records.py -v
```

Expected: 10 passed. If `tantivy` fails to install on this platform, stop and report it — the spec fixes a support matrix (manylinux x86-64/aarch64, macOS arm64, Windows x86-64) and a platform outside it is a decision for the user, never a silent fallback.

- [x] **Step 9: Lint and type-check**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
```

Expected: clean. If `ruff format --check` complains, run `python -m ruff format src tests` and re-run.

- [x] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore README.md src/xwalk tests/test_records.py
git commit -m "feat: project scaffolding and core record types"
```

---

## Task 2: Fingerprinting and result keys

**Files:**
- Create: `src/xwalk/fingerprint.py`
- Test: `tests/test_fingerprint.py`

**Interfaces:**
- Consumes: `Record` from Task 1.
- Produces: `canonical_json(value: Any) -> str`; `hash_value(value: Any) -> str` (16 lowercase hex chars); `hash_record(record: Record) -> str`; `run_fingerprint(**components: Any) -> str`; `result_key(run_fp: str, source_id: str, source_hash: str) -> str`. Task 15 (ledger) and Task 18 (batch) key everything on these.

**Why this matters:** resume keyed on `source_id` alone is wrong. The same ID can carry changed field values, or have been processed under a different prompt, target snapshot, retriever set or model. Change any of those and prior results must *stop* matching, so you get a fresh run rather than a silently mixed one.

- [x] **Step 1: Write the failing test**

Create `tests/test_fingerprint.py`:

```python
from datetime import date

import pytest

from xwalk.fingerprint import canonical_json, hash_record, hash_value, result_key, run_fingerprint
from xwalk.records import Record


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_is_insertion_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_normalises_sets_deterministically():
    assert canonical_json({"tags": {"b", "a"}}) == '{"tags":["a","b"]}'


def test_canonical_json_renders_tuples_as_lists():
    assert canonical_json(("a", "b")) == '["a","b"]'


def test_canonical_json_handles_dates():
    assert canonical_json(date(2026, 7, 26)) == '"2026-07-26"'


def test_canonical_json_rejects_unserialisable_values():
    with pytest.raises(TypeError, match="not serialisable"):
        canonical_json({"f": object()})


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"score": float("nan")})


def test_hash_value_is_sixteen_hex_chars():
    digest = hash_value({"a": 1})
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_value_is_stable_across_calls():
    assert hash_value({"a": [1, 2]}) == hash_value({"a": [1, 2]})


def test_hash_record_changes_when_a_field_value_changes():
    before = hash_record(Record(id="s1", fields={"name": "glucose"}))
    after = hash_record(Record(id="s1", fields={"name": "fructose"}))
    assert before != after


def test_hash_record_ignores_field_insertion_order():
    a = hash_record(Record(id="s1", fields={"name": "x", "city": "y"}))
    b = hash_record(Record(id="s1", fields={"city": "y", "name": "x"}))
    assert a == b


def test_run_fingerprint_changes_when_any_component_changes():
    base = dict(model="gpt-4o", target="t1", policy={"accept_at": 0.6})
    assert run_fingerprint(**base) != run_fingerprint(**{**base, "model": "gpt-4o-mini"})
    assert run_fingerprint(**base) != run_fingerprint(**{**base, "policy": {"accept_at": 0.7}})


def test_run_fingerprint_ignores_keyword_order():
    assert run_fingerprint(a=1, b=2) == run_fingerprint(b=2, a=1)


def test_result_key_is_not_vulnerable_to_boundary_collisions():
    """('a', 'bc') and ('ab', 'c') must not collide, which naive concatenation allows."""
    assert result_key("fp", "a", "bc") != result_key("fp", "ab", "c")


def test_result_key_changes_with_run_fingerprint():
    assert result_key("fp1", "s1", "h1") != result_key("fp2", "s1", "h1")


def test_result_key_changes_with_source_hash():
    assert result_key("fp1", "s1", "h1") != result_key("fp1", "s1", "h2")
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fingerprint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.fingerprint'`.

- [x] **Step 3: Write `src/xwalk/fingerprint.py`**

```python
"""Deterministic hashing of records and run configuration.

Everything that decides whether a prior result is still valid flows through here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xwalk.records import Record

_DIGEST_CHARS = 16


def _normalise(value: Any) -> Any:
    """Reduce a value to JSON-safe primitives, deterministically."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical_json(v) for v in value) and [
            _normalise(v) for v in sorted(value, key=lambda x: canonical_json(x))
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalise(v) for v in value]
    raise TypeError(f"value of type {type(value).__name__} is not serialisable for hashing")


def canonical_json(value: Any) -> str:
    """A byte-stable JSON rendering: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_value(value: Any) -> str:
    """A 16-hex-char digest. Short enough to read in a filename, wide enough to be safe."""
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_DIGEST_CHARS]


def hash_record(record: Record) -> str:
    """Digest of a record's identity *and* content."""
    return hash_value({"id": record.id, "fields": record.fields})


def run_fingerprint(**components: Any) -> str:
    """Digest of everything that would invalidate prior results.

    Callers pass the target snapshot fingerprint, retriever fingerprints, prompt and
    slots hashes, model and provider identity, generation parameters, the match policy,
    and the library version.
    """
    return hash_value(components)


def result_key(run_fp: str, source_id: str, source_hash: str) -> str:
    """Stable identity of one source record under one run configuration.

    The parts go in as a list, never concatenated: ``"a" + "bc"`` and ``"ab" + "c"``
    produce the same string, and a resume key that can collide is worse than none.
    """
    return hash_value([run_fp, source_id, source_hash])
```

- [x] **Step 4: Simplify the set branch**

The `_normalise` set branch above is deliberately over-complicated so you notice it. Replace it with:

```python
    if isinstance(value, (set, frozenset)):
        return sorted((canonical_json(v) for v in value))
```

Sets have no order, so hashing their *rendered elements* sorted is both deterministic and honest. Re-run the test — `test_canonical_json_normalises_sets_deterministically` expects `'{"tags":["a","b"]}'`, and rendered strings `'"a"'`/`'"b"'` would produce `["\"a\"","\"b\""]`. Fix the implementation to sort the *normalised* elements by their canonical rendering instead:

```python
    if isinstance(value, (set, frozenset)):
        normalised = [_normalise(v) for v in value]
        return sorted(normalised, key=lambda v: json.dumps(v, sort_keys=True, default=str))
```

- [x] **Step 5: Run the tests**

Run: `python -m pytest tests/test_fingerprint.py -v`
Expected: 16 passed.

- [x] **Step 6: Lint and type-check**

Run: `python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: clean.

- [x] **Step 7: Commit**

```bash
git add src/xwalk/fingerprint.py tests/test_fingerprint.py
git commit -m "feat: deterministic fingerprinting and collision-safe result keys"
```

---

## Task 3: The four templates

**Files:**
- Create: `src/xwalk/templates.py`
- Test: `tests/test_templates.py`

**Interfaces:**
- Consumes: `Record` from Task 1, `hash_value` from Task 2.
- Produces: `TemplateSet(query: str, context: str, doc: str, candidate: str)` — a frozen dataclass holding *template source strings*, with methods `render_query(record) -> str`, `render_context(record) -> str`, `render_doc(record) -> str`, `render_candidate(record) -> str`, and property `fingerprint -> str`. Also `TemplateError`. Tasks 5, 6, 11, 12 and 17 all render through this.

**Design note:** templates receive the record's fields as top-level variables plus `id`, so `"{{ mention }}"` works directly. Missing fields render as the empty string, *not* `StrictUndefined` — real source rows are sparse, and a missing optional column must not abort a 100k-row run. Template *syntax* errors, by contrast, raise at `TemplateSet` construction so a typo surfaces before any LLM call is billed.

- [x] **Step 1: Write the failing test**

Create `tests/test_templates.py`:

```python
import pytest

from xwalk.records import Record
from xwalk.templates import TemplateError, TemplateSet

TINY = dict(
    query="{{ mention }}",
    context=(
        "{% if context_left %}{{ context_left }} [{{ mention }}] {{ context_right }}{% endif %}"
    ),
    doc="{{ label }} {{ synonyms | join(' ') }}",
    candidate="ID: {{ id }}\nLabel: {{ label }}",
)


def test_renders_a_single_field():
    ts = TemplateSet(**TINY)
    assert ts.render_query(Record(id="s1", fields={"mention": "glucose"})) == "glucose"


def test_missing_field_renders_empty_not_an_error():
    ts = TemplateSet(**TINY)
    assert ts.render_context(Record(id="s1", fields={"mention": "glucose"})) == ""


def test_record_id_is_available_as_id():
    ts = TemplateSet(**TINY)
    out = ts.render_candidate(Record(id="CHEBI:17234", fields={"label": "glucose"}))
    assert out == "ID: CHEBI:17234\nLabel: glucose"


def test_list_fields_work_with_filters():
    ts = TemplateSet(**TINY)
    record = Record(id="t1", fields={"label": "glucose", "synonyms": ["dextrose", "grape sugar"]})
    assert ts.render_doc(record) == "glucose dextrose grape sugar"


def test_whitespace_is_collapsed_and_trimmed():
    ts = TemplateSet(query="  {{ a }}   {{ b }}  ", context="", doc="", candidate="")
    assert ts.render_query(Record(id="s1", fields={"a": "x", "b": "y"})) == "x y"


def test_blank_lines_from_skipped_conditionals_are_removed():
    ts = TemplateSet(
        query="",
        context="",
        doc="",
        candidate="ID: {{ id }}\n{% if defn %}Def: {{ defn }}\n{% endif %}Label: {{ label }}",
    )
    out = ts.render_candidate(Record(id="t1", fields={"label": "glucose"}))
    assert out == "ID: t1\nLabel: glucose"


def test_syntax_error_raises_at_construction_not_at_render():
    with pytest.raises(TemplateError, match="query"):
        TemplateSet(query="{{ unclosed ", context="", doc="", candidate="")


def test_fingerprint_changes_when_any_template_changes():
    a = TemplateSet(**TINY)
    b = TemplateSet(**{**TINY, "doc": "{{ label }}"})
    assert a.fingerprint != b.fingerprint


def test_fingerprint_is_stable_for_identical_templates():
    assert TemplateSet(**TINY).fingerprint == TemplateSet(**TINY).fingerprint


def test_id_field_in_fields_does_not_shadow_record_id():
    ts = TemplateSet(query="{{ id }}", context="", doc="", candidate="")
    record = Record(id="real", fields={"id": "spoofed"})
    assert ts.render_query(record) == "real"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.templates'`.

- [x] **Step 3: Write `src/xwalk/templates.py`**

```python
"""The four templates that carry the entire domain mapping.

| Template    | Input         | Produces                        |
|-------------|---------------|---------------------------------|
| `query`     | source record | retrieval query string          |
| `context`   | source record | context block shown to the LLM  |
| `doc`       | target record | indexed text                    |
| `candidate` | target record | one entry in the candidate list |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, Template, Undefined
from jinja2 import TemplateSyntaxError as JinjaSyntaxError

from xwalk.fingerprint import hash_value
from xwalk.records import Record


class TemplateError(Exception):
    """A template failed to compile or to render."""


class _EmptyUndefined(Undefined):
    """Missing fields render as an empty string.

    Source collections are sparse in practice. Aborting a 100k-row run because one row
    lacks an optional column is the wrong trade; a silently-empty slot is visible in the
    rendered output and in the attempt trace.
    """

    def __str__(self) -> str:
        return ""


_BLANK_LINES = re.compile(r"\n\s*\n+")
_SPACES = re.compile(r"[ \t]+")


def _tidy(text: str) -> str:
    """Collapse the whitespace that conditionals leave behind."""
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return _BLANK_LINES.sub("\n", joined).strip()


@dataclass(frozen=True)
class TemplateSet:
    """Holds template *source*; compiles on construction so typos fail fast."""

    query: str
    context: str
    doc: str
    candidate: str
    _compiled: dict[str, Template] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        env = Environment(undefined=_EmptyUndefined, keep_trailing_newline=False)
        for name in ("query", "context", "doc", "candidate"):
            source = getattr(self, name)
            try:
                self._compiled[name] = env.from_string(source)
            except JinjaSyntaxError as exc:
                raise TemplateError(f"{name} template failed to compile: {exc}") from exc

    def _render(self, name: str, record: Record) -> str:
        variables: dict[str, Any] = dict(record.fields)
        variables["id"] = record.id  # record identity always wins over a field named "id"
        try:
            return _tidy(self._compiled[name].render(**variables))
        except Exception as exc:  # noqa: BLE001 - any Jinja runtime error is a template error
            raise TemplateError(f"{name} template failed on record {record.id!r}: {exc}") from exc

    def render_query(self, record: Record) -> str:
        return self._render("query", record)

    def render_context(self, record: Record) -> str:
        return self._render("context", record)

    def render_doc(self, record: Record) -> str:
        return self._render("doc", record)

    def render_candidate(self, record: Record) -> str:
        return self._render("candidate", record)

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "query": self.query,
                "context": self.context,
                "doc": self.doc,
                "candidate": self.candidate,
            }
        )
```

- [x] **Step 4: Run the tests**

Run: `python -m pytest tests/test_templates.py -v`
Expected: 10 passed.

- [x] **Step 5: Lint and type-check**

Run: `python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: clean. `_compiled` as a mutable default on a frozen dataclass is intentional and `compare=False` keeps equality on the source strings; if mypy objects to assigning into it from `__post_init__`, that is fine — the dict itself is mutable, only the field binding is frozen.

- [x] **Step 6: Commit**

```bash
git add src/xwalk/templates.py tests/test_templates.py
git commit -m "feat: four-template rendering with fail-fast compilation"
```

---

## Task 4: Record sources and the in-memory target store

**Files:**
- Create: `src/xwalk/sources/__init__.py`, `src/xwalk/sources/base.py`, `src/xwalk/sources/tabular.py`
- Create: `src/xwalk/stores/__init__.py`, `src/xwalk/stores/base.py`, `src/xwalk/stores/memory.py`
- Create: `tests/fixtures/targets_tiny.csv`, `tests/fixtures/sources_tiny.csv`
- Test: `tests/test_sources.py`, `tests/test_stores.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `Record` (Task 1), `hash_record`/`hash_value` (Task 2).
- Produces: `RecordSource` protocol (`Iterable[Record]`); `csv_source(path, *, id_column, delimiter=",", multivalue_columns=(), multivalue_sep="|") -> Iterator[Record]`; `jsonl_source(path, *, id_field) -> Iterator[Record]`; `TargetStore` protocol with `fingerprint: str`, `get(record_id) -> Record`, `get_many(record_ids) -> Sequence[Record]`, `__len__`, `__iter__`; `MemoryStore.from_source(source) -> MemoryStore`. Tasks 5, 6, 12, 16 consume these.

**Design note:** `RecordSource` is the *entire* extension contract for input data. It replaces the paper repo's `HOOK_REGISTRY` — adding a dataset is writing a generator in your own code, not editing library code. Keeping `TargetStore` separate from `Retriever` is what makes "plug in Elasticsearch for scale" real; otherwise the retriever still has to hold every target record in memory.

- [x] **Step 1: Create the fixtures**

`tests/fixtures/targets_tiny.csv`:

```csv
id,label,synonyms,definition
CHEBI:17234,glucose,dextrose|grape sugar,A monosaccharide sugar.
CHEBI:28757,fructose,fruit sugar|levulose,A ketohexose sugar.
CHEBI:17992,sucrose,table sugar|saccharose,A disaccharide of glucose and fructose.
CHEBI:17716,lactose,milk sugar,A disaccharide found in milk.
CHEBI:15903,beta-D-glucose,,The beta-anomer of D-glucose.
```

`tests/fixtures/sources_tiny.csv`:

```csv
mention_id,mention,context_left,context_right
s1,glucose,blood ,levels were elevated
s2,dextrose,intravenous ,was administered
s3,table sugar,a spoonful of ,in the coffee
s4,unobtainium,a sample of ,was requested
```

`s4` is deliberately unmatchable — it exercises `NO_CANDIDATES` and abstention end-to-end.

- [x] **Step 2: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def targets_csv() -> Path:
    return FIXTURES / "targets_tiny.csv"


@pytest.fixture
def sources_csv() -> Path:
    return FIXTURES / "sources_tiny.csv"
```

- [x] **Step 3: Write the failing tests**

Create `tests/test_sources.py`:

```python
import json

import pytest

from xwalk.sources.tabular import csv_source, jsonl_source


def test_csv_source_yields_one_record_per_row(targets_csv):
    records = list(csv_source(targets_csv, id_column="id"))
    assert len(records) == 5
    assert records[0].id == "CHEBI:17234"


def test_csv_source_drops_the_id_column_from_fields(targets_csv):
    record = next(iter(csv_source(targets_csv, id_column="id")))
    assert "id" not in record.fields
    assert record.fields["label"] == "glucose"


def test_csv_source_splits_multivalue_columns(targets_csv):
    record = next(iter(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"])))
    assert record.fields["synonyms"] == ["dextrose", "grape sugar"]


def test_csv_source_yields_empty_list_for_a_blank_multivalue_cell(targets_csv):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    assert records[4].fields["synonyms"] == []


def test_csv_source_rejects_a_missing_id_column(targets_csv):
    with pytest.raises(ValueError, match="id_column"):
        list(csv_source(targets_csv, id_column="nope"))


def test_csv_source_rejects_a_blank_id(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,label\n,glucose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        list(csv_source(path, id_column="id"))


def test_csv_source_is_lazy(targets_csv):
    """A 10M-row CSV must not be materialised to read the first record."""
    import types

    assert isinstance(csv_source(targets_csv, id_column="id"), types.GeneratorType)


def test_jsonl_source_reads_nested_values(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"pk": "a", "name": "x", "tags": ["t1", "t2"]}) + "\n", encoding="utf-8"
    )
    record = next(iter(jsonl_source(path, id_field="pk")))
    assert record.id == "a"
    assert record.fields["tags"] == ["t1", "t2"]


def test_jsonl_source_skips_blank_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"pk":"a"}\n\n{"pk":"b"}\n', encoding="utf-8")
    assert [r.id for r in jsonl_source(path, id_field="pk")] == ["a", "b"]


def test_jsonl_source_reports_the_line_number_on_bad_json(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"pk":"a"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(jsonl_source(path, id_field="pk"))


def test_jsonl_source_coerces_a_numeric_id_to_string(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"pk": 3, "name": "x"}\n', encoding="utf-8")
    assert next(iter(jsonl_source(path, id_field="pk"))).id == "3"
```

Create `tests/test_stores.py`:

```python
import pytest

from xwalk.records import Record
from xwalk.sources.tabular import csv_source
from xwalk.stores.memory import MemoryStore


def test_from_source_indexes_every_record(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    assert len(store) == 5


def test_get_returns_the_record(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    assert store.get("CHEBI:17234").fields["label"] == "glucose"


def test_get_raises_keyerror_for_an_unknown_id(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    with pytest.raises(KeyError):
        store.get("CHEBI:99999")


def test_get_many_preserves_request_order(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    got = store.get_many(["CHEBI:17992", "CHEBI:17234"])
    assert [r.id for r in got] == ["CHEBI:17992", "CHEBI:17234"]


def test_get_many_skips_unknown_ids_without_raising(targets_csv):
    """A retriever backed by a stale index can name a since-deleted record."""
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    got = store.get_many(["CHEBI:17234", "GONE:1"])
    assert [r.id for r in got] == ["CHEBI:17234"]


def test_duplicate_ids_are_rejected():
    dupes = [Record(id="a", fields={"n": 1}), Record(id="a", fields={"n": 2})]
    with pytest.raises(ValueError, match="duplicate"):
        MemoryStore.from_source(dupes)


def test_fingerprint_is_order_independent():
    a = MemoryStore.from_source([Record(id="a", fields={}), Record(id="b", fields={})])
    b = MemoryStore.from_source([Record(id="b", fields={}), Record(id="a", fields={})])
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_content_changes():
    a = MemoryStore.from_source([Record(id="a", fields={"n": 1})])
    b = MemoryStore.from_source([Record(id="a", fields={"n": 2})])
    assert a.fingerprint != b.fingerprint


def test_iterating_the_store_yields_records(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    assert {r.id for r in store} == {
        "CHEBI:17234",
        "CHEBI:28757",
        "CHEBI:17992",
        "CHEBI:17716",
        "CHEBI:15903",
    }
```

- [x] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_sources.py tests/test_stores.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.sources'`.

- [x] **Step 5: Write `src/xwalk/sources/base.py`**

```python
"""The entire extension contract for input data."""

from __future__ import annotations

from collections.abc import Iterable

from xwalk.records import Record

# A RecordSource is anything iterable that yields Records. A list works. A generator
# reading 10M rows works. A function querying your warehouse works. Nothing in xwalk
# needs to know your dataset exists.
RecordSource = Iterable[Record]

__all__ = ["RecordSource"]
```

`src/xwalk/sources/__init__.py`:

```python
from xwalk.sources.base import RecordSource
from xwalk.sources.tabular import csv_source, jsonl_source

__all__ = ["RecordSource", "csv_source", "jsonl_source"]
```

- [x] **Step 6: Write `src/xwalk/sources/tabular.py`**

```python
"""CSV / TSV / JSONL record sources. Lazy by construction."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from xwalk.records import Record

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def csv_source(
    path: str | Path,
    *,
    id_column: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    multivalue_columns: Sequence[str] = (),
    multivalue_sep: str = "|",
) -> Iterator[Record]:
    """Yield one Record per row. The id column is removed from `fields`.

    `multivalue_columns` are split on `multivalue_sep` into lists, with blank parts
    dropped — a blank cell becomes `[]`, never `[""]`.
    """
    multivalue = set(multivalue_columns)
    path = Path(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or id_column not in reader.fieldnames:
            raise ValueError(
                f"id_column {id_column!r} not found in {path}; columns are {reader.fieldnames!r}"
            )
        for line_no, row in enumerate(reader, start=2):  # header is line 1
            raw_id = (row.pop(id_column) or "").strip()
            if not raw_id:
                raise ValueError(f"{path}: row {line_no} has a blank {id_column!r}")
            fields: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue  # extra columns beyond the header
                text = value or ""
                if key in multivalue:
                    fields[key] = [p.strip() for p in text.split(multivalue_sep) if p.strip()]
                else:
                    fields[key] = text.strip()
            yield Record(id=raw_id, fields=fields)


def jsonl_source(
    path: str | Path,
    *,
    id_field: str,
    encoding: str = "utf-8",
) -> Iterator[Record]:
    """Yield one Record per non-blank line. The id field is removed from `fields`."""
    path = Path(path)
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no} is not valid JSON: {exc}") from exc
            if id_field not in obj:
                raise ValueError(f"{path}: line {line_no} has no {id_field!r}")
            raw_id = str(obj.pop(id_field)).strip()
            if not raw_id:
                raise ValueError(f"{path}: line {line_no} has a blank {id_field!r}")
            yield Record(id=raw_id, fields=obj)
```

- [x] **Step 7: Write `src/xwalk/stores/base.py`**

```python
"""Target storage, deliberately separate from retrieval.

Retrievers return IDs and ranks; the store turns IDs into Records. Keeping them apart
is what makes "plug in an external search backend for scale" actually work — otherwise
the retriever still has to hold every target record in memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from xwalk.records import Record


@runtime_checkable
class TargetStore(Protocol):
    @property
    def fingerprint(self) -> str:
        """Digest of the snapshot. Feeds the run fingerprint."""

    def get(self, record_id: str) -> Record:
        """Return one record, or raise KeyError."""

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        """Return records in request order, silently skipping unknown IDs."""


__all__ = ["TargetStore"]
```

`src/xwalk/stores/__init__.py`:

```python
from xwalk.stores.base import TargetStore
from xwalk.stores.memory import MemoryStore

__all__ = ["MemoryStore", "TargetStore"]
```

- [x] **Step 8: Write `src/xwalk/stores/memory.py`**

```python
"""In-memory target store. The default for collections that fit in RAM."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record


class MemoryStore:
    """Dict-backed TargetStore built eagerly from any iterable of Records."""

    def __init__(self, records: dict[str, Record], fingerprint: str) -> None:
        self._records = records
        self._fingerprint = fingerprint

    @classmethod
    def from_source(cls, source: Iterable[Record]) -> MemoryStore:
        records: dict[str, Record] = {}
        digests: list[str] = []
        for record in source:
            if record.id in records:
                raise ValueError(f"duplicate target record id: {record.id!r}")
            records[record.id] = record
            digests.append(hash_record(record))
        return cls(records, hash_value(sorted(digests)))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def get(self, record_id: str) -> Record:
        return self._records[record_id]

    def get_many(self, record_ids: Sequence[str]) -> Sequence[Record]:
        # Unknown IDs are skipped rather than raising: a retriever backed by a stale
        # index can legitimately name a record that has since been removed.
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records.values())

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._records
```

- [x] **Step 9: Run the tests**

Run: `python -m pytest tests/test_sources.py tests/test_stores.py -v`
Expected: 20 passed.

- [x] **Step 10: Verify `MemoryStore` satisfies the protocol**

Add to `tests/test_stores.py`:

```python
def test_memory_store_satisfies_the_target_store_protocol():
    from xwalk.stores.base import TargetStore

    store = MemoryStore.from_source([Record(id="a", fields={})])
    assert isinstance(store, TargetStore)
```

Run: `python -m pytest tests/test_stores.py -v` — Expected: 10 passed.

- [x] **Step 11: Lint and type-check**

Run: `python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: clean.

- [x] **Step 12: Commit**

```bash
git add src/xwalk/sources src/xwalk/stores tests/
git commit -m "feat: record sources and in-memory target store"
```

---

## Task 5: Retriever protocol and the BM25 retriever

**Files:**
- Create: `src/xwalk/retrieval/__init__.py`, `src/xwalk/retrieval/base.py`, `src/xwalk/retrieval/bm25.py`
- Test: `tests/test_bm25.py`

**Interfaces:**
- Consumes: `Record` (Task 1), `hash_value` (Task 2), `TemplateSet` (Task 3), `TargetStore`/`MemoryStore` (Task 4).
- Produces: `SearchRequest(text: str, limit: int, filters: Mapping[str, Any] | None = None, source_record: Record | None = None)`; `Retriever` protocol with `name: str`, `fingerprint: str`, `default_limit: int`, `async search(request) -> Sequence[RetrievalHit]`; `RetrieverError`; `sanitise_query(text) -> str`; `BM25Retriever.build(records, templates, index_dir, *, name="bm25", exact_fields=(), default_limit=20) -> BM25Retriever` and `BM25Retriever.open(index_dir, *, name=None) -> BM25Retriever`, with attribute `empty_doc_count: int`. Tasks 6, 17 consume these.

**Why the first step is a scratch script:** this plan will not assert a third-party API it has not verified. Tantivy's Python bindings have changed shape across releases. Verify, then write.

**Why there is an exact-match field.** `test_exact_label_match_ranks_first` is a *requirement*, and plain BM25 over one concatenated text field does not satisfy it. On this plan's own fixture, the query `glucose` ranks `CHEBI:15903` (*beta-D-glucose*) **above** `CHEBI:17234` (*glucose*): the former tokenises to `beta`/`d`/`glucose`, contains `glucose` twice counting `D-glucose` in its definition, and is the shorter document, so term frequency and length normalisation both favour it. Field boosting does not fix this — it scales both documents. The fix is a second field, `exact`, indexed with Tantivy's `raw` tokeniser holding the whole lower-cased value of each field named in `exact_fields` (label and synonyms), queried as a boosted term query and OR-ed with the ordinary BM25 text query. An exact whole-string hit then dominates, and everything else degrades to normal BM25. With `exact_fields=()` the retriever behaves exactly as a plain BM25 index.

`default_limit` lives on the retriever because the spec is explicit that "retrieval depth `k` belongs to each retriever's own configuration — a BM25 and a dense retriever have no reason to share a depth." `Matcher.retriever_limit` is only the fallback for a retriever that does not declare one.

- [ ] **Step 1: Verify the Tantivy API before writing any code against it**

```bash
python -m pip install "tantivy>=0.22"
python - <<'PY'
import tantivy, tempfile, pathlib
print("tantivy", getattr(tantivy, "__version__", "unknown"))
print("exports:", sorted(n for n in dir(tantivy) if not n.startswith("_")))
sb = tantivy.SchemaBuilder()
print("SchemaBuilder methods:", sorted(n for n in dir(sb) if not n.startswith("_")))
sb.add_text_field("record_id", stored=True)
sb.add_text_field("text", stored=False)
schema = sb.build()
d = pathlib.Path(tempfile.mkdtemp())
index = tantivy.Index(schema, path=str(d))
print("Index methods:", sorted(n for n in dir(index) if not n.startswith("_")))
w = index.writer()
print("writer methods:", sorted(n for n in dir(w) if not n.startswith("_")))
w.add_document(tantivy.Document(record_id="A", text="glucose dextrose"))
w.add_document(tantivy.Document(record_id="B", text="fructose"))
w.commit()
index.reload()
s = index.searcher()
q = index.parse_query("glucose", ["text"])
res = s.search(q, 5)
print("search result type:", type(res), sorted(n for n in dir(res) if not n.startswith("_")))
print("hits:", res.hits)
score, addr = res.hits[0]
doc = s.doc(addr)
print("doc:", doc, "record_id ->", doc["record_id"])
PY
```

Then probe the second group of names, which the exact-match field needs:

```bash
python - <<'PY'
import tantivy, inspect
print("exports:", sorted(n for n in dir(tantivy) if not n.startswith("_")))
print("add_text_field:", inspect.signature(tantivy.SchemaBuilder.add_text_field))
print("parse_query:", inspect.signature(tantivy.Index.parse_query))
print("Occur:", [n for n in dir(tantivy.Occur) if not n.startswith("_")])
for n in ("term_query", "boost_query", "boolean_query"):
    print(n, "->", (getattr(tantivy.Query, n).__doc__ or "?").strip().splitlines()[0])
PY
```

Record the real names in a scratch note. If any of `SchemaBuilder`, `add_text_field(..., tokenizer_name="raw")`, `Index(schema, path=...)`, `index.writer()`, `writer.add_document(tantivy.Document(...))`, `document.add_text(field, value)`, `writer.commit()`, `index.reload()`, `index.searcher()`, `index.parse_query(text, [field])`, `tantivy.Query.term_query(schema, field, value)`, `tantivy.Query.boost_query(query, boost)`, `tantivy.Query.boolean_query([(tantivy.Occur.Should, q), ...])`, `result.hits` as `(score, address)` pairs, or `searcher.doc(address)[field]` returning a **list** differ from the above, adapt Step 5's code to what you observed and note the deviation in the commit message. **Do not adapt the tests** — they are written against behaviour, not against Tantivy.

Verified against `tantivy 0.26.0` (index format v7) at the time of writing: every name above exists with these shapes, and `searcher.doc(address)["record_id"]` returns a one-element list.

- [ ] **Step 2: Write the failing test**

Create `tests/test_bm25.py`:

```python
import pytest

from xwalk.retrieval.base import SearchRequest
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source
from xwalk.templates import TemplateSet

DOC_TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="",
    doc="{{ label }} {{ synonyms | join(' ') }} {{ definition }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)


EXACT_FIELDS = ("label", "synonyms")


@pytest.fixture
def retriever(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    return BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "idx", exact_fields=EXACT_FIELDS)


async def test_exact_label_match_ranks_first(retriever):
    """Plain BM25 puts beta-D-glucose first here — it contains 'glucose' twice in a
    shorter document. The exact field is what makes this assertion a requirement."""
    hits = await retriever.search(SearchRequest(text="glucose", limit=5))
    assert hits[0].record_id == "CHEBI:17234"


async def test_an_exact_synonym_outranks_a_partial_label_match(retriever):
    hits = await retriever.search(SearchRequest(text="table sugar", limit=5))
    assert hits[0].record_id == "CHEBI:17992"


async def test_without_exact_fields_it_is_a_plain_bm25_index(targets_csv, tmp_path):
    """`exact_fields=()` must still build, search, and return hits — the exact field is
    an opt-in ranking aid, never a requirement for the retriever to function."""
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    plain = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "plain")
    hits = await plain.search(SearchRequest(text="glucose", limit=5))
    assert [h.record_id for h in hits]


async def test_default_limit_is_exposed_for_the_matcher(retriever):
    assert retriever.default_limit == 20


async def test_synonyms_are_searchable(retriever):
    hits = await retriever.search(SearchRequest(text="dextrose", limit=5))
    assert hits[0].record_id == "CHEBI:17234"


async def test_ranks_are_one_based_and_contiguous(retriever):
    hits = await retriever.search(SearchRequest(text="sugar", limit=5))
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


async def test_hits_carry_the_retriever_name(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=5))
    assert all(h.retriever == "bm25" for h in hits)


async def test_limit_is_respected(retriever):
    hits = await retriever.search(SearchRequest(text="sugar", limit=2))
    assert len(hits) <= 2


async def test_no_match_returns_empty_not_an_error(retriever):
    assert await retriever.search(SearchRequest(text="unobtainium", limit=5)) == []


async def test_blank_query_returns_empty(retriever):
    assert await retriever.search(SearchRequest(text="   ", limit=5)) == []


async def test_query_syntax_characters_are_treated_as_text(retriever):
    """Real mentions contain +, -, :, (, ), ^, ~, AND, OR. None may reach the parser."""
    for text in ["glucose:", "glucose (D-)", "glucose AND fructose", "+glucose^2", "gluc*ose~"]:
        hits = await retriever.search(SearchRequest(text=text, limit=5))
        assert isinstance(hits, list)  # no exception, no parser error


async def test_open_reuses_a_built_index(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    built = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "idx", exact_fields=EXACT_FIELDS)
    reopened = BM25Retriever.open(tmp_path / "idx")
    assert reopened.fingerprint == built.fingerprint
    hits = await reopened.search(SearchRequest(text="glucose", limit=5))
    assert hits[0].record_id == "CHEBI:17234"


def test_open_restores_the_name_and_exact_fields_from_meta(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    BM25Retriever.build(
        records, DOC_TEMPLATES, tmp_path / "idx", name="lexical", exact_fields=EXACT_FIELDS
    )
    reopened = BM25Retriever.open(tmp_path / "idx")
    assert reopened.name == "lexical"


def test_fingerprint_changes_with_the_exact_fields(targets_csv, tmp_path):
    """Exact fields change ranking, so they must change the index fingerprint."""
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    a = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "a", exact_fields=EXACT_FIELDS)
    b = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "b")
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_the_doc_template(targets_csv, tmp_path):
    records = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    a = BM25Retriever.build(records, DOC_TEMPLATES, tmp_path / "a")
    other = TemplateSet(query="", context="", doc="{{ label }}", candidate="")
    b = BM25Retriever.build(records, other, tmp_path / "b")
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_the_target_content(tmp_path):
    from xwalk.records import Record

    a = BM25Retriever.build([Record(id="x", fields={"label": "a"})], DOC_TEMPLATES, tmp_path / "a")
    b = BM25Retriever.build([Record(id="x", fields={"label": "b"})], DOC_TEMPLATES, tmp_path / "b")
    assert a.fingerprint != b.fingerprint


def test_records_with_empty_rendered_docs_are_reported(tmp_path):
    from xwalk.records import Record

    r = BM25Retriever.build(
        [Record(id="x", fields={"label": ""}), Record(id="y", fields={"label": "glucose"})],
        DOC_TEMPLATES,
        tmp_path / "idx",
    )
    assert r.empty_doc_count == 1


def test_bm25_satisfies_the_retriever_protocol(retriever):
    from xwalk.retrieval.base import Retriever

    assert isinstance(retriever, Retriever)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.retrieval'`.

- [ ] **Step 4: Write `src/xwalk/retrieval/base.py`**

```python
"""The retrieval contract. Implement this to plug in any backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from xwalk.records import Record, RetrievalHit


class RetrieverError(Exception):
    """A retriever failed. The matcher degrades to the remaining retrievers."""


@dataclass(frozen=True)
class SearchRequest:
    """A request object rather than `search(query, k)`.

    Filters and field-aware retrieval can then be added without a breaking protocol
    change, and a backend that ignores `source_record` stays correct.
    """

    text: str
    limit: int
    filters: Mapping[str, Any] | None = None
    source_record: Record | None = None


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier. Appears in RetrievalHit.retriever and in diagnostics."""

    @property
    def fingerprint(self) -> str:
        """Digest of index content and configuration. Feeds the run fingerprint."""

    @property
    def default_limit(self) -> int:
        """This retriever's own retrieval depth.

        Depth belongs here rather than on the matcher: a BM25 index and a dense index
        have no reason to share a `k`. The matcher supplies a fallback only.
        """

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        """Return hits ranked best-first with 1-based contiguous ranks."""


__all__ = ["Retriever", "RetrieverError", "SearchRequest"]
```

`src/xwalk/retrieval/__init__.py` — **note what is *not* here.** `fusion` does not exist until Task 6, and `__init__.py` runs before any submodule import, so naming it now would make `from xwalk.retrieval.base import SearchRequest` raise `ModuleNotFoundError` and fail all of this task's tests for the wrong reason. Task 6 Step 5 adds the fusion line.

```python
from xwalk.retrieval.base import Retriever, RetrieverError, SearchRequest
from xwalk.retrieval.bm25 import BM25Retriever

__all__ = [
    "BM25Retriever",
    "Retriever",
    "RetrieverError",
    "SearchRequest",
]
```

- [ ] **Step 5: Write `src/xwalk/retrieval/bm25.py`**

Adapt the Tantivy calls to whatever Step 1 observed.

```python
"""BM25 retrieval over Tantivy.

Tantivy is a hard dependency with a declared support matrix (manylinux x86-64 and
aarch64, macOS arm64, Windows x86-64). There is deliberately no automatic fallback to
another engine: a silent switch would change ranking behaviour and quietly undermine
reproducibility. On an unsupported platform the user selects an alternate retriever
explicitly.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import tantivy

from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest
from xwalk.templates import TemplateSet

_META_FILE = "xwalk_meta.json"

# Tantivy's query parser gives these characters meaning. Real mentions contain them —
# "glucose (D-)", "IL-2", "5-HT2A", "Ca2+" — so they are stripped rather than escaped.
# BM25 over a tokenised text field does not need the operators.
_QUERY_SYNTAX = re.compile(r"[+\-!(){}\[\]^\"~*?:\\/]|(?<!\w)(?:AND|OR|NOT|IN)(?!\w)")
_WS = re.compile(r"\s+")


def sanitise_query(text: str) -> str:
    """Reduce arbitrary text to something the query parser cannot misinterpret."""
    return _WS.sub(" ", _QUERY_SYNTAX.sub(" ", text)).strip()


# An exact whole-string hit on a label or synonym must outrank a document that merely
# happens to contain the query term more often. 10.0 is comfortably above any BM25 score
# the tokenised `text` field produces on a collection of this shape.
_EXACT_BOOST = 10.0

# Tantivy's `writer()` defaults are heap_size=128 MB and num_threads=0, where 0 means
# "one indexing thread per core" — and the heap is allocated *per thread*. On a 64-core
# machine that is an 8 GB arena, which costs ~7 s to set up whether the index holds five
# documents or five million. These defaults are deliberately modest; raise them for large
# collections. Tantivy's own floor is 15 MB.
_WRITER_HEAP_BYTES = 50_000_000
_WRITER_THREADS = 1


def _build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("record_id", stored=True)
    # `raw` = no tokenisation: the whole field value is one term, so "glucose" matches
    # the label "glucose" but not the label "beta-D-glucose".
    builder.add_text_field("exact", stored=False, tokenizer_name="raw")
    builder.add_text_field("text", stored=False)
    return builder.build()


def _exact_forms(record: Record, exact_fields: Sequence[str]) -> list[str]:
    """Every whole-string form of a record that should count as an exact match."""
    forms: list[str] = []
    for field_name in exact_fields:
        value = record.fields.get(field_name)
        if value is None:
            continue
        candidates = [value] if isinstance(value, str) else list(value)
        for candidate in candidates:
            text = str(candidate).strip().lower()
            if text and text not in forms:
                forms.append(text)
    return forms


class BM25Retriever:
    """Tantivy-backed BM25, with an opt-in exact-match field. Reuse via `open`."""

    def __init__(
        self,
        index: tantivy.Index,
        *,
        name: str,
        fingerprint: str,
        exact_fields: Sequence[str] = (),
        default_limit: int = 20,
        empty_doc_count: int = 0,
    ) -> None:
        self._index = index
        self._name = name
        self._fingerprint = fingerprint
        self._exact_fields = tuple(exact_fields)
        self._default_limit = default_limit
        self.empty_doc_count = empty_doc_count

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def default_limit(self) -> int:
        return self._default_limit

    @classmethod
    def build(
        cls,
        records: Iterable[Record],
        templates: TemplateSet,
        index_dir: str | Path,
        *,
        name: str = "bm25",
        exact_fields: Sequence[str] = (),
        default_limit: int = 20,
        writer_heap_bytes: int = _WRITER_HEAP_BYTES,
        writer_threads: int = _WRITER_THREADS,
    ) -> BM25Retriever:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        index = tantivy.Index(_build_schema(), path=str(index_dir))
        writer = index.writer(heap_size=writer_heap_bytes, num_threads=writer_threads)

        digests: list[str] = []
        empty = 0
        for record in records:
            text = templates.render_doc(record)
            if not text:
                empty += 1
            document = tantivy.Document(record_id=record.id, text=text)
            for form in _exact_forms(record, exact_fields):
                document.add_text("exact", form)
            writer.add_document(document)
            digests.append(hash_record(record))
        writer.commit()
        index.reload()

        fingerprint = hash_value(
            {
                "engine": "tantivy-bm25",
                "doc_template": templates.doc,
                "exact_fields": list(exact_fields),
                "records": sorted(digests),
            }
        )
        (index_dir / _META_FILE).write_text(
            json.dumps(
                {
                    "engine": "tantivy-bm25",
                    "name": name,
                    "fingerprint": fingerprint,
                    "exact_fields": list(exact_fields),
                    "default_limit": default_limit,
                    "doc_count": len(digests),
                    "empty_doc_count": empty,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(
            index,
            name=name,
            fingerprint=fingerprint,
            exact_fields=exact_fields,
            default_limit=default_limit,
            empty_doc_count=empty,
        )

    @classmethod
    def open(cls, index_dir: str | Path, *, name: str | None = None) -> BM25Retriever:
        """Reopen a built index. `name` overrides the one recorded at build time."""
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{index_dir} is not an xwalk BM25 index (no {_META_FILE}); build it first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = tantivy.Index(_build_schema(), path=str(index_dir))
        index.reload()
        return cls(
            index,
            name=name or meta["name"],
            fingerprint=meta["fingerprint"],
            exact_fields=meta.get("exact_fields", ()),
            default_limit=meta.get("default_limit", 20),
            empty_doc_count=meta.get("empty_doc_count", 0),
        )

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        cleaned = sanitise_query(text)
        if not cleaned:
            return []
        searcher = self._index.searcher()
        text_query = self._index.parse_query(cleaned, ["text"])
        if self._exact_fields:
            exact_query = tantivy.Query.boost_query(
                tantivy.Query.term_query(self._index.schema, "exact", cleaned.lower()),
                _EXACT_BOOST,
            )
            query = tantivy.Query.boolean_query(
                [
                    (tantivy.Occur.Should, exact_query),
                    (tantivy.Occur.Should, text_query),
                ]
            )
        else:
            query = text_query
        result = searcher.search(query, limit)
        hits: list[RetrievalHit] = []
        for rank, (score, address) in enumerate(result.hits, start=1):
            doc = searcher.doc(address)
            record_id = doc["record_id"][0]
            hits.append(
                RetrievalHit(
                    record_id=record_id,
                    retriever=self._name,
                    raw_score=float(score),
                    rank=rank,
                )
            )
        return hits

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        # Tantivy's search is synchronous CPU work; a thread keeps the event loop free
        # so N retrievers genuinely run concurrently.
        return await asyncio.to_thread(self._search_sync, request.text, request.limit)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_bm25.py -v`
Expected: 18 passed, in well under a second with `TMPDIR` on tmpfs. If a ranking assertion fails, fix the *schema, sanitiser, or doc template* — never the assertion. "Exact label match ranks first" is a requirement, and the `exact` field is how it is met; see the note at the top of this task for why plain BM25 cannot meet it on this fixture.

- [ ] **Step 7: Lint and type-check**

Run: `python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: clean. `tantivy` ships no stubs; if mypy complains, add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = "tantivy.*"
ignore_missing_imports = true
```

- [ ] **Step 8: Commit**

```bash
git add src/xwalk/retrieval tests/test_bm25.py pyproject.toml
git commit -m "feat: Retriever protocol and Tantivy BM25 retriever"
```

---

## Task 6: Reciprocal rank fusion

**Files:**
- Create: `src/xwalk/retrieval/fusion.py`
- Modify: `src/xwalk/retrieval/__init__.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `RetrievalHit`, `Candidate` (Task 1), `TargetStore` (Task 4).
- Produces: `reciprocal_rank_fusion(hit_groups: Sequence[Sequence[RetrievalHit]], store: TargetStore, *, k: int = 60) -> list[Candidate]`. Task 17 (matcher) is the only caller.

**Design note:** RRF scores each hit `1 / (k + rank)` and sums across retrievers. It needs no score calibration between retrievers, which matters because BM25 scores and cosine similarities are not comparable. `k=60` is the standard default. Every fused `Candidate` carries `evidence` — the full set of `RetrievalHit`s that produced it — and that is exactly what the Phase 2 ceiling diagnostic reads.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
import pytest

from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.stores.memory import MemoryStore


@pytest.fixture
def store():
    return MemoryStore.from_source(
        [Record(id=letter, fields={"label": letter}) for letter in "ABCDE"]
    )


def hit(record_id, retriever, rank, score=1.0):
    return RetrievalHit(record_id=record_id, retriever=retriever, raw_score=score, rank=rank)


def test_single_retriever_preserves_its_order(store):
    hits = [hit("A", "bm25", 1), hit("B", "bm25", 2), hit("C", "bm25", 3)]
    assert [c.id for c in reciprocal_rank_fusion([hits], store)] == ["A", "B", "C"]


def test_agreement_beats_a_single_first_place(store):
    """B is second for both retrievers; A is first for one and absent from the other.

    RRF: A = 1/61 = 0.01639. B = 2 * 1/62 = 0.03226. Consensus wins, which is the
    entire reason for fusing rather than concatenating.
    """
    bm25 = [hit("A", "bm25", 1), hit("B", "bm25", 2)]
    dense = [hit("C", "dense", 1), hit("B", "dense", 2)]
    assert reciprocal_rank_fusion([bm25, dense], store)[0].id == "B"


def test_score_matches_the_rrf_formula(store):
    fused = reciprocal_rank_fusion([[hit("A", "bm25", 1)]], store, k=60)
    assert fused[0].fused_score == pytest.approx(1 / 61)


def test_k_is_configurable(store):
    fused = reciprocal_rank_fusion([[hit("A", "bm25", 1)]], store, k=10)
    assert fused[0].fused_score == pytest.approx(1 / 11)


def test_evidence_collects_every_retriever_that_surfaced_the_record(store):
    bm25 = [hit("A", "bm25", 1)]
    dense = [hit("A", "dense", 4)]
    [candidate] = reciprocal_rank_fusion([bm25, dense], store)
    assert {(e.retriever, e.rank) for e in candidate.evidence} == {("bm25", 1), ("dense", 4)}


def test_evidence_order_is_deterministic(store):
    bm25 = [hit("A", "bm25", 1)]
    dense = [hit("A", "dense", 4)]
    forward = reciprocal_rank_fusion([bm25, dense], store)[0].evidence
    backward = reciprocal_rank_fusion([dense, bm25], store)[0].evidence
    assert [(e.retriever, e.rank) for e in forward] == [(e.retriever, e.rank) for e in backward]


def test_ties_break_on_record_id(store):
    a = [hit("B", "r1", 1)]
    b = [hit("A", "r2", 1)]
    assert [c.id for c in reciprocal_rank_fusion([a, b], store)] == ["A", "B"]


def test_ids_missing_from_the_store_are_dropped(store):
    hits = [hit("A", "bm25", 1), hit("GONE", "bm25", 2)]
    assert [c.id for c in reciprocal_rank_fusion([hits], store)] == ["A"]


def test_empty_input_yields_no_candidates(store):
    assert reciprocal_rank_fusion([], store) == []
    assert reciprocal_rank_fusion([[], []], store) == []


def test_duplicate_hits_from_one_retriever_count_once(store):
    """A malformed backend returning the same ID twice must not double its score."""
    hits = [hit("A", "bm25", 1), hit("A", "bm25", 2)]
    [candidate] = reciprocal_rank_fusion([hits], store)
    assert candidate.fused_score == pytest.approx(1 / 61)
    assert len(candidate.evidence) == 1


def test_rejects_a_non_positive_k(store):
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([[hit("A", "bm25", 1)]], store, k=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.retrieval.fusion'`.

- [ ] **Step 3: Write `src/xwalk/retrieval/fusion.py`**

```python
"""Reciprocal rank fusion.

Combines rankings from retrievers whose raw scores are not comparable — a BM25 score
and a cosine similarity have no shared scale — by using rank position alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from xwalk.records import Candidate, RetrievalHit
from xwalk.stores.base import TargetStore


def reciprocal_rank_fusion(
    hit_groups: Sequence[Sequence[RetrievalHit]],
    store: TargetStore,
    *,
    k: int = 60,
) -> list[Candidate]:
    """Fuse per-retriever rankings into scored candidates carrying their evidence.

    Score is the sum of `1 / (k + rank)` over the retrievers that surfaced the record.
    Records absent from the store are dropped: a stale index may name a deleted record.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    evidence: dict[str, dict[str, RetrievalHit]] = {}

    for group in hit_groups:
        for hit in group:
            per_retriever = evidence.setdefault(hit.record_id, {})
            if hit.retriever in per_retriever:
                continue  # one vote per retriever, best rank wins by arrival order
            per_retriever[hit.retriever] = hit
            scores[hit.record_id] = scores.get(hit.record_id, 0.0) + 1.0 / (k + hit.rank)

    if not scores:
        return []

    records = {r.id: r for r in store.get_many(sorted(scores))}

    candidates = [
        Candidate(
            record=records[record_id],
            fused_score=score,
            # sorted so the trace is byte-identical regardless of retriever completion order
            evidence=tuple(
                sorted(evidence[record_id].values(), key=lambda h: (h.retriever, h.rank))
            ),
        )
        for record_id, score in scores.items()
        if record_id in records
    ]
    # descending score, then ascending id — a stable, reproducible order
    candidates.sort(key=lambda c: (-c.fused_score, c.id))
    return candidates
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: 11 passed.

- [ ] **Step 5: Write the real `src/xwalk/retrieval/__init__.py`**

```python
from xwalk.retrieval.base import Retriever, RetrieverError, SearchRequest
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.fusion import reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "Retriever",
    "RetrieverError",
    "SearchRequest",
    "reciprocal_rank_fusion",
]
```

- [ ] **Step 6: Run the whole suite, lint, and type-check**

Run: `python -m pytest -q && python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/xwalk/retrieval tests/test_fusion.py
git commit -m "feat: reciprocal rank fusion with per-retriever evidence"
```

---

## Task 7: LLM protocol, response parsing, and `FakeLLM`

**Files:**
- Create: `src/xwalk/llm/__init__.py`, `src/xwalk/llm/base.py`, `src/xwalk/llm/parsing.py`, `src/xwalk/llm/fake.py`
- Test: `tests/test_parsing.py`, `tests/test_fake_llm.py`

**Interfaces:**
- Consumes: `Usage` (Task 1), `hash_value` (Task 2).
- Produces: `LLMCapabilities`; `LLMRequest`; `LLMResponse`; `LLMClient` protocol; `LLMError`/`LLMRetryableError`/`LLMFatalError`/`ParseError`; `strip_thinking(text) -> str`; `extract_json_object(text) -> str | None`; `repair_json(text) -> str`; `parse_json_object(text) -> dict[str, Any]`; `FakeLLM`. Tasks 8, 11, 12, 13, 17 all consume these.

**Why `FakeLLM` comes before the real client:** today's pipeline cannot be tested without a provider. `FakeLLM` is the single biggest testability win in the rewrite — the entire matcher loop (retry paths, proposal routing, verify band, disagreement, audit sampling, exhaustion, error handling) becomes testable offline. Build it first so every later stage has it.

- [ ] **Step 1: Write the failing parsing test**

Create `tests/test_parsing.py`:

```python
import pytest

from xwalk.llm.base import ParseError
from xwalk.llm.parsing import extract_json_object, parse_json_object, repair_json, strip_thinking


def test_strips_a_complete_think_block():
    assert strip_thinking("<think>hmm</think>ANSWER") == "ANSWER"


def test_strips_thinking_and_reasoning_tags():
    assert strip_thinking("<thinking>a</thinking>X<reasoning>b</reasoning>Y") == "XY"


def test_strips_a_multiline_think_block():
    assert strip_thinking('<think>\nline1\nline2\n</think>\n{"a": 1}') == '{"a": 1}'


def test_handles_a_dangling_closing_tag():
    """Some models emit reasoning with no opening tag. Everything before </think> is reasoning."""
    assert strip_thinking('I should pick C01.\n</think>\n{"chosen_key": "C01"}') == (
        '{"chosen_key": "C01"}'
    )


def test_handles_an_unclosed_opening_tag():
    """Truncated output: an opening tag with no close means nothing usable follows."""
    assert strip_thinking("prefix<think>reasoning that never ends") == "prefix"


def test_leaves_ordinary_text_alone():
    assert strip_thinking('{"a": 1}') == '{"a": 1}'


def test_is_case_insensitive_about_tags():
    assert strip_thinking("<THINK>x</THINK>Y") == "Y"


def test_extracts_json_from_a_fenced_block():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json_object(text) == '{"a": 1}'


def test_extracts_json_from_an_unlabelled_fence():
    assert extract_json_object('```\n{"a": 1}\n```') == '{"a": 1}'


def test_extracts_a_bare_object_surrounded_by_prose():
    assert extract_json_object('Sure. {"a": 1} Done.') == '{"a": 1}'


def test_extracts_a_balanced_object_containing_nested_braces():
    assert extract_json_object('x {"a": {"b": 2}} y') == '{"a": {"b": 2}}'


def test_ignores_braces_inside_strings():
    assert extract_json_object('{"a": "} not the end"}') == '{"a": "} not the end"}'


def test_returns_none_when_there_is_no_object():
    assert extract_json_object("no json here") is None


def test_repair_removes_a_trailing_comma():
    assert repair_json('{"a": 1,}') == '{"a": 1}'


def test_repair_removes_a_trailing_comma_in_a_list():
    assert repair_json('{"a": [1, 2,]}') == '{"a": [1, 2]}'


def test_parse_handles_a_clean_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_handles_thinking_plus_fence_plus_trailing_comma():
    raw = '<think>deciding</think>\n```json\n{"chosen_key": "C01", "confidence": 0.9,}\n```'
    assert parse_json_object(raw) == {"chosen_key": "C01", "confidence": 0.9}


def test_parse_raises_on_unrecoverable_output():
    with pytest.raises(ParseError):
        parse_json_object("I refuse to answer.")


def test_parse_raises_when_the_value_is_not_an_object():
    with pytest.raises(ParseError, match="object"):
        parse_json_object("[1, 2, 3]")


def test_parse_error_carries_the_raw_text_for_the_trace():
    with pytest.raises(ParseError) as exc:
        parse_json_object("garbage")
    assert exc.value.raw == "garbage"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.llm'`.

- [ ] **Step 3: Write `src/xwalk/llm/base.py`**

```python
"""The LLM contract.

Interchangeability between OpenAI-compatible endpoints is genuinely partial: Google
offers an OpenAI-compatible Gemini endpoint while recommending the native API, and
Anthropic describes its compatibility layer as primarily for testing and notes that
strict schema enforcement may be ignored. So adapters *declare* what they do — and the
declaration only decides what to **ask** for, never whether to **trust** the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from xwalk.records import Usage


class LLMError(Exception):
    """Base class for provider failures."""


class LLMRetryableError(LLMError):
    """Transient: 408, 409, 429, 5xx, timeouts, connection resets."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMFatalError(LLMError):
    """Non-retryable: auth failure, unknown model, malformed request."""


class ParseError(LLMError):
    """The response could not be reduced to a JSON object."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class LLMCapabilities:
    """What an adapter claims it can do. Defaults are all-false: assume nothing."""

    json_schema: bool = False
    strict_schema: bool = False
    usage_reporting: bool = False
    seed: bool = False
    native_retry_after: bool = False


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: Mapping[str, Any] | None = None
    schema_name: str = "response"
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage
    model: str
    structured: bool = False
    finish_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """Digest of provider identity, model, generation params, and adapter version."""

    @property
    def capabilities(self) -> LLMCapabilities: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


__all__ = [
    "LLMCapabilities",
    "LLMClient",
    "LLMError",
    "LLMFatalError",
    "LLMRequest",
    "LLMResponse",
    "LLMRetryableError",
    "ParseError",
]
```

- [ ] **Step 4: Write `src/xwalk/llm/parsing.py`**

```python
"""Turning whatever the model said into a JSON object.

Three behaviours port over from the paper repo because they were learned the hard way:
thinking-block stripping (including the dangling-closing-tag case), fenced-block
extraction, and modest repair. Anything these cannot salvage becomes UNRESOLVED_OUTPUT
and routes to human review — a malformed answer is a signal, not a non-match.
"""

from __future__ import annotations

import json
import re
from typing import Any

from xwalk.llm.base import ParseError

_TAGS = ("think", "thinking", "reasoning")
_PAIRED = re.compile(r"<(" + "|".join(_TAGS) + r")\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_DANGLING_CLOSE = re.compile(r"^.*?</(?:" + "|".join(_TAGS) + r")\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_OPEN = re.compile(r"<(?:" + "|".join(_TAGS) + r")\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks, closed or not."""
    out = _PAIRED.sub("", text)
    if re.search(r"</(?:" + "|".join(_TAGS) + r")\s*>", out, re.IGNORECASE):
        # A closing tag with no opener: everything up to and including it was reasoning.
        out = _DANGLING_CLOSE.sub("", out)
    out = _UNCLOSED_OPEN.sub("", out)  # truncated output: an opener with no close
    return out.strip()


def extract_json_object(text: str) -> str | None:
    """Find the first balanced JSON object, preferring a fenced block."""
    fenced = _FENCE.search(text)
    haystack = fenced.group(1) if fenced else text

    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(haystack):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return haystack[start : i + 1]
    return None


def repair_json(text: str) -> str:
    """Only repairs that cannot change meaning. Trailing commas, and nothing more."""
    return _TRAILING_COMMA.sub(r"\1", text)


def parse_json_object(raw: str) -> dict[str, Any]:
    """strip thinking -> direct parse -> extract -> repair. Raise ParseError if all fail."""
    cleaned = strip_thinking(raw)
    for attempt in (cleaned, extract_json_object(cleaned), repair_json(cleaned)):
        if not attempt:
            continue
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            raise ParseError(f"expected a JSON object, got {type(value).__name__}", raw=raw)
        return value

    extracted = extract_json_object(cleaned)
    if extracted:
        try:
            repaired = json.loads(repair_json(extracted))
        except json.JSONDecodeError:
            repaired = None
        if isinstance(repaired, dict):
            return repaired
    raise ParseError("could not extract a JSON object from the response", raw=raw)
```

- [ ] **Step 5: Run the parsing tests**

Run: `python -m pytest tests/test_parsing.py -v`
Expected: 20 passed.

- [ ] **Step 6: Write the failing `FakeLLM` test**

Create `tests/test_fake_llm.py`:

```python
import pytest

from xwalk.llm.base import LLMClient, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.fake import FakeLLM

REQ = LLMRequest(system="s", user="u")


async def test_returns_scripted_responses_in_order():
    llm = FakeLLM(["first", "second"])
    assert (await llm.complete(REQ)).text == "first"
    assert (await llm.complete(REQ)).text == "second"


async def test_records_every_request():
    llm = FakeLLM(["ok"])
    await llm.complete(LLMRequest(system="sys", user="hello"))
    assert llm.requests[0].user == "hello"


async def test_raises_a_scripted_exception():
    llm = FakeLLM([LLMRetryableError("429")])
    with pytest.raises(LLMRetryableError):
        await llm.complete(REQ)


async def test_running_out_of_script_is_a_loud_failure():
    llm = FakeLLM(["only one"])
    await llm.complete(REQ)
    with pytest.raises(AssertionError, match="exhausted"):
        await llm.complete(REQ)


async def test_a_handler_can_respond_based_on_the_request():
    def handler(request: LLMRequest) -> str:
        return "picked" if "C01" in request.user else "none"

    llm = FakeLLM(handler=handler)
    assert (await llm.complete(LLMRequest(system="", user="[C01] ID: x"))).text == "picked"
    assert (await llm.complete(LLMRequest(system="", user="nothing"))).text == "none"


async def test_usage_is_reported_and_call_count_accumulates():
    llm = FakeLLM(["a", "b"])
    first = await llm.complete(REQ)
    second = await llm.complete(REQ)
    assert first.usage.calls == 1 and second.usage.calls == 1
    assert llm.total_usage.calls == 2


async def test_capabilities_are_configurable():
    from xwalk.llm.base import LLMCapabilities

    llm = FakeLLM(["x"], capabilities=LLMCapabilities(json_schema=True))
    assert llm.capabilities.json_schema is True


async def test_default_capabilities_claim_nothing():
    llm = FakeLLM(["x"])
    caps = llm.capabilities
    assert not any(
        [
            caps.json_schema,
            caps.strict_schema,
            caps.usage_reporting,
            caps.seed,
            caps.native_retry_after,
        ]
    )


def test_fingerprint_changes_with_the_script():
    assert FakeLLM(["a"]).fingerprint != FakeLLM(["b"]).fingerprint


def test_fake_satisfies_the_llm_client_protocol():
    assert isinstance(FakeLLM(["x"]), LLMClient)


async def test_fatal_errors_pass_through_unchanged():
    llm = FakeLLM([LLMFatalError("bad key")])
    with pytest.raises(LLMFatalError, match="bad key"):
        await llm.complete(REQ)
```

- [ ] **Step 7: Write `src/xwalk/llm/fake.py`**

```python
"""A scripted LLM client. The reason the whole matcher loop is testable offline."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from xwalk.fingerprint import hash_value
from xwalk.llm.base import LLMCapabilities, LLMRequest, LLMResponse
from xwalk.records import Usage

Scripted = str | BaseException
Handler = Callable[[LLMRequest], Scripted]


class FakeLLM:
    """Either a fixed script (consumed in order) or a handler that inspects the request.

    Exhausting the script raises AssertionError rather than looping or returning a
    default: a test that makes an unexpected extra call has found a real bug.
    """

    def __init__(
        self,
        script: Sequence[Scripted] | None = None,
        *,
        handler: Handler | None = None,
        capabilities: LLMCapabilities | None = None,
        model: str = "fake",
        prompt_tokens: int = 100,
        completion_tokens: int = 20,
    ) -> None:
        if (script is None) == (handler is None):
            raise ValueError("provide exactly one of script or handler")
        self._script = list(script or [])
        self._handler = handler
        self._capabilities = capabilities or LLMCapabilities()
        self._model = model
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._index = 0
        self.requests: list[LLMRequest] = []
        self.total_usage = Usage.zero()

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "adapter": "fake",
                "model": self._model,
                "script": [s if isinstance(s, str) else type(s).__name__ for s in self._script],
                "handler": self._handler.__name__ if self._handler else None,
            }
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._handler is not None:
            item: Scripted = self._handler(request)
        else:
            assert self._index < len(self._script), (
                f"FakeLLM script exhausted after {self._index} call(s); "
                f"unexpected request:\n{request.user[:400]}"
            )
            item = self._script[self._index]
            self._index += 1

        if isinstance(item, BaseException):
            raise item

        usage = Usage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            calls=1,
        )
        self.total_usage = self.total_usage + usage
        return LLMResponse(
            text=item,
            usage=usage,
            model=self._model,
            structured=bool(request.schema and self._capabilities.json_schema),
        )
```

`src/xwalk/llm/__init__.py`:

```python
from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMError,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
    ParseError,
)
from xwalk.llm.fake import FakeLLM
from xwalk.llm.parsing import parse_json_object, strip_thinking

__all__ = [
    "FakeLLM",
    "LLMCapabilities",
    "LLMClient",
    "LLMError",
    "LLMFatalError",
    "LLMRequest",
    "LLMResponse",
    "LLMRetryableError",
    "ParseError",
    "parse_json_object",
    "strip_thinking",
]
```

- [ ] **Step 8: Run the tests, lint, type-check**

Run: `python -m pytest tests/test_parsing.py tests/test_fake_llm.py -v && python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: 31 passed, clean lint and types.

- [ ] **Step 9: Commit**

```bash
git add src/xwalk/llm tests/test_parsing.py tests/test_fake_llm.py
git commit -m "feat: LLM protocol, robust response parsing, and FakeLLM"
```

---

## Task 8: OpenAI-compatible client

**Files:**
- Create: `src/xwalk/llm/openai_compat.py`
- Modify: `src/xwalk/llm/__init__.py`
- Test: `tests/test_openai_compat.py`

**Interfaces:**
- Consumes: everything from Task 7.
- Produces: `OpenAICompatClient(base_url, model, *, api_key=None, capabilities=None, profile="unknown", temperature=0.0, max_tokens=1024, timeout=120.0, max_retries=5, backoff_base=1.0, backoff_cap=30.0, transport=None, extra_headers=None)`; `CAPABILITY_PROFILES: Mapping[str, LLMCapabilities]`. Task 17 consumes it; Task 18 reads its `fingerprint`.

**Design note:** tests inject an `httpx.MockTransport`, so this task is fully offline apart from one `@pytest.mark.integration` test that skips without `XWALK_TEST_API_KEY`. Never trust a capability claim: even `strict_schema=True` output goes through `parse_json_object` at the call sites.

- [ ] **Step 1: Write the failing test**

Create `tests/test_openai_compat.py`:

```python
import json
import os

import httpx
import pytest

from xwalk.llm.base import LLMCapabilities, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.openai_compat import CAPABILITY_PROFILES, OpenAICompatClient

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def ok_body(content: str = '{"a": "x"}') -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        "model": "test-model",
    }


def client(handler, **kwargs) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="k",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_returns_the_message_content():
    llm = client(lambda r: httpx.Response(200, json=ok_body()))
    assert (await llm.complete(LLMRequest(system="s", user="u"))).text == '{"a": "x"}'


async def test_sends_system_and_user_messages():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    await client(handler).complete(LLMRequest(system="SYS", user="USR"))
    assert seen["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


async def test_sends_the_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_body())

    await client(handler).complete(LLMRequest(system="", user="u"))
    assert seen["auth"] == "Bearer k"


async def test_records_usage():
    llm = client(lambda r: httpx.Response(200, json=ok_body()))
    usage = (await llm.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (11, 3, 1)


async def test_missing_usage_block_is_not_an_error():
    body = ok_body()
    del body["usage"]
    llm = client(lambda r: httpx.Response(200, json=body))
    assert (await llm.complete(LLMRequest(system="", user="u"))).usage.calls == 1


async def test_requests_structured_output_only_when_declared():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    without = client(handler, capabilities=LLMCapabilities(json_schema=False))
    await without.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert "response_format" not in seen[-1]

    with_ = client(handler, capabilities=LLMCapabilities(json_schema=True))
    await with_.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert seen[-1]["response_format"]["type"] == "json_schema"


async def test_strict_flag_follows_the_declared_capability():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    llm = client(handler, capabilities=LLMCapabilities(json_schema=True, strict_schema=True))
    await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert seen[-1]["response_format"]["json_schema"]["strict"] is True


async def test_falls_back_to_prompt_only_json_when_the_provider_rejects_the_format():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(
                400, json={"error": {"message": "response_format is not supported"}}
            )
        return httpx.Response(200, json=ok_body())

    llm = client(handler, capabilities=LLMCapabilities(json_schema=True))
    response = await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert response.text == '{"a": "x"}'
    assert response.structured is False
    assert len(calls) == 2


async def test_the_fallback_is_remembered_for_later_calls():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return httpx.Response(200, json=ok_body())

    llm = client(handler, capabilities=LLMCapabilities(json_schema=True))
    await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    await llm.complete(LLMRequest(system="", user="u", schema=SCHEMA))
    assert sum("response_format" in c for c in calls) == 1


async def test_retries_on_429_then_succeeds():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=ok_body())

    llm = client(lambda r: handler(r), backoff_base=0.0)
    assert (await llm.complete(LLMRequest(system="", user="u"))).text == '{"a": "x"}'
    assert state["n"] == 2


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
async def test_retryable_statuses_eventually_raise_retryable(status):
    llm = client(lambda r: httpx.Response(status, json={}), max_retries=2, backoff_base=0.0)
    with pytest.raises(LLMRetryableError):
        await llm.complete(LLMRequest(system="", user="u"))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_non_retryable_statuses_raise_fatal_immediately(status):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(LLMFatalError):
        await client(handler, backoff_base=0.0).complete(LLMRequest(system="", user="u"))
    assert calls["n"] == 1


def retry_after_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, headers={"retry-after": "7"}, json={})


async def test_honours_retry_after_when_declared():
    """`max_retries=0` so the header is read but never slept on — the test must not
    actually wait 7 seconds to assert that it parsed 7 seconds."""
    declared = client(
        retry_after_handler,
        capabilities=LLMCapabilities(native_retry_after=True),
        max_retries=0,
        backoff_base=0.0,
    )
    with pytest.raises(LLMRetryableError) as exc:
        await declared.complete(LLMRequest(system="", user="u"))
    assert exc.value.retry_after == 7.0


async def test_ignores_retry_after_when_not_declared():
    undeclared = client(
        retry_after_handler,
        capabilities=LLMCapabilities(native_retry_after=False),
        max_retries=0,
        backoff_base=0.0,
    )
    with pytest.raises(LLMRetryableError) as exc:
        await undeclared.complete(LLMRequest(system="", user="u"))
    assert exc.value.retry_after is None


async def test_a_connection_error_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMRetryableError):
        await client(handler, max_retries=1, backoff_base=0.0).complete(
            LLMRequest(system="", user="u")
        )


async def test_an_empty_choices_array_is_fatal():
    llm = client(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMFatalError, match="no choices"):
        await llm.complete(LLMRequest(system="", user="u"))


def test_profiles_exist_for_the_endpoints_we_document():
    for name in ("openai", "anthropic-compat", "google-compat", "vllm", "unknown"):
        assert name in CAPABILITY_PROFILES


def test_the_unknown_profile_claims_nothing():
    caps = CAPABILITY_PROFILES["unknown"]
    assert not any(
        [
            caps.json_schema,
            caps.strict_schema,
            caps.usage_reporting,
            caps.seed,
            caps.native_retry_after,
        ]
    )


def test_fingerprint_changes_with_the_model():
    a = client(lambda r: httpx.Response(200, json=ok_body()))
    b = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="other-model",
        api_key="k",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    assert a.fingerprint != b.fingerprint


def test_fingerprint_ignores_the_api_key():
    """Rotating a key must not invalidate a resumable run — and must not leak into it."""
    a = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        api_key="key-one",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    b = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        model="m",
        api_key="key-two",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=ok_body())),
    )
    assert a.fingerprint == b.fingerprint


@pytest.mark.integration
async def test_against_a_real_provider():
    key = os.environ.get("XWALK_TEST_API_KEY")
    base = os.environ.get("XWALK_TEST_BASE_URL")
    model = os.environ.get("XWALK_TEST_MODEL")
    if not (key and base and model):
        pytest.skip("set XWALK_TEST_API_KEY, XWALK_TEST_BASE_URL, XWALK_TEST_MODEL")
    llm = OpenAICompatClient(base_url=base, model=model, api_key=key, profile="openai")
    response = await llm.complete(
        LLMRequest(
            system="Reply with JSON only.",
            user='Return exactly {"a": "x"}',
            schema=SCHEMA,
            max_tokens=64,
        )
    )
    from xwalk.llm.parsing import parse_json_object

    assert parse_json_object(response.text)["a"] == "x"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_openai_compat.py -v -m "not integration"`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.llm.openai_compat'`.

- [ ] **Step 3: Write `src/xwalk/llm/openai_compat.py`**

```python
"""One well-built OpenAI-compatible client, with declared capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

from xwalk.fingerprint import hash_value
from xwalk.llm.base import (
    LLMCapabilities,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
)
from xwalk.records import Usage

ADAPTER_VERSION = 1

RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

CAPABILITY_PROFILES: Mapping[str, LLMCapabilities] = {
    # Full OpenAI Chat Completions.
    "openai": LLMCapabilities(
        json_schema=True,
        strict_schema=True,
        usage_reporting=True,
        seed=True,
        native_retry_after=True,
    ),
    # Anthropic's OpenAI compatibility layer is documented as primarily for testing, and
    # strict schema enforcement may be ignored — so we do not ask for it.
    "anthropic-compat": LLMCapabilities(
        json_schema=False,
        strict_schema=False,
        usage_reporting=True,
        seed=False,
        native_retry_after=True,
    ),
    # Gemini's OpenAI-compatible endpoint accepts json_schema but is not strict.
    "google-compat": LLMCapabilities(
        json_schema=True,
        strict_schema=False,
        usage_reporting=True,
        seed=False,
        native_retry_after=False,
    ),
    # vLLM / SGLang with guided decoding.
    "vllm": LLMCapabilities(
        json_schema=True,
        strict_schema=True,
        usage_reporting=True,
        seed=True,
        native_retry_after=False,
    ),
    # The safe default: ask for nothing, validate everything.
    "unknown": LLMCapabilities(),
}


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        return str(error.get("message", error))[:500]
    return str(error)[:500]


class OpenAICompatClient:
    """Chat Completions over httpx, with backoff and structured-output fallback."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        capabilities: LLMCapabilities | None = None,
        profile: str = "unknown",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if profile not in CAPABILITY_PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; choose from {sorted(CAPABILITY_PROFILES)}"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._profile = profile
        self._capabilities = capabilities or CAPABILITY_PROFILES[profile]
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._extra_headers = dict(extra_headers or {})
        self._structured_disabled = False
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    @property
    def fingerprint(self) -> str:
        # The API key is deliberately absent: rotating a key must not invalidate a
        # resumable run, and a secret must never reach a manifest on disk.
        return hash_value(
            {
                "adapter": "openai_compat",
                "adapter_version": ADAPTER_VERSION,
                "base_url": self._base_url,
                "model": self._model,
                "profile": self._profile,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        )

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self._max_tokens,
        }
        if structured and request.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": dict(request.schema),
                    "strict": self._capabilities.strict_schema,
                },
            }
        if request.seed is not None and self._capabilities.seed:
            body["seed"] = request.seed
        body.update(request.extra)
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self._extra_headers}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        return await self._client.post(
            f"{self._base_url}/chat/completions", json=body, headers=self._headers()
        )

    def _retry_after(self, response: httpx.Response) -> float | None:
        if not self._capabilities.native_retry_after:
            return None
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        structured = (
            request.schema is not None
            and self._capabilities.json_schema
            and not self._structured_disabled
        )

        last_error = "unknown"
        last_retry_after: float | None = None
        for attempt in range(self._max_retries + 1):
            body = self._body(request, structured=structured)
            try:
                response = await self._post(body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, structured=structured)

                message = _error_message(response)
                if (
                    structured
                    and response.status_code in (400, 404, 422)
                    and "response_format" in message.lower()
                ):
                    # The provider claimed json_schema support and rejected it anyway.
                    # Drop to prompt-only JSON and remember, so we ask once, not per call.
                    self._structured_disabled = True
                    structured = False
                    continue
                if response.status_code not in RETRYABLE_STATUSES:
                    raise LLMFatalError(f"HTTP {response.status_code}: {message}")
                last_error = f"HTTP {response.status_code}: {message}"
                last_retry_after = self._retry_after(response)

            if attempt < self._max_retries:
                delay = last_retry_after or min(
                    self._backoff_cap, self._backoff_base * (2**attempt)
                )
                if delay:
                    await asyncio.sleep(delay)

        raise LLMRetryableError(
            f"exhausted {self._max_retries} retries: {last_error}",
            retry_after=last_retry_after,
        )

    def _parse(self, response: httpx.Response, *, structured: bool) -> LLMResponse:
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise LLMFatalError("provider returned no choices")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        raw_usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                completion_tokens=int(raw_usage.get("completion_tokens", 0)),
                calls=1,
            ),
            model=str(payload.get("model", self._model)),
            structured=structured,
            finish_reason=choices[0].get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_openai_compat.py -v -m "not integration"`
Expected: 30 passed (20 test functions, two of which parametrise status codes); the integration test is deselected.

Then confirm the integration test skips cleanly rather than erroring:
Run: `python -m pytest tests/test_openai_compat.py -v -m integration`
Expected: 1 skipped.

- [ ] **Step 5: Export it and re-run everything**

Add `OpenAICompatClient` and `CAPABILITY_PROFILES` to `src/xwalk/llm/__init__.py`'s imports and `__all__`.

Run: `python -m pytest -q -m "not integration" && python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/xwalk/llm tests/test_openai_compat.py
git commit -m "feat: OpenAI-compatible client with capability profiles and backoff"
```

---

## Task 9: Opaque candidate keys and exact resolution

**Files:**
- Create: `src/xwalk/stages/__init__.py`, `src/xwalk/stages/keying.py`
- Test: `tests/test_keying.py`

**Interfaces:**
- Consumes: `Candidate`, `Record` (Task 1), `TemplateSet` (Task 3).
- Produces: `Resolution` (StrEnum-like `Enum` of `exact_key`, `abstain`, `unresolved`, `legacy_exact_id`, `legacy_fuzzy`, `legacy_rank`); `ResolvedChoice(record_id: str | None, resolution: Resolution, raw: str | None)`; `KeyedCandidates(order, by_key, issued, rendered)`; `assign_keys(candidates, templates) -> KeyedCandidates`; `resolve_key(raw, keyed, *, legacy=False) -> ResolvedChoice`. Tasks 11, 12, 13, 17 consume these.

**This is the highest-stakes task in Phase 1.** The paper repo's resolver (`src/pipeline.py:67`) ends in a branch that treats a bare integer as a 1-based candidate rank. With numeric target IDs — `NCBIGene:3` — a hallucinated ID that is *not* in the candidate set falls through and **silently becomes a different, real mapping**. Opaque keys exist to make that impossible.

- [ ] **Step 1: Write the failing test**

Create `tests/test_keying.py`:

```python
import pytest

from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import Resolution, assign_keys, resolve_key
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(query="", context="", doc="", candidate="ID: {{ id }}\nLabel: {{ label }}")


def candidate(record_id: str, label: str, score: float = 0.5) -> Candidate:
    return Candidate(
        record=Record(id=record_id, fields={"label": label}),
        fused_score=score,
        evidence=(RetrievalHit(record_id=record_id, retriever="bm25", raw_score=1.0, rank=1),),
    )


@pytest.fixture
def keyed():
    return assign_keys(
        [candidate("NCBIGene:3", "A2MP1"), candidate("NCBIGene:7", "AANAT")], TEMPLATES
    )


def test_keys_are_one_based_and_zero_padded(keyed):
    assert keyed.order == ("C01", "C02")


def test_key_width_grows_past_ninety_nine():
    many = [candidate(f"T{i}", f"L{i}") for i in range(120)]
    keyed = assign_keys(many, TEMPLATES)
    assert keyed.order[0] == "C001" and keyed.order[-1] == "C120"


def test_rendered_block_shows_the_key_before_the_record(keyed):
    assert keyed.rendered.startswith("[C01] ID: NCBIGene:3")


def test_rendered_block_contains_every_candidate(keyed):
    assert "[C02]" in keyed.rendered and "AANAT" in keyed.rendered


def test_issued_maps_keys_to_record_ids(keyed):
    assert keyed.issued == {"C01": "NCBIGene:3", "C02": "NCBIGene:7"}


def test_empty_candidate_list_produces_no_keys():
    keyed = assign_keys([], TEMPLATES)
    assert keyed.order == () and keyed.rendered == ""


# --- resolution: the happy paths -------------------------------------------------


def test_resolves_an_issued_key(keyed):
    choice = resolve_key("C01", keyed)
    assert choice.record_id == "NCBIGene:3"
    assert choice.resolution is Resolution.EXACT_KEY


def test_tolerates_surrounding_whitespace_and_brackets(keyed):
    for raw in [" C01 ", "[C01]", "\nC01\n", '"C01"']:
        assert resolve_key(raw, keyed).record_id == "NCBIGene:3"


def test_key_case_is_normalised(keyed):
    """Safe here and only here: the key space is issued by us, so c01 and C01 cannot
    denote two different records. Identifier case-folding is a different matter."""
    assert resolve_key("c01", keyed).record_id == "NCBIGene:3"


@pytest.mark.parametrize("raw", [None, "null", "NULL", "none", "", "  ", "no_match", "NO MATCH"])
def test_abstention_is_recognised(raw, keyed):
    choice = resolve_key(raw, keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.ABSTAIN


# --- resolution: the adversarial cases opaque keys exist to prevent ---------------


def test_a_hallucinated_numeric_id_never_becomes_a_rank(keyed):
    """The exact failure mode of the paper repo's resolver: '3' must not mean C03,
    and must not become NCBIGene:3 either."""
    choice = resolve_key("3", keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_a_raw_target_id_does_not_resolve(keyed):
    choice = resolve_key("NCBIGene:3", keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_an_id_differing_only_by_case_does_not_resolve(keyed):
    assert resolve_key("ncbigene:3", keyed).resolution is Resolution.UNRESOLVED


def test_a_key_not_issued_this_attempt_does_not_resolve(keyed):
    assert resolve_key("C09", keyed).resolution is Resolution.UNRESOLVED


def test_a_prefix_stripped_id_does_not_resolve(keyed):
    assert resolve_key("3", keyed).record_id is None


def test_an_id_from_another_namespace_does_not_resolve(keyed):
    """The spec's third adversarial case. `CHEBI:3` shares its local part with the
    candidate `NCBIGene:3`, which is exactly the collision CURIE-prefix stripping
    creates. Strict mode must not see them as the same entity."""
    choice = resolve_key("CHEBI:3", keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_legacy_mode_also_refuses_a_foreign_namespace_suffix_collision(keyed):
    """Legacy mode keeps a suffix map for reproducing prior work. It must key on the
    bare suffix only, never accept a *different* namespace that happens to end in it —
    otherwise `CHEBI:3` silently becomes the real record `NCBIGene:3`."""
    choice = resolve_key("CHEBI:3", keyed, legacy=True)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_a_key_issued_by_a_previous_attempt_does_not_resolve():
    """Keys are issued per attempt. A five-candidate attempt issues C05; if the next
    attempt has two candidates, C05 must not resolve against it."""
    wide = assign_keys([candidate(f"T{i}", f"L{i}") for i in range(5)], TEMPLATES)
    narrow = assign_keys([candidate("T0", "L0"), candidate("T1", "L1")], TEMPLATES)
    assert resolve_key("C05", wide).record_id == "T4"
    assert resolve_key("C05", narrow).resolution is Resolution.UNRESOLVED


def test_prose_does_not_resolve(keyed):
    assert resolve_key("I think it is C01, the first one", keyed).resolution is (
        Resolution.UNRESOLVED
    )


def test_the_raw_answer_is_always_preserved_for_the_trace(keyed):
    assert resolve_key("banana", keyed).raw == "banana"


# --- legacy mode -----------------------------------------------------------------


def test_legacy_mode_accepts_an_exact_record_id(keyed):
    choice = resolve_key("NCBIGene:3", keyed, legacy=True)
    assert choice.record_id == "NCBIGene:3"
    assert choice.resolution is Resolution.LEGACY_EXACT_ID


def test_legacy_mode_accepts_a_case_insensitive_id_but_flags_it(keyed):
    choice = resolve_key("ncbigene:3", keyed, legacy=True)
    assert choice.record_id == "NCBIGene:3"
    assert choice.resolution is Resolution.LEGACY_FUZZY


def test_legacy_mode_accepts_a_bare_rank_but_flags_it(keyed):
    choice = resolve_key("2", keyed, legacy=True)
    assert choice.record_id == "NCBIGene:7"
    assert choice.resolution is Resolution.LEGACY_RANK


def test_legacy_mode_still_prefers_an_exact_key(keyed):
    assert resolve_key("C02", keyed, legacy=True).resolution is Resolution.EXACT_KEY


def test_legacy_mode_still_fails_on_a_genuinely_unknown_id(keyed):
    assert resolve_key("CHEBI:99999", keyed, legacy=True).resolution is Resolution.UNRESOLVED


def test_legacy_rank_out_of_range_does_not_resolve(keyed):
    assert resolve_key("99", keyed, legacy=True).resolution is Resolution.UNRESOLVED


def test_every_non_exact_resolution_is_distinguishable_from_exact(keyed):
    """Task 14 relies on this: anything other than EXACT_KEY or ABSTAIN forces review."""
    fuzzy = {Resolution.LEGACY_EXACT_ID, Resolution.LEGACY_FUZZY, Resolution.LEGACY_RANK}
    assert Resolution.EXACT_KEY not in fuzzy and Resolution.ABSTAIN not in fuzzy
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_keying.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.stages'`.

- [ ] **Step 3: Write `src/xwalk/stages/keying.py`**

```python
"""Opaque candidate keys.

Candidates are shown to the model under temporary keys assigned per attempt. The model
returns a key or null. Resolution is an exact dictionary lookup against the keys issued
*for that attempt*; anything else is UNRESOLVED.

This exists because heuristic ID resolution is unsafe as a default. Its worst branch —
treating a bare integer as a candidate rank — turns a hallucinated ID into a different,
real mapping whenever target IDs are numeric. Opaque keys also remove the sentinel
muddle: `null` is unambiguous where "-1" and "0" are values some scheme legitimately uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from xwalk.records import Candidate
from xwalk.templates import TemplateSet

_ABSTAIN_TOKENS = frozenset(
    {"", "null", "none", "nil", "no_match", "nomatch", "no match", "n/a", "na"}
)
_TRIM = " \t\r\n\"'`[]().,;:"


class Resolution(Enum):
    """How a raw model answer became (or failed to become) a record id."""

    EXACT_KEY = "exact_key"
    ABSTAIN = "abstain"
    UNRESOLVED = "unresolved"
    LEGACY_EXACT_ID = "legacy_exact_id"
    LEGACY_FUZZY = "legacy_fuzzy"
    LEGACY_RANK = "legacy_rank"


@dataclass(frozen=True)
class ResolvedChoice:
    record_id: str | None
    resolution: Resolution
    raw: str | None


@dataclass(frozen=True)
class KeyedCandidates:
    order: tuple[str, ...]
    by_key: Mapping[str, Candidate]
    issued: Mapping[str, str]
    rendered: str

    def __len__(self) -> int:
        return len(self.order)


def assign_keys(candidates: Sequence[Candidate], templates: TemplateSet) -> KeyedCandidates:
    """Assign C01, C02, ... in the order given, and render the candidate block."""
    if not candidates:
        return KeyedCandidates(order=(), by_key={}, issued={}, rendered="")

    width = max(2, len(str(len(candidates))))
    order: list[str] = []
    by_key: dict[str, Candidate] = {}
    issued: dict[str, str] = {}
    blocks: list[str] = []

    for index, candidate in enumerate(candidates, start=1):
        key = f"C{index:0{width}d}"
        order.append(key)
        by_key[key] = candidate
        issued[key] = candidate.id
        blocks.append(f"[{key}] {templates.render_candidate(candidate.record)}")

    return KeyedCandidates(
        order=tuple(order), by_key=by_key, issued=issued, rendered="\n\n".join(blocks)
    )


def _normalise(raw: str) -> str:
    return raw.strip().strip(_TRIM).strip()


def resolve_key(
    raw: str | None,
    keyed: KeyedCandidates,
    *,
    legacy: bool = False,
) -> ResolvedChoice:
    """Turn the model's answer into a record id, or refuse.

    `legacy=True` re-enables heuristic ID resolution for reproducing prior work. Every
    non-exact path is tagged so Task 14 can force NEEDS_REVIEW on it.
    """
    if raw is None:
        return ResolvedChoice(None, Resolution.ABSTAIN, raw)

    text = _normalise(raw)
    if text.lower() in _ABSTAIN_TOKENS:
        return ResolvedChoice(None, Resolution.ABSTAIN, raw)

    # The key space is ours, so upper-casing cannot merge two distinct records.
    upper = text.upper()
    if upper in keyed.issued:
        return ResolvedChoice(keyed.issued[upper], Resolution.EXACT_KEY, raw)

    if not legacy:
        return ResolvedChoice(None, Resolution.UNRESOLVED, raw)

    ids = [keyed.issued[key] for key in keyed.order]
    if text in ids:
        return ResolvedChoice(text, Resolution.LEGACY_EXACT_ID, raw)

    lowered = {rid.lower(): rid for rid in ids}
    if text.lower() in lowered:
        return ResolvedChoice(lowered[text.lower()], Resolution.LEGACY_FUZZY, raw)

    suffixes = {rid.rsplit(":", 1)[-1]: rid for rid in ids}
    if text in suffixes:
        return ResolvedChoice(suffixes[text], Resolution.LEGACY_FUZZY, raw)

    if text.isdigit():
        rank = int(text)
        if 1 <= rank <= len(ids):
            return ResolvedChoice(ids[rank - 1], Resolution.LEGACY_RANK, raw)

    return ResolvedChoice(None, Resolution.UNRESOLVED, raw)
```

Note the ordering inside legacy mode: exact ID, then case-insensitive, then suffix, then rank — and the rank branch is reached only after every identifier interpretation has failed. Outside legacy mode it does not exist at all.

`src/xwalk/stages/__init__.py`:

```python
"""Stages: keying, selection, gating, rewriting, proposal routing."""
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_keying.py -v`
Expected: 34 passed (24 test functions, one of which parametrises 8 abstention tokens, plus the 3 adversarial cases below). The `test_legacy_rank_out_of_range_does_not_resolve` case with `"99"` and a suffix map containing `"3"` and `"7"` must reach the rank branch and fail on range — verify that.

The spec names four adversarial resolution cases that opaque keys exist to prevent, and each must reach `UNRESOLVED`, never a real record: a numeric hallucinated ID, an ID differing only by case, **an ID valid in another namespace**, and a key not issued this attempt. All four are now covered, the third by `test_an_id_from_another_namespace_does_not_resolve` and its legacy-mode twin. The legacy suffix map keys on the bare local part, so `"CHEBI:3"` never matches `"3"` — verify that branch rather than assuming it.

- [ ] **Step 5: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/stages tests/test_keying.py
git commit -m "feat: opaque candidate keys with exact-only resolution"
```

---

## Task 10: Prompt skeletons, slots, and contract validation

**Files:**
- Create: `src/xwalk/prompts/__init__.py`, `src/xwalk/prompts/contract.py`
- Create: `src/xwalk/prompts/base/select.j2`, `score.j2`, `verify.j2`, `rewrite.j2`
- Create: `examples/chemistry/slots.yaml`
- Modify: `pyproject.toml` (ship `.j2` files in the wheel)
- Test: `tests/test_prompt_contract.py`

**Interfaces:**
- Consumes: `Record` (Task 1), `hash_value` (Task 2).
- Produces: `RubricRow`; `PromptSlots` (pydantic model, with `load_slots(path) -> PromptSlots`); `PromptSet.from_slots(slots, *, base_dir=None) -> PromptSet` with `render_select(...)`, `render_score(...)`, `render_verify(...)`, `render_rewrite(...)`, `fingerprint`; `SELECT_SCHEMA`, `SCORE_SCHEMA`, `VERIFY_SCHEMA`, `REWRITE_SCHEMA`; `BASE_DIR` (the shipped skeleton directory); `validate_contract(prompts) -> None` raising `ContractError`. Tasks 11, 12, 13 consume these; Phase 2's optimizer rewrites only the slots.

**Design note:** the skeleton owns the fixed structure — role line, input blocks, opaque-key candidate list, rubric table, hard rules, JSON output contract. Domain content lives only in the slots file. A bad draft can therefore produce a poor rubric but never a broken prompt.

- [ ] **Step 1: Write `src/xwalk/prompts/base/select.j2`**

```jinja
You are matching a {{ slots.entity_noun }} to at most one {{ slots.target_noun }}.
Domain: {{ slots.domain_brief }}

## Source record
{% for key, value in source_fields.items() %}- {{ key }}: {{ value }}
{% endfor %}
{% if context %}
## Context
{{ context }}
{% endif %}
## Candidates
{{ candidate_block }}
{% if slots.disambiguation_steps %}
## How to disambiguate
{{ slots.disambiguation_steps }}
{% endif %}
{% if slots.hard_rules %}
## Hard rules
{% for rule in slots.hard_rules %}- {{ rule }}
{% endfor %}
{% endif %}
## Rules
- Choose the single best candidate, or abstain.
- Answer with the bracketed key exactly as shown (for example C01), or null to abstain.
- Never invent a key. Never answer with an identifier, a name, or a position number.
- Abstain when no candidate is the same entity as the source record.

## Output
Return only a JSON object of exactly this shape, where `chosen_key` is one of the keys
above or null, and `confidence_score` is between 0.0 and 1.0:
{"chosen_key": "C01", "confidence_score": 0.9, "explanation": "one sentence"}
```

The example must be **literal, parseable JSON**. `contract.py` parses whatever follows
`## Output` and asserts it has the schema's required keys, so a pseudo-JSON sketch like
`{"chosen_key": "C01" or null}` fails validation outright. The variability belongs in the
prose above the example, never inside it.

- [ ] **Step 2: Write `src/xwalk/prompts/base/score.j2`**

The gate scores against the **full source record and context**, never the retrieval query — the query is deliberately lossy, and for a multi-field source it may omit the very fields needed to disambiguate.

```jinja
You are judging a proposed match between a {{ slots.entity_noun }} and a {{ slots.target_noun }}.
Domain: {{ slots.domain_brief }}

## Source record
{% for key, value in source_fields.items() %}- {{ key }}: {{ value }}
{% endfor %}
{% if context %}
## Context
{{ context }}
{% endif %}
## Proposed match
{{ chosen_block }}
{% if other_candidates %}
## Other candidates that were available
{{ other_candidates }}
{% endif %}
## Rubric
| Score | Name | Applies when |
|---|---|---|
{% for row in slots.rubric %}| {{ row.score }} | {{ row.name }} | {{ row.when }}{% if row.example %} (e.g. {{ row.example }}){% endif %} |
{% endfor %}
{% if slots.hard_rules %}
## Hard rules
{% for rule in slots.hard_rules %}- {{ rule }}
{% endfor %}
{% endif %}
## Rules
- Judge the proposed match against the whole source record above, not against any search string.
- If the score is below {{ review_floor }}, you may propose better options.
- Propose a candidate key only if it appears in "Other candidates that were available".
- Propose a search query only for an entity that is not present in any listed candidate.

## Output
Return only a JSON object of exactly this shape. `confidence_score` is between 0.0 and
1.0; both lists may be empty:
{"confidence_score": 0.7, "explanation": "one sentence", "better_candidate_keys": ["C03"], "better_queries": ["alternative search string"]}
```

- [ ] **Step 3: Write `src/xwalk/prompts/base/verify.j2`**

Verification returns an independent **verdict**, not a second number to be `min()`-ed with the first. Two uncalibrated scores cannot distinguish agreement-with-different-numbers from genuine disagreement.

```jinja
You are independently verifying a proposed match. Do not assume the proposal is correct.
Domain: {{ slots.domain_brief }}

## Source record
{% for key, value in source_fields.items() %}- {{ key }}: {{ value }}
{% endfor %}
{% if context %}
## Context
{{ context }}
{% endif %}
## Proposed match
{{ chosen_block }}
{% if other_candidates %}
## Other candidates
{{ other_candidates }}
{% endif %}
{% if slots.hard_rules %}
## Hard rules
{% for rule in slots.hard_rules %}- {{ rule }}
{% endfor %}
{% endif %}
## Rules
- Answer "support" if the proposed match is the same entity as the source record.
- Answer "disagree" if a different listed candidate is the correct match, and name its key.
- Answer "no_match" if no listed candidate is the correct match.

## Output
Return only a JSON object of exactly this shape. `decision` is one of support, disagree,
or no_match; `preferred_key` is one of the keys above or null:
{"decision": "support", "preferred_key": null, "confidence_score": 0.8, "explanation": "one sentence"}
```

- [ ] **Step 4: Write `src/xwalk/prompts/base/rewrite.j2`**

```jinja
You are improving a search query used to find a {{ slots.target_noun }} for a
{{ slots.entity_noun }}.
Domain: {{ slots.domain_brief }}

## Source record
{% for key, value in source_fields.items() %}- {{ key }}: {{ value }}
{% endfor %}
{% if context %}
## Context
{{ context }}
{% endif %}
## Queries already tried
{% for q in previous_queries %}- {{ q }}
{% endfor %}
{% if best_candidates %}
## Best candidates those queries returned
{{ best_candidates }}
{% endif %}
## Rules
- Propose up to {{ max_queries }} new search queries that are materially different from
  the queries already tried.
- Expand abbreviations, use the full formal name, or drop qualifiers that are not part of
  the entity name.
- Do not repeat a query that has already been tried.

## Output
Return only this JSON object:
{"queries": ["first alternative", "second alternative"], "explanation": "one sentence"}
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_prompt_contract.py`:

```python
import json

import pytest
import yaml

from xwalk.prompts.contract import (
    BASE_DIR,
    ContractError,
    PromptSet,
    PromptSlots,
    load_slots,
    validate_contract,
)


def skeleton_dir(tmp_path, **overrides: str):
    """Copy the shipped skeletons into a temp dir, replacing the named ones."""
    base = tmp_path / "base"
    base.mkdir()
    for name in ("select", "score", "verify", "rewrite"):
        text = overrides.get(name, (BASE_DIR / f"{name}.j2").read_text(encoding="utf-8"))
        (base / f"{name}.j2").write_text(text, encoding="utf-8")
    return base


SLOTS = PromptSlots(
    entity_noun="chemical entity mention",
    target_noun="ChEBI ontology term",
    domain_brief="biomedical chemistry nomenclature",
    rubric=[
        {
            "score": 1.0,
            "name": "Certain",
            "when": "exact match to label or synonym",
            "example": "garlic -> Garlic",
        },
        {
            "score": 0.9,
            "name": "High",
            "when": "normalized form or common abbreviation",
            "example": "ASA -> aspirin",
        },
        {
            "score": 0.6,
            "name": "Plausible",
            "when": "a specific instance of the candidate",
            "example": "Fuji apple -> apple",
        },
        {
            "score": 0.4,
            "name": "Speculative",
            "when": "related by broad category only",
            "example": "sugar -> carbohydrate",
        },
    ],
    hard_rules=["Any difference in a regulated substance's name or number means a distinct entity"],
)


def test_slots_reject_an_empty_rubric():
    with pytest.raises(ValueError, match="rubric"):
        PromptSlots(entity_noun="a", target_noun="b", domain_brief="c", rubric=[])


def test_slots_reject_a_score_outside_zero_to_one():
    with pytest.raises(ValueError, match="between 0 and 1"):
        PromptSlots(
            entity_noun="a",
            target_noun="b",
            domain_brief="c",
            rubric=[
                {"score": 1.5, "name": "x", "when": "y"},
                {"score": 0.5, "name": "z", "when": "w"},
            ],
        )


def test_slots_require_strictly_decreasing_scores():
    with pytest.raises(ValueError, match="decreasing"):
        PromptSlots(
            entity_noun="a",
            target_noun="b",
            domain_brief="c",
            rubric=[
                {"score": 0.5, "name": "x", "when": "y"},
                {"score": 0.9, "name": "z", "when": "w"},
            ],
        )


def test_slots_require_at_least_two_rubric_rows():
    with pytest.raises(ValueError, match="at least 2"):
        PromptSlots(
            entity_noun="a",
            target_noun="b",
            domain_brief="c",
            rubric=[{"score": 0.9, "name": "x", "when": "y"}],
        )


def test_load_slots_reads_yaml(tmp_path):
    path = tmp_path / "slots.yaml"
    path.write_text(yaml.safe_dump(json.loads(SLOTS.model_dump_json())), encoding="utf-8")
    assert load_slots(path).entity_noun == "chemical entity mention"


def test_select_prompt_contains_the_candidate_block():
    prompts = PromptSet.from_slots(SLOTS)
    rendered = prompts.render_select(
        source_fields={"mention": "glucose"},
        context="blood [glucose] levels",
        candidate_block="[C01] ID: CHEBI:17234 Label: glucose",
    )
    assert "[C01] ID: CHEBI:17234" in rendered


def test_select_prompt_forbids_answering_with_an_identifier():
    rendered = PromptSet.from_slots(SLOTS).render_select(
        source_fields={"mention": "x"}, context="", candidate_block="[C01] ID: a"
    )
    assert "Never answer with an identifier" in rendered


def test_select_prompt_omits_the_context_heading_when_there_is_no_context():
    rendered = PromptSet.from_slots(SLOTS).render_select(
        source_fields={"mention": "x"}, context="", candidate_block="[C01] ID: a"
    )
    assert "## Context" not in rendered


def test_score_prompt_shows_the_whole_source_record_not_a_query():
    rendered = PromptSet.from_slots(SLOTS).render_score(
        source_fields={"symbol": "TP53", "organism": "Homo sapiens"},
        context="",
        chosen_block="[C01] ID: NCBIGene:7157",
        other_candidates="[C02] ID: NCBIGene:22059",
        review_floor=0.4,
    )
    assert "symbol: TP53" in rendered and "organism: Homo sapiens" in rendered


def test_score_prompt_renders_every_rubric_row():
    rendered = PromptSet.from_slots(SLOTS).render_score(
        source_fields={"a": "b"},
        context="",
        chosen_block="x",
        other_candidates="",
        review_floor=0.4,
    )
    for row in SLOTS.rubric:
        assert row.name in rendered


def test_verify_prompt_offers_all_three_decisions():
    rendered = PromptSet.from_slots(SLOTS).render_verify(
        source_fields={"a": "b"}, context="", chosen_block="x", other_candidates=""
    )
    for decision in ("support", "disagree", "no_match"):
        assert decision in rendered


def test_rewrite_prompt_lists_previous_queries():
    rendered = PromptSet.from_slots(SLOTS).render_rewrite(
        source_fields={"a": "b"},
        context="",
        previous_queries=["glucose", "dextrose"],
        best_candidates="",
        max_queries=2,
    )
    assert "- glucose" in rendered and "- dextrose" in rendered


def test_hard_rules_appear_in_select_score_and_verify():
    prompts = PromptSet.from_slots(SLOTS)
    rule = SLOTS.hard_rules[0]
    assert rule in prompts.render_select(source_fields={}, context="", candidate_block="x")
    assert rule in prompts.render_score(
        source_fields={}, context="", chosen_block="x", other_candidates="", review_floor=0.4
    )
    assert rule in prompts.render_verify(
        source_fields={}, context="", chosen_block="x", other_candidates=""
    )


def test_the_declared_output_examples_parse_as_their_schemas():
    validate_contract(PromptSet.from_slots(SLOTS))  # raises on failure


def test_contract_validation_rejects_a_skeleton_missing_the_candidate_block(tmp_path):
    base = skeleton_dir(tmp_path, select="no blocks here\n")
    with pytest.raises(ContractError, match="candidate_block"):
        validate_contract(PromptSet.from_slots(SLOTS, base_dir=base))


def test_contract_validation_rejects_a_duplicated_input_block(tmp_path):
    """The spec requires every input block to appear *exactly once*. Rendering the
    candidate list twice would show the model two lists and is not merely untidy."""
    original = (BASE_DIR / "select.j2").read_text(encoding="utf-8")
    base = skeleton_dir(tmp_path, select=original + "\n## Candidates\n{{ candidate_block }}\n")
    with pytest.raises(ContractError, match="exactly once"):
        validate_contract(PromptSet.from_slots(SLOTS, base_dir=base))


def test_contract_validation_rejects_a_skeleton_that_lost_the_key_instruction(tmp_path):
    """Without 'Never answer with an identifier' the model may answer with a target ID,
    which resolves to UNRESOLVED_OUTPUT every time. An optimizer must not be able to
    delete it while leaving a prompt that still looks valid."""
    original = (BASE_DIR / "select.j2").read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in original.splitlines() if "Never answer with an identifier" not in line
    )
    base = skeleton_dir(tmp_path, select=stripped + "\n")
    with pytest.raises(ContractError, match="key instruction"):
        validate_contract(PromptSet.from_slots(SLOTS, base_dir=base))


def test_fingerprint_changes_when_a_slot_changes():
    a = PromptSet.from_slots(SLOTS)
    b = PromptSet.from_slots(SLOTS.model_copy(update={"domain_brief": "legal citations"}))
    assert a.fingerprint != b.fingerprint


def test_fingerprint_covers_the_skeleton_text_too(tmp_path):
    from xwalk.prompts.contract import BASE_DIR

    base = tmp_path / "base"
    base.mkdir()
    for name in ("select.j2", "score.j2", "verify.j2", "rewrite.j2"):
        (base / name).write_text((BASE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    (base / "select.j2").write_text(
        (base / "select.j2").read_text(encoding="utf-8") + "\nextra line\n", encoding="utf-8"
    )
    assert PromptSet.from_slots(SLOTS).fingerprint != (
        PromptSet.from_slots(SLOTS, base_dir=base).fingerprint
    )


def test_the_shipped_chemistry_example_validates():
    from pathlib import Path

    # Anchored to this file, not to the working directory the suite happens to run in.
    path = Path(__file__).resolve().parent.parent / "examples" / "chemistry" / "slots.yaml"
    validate_contract(PromptSet.from_slots(load_slots(path)))
```

- [ ] **Step 6: Write `src/xwalk/prompts/contract.py`**

```python
"""Prompt skeletons plus the contract that keeps them safe to modify.

The skeleton owns the fixed structure. Slots own domain content. Everything that could
break the machine-readable contract lives in the skeleton, so an LLM-authored or
optimizer-mutated slots file can produce a poor rubric but never a broken prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field, field_validator, model_validator

from xwalk.fingerprint import hash_value
from xwalk.llm.parsing import parse_json_object

BASE_DIR = Path(__file__).parent / "base"

SELECT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "chosen_key": {"type": ["string", "null"]},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
    },
    "required": ["chosen_key", "confidence_score", "explanation"],
    "additionalProperties": False,
}

SCORE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
        "better_candidate_keys": {"type": "array", "items": {"type": "string"}},
        "better_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confidence_score", "explanation"],
    "additionalProperties": False,
}

VERIFY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["support", "disagree", "no_match"]},
        "preferred_key": {"type": ["string", "null"]},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
    },
    "required": ["decision", "explanation"],
    "additionalProperties": False,
}

REWRITE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["queries"],
    "additionalProperties": False,
}


class ContractError(Exception):
    """A skeleton no longer satisfies the machine-readable contract."""


class RubricRow(BaseModel):
    score: float
    name: str
    when: str
    example: str = ""


class PromptSlots(BaseModel):
    entity_noun: str
    target_noun: str
    domain_brief: str
    rubric: list[RubricRow] = Field(min_length=1)
    hard_rules: list[str] = Field(default_factory=list)
    disambiguation_steps: str = ""

    @field_validator("rubric")
    @classmethod
    def _check_rubric(cls, rows: list[RubricRow]) -> list[RubricRow]:
        if len(rows) < 2:
            raise ValueError("rubric needs at least 2 rows to be usable")
        for row in rows:
            if not 0.0 <= row.score <= 1.0:
                raise ValueError(f"rubric score {row.score} must be between 0 and 1")
        scores = [row.score for row in rows]
        # Lengths differ by one on purpose: each row is compared with its successor.
        if any(b >= a for a, b in zip(scores, scores[1:], strict=False)):
            raise ValueError("rubric scores must be strictly decreasing as declared")
        return rows

    @model_validator(mode="after")
    def _check_nouns(self) -> PromptSlots:
        for name in ("entity_noun", "target_noun", "domain_brief"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        return self


def load_slots(path: str | Path) -> PromptSlots:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PromptSlots.model_validate(data)


@dataclass(frozen=True)
class PromptSet:
    slots: PromptSlots
    skeletons: Mapping[str, str]

    @classmethod
    def from_slots(cls, slots: PromptSlots, *, base_dir: Path | None = None) -> PromptSet:
        directory = base_dir or BASE_DIR
        skeletons = {
            name: (directory / f"{name}.j2").read_text(encoding="utf-8")
            for name in ("select", "score", "verify", "rewrite")
        }
        return cls(slots=slots, skeletons=skeletons)

    def _render(self, name: str, **variables: Any) -> str:
        env = Environment(
            loader=FileSystemLoader(str(BASE_DIR)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )
        template = env.from_string(self.skeletons[name])
        return template.render(slots=self.slots, **variables).strip()

    def render_select(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        candidate_block: str,
    ) -> str:
        return self._render(
            "select",
            source_fields=source_fields,
            context=context,
            candidate_block=candidate_block,
        )

    def render_score(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        chosen_block: str,
        other_candidates: str,
        review_floor: float,
    ) -> str:
        return self._render(
            "score",
            source_fields=source_fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
            review_floor=review_floor,
        )

    def render_verify(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        chosen_block: str,
        other_candidates: str,
    ) -> str:
        return self._render(
            "verify",
            source_fields=source_fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
        )

    def render_rewrite(
        self,
        *,
        source_fields: Mapping[str, Any],
        context: str,
        previous_queries: Sequence[str],
        best_candidates: str,
        max_queries: int,
    ) -> str:
        return self._render(
            "rewrite",
            source_fields=source_fields,
            context=context,
            previous_queries=list(previous_queries),
            best_candidates=best_candidates,
            max_queries=max_queries,
        )

    @property
    def fingerprint(self) -> str:
        return hash_value(
            {
                "slots": json.loads(self.slots.model_dump_json()),
                "skeletons": dict(sorted(self.skeletons.items())),
            }
        )


# Section headings that must be present, and present exactly once. A skeleton that
# renders "## Candidates" twice would show the model two candidate lists.
_REQUIRED_SECTIONS: Mapping[str, tuple[str, ...]] = {
    "select": ("## Source record", "## Candidates", "## Output"),
    "score": ("## Source record", "## Rubric", "## Output"),
    "verify": ("## Source record", "## Proposed match", "## Output"),
    "rewrite": ("## Source record", "## Output"),
}

# The instruction that makes opaque keys safe. If an optimizer deletes it, the model is
# free to answer with an identifier and every answer becomes UNRESOLVED_OUTPUT.
_KEY_INSTRUCTION = "Never answer with an identifier"

# Typed individually rather than as one heterogeneous dict: `dict(a="s", b=0.4, c=2)`
# infers `dict[str, object]`, and every use site then fails `mypy --strict` on arg-type.
_SOURCE_FIELDS: Mapping[str, Any] = {"mention": "glucose", "organism": "Homo sapiens"}
_CONTEXT = "blood [glucose] levels were elevated"
_CANDIDATE_BLOCK = "[C01] ID: T1 Label: glucose\n\n[C02] ID: T2 Label: fructose"
_CHOSEN_BLOCK = "[C01] ID: T1 Label: glucose"
_OTHER_CANDIDATES = "[C02] ID: T2 Label: fructose"
_REVIEW_FLOOR = 0.4
_PREVIOUS_QUERIES: Sequence[str] = ("glucose",)
_BEST_CANDIDATES = "[C01] ID: T1"
_MAX_QUERIES = 2

# Rendered inputs that must survive into each prompt, exactly once.
_REQUIRED_INPUTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "select": (("candidate_block", _CANDIDATE_BLOCK), ("context", _CONTEXT)),
    "score": (("chosen_block", _CHOSEN_BLOCK), ("context", _CONTEXT)),
    "verify": (("chosen_block", _CHOSEN_BLOCK), ("context", _CONTEXT)),
    "rewrite": (("context", _CONTEXT),),
}

_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "select": SELECT_SCHEMA,
    "score": SCORE_SCHEMA,
    "verify": VERIFY_SCHEMA,
    "rewrite": REWRITE_SCHEMA,
}


def validate_contract(prompts: PromptSet) -> None:
    """Render every skeleton and assert the machine-readable contract survived.

    Run this before anything is saved — a drafting model or an optimizer must not be
    able to ship a prompt whose output cannot be parsed, whose input blocks went
    missing or got duplicated, or whose key instruction was edited away.

    Order matters: the rendered-input check runs first so that a skeleton which dropped
    a block reports *that*, rather than a downstream missing-heading error.
    """
    rendered = {
        "select": prompts.render_select(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            candidate_block=_CANDIDATE_BLOCK,
        ),
        "score": prompts.render_score(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            chosen_block=_CHOSEN_BLOCK,
            other_candidates=_OTHER_CANDIDATES,
            review_floor=_REVIEW_FLOOR,
        ),
        "verify": prompts.render_verify(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            chosen_block=_CHOSEN_BLOCK,
            other_candidates=_OTHER_CANDIDATES,
        ),
        "rewrite": prompts.render_rewrite(
            source_fields=_SOURCE_FIELDS,
            context=_CONTEXT,
            previous_queries=_PREVIOUS_QUERIES,
            best_candidates=_BEST_CANDIDATES,
            max_queries=_MAX_QUERIES,
        ),
    }

    for name, text in rendered.items():
        for label, block in _REQUIRED_INPUTS[name]:
            count = text.count(block)
            if count != 1:
                raise ContractError(
                    f"{name} prompt rendered {label} {count} times; it must appear exactly once"
                )

        for section in _REQUIRED_SECTIONS[name]:
            count = text.count(section)
            if count != 1:
                raise ContractError(
                    f"{name} prompt contains section {section!r} {count} times; "
                    f"it must appear exactly once"
                )

        if name == "select" and _KEY_INSTRUCTION not in text:
            raise ContractError(
                "select prompt lost the key instruction "
                f"({_KEY_INSTRUCTION!r}); without it the model may answer with an identifier"
            )

        tail = text.split("## Output", 1)[1]
        try:
            example = parse_json_object(tail)
        except Exception as exc:  # noqa: BLE001
            raise ContractError(f"{name} output block is not a parseable example: {exc}") from exc

        required = _SCHEMAS[name].get("required", [])
        missing = [key for key in required if key not in example]
        if missing:
            raise ContractError(f"{name} output example is missing keys: {missing}")
```

`src/xwalk/prompts/__init__.py`:

```python
from xwalk.prompts.contract import (
    REWRITE_SCHEMA,
    SCORE_SCHEMA,
    SELECT_SCHEMA,
    VERIFY_SCHEMA,
    ContractError,
    PromptSet,
    PromptSlots,
    RubricRow,
    load_slots,
    validate_contract,
)

__all__ = [
    "REWRITE_SCHEMA",
    "SCORE_SCHEMA",
    "SELECT_SCHEMA",
    "VERIFY_SCHEMA",
    "ContractError",
    "PromptSet",
    "PromptSlots",
    "RubricRow",
    "load_slots",
    "validate_contract",
]
```

- [ ] **Step 7: Write `examples/chemistry/slots.yaml`**

```yaml
entity_noun: chemical entity mention
target_noun: ChEBI ontology term
domain_brief: biomedical chemistry nomenclature
rubric:
  - score: 1.0
    name: Certain
    when: exact case-insensitive match to the candidate label or one of its synonyms
    example: garlic -> Garlic
  - score: 0.9
    name: High
    when: a normalized form, plural, or common abbreviation of the candidate
    example: ASA -> acetylsalicylic acid
  - score: 0.6
    name: Plausible
    when: the source names a specific instance of the candidate class
    example: Fuji apple -> apple
  - score: 0.4
    name: Speculative
    when: related only by broad category
    example: sugar -> carbohydrate
hard_rules:
  - For regulated substances, any difference in name or number means distinct entities
  - A salt, ester, or hydrate is a distinct entity from its parent compound
```

- [ ] **Step 8: Ship the `.j2` files in the wheel**

Add to `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/xwalk/prompts/base" = "xwalk/prompts/base"
```

If hatchling already includes package data under `src/xwalk`, verify with `python -m pip install -e . && python -c "from xwalk.prompts.contract import BASE_DIR; print(sorted(p.name for p in BASE_DIR.iterdir()))"` — expect all four `.j2` names.

- [ ] **Step 9: Run the tests**

Run: `python -m pytest tests/test_prompt_contract.py -v`
Expected: 20 passed.

- [ ] **Step 10: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/prompts examples pyproject.toml tests/test_prompt_contract.py
git commit -m "feat: contract-safe prompt skeletons with validated domain slots"
```

---

## Task 11: The selector stage and selector budget

**Files:**
- Create: `src/xwalk/stages/select.py`
- Test: `tests/test_select.py`

**Interfaces:**
- Consumes: `Candidate`, `Usage` (Task 1), `TemplateSet` (Task 3), `LLMClient`/`LLMRequest`/`ParseError`/`parse_json_object` (Task 7), `assign_keys`/`resolve_key`/`KeyedCandidates`/`ResolvedChoice`/`Resolution` (Task 9), `PromptSet`/`SELECT_SCHEMA` (Task 10).
- Produces: `SelectorPolicy(max_candidates: int = 30, max_candidate_tokens: int = 8_000)`; `apply_budget(candidates, policy) -> tuple[list[Candidate], int]`; `SelectionOutcome(choice, confidence, explanation, raw, keyed, truncated, usage, error)`; `Selector(llm, prompts, templates, *, policy=None, legacy_id_resolution=False)` with `async select(source_record, context, candidates) -> SelectionOutcome`. Task 17 consumes this.

**Design note:** the budget is **not** a retrieval-depth setting. Retrieval depth `k` belongs to each retriever; the budget governs how much of the *fused* list reaches the model. Truncation is deterministic and the count is recorded, because Phase 2's ceiling diagnostic must separate "never retrieved" from "retrieved but cut by the budget" — those have different fixes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_select.py`:

```python
import json

import pytest

from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import Resolution
from xwalk.stages.select import Selector, SelectorPolicy, apply_budget
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="", doc="", candidate="ID: {{ id }} Label: {{ label }}"
)
SLOTS = PromptSlots(
    entity_noun="mention",
    target_noun="term",
    domain_brief="test domain",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact"},
        {"score": 0.4, "name": "Weak", "when": "vague"},
    ],
)
PROMPTS = PromptSet.from_slots(SLOTS)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


def cand(record_id: str, label: str, score: float) -> Candidate:
    return Candidate(
        record=Record(id=record_id, fields={"label": label}),
        fused_score=score,
        evidence=(RetrievalHit(record_id=record_id, retriever="bm25", raw_score=1.0, rank=1),),
    )


CANDIDATES = [cand("T1", "glucose", 0.9), cand("T2", "fructose", 0.5)]


def reply(**kwargs) -> str:
    body = {"chosen_key": None, "confidence_score": 0.0, "explanation": ""}
    body.update(kwargs)
    return json.dumps(body)


# --- budget ---------------------------------------------------------------------


def test_budget_keeps_everything_when_under_the_cap():
    kept, dropped = apply_budget(CANDIDATES, SelectorPolicy())
    assert len(kept) == 2 and dropped == 0


def test_budget_truncates_to_max_candidates_and_reports_the_count():
    many = [cand(f"T{i}", f"L{i}", 1.0 - i / 100) for i in range(50)]
    kept, dropped = apply_budget(many, SelectorPolicy(max_candidates=10))
    assert len(kept) == 10 and dropped == 40


def test_budget_keeps_the_highest_scoring_candidates():
    many = [cand(f"T{i}", f"L{i}", i / 100) for i in range(10)]
    kept, _ = apply_budget(
        sorted(many, key=lambda c: -c.fused_score), SelectorPolicy(max_candidates=3)
    )
    assert [c.id for c in kept] == ["T9", "T8", "T7"]


def test_budget_truncation_is_deterministic_across_runs():
    many = [cand(f"T{i}", f"L{i}", 0.5) for i in range(20)]  # all tied
    a, _ = apply_budget(many, SelectorPolicy(max_candidates=5))
    b, _ = apply_budget(list(reversed(many)), SelectorPolicy(max_candidates=5))
    assert [c.id for c in a] == [c.id for c in b]


def test_token_budget_trims_further_than_the_count_budget():
    long_label = "x" * 4000
    many = [cand(f"T{i}", long_label, 1.0 - i / 100) for i in range(20)]
    kept, dropped = apply_budget(many, SelectorPolicy(max_candidates=20, max_candidate_tokens=2000))
    assert len(kept) < 20 and dropped == 20 - len(kept)


def test_token_budget_always_keeps_at_least_one_candidate():
    huge = [cand("T1", "y" * 100_000, 1.0)]
    kept, dropped = apply_budget(huge, SelectorPolicy(max_candidate_tokens=10))
    assert len(kept) == 1 and dropped == 0


# --- selection ------------------------------------------------------------------


async def test_resolves_a_chosen_key_to_a_record_id():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.95, explanation="exact")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id == "T1"
    assert outcome.choice.resolution is Resolution.EXACT_KEY
    assert outcome.confidence == 0.95


async def test_abstention_yields_no_record_id():
    llm = FakeLLM([reply(chosen_key=None, confidence_score=0.0, explanation="none fit")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id is None
    assert outcome.choice.resolution is Resolution.ABSTAIN


async def test_a_hallucinated_id_does_not_resolve_to_a_record():
    llm = FakeLLM([reply(chosen_key="T1", confidence_score=0.99, explanation="sure")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id is None
    assert outcome.choice.resolution is Resolution.UNRESOLVED


async def test_a_bare_integer_does_not_resolve_to_a_record():
    llm = FakeLLM([reply(chosen_key="1", confidence_score=0.99, explanation="the first")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.record_id is None


async def test_the_prompt_carries_the_keyed_candidate_block():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert "[C01] ID: T1" in llm.requests[0].user


async def test_the_prompt_carries_the_context_when_present():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "blood [glucose] levels", CANDIDATES)
    assert "blood [glucose] levels" in llm.requests[0].user


async def test_the_request_carries_the_select_schema():
    from xwalk.prompts.contract import SELECT_SCHEMA

    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert llm.requests[0].schema == SELECT_SCHEMA


async def test_malformed_output_is_unresolved_not_an_exception():
    llm = FakeLLM(["I decline to answer."])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.resolution is Resolution.UNRESOLVED
    assert outcome.error is not None


async def test_output_with_a_wrong_confidence_type_is_unresolved():
    llm = FakeLLM(['{"chosen_key": "C01", "confidence_score": "high", "explanation": "x"}'])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.choice.resolution is Resolution.UNRESOLVED


async def test_a_confidence_outside_zero_to_one_is_clamped():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=1.7, explanation="x")])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.confidence == 1.0


async def test_provider_errors_propagate_for_the_matcher_to_classify():
    llm = FakeLLM([LLMRetryableError("429")])
    with pytest.raises(LLMRetryableError):
        await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)


async def test_no_candidates_short_circuits_without_calling_the_model():
    llm = FakeLLM([])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", [])
    assert outcome.choice.resolution is Resolution.ABSTAIN
    assert llm.requests == []
    assert outcome.usage.calls == 0


async def test_truncation_count_is_reported_on_the_outcome():
    many = [cand(f"T{i}", f"L{i}", 1.0 - i / 100) for i in range(40)]
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    outcome = await Selector(
        llm, PROMPTS, TEMPLATES, policy=SelectorPolicy(max_candidates=10)
    ).select(SOURCE, "", many)
    assert outcome.truncated == 30
    assert len(outcome.keyed) == 10


async def test_legacy_mode_accepts_a_raw_id_and_flags_it():
    llm = FakeLLM([reply(chosen_key="T1", confidence_score=0.9)])
    outcome = await Selector(llm, PROMPTS, TEMPLATES, legacy_id_resolution=True).select(
        SOURCE, "", CANDIDATES
    )
    assert outcome.choice.record_id == "T1"
    assert outcome.choice.resolution is Resolution.LEGACY_EXACT_ID


async def test_usage_is_recorded():
    llm = FakeLLM([reply(chosen_key="C01", confidence_score=0.9)])
    outcome = await Selector(llm, PROMPTS, TEMPLATES).select(SOURCE, "", CANDIDATES)
    assert outcome.usage.calls == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_select.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.stages.select'`.

- [ ] **Step 3: Write `src/xwalk/stages/select.py`**

```python
"""Selection: show the model a keyed candidate list, get back one key or null."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.prompts.contract import SELECT_SCHEMA, PromptSet
from xwalk.records import Candidate, Record, Usage
from xwalk.stages.keying import (
    KeyedCandidates,
    Resolution,
    ResolvedChoice,
    assign_keys,
    resolve_key,
)
from xwalk.templates import TemplateSet

_CHARS_PER_TOKEN = 4  # a deliberately crude estimate; the budget is a guard rail, not a meter


@dataclass(frozen=True)
class SelectorPolicy:
    """How much of the fused candidate list reaches the model.

    This is not retrieval depth — each retriever owns its own `k`. With N retrievers the
    fused list grows without bound, and this is the separate constraint that governs it.
    """

    max_candidates: int = 30
    max_candidate_tokens: int = 8_000


@dataclass(frozen=True)
class SelectionOutcome:
    choice: ResolvedChoice
    confidence: float | None
    explanation: str
    raw: str
    keyed: KeyedCandidates
    truncated: int
    usage: Usage
    error: str | None = None


def apply_budget(
    candidates: Sequence[Candidate], policy: SelectorPolicy
) -> tuple[list[Candidate], int]:
    """Deterministically trim the fused list. Returns (kept, dropped_count)."""
    ordered = sorted(candidates, key=lambda c: (-c.fused_score, c.id))
    kept = ordered[: policy.max_candidates]

    budget_chars = policy.max_candidate_tokens * _CHARS_PER_TOKEN
    total = 0
    trimmed: list[Candidate] = []
    for candidate in kept:
        # a cheap proxy for the rendered size; the exact render happens after keying
        size = sum(len(str(v)) for v in candidate.record.fields.values()) + len(candidate.id) + 16
        if trimmed and total + size > budget_chars:
            break
        trimmed.append(candidate)
        total += size

    return trimmed, len(candidates) - len(trimmed)


def _coerce_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


class Selector:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        policy: SelectorPolicy | None = None,
        legacy_id_resolution: bool = False,
        system: str = "You return JSON only. No prose, no code fences.",
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._policy = policy or SelectorPolicy()
        self._legacy = legacy_id_resolution
        self._system = system
        self._max_tokens = max_tokens

    async def select(
        self,
        source: Record,
        context: str,
        candidates: Sequence[Candidate],
    ) -> SelectionOutcome:
        kept, dropped = apply_budget(candidates, self._policy)
        keyed = assign_keys(kept, self._templates)

        if not keyed.order:
            return SelectionOutcome(
                choice=ResolvedChoice(None, Resolution.ABSTAIN, None),
                confidence=None,
                explanation="no candidates were retrieved",
                raw="",
                keyed=keyed,
                truncated=dropped,
                usage=Usage.zero(),
            )

        prompt = self._prompts.render_select(
            source_fields=source.fields,
            context=context,
            candidate_block=keyed.rendered,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=self._system,
                user=prompt,
                schema=SELECT_SCHEMA,
                schema_name="selection",
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            return SelectionOutcome(
                choice=ResolvedChoice(None, Resolution.UNRESOLVED, response.text),
                confidence=None,
                explanation="",
                raw=response.text,
                keyed=keyed,
                truncated=dropped,
                usage=response.usage,
                error=str(exc),
            )

        raw_key = payload.get("chosen_key")
        if raw_key is not None and not isinstance(raw_key, str):
            raw_key = str(raw_key)
        confidence = _coerce_confidence(payload.get("confidence_score"))
        explanation = str(payload.get("explanation", ""))

        choice = resolve_key(raw_key, keyed, legacy=self._legacy)

        # A choice we cannot score is a choice we cannot classify.
        error: str | None = None
        if choice.resolution is Resolution.EXACT_KEY and confidence is None:
            choice = ResolvedChoice(None, Resolution.UNRESOLVED, raw_key)
            error = "confidence_score was missing or not a number"
        elif choice.resolution is Resolution.UNRESOLVED:
            error = f"model returned {raw_key!r}, which is not a key issued this attempt"

        return SelectionOutcome(
            choice=choice,
            confidence=confidence,
            explanation=explanation,
            raw=response.text,
            keyed=keyed,
            truncated=dropped,
            usage=response.usage,
            error=error,
        )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_select.py -v`
Expected: 21 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/stages/select.py tests/test_select.py
git commit -m "feat: selector stage with deterministic candidate budget"
```

---

## Task 12: The gate — scorer and verifier

**Files:**
- Create: `src/xwalk/stages/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 7, 9, 10, and `RetryProposal` (defined in Task 13 — **define `RetryProposal` in `records.py` as part of this task** so `gate.py` can emit proposals; Task 13 then adds only the routing logic).
- Produces: `RetryProposal(kind: Literal["candidate","query"], value: str, source: Literal["scorer","rewriter"])` in `records.py`; `ScoreOutcome(score, explanation, proposals, raw, usage, error)`; `VerifierVerdict(decision, preferred_key, confidence, explanation, raw, usage, error)`; `Scorer(llm, prompts, templates, *, review_floor=0.4)` with `async score(source, context, keyed, chosen_key) -> ScoreOutcome`; `Verifier(llm, prompts, templates)` with `async verify(source, context, keyed, chosen_key) -> VerifierVerdict`. Tasks 13, 14, 17 consume these.

**Two design points the tests enforce:**

1. **The gate scores against the full source record and context, never the retrieval query.** The query is deliberately lossy — for a multi-field source it may omit exactly the fields needed to disambiguate — so scoring against it evaluates the wrong thing.
2. **Verification returns a verdict, not a second number.** `min(primary, verifier)` over two uncalibrated scores is arbitrary and cannot distinguish agreement-with-different-numbers from genuine disagreement. `disagree` or `no_match` forces review regardless of the arithmetic. When the verifier names a different `preferred_key`, that key is **recorded and flagged, not chased** — re-entering selection on the verifier's preference would make loop termination depend on two models negotiating.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gate.py`:

```python
import json

import pytest

from xwalk.llm.fake import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.keying import assign_keys
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ symbol }}", context="", doc="", candidate="ID: {{ id }} Label: {{ label }}"
)
SLOTS = PromptSlots(
    entity_noun="gene mention",
    target_noun="NCBI gene",
    domain_brief="gene nomenclature",
    rubric=[
        {"score": 1.0, "name": "Certain", "when": "exact symbol and organism match"},
        {"score": 0.4, "name": "Weak", "when": "symbol matches, organism unknown"},
    ],
)
PROMPTS = PromptSet.from_slots(SLOTS)
SOURCE = Record(id="s1", fields={"symbol": "TP53", "organism": "Homo sapiens"})


def cand(record_id: str, label: str) -> Candidate:
    return Candidate(
        record=Record(id=record_id, fields={"label": label}),
        fused_score=0.5,
        evidence=(RetrievalHit(record_id=record_id, retriever="bm25", raw_score=1.0, rank=1),),
    )


KEYED = assign_keys([cand("NCBIGene:7157", "TP53"), cand("NCBIGene:22059", "Trp53")], TEMPLATES)


def score_reply(**kwargs) -> str:
    body = {"confidence_score": 0.5, "explanation": ""}
    body.update(kwargs)
    return json.dumps(body)


def verify_reply(**kwargs) -> str:
    body = {"decision": "support", "explanation": ""}
    body.update(kwargs)
    return json.dumps(body)


# --- scorer ----------------------------------------------------------------------


async def test_scorer_returns_the_confidence():
    llm = FakeLLM([score_reply(confidence_score=0.83, explanation="symbol and organism agree")])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert outcome.score == 0.83


async def test_scorer_sees_every_source_field_not_just_the_query():
    llm = FakeLLM([score_reply(confidence_score=0.8)])
    await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    prompt = llm.requests[0].user
    assert "TP53" in prompt and "Homo sapiens" in prompt


async def test_scorer_shows_the_chosen_candidate_and_the_others_separately():
    llm = FakeLLM([score_reply(confidence_score=0.8)])
    await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    prompt = llm.requests[0].user
    assert "## Proposed match" in prompt
    assert "[C01] ID: NCBIGene:7157" in prompt
    assert "[C02] ID: NCBIGene:22059" in prompt


async def test_scorer_emits_candidate_proposals_for_issued_keys():
    llm = FakeLLM([score_reply(confidence_score=0.3, better_candidate_keys=["C02"])])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert [(p.kind, p.value, p.source) for p in outcome.proposals] == [
        ("candidate", "C02", "scorer")
    ]


async def test_scorer_emits_query_proposals_separately():
    llm = FakeLLM([score_reply(confidence_score=0.3, better_queries=["tumour protein p53"])])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert [(p.kind, p.value) for p in outcome.proposals] == [("query", "tumour protein p53")]


async def test_scorer_keeps_candidate_and_query_proposals_distinct():
    """The paper repo funnels both into one query queue and re-runs full retrieval on
    records it already has in hand. They are different objects."""
    llm = FakeLLM(
        [score_reply(confidence_score=0.3, better_candidate_keys=["C02"], better_queries=["p53"])]
    )
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    kinds = {p.kind for p in outcome.proposals}
    assert kinds == {"candidate", "query"}


async def test_scorer_drops_blank_and_duplicate_proposals():
    llm = FakeLLM([score_reply(confidence_score=0.3, better_queries=["p53", "  ", "p53", ""])])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert [p.value for p in outcome.proposals] == ["p53"]


async def test_scorer_malformed_output_yields_a_zero_score_and_an_error():
    llm = FakeLLM(["not json"])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert outcome.score is None and outcome.error is not None


async def test_scorer_clamps_an_out_of_range_score():
    llm = FakeLLM([score_reply(confidence_score=-2)])
    outcome = await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    assert outcome.score == 0.0


async def test_scorer_rejects_an_unissued_chosen_key():
    llm = FakeLLM([])
    with pytest.raises(KeyError):
        await Scorer(llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C99")


# --- verifier --------------------------------------------------------------------


async def test_verifier_supports():
    llm = FakeLLM([verify_reply(decision="support", confidence_score=0.9)])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "support"


async def test_verifier_disagrees_and_names_a_preferred_key():
    llm = FakeLLM([verify_reply(decision="disagree", preferred_key="C02")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree" and verdict.preferred_key == "C02"


async def test_verifier_returns_no_match():
    llm = FakeLLM([verify_reply(decision="no_match")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "no_match"


async def test_verifier_drops_a_preferred_key_that_was_never_issued():
    """A preferred key we cannot resolve is noise, not a lead."""
    llm = FakeLLM([verify_reply(decision="disagree", preferred_key="C99")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree" and verdict.preferred_key is None


async def test_verifier_treats_an_unknown_decision_as_disagreement():
    """Fail towards review, never towards silent acceptance."""
    llm = FakeLLM([verify_reply(decision="maybe")])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree"


async def test_verifier_malformed_output_is_disagreement_with_an_error():
    llm = FakeLLM(["???"])
    verdict = await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert verdict.decision == "disagree" and verdict.error is not None


async def test_verifier_sees_the_full_source_record():
    llm = FakeLLM([verify_reply()])
    await Verifier(llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert "Homo sapiens" in llm.requests[0].user


async def test_verifier_prompt_is_not_the_scorer_prompt():
    """An independent verdict requires an independent framing."""
    scorer_llm = FakeLLM([score_reply(confidence_score=0.7)])
    verifier_llm = FakeLLM([verify_reply()])
    await Scorer(scorer_llm, PROMPTS, TEMPLATES).score(SOURCE, "", KEYED, "C01")
    await Verifier(verifier_llm, PROMPTS, TEMPLATES).verify(SOURCE, "", KEYED, "C01")
    assert scorer_llm.requests[0].user != verifier_llm.requests[0].user
```

- [ ] **Step 2: Add `RetryProposal` to `src/xwalk/records.py`**

Merge the import into the existing block at the top of the file rather than pasting it here — a mid-file `import` is `E402` and fails the lint gate.

```python
from typing import Literal


@dataclass(frozen=True)
class RetryProposal:
    """A lead for the next attempt.

    `kind="candidate"` means "look again at a record we already retrieved" — no new
    search. `kind="query"` means "run a genuinely new search". Conflating the two makes
    the loop re-retrieve records it already has in hand.
    """

    kind: Literal["candidate", "query"]
    value: str
    source: Literal["scorer", "rewriter"]
```

Add `RetryProposal` to `src/xwalk/__init__.py`'s imports and `__all__`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.stages.gate'`.

- [ ] **Step 4: Write `src/xwalk/stages/gate.py`**

```python
"""The gate: an independent score, and — when it is uncertain — an independent verdict."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.prompts.contract import SCORE_SCHEMA, VERIFY_SCHEMA, PromptSet
from xwalk.records import Record, RetryProposal, Usage
from xwalk.stages.keying import KeyedCandidates
from xwalk.templates import TemplateSet

Decision = Literal["support", "disagree", "no_match"]
_SYSTEM = "You return JSON only. No prose, no code fences."


@dataclass(frozen=True)
class ScoreOutcome:
    score: float | None
    explanation: str
    proposals: tuple[RetryProposal, ...]
    raw: str
    usage: Usage
    error: str | None = None


@dataclass(frozen=True)
class VerifierVerdict:
    decision: Decision
    preferred_key: str | None
    confidence: float | None
    explanation: str
    raw: str
    usage: Usage
    error: str | None = None


def _clamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _blocks(keyed: KeyedCandidates, chosen_key: str) -> tuple[str, str]:
    """Split the rendered candidate list into (chosen, others)."""
    if chosen_key not in keyed.by_key:
        raise KeyError(f"{chosen_key!r} was not issued for this attempt")
    entries = keyed.rendered.split("\n\n")
    chosen = next(e for e in entries if e.startswith(f"[{chosen_key}]"))
    others = [e for e in entries if not e.startswith(f"[{chosen_key}]")]
    return chosen, "\n\n".join(others)


def _clean(values: object, kind: Literal["candidate", "query"], issued: Sequence[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        if kind == "candidate" and text.upper() not in issued:
            continue  # naming a key we never issued is a hallucination, not a lead
        seen.append(text.upper() if kind == "candidate" else text)
    return seen


class Scorer:
    """Judges the selected match against the whole source record."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        review_floor: float = 0.4,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._review_floor = review_floor
        self._max_tokens = max_tokens

    async def score(
        self,
        source: Record,
        context: str,
        keyed: KeyedCandidates,
        chosen_key: str,
    ) -> ScoreOutcome:
        chosen_block, other_candidates = _blocks(keyed, chosen_key)
        prompt = self._prompts.render_score(
            source_fields=source.fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
            review_floor=self._review_floor,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=prompt,
                schema=SCORE_SCHEMA,
                schema_name="score",
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            return ScoreOutcome(
                score=None,
                explanation="",
                proposals=(),
                raw=response.text,
                usage=response.usage,
                error=str(exc),
            )

        score = _clamp(payload.get("confidence_score"))
        issued = list(keyed.order)
        proposals = tuple(
            RetryProposal(kind="candidate", value=key, source="scorer")
            for key in _clean(payload.get("better_candidate_keys"), "candidate", issued)
        ) + tuple(
            RetryProposal(kind="query", value=query, source="scorer")
            for query in _clean(payload.get("better_queries"), "query", issued)
        )

        return ScoreOutcome(
            score=score,
            explanation=str(payload.get("explanation", "")),
            proposals=proposals,
            raw=response.text,
            usage=response.usage,
            error=None if score is not None else "confidence_score was missing or not a number",
        )


class Verifier:
    """A second, independent opinion. Returns a verdict, not a number to average."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._max_tokens = max_tokens

    async def verify(
        self,
        source: Record,
        context: str,
        keyed: KeyedCandidates,
        chosen_key: str,
    ) -> VerifierVerdict:
        chosen_block, other_candidates = _blocks(keyed, chosen_key)
        prompt = self._prompts.render_verify(
            source_fields=source.fields,
            context=context,
            chosen_block=chosen_block,
            other_candidates=other_candidates,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=prompt,
                schema=VERIFY_SCHEMA,
                schema_name="verification",
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            # An unreadable second opinion is not a supporting one.
            return VerifierVerdict(
                decision="disagree",
                preferred_key=None,
                confidence=None,
                explanation="",
                raw=response.text,
                usage=response.usage,
                error=str(exc),
            )

        raw_decision = str(payload.get("decision", "")).strip().lower()
        decision: Decision = (
            raw_decision if raw_decision in ("support", "disagree", "no_match") else "disagree"  # type: ignore[assignment]
        )
        error = None if raw_decision == decision else f"unrecognised decision {raw_decision!r}"

        raw_preferred = payload.get("preferred_key")
        preferred: str | None = None
        if isinstance(raw_preferred, str) and raw_preferred.strip().upper() in keyed.by_key:
            preferred = raw_preferred.strip().upper()

        return VerifierVerdict(
            decision=decision,
            preferred_key=preferred,
            confidence=_clamp(payload.get("confidence_score")),
            explanation=str(payload.get("explanation", "")),
            raw=response.text,
            usage=response.usage,
            error=error,
        )
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_gate.py -v`
Expected: 18 passed.

- [ ] **Step 6: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/stages/gate.py src/xwalk/records.py src/xwalk/__init__.py tests/test_gate.py
git commit -m "feat: scorer and verdict-returning verifier"
```

---

## Task 13: Query rewriting and proposal routing

**Files:**
- Create: `src/xwalk/stages/rewrite.py`, `src/xwalk/stages/proposals.py`
- Test: `tests/test_rewrite.py`, `tests/test_proposals.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 7, 9, 10, 12.
- Produces: `QueryRewriter(llm, prompts, templates, *, max_queries=2)` with `async rewrite(source, context, previous_queries, keyed) -> RewriteOutcome(proposals, explanation, raw, usage, error)`; `RoutedProposals(candidate_keys, queries, dropped)`; `route_proposals(proposals, issued_keys, seen_queries) -> RoutedProposals`; `normalise_query(text) -> str`. Task 17 consumes these.

- [ ] **Step 1: Write the failing routing test**

Create `tests/test_proposals.py`:

```python
from xwalk.records import RetryProposal
from xwalk.stages.proposals import normalise_query, route_proposals

ISSUED = ("C01", "C02", "C03")


def prop(kind, value, source="scorer"):
    return RetryProposal(kind=kind, value=value, source=source)


def test_candidate_proposals_naming_issued_keys_are_kept():
    routed = route_proposals([prop("candidate", "C02")], ISSUED, seen_queries=set())
    assert routed.candidate_keys == ("C02",)


def test_candidate_proposals_naming_unissued_keys_are_dropped():
    routed = route_proposals([prop("candidate", "C99")], ISSUED, seen_queries=set())
    assert routed.candidate_keys == ()
    assert routed.dropped[0][0].value == "C99"


def test_a_dropped_candidate_is_never_promoted_to_a_query():
    """A candidate proposal naming something not in the set is a hallucination, not a
    new lead. Promoting it would run a full retrieval on invented text."""
    routed = route_proposals([prop("candidate", "C99")], ISSUED, seen_queries=set())
    assert routed.queries == ()


def test_the_drop_reason_is_recorded():
    routed = route_proposals([prop("candidate", "C99")], ISSUED, seen_queries=set())
    assert "not issued" in routed.dropped[0][1]


def test_query_proposals_enter_the_query_queue():
    routed = route_proposals([prop("query", "tumour protein p53")], ISSUED, seen_queries=set())
    assert routed.queries == ("tumour protein p53",)


def test_queries_already_tried_are_dropped():
    routed = route_proposals([prop("query", "TP53")], ISSUED, seen_queries={"tp53"})
    assert routed.queries == ()
    assert "already tried" in routed.dropped[0][1]


def test_query_deduplication_ignores_case_and_whitespace():
    routed = route_proposals([prop("query", "  TP53  ")], ISSUED, seen_queries={"tp53"})
    assert routed.queries == ()


def test_duplicate_queries_within_one_batch_are_collapsed():
    routed = route_proposals(
        [prop("query", "p53"), prop("query", "P53")], ISSUED, seen_queries=set()
    )
    assert routed.queries == ("p53",)


def test_blank_proposals_are_dropped():
    routed = route_proposals([prop("query", "   ")], ISSUED, seen_queries=set())
    assert routed.queries == () and len(routed.dropped) == 1


def test_candidate_keys_are_deduplicated_and_order_preserved():
    routed = route_proposals(
        [prop("candidate", "C03"), prop("candidate", "C01"), prop("candidate", "C03")],
        ISSUED,
        seen_queries=set(),
    )
    assert routed.candidate_keys == ("C03", "C01")


def test_candidate_key_matching_is_case_insensitive():
    routed = route_proposals([prop("candidate", "c02")], ISSUED, seen_queries=set())
    assert routed.candidate_keys == ("C02",)


def test_normalise_query_lowercases_and_collapses_whitespace():
    assert normalise_query("  TP53   gene ") == "tp53 gene"


def test_empty_input_routes_to_nothing():
    routed = route_proposals([], ISSUED, seen_queries=set())
    assert routed.candidate_keys == () and routed.queries == () and routed.dropped == ()
```

Create `tests/test_rewrite.py`:

```python
import json

from xwalk.llm.fake import FakeLLM
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import assign_keys
from xwalk.stages.rewrite import QueryRewriter
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}", context="", doc="", candidate="ID: {{ id }} Label: {{ label }}"
)
PROMPTS = PromptSet.from_slots(
    PromptSlots(
        entity_noun="mention",
        target_noun="term",
        domain_brief="test",
        rubric=[
            {"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.4, "name": "Weak", "when": "vague"},
        ],
    )
)
SOURCE = Record(id="s1", fields={"mention": "ASA"})
KEYED = assign_keys(
    [
        Candidate(
            record=Record(id="T1", fields={"label": "aspirin"}),
            fused_score=0.5,
            evidence=(RetrievalHit(record_id="T1", retriever="bm25", raw_score=1.0, rank=1),),
        )
    ],
    TEMPLATES,
)


async def test_returns_query_proposals():
    llm = FakeLLM([json.dumps({"queries": ["acetylsalicylic acid"], "explanation": "expanded"})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert [(p.kind, p.value, p.source) for p in outcome.proposals] == [
        ("query", "acetylsalicylic acid", "rewriter")
    ]


async def test_all_proposals_from_the_rewriter_are_queries():
    llm = FakeLLM([json.dumps({"queries": ["a", "b"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert all(p.kind == "query" for p in outcome.proposals)


async def test_respects_max_queries():
    llm = FakeLLM([json.dumps({"queries": ["a", "b", "c", "d"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES, max_queries=2).rewrite(
        SOURCE, "", ["ASA"], KEYED
    )
    assert len(outcome.proposals) == 2


async def test_repeats_of_previous_queries_are_dropped():
    llm = FakeLLM([json.dumps({"queries": ["ASA", "acetylsalicylic acid"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert [p.value for p in outcome.proposals] == ["acetylsalicylic acid"]


async def test_the_prompt_lists_previous_queries():
    llm = FakeLLM([json.dumps({"queries": ["x"], "explanation": ""})])
    await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA", "aspirin"], KEYED)
    prompt = llm.requests[0].user
    assert "- ASA" in prompt and "- aspirin" in prompt


async def test_malformed_output_yields_no_proposals_and_an_error():
    llm = FakeLLM(["nope"])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(SOURCE, "", ["ASA"], KEYED)
    assert outcome.proposals == () and outcome.error is not None


async def test_works_when_there_were_no_candidates():
    from xwalk.stages.keying import assign_keys as ak

    llm = FakeLLM([json.dumps({"queries": ["acetylsalicylic acid"], "explanation": ""})])
    outcome = await QueryRewriter(llm, PROMPTS, TEMPLATES).rewrite(
        SOURCE, "", ["ASA"], ak([], TEMPLATES)
    )
    assert len(outcome.proposals) == 1
```

- [ ] **Step 2: Run both to verify they fail**

Run: `python -m pytest tests/test_proposals.py tests/test_rewrite.py -v`
Expected: FAIL — modules do not exist.

- [ ] **Step 3: Write `src/xwalk/stages/proposals.py`**

```python
"""Routing retry proposals.

Two different objects share the name "retry lead", and conflating them causes real
waste: the paper repo asks the scorer for alternatives drawn from the candidates it was
just shown, then pushes those labels into the query queue and re-runs full retrieval on
records it already has in hand.
"""

from __future__ import annotations

import re
from collections.abc import Sequence, Set
from dataclasses import dataclass

from xwalk.records import RetryProposal

_WS = re.compile(r"\s+")


def normalise_query(text: str) -> str:
    """The comparison form for query deduplication."""
    return _WS.sub(" ", text.strip().lower())


@dataclass(frozen=True)
class RoutedProposals:
    candidate_keys: tuple[str, ...]
    queries: tuple[str, ...]
    dropped: tuple[tuple[RetryProposal, str], ...]


def route_proposals(
    proposals: Sequence[RetryProposal],
    issued_keys: Sequence[str],
    seen_queries: Set[str],
) -> RoutedProposals:
    """Split proposals into re-examinations and new searches, dropping the rest.

    `seen_queries` holds already-normalised query strings.
    """
    issued = {key.upper() for key in issued_keys}
    keys: list[str] = []
    queries: list[str] = []
    seen_normalised = set(seen_queries)
    dropped: list[tuple[RetryProposal, str]] = []

    for proposal in proposals:
        value = proposal.value.strip()
        if not value:
            dropped.append((proposal, "blank value"))
            continue

        if proposal.kind == "candidate":
            key = value.upper()
            if key not in issued:
                # Not a lead — an invented key. Recorded, never promoted to a query.
                dropped.append((proposal, f"candidate key {value!r} was not issued this attempt"))
            elif key in keys:
                dropped.append((proposal, "duplicate candidate key"))
            else:
                keys.append(key)
            continue

        normalised = normalise_query(value)
        if normalised in seen_normalised:
            dropped.append((proposal, "query already tried"))
            continue
        seen_normalised.add(normalised)
        queries.append(value)

    return RoutedProposals(
        candidate_keys=tuple(keys), queries=tuple(queries), dropped=tuple(dropped)
    )
```

- [ ] **Step 4: Write `src/xwalk/stages/rewrite.py`**

```python
"""Query reformulation. Produces query proposals only — never candidate proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.llm.base import LLMClient, LLMRequest, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.prompts.contract import REWRITE_SCHEMA, PromptSet
from xwalk.records import Record, RetryProposal, Usage
from xwalk.stages.keying import KeyedCandidates
from xwalk.stages.proposals import normalise_query
from xwalk.templates import TemplateSet

_SYSTEM = "You return JSON only. No prose, no code fences."


@dataclass(frozen=True)
class RewriteOutcome:
    proposals: tuple[RetryProposal, ...]
    explanation: str
    raw: str
    usage: Usage
    error: str | None = None


class QueryRewriter:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptSet,
        templates: TemplateSet,
        *,
        max_queries: int = 2,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._templates = templates
        self._max_queries = max_queries
        self._max_tokens = max_tokens

    async def rewrite(
        self,
        source: Record,
        context: str,
        previous_queries: Sequence[str],
        keyed: KeyedCandidates,
    ) -> RewriteOutcome:
        prompt = self._prompts.render_rewrite(
            source_fields=source.fields,
            context=context,
            previous_queries=list(previous_queries),
            best_candidates=keyed.rendered,
            max_queries=self._max_queries,
        )
        response = await self._llm.complete(
            LLMRequest(
                system=_SYSTEM,
                user=prompt,
                schema=REWRITE_SCHEMA,
                schema_name="rewrite",
                max_tokens=self._max_tokens,
            )
        )

        try:
            payload = parse_json_object(response.text)
        except ParseError as exc:
            return RewriteOutcome(
                proposals=(),
                explanation="",
                raw=response.text,
                usage=response.usage,
                error=str(exc),
            )

        seen = {normalise_query(q) for q in previous_queries}
        proposals: list[RetryProposal] = []
        for value in payload.get("queries", []) or []:
            if not isinstance(value, str):
                continue
            text = value.strip()
            normalised = normalise_query(text)
            if not text or normalised in seen:
                continue
            seen.add(normalised)
            proposals.append(RetryProposal(kind="query", value=text, source="rewriter"))
            if len(proposals) >= self._max_queries:
                break

        return RewriteOutcome(
            proposals=tuple(proposals),
            explanation=str(payload.get("explanation", "")),
            raw=response.text,
            usage=response.usage,
        )
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_proposals.py tests/test_rewrite.py -v`
Expected: 20 passed (13 + 7).

- [ ] **Step 6: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/stages/rewrite.py src/xwalk/stages/proposals.py tests/test_rewrite.py tests/test_proposals.py
git commit -m "feat: query rewriter and candidate-vs-query proposal routing"
```

---

## Task 14: Result types, match policy, and status derivation

**Files:**
- Modify: `src/xwalk/records.py` (add `MatchStatus`, `DecisionReason`, `Attempt`, `MatchResult`)
- Create: `src/xwalk/policy.py`
- Modify: `src/xwalk/__init__.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: Tasks 1, 9, 12.
- Produces: `MatchStatus` enum; `DecisionReason` enum; `Attempt`; `MatchResult`; `MatchPolicy(max_attempts=4, accept_at=0.6, review_floor=0.4, verify_band=(0.6, 0.8), audit_rate=0.0, concurrency=32, legacy_id_resolution=False, retriever_timeout=60.0)`; `derive_status(attempts, policy) -> tuple[MatchStatus, DecisionReason, Attempt | None]`; `should_verify(score, policy) -> bool`; `should_audit(score, policy, run_fp, source_id) -> bool`. Task 17 consumes these.

**Why status and reason are separate:** a single `REJECTED` status cannot tell rejected-by-model from rejected-by-policy from rejected-by-human. Recording the outcome (`reason`) separately from its classification (`status`) means `accept_at` can be retuned and an existing run reclassified *without* the original outcome having been destroyed.

**Precedence order, in full.** A low-confidence pick outranks an earlier explicit abstention — the model found *something* worth surfacing. Unresolvable output routes to review rather than silence, because a malformed answer is a signal, not a non-match.

1. Every attempt failed with an infrastructure error → `FAILED` / `RETRIEVER_FAILURE` or `PROVIDER_FAILURE`.
2. Verification disagreed → `NEEDS_REVIEW` / `VERIFIER_DISAGREEMENT`.
3. A resolvable scored choice exists → classify by score: `≥ accept_at` → `MATCHED / ACCEPT_THRESHOLD`; `[review_floor, accept_at)` → `NEEDS_REVIEW / BELOW_ACCEPT_THRESHOLD`; `< review_floor` → `UNMATCHED / BELOW_REVIEW_FLOOR`.
4. No resolvable choice, selector abstained → `UNMATCHED / SELECTOR_ABSTAINED`.
5. No candidates were ever retrieved → `UNMATCHED / NO_CANDIDATES`.
6. Candidates existed but output never resolved → `NEEDS_REVIEW / UNRESOLVED_OUTPUT`.

Plus one rule from Task 9: **any non-exact resolution (legacy mode) forces `NEEDS_REVIEW`** even when the score clears `accept_at`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_policy.py`:

```python
import pytest

from xwalk.policy import MatchPolicy, derive_status, should_audit, should_verify
from xwalk.records import Attempt, DecisionReason, MatchStatus, Usage
from xwalk.stages.keying import Resolution

POLICY = MatchPolicy()


def attempt(**kwargs) -> Attempt:
    base = dict(
        index=0,
        query="q",
        proposal=None,
        candidates=(),
        candidate_count=0,
        candidates_truncated=0,
        issued_keys={},
        raw_selection=None,
        chosen_id=None,
        resolution=Resolution.ABSTAIN.value,
        primary_score=None,
        explanation="",
        verifier_decision=None,
        verifier_score=None,
        verifier_preferred_id=None,
        audited=False,
        dropped_proposals=(),
        reason=None,
        error=None,
        usage=Usage.zero(),
    )
    base.update(kwargs)
    return Attempt(**base)


def matched(score: float, **kwargs) -> Attempt:
    # Defaults merged rather than passed through, so a caller may override `resolution`
    # (which `test_a_legacy_resolution_forces_review_even_at_a_high_score` does).
    base = dict(
        chosen_id="T1",
        resolution=Resolution.EXACT_KEY.value,
        primary_score=score,
        candidate_count=3,
    )
    base.update(kwargs)
    return attempt(**base)


# --- classification --------------------------------------------------------------


def test_a_high_score_is_matched():
    status, reason, best = derive_status([matched(0.9)], POLICY)
    assert status is MatchStatus.MATCHED
    assert reason is DecisionReason.ACCEPT_THRESHOLD
    assert best is not None and best.chosen_id == "T1"


def test_a_score_in_the_review_band_needs_review():
    status, reason, _ = derive_status([matched(0.5)], POLICY)
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.BELOW_ACCEPT_THRESHOLD


def test_a_score_below_the_floor_is_unmatched():
    status, reason, _ = derive_status([matched(0.2)], POLICY)
    assert status is MatchStatus.UNMATCHED
    assert reason is DecisionReason.BELOW_REVIEW_FLOOR


def test_the_accept_threshold_is_inclusive():
    status, _, _ = derive_status([matched(0.6)], POLICY)
    assert status is MatchStatus.MATCHED


def test_the_review_floor_is_inclusive():
    status, reason, _ = derive_status([matched(0.4)], POLICY)
    assert reason is DecisionReason.BELOW_ACCEPT_THRESHOLD


# --- abstention, no candidates, unresolved ---------------------------------------


def test_an_explicit_abstention_is_unmatched():
    status, reason, _ = derive_status([attempt(candidate_count=3)], POLICY)
    assert status is MatchStatus.UNMATCHED
    assert reason is DecisionReason.SELECTOR_ABSTAINED


def test_no_candidates_is_distinguishable_from_abstention():
    status, reason, _ = derive_status([attempt(candidate_count=0)], POLICY)
    assert reason is DecisionReason.NO_CANDIDATES


def test_unresolved_output_routes_to_review_not_to_silence():
    status, reason, _ = derive_status(
        [attempt(candidate_count=3, resolution=Resolution.UNRESOLVED.value, raw_selection="???")],
        POLICY,
    )
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.UNRESOLVED_OUTPUT


# --- precedence ------------------------------------------------------------------


def test_a_weak_pick_outranks_an_earlier_abstention():
    """The model found something worth surfacing; that beats an earlier shrug."""
    status, reason, best = derive_status([attempt(candidate_count=3), matched(0.45)], POLICY)
    assert reason is DecisionReason.BELOW_ACCEPT_THRESHOLD
    assert best is not None and best.primary_score == 0.45


def test_the_highest_scoring_attempt_wins():
    _, _, best = derive_status([matched(0.5), matched(0.8), matched(0.3)], POLICY)
    assert best is not None and best.primary_score == 0.8


def test_ties_resolve_to_the_earlier_attempt():
    _, _, best = derive_status([matched(0.8, index=0), matched(0.8, index=1)], POLICY)
    assert best is not None and best.index == 0


def test_verifier_disagreement_overrides_a_passing_score():
    status, reason, _ = derive_status([matched(0.95, verifier_decision="disagree")], POLICY)
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.VERIFIER_DISAGREEMENT


def test_verifier_no_match_also_forces_review():
    status, reason, _ = derive_status([matched(0.95, verifier_decision="no_match")], POLICY)
    assert reason is DecisionReason.VERIFIER_DISAGREEMENT


def test_verifier_support_leaves_the_classification_alone():
    status, reason, _ = derive_status([matched(0.95, verifier_decision="support")], POLICY)
    assert status is MatchStatus.MATCHED


def test_a_legacy_resolution_forces_review_even_at_a_high_score():
    status, reason, _ = derive_status(
        [matched(0.95, resolution=Resolution.LEGACY_RANK.value)], POLICY
    )
    assert status is MatchStatus.NEEDS_REVIEW
    assert reason is DecisionReason.UNRESOLVED_OUTPUT


# --- failure ---------------------------------------------------------------------


def test_all_attempts_erroring_is_failed_not_unmatched():
    """A provider 500 is not evidence of a non-match."""
    status, reason, _ = derive_status(
        [attempt(error="HTTP 500", reason=DecisionReason.PROVIDER_FAILURE)] * 2, POLICY
    )
    assert status is MatchStatus.FAILED
    assert reason is DecisionReason.PROVIDER_FAILURE


def test_retriever_failure_is_distinguishable_from_provider_failure():
    status, reason, _ = derive_status(
        [attempt(error="timeout", reason=DecisionReason.RETRIEVER_FAILURE)], POLICY
    )
    assert reason is DecisionReason.RETRIEVER_FAILURE


def test_one_good_attempt_beats_one_failed_attempt():
    status, _, _ = derive_status(
        [attempt(error="HTTP 500", reason=DecisionReason.PROVIDER_FAILURE), matched(0.9)], POLICY
    )
    assert status is MatchStatus.MATCHED


def test_no_attempts_at_all_is_failed():
    status, reason, best = derive_status([], POLICY)
    assert status is MatchStatus.FAILED and best is None


# --- verify band and audit -------------------------------------------------------


def test_verify_band_is_inclusive_at_both_ends():
    policy = MatchPolicy(verify_band=(0.6, 0.8))
    assert should_verify(0.6, policy) and should_verify(0.8, policy)
    assert not should_verify(0.59, policy) and not should_verify(0.81, policy)


def test_verify_band_of_none_disables_verification():
    assert not should_verify(0.7, MatchPolicy(verify_band=None))


def test_a_none_score_is_never_verified():
    assert not should_verify(None, MatchPolicy())


def test_audit_is_off_by_default():
    assert not should_audit(0.95, MatchPolicy(), "fp", "s1")


def test_audit_only_applies_above_the_verify_band():
    policy = MatchPolicy(audit_rate=1.0, verify_band=(0.6, 0.8))
    assert should_audit(0.95, policy, "fp", "s1")
    assert not should_audit(0.7, policy, "fp", "s1")


def test_audit_sampling_is_deterministic_for_the_same_run_and_record():
    policy = MatchPolicy(audit_rate=0.5)
    first = should_audit(0.95, policy, "fp", "s1")
    second = should_audit(0.95, policy, "fp", "s1")
    assert first == second


def test_audit_sampling_differs_across_records():
    policy = MatchPolicy(audit_rate=0.5)
    picks = [should_audit(0.95, policy, "fp", f"s{i}") for i in range(200)]
    assert 0 < sum(picks) < 200


def test_audit_rate_is_approximately_honoured():
    policy = MatchPolicy(audit_rate=0.2)
    picks = [should_audit(0.95, policy, "fp", f"s{i}") for i in range(2000)]
    assert 0.15 < sum(picks) / 2000 < 0.25


# --- policy validation -----------------------------------------------------------


def test_review_floor_above_accept_at_is_rejected():
    with pytest.raises(ValueError, match="review_floor"):
        MatchPolicy(accept_at=0.5, review_floor=0.7)


def test_a_negative_audit_rate_is_rejected():
    with pytest.raises(ValueError, match="audit_rate"):
        MatchPolicy(audit_rate=-0.1)


def test_an_inverted_verify_band_is_rejected():
    with pytest.raises(ValueError, match="verify_band"):
        MatchPolicy(verify_band=(0.8, 0.6))


def test_max_attempts_below_one_is_rejected():
    with pytest.raises(ValueError, match="max_attempts"):
        MatchPolicy(max_attempts=0)
```

- [ ] **Step 2: Add the result types to `src/xwalk/records.py`**

As in Task 12, the import belongs in the block at the top of the file, not where it appears below.

```python
from enum import Enum


class MatchStatus(Enum):
    MATCHED = "matched"
    NEEDS_REVIEW = "needs_review"
    UNMATCHED = "unmatched"
    FAILED = "failed"


class DecisionReason(Enum):
    ACCEPT_THRESHOLD = "accept_threshold"
    BELOW_ACCEPT_THRESHOLD = "below_accept_threshold"
    BELOW_REVIEW_FLOOR = "below_review_floor"
    SELECTOR_ABSTAINED = "selector_abstained"
    NO_CANDIDATES = "no_candidates"
    UNRESOLVED_OUTPUT = "unresolved_output"
    RETRIEVER_FAILURE = "retriever_failure"
    PROVIDER_FAILURE = "provider_failure"
    VERIFIER_DISAGREEMENT = "verifier_disagreement"


@dataclass(frozen=True)
class Attempt:
    """One pass through retrieve -> select -> gate. The complete audit trail."""

    index: int
    query: str
    proposal: RetryProposal | None
    candidates: tuple[Candidate, ...]
    candidate_count: int
    candidates_truncated: int
    issued_keys: Mapping[str, str]
    raw_selection: str | None
    chosen_id: str | None
    resolution: str
    primary_score: float | None
    explanation: str
    verifier_decision: str | None
    verifier_score: float | None
    verifier_preferred_id: str | None
    audited: bool
    dropped_proposals: tuple[tuple[str, str], ...]
    reason: DecisionReason | None
    error: str | None
    usage: Usage


@dataclass(frozen=True)
class MatchResult:
    result_key: str
    source_id: str
    source_hash: str
    matched_id: str | None
    matched_record: Record | None
    confidence: float | None
    status: MatchStatus
    reason: DecisionReason
    explanation: str
    candidates: tuple[Candidate, ...]
    attempts: tuple[Attempt, ...]
    usage: Usage
    run_fingerprint: str
```

`candidates` on `Attempt` may be stored empty when a run is configured for compact traces; `candidate_count` always holds the true number, so the ceiling diagnostic and `NO_CANDIDATES` derivation never depend on trace verbosity.

Two fields are worth their own justification, because both exist to keep a spec promise:

- **`explanation`** carries the selector's own words for choosing this candidate. `MatchResult.explanation` is derived from it. Without this field the only string available to the result is `Attempt.error`, and a `MatchResult.explanation` that always holds an error message is a lie — it would read as an explanation of the match to every consumer of `mapping.csv`.
- **`dropped_proposals`** is a tuple of `(value, reason)` pairs. The spec says a candidate proposal naming something outside the current set "is dropped and recorded on the attempt rather than silently promoted to a query." `route_proposals` already computes the reasons; without somewhere to put them the recording half of that sentence is unimplemented, and a run gives no signal that a model kept inventing keys.

`Attempt` has no field defaults on purpose — every construction site must state every field, so adding one later cannot silently leave a stale value behind. That also means these two fields must be present from *this* task: retrofitting them in Task 17 would break `serde.py`, both test fixtures, and two construction sites inside `matcher.py`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.policy'`.

- [ ] **Step 4: Write `src/xwalk/policy.py`**

```python
"""Policy: cost controls, classification thresholds, and status derivation.

`verify_band` is a *cost control* — buy a second opinion only when the first is
uncertain. `accept_at` and `review_floor` are *classification*. The paper repo conflates
these; separating them lets each be tuned without disturbing the other.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.records import Attempt, DecisionReason, MatchStatus
from xwalk.stages.keying import Resolution

_EXACT_RESOLUTIONS = frozenset({Resolution.EXACT_KEY.value})
_FAILURE_REASONS = frozenset({DecisionReason.RETRIEVER_FAILURE, DecisionReason.PROVIDER_FAILURE})


@dataclass(frozen=True)
class MatchPolicy:
    max_attempts: int = 4
    accept_at: float = 0.6
    review_floor: float = 0.4
    verify_band: tuple[float, float] | None = (0.6, 0.8)
    audit_rate: float = 0.0
    concurrency: int = 32
    legacy_id_resolution: bool = False
    retriever_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if not 0.0 <= self.review_floor <= self.accept_at <= 1.0:
            raise ValueError(
                f"need 0 <= review_floor ({self.review_floor}) <= accept_at ({self.accept_at}) <= 1"
            )
        if not 0.0 <= self.audit_rate <= 1.0:
            raise ValueError(f"audit_rate must be in [0, 1], got {self.audit_rate}")
        if self.verify_band is not None:
            low, high = self.verify_band
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(
                    f"verify_band must be an ordered pair in [0, 1], got {self.verify_band}"
                )
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")


def should_verify(score: float | None, policy: MatchPolicy) -> bool:
    """Verification is bought only where the primary score is genuinely uncertain."""
    if score is None or policy.verify_band is None:
        return False
    low, high = policy.verify_band
    return low <= score <= high


def should_audit(
    score: float | None,
    policy: MatchPolicy,
    run_fingerprint: str,
    source_id: str,
) -> bool:
    """Sample a fraction of otherwise-automatic high-confidence matches.

    High-confidence errors exist and are by definition invisible. Sampling decides
    *which* results get a second opinion, never whether the opinion counts — an audit
    verdict is honoured exactly like any other verdict.

    Seeded from (run_fingerprint, source_id) so audits are reproducible across resumes
    rather than re-rolled each time a run restarts.
    """
    if policy.audit_rate <= 0.0 or score is None:
        return False
    if policy.verify_band is not None and score <= policy.verify_band[1]:
        return False  # already covered by verification
    digest = hashlib.sha256(f"{run_fingerprint}\x00{source_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return draw < policy.audit_rate


def _is_failure(attempt: Attempt) -> bool:
    return attempt.reason in _FAILURE_REASONS


def _has_choice(attempt: Attempt) -> bool:
    return attempt.chosen_id is not None and attempt.primary_score is not None


def derive_status(
    attempts: Sequence[Attempt],
    policy: MatchPolicy,
) -> tuple[MatchStatus, DecisionReason, Attempt | None]:
    """Reduce a list of attempts to one status, one reason, and the winning attempt."""
    if not attempts:
        return MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE, None

    if all(_is_failure(a) for a in attempts):
        reason = attempts[-1].reason or DecisionReason.PROVIDER_FAILURE
        return MatchStatus.FAILED, reason, None

    scored = [a for a in attempts if _has_choice(a)]
    if scored:
        # highest score wins; ties go to the earlier attempt
        best = min(scored, key=lambda a: (-(a.primary_score or 0.0), a.index))

        if best.verifier_decision in ("disagree", "no_match"):
            return MatchStatus.NEEDS_REVIEW, DecisionReason.VERIFIER_DISAGREEMENT, best

        if best.resolution not in _EXACT_RESOLUTIONS:
            # legacy or otherwise inexact resolution never becomes an automatic match
            return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT, best

        score = best.primary_score or 0.0
        if score >= policy.accept_at:
            return MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD, best
        if score >= policy.review_floor:
            return MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD, best
        return MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR, best

    if any(a.resolution == Resolution.UNRESOLVED.value for a in attempts):
        last = next(a for a in reversed(attempts) if a.resolution == Resolution.UNRESOLVED.value)
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT, last

    if any(a.candidate_count > 0 for a in attempts):
        last = next(a for a in reversed(attempts) if a.candidate_count > 0)
        return MatchStatus.UNMATCHED, DecisionReason.SELECTOR_ABSTAINED, last

    return MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES, attempts[-1]
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_policy.py -v`
Expected: 31 passed. `test_a_weak_pick_outranks_an_earlier_abstention` is the one that pins precedence rule 3 above rule 4 — if it fails, the ordering in `derive_status` is wrong, not the test.

- [ ] **Step 6: Export the new names**

Add `Attempt`, `DecisionReason`, `MatchResult`, `MatchStatus`, `MatchPolicy` to `src/xwalk/__init__.py`, keeping `__version__` above the import block. Task 18 Step 3 writes the final version of this file.

- [ ] **Step 7: Lint, type-check, commit**

```bash
python -m pytest -q -m "not integration"
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/records.py src/xwalk/policy.py src/xwalk/__init__.py tests/test_policy.py
git commit -m "feat: result types, match policy, and deterministic status derivation"
```

---

## Task 15: The SQLite run ledger

**Files:**
- Create: `src/xwalk/ledger.py`, `src/xwalk/serde.py`, `src/xwalk/llm/cache.py`, `tests/__init__.py`
- Modify: `src/xwalk/llm/__init__.py`
- Test: `tests/test_serde.py`, `tests/test_ledger.py`, `tests/test_llm_cache.py`

**Interfaces:**
- Consumes: `MatchResult`, `Attempt`, `Candidate`, `Record`, `RetrievalHit`, `Usage`, `MatchStatus`, `DecisionReason`, `RetryProposal` (Tasks 1, 12, 14), `hash_value` (Task 2).
- Produces: `result_to_dict(result) -> dict`, `result_from_dict(data) -> MatchResult` in `serde.py`; `Ledger.open(path) -> Ledger` with `async put_result(result)`, `get_result(result_key) -> MatchResult | None`, `has_result(result_key) -> bool`, `iter_results(run_fingerprint) -> Iterator[MatchResult]`, `count(run_fingerprint) -> int`, `count_by_status(run_fingerprint) -> dict[MatchStatus, int]`, `duplicate_targets(run_fingerprint) -> dict[str, list[str]]`, `get_cached(cache_key) -> str | None`, `async put_cached(cache_key, text)`, `put_manifest(run_fingerprint, manifest)`, `get_manifest(run_fingerprint) -> dict | None`, `put_review(row)`, `iter_reviews(run_fingerprint)`, `close()`; `llm_cache_key(client, request) -> str` and `CachingLLM(inner, ledger, *, read=True, write=True)` in `llm/cache.py`. Tasks 16 and 18 consume the ledger; `CachingLLM` wraps any client the user passes to `Matcher`.

**Why SQLite and not JSONL:** concurrent append to JSONL invites partial final lines and duplicate records, which complicates exactly the resume logic that has to be trustworthy. `results.jsonl`, `mapping.csv` and `manifest.json` become **exports**, regenerated from the ledger on demand. Writes go through a single writer coroutine; WAL keeps concurrent readers working.

- [ ] **Step 1: Write the failing serde test**

Create `tests/test_serde.py`:

```python
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)
from xwalk.serde import result_from_dict, result_to_dict


def sample_result() -> MatchResult:
    record = Record(id="T1", fields={"label": "glucose", "synonyms": ["dextrose"]})
    hit = RetrievalHit(record_id="T1", retriever="bm25", raw_score=3.5, rank=1)
    candidate = Candidate(record=record, fused_score=0.016, evidence=(hit,))
    attempt = Attempt(
        index=0,
        query="glucose",
        proposal=RetryProposal(kind="query", value="dextrose", source="rewriter"),
        candidates=(candidate,),
        candidate_count=1,
        candidates_truncated=3,
        issued_keys={"C01": "T1"},
        raw_selection='{"chosen_key": "C01"}',
        chosen_id="T1",
        resolution="exact_key",
        primary_score=0.91,
        explanation="exact synonym match",
        verifier_decision="support",
        verifier_score=0.88,
        verifier_preferred_id=None,
        audited=True,
        dropped_proposals=(("C99", "candidate key 'C99' was not issued this attempt"),),
        reason=DecisionReason.ACCEPT_THRESHOLD,
        error=None,
        usage=Usage(prompt_tokens=100, completion_tokens=20, calls=2),
    )
    return MatchResult(
        result_key="rk1",
        source_id="s1",
        source_hash="sh1",
        matched_id="T1",
        matched_record=record,
        confidence=0.91,
        status=MatchStatus.MATCHED,
        reason=DecisionReason.ACCEPT_THRESHOLD,
        explanation="exact synonym match",
        candidates=(candidate,),
        attempts=(attempt,),
        usage=Usage(prompt_tokens=100, completion_tokens=20, calls=2),
        run_fingerprint="fp1",
    )


def test_round_trip_preserves_everything():
    original = sample_result()
    assert result_from_dict(result_to_dict(original)) == original


def test_enums_serialise_as_their_values():
    data = result_to_dict(sample_result())
    assert data["status"] == "matched"
    assert data["reason"] == "accept_threshold"


def test_evidence_survives_the_round_trip():
    restored = result_from_dict(result_to_dict(sample_result()))
    assert restored.candidates[0].evidence[0].retriever == "bm25"


def test_a_none_matched_record_round_trips():
    original = sample_result()
    unmatched = MatchResult(
        **{
            **original.__dict__,
            "matched_id": None,
            "matched_record": None,
            "status": MatchStatus.UNMATCHED,
            "reason": DecisionReason.NO_CANDIDATES,
        }
    )
    assert result_from_dict(result_to_dict(unmatched)).matched_record is None


def test_a_none_proposal_round_trips():
    original = sample_result()
    attempt = Attempt(**{**original.attempts[0].__dict__, "proposal": None})
    result = MatchResult(**{**original.__dict__, "attempts": (attempt,)})
    assert result_from_dict(result_to_dict(result)).attempts[0].proposal is None


def test_the_serialised_form_is_json_safe():
    import json

    json.dumps(result_to_dict(sample_result()))  # must not raise
```

- [ ] **Step 2: Write `src/xwalk/serde.py`**

```python
"""Converting results to and from plain JSON-safe dicts.

The ledger stores one JSON blob per result. Keeping the conversion here means the
storage layer never has to know the shape of a MatchResult.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)


def _record_to_dict(record: Record) -> dict[str, Any]:
    return {"id": record.id, "fields": dict(record.fields)}


def _record_from_dict(data: Mapping[str, Any]) -> Record:
    return Record(id=data["id"], fields=data["fields"])


def _usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "calls": usage.calls,
    }


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "record": _record_to_dict(candidate.record),
        "fused_score": candidate.fused_score,
        "evidence": [
            {
                "record_id": hit.record_id,
                "retriever": hit.retriever,
                "raw_score": hit.raw_score,
                "rank": hit.rank,
            }
            for hit in candidate.evidence
        ],
    }


def _candidate_from_dict(data: Mapping[str, Any]) -> Candidate:
    return Candidate(
        record=_record_from_dict(data["record"]),
        fused_score=data["fused_score"],
        evidence=tuple(RetrievalHit(**hit) for hit in data["evidence"]),
    )


def _attempt_to_dict(attempt: Attempt) -> dict[str, Any]:
    return {
        "index": attempt.index,
        "query": attempt.query,
        "proposal": (
            None
            if attempt.proposal is None
            else {
                "kind": attempt.proposal.kind,
                "value": attempt.proposal.value,
                "source": attempt.proposal.source,
            }
        ),
        "candidates": [_candidate_to_dict(c) for c in attempt.candidates],
        "candidate_count": attempt.candidate_count,
        "candidates_truncated": attempt.candidates_truncated,
        "issued_keys": dict(attempt.issued_keys),
        "raw_selection": attempt.raw_selection,
        "chosen_id": attempt.chosen_id,
        "resolution": attempt.resolution,
        "primary_score": attempt.primary_score,
        "explanation": attempt.explanation,
        "verifier_decision": attempt.verifier_decision,
        "verifier_score": attempt.verifier_score,
        "verifier_preferred_id": attempt.verifier_preferred_id,
        "audited": attempt.audited,
        # JSON has no tuples; restored as tuple-of-tuples on the way back in.
        "dropped_proposals": [list(pair) for pair in attempt.dropped_proposals],
        "reason": None if attempt.reason is None else attempt.reason.value,
        "error": attempt.error,
        "usage": _usage_to_dict(attempt.usage),
    }


def _attempt_from_dict(data: Mapping[str, Any]) -> Attempt:
    return Attempt(
        index=data["index"],
        query=data["query"],
        proposal=None if data["proposal"] is None else RetryProposal(**data["proposal"]),
        candidates=tuple(_candidate_from_dict(c) for c in data["candidates"]),
        candidate_count=data["candidate_count"],
        candidates_truncated=data["candidates_truncated"],
        issued_keys=data["issued_keys"],
        raw_selection=data["raw_selection"],
        chosen_id=data["chosen_id"],
        resolution=data["resolution"],
        primary_score=data["primary_score"],
        explanation=data["explanation"],
        verifier_decision=data["verifier_decision"],
        verifier_score=data["verifier_score"],
        verifier_preferred_id=data["verifier_preferred_id"],
        audited=data["audited"],
        dropped_proposals=tuple((str(pair[0]), str(pair[1])) for pair in data["dropped_proposals"]),
        reason=None if data["reason"] is None else DecisionReason(data["reason"]),
        error=data["error"],
        usage=Usage(**data["usage"]),
    )


def result_to_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "result_key": result.result_key,
        "source_id": result.source_id,
        "source_hash": result.source_hash,
        "matched_id": result.matched_id,
        "matched_record": (
            None if result.matched_record is None else _record_to_dict(result.matched_record)
        ),
        "confidence": result.confidence,
        "status": result.status.value,
        "reason": result.reason.value,
        "explanation": result.explanation,
        "candidates": [_candidate_to_dict(c) for c in result.candidates],
        "attempts": [_attempt_to_dict(a) for a in result.attempts],
        "usage": _usage_to_dict(result.usage),
        "run_fingerprint": result.run_fingerprint,
    }


def result_from_dict(data: Mapping[str, Any]) -> MatchResult:
    return MatchResult(
        result_key=data["result_key"],
        source_id=data["source_id"],
        source_hash=data["source_hash"],
        matched_id=data["matched_id"],
        matched_record=(
            None if data["matched_record"] is None else _record_from_dict(data["matched_record"])
        ),
        confidence=data["confidence"],
        status=MatchStatus(data["status"]),
        reason=DecisionReason(data["reason"]),
        explanation=data["explanation"],
        candidates=tuple(_candidate_from_dict(c) for c in data["candidates"]),
        attempts=tuple(_attempt_from_dict(a) for a in data["attempts"]),
        usage=Usage(**data["usage"]),
        run_fingerprint=data["run_fingerprint"],
    )
```

- [ ] **Step 3: Run the serde tests**

Run: `python -m pytest tests/test_serde.py -v`
Expected: 6 passed. `Record.fields` restores as a plain dict, so equality holds against the original `Mapping`.

- [ ] **Step 4: Create `tests/__init__.py`**

From here on, `tests/test_ledger.py`, `tests/test_review.py` and `tests/test_batch.py` import shared fixtures from sibling test modules (`from tests.test_serde import sample_result`). That only works if `tests/` is a package.

```bash
touch tests/__init__.py
```

- [ ] **Step 5: Write the failing ledger test**

Create `tests/test_ledger.py`:

```python
import asyncio

import pytest

from tests.test_serde import sample_result
from xwalk.ledger import Ledger
from xwalk.records import MatchStatus


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    yield led
    led.close()


async def test_put_then_get_round_trips(ledger):
    result = sample_result()
    await ledger.put_result(result)
    assert ledger.get_result(result.result_key) == result


def test_get_returns_none_for_an_unknown_key(ledger):
    assert ledger.get_result("nope") is None


async def test_has_result_is_true_after_a_put(ledger):
    result = sample_result()
    assert not ledger.has_result(result.result_key)
    await ledger.put_result(result)
    assert ledger.has_result(result.result_key)


async def test_putting_the_same_key_twice_replaces_rather_than_duplicates(ledger):
    result = sample_result()
    await ledger.put_result(result)
    await ledger.put_result(result)
    assert ledger.count(result.run_fingerprint) == 1


async def test_iter_results_is_scoped_to_a_run_fingerprint(ledger):
    from xwalk.records import MatchResult

    first = sample_result()
    second = MatchResult(**{**first.__dict__, "result_key": "rk2", "run_fingerprint": "fp2"})
    await ledger.put_result(first)
    await ledger.put_result(second)
    assert [r.result_key for r in ledger.iter_results("fp1")] == ["rk1"]
    assert ledger.count("fp2") == 1


async def test_iter_results_is_ordered_by_source_id(ledger):
    from xwalk.records import MatchResult

    first = sample_result()
    for sid in ["s3", "s1", "s2"]:
        await ledger.put_result(
            MatchResult(**{**first.__dict__, "result_key": f"rk-{sid}", "source_id": sid})
        )
    assert [r.source_id for r in ledger.iter_results("fp1")] == ["s1", "s2", "s3"]


async def test_status_is_queryable_without_deserialising_every_blob(ledger):
    await ledger.put_result(sample_result())
    assert ledger.count_by_status("fp1") == {MatchStatus.MATCHED: 1}


# --- cache ----------------------------------------------------------------------


async def test_cache_round_trips(ledger):
    await ledger.put_cached("ck1", '{"a": 1}')
    assert ledger.get_cached("ck1") == '{"a": 1}'


def test_cache_miss_returns_none(ledger):
    assert ledger.get_cached("nope") is None


# --- manifest -------------------------------------------------------------------


async def test_manifest_round_trips(ledger):
    ledger.put_manifest("fp1", {"model": "gpt-4o", "target": "t1"})
    assert ledger.get_manifest("fp1") == {"model": "gpt-4o", "target": "t1"}


def test_manifest_miss_returns_none(ledger):
    assert ledger.get_manifest("nope") is None


# --- durability -----------------------------------------------------------------


async def test_results_survive_reopening_the_file(tmp_path):
    path = tmp_path / "run.sqlite"
    led = Ledger.open(path)
    await led.put_result(sample_result())
    led.close()

    reopened = Ledger.open(path)
    assert reopened.has_result("rk1")
    reopened.close()


async def test_wal_mode_is_enabled(ledger):
    assert ledger.journal_mode.lower() == "wal"


async def test_concurrent_writes_all_land(ledger):
    from xwalk.records import MatchResult

    base = sample_result()
    results = [
        MatchResult(**{**base.__dict__, "result_key": f"rk{i}", "source_id": f"s{i}"})
        for i in range(50)
    ]
    await asyncio.gather(*(ledger.put_result(r) for r in results))
    assert ledger.count("fp1") == 50


async def test_a_crash_mid_run_leaves_completed_results_readable(tmp_path):
    """Simulate a kill: write some results, drop the object without closing, reopen."""
    from xwalk.records import MatchResult

    path = tmp_path / "run.sqlite"
    led = Ledger.open(path)
    base = sample_result()
    for i in range(5):
        await led.put_result(
            MatchResult(**{**base.__dict__, "result_key": f"rk{i}", "source_id": f"s{i}"})
        )
    del led  # no close(), no flush

    reopened = Ledger.open(path)
    assert reopened.count("fp1") == 5
    reopened.close()
```

- [ ] **Step 6: Write `src/xwalk/ledger.py`**

```python
"""The run ledger: transactional, resumable execution state.

Every completed result is committed as it finishes, so a batch run never loses work.
JSONL/CSV/manifest files are exports regenerated from here, never the source of truth.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from xwalk.records import MatchResult, MatchStatus
from xwalk.serde import result_from_dict, result_to_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    result_key       TEXT PRIMARY KEY,
    run_fingerprint  TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    source_hash      TEXT NOT NULL,
    status           TEXT NOT NULL,
    reason           TEXT NOT NULL,
    matched_id       TEXT,
    confidence       REAL,
    blob             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_by_run ON results(run_fingerprint, source_id);
CREATE INDEX IF NOT EXISTS results_by_target ON results(run_fingerprint, matched_id);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manifests (
    run_fingerprint TEXT PRIMARY KEY,
    blob            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    result_key        TEXT NOT NULL,
    run_fingerprint   TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_hash       TEXT NOT NULL,
    proposed_target_id TEXT,
    decision          TEXT NOT NULL,
    corrected_target_id TEXT,
    reviewer          TEXT NOT NULL,
    review_note       TEXT NOT NULL DEFAULT '',
    reviewed_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reviews_by_run ON reviews(run_fingerprint, result_key);
"""


class Ledger:
    """SQLite in WAL mode, with a lock serialising writers.

    Reads go straight through — WAL allows concurrent readers. Writes take an asyncio
    lock so a batch run with concurrency 32 never interleaves two transactions.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._write_lock = asyncio.Lock()

    @classmethod
    def open(cls, path: str | Path) -> Ledger:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        return cls(conn)

    @property
    def journal_mode(self) -> str:
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    # --- results ---------------------------------------------------------------

    async def put_result(self, result: MatchResult) -> None:
        blob = json.dumps(result_to_dict(result), ensure_ascii=False)
        async with self._write_lock:
            self._conn.execute(
                """
                INSERT INTO results
                    (result_key, run_fingerprint, source_id, source_hash,
                     status, reason, matched_id, confidence, blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(result_key) DO UPDATE SET
                    run_fingerprint=excluded.run_fingerprint,
                    source_id=excluded.source_id,
                    source_hash=excluded.source_hash,
                    status=excluded.status,
                    reason=excluded.reason,
                    matched_id=excluded.matched_id,
                    confidence=excluded.confidence,
                    blob=excluded.blob
                """,
                (
                    result.result_key,
                    result.run_fingerprint,
                    result.source_id,
                    result.source_hash,
                    result.status.value,
                    result.reason.value,
                    result.matched_id,
                    result.confidence,
                    blob,
                ),
            )

    def get_result(self, result_key: str) -> MatchResult | None:
        row = self._conn.execute(
            "SELECT blob FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return None if row is None else result_from_dict(json.loads(row["blob"]))

    def has_result(self, result_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE result_key = ?", (result_key,)
        ).fetchone()
        return row is not None

    def iter_results(self, run_fingerprint: str) -> Iterator[MatchResult]:
        cursor = self._conn.execute(
            "SELECT blob FROM results WHERE run_fingerprint = ? ORDER BY source_id",
            (run_fingerprint,),
        )
        for row in cursor:
            yield result_from_dict(json.loads(row["blob"]))

    def count(self, run_fingerprint: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM results WHERE run_fingerprint = ?", (run_fingerprint,)
        ).fetchone()
        return int(row["n"])

    def count_by_status(self, run_fingerprint: str) -> dict[MatchStatus, int]:
        cursor = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM results WHERE run_fingerprint = ? GROUP BY status",
            (run_fingerprint,),
        )
        return {MatchStatus(row["status"]): int(row["n"]) for row in cursor}

    def duplicate_targets(self, run_fingerprint: str) -> dict[str, list[str]]:
        """Several source records selected the same target. Reported, never resolved."""
        cursor = self._conn.execute(
            """
            SELECT matched_id, GROUP_CONCAT(source_id) AS sources
            FROM results
            WHERE run_fingerprint = ? AND matched_id IS NOT NULL
            GROUP BY matched_id HAVING COUNT(*) > 1
            """,
            (run_fingerprint,),
        )
        return {row["matched_id"]: sorted(row["sources"].split(",")) for row in cursor}

    # --- llm cache -------------------------------------------------------------

    def get_cached(self, cache_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return None if row is None else str(row["response"])

    async def put_cached(self, cache_key: str, response: str) -> None:
        async with self._write_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, response) VALUES (?, ?)",
                (cache_key, response),
            )

    # --- manifest --------------------------------------------------------------

    def put_manifest(self, run_fingerprint: str, manifest: Mapping[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO manifests (run_fingerprint, blob) VALUES (?, ?)",
            (run_fingerprint, json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
        )

    def get_manifest(self, run_fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT blob FROM manifests WHERE run_fingerprint = ?", (run_fingerprint,)
        ).fetchone()
        return None if row is None else dict(json.loads(row["blob"]))

    # --- reviews ---------------------------------------------------------------

    def put_review(self, row: Mapping[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO reviews
                (result_key, run_fingerprint, source_id, source_hash, proposed_target_id,
                 decision, corrected_target_id, reviewer, review_note, reviewed_at)
            VALUES (:result_key, :run_fingerprint, :source_id, :source_hash,
                    :proposed_target_id, :decision, :corrected_target_id, :reviewer,
                    :review_note, :reviewed_at)
            """,
            dict(row),
        )

    def iter_reviews(self, run_fingerprint: str) -> Iterator[dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM reviews WHERE run_fingerprint = ? ORDER BY review_id",
            (run_fingerprint,),
        )
        for row in cursor:
            yield dict(row)

    def close(self) -> None:
        self._conn.close()
```

`isolation_level=None` puts the connection in autocommit, so every statement is its own durable transaction — that is what makes the crash-mid-run test pass without an explicit flush.

- [ ] **Step 7: Run the ledger tests**

Run: `python -m pytest tests/test_ledger.py -v`
Expected: 15 passed.

- [ ] **Step 8: Write the failing cache test**

The `llm_cache` table now exists but nothing writes to it. The spec is explicit that the cache is part of the ledger *and* explicit about the key: "`(model, rendered_prompt)` is not a sufficient LLM cache key. The key covers the complete request body, the schema, provider identity, model parameters, and adapter version." A cache keyed on less than that returns a stale answer when the schema or temperature changes — the answer to a question that was not asked.

Create `tests/test_llm_cache.py`:

```python
import pytest

from xwalk.ledger import Ledger
from xwalk.llm.base import LLMCapabilities, LLMClient, LLMRequest
from xwalk.llm.cache import CachingLLM, llm_cache_key
from xwalk.llm.fake import FakeLLM

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    yield led
    led.close()


def wrap(ledger, script):
    inner = FakeLLM(script)
    return inner, CachingLLM(inner, ledger)


async def test_a_repeated_request_is_served_from_the_cache(ledger):
    inner, cached = wrap(ledger, ["first"])
    request = LLMRequest(system="s", user="u")
    assert (await cached.complete(request)).text == "first"
    assert (await cached.complete(request)).text == "first"
    assert len(inner.requests) == 1  # the script would have raised on a second call


async def test_a_different_user_prompt_misses(ledger):
    inner, cached = wrap(ledger, ["a", "b"])
    assert (await cached.complete(LLMRequest(system="s", user="one"))).text == "a"
    assert (await cached.complete(LLMRequest(system="s", user="two"))).text == "b"


async def test_a_different_system_prompt_misses(ledger):
    inner, cached = wrap(ledger, ["a", "b"])
    await cached.complete(LLMRequest(system="one", user="u"))
    await cached.complete(LLMRequest(system="two", user="u"))
    assert len(inner.requests) == 2


def test_the_key_is_not_vulnerable_to_prompt_boundary_collisions(ledger):
    """system='ab', user='c' and system='a', user='bc' are different requests."""
    llm = FakeLLM(["x"])
    assert llm_cache_key(llm, LLMRequest(system="ab", user="c")) != llm_cache_key(
        llm, LLMRequest(system="a", user="bc")
    )


def test_the_key_covers_the_schema(ledger):
    llm = FakeLLM(["x"])
    assert llm_cache_key(llm, LLMRequest(system="", user="u")) != llm_cache_key(
        llm, LLMRequest(system="", user="u", schema=SCHEMA)
    )


def test_the_key_covers_generation_parameters(ledger):
    llm = FakeLLM(["x"])
    assert llm_cache_key(llm, LLMRequest(system="", user="u", temperature=0.0)) != llm_cache_key(
        llm, LLMRequest(system="", user="u", temperature=0.7)
    )


def test_the_key_covers_provider_identity(ledger):
    """Same prompt, different client — never the same cache entry."""
    request = LLMRequest(system="", user="u")
    assert llm_cache_key(FakeLLM(["x"]), request) != llm_cache_key(FakeLLM(["y"]), request)


async def test_a_cache_hit_reports_no_token_spend(ledger):
    inner, cached = wrap(ledger, ["first"])
    request = LLMRequest(system="s", user="u")
    await cached.complete(request)
    second = await cached.complete(request)
    assert second.usage.calls == 0
    assert second.usage.total_tokens == 0


async def test_hit_and_miss_counts_are_tracked(ledger):
    inner, cached = wrap(ledger, ["first"])
    request = LLMRequest(system="s", user="u")
    await cached.complete(request)
    await cached.complete(request)
    assert (cached.hits, cached.misses) == (1, 1)


async def test_write_can_be_disabled(ledger):
    inner = FakeLLM(["a", "b"])
    cached = CachingLLM(inner, ledger, write=False)
    request = LLMRequest(system="s", user="u")
    await cached.complete(request)
    await cached.complete(request)
    assert len(inner.requests) == 2


async def test_the_cache_survives_reopening_the_ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    await CachingLLM(FakeLLM(["first"]), led).complete(LLMRequest(system="s", user="u"))
    led.close()

    reopened = Ledger.open(tmp_path / "run.sqlite")
    # The same script gives the same client fingerprint, hence the same cache key. A
    # client scripted differently is a *different* provider identity and must miss —
    # that is the point of putting `client.fingerprint` in the key.
    fresh = FakeLLM(["first"])
    response = await CachingLLM(fresh, reopened).complete(LLMRequest(system="s", user="u"))
    reopened.close()
    assert response.text == "first"
    assert fresh.requests == []  # answered from disk, never delegated


def test_identity_delegates_to_the_inner_client(ledger):
    inner = FakeLLM(["x"], capabilities=LLMCapabilities(json_schema=True), model="m1")
    cached = CachingLLM(inner, ledger)
    assert cached.model == "m1"
    assert cached.capabilities.json_schema is True
    # The wrapper must be invisible to run fingerprinting: caching changes how an
    # answer was obtained, never what the answer means.
    assert cached.fingerprint == inner.fingerprint


def test_caching_llm_satisfies_the_llm_client_protocol(ledger):
    assert isinstance(CachingLLM(FakeLLM(["x"]), ledger), LLMClient)
```

- [ ] **Step 9: Write `src/xwalk/llm/cache.py`**

```python
"""Ledger-backed LLM response caching.

Wraps any `LLMClient`. Identity (`model`, `capabilities`, `fingerprint`) delegates to
the inner client, so wrapping a client never changes a run fingerprint — caching changes
how an answer was obtained, never what it means.
"""

from __future__ import annotations

from xwalk.fingerprint import hash_value
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMCapabilities, LLMClient, LLMRequest, LLMResponse
from xwalk.records import Usage

# Bump when the stored representation changes, to invalidate old entries rather than
# misread them.
CACHE_VERSION = 1


def llm_cache_key(client: LLMClient, request: LLMRequest) -> str:
    """Everything the provider actually sees, plus who it was sent to.

    The parts go in as a mapping rather than a concatenated string: `system="ab"` with
    `user="c"` must not key the same as `system="a"` with `user="bc"`.
    """
    return hash_value(
        {
            "cache_version": CACHE_VERSION,
            # Covers adapter version, endpoint, model, and default generation params.
            "client": client.fingerprint,
            "system": request.system,
            "user": request.user,
            "schema": dict(request.schema) if request.schema is not None else None,
            "schema_name": request.schema_name,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "extra": dict(request.extra),
        }
    )


class CachingLLM:
    """An `LLMClient` that serves repeats of an identical request from the ledger."""

    def __init__(
        self,
        inner: LLMClient,
        ledger: Ledger,
        *,
        read: bool = True,
        write: bool = True,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._read = read
        self._write = write
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._inner.capabilities

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = llm_cache_key(self._inner, request)

        if self._read:
            cached = self._ledger.get_cached(key)
            if cached is not None:
                self.hits += 1
                # Zero usage is the honest number: a cache hit spends no tokens, so a
                # resumed run's reported cost stays a cost, not a replayed estimate.
                return LLMResponse(
                    text=cached,
                    usage=Usage.zero(),
                    model=self._inner.model,
                    structured=False,
                    finish_reason="cached",
                )

        response = await self._inner.complete(request)
        self.misses += 1
        if self._write:
            await self._ledger.put_cached(key, response.text)
        return response
```

Add `CachingLLM` and `llm_cache_key` to `src/xwalk/llm/__init__.py`.

- [ ] **Step 10: Run the cache tests**

Run: `python -m pytest tests/test_llm_cache.py -v`
Expected: 13 passed.

- [ ] **Step 11: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/ledger.py src/xwalk/serde.py src/xwalk/llm/cache.py src/xwalk/llm/__init__.py tests/__init__.py tests/test_serde.py tests/test_ledger.py tests/test_llm_cache.py
git commit -m "feat: SQLite WAL run ledger with results, cache, manifest, and reviews"
```

---

## Task 16: The immutable review overlay

**Files:**
- Create: `src/xwalk/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `Ledger` (Task 15), `MatchResult`/`MatchStatus`/`DecisionReason` (Task 14).
- Produces: `ReviewDecision` enum (`accept`, `reject`, `replace`, `no_match`, `defer`); `ReviewRow` dataclass; `SnapshotMismatch` exception; `export_review(ledger, run_fingerprint, out_path, *, statuses=(NEEDS_REVIEW,)) -> int`; `read_review(path) -> list[ReviewRow]`; `ApplyReport(applied, rejected)`; `AdjudicatedResult(result_key, source_id, model_target_id, model_status, confidence, final_target_id, final_status, reviewer, review_note, reviewed_at)`; `apply_review(ledger, rows, *, target_store_fingerprint) -> ApplyReport`; `adjudicated(ledger, run_fingerprint) -> Iterator[AdjudicatedResult]`. Task 18 exports adjudicated rows.

**The invariant this task exists to protect:** applying review never overwrites model output. Three layers are preserved — the original model result, the reviewer decision, and the adjudicated result derived from both. And **applying a review fails if its source or target snapshot no longer matches the originating run**: a decision made against different data is not a decision about *this* data, and silently applying it would corrupt the mapping in the least detectable way possible.

- [ ] **Step 1: Write the failing test**

Create `tests/test_review.py`:

```python
import csv

import pytest

from tests.test_serde import sample_result
from xwalk.ledger import Ledger
from xwalk.records import DecisionReason, MatchResult, MatchStatus
from xwalk.review import (
    SnapshotMismatch,
    adjudicated,
    apply_review,
    export_review,
    read_review,
)


def needs_review(**kwargs) -> MatchResult:
    base = sample_result()
    return MatchResult(
        **{
            **base.__dict__,
            "status": MatchStatus.NEEDS_REVIEW,
            "reason": DecisionReason.BELOW_ACCEPT_THRESHOLD,
            **kwargs,
        }
    )


@pytest.fixture
async def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    led.put_manifest("fp1", {"target_fingerprint": "tf1"})
    await led.put_result(needs_review())
    yield led
    led.close()


# --- export ---------------------------------------------------------------------


async def test_export_writes_one_row_per_reviewable_result(ledger, tmp_path):
    out = tmp_path / "review.csv"
    assert export_review(ledger, "fp1", out) == 1
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1


async def test_export_carries_the_identity_columns(ledger, tmp_path):
    out = tmp_path / "review.csv"
    export_review(ledger, "fp1", out)
    row = next(iter(csv.DictReader(out.open(encoding="utf-8"))))
    for column in (
        "result_key",
        "run_fingerprint",
        "source_id",
        "source_hash",
        "proposed_target_id",
        "decision",
        "corrected_target_id",
        "reviewer",
        "review_note",
        "reviewed_at",
    ):
        assert column in row


async def test_export_leaves_the_decision_columns_blank_for_the_reviewer(ledger, tmp_path):
    out = tmp_path / "review.csv"
    export_review(ledger, "fp1", out)
    row = next(iter(csv.DictReader(out.open(encoding="utf-8"))))
    assert row["decision"] == "" and row["corrected_target_id"] == ""


async def test_export_only_includes_the_requested_statuses(ledger, tmp_path):
    await ledger.put_result(
        needs_review(
            result_key="rk2",
            source_id="s2",
            status=MatchStatus.MATCHED,
            reason=DecisionReason.ACCEPT_THRESHOLD,
        )
    )
    out = tmp_path / "review.csv"
    assert export_review(ledger, "fp1", out) == 1


async def test_export_can_include_matched_rows_for_spot_checking(ledger, tmp_path):
    await ledger.put_result(
        needs_review(
            result_key="rk2",
            source_id="s2",
            status=MatchStatus.MATCHED,
            reason=DecisionReason.ACCEPT_THRESHOLD,
        )
    )
    out = tmp_path / "review.csv"
    count = export_review(
        ledger, "fp1", out, statuses=(MatchStatus.NEEDS_REVIEW, MatchStatus.MATCHED)
    )
    assert count == 2


# --- apply ----------------------------------------------------------------------


def write_review(path, **overrides) -> None:
    row = {
        "result_key": "rk1",
        "run_fingerprint": "fp1",
        "source_id": "s1",
        "source_hash": "sh1",
        "proposed_target_id": "T1",
        "decision": "accept",
        "corrected_target_id": "",
        "reviewer": "jan",
        "review_note": "",
        "reviewed_at": "2026-07-26T10:00:00Z",
    }
    row.update(overrides)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


async def test_apply_accepts_a_decision(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path)
    report = apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    assert report.applied == 1 and report.rejected == []


async def test_apply_never_mutates_the_original_result(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="reject")
    before = ledger.get_result("rk1")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    assert ledger.get_result("rk1") == before


async def test_apply_records_the_decision_in_the_overlay(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="T9")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    stored = list(ledger.iter_reviews("fp1"))
    assert stored[0]["decision"] == "replace" and stored[0]["corrected_target_id"] == "T9"


async def test_apply_rejects_a_stale_source_hash(ledger, tmp_path):
    """A decision made against different data is not a decision about this data."""
    path = tmp_path / "r.csv"
    write_review(path, source_hash="DIFFERENT")
    with pytest.raises(SnapshotMismatch, match="source"):
        apply_review(ledger, read_review(path), target_store_fingerprint="tf1")


async def test_apply_rejects_a_stale_target_snapshot(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path)
    with pytest.raises(SnapshotMismatch, match="target"):
        apply_review(ledger, read_review(path), target_store_fingerprint="DIFFERENT")


async def test_apply_rejects_an_unknown_result_key(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, result_key="nope")
    with pytest.raises(SnapshotMismatch, match="result_key"):
        apply_review(ledger, read_review(path), target_store_fingerprint="tf1")


async def test_apply_rejects_an_unknown_decision(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="maybe")
    with pytest.raises(ValueError, match="decision"):
        read_review(path)


async def test_apply_requires_a_corrected_id_for_replace(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="")
    with pytest.raises(ValueError, match="corrected_target_id"):
        read_review(path)


async def test_apply_skips_blank_decisions_without_failing(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="")
    rows = read_review(path)
    assert rows == []


async def test_apply_requires_a_reviewer(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, reviewer="")
    with pytest.raises(ValueError, match="reviewer"):
        read_review(path)


# --- adjudicated view -----------------------------------------------------------


async def test_adjudicated_accept_keeps_the_model_target(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="accept")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_target_id == "T1" and row.final_status is MatchStatus.MATCHED


async def test_adjudicated_reject_clears_the_target(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="reject")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_target_id is None and row.final_status is MatchStatus.UNMATCHED


async def test_adjudicated_replace_uses_the_corrected_target(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="T9")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_target_id == "T9" and row.final_status is MatchStatus.MATCHED


async def test_adjudicated_defer_leaves_the_row_in_review(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="defer")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_status is MatchStatus.NEEDS_REVIEW


async def test_adjudicated_preserves_the_model_answer_alongside_the_final_one(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="replace", corrected_target_id="T9")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.model_target_id == "T1" and row.final_target_id == "T9"


async def test_the_latest_review_wins_when_a_row_is_reviewed_twice(ledger, tmp_path):
    path = tmp_path / "r.csv"
    write_review(path, decision="accept")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    write_review(path, decision="reject")
    apply_review(ledger, read_review(path), target_store_fingerprint="tf1")
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.final_status is MatchStatus.UNMATCHED
    assert len(list(ledger.iter_reviews("fp1"))) == 2  # both decisions retained


async def test_unreviewed_rows_appear_with_their_model_status(ledger, tmp_path):
    [row] = list(adjudicated(ledger, "fp1"))
    assert row.reviewer is None and row.final_status is MatchStatus.NEEDS_REVIEW
```

- [ ] **Step 2: Write `src/xwalk/review.py`**

```python
"""Human review as an immutable overlay.

Three layers are preserved and never collapsed: the model result (in `results`), the
reviewer decision (in `reviews`), and the adjudicated view derived from both.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from xwalk.ledger import Ledger
from xwalk.records import MatchStatus

COLUMNS = (
    "result_key",
    "run_fingerprint",
    "source_id",
    "source_hash",
    "proposed_target_id",
    "decision",
    "corrected_target_id",
    "reviewer",
    "review_note",
    "reviewed_at",
)


class SnapshotMismatch(Exception):
    """A review was made against data that no longer matches this run."""


class ReviewDecision(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REPLACE = "replace"
    NO_MATCH = "no_match"
    DEFER = "defer"


@dataclass(frozen=True)
class ReviewRow:
    result_key: str
    run_fingerprint: str
    source_id: str
    source_hash: str
    proposed_target_id: str | None
    decision: ReviewDecision
    corrected_target_id: str | None
    reviewer: str
    review_note: str
    reviewed_at: str


@dataclass(frozen=True)
class ApplyReport:
    applied: int
    rejected: list[tuple[str, str]]


@dataclass(frozen=True)
class AdjudicatedResult:
    result_key: str
    source_id: str
    model_target_id: str | None
    model_status: MatchStatus
    confidence: float | None
    final_target_id: str | None
    final_status: MatchStatus
    reviewer: str | None
    review_note: str
    reviewed_at: str | None


def export_review(
    ledger: Ledger,
    run_fingerprint: str,
    out_path: str | Path,
    *,
    statuses: Sequence[MatchStatus] = (MatchStatus.NEEDS_REVIEW,),
) -> int:
    """Write a CSV with identity columns filled and decision columns blank."""
    wanted = set(statuses)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for result in ledger.iter_results(run_fingerprint):
            if result.status not in wanted:
                continue
            writer.writerow(
                {
                    "result_key": result.result_key,
                    "run_fingerprint": result.run_fingerprint,
                    "source_id": result.source_id,
                    "source_hash": result.source_hash,
                    "proposed_target_id": result.matched_id or "",
                    "decision": "",
                    "corrected_target_id": "",
                    "reviewer": "",
                    "review_note": "",
                    "reviewed_at": "",
                }
            )
            written += 1
    return written


def read_review(path: str | Path) -> list[ReviewRow]:
    """Parse a reviewed CSV. Blank decisions are skipped; bad ones raise."""
    rows: list[ReviewRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_no, raw in enumerate(csv.DictReader(handle), start=2):
            decision_text = (raw.get("decision") or "").strip().lower()
            if not decision_text:
                continue
            try:
                decision = ReviewDecision(decision_text)
            except ValueError as exc:
                raise ValueError(
                    f"row {line_no}: unknown decision {decision_text!r}; "
                    f"expected one of {[d.value for d in ReviewDecision]}"
                ) from exc

            corrected = (raw.get("corrected_target_id") or "").strip() or None
            if decision is ReviewDecision.REPLACE and not corrected:
                raise ValueError(f"row {line_no}: decision 'replace' needs corrected_target_id")

            reviewer = (raw.get("reviewer") or "").strip()
            if not reviewer:
                raise ValueError(f"row {line_no}: reviewer must be set for an applied decision")

            rows.append(
                ReviewRow(
                    result_key=(raw.get("result_key") or "").strip(),
                    run_fingerprint=(raw.get("run_fingerprint") or "").strip(),
                    source_id=(raw.get("source_id") or "").strip(),
                    source_hash=(raw.get("source_hash") or "").strip(),
                    proposed_target_id=(raw.get("proposed_target_id") or "").strip() or None,
                    decision=decision,
                    corrected_target_id=corrected,
                    reviewer=reviewer,
                    review_note=(raw.get("review_note") or "").strip(),
                    reviewed_at=(raw.get("reviewed_at") or "").strip(),
                )
            )
    return rows


def apply_review(
    ledger: Ledger,
    rows: Sequence[ReviewRow],
    *,
    target_store_fingerprint: str,
) -> ApplyReport:
    """Validate every row against the originating run, then append to the overlay.

    Validation is all-or-nothing: one stale row fails the whole application, because a
    partially-applied review file is worse than an unapplied one.
    """
    for row in rows:
        result = ledger.get_result(row.result_key)
        if result is None:
            raise SnapshotMismatch(f"unknown result_key {row.result_key!r}")
        if result.source_hash != row.source_hash:
            raise SnapshotMismatch(
                f"{row.result_key}: source record changed since the run "
                f"({row.source_hash!r} -> {result.source_hash!r}); re-run before applying"
            )
        manifest = ledger.get_manifest(row.run_fingerprint) or {}
        recorded_target = manifest.get("target_fingerprint")
        if recorded_target is not None and recorded_target != target_store_fingerprint:
            raise SnapshotMismatch(
                f"{row.result_key}: target snapshot changed since the run "
                f"({recorded_target!r} -> {target_store_fingerprint!r}); re-run before applying"
            )

    for row in rows:
        ledger.put_review(
            {
                "result_key": row.result_key,
                "run_fingerprint": row.run_fingerprint,
                "source_id": row.source_id,
                "source_hash": row.source_hash,
                "proposed_target_id": row.proposed_target_id,
                "decision": row.decision.value,
                "corrected_target_id": row.corrected_target_id,
                "reviewer": row.reviewer,
                "review_note": row.review_note,
                "reviewed_at": row.reviewed_at,
            }
        )
    return ApplyReport(applied=len(rows), rejected=[])


def adjudicated(ledger: Ledger, run_fingerprint: str) -> Iterator[AdjudicatedResult]:
    """The derived view: model answer plus the latest reviewer decision, if any."""
    latest: dict[str, dict[str, object]] = {}
    for review in ledger.iter_reviews(run_fingerprint):
        latest[str(review["result_key"])] = review  # later rows overwrite earlier ones

    for result in ledger.iter_results(run_fingerprint):
        # A distinct name from the loop variable above: that one is a row, this one is
        # an optional lookup, and mypy --strict will not let one binding be both.
        decision_row = latest.get(result.result_key)
        if decision_row is None:
            yield AdjudicatedResult(
                result_key=result.result_key,
                source_id=result.source_id,
                model_target_id=result.matched_id,
                model_status=result.status,
                confidence=result.confidence,
                final_target_id=result.matched_id,
                final_status=result.status,
                reviewer=None,
                review_note="",
                reviewed_at=None,
            )
            continue

        decision = ReviewDecision(str(decision_row["decision"]))
        if decision is ReviewDecision.ACCEPT:
            final_id, final_status = result.matched_id, MatchStatus.MATCHED
        elif decision is ReviewDecision.REPLACE:
            final_id = str(decision_row["corrected_target_id"])
            final_status = MatchStatus.MATCHED
        elif decision in (ReviewDecision.REJECT, ReviewDecision.NO_MATCH):
            final_id, final_status = None, MatchStatus.UNMATCHED
        else:  # DEFER
            final_id, final_status = result.matched_id, MatchStatus.NEEDS_REVIEW

        yield AdjudicatedResult(
            result_key=result.result_key,
            source_id=result.source_id,
            model_target_id=result.matched_id,
            model_status=result.status,
            confidence=result.confidence,
            final_target_id=final_id,
            final_status=final_status,
            reviewer=str(decision_row["reviewer"]),
            review_note=str(decision_row["review_note"]),
            reviewed_at=str(decision_row["reviewed_at"]),
        )
```

- [ ] **Step 3: Run the tests**

Run: `python -m pytest tests/test_review.py -v`
Expected: 22 passed. `test_apply_accepts_a_decision` requires the manifest written by the `ledger` fixture — if it fails with a `SnapshotMismatch`, check that `put_manifest` stored `target_fingerprint`.

- [ ] **Step 4: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/review.py tests/test_review.py
git commit -m "feat: immutable review overlay with snapshot-mismatch refusal"
```

---

## Task 17: The matcher loop

**Files:**
- Create: `src/xwalk/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: everything from Tasks 1–14.
- Produces: `Matcher(*, templates, retrievers, store, selector, scorer, verifier, rewriter, policy=None, run_fingerprint="", retriever_limit=20, rrf_k=60, keep_candidates_in_trace=True)` — **every argument is keyword-only** — with `async match(record) -> MatchResult`, `match_sync(record) -> MatchResult`, and read-only properties `run_fingerprint`, `policy`, `store_fingerprint` (Task 18 reads all three). Task 18 drives it.

**The loop, per attempt:**

1. Render `query` and `context` from the source record (first attempt) or take the next queued query proposal.
2. Search every retriever concurrently with a per-retriever timeout. One retriever timing out degrades to the others rather than failing the match, and the degradation is recorded.
3. Fuse hits by RRF into `Candidate`s carrying their evidence, resolving IDs via the `TargetStore`.
4. No candidates → record the attempt and move to the next queued query.
5. Assign opaque keys, apply the selector budget, render the candidate list, select.
6. Score the choice against the **full rendered source record and context**, not the retrieval query.
7. If the score falls in `verify_band` — or the record is sampled for audit above the band — run verification.
8. Score ≥ `accept_at` → return immediately. Otherwise collect retry proposals and continue.

On exhaustion, `derive_status` picks the best attempt.

- [ ] **Step 1: Write the failing test**

Create `tests/test_matcher.py`:

```python
import asyncio
import json

from xwalk.llm.base import LLMFatalError, LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, PromptSlots
from xwalk.records import DecisionReason, MatchStatus, Record, RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context=(
        "{% if context_left %}{{ context_left }} [{{ mention }}] {{ context_right }}{% endif %}"
    ),
    doc="{{ label }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)
PROMPTS = PromptSet.from_slots(
    PromptSlots(
        entity_noun="mention",
        target_noun="term",
        domain_brief="test",
        rubric=[
            {"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.4, "name": "Weak", "when": "vague"},
        ],
    )
)
STORE = MemoryStore.from_source(
    [
        Record(id="T1", fields={"label": "glucose"}),
        Record(id="T2", fields={"label": "fructose"}),
        Record(id="T3", fields={"label": "sucrose"}),
    ]
)
SOURCE = Record(
    id="s1", fields={"mention": "glucose", "context_left": "blood", "context_right": "levels"}
)


class ScriptedRetriever:
    """Returns a fixed hit list per query text; unknown queries return nothing."""

    def __init__(self, by_query, *, name="bm25", fail=False, hang=False, default_limit=20):
        self._by_query = by_query
        self._name = name
        self._fail = fail
        self._hang = hang
        self._default_limit = default_limit
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return f"scripted-{self._name}"

    @property
    def default_limit(self) -> int:
        return self._default_limit

    async def search(self, request: SearchRequest):
        self.queries.append(request.text)
        if self._fail:
            raise RetrieverError("index unavailable")
        if self._hang:
            await asyncio.sleep(3600)
        ids = self._by_query.get(request.text, [])
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(ids, start=1)
        ]


def select_reply(key, score=0.9, explanation="ok"):
    return json.dumps({"chosen_key": key, "confidence_score": score, "explanation": explanation})


def score_reply(score, **extra):
    return json.dumps({"confidence_score": score, "explanation": "", **extra})


def verify_reply(decision="support", **extra):
    return json.dumps({"decision": decision, "explanation": "", **extra})


def rewrite_reply(*queries):
    return json.dumps({"queries": list(queries), "explanation": ""})


def build(llm, retriever, *, policy=None):
    policy = policy or MatchPolicy()
    return Matcher(
        templates=TEMPLATES,
        retrievers=[retriever],
        store=STORE,
        selector=Selector(
            llm, PROMPTS, TEMPLATES, legacy_id_resolution=policy.legacy_id_resolution
        ),
        scorer=Scorer(llm, PROMPTS, TEMPLATES, review_floor=policy.review_floor),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=policy,
        run_fingerprint="fp1",
    )


# --- happy path -----------------------------------------------------------------


async def test_a_confident_match_returns_immediately():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    retriever = ScriptedRetriever({"glucose": ["T1", "T2"]})
    result = await build(llm, retriever).match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert result.matched_id == "T1"
    assert result.confidence == 0.95
    assert len(result.attempts) == 1


async def test_the_matched_record_is_attached():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.matched_record is not None
    assert result.matched_record.fields["label"] == "glucose"


async def test_the_first_query_comes_from_the_query_template():
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose"]


async def test_the_context_reaches_the_selector():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert "blood [glucose] levels" in llm.requests[0].user


async def test_usage_sums_across_stages():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.usage.calls == 2


async def test_the_result_key_is_stable_for_the_same_record_and_run():
    def make():
        return build(
            FakeLLM([select_reply("C01"), score_reply(0.95)]),
            ScriptedRetriever({"glucose": ["T1"]}),
        )

    first = await make().match(SOURCE)
    second = await make().match(SOURCE)
    assert first.result_key == second.result_key


async def test_the_result_key_changes_when_a_field_value_changes():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    a = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    llm2 = FakeLLM([select_reply("C01"), score_reply(0.95)])
    changed = Record(id="s1", fields={**SOURCE.fields, "context_right": "was low"})
    b = await build(llm2, ScriptedRetriever({"glucose": ["T1"]})).match(changed)
    assert a.result_key != b.result_key


# --- retry paths ----------------------------------------------------------------


async def test_a_low_score_triggers_a_rewrite_and_a_second_attempt():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3),
            rewrite_reply("dextrose"),
            select_reply("C01", 0.95),
            score_reply(0.95),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T2"], "dextrose": ["T1"]})
    result = await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]
    assert result.matched_id == "T1"
    assert len(result.attempts) == 2


async def test_a_scorer_query_proposal_is_used_before_the_rewriter_is_called():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3, better_queries=["dextrose"]),
            select_reply("C01", 0.95),
            score_reply(0.95),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T2"], "dextrose": ["T1"]})
    result = await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]
    assert result.status is MatchStatus.MATCHED


async def test_a_candidate_proposal_is_rescored_without_new_retrieval():
    """The proposed candidate is already in hand; re-running retrieval would be waste."""
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3, better_candidate_keys=["C02"]),
            score_reply(0.95),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1", "T2"]})
    result = await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose"]  # exactly one retrieval
    assert result.matched_id == "T2"
    assert result.status is MatchStatus.MATCHED


async def test_a_hallucinated_candidate_proposal_is_dropped_not_searched():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.3, better_candidate_keys=["C99"]),
            rewrite_reply("dextrose"),
            select_reply("C01", 0.9),
            score_reply(0.9),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "dextrose": ["T2"]})
    await build(llm, retriever).match(SOURCE)
    assert "C99" not in retriever.queries


async def test_the_loop_stops_at_max_attempts():
    llm = FakeLLM(
        [select_reply("C01", 0.3), score_reply(0.3), rewrite_reply("q2")] * 2
        + [select_reply("C01", 0.3), score_reply(0.3)]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "q2": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert len(result.attempts) == 2


async def test_a_query_is_never_searched_twice():
    llm = FakeLLM(
        [
            select_reply("C01", 0.3),
            score_reply(0.3, better_queries=["glucose"]),  # the query already tried
            rewrite_reply("dextrose"),
            select_reply("C01", 0.9),
            score_reply(0.9),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "dextrose": ["T2"]})
    await build(llm, retriever).match(SOURCE)
    assert retriever.queries == ["glucose", "dextrose"]


async def test_the_best_attempt_wins_on_exhaustion():
    llm = FakeLLM(
        [
            select_reply("C01", 0.5),
            score_reply(0.5),
            rewrite_reply("q2"),
            select_reply("C01", 0.2),
            score_reply(0.2),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1"], "q2": ["T2"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert result.confidence == 0.5
    assert result.matched_id == "T1"


# --- abstention and no candidates -----------------------------------------------


async def test_no_candidates_anywhere_is_unmatched_with_that_reason():
    llm = FakeLLM([rewrite_reply()])
    retriever = ScriptedRetriever({})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=1)).match(SOURCE)
    assert result.status is MatchStatus.UNMATCHED
    assert result.reason is DecisionReason.NO_CANDIDATES


async def test_an_explicit_abstention_is_unmatched_with_its_own_reason():
    llm = FakeLLM([select_reply(None), rewrite_reply()])
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=1)).match(SOURCE)
    assert result.reason is DecisionReason.SELECTOR_ABSTAINED


async def test_the_scorer_is_not_called_after_an_abstention():
    """With max_attempts=1 the loop stops after the single attempt, so the only call is
    the selector's. What this pins is that no scoring prompt was ever sent."""
    llm = FakeLLM([select_reply(None)])
    await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    assert len(llm.requests) == 1
    assert all("## Rubric" not in r.user for r in llm.requests)


async def test_unresolvable_output_routes_to_review():
    llm = FakeLLM([select_reply("T1"), rewrite_reply()])  # an id, not a key
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.UNRESOLVED_OUTPUT


# --- verification ---------------------------------------------------------------


async def test_a_score_in_the_verify_band_triggers_verification():
    llm = FakeLLM([select_reply("C01"), score_reply(0.7), verify_reply("support")])
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(verify_band=(0.6, 0.8))
    ).match(SOURCE)
    assert result.attempts[0].verifier_decision == "support"
    assert result.status is MatchStatus.MATCHED


async def test_a_score_above_the_band_is_not_verified():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.attempts[0].verifier_decision is None


async def test_verifier_disagreement_forces_review():
    llm = FakeLLM(
        [select_reply("C01"), score_reply(0.7), verify_reply("disagree", preferred_key="C02")]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(verify_band=(0.6, 0.8), max_attempts=1),
    ).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.reason is DecisionReason.VERIFIER_DISAGREEMENT


async def test_the_verifiers_preferred_key_is_recorded_but_not_chased():
    """Re-entering selection on the verifier's preference would make termination depend
    on two models negotiating. Bounded loop, honest flag, human decides."""
    llm = FakeLLM(
        [select_reply("C01"), score_reply(0.7), verify_reply("disagree", preferred_key="C02")]
    )
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1", "T2"]}),
        policy=MatchPolicy(verify_band=(0.6, 0.8), max_attempts=1),
    ).match(SOURCE)
    assert result.attempts[0].verifier_preferred_id == "T2"
    assert result.matched_id == "T1"  # unchanged


async def test_an_audit_verdict_is_honoured_not_merely_counted():
    llm = FakeLLM([select_reply("C01"), score_reply(0.99), verify_reply("no_match")])
    result = await build(
        llm,
        ScriptedRetriever({"glucose": ["T1"]}),
        policy=MatchPolicy(audit_rate=1.0, verify_band=(0.6, 0.8), max_attempts=1),
    ).match(SOURCE)
    assert result.status is MatchStatus.NEEDS_REVIEW
    assert result.attempts[0].audited is True


async def test_audit_is_off_by_default():
    llm = FakeLLM([select_reply("C01"), score_reply(0.99)])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.attempts[0].audited is False


# --- error handling -------------------------------------------------------------


async def test_one_retriever_failing_degrades_to_the_others():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    bad = ScriptedRetriever({}, name="dense", fail=True)
    matcher = Matcher(
        templates=TEMPLATES,
        retrievers=[good, bad],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(),
        run_fingerprint="fp1",
    )
    result = await matcher.match(SOURCE)
    assert result.status is MatchStatus.MATCHED
    assert "dense" in (result.attempts[0].error or "")


async def test_a_retriever_timeout_degrades_rather_than_failing_the_match():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    slow = ScriptedRetriever({}, name="slow", hang=True)
    matcher = Matcher(
        templates=TEMPLATES,
        retrievers=[good, slow],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(retriever_timeout=0.05),
        run_fingerprint="fp1",
    )
    result = await matcher.match(SOURCE)
    assert result.status is MatchStatus.MATCHED


async def test_every_retriever_failing_is_a_retriever_failure_not_a_non_match():
    llm = FakeLLM([])
    bad = ScriptedRetriever({}, name="bm25", fail=True)
    matcher = Matcher(
        templates=TEMPLATES,
        retrievers=[bad],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=MatchPolicy(max_attempts=1),
        run_fingerprint="fp1",
    )
    result = await matcher.match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert result.reason is DecisionReason.RETRIEVER_FAILURE


async def test_a_provider_failure_is_failed_not_unmatched():
    llm = FakeLLM([LLMRetryableError("429 exhausted")])
    result = await build(
        llm, ScriptedRetriever({"glucose": ["T1"]}), policy=MatchPolicy(max_attempts=1)
    ).match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert result.reason is DecisionReason.PROVIDER_FAILURE


async def test_a_fatal_provider_error_stops_the_loop_immediately():
    llm = FakeLLM([LLMFatalError("invalid api key")])
    result = await build(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert result.status is MatchStatus.FAILED
    assert len(result.attempts) == 1  # no point retrying a bad key


async def test_one_failed_attempt_does_not_sink_a_later_good_one():
    llm = FakeLLM([LLMRetryableError("429"), select_reply("C01", 0.95), score_reply(0.95)])
    retriever = ScriptedRetriever({"glucose": ["T1"]})
    result = await build(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert result.status is MatchStatus.MATCHED


# --- sync facade ----------------------------------------------------------------


def test_match_sync_works_outside_an_event_loop():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = build(llm, ScriptedRetriever({"glucose": ["T1"]})).match_sync(SOURCE)
    assert result.status is MatchStatus.MATCHED
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_matcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xwalk.matcher'`.

- [ ] **Step 3: Write `src/xwalk/matcher.py`**

```python
"""The matching loop. Orchestration only — every decision lives in a stage or in policy."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from xwalk.fingerprint import hash_record, result_key
from xwalk.llm.base import LLMError, LLMFatalError
from xwalk.policy import MatchPolicy, derive_status, should_audit, should_verify
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)
from xwalk.retrieval.base import Retriever, SearchRequest
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.keying import KeyedCandidates, Resolution
from xwalk.stages.proposals import RoutedProposals, normalise_query, route_proposals
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet

_NO_PROPOSALS = RoutedProposals(candidate_keys=(), queries=(), dropped=())
_INFRASTRUCTURE_FAILURES = frozenset(
    {DecisionReason.RETRIEVER_FAILURE, DecisionReason.PROVIDER_FAILURE}
)


class Matcher:
    def __init__(
        self,
        *,
        templates: TemplateSet,
        retrievers: Sequence[Retriever],
        store: TargetStore,
        selector: Selector,
        scorer: Scorer,
        verifier: Verifier,
        rewriter: QueryRewriter,
        policy: MatchPolicy | None = None,
        run_fingerprint: str = "",
        retriever_limit: int = 20,
        rrf_k: int = 60,
        keep_candidates_in_trace: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self._templates = templates
        self._retrievers = list(retrievers)
        self._store = store
        self._selector = selector
        self._scorer = scorer
        self._verifier = verifier
        self._rewriter = rewriter
        self._policy = policy or MatchPolicy()
        self._run_fingerprint = run_fingerprint
        self._retriever_limit = retriever_limit
        self._rrf_k = rrf_k
        self._keep_candidates = keep_candidates_in_trace

    @property
    def run_fingerprint(self) -> str:
        return self._run_fingerprint

    @property
    def policy(self) -> MatchPolicy:
        return self._policy

    @property
    def store_fingerprint(self) -> str:
        return self._store.fingerprint

    # --- retrieval -------------------------------------------------------------

    async def _retrieve(self, query: str, source: Record) -> tuple[list[Candidate], list[str]]:
        """Search every retriever concurrently. Returns (candidates, degradation notes).

        Depth comes from each retriever's own `default_limit`; `retriever_limit` is only
        the fallback for a backend that does not declare one.
        """

        async def one(retriever: Retriever) -> Sequence[RetrievalHit]:
            limit = getattr(retriever, "default_limit", None) or self._retriever_limit
            request = SearchRequest(text=query, limit=limit, source_record=source)
            return await asyncio.wait_for(
                retriever.search(request), timeout=self._policy.retriever_timeout
            )

        outcomes = await asyncio.gather(*(one(r) for r in self._retrievers), return_exceptions=True)

        groups: list[Sequence[RetrievalHit]] = []
        notes: list[str] = []
        for retriever, outcome in zip(self._retrievers, outcomes, strict=True):
            # Order matters: TimeoutError is an Exception, so it must be tested first.
            if isinstance(outcome, asyncio.TimeoutError):
                notes.append(f"{retriever.name}: timed out after {self._policy.retriever_timeout}s")
            elif isinstance(outcome, BaseException):
                notes.append(f"{retriever.name}: {outcome}")
            else:
                groups.append(outcome)

        if not groups:
            return [], notes
        return reciprocal_rank_fusion(groups, self._store, k=self._rrf_k), notes

    # --- one attempt -----------------------------------------------------------

    async def _attempt(
        self,
        source: Record,
        context: str,
        query: str,
        proposal: RetryProposal | None,
        index: int,
        reuse: tuple[KeyedCandidates, list[Candidate]] | None,
        forced_key: str | None,
        seen_queries: AbstractSet[str],
    ) -> tuple[Attempt, RoutedProposals, KeyedCandidates | None]:
        """Run one retrieve -> select -> score -> verify pass.

        Returns the attempt, its *already routed* proposals, and the `KeyedCandidates`
        actually issued — the caller keeps that object so a candidate proposal can be
        re-examined without re-running retrieval. It is never reconstructed from
        `issued_keys`: `rendered` and `by_key` cannot be recovered from a key->id map,
        and the scorer needs both.
        """
        usage = Usage.zero()
        notes: list[str] = []
        keyed: KeyedCandidates | None = None
        truncated = 0

        if reuse is not None:
            keyed, candidates = reuse
        else:
            candidates, notes = await self._retrieve(query, source)

        def build(
            *,
            chosen_id: str | None = None,
            resolution: str = Resolution.ABSTAIN.value,
            raw: str | None = None,
            score: float | None = None,
            explanation: str = "",
            verifier_decision: str | None = None,
            verifier_score: float | None = None,
            verifier_preferred_id: str | None = None,
            audited: bool = False,
            dropped: tuple[tuple[str, str], ...] = (),
            reason: DecisionReason | None = None,
            error: str | None = None,
        ) -> Attempt:
            joined = "; ".join(notes) if notes else None
            return Attempt(
                index=index,
                query=query,
                proposal=proposal,
                candidates=tuple(candidates) if self._keep_candidates else (),
                candidate_count=len(candidates),
                candidates_truncated=truncated,
                issued_keys=dict(keyed.issued) if keyed is not None else {},
                raw_selection=raw,
                chosen_id=chosen_id,
                resolution=resolution,
                primary_score=score,
                explanation=explanation,
                verifier_decision=verifier_decision,
                verifier_score=verifier_score,
                verifier_preferred_id=verifier_preferred_id,
                audited=audited,
                dropped_proposals=dropped,
                reason=reason,
                error=error if error else joined,
                usage=usage,
            )

        # Every retriever failed. A dead index is not evidence of a non-match.
        if not candidates and notes and len(notes) == len(self._retrievers):
            return build(reason=DecisionReason.RETRIEVER_FAILURE), _NO_PROPOSALS, None

        if not candidates:
            return build(reason=DecisionReason.NO_CANDIDATES), _NO_PROPOSALS, None

        # --- selection ---
        if forced_key is not None and keyed is not None:
            # A candidate proposal: the record is already in hand, so no new retrieval
            # and no second selector call. Go straight to scoring it.
            chosen_key: str = forced_key
            # Annotated because the selector branch below assigns `str | None` here.
            chosen_id: str | None = keyed.issued[forced_key]
            resolution = Resolution.EXACT_KEY.value
            selection_raw = f"(reused candidate proposal {forced_key})"
            selection_explanation = ""
        else:
            try:
                selection = await self._selector.select(source, context, candidates)
            except LLMFatalError:
                raise  # a bad key or unknown model will not fix itself; let match() stop
            except LLMError as exc:
                return (
                    build(reason=DecisionReason.PROVIDER_FAILURE, error=f"selector: {exc}"),
                    _NO_PROPOSALS,
                    None,
                )
            usage = usage + selection.usage
            keyed = selection.keyed
            truncated = selection.truncated
            chosen_id = selection.choice.record_id
            resolution = selection.choice.resolution.value
            selection_raw = selection.raw
            selection_explanation = selection.explanation
            resolved_key = next((k for k, rid in keyed.issued.items() if rid == chosen_id), None)

            if chosen_id is None or resolved_key is None:
                # Abstention or unresolvable output. Both are terminal for this attempt;
                # neither is worth a scorer call.
                return (
                    build(
                        resolution=resolution,
                        raw=selection_raw,
                        explanation=selection_explanation,
                        error=selection.error,
                    ),
                    _NO_PROPOSALS,
                    keyed,
                )
            # Bound to a narrowed local so `chosen_key` is `str`, not `str | None`, in
            # both branches — the scorer and verifier both require `str`.
            chosen_key = resolved_key

        assert keyed is not None

        # --- scoring, against the full source record, never the retrieval query ---
        try:
            scored = await self._scorer.score(source, context, keyed, chosen_key)
        except LLMFatalError:
            raise
        except LLMError as exc:
            return (
                build(
                    chosen_id=chosen_id,
                    resolution=resolution,
                    raw=selection_raw,
                    explanation=selection_explanation,
                    reason=DecisionReason.PROVIDER_FAILURE,
                    error=f"scorer: {exc}",
                ),
                _NO_PROPOSALS,
                keyed,
            )
        usage = usage + scored.usage

        if scored.score is None:
            # The gate produced no usable number. Falling back to the selector's own
            # self-reported confidence would let a malformed answer become an automatic
            # MATCH on a score the gate never gave; the spec routes malformed output to
            # review instead.
            return (
                build(
                    chosen_id=chosen_id,
                    resolution=Resolution.UNRESOLVED.value,
                    raw=selection_raw,
                    explanation=selection_explanation,
                    error=f"scorer: {scored.error}",
                ),
                _NO_PROPOSALS,
                keyed,
            )

        score = scored.score

        # --- verification, as a cost control; audit, as a sample of the invisible ---
        verdict = None
        audited = False
        if should_verify(score, self._policy):
            verdict = await self._verifier.verify(source, context, keyed, chosen_key)
        elif should_audit(score, self._policy, self._run_fingerprint, source.id):
            verdict = await self._verifier.verify(source, context, keyed, chosen_key)
            audited = True
        if verdict is not None:
            usage = usage + verdict.usage

        routed = route_proposals(scored.proposals, keyed.order, seen_queries)

        attempt = build(
            chosen_id=chosen_id,
            resolution=resolution,
            raw=selection_raw,
            score=score,
            explanation=scored.explanation or selection_explanation,
            verifier_decision=None if verdict is None else verdict.decision,
            verifier_score=None if verdict is None else verdict.confidence,
            verifier_preferred_id=(
                None
                if verdict is None or verdict.preferred_key is None
                else keyed.issued.get(verdict.preferred_key)
            ),
            audited=audited,
            dropped=tuple((p.value, why) for p, why in routed.dropped),
        )
        return attempt, routed, keyed

    # --- the loop --------------------------------------------------------------

    def _is_acceptable(self, attempt: Attempt) -> bool:
        """Good enough to stop early: a scored, exactly-resolved, unchallenged match."""
        return (
            attempt.primary_score is not None
            and attempt.primary_score >= self._policy.accept_at
            and attempt.resolution == Resolution.EXACT_KEY.value
            and attempt.verifier_decision not in ("disagree", "no_match")
        )

    def _fatal_attempt(
        self, index: int, query: str, proposal: RetryProposal | None, exc: Exception
    ) -> Attempt:
        return Attempt(
            index=index,
            query=query,
            proposal=proposal,
            candidates=(),
            candidate_count=0,
            candidates_truncated=0,
            issued_keys={},
            raw_selection=None,
            chosen_id=None,
            resolution=Resolution.ABSTAIN.value,
            primary_score=None,
            explanation="",
            verifier_decision=None,
            verifier_score=None,
            verifier_preferred_id=None,
            audited=False,
            dropped_proposals=(),
            reason=DecisionReason.PROVIDER_FAILURE,
            error=str(exc),
            usage=Usage.zero(),
        )

    async def match(self, source: Record) -> MatchResult:
        context = self._templates.render_context(source)
        first_query = self._templates.render_query(source)

        queue: list[tuple[str, RetryProposal | None]] = [(first_query, None)]
        seen_queries = {normalise_query(first_query)}
        tried_queries: list[str] = []  # original casing, for the rewriter's prompt
        candidate_queue: list[str] = []
        attempts: list[Attempt] = []
        all_candidates: list[Candidate] = []
        last_keyed: KeyedCandidates | None = None
        last_candidates: list[Candidate] = []
        last_query = first_query

        for index in range(self._policy.max_attempts):
            reuse: tuple[KeyedCandidates, list[Candidate]] | None = None
            forced_key: str | None = None
            proposal: RetryProposal | None = None

            if candidate_queue and last_keyed is not None:
                forced_key = candidate_queue.pop(0)
                proposal = RetryProposal(kind="candidate", value=forced_key, source="scorer")
                reuse = (last_keyed, last_candidates)
                query = last_query
            elif queue:
                query, proposal = queue.pop(0)
                last_query = query
                if query not in tried_queries:
                    tried_queries.append(query)
            else:
                break

            try:
                attempt, routed, keyed = await self._attempt(
                    source, context, query, proposal, index, reuse, forced_key, seen_queries
                )
            except LLMFatalError as exc:
                attempts.append(self._fatal_attempt(index, query, proposal, exc))
                break

            attempts.append(attempt)
            if attempt.candidates:
                all_candidates = list(attempt.candidates)
            if keyed is not None and keyed.order:
                last_keyed = keyed
                last_candidates = list(attempt.candidates) or last_candidates

            if self._is_acceptable(attempt):
                break
            if index + 1 >= self._policy.max_attempts:
                break

            if attempt.reason in _INFRASTRUCTURE_FAILURES:
                # Not evidence about this record. Retry the same query rather than
                # asking the rewriter to invent a new one for a provider outage.
                queue.insert(0, (query, proposal))
                continue

            candidate_queue.extend(routed.candidate_keys)
            for text in routed.queries:
                seen_queries.add(normalise_query(text))
                queue.append((text, RetryProposal(kind="query", value=text, source="scorer")))

            if not queue and not candidate_queue:
                rewritten = await self._rewriter.rewrite(
                    source, context, tried_queries, last_keyed or _empty_keyed()
                )
                for out in rewritten.proposals:
                    normalised = normalise_query(out.value)
                    if normalised in seen_queries:
                        continue
                    seen_queries.add(normalised)
                    queue.append((out.value, out))
                if not queue:
                    break

        status, reason, best = derive_status(attempts, self._policy)

        matched_id = None if best is None else best.chosen_id
        if status in (MatchStatus.UNMATCHED, MatchStatus.FAILED):
            matched_id = None

        record = None
        if matched_id is not None:
            try:
                record = self._store.get(matched_id)
            except KeyError:
                record = None

        source_hash = hash_record(source)
        return MatchResult(
            result_key=result_key(self._run_fingerprint, source.id, source_hash),
            source_id=source.id,
            source_hash=source_hash,
            matched_id=matched_id,
            matched_record=record,
            confidence=None if best is None else best.primary_score,
            status=status,
            reason=reason,
            explanation="" if best is None else best.explanation,
            candidates=tuple(all_candidates) if self._keep_candidates else (),
            attempts=tuple(attempts),
            usage=sum((a.usage for a in attempts), Usage.zero()),
            run_fingerprint=self._run_fingerprint,
        )

    def match_sync(self, source: Record) -> MatchResult:
        return asyncio.run(self.match(source))


def _empty_keyed() -> KeyedCandidates:
    return KeyedCandidates(order=(), by_key={}, issued={}, rendered="")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_matcher.py -v`
Expected: 31 passed.

Five decisions in the code above are load-bearing. If a test fails, check these before changing anything:

1. **`_attempt` returns the `KeyedCandidates` it built.** A candidate proposal is re-scored against that same object. It cannot be rebuilt from `Attempt.issued_keys`, because a key→id map carries neither `rendered` nor `by_key`, and the scorer's `_blocks()` needs both — it would raise `StopIteration` on an empty `rendered`.
2. **`LLMFatalError` is re-raised, not absorbed.** `LLMFatalError` subclasses `LLMError`, so a bare `except LLMError` around the selector call would swallow it and make the `except LLMFatalError` in `match()` unreachable. An invalid API key would then burn every remaining attempt.
3. **An infrastructure failure re-queues the same query.** A provider 500 says nothing about this record, so the next attempt retries the same search rather than asking the rewriter for a new one. Without this, `test_one_failed_attempt_does_not_sink_a_later_good_one` cannot pass: the rewriter would consume the next scripted response and the loop would exit with every attempt failed.
4. **A scorer that returns no usable score does not fall back to the selector's confidence.** That fallback would let malformed gate output become `MATCHED / ACCEPT_THRESHOLD` on a number the gate never produced. The attempt is marked `UNRESOLVED` and routes to review, which is what the spec's error-handling contract requires.
5. **`TimeoutError` is tested before `BaseException`** in `_retrieve`, because it is one.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q -m "not integration"`
Expected: all green.

- [ ] **Step 6: Lint, type-check, commit**

```bash
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/matcher.py src/xwalk/records.py src/xwalk/serde.py tests/
git commit -m "feat: the matching loop with retry routing, verification, and degradation"
```

---

## Task 18: Batch runner, resume, and exports

**Files:**
- Create: `src/xwalk/batch.py`
- Modify: `src/xwalk/__init__.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_run_fingerprint(*, templates, prompts, store, retrievers, llm, policy, selector_policy) -> str`; `BatchReport` with `total`, `by_status()`, `needs_review()`, `duplicate_targets()`, `usage`, `run_fingerprint`; `async run_batch(matcher, source, out, *, resume=True, manifest_extra=None, progress=None) -> BatchReport`; `run_batch_sync(...)`; `export_results_jsonl(ledger, run_fp, path)`, `export_mapping_csv(ledger, run_fp, path, *, use_review=False)`, `export_manifest(ledger, run_fp, path)`.

**This is the task that makes the phase's promise true:** a user points at two CSVs and gets a reviewed mapping table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_batch.py`:

```python
import csv
import json

import pytest

from tests.test_matcher import (
    PROMPTS,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    score_reply,
    select_reply,
)
from xwalk.batch import build_run_fingerprint, export_mapping_csv, run_batch
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import MatchStatus, Record
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

SOURCES = [
    Record(id="s1", fields={"mention": "glucose"}),
    Record(id="s2", fields={"mention": "fructose"}),
]


def make_matcher(llm, retriever, *, run_fp="fp1", policy=None):
    policy = policy or MatchPolicy()
    return Matcher(
        templates=TEMPLATES,
        retrievers=[retriever],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=policy,
        run_fingerprint=run_fp,
    )


def two_good_matches() -> FakeLLM:
    def handler(request):
        if "## Candidates" in request.user:
            return select_reply("C01")
        return score_reply(0.95)

    return FakeLLM(handler=handler)


RETRIEVER = {"glucose": ["T1"], "fructose": ["T2"]}


# --- fingerprint -----------------------------------------------------------------


def test_fingerprint_changes_with_the_policy():
    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        selector_policy=SelectorPolicy(),
    )
    a = build_run_fingerprint(policy=MatchPolicy(accept_at=0.6), **kwargs)
    b = build_run_fingerprint(policy=MatchPolicy(accept_at=0.7), **kwargs)
    assert a != b


def test_fingerprint_changes_with_the_selector_budget():
    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        policy=MatchPolicy(),
    )
    a = build_run_fingerprint(selector_policy=SelectorPolicy(max_candidates=30), **kwargs)
    b = build_run_fingerprint(selector_policy=SelectorPolicy(max_candidates=10), **kwargs)
    assert a != b


def test_fingerprint_changes_with_the_target_snapshot():
    from xwalk.stores.memory import MemoryStore

    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        policy=MatchPolicy(),
        selector_policy=SelectorPolicy(),
    )
    other = MemoryStore.from_source([Record(id="T1", fields={"label": "changed"})])
    assert build_run_fingerprint(store=STORE, **kwargs) != build_run_fingerprint(
        store=other, **kwargs
    )


def test_fingerprint_is_stable_across_identical_configurations():
    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        policy=MatchPolicy(),
        selector_policy=SelectorPolicy(),
    )
    assert build_run_fingerprint(**kwargs) == build_run_fingerprint(**kwargs)


# --- running ----------------------------------------------------------------------


async def test_every_source_record_produces_a_result(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert report.total == 2


async def test_results_are_grouped_by_status(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert report.by_status() == {MatchStatus.MATCHED: 2}


async def test_usage_is_aggregated(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert report.usage.calls == 4  # two records, select + score each


async def test_the_ledger_file_is_created(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert (tmp_path / "run" / "ledger.sqlite").exists()


async def test_a_progress_callback_fires_once_per_record(tmp_path):
    seen = []
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    await run_batch(matcher, SOURCES, out=tmp_path / "run", progress=seen.append)
    assert len(seen) == 2


async def test_concurrency_is_bounded_by_the_policy(tmp_path):
    import asyncio

    live = {"now": 0, "max": 0}

    class CountingRetriever(ScriptedRetriever):
        async def search(self, request):
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            await asyncio.sleep(0.01)
            try:
                return await super().search(request)
            finally:
                live["now"] -= 1

    many = [Record(id=f"s{i}", fields={"mention": "glucose"}) for i in range(20)]
    matcher = make_matcher(
        two_good_matches(), CountingRetriever(RETRIEVER), policy=MatchPolicy(concurrency=3)
    )
    await run_batch(matcher, many, out=tmp_path / "run")
    assert live["max"] <= 3


# --- resume -----------------------------------------------------------------------


async def test_resume_skips_records_already_completed(tmp_path):
    out = tmp_path / "run"
    first = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), first), SOURCES, out=out)

    second = ScriptedRetriever(RETRIEVER)
    report = await run_batch(
        make_matcher(two_good_matches(), second), SOURCES, out=out, resume=True
    )
    assert second.queries == []  # nothing re-run
    assert report.total == 2


async def test_resume_false_reruns_everything(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    second = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), second), SOURCES, out=out, resume=False)
    assert len(second.queries) == 2


async def test_a_changed_source_record_is_reprocessed_on_resume(tmp_path):
    """Same id, different content — the prior result is not about this record."""
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    changed = [Record(id="s1", fields={"mention": "glucose", "note": "new"}), SOURCES[1]]
    second = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), second), changed, out=out, resume=True)
    assert second.queries == ["glucose"]


async def test_a_changed_run_fingerprint_forces_a_fresh_run(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER), run_fp="fp1"),
        SOURCES,
        out=out,
    )
    second = ScriptedRetriever(RETRIEVER)
    await run_batch(
        make_matcher(two_good_matches(), second, run_fp="fp2"), SOURCES, out=out, resume=True
    )
    assert len(second.queries) == 2


async def test_completed_work_survives_a_crash_mid_run(tmp_path):
    """The point of committing each result as it finishes."""
    out = tmp_path / "run"
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated crash")
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.95)

    matcher = make_matcher(
        FakeLLM(handler=handler), ScriptedRetriever(RETRIEVER), policy=MatchPolicy(concurrency=1)
    )
    with pytest.raises(RuntimeError):
        await run_batch(matcher, SOURCES, out=out)

    resumed = ScriptedRetriever(RETRIEVER)
    report = await run_batch(
        make_matcher(two_good_matches(), resumed), SOURCES, out=out, resume=True
    )
    assert report.total == 2
    assert len(resumed.queries) == 1  # only the unfinished record re-ran


# --- reporting --------------------------------------------------------------------


async def test_duplicate_targets_are_reported_not_resolved(tmp_path):
    """Reporting is not solving. The library surfaces the conflict and does nothing."""
    both_to_t1 = [
        Record(id="s1", fields={"mention": "glucose"}),
        Record(id="s2", fields={"mention": "glucose"}),
    ]
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, both_to_t1, out=tmp_path / "run")
    assert report.duplicate_targets() == {"T1": ["s1", "s2"]}
    assert report.by_status() == {MatchStatus.MATCHED: 2}  # both still matched


async def test_needs_review_lists_the_review_bucket(tmp_path):
    def handler(request):
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.5)

    matcher = make_matcher(FakeLLM(handler=handler), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert {r.source_id for r in report.needs_review()} == {"s1", "s2"}


# --- exports ----------------------------------------------------------------------


async def test_mapping_csv_has_one_row_per_source_record(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    rows = list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))
    assert len(rows) == 2


async def test_mapping_csv_columns_are_the_documented_set(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    row = next(iter(csv.DictReader((out / "mapping.csv").open(encoding="utf-8"))))
    assert list(row) == [
        "source_id",
        "matched_id",
        "confidence",
        "status",
        "reason",
        "explanation",
        "attempts",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
    ]


async def test_results_jsonl_has_one_object_per_line(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    lines = (out / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["source_id"] == "s1"


async def test_the_manifest_records_the_run_fingerprint_and_components(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_fingerprint"] == "fp1"
    assert "target_fingerprint" in manifest
    assert "counts" in manifest


async def test_the_manifest_never_contains_an_api_key(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)),
        SOURCES,
        out=out,
        manifest_extra={"model": "fake"},
    )
    text = (out / "manifest.json").read_text(encoding="utf-8").lower()
    assert "api_key" not in text and "authorization" not in text


async def test_exports_are_regenerable_from_the_ledger_alone(tmp_path):
    from xwalk.ledger import Ledger

    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    (out / "mapping.csv").unlink()
    ledger = Ledger.open(out / "ledger.sqlite")
    export_mapping_csv(ledger, "fp1", out / "mapping.csv")
    ledger.close()
    assert len(list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))) == 2


async def test_mapping_csv_can_be_written_from_the_adjudicated_view(tmp_path):
    from xwalk.ledger import Ledger
    from xwalk.review import ReviewDecision, ReviewRow, apply_review

    out = tmp_path / "run"

    def handler(request):
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.5)

    await run_batch(
        make_matcher(FakeLLM(handler=handler), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )

    ledger = Ledger.open(out / "ledger.sqlite")
    result = next(iter(ledger.iter_results("fp1")))
    apply_review(
        ledger,
        [
            ReviewRow(
                result_key=result.result_key,
                run_fingerprint="fp1",
                source_id=result.source_id,
                source_hash=result.source_hash,
                proposed_target_id=result.matched_id,
                decision=ReviewDecision.ACCEPT,
                corrected_target_id=None,
                reviewer="jan",
                review_note="",
                reviewed_at="2026-07-26T10:00:00Z",
            )
        ],
        target_store_fingerprint=STORE.fingerprint,
    )
    export_mapping_csv(ledger, "fp1", out / "adjudicated.csv", use_review=True)
    ledger.close()

    rows = {
        r["source_id"]: r for r in csv.DictReader((out / "adjudicated.csv").open(encoding="utf-8"))
    }
    assert rows[result.source_id]["status"] == "matched"


# --- sync facade ------------------------------------------------------------------


def test_run_batch_sync_works_outside_an_event_loop(tmp_path):
    from xwalk.batch import run_batch_sync

    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = run_batch_sync(matcher, SOURCES, out=tmp_path / "run")
    assert report.total == 2
```

- [ ] **Step 2: Write `src/xwalk/batch.py`**

```python
"""Batch matching: the mapping platform.

Point at a source, get a resumable run directory containing a ledger and three exports.
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xwalk import __version__
from xwalk.fingerprint import hash_record, hash_value, result_key
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet
from xwalk.records import MatchResult, MatchStatus, Record, Usage
from xwalk.retrieval.base import Retriever
from xwalk.review import adjudicated
from xwalk.serde import result_to_dict
from xwalk.stages.select import SelectorPolicy
from xwalk.stores.base import TargetStore
from xwalk.templates import TemplateSet

MAPPING_COLUMNS = (
    "source_id",
    "matched_id",
    "confidence",
    "status",
    "reason",
    "explanation",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "llm_calls",
)


def build_run_fingerprint(
    *,
    templates: TemplateSet,
    prompts: PromptSet,
    store: TargetStore,
    retrievers: Sequence[Retriever],
    llm: LLMClient,
    policy: MatchPolicy,
    selector_policy: SelectorPolicy,
    retriever_limit: int = 20,
    rrf_k: int = 60,
) -> str:
    """Everything whose change should invalidate prior results.

    Deliberately excludes credentials, output paths, and concurrency — none of them
    change what a result means. `retriever_limit` and `rrf_k` must match the values
    handed to `Matcher`, or a resumed run will reuse results produced at another depth.
    """
    return hash_value(
        {
            "library_version": __version__,
            "templates": templates.fingerprint,
            "prompts": prompts.fingerprint,
            "target": store.fingerprint,
            "retrievers": sorted(f"{r.name}:{r.fingerprint}" for r in retrievers),
            # Depth and fusion constant change which candidates exist at all, so they
            # belong here: raising k from 20 to 100 must not silently reuse old results.
            "retrieval": {
                "depths": sorted(
                    f"{r.name}:{getattr(r, 'default_limit', retriever_limit)}" for r in retrievers
                ),
                "retriever_limit": retriever_limit,
                "rrf_k": rrf_k,
                "retriever_timeout": policy.retriever_timeout,
            },
            "llm": llm.fingerprint,
            "policy": {
                "max_attempts": policy.max_attempts,
                "accept_at": policy.accept_at,
                "review_floor": policy.review_floor,
                "verify_band": list(policy.verify_band) if policy.verify_band else None,
                "audit_rate": policy.audit_rate,
                "legacy_id_resolution": policy.legacy_id_resolution,
            },
            "selector_policy": {
                "max_candidates": selector_policy.max_candidates,
                "max_candidate_tokens": selector_policy.max_candidate_tokens,
            },
        }
    )


@dataclass(frozen=True)
class BatchReport:
    run_fingerprint: str
    out_dir: Path
    total: int
    usage: Usage
    _ledger_path: Path

    def _ledger(self) -> Ledger:
        return Ledger.open(self._ledger_path)

    def by_status(self) -> dict[MatchStatus, int]:
        ledger = self._ledger()
        try:
            return ledger.count_by_status(self.run_fingerprint)
        finally:
            ledger.close()

    def needs_review(self) -> list[MatchResult]:
        ledger = self._ledger()
        try:
            return [
                r
                for r in ledger.iter_results(self.run_fingerprint)
                if r.status is MatchStatus.NEEDS_REVIEW
            ]
        finally:
            ledger.close()

    def duplicate_targets(self) -> dict[str, list[str]]:
        ledger = self._ledger()
        try:
            return ledger.duplicate_targets(self.run_fingerprint)
        finally:
            ledger.close()


async def run_batch(
    matcher: Matcher,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    manifest_extra: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open(out_dir / "ledger.sqlite")
    run_fp = matcher.run_fingerprint
    policy = matcher.policy

    ledger.put_manifest(
        run_fp,
        {
            "run_fingerprint": run_fp,
            "library_version": __version__,
            "target_fingerprint": matcher.store_fingerprint,
            **dict(manifest_extra or {}),
        },
    )

    semaphore = asyncio.Semaphore(policy.concurrency)
    total_usage = Usage.zero()
    completed = 0

    async def one(record: Record) -> None:
        nonlocal total_usage, completed
        key = result_key(run_fp, record.id, hash_record(record))
        if resume and ledger.has_result(key):
            completed += 1
            return
        async with semaphore:
            result = await matcher.match(record)
            # Commit inside the slot. If the write happened after release, a crash in
            # the next record could interleave ahead of this one's commit, and the
            # resume guarantee this whole module exists for would be probabilistic.
            await ledger.put_result(result)
        total_usage = total_usage + result.usage
        completed += 1
        if progress is not None:
            progress(result)

    try:
        # Chunked so an unbounded source does not materialise every task at once.
        pending: list[asyncio.Task[None]] = []
        for record in source:
            pending.append(asyncio.create_task(one(record)))
            if len(pending) >= policy.concurrency * 4:
                await asyncio.gather(*pending)
                pending = []
        if pending:
            await asyncio.gather(*pending)

        export_results_jsonl(ledger, run_fp, out_dir / "results.jsonl")
        export_mapping_csv(ledger, run_fp, out_dir / "mapping.csv")
        export_manifest(ledger, run_fp, out_dir / "manifest.json")

        return BatchReport(
            run_fingerprint=run_fp,
            out_dir=out_dir,
            total=ledger.count(run_fp),
            usage=total_usage,
            _ledger_path=out_dir / "ledger.sqlite",
        )
    finally:
        ledger.close()


def run_batch_sync(
    matcher: Matcher,
    source: Iterable[Record],
    *,
    out: str | Path,
    resume: bool = True,
    manifest_extra: Mapping[str, Any] | None = None,
    progress: Callable[[MatchResult], None] | None = None,
) -> BatchReport:
    return asyncio.run(
        run_batch(
            matcher,
            source,
            out=out,
            resume=resume,
            manifest_extra=manifest_extra,
            progress=progress,
        )
    )


# --- exports --------------------------------------------------------------------


def export_results_jsonl(ledger: Ledger, run_fingerprint: str, path: str | Path) -> int:
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for result in ledger.iter_results(run_fingerprint):
            handle.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")
            written += 1
    return written


def export_mapping_csv(
    ledger: Ledger,
    run_fingerprint: str,
    path: str | Path,
    *,
    use_review: bool = False,
) -> int:
    """The deliverable. `use_review=True` writes the adjudicated view instead."""
    path = Path(path)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MAPPING_COLUMNS))
        writer.writeheader()

        if use_review:
            for row in adjudicated(ledger, run_fingerprint):
                writer.writerow(
                    {
                        "source_id": row.source_id,
                        "matched_id": row.final_target_id or "",
                        "confidence": "" if row.confidence is None else row.confidence,
                        "status": row.final_status.value,
                        "reason": "reviewed" if row.reviewer else row.model_status.value,
                        "explanation": row.review_note,
                        "attempts": "",
                        "prompt_tokens": "",
                        "completion_tokens": "",
                        "llm_calls": "",
                    }
                )
                written += 1
            return written

        for result in ledger.iter_results(run_fingerprint):
            writer.writerow(
                {
                    "source_id": result.source_id,
                    "matched_id": result.matched_id or "",
                    "confidence": "" if result.confidence is None else result.confidence,
                    "status": result.status.value,
                    "reason": result.reason.value,
                    "explanation": result.explanation,
                    "attempts": len(result.attempts),
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "llm_calls": result.usage.calls,
                }
            )
            written += 1
    return written


def export_manifest(ledger: Ledger, run_fingerprint: str, path: str | Path) -> None:
    manifest = dict(ledger.get_manifest(run_fingerprint) or {})
    manifest["counts"] = {
        status.value: count for status, count in ledger.count_by_status(run_fingerprint).items()
    }
    manifest["duplicate_targets"] = ledger.duplicate_targets(run_fingerprint)
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
```

- [ ] **Step 3: Write the final `src/xwalk/__init__.py`**

`Matcher` and `TemplateSet` are the two names the README quickstart imports from the package root, and neither has been exported yet. Write the whole file:

```python
"""xwalk — LLM-RAG record matching between two collections."""

# __version__ is defined FIRST, before any submodule import. `batch.py` does
# `from xwalk import __version__`, so if a batch name were re-exported below while
# __version__ was still unbound, importing xwalk would raise
# "cannot import name '__version__' from partially initialized module".
__version__ = "0.1.0.dev0"

from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import (
    Attempt,
    Candidate,
    DecisionReason,
    MatchResult,
    MatchStatus,
    Record,
    RetrievalHit,
    RetryProposal,
    Usage,
)
from xwalk.templates import TemplateSet

__all__ = [
    "Attempt",
    "Candidate",
    "DecisionReason",
    "MatchPolicy",
    "MatchResult",
    "MatchStatus",
    "Matcher",
    "Record",
    "RetrievalHit",
    "RetryProposal",
    "TemplateSet",
    "Usage",
    "__version__",
]
```

`run_batch` is deliberately **not** re-exported here. It lives at `xwalk.batch.run_batch`, and importing it from the package root would create exactly the cycle the comment above warns about. `Matcher` and `TemplateSet` are safe because neither module imports `xwalk` itself.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_batch.py -v`
Expected: 25 passed.

`test_completed_work_survives_a_crash_mid_run` is the important one: it asserts the whole reason for the ledger. If it fails because the crash aborts before the first result commits, check that `run_batch` awaits `ledger.put_result` *before* the next record starts — with `concurrency=1` the ordering must be strict.

- [ ] **Step 5: Add the end-to-end fixture test**

Append to `tests/test_batch.py` — the phase's promise, exercised against the real CSV fixtures and the real BM25 retriever:

```python
async def test_two_csvs_in_a_mapping_table_out(tmp_path, targets_csv, sources_csv):
    from xwalk.retrieval.bm25 import BM25Retriever
    from xwalk.sources.tabular import csv_source
    from xwalk.stores.memory import MemoryStore
    from xwalk.templates import TemplateSet

    templates = TemplateSet(
        query="{{ mention }}",
        context="{{ context_left }} [{{ mention }}] {{ context_right }}",
        doc="{{ label }} {{ synonyms | join(' ') }}",
        candidate="ID: {{ id }} Label: {{ label }}",
    )
    targets = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    store = MemoryStore.from_source(targets)
    retriever = BM25Retriever.build(
        targets, templates, tmp_path / "idx", exact_fields=("label", "synonyms")
    )

    def handler(request):
        if "## Candidates" in request.user:
            return select_reply("C01") if "[C01]" in request.user else select_reply(None)
        return score_reply(0.95)

    llm = FakeLLM(handler=handler)
    matcher = Matcher(
        templates=templates,
        retrievers=[retriever],
        store=store,
        selector=Selector(llm, PROMPTS, templates),
        scorer=Scorer(llm, PROMPTS, templates),
        verifier=Verifier(llm, PROMPTS, templates),
        rewriter=QueryRewriter(llm, PROMPTS, templates),
        policy=MatchPolicy(max_attempts=1),
        run_fingerprint="e2e",
    )
    report = await run_batch(
        matcher, csv_source(sources_csv, id_column="mention_id"), out=tmp_path / "run"
    )

    assert report.total == 4
    rows = {
        r["source_id"]: r
        for r in csv.DictReader((tmp_path / "run" / "mapping.csv").open(encoding="utf-8"))
    }
    assert rows["s1"]["matched_id"] == "CHEBI:17234"  # glucose
    assert rows["s2"]["matched_id"] == "CHEBI:17234"  # dextrose, via synonym
    assert rows["s4"]["status"] in ("unmatched", "needs_review")  # unobtainium
```

Run: `python -m pytest tests/test_batch.py -v` — Expected: 26 passed. If `s2` does not resolve through the synonym field, the `doc` template is not indexing synonyms; fix the template in the test, not the assertion.

- [ ] **Step 6: Run everything, lint, type-check**

```bash
python -m pytest -q -m "not integration"
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
```

- [ ] **Step 7: Write the README quickstart**

Replace `README.md` with a runnable example, so the phase's promise is documented where a new user will find it:

````markdown
# xwalk

Match records from any collection to any other, using retrieval plus an LLM.

```python
import os

from xwalk import Matcher, MatchPolicy, TemplateSet
from xwalk.batch import build_run_fingerprint, run_batch_sync
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.stores.memory import MemoryStore

templates = TemplateSet(
    query="{{ mention }}",
    context="{{ context_left }} [{{ mention }}] {{ context_right }}",
    doc="{{ label }} {{ synonyms | join(' ') }}",
    candidate="ID: {{ id }} Label: {{ label }}",
)

targets = list(csv_source("targets.csv", id_column="id", multivalue_columns=["synonyms"]))
store = MemoryStore.from_source(targets)
# exact_fields makes a whole-string hit on a label or synonym outrank a document that
# merely contains the query term more often. Omit it for plain BM25.
retriever = BM25Retriever.build(
    targets, templates, "index/", exact_fields=("label", "synonyms")
)

llm = OpenAICompatClient(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    profile="openai",
)
prompts = PromptSet.from_slots(load_slots("slots.yaml"))
policy, selector_policy = MatchPolicy(), SelectorPolicy()

matcher = Matcher(
    templates=templates,
    retrievers=[retriever],
    store=store,
    selector=Selector(llm, prompts, templates, policy=selector_policy),
    scorer=Scorer(llm, prompts, templates, review_floor=policy.review_floor),
    verifier=Verifier(llm, prompts, templates),
    rewriter=QueryRewriter(llm, prompts, templates),
    policy=policy,
    run_fingerprint=build_run_fingerprint(
        templates=templates,
        prompts=prompts,
        store=store,
        retrievers=[retriever],
        llm=llm,
        policy=policy,
        selector_policy=selector_policy,
    ),
)

report = run_batch_sync(matcher, csv_source("sources.csv", id_column="mention_id"), out="run/")
print(report.by_status())
print(report.duplicate_targets())
```

`run/mapping.csv` is the deliverable. `run/ledger.sqlite` makes the run resumable —
re-running with `resume=True` skips completed records and re-runs anything whose source
record or run configuration changed.

## Review

```python
from xwalk.ledger import Ledger
from xwalk.review import apply_review, export_review, read_review

ledger = Ledger.open("run/ledger.sqlite")
export_review(ledger, report.run_fingerprint, "review.csv")
# fill in `decision` (accept / reject / replace / no_match / defer) and `reviewer`
apply_review(ledger, read_review("review.csv"), target_store_fingerprint=store.fingerprint)
```

Review never overwrites model output; `export_mapping_csv(..., use_review=True)` writes
the adjudicated view alongside it.
````

- [ ] **Step 8: Commit**

```bash
git add src/xwalk/batch.py src/xwalk/matcher.py src/xwalk/__init__.py README.md tests/test_batch.py
git commit -m "feat: batch runner with resume, reporting, and exports"
```

---

## Definition of done for Phase 1

- [ ] `python -m pytest -q -m "not integration"` — all green. Expect **404 tests** across 23 files, in a couple of seconds with `TMPDIR` on tmpfs.
- [ ] `python -m pytest -q -m integration` — 1 skipped without keys; passes with them.
- [ ] `python -m ruff check src tests && python -m ruff format --check src tests` — clean.
- [ ] `python -m mypy` — clean under `--strict`.
- [ ] `pip install .` in a fresh venv installs **no** torch, sentence-transformers, faiss, rdflib, or sqlalchemy. Verify with `pip list`.
- [ ] The README quickstart runs end-to-end against `tests/fixtures/*.csv` with a real provider.
- [ ] Grep the repo for `crosswalk` — zero hits.
- [ ] `git log --oneline` shows one commit per task, each with tests.

## Plan self-review notes

Checked against the spec; the following are **deliberately deferred**, not omissions:

| Spec item | Deferred to |
|---|---|
| Dense retrieval (`retrieval/dense.py`), `[dense]` extra | Phase 3 |
| Ontology (OWL/OBO) and SQL sources | Phase 3 |
| `config.py` job spec, `cli/` | Phase 3 (Phase 1 is SDK-only; the README shows the API) |
| `llm/litellm.py` | Phase 3 |
| `evaluate/*` — gold, metrics, ceiling, ablate, compare | Phase 2 |
| `prompts/author.py`, `prompts/optimize.py` | Phase 2 |
| Porting the four datasets as examples | Phase 3 |
| CI matrix and Tantivy install smoke tests | Phase 3 (Task 5 Step 1 verifies the API locally now) |
| Parquet source (spec's `tabular.py` lists `csv / tsv / jsonl / parquet`) | Phase 3 — needs `pyarrow`, so it belongs behind an extra, and the base install must stay small. TSV is already reachable via `csv_source(..., delimiter="\t")`. |
| Emitting confirmed review pairs as a gold file | Phase 2 — the consumer is the prompt optimizer, and `evaluate/gold.py` is where the gold format is defined. Task 16's `adjudicated()` already exposes everything such an exporter needs. |
| A documented alternate retriever for unsupported platforms | Phase 3, with the platform matrix. Never automatic: the spec is explicit that a silent engine switch would change ranking and undermine reproducibility. |

Three items the spec lists that Phase 1 **does** cover and must not be skipped: the
contract-safe skeletons plus `contract.py` (Task 10 — only LLM-*assisted authoring* is
deferred), duplicate-target reporting (Task 18 — reported, never resolved), and the LLM
response cache with the spec's full key derivation (Task 15 — the ledger table alone is
not enough; `CachingLLM` is what makes it a cache rather than an unused table).

## Open spec disagreements — decide before Task 1

The spec wins over this plan. These four points are places where the plan still differs,
each deliberate and each cheap to change if the ruling goes the other way.

| # | Spec says | Plan does | Why, and what changing it costs |
|---|---|---|---|
| 1 | "**Tiny fixture target of ~50 records** as CSV, so retrieval tests run in milliseconds" | 5 target rows, 4 source rows | The stated *rationale* — millisecond retrieval tests with BM25 only — is met: the suite runs in ~2 s. Five rows is what the deterministic ranking assertions in Tasks 5 and 18 are written against, and each was verified by hand. Growing to 50 means re-deriving every expected ranking; it buys realistic near-miss ranking and genuine selector-budget pressure, which today's budget tests only reach with synthetic candidates. **Recommendation: keep 5 for Phase 1, grow the fixture in Phase 2** where the ceiling diagnostic needs a population to be meaningful. |
| 2 | "Resolution is an **exact dictionary lookup** against the keys issued for that attempt; anything else is `UNRESOLVED_OUTPUT`" and "`null` is unambiguous" | `resolve_key` strips surrounding whitespace, quotes and brackets, upper-cases the key, and treats `none`/`nil`/`no_match`/`n/a`/`na` as abstention alongside `null` | The key space is issued by the library, so no tolerant path can reach a *wrong* record — every one of the spec's four adversarial cases is tested and still resolves to `UNRESOLVED`. The tolerance only ever maps to abstention or to the key the model plainly meant. But it *is* laxer than the letter of the spec, and every fuzzy hit is still labelled `EXACT_KEY`. **Recommendation: keep, and note the deviation in the module docstring.** Tightening is a one-line change to `_normalise` plus `_ABSTAIN_TOKENS`. |
| 3 | "A **single writer coroutine** serialises writes" | One `asyncio.Lock` around each write on a shared connection | Same guarantee inside one event loop, less machinery, and no queue to drain on shutdown. It is a mechanism substitution, not a weaker invariant — `test_concurrent_writes_all_land` pins the behaviour the spec is asking for. **Recommendation: keep**; the plan's architecture paragraph has been left promising "a single writer coroutine" and should be reworded if you agree. |
| 4 | Review rows carry `result_id` | `result_key` throughout | Matches `MatchResult.result_key`, so the exported CSV column and the field it refers to have one name. **Recommendation: keep**; rename is mechanical if you prefer the spec's word. |

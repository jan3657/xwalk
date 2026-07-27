# xwalk Phase 3 — Breadth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make xwalk installable and usable by someone who is not you — dense retrieval and ontology/SQL sources behind extras, ablation and run comparison, a YAML job spec, a CLI, the four paper datasets ported as working examples, and CI on the declared platform matrix.

**Architecture:** Every new capability plugs into a protocol that already exists. `DenseRetriever` is another `Retriever`; `owl_source`/`obo_source`/`sql_source` are `RecordSource`s; nothing in `matcher.py` changes. `config.py` turns a YAML job spec into exactly the objects the Phase 1 README constructs by hand, and `cli/` is a thin argument-parsing shell over `config.py` — the SDK stays primary.

**Tech Stack:** Phase 1 + 2, plus optional extras: `[dense]` (sentence-transformers, torch, faiss-cpu), `[ontology]` (rdflib), `[sql]` (sqlalchemy), `[litellm]`, `[all]`. The base install gains nothing.

**Prerequisite:** Phases 1 and 2 complete, with both "Definition of done" checklists passing.

**Spec:** `docs/superpowers/specs/2026-07-26-xwalk-generalized-matching-library-design.md`.

## Global Constraints

Every task's requirements implicitly include this section, plus Phases 1 and 2's.

- **The base install never grows.** Every dependency added in this phase sits behind an extra, and every module using one imports it lazily inside the function or guarded at module top with a clear `ImportError` message naming the extra.
- **A missing extra produces an actionable error**, not an `ImportError` traceback: `"xwalk[dense] is required for DenseRetriever; install with: pip install 'xwalk[dense]'"`.
- **Every optional-dependency test is marked** (`@pytest.mark.dense`, `@pytest.mark.ontology`, `@pytest.mark.sql`) and skips cleanly when the extra is absent. `pytest -q` on a base install must stay green.
- **The CLI is a shell.** Any logic worth testing lives in a library function the CLI calls; CLI tests assert wiring and exit codes, not behaviour.
- **No automatic retriever substitution, ever.** On an unsupported platform the user selects an alternate explicitly. A silent switch changes ranking and undermines reproducibility.
- **Ported datasets ship sample slices only.** Full data stays in the paper repo. No file over 1 MB enters git.
- **`mypy --strict`, `ruff check`, `ruff format --check` pass at the end of every task.** Commit at the end of every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/xwalk/_extras.py` | `require(extra, module)` — the one place an extra's absence is explained |
| `src/xwalk/retrieval/dense.py` | `DenseRetriever` — encoder + FAISS |
| `src/xwalk/sources/ontology.py` | `owl_source` (rdflib), `obo_source` (stanza parser, no deps) |
| `src/xwalk/sources/sql.py` | `sql_source` (SQLAlchemy) |
| `src/xwalk/llm/litellm.py` | `LiteLLMClient` adapter |
| `src/xwalk/evaluate/ablate.py` | flag-flipping wrapper over eval |
| `src/xwalk/evaluate/compare.py` | run comparison table |
| `src/xwalk/config.py` | `JobSpec` (pydantic) → constructed objects |
| `src/xwalk/cli/__init__.py`, `cli/main.py` | argument parsing and dispatch |
| `examples/{chebi,ncbi_disease,nlm_gene,cafeteria_fcd}/` | job.yaml, slots.yaml, sample slices |
| `.github/workflows/ci.yml` | platform matrix, extras matrix, Tantivy smoke test |

---

## Task 1: Optional-extra plumbing

**Files:**
- Create: `src/xwalk/_extras.py`
- Modify: `pyproject.toml` (extras + pytest markers)
- Test: `tests/test_extras.py`

**Interfaces:**
- Produces: `MissingExtra` exception; `require(extra: str, module: str, *, purpose: str) -> ModuleType`. Tasks 2–4 and 6 use it.

- [ ] **Step 1: Add the extras and markers to `pyproject.toml`**

```toml
[project.optional-dependencies]
dense = ["sentence-transformers>=3.0", "torch>=2.0", "faiss-cpu>=1.8"]
ontology = ["rdflib>=7.0"]
sql = ["sqlalchemy>=2.0"]
litellm = ["litellm>=1.40"]
all = ["xwalk[dense,ontology,sql,litellm]"]
dev = [
    "pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.6", "mypy>=1.11", "types-PyYAML",
]

[project.scripts]
xwalk = "xwalk.cli.main:main"
```

Extend `[tool.pytest.ini_options] markers`:

```toml
markers = [
    "integration: hits a real LLM provider; skipped without an API key",
    "dense: needs xwalk[dense]",
    "ontology: needs xwalk[ontology]",
    "sql: needs xwalk[sql]",
]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_extras.py`:

```python
import pytest

from xwalk._extras import MissingExtra, require


def test_require_returns_an_installed_module():
    module = require("base", "json", purpose="testing")
    assert module.dumps({"a": 1}) == '{"a": 1}'


def test_a_missing_module_raises_missing_extra():
    with pytest.raises(MissingExtra):
        require("dense", "definitely_not_installed_xyz", purpose="DenseRetriever")


def test_the_error_names_the_extra_and_the_install_command():
    with pytest.raises(MissingExtra) as exc:
        require("dense", "definitely_not_installed_xyz", purpose="DenseRetriever")
    message = str(exc.value)
    assert "xwalk[dense]" in message
    assert "pip install" in message
    assert "DenseRetriever" in message


def test_missing_extra_is_an_import_error_subclass():
    """So `except ImportError` in user code still works."""
    assert issubclass(MissingExtra, ImportError)


def test_submodules_are_importable():
    module = require("base", "os.path", purpose="testing")
    assert hasattr(module, "join")
```

- [ ] **Step 3: Write `src/xwalk/_extras.py`**

```python
"""One place where a missing optional dependency is explained.

The base install stays small on purpose: matching two CSVs with an API-hosted model
should not download torch. When a user reaches for something that does need it, the
error must say which extra to install, not raise a bare ImportError from a library
they have never heard of.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtra(ImportError):
    """An optional dependency is required but not installed."""


def require(extra: str, module: str, *, purpose: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtra(
            f"{purpose} requires the optional dependency {module!r}, which is part of "
            f"xwalk[{extra}].\n\n    pip install 'xwalk[{extra}]'\n"
        ) from exc
```

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_extras.py -v      # expect 5 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/_extras.py pyproject.toml tests/test_extras.py
git commit -m "feat: optional-extra plumbing with actionable install errors"
```

---

## Task 2: Dense retrieval

**Files:**
- Create: `src/xwalk/retrieval/dense.py`
- Modify: `src/xwalk/retrieval/__init__.py`
- Test: `tests/test_dense.py`

**Interfaces:**
- Consumes: `Record`/`RetrievalHit` (P1), `TemplateSet` (P1), `hash_value`/`hash_record` (P1), `require` (Task 1).
- Produces: `Encoder` Protocol (`name`, `dimension`, `encode(texts, *, is_query) -> list[list[float]]`); `SentenceTransformerEncoder(model_name, *, device=None, normalize=True, query_prefix="", doc_prefix="", batch_size=32)`; `DenseRetriever.build(records, templates, index_dir, encoder, *, name=None)`, `DenseRetriever.open(index_dir, encoder, *, name=None)`.

**Design note — the SapBERT problem.** The paper repo hardcodes SapBERT as "the second dense model", which is meaningless outside biomedicine. Here there is no second dense model: `Matcher` takes a list of retrievers, and running two encoders means constructing two `DenseRetriever`s with two `SentenceTransformerEncoder`s. The domain-specific choice is now a line in the user's config, not a branch in library code.

The `Encoder` protocol exists so a user can plug in an API embedding service without installing torch at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dense.py`:

```python
import math

import pytest

from xwalk.records import Record
from xwalk.retrieval.base import Retriever, SearchRequest
from xwalk.retrieval.dense import DenseRetriever
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(query="{{ mention }}", context="", doc="{{ label }}",
                        candidate="ID: {{ id }}")

RECORDS = [
    Record(id="T1", fields={"label": "glucose"}),
    Record(id="T2", fields={"label": "fructose"}),
    Record(id="T3", fields={"label": "automobile"}),
]


class ToyEncoder:
    """Deterministic, dependency-free: a character-histogram embedding.

    Enough structure for 'glucose' to sit closer to 'fructose' than to 'automobile',
    which is all the retriever's contract requires. Tests must not depend on a
    downloaded model.
    """

    name = "toy"
    dimension = 26

    def encode(self, texts, *, is_query: bool = False):
        vectors = []
        for text in texts:
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@pytest.fixture
def retriever(tmp_path):
    return DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())


async def test_an_exact_match_ranks_first(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=3))
    assert hits[0].record_id == "T1"


async def test_ranks_are_one_based_and_contiguous(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=3))
    assert [h.rank for h in hits] == [1, 2, 3]


async def test_a_semantically_closer_record_outranks_a_distant_one(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=3))
    order = [h.record_id for h in hits]
    assert order.index("T2") < order.index("T3")


async def test_hits_carry_the_encoder_derived_name(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=1))
    assert hits[0].retriever == "dense:toy"


async def test_limit_is_respected(retriever):
    assert len(await retriever.search(SearchRequest(text="glucose", limit=2))) == 2


async def test_a_blank_query_returns_nothing(retriever):
    assert await retriever.search(SearchRequest(text="   ", limit=3)) == []


async def test_scores_are_reported(retriever):
    hits = await retriever.search(SearchRequest(text="glucose", limit=1))
    assert hits[0].raw_score is not None


async def test_open_reuses_a_built_index(tmp_path):
    built = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())
    reopened = DenseRetriever.open(tmp_path / "idx", ToyEncoder())
    assert reopened.fingerprint == built.fingerprint
    hits = await reopened.search(SearchRequest(text="glucose", limit=1))
    assert hits[0].record_id == "T1"


def test_opening_a_non_index_directory_is_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="xwalk"):
        DenseRetriever.open(tmp_path / "empty", ToyEncoder())


def test_opening_with_a_different_encoder_is_refused(tmp_path):
    """A vector index built by one encoder is meaningless to another."""
    DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())

    class OtherEncoder(ToyEncoder):
        name = "other"

    with pytest.raises(ValueError, match="encoder"):
        DenseRetriever.open(tmp_path / "idx", OtherEncoder())


def test_a_dimension_mismatch_is_refused(tmp_path):
    class BadEncoder(ToyEncoder):
        dimension = 12

    with pytest.raises(ValueError, match="dimension"):
        DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", BadEncoder())


def test_fingerprint_changes_with_the_encoder(tmp_path):
    class OtherEncoder(ToyEncoder):
        name = "other"

    a = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "a", ToyEncoder())
    b = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "b", OtherEncoder())
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_with_the_records(tmp_path):
    a = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "a", ToyEncoder())
    b = DenseRetriever.build(RECORDS[:2], TEMPLATES, tmp_path / "b", ToyEncoder())
    assert a.fingerprint != b.fingerprint


def test_dense_satisfies_the_retriever_protocol(retriever):
    assert isinstance(retriever, Retriever)


def test_two_dense_retrievers_can_coexist_with_distinct_names(tmp_path):
    """This is what replaces the paper repo's hardcoded 'second dense model'."""
    class OtherEncoder(ToyEncoder):
        name = "domain"

    a = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "a", ToyEncoder())
    b = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "b", OtherEncoder())
    assert a.name != b.name


def test_a_query_prefix_is_applied_only_to_queries(tmp_path):
    seen = []

    class RecordingEncoder(ToyEncoder):
        def encode(self, texts, *, is_query: bool = False):
            seen.append((is_query, list(texts)))
            return super().encode(texts, is_query=is_query)

    encoder = RecordingEncoder()
    retriever = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", encoder)
    import asyncio

    asyncio.run(retriever.search(SearchRequest(text="glucose", limit=1)))
    assert seen[0][0] is False   # documents
    assert seen[-1][0] is True   # query


@pytest.mark.dense
def test_the_sentence_transformer_encoder_loads():
    pytest.importorskip("sentence_transformers")
    from xwalk.retrieval.dense import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = encoder.encode(["glucose"], is_query=True)
    assert len(vectors) == 1 and len(vectors[0]) == encoder.dimension


def test_dense_import_without_the_extra_gives_an_actionable_error(monkeypatch):
    """Importing the module must work; only *using* faiss should demand the extra."""
    import xwalk.retrieval.dense as module

    assert hasattr(module, "DenseRetriever")
```

- [ ] **Step 2: Write `src/xwalk/retrieval/dense.py`**

Note the storage choice: vectors are persisted as a plain binary file plus a JSON sidecar, and FAISS is used *if available* for search speed, falling back to an exact NumPy-free dot product over the stored vectors. That keeps the toy-encoder tests dependency-free while giving real users FAISS.

```python
"""Dense retrieval.

There is deliberately no "second dense model" concept. `Matcher` takes a list of
retrievers; running two encoders means constructing two DenseRetrievers. The paper repo
hardcoded SapBERT as the second dense model, which is meaningless outside biomedicine —
here it is a line in the user's config.

The `Encoder` protocol means an API embedding service can be plugged in with no torch
installed at all.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

from xwalk._extras import require
from xwalk.fingerprint import hash_record, hash_value
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest
from xwalk.templates import TemplateSet

_META_FILE = "xwalk_meta.json"
_VECTOR_FILE = "vectors.f32"
_IDS_FILE = "record_ids.json"


@runtime_checkable
class Encoder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        """Return one unit-normalised vector per text."""


class SentenceTransformerEncoder:
    """The default encoder. Requires xwalk[dense]."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize: bool = True,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        st = require("dense", "sentence_transformers", purpose="SentenceTransformerEncoder")
        self._model = st.SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self._query_prefix if is_query else self._doc_prefix
        prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            prepared,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(v) for v in row] for row in vectors]


def _write_vectors(path: Path, vectors: Sequence[Sequence[float]]) -> None:
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack(f"<{len(vector)}f", *vector))


def _read_vectors(path: Path, dimension: int) -> list[list[float]]:
    raw = path.read_bytes()
    size = dimension * 4
    return [
        list(struct.unpack(f"<{dimension}f", raw[i : i + size]))
        for i in range(0, len(raw), size)
    ]


class DenseRetriever:
    def __init__(
        self,
        *,
        vectors: Sequence[Sequence[float]],
        record_ids: Sequence[str],
        encoder: Encoder,
        fingerprint: str,
        name: str,
    ) -> None:
        self._vectors = [list(v) for v in vectors]
        self._record_ids = list(record_ids)
        self._encoder = encoder
        self._fingerprint = fingerprint
        self._name = name
        self._faiss_index = self._try_build_faiss()

    # --- protocol ---------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    # --- construction -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        records: Iterable[Record],
        templates: TemplateSet,
        index_dir: str | Path,
        encoder: Encoder,
        *,
        name: str | None = None,
        batch: int = 256,
    ) -> DenseRetriever:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        record_ids: list[str] = []
        texts: list[str] = []
        digests: list[str] = []
        for record in records:
            record_ids.append(record.id)
            texts.append(templates.render_doc(record))
            digests.append(hash_record(record))

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            vectors.extend(encoder.encode(texts[start : start + batch], is_query=False))

        for vector in vectors:
            if len(vector) != encoder.dimension:
                raise ValueError(
                    f"encoder {encoder.name!r} declares dimension {encoder.dimension} "
                    f"but produced a vector of length {len(vector)}"
                )

        fingerprint = hash_value(
            {
                "engine": "dense",
                "encoder": encoder.name,
                "dimension": encoder.dimension,
                "doc_template": templates.doc,
                "records": sorted(digests),
            }
        )

        _write_vectors(index_dir / _VECTOR_FILE, vectors)
        (index_dir / _IDS_FILE).write_text(json.dumps(record_ids), encoding="utf-8")
        (index_dir / _META_FILE).write_text(
            json.dumps(
                {
                    "engine": "dense",
                    "name": name or f"dense:{encoder.name}",
                    "encoder": encoder.name,
                    "dimension": encoder.dimension,
                    "fingerprint": fingerprint,
                    "doc_count": len(record_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(
            vectors=vectors,
            record_ids=record_ids,
            encoder=encoder,
            fingerprint=fingerprint,
            name=name or f"dense:{encoder.name}",
        )

    @classmethod
    def open(
        cls, index_dir: str | Path, encoder: Encoder, *, name: str | None = None
    ) -> DenseRetriever:
        index_dir = Path(index_dir)
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{index_dir} is not an xwalk dense index (no {_META_FILE}); build it first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["encoder"] != encoder.name:
            raise ValueError(
                f"index was built with encoder {meta['encoder']!r} but {encoder.name!r} "
                f"was supplied; a vector index is meaningless to a different encoder"
            )
        if meta["dimension"] != encoder.dimension:
            raise ValueError(
                f"index dimension {meta['dimension']} != encoder dimension {encoder.dimension}"
            )
        return cls(
            vectors=_read_vectors(index_dir / _VECTOR_FILE, meta["dimension"]),
            record_ids=json.loads((index_dir / _IDS_FILE).read_text(encoding="utf-8")),
            encoder=encoder,
            fingerprint=meta["fingerprint"],
            name=name or meta["name"],
        )

    # --- search -----------------------------------------------------------------

    def _try_build_faiss(self) -> object | None:
        """FAISS if available; otherwise exact search over the stored vectors.

        This is a *speed* fallback with identical semantics on normalised vectors
        (IndexFlatIP is exact), not a silent ranking change.
        """
        if not self._vectors:
            return None
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np
        except ImportError:
            return None
        index = faiss.IndexFlatIP(len(self._vectors[0]))
        index.add(np.asarray(self._vectors, dtype="float32"))
        return index

    def _search_sync(self, text: str, limit: int) -> list[RetrievalHit]:
        if not text.strip() or not self._vectors:
            return []
        query = self._encoder.encode([text], is_query=True)[0]

        if self._faiss_index is not None:
            import numpy as np

            scores, indices = self._faiss_index.search(  # type: ignore[attr-defined]
                np.asarray([query], dtype="float32"), min(limit, len(self._vectors))
            )
            pairs = [(float(s), int(i)) for s, i in zip(scores[0], indices[0]) if i >= 0]
        else:
            pairs = sorted(
                (
                    (sum(a * b for a, b in zip(query, vector)), i)
                    for i, vector in enumerate(self._vectors)
                ),
                key=lambda p: (-p[0], p[1]),
            )[:limit]

        return [
            RetrievalHit(
                record_id=self._record_ids[index],
                retriever=self._name,
                raw_score=score,
                rank=rank,
            )
            for rank, (score, index) in enumerate(pairs, start=1)
        ]

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return await asyncio.to_thread(self._search_sync, request.text, request.limit)
```

Add `DenseRetriever`, `Encoder`, `SentenceTransformerEncoder` to `src/xwalk/retrieval/__init__.py`, importing `SentenceTransformerEncoder` lazily is unnecessary — its constructor is what calls `require`.

- [ ] **Step 3: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_dense.py -v -m "not dense"   # expect 17 passed
python -m pytest tests/test_dense.py -v -m dense         # 1 skipped without the extra
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/retrieval tests/test_dense.py
git commit -m "feat: dense retrieval with a pluggable encoder protocol"
```

---

## Task 3: Ontology sources — OWL and OBO

**Files:**
- Create: `src/xwalk/sources/ontology.py`
- Modify: `src/xwalk/sources/__init__.py`
- Create: `tests/fixtures/tiny.obo`, `tests/fixtures/tiny.owl`
- Test: `tests/test_ontology_source.py`

**Interfaces:**
- Consumes: `Record` (P1), `require` (Task 1).
- Produces: `obo_source(path, *, id_prefix=None, include_obsolete=False) -> Iterator[Record]`; `owl_source(path, *, format=None, id_prefix=None, label_predicates=..., synonym_predicates=..., definition_predicates=..., include_obsolete=False) -> Iterator[Record]`; `curie(uri) -> str`.

**Design note:** OBO gets a hand-rolled stanza parser and needs **no** dependency — the format is line-oriented and 60 lines of parsing beats pulling rdflib for it. OWL uses rdflib behind `[ontology]`. Both emit the same field names (`label`, `synonyms`, `definition`, `parents`, `obsolete`) so one `doc` template works against either.

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/tiny.obo`:

```
format-version: 1.2
ontology: tiny

[Term]
id: CHEBI:17234
name: glucose
def: "A monosaccharide sugar." [PMID:1]
synonym: "dextrose" EXACT []
synonym: "grape sugar" RELATED []
is_a: CHEBI:16646 ! carbohydrate

[Term]
id: CHEBI:28757
name: fructose
def: "A ketohexose sugar." []
synonym: "fruit sugar" EXACT []
is_a: CHEBI:16646 ! carbohydrate

[Term]
id: CHEBI:00000
name: obsolete thing
is_obsolete: true

[Typedef]
id: part_of
name: part of
```

`tests/fixtures/tiny.owl`:

```xml
<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#"
         xmlns:obo="http://purl.obolibrary.org/obo/">
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_17234">
    <rdfs:label>glucose</rdfs:label>
    <obo:IAO_0000115>A monosaccharide sugar.</obo:IAO_0000115>
    <oboInOwl:hasExactSynonym>dextrose</oboInOwl:hasExactSynonym>
    <oboInOwl:hasRelatedSynonym>grape sugar</oboInOwl:hasRelatedSynonym>
    <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/CHEBI_16646"/>
  </owl:Class>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_28757">
    <rdfs:label>fructose</rdfs:label>
    <oboInOwl:hasExactSynonym>fruit sugar</oboInOwl:hasExactSynonym>
  </owl:Class>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_00000">
    <rdfs:label>obsolete thing</rdfs:label>
    <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
  </owl:Class>
</rdf:RDF>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_ontology_source.py`:

```python
import pytest

from tests.conftest import FIXTURES
from xwalk.sources.ontology import curie, obo_source, owl_source

OBO = FIXTURES / "tiny.obo"
OWL = FIXTURES / "tiny.owl"


# --- OBO (no dependency) ----------------------------------------------------------

def test_obo_yields_one_record_per_term():
    records = list(obo_source(OBO))
    assert [r.id for r in records] == ["CHEBI:17234", "CHEBI:28757"]


def test_obo_skips_obsolete_terms_by_default():
    assert "CHEBI:00000" not in {r.id for r in obo_source(OBO)}


def test_obo_can_include_obsolete_terms():
    ids = {r.id for r in obo_source(OBO, include_obsolete=True)}
    assert "CHEBI:00000" in ids


def test_obo_ignores_non_term_stanzas():
    assert "part_of" not in {r.id for r in obo_source(OBO)}


def test_obo_extracts_the_label():
    record = next(iter(obo_source(OBO)))
    assert record.fields["label"] == "glucose"


def test_obo_extracts_all_synonyms():
    record = next(iter(obo_source(OBO)))
    assert record.fields["synonyms"] == ["dextrose", "grape sugar"]


def test_obo_strips_the_definition_reference_list():
    record = next(iter(obo_source(OBO)))
    assert record.fields["definition"] == "A monosaccharide sugar."


def test_obo_extracts_parents_without_the_trailing_comment():
    record = next(iter(obo_source(OBO)))
    assert record.fields["parents"] == ["CHEBI:16646"]


def test_obo_records_the_obsolete_flag():
    record = next(iter(obo_source(OBO, include_obsolete=True)))
    assert record.fields["obsolete"] is False


def test_obo_filters_by_id_prefix():
    assert list(obo_source(OBO, id_prefix="GO:")) == []


def test_obo_is_lazy():
    import types

    assert isinstance(obo_source(OBO), types.GeneratorType)


# --- OWL (needs xwalk[ontology]) --------------------------------------------------

@pytest.mark.ontology
def test_owl_yields_one_record_per_class():
    pytest.importorskip("rdflib")
    assert {r.id for r in owl_source(OWL)} == {"CHEBI:17234", "CHEBI:28757"}


@pytest.mark.ontology
def test_owl_produces_the_same_field_names_as_obo():
    pytest.importorskip("rdflib")
    owl_record = next(r for r in owl_source(OWL) if r.id == "CHEBI:17234")
    obo_record = next(r for r in obo_source(OBO) if r.id == "CHEBI:17234")
    assert set(owl_record.fields) == set(obo_record.fields)


@pytest.mark.ontology
def test_owl_extracts_label_synonyms_and_definition():
    pytest.importorskip("rdflib")
    record = next(r for r in owl_source(OWL) if r.id == "CHEBI:17234")
    assert record.fields["label"] == "glucose"
    assert set(record.fields["synonyms"]) == {"dextrose", "grape sugar"}
    assert record.fields["definition"] == "A monosaccharide sugar."


@pytest.mark.ontology
def test_owl_extracts_parents_as_curies():
    pytest.importorskip("rdflib")
    record = next(r for r in owl_source(OWL) if r.id == "CHEBI:17234")
    assert record.fields["parents"] == ["CHEBI:16646"]


@pytest.mark.ontology
def test_owl_skips_deprecated_classes_by_default():
    pytest.importorskip("rdflib")
    assert "CHEBI:00000" not in {r.id for r in owl_source(OWL)}


@pytest.mark.ontology
def test_owl_synonyms_are_sorted_for_determinism():
    pytest.importorskip("rdflib")
    first = [r.fields["synonyms"] for r in owl_source(OWL)]
    second = [r.fields["synonyms"] for r in owl_source(OWL)]
    assert first == second


def test_owl_without_the_extra_gives_an_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("rdflib"):
            raise ImportError("no rdflib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    from xwalk._extras import MissingExtra

    with pytest.raises(MissingExtra, match=r"xwalk\[ontology\]"):
        list(owl_source(OWL))


# --- curie ------------------------------------------------------------------------

def test_curie_converts_an_obo_purl():
    assert curie("http://purl.obolibrary.org/obo/CHEBI_17234") == "CHEBI:17234"


def test_curie_leaves_an_unrecognised_uri_alone():
    assert curie("http://example.org/thing") == "http://example.org/thing"


def test_curie_handles_a_fragment_uri():
    assert curie("http://example.org/onto#Term_1") == "Term_1"
```

- [ ] **Step 3: Write `src/xwalk/sources/ontology.py`**

```python
"""Ontology record sources.

OBO is parsed here directly — the format is line-oriented stanzas, and sixty lines of
parsing beats pulling rdflib in for it. OWL uses rdflib behind xwalk[ontology].

Both emit the same field names — `label`, `synonyms`, `definition`, `parents`,
`obsolete` — so a single `doc` template works against either.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Sequence

from xwalk._extras import require
from xwalk.records import Record

_OBO_PURL = re.compile(r"^http://purl\.obolibrary\.org/obo/([A-Za-z0-9]+)_(.+)$")
_QUOTED = re.compile(r'^"((?:[^"\\]|\\.)*)"')

DEFAULT_LABEL_PREDICATES = ("http://www.w3.org/2000/01/rdf-schema#label",)
DEFAULT_SYNONYM_PREDICATES = (
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
)
DEFAULT_DEFINITION_PREDICATES = (
    "http://purl.obolibrary.org/obo/IAO_0000115",
    "http://www.w3.org/2004/02/skos/core#definition",
)


def curie(uri: str) -> str:
    """Compact an OBO PURL or fragment URI. Anything else passes through unchanged."""
    match = _OBO_PURL.match(uri)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri


def _obo_value(raw: str) -> str:
    """Strip a trailing ` ! comment` and surrounding whitespace."""
    return raw.split(" ! ", 1)[0].strip()


def _obo_quoted(raw: str) -> str:
    """`def:` and `synonym:` values start with a quoted string, then metadata."""
    match = _QUOTED.match(raw.strip())
    return match.group(1).replace('\\"', '"') if match else _obo_value(raw)


def obo_source(
    path: str | Path,
    *,
    id_prefix: str | None = None,
    include_obsolete: bool = False,
    encoding: str = "utf-8",
) -> Iterator[Record]:
    """Yield one Record per `[Term]` stanza. No dependencies."""
    path = Path(path)
    stanza_type: str | None = None
    fields: dict[str, Any] = {}
    term_id: str | None = None

    def finish() -> Record | None:
        if stanza_type != "Term" or not term_id:
            return None
        if id_prefix and not term_id.startswith(id_prefix):
            return None
        if fields.get("obsolete") and not include_obsolete:
            return None
        return Record(
            id=term_id,
            fields={
                "label": fields.get("label", ""),
                "synonyms": fields.get("synonyms", []),
                "definition": fields.get("definition", ""),
                "parents": fields.get("parents", []),
                "obsolete": bool(fields.get("obsolete", False)),
            },
        )

    with path.open("r", encoding=encoding) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                record = finish()
                if record is not None:
                    yield record
                stanza_type = stripped[1:-1]
                fields = {}
                term_id = None
                continue
            if not stripped or ":" not in stripped or stanza_type is None:
                continue

            key, _, raw = stripped.partition(":")
            raw = raw.strip()
            if key == "id":
                term_id = _obo_value(raw)
            elif key == "name":
                fields["label"] = _obo_value(raw)
            elif key == "def":
                fields["definition"] = _obo_quoted(raw)
            elif key == "synonym":
                fields.setdefault("synonyms", []).append(_obo_quoted(raw))
            elif key == "is_a":
                fields.setdefault("parents", []).append(_obo_value(raw))
            elif key == "is_obsolete":
                fields["obsolete"] = _obo_value(raw).lower() == "true"

    record = finish()
    if record is not None:
        yield record


def owl_source(
    path: str | Path,
    *,
    format: str | None = None,
    id_prefix: str | None = None,
    label_predicates: Sequence[str] = DEFAULT_LABEL_PREDICATES,
    synonym_predicates: Sequence[str] = DEFAULT_SYNONYM_PREDICATES,
    definition_predicates: Sequence[str] = DEFAULT_DEFINITION_PREDICATES,
    include_obsolete: bool = False,
) -> Iterator[Record]:
    """Yield one Record per owl:Class. Requires xwalk[ontology]."""
    rdflib = require("ontology", "rdflib", purpose="owl_source")

    graph = rdflib.Graph()
    graph.parse(str(path), format=format)

    rdf_type = rdflib.RDF.type
    owl_class = rdflib.OWL.Class
    subclass_of = rdflib.RDFS.subClassOf
    deprecated = rdflib.OWL.deprecated

    labels = [rdflib.URIRef(p) for p in label_predicates]
    synonyms = [rdflib.URIRef(p) for p in synonym_predicates]
    definitions = [rdflib.URIRef(p) for p in definition_predicates]

    for subject in sorted(set(graph.subjects(rdf_type, owl_class)), key=str):
        if not isinstance(subject, rdflib.URIRef):
            continue  # skip blank-node class expressions
        record_id = curie(str(subject))
        if id_prefix and not record_id.startswith(id_prefix):
            continue

        is_obsolete = any(str(o).lower() == "true" for o in graph.objects(subject, deprecated))
        if is_obsolete and not include_obsolete:
            continue

        label = next(
            (str(o) for p in labels for o in graph.objects(subject, p)), ""
        )
        definition = next(
            (str(o) for p in definitions for o in graph.objects(subject, p)), ""
        )
        synonym_values = sorted(
            {str(o) for p in synonyms for o in graph.objects(subject, p)}
        )
        parents = sorted(
            curie(str(o))
            for o in graph.objects(subject, subclass_of)
            if isinstance(o, rdflib.URIRef)
        )

        yield Record(
            id=record_id,
            fields={
                "label": label,
                "synonyms": synonym_values,
                "definition": definition,
                "parents": parents,
                "obsolete": is_obsolete,
            },
        )
```

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_ontology_source.py -v -m "not ontology"   # expect 15 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/sources tests/
git commit -m "feat: OBO and OWL record sources with a shared field shape"
```

---

## Task 4: SQL source and the LiteLLM adapter

**Files:**
- Create: `src/xwalk/sources/sql.py`, `src/xwalk/llm/litellm.py`
- Test: `tests/test_sql_source.py`, `tests/test_litellm.py`

**Interfaces:**
- Produces: `sql_source(url, query, *, id_column, params=None, multivalue_columns=(), multivalue_sep="|", chunk_size=1000) -> Iterator[Record]`; `LiteLLMClient(model, *, capabilities=None, temperature=0.0, max_tokens=1024, **kwargs)` satisfying `LLMClient`.

**Design note on SQL:** streamed with a server-side cursor in chunks, because "point at your warehouse" is exactly the case where materialising the result set defeats the purpose. `sqlite:///:memory:` is used in tests, so the extra is real but the test is fast.

- [ ] **Step 1: Write the failing SQL test**

Create `tests/test_sql_source.py`:

```python
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytestmark = pytest.mark.sql

from xwalk.sources.sql import sql_source  # noqa: E402


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = sqlalchemy.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE compounds (id TEXT, label TEXT, synonyms TEXT, tax INTEGER)"
        ))
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO compounds VALUES (:i, :l, :s, :t)"
            ),
            [
                {"i": "T1", "l": "glucose", "s": "dextrose|grape sugar", "t": 9606},
                {"i": "T2", "l": "fructose", "s": "fruit sugar", "t": 9606},
                {"i": "T3", "l": "sucrose", "s": "", "t": 10090},
            ],
        )
    return url


def test_yields_one_record_per_row(db):
    records = list(sql_source(db, "SELECT * FROM compounds", id_column="id"))
    assert [r.id for r in records] == ["T1", "T2", "T3"]


def test_drops_the_id_column_from_fields(db):
    record = next(iter(sql_source(db, "SELECT * FROM compounds", id_column="id")))
    assert "id" not in record.fields and record.fields["label"] == "glucose"


def test_splits_multivalue_columns(db):
    record = next(iter(sql_source(db, "SELECT * FROM compounds", id_column="id",
                                  multivalue_columns=["synonyms"])))
    assert record.fields["synonyms"] == ["dextrose", "grape sugar"]


def test_a_blank_multivalue_cell_becomes_an_empty_list(db):
    records = list(sql_source(db, "SELECT * FROM compounds", id_column="id",
                              multivalue_columns=["synonyms"]))
    assert records[2].fields["synonyms"] == []


def test_bound_parameters_are_supported(db):
    records = list(sql_source(db, "SELECT * FROM compounds WHERE tax = :tax",
                              id_column="id", params={"tax": 10090}))
    assert [r.id for r in records] == ["T3"]


def test_a_missing_id_column_is_a_clear_error(db):
    with pytest.raises(ValueError, match="id_column"):
        list(sql_source(db, "SELECT label FROM compounds", id_column="id"))


def test_a_null_id_is_rejected(db, tmp_path):
    engine = sqlalchemy.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("INSERT INTO compounds VALUES (NULL, 'x', '', 1)"))
    with pytest.raises(ValueError, match="blank"):
        list(sql_source(db, "SELECT * FROM compounds", id_column="id"))


def test_a_numeric_id_is_coerced_to_string(db):
    records = list(sql_source(db, "SELECT tax AS id, label FROM compounds", id_column="id"))
    assert records[0].id == "9606"


def test_the_source_is_lazy(db):
    import types

    assert isinstance(sql_source(db, "SELECT * FROM compounds", id_column="id"),
                      types.GeneratorType)
```

- [ ] **Step 2: Write `src/xwalk/sources/sql.py`**

```python
"""SQL record source. Requires xwalk[sql].

Streamed in chunks rather than materialised: "point at your warehouse" is exactly the
case where loading the whole result set defeats the purpose.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

from xwalk._extras import require
from xwalk.records import Record


def sql_source(
    url: str,
    query: str,
    *,
    id_column: str,
    params: Mapping[str, Any] | None = None,
    multivalue_columns: Sequence[str] = (),
    multivalue_sep: str = "|",
    chunk_size: int = 1000,
) -> Iterator[Record]:
    sa = require("sql", "sqlalchemy", purpose="sql_source")

    multivalue = set(multivalue_columns)
    engine = sa.create_engine(url)
    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True).execute(
            sa.text(query), dict(params or {})
        )
        columns = list(result.keys())
        if id_column not in columns:
            raise ValueError(
                f"id_column {id_column!r} not in the query result; columns are {columns!r}"
            )

        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break
            for row in rows:
                mapping = dict(zip(columns, row))
                raw_id = mapping.pop(id_column)
                if raw_id is None or not str(raw_id).strip():
                    raise ValueError(f"row with a blank {id_column!r}: {mapping!r}")
                fields: dict[str, Any] = {}
                for key, value in mapping.items():
                    if key in multivalue:
                        text = "" if value is None else str(value)
                        fields[key] = [
                            p.strip() for p in text.split(multivalue_sep) if p.strip()
                        ]
                    else:
                        fields[key] = "" if value is None else value
                yield Record(id=str(raw_id).strip(), fields=fields)
```

- [ ] **Step 3: Write the LiteLLM adapter and its test**

Create `tests/test_litellm.py`:

```python
import pytest

from xwalk.llm.base import LLMClient, LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.litellm import LiteLLMClient


class FakeCompletion:
    def __init__(self, content, *, prompt=10, completion=3):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": content})(),
                           "finish_reason": "stop"})()
        ]
        self.usage = type("U", (), {"prompt_tokens": prompt, "completion_tokens": completion})()
        self.model = "fake/model"


async def test_returns_the_content(monkeypatch):
    client = LiteLLMClient("fake/model")
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(FakeCompletion('{"a":1}')))
    response = await client.complete(LLMRequest(system="s", user="u"))
    assert response.text == '{"a":1}'


async def test_reports_usage(monkeypatch):
    client = LiteLLMClient("fake/model")
    monkeypatch.setattr(client, "_acompletion", lambda **kw: _async(FakeCompletion("x")))
    usage = (await client.complete(LLMRequest(system="", user="u"))).usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (10, 3, 1)


async def test_sends_system_and_user_messages(monkeypatch):
    seen = {}
    client = LiteLLMClient("fake/model")

    def capture(**kwargs):
        seen.update(kwargs)
        return _async(FakeCompletion("x"))

    monkeypatch.setattr(client, "_acompletion", capture)
    await client.complete(LLMRequest(system="SYS", user="USR"))
    assert seen["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


async def test_a_rate_limit_becomes_retryable(monkeypatch):
    client = LiteLLMClient("fake/model")

    def boom(**kwargs):
        raise RuntimeError("RateLimitError: too many requests")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMRetryableError):
        await client.complete(LLMRequest(system="", user="u"))


async def test_an_auth_error_becomes_fatal(monkeypatch):
    client = LiteLLMClient("fake/model")

    def boom(**kwargs):
        raise RuntimeError("AuthenticationError: bad key")

    monkeypatch.setattr(client, "_acompletion", boom)
    with pytest.raises(LLMFatalError):
        await client.complete(LLMRequest(system="", user="u"))


def test_default_capabilities_claim_nothing():
    caps = LiteLLMClient("fake/model").capabilities
    assert not caps.json_schema and not caps.strict_schema


def test_fingerprint_changes_with_the_model():
    assert LiteLLMClient("a/b").fingerprint != LiteLLMClient("c/d").fingerprint


def test_satisfies_the_llm_client_protocol():
    assert isinstance(LiteLLMClient("fake/model"), LLMClient)


async def _async(value):
    return value
```

Create `src/xwalk/llm/litellm.py`:

```python
"""LiteLLM adapter. Requires xwalk[litellm].

Capabilities default to all-false: LiteLLM proxies a hundred providers with wildly
different structured-output support, and claiming support we cannot verify is exactly
what the capability system exists to prevent. Set them explicitly per model.
"""

from __future__ import annotations

from typing import Any

from xwalk._extras import require
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

_RETRYABLE_MARKERS = ("ratelimit", "timeout", "serviceunavailable", "internalserver",
                      "apiconnection", "overloaded")
_FATAL_MARKERS = ("authentication", "permissiondenied", "notfound", "badrequest",
                  "invalidrequest")


class LiteLLMClient:
    def __init__(
        self,
        model: str,
        *,
        capabilities: LLMCapabilities | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._capabilities = capabilities or LLMCapabilities()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs
        self._module: Any = None

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
                "adapter": "litellm",
                "adapter_version": ADAPTER_VERSION,
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "kwargs": {k: str(v) for k, v in sorted(self._kwargs.items())},
            }
        )

    async def _acompletion(self, **kwargs: Any) -> Any:
        if self._module is None:
            self._module = require("litellm", "litellm", purpose="LiteLLMClient")
        return await self._module.acompletion(**kwargs)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self._max_tokens,
            **self._kwargs,
        }
        structured = request.schema is not None and self._capabilities.json_schema
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": dict(request.schema or {}),
                    "strict": self._capabilities.strict_schema,
                },
            }

        try:
            completion = await self._acompletion(**body)
        except Exception as exc:  # noqa: BLE001 - LiteLLM raises many provider types
            marker = f"{type(exc).__name__}{exc}".lower().replace("_", "")
            if any(m in marker for m in _FATAL_MARKERS):
                raise LLMFatalError(str(exc)) from exc
            if any(m in marker for m in _RETRYABLE_MARKERS):
                raise LLMRetryableError(str(exc)) from exc
            raise LLMFatalError(str(exc)) from exc

        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMFatalError("provider returned no choices")
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=getattr(choices[0].message, "content", "") or "",
            usage=Usage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                calls=1,
            ),
            model=str(getattr(completion, "model", self._model)),
            structured=structured,
            finish_reason=getattr(choices[0], "finish_reason", None),
        )
```

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_sql_source.py tests/test_litellm.py -v
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/sources/sql.py src/xwalk/llm/litellm.py tests/
git commit -m "feat: SQL record source and LiteLLM adapter"
```

---

## Task 5: Ablation and run comparison

**Files:**
- Create: `src/xwalk/evaluate/ablate.py`, `src/xwalk/evaluate/compare.py`
- Test: `tests/test_ablate.py`, `tests/test_compare.py`

**Interfaces:**
- Produces: `Ablation` (frozen: `name`, `description`, `apply: Callable[[MatcherConfig], MatcherConfig]`); `standard_ablations(retriever_names) -> list[Ablation]`; `AblationRow`/`AblationReport`; `async ablate(matcher_factory, records, gold, *, ablations=None, out=None) -> AblationReport`; `RunSummary`; `summarise_run(ledger, run_fingerprint, gold, *, label) -> RunSummary`; `compare_runs(summaries) -> str` and `compare_runs_dict(summaries) -> list[dict]`.

**Design note:** ablation flips one flag and reports the delta. The standard set is: drop each retriever in turn, disable the verifier (`verify_band=None`), disable retries (`max_attempts=1`), and halve the selector budget. The last one is there precisely because the ceiling decomposition separates budget misses — an ablation that confirms the diagnosis is worth more than one that only measures a component.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ablate.py`:

```python
import pytest

from tests.test_matcher import PROMPTS, STORE, TEMPLATES, ScriptedRetriever, score_reply, select_reply
from xwalk.evaluate.ablate import MatcherConfig, ablate, standard_ablations
from xwalk.evaluate.gold import GoldSet
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import Record
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

SOURCES = [Record(id=f"s{i}", fields={"mention": "glucose"}) for i in range(6)]
GOLD = GoldSet({f"s{i}": frozenset({"T1"}) for i in range(6)})


def factory(config: MatcherConfig) -> Matcher:
    def handler(request):
        if "## Candidates" in request.user:
            return select_reply("C01")
        return score_reply(0.95)

    llm = FakeLLM(handler=handler)
    retrievers = [
        ScriptedRetriever({"glucose": ["T1", "T2"]}, name=name)
        for name in config.retriever_names
    ] or [ScriptedRetriever({}, name="none")]
    return Matcher(
        templates=TEMPLATES, retrievers=retrievers, store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES, policy=config.selector_policy),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=config.policy, run_fingerprint=config.name,
    )


BASE = MatcherConfig(
    name="baseline", retriever_names=("bm25", "dense"),
    policy=MatchPolicy(), selector_policy=SelectorPolicy(),
)


def test_standard_ablations_cover_every_retriever():
    names = {a.name for a in standard_ablations(("bm25", "dense"))}
    assert "no_bm25" in names and "no_dense" in names


def test_standard_ablations_cover_verifier_retries_and_budget():
    names = {a.name for a in standard_ablations(("bm25",))}
    assert {"no_verifier", "no_retries", "half_budget"} <= names


def test_dropping_a_retriever_removes_it_from_the_config():
    ablation = next(a for a in standard_ablations(("bm25", "dense")) if a.name == "no_dense")
    assert ablation.apply(BASE).retriever_names == ("bm25",)


def test_disabling_the_verifier_clears_the_band():
    ablation = next(a for a in standard_ablations(("bm25",)) if a.name == "no_verifier")
    assert ablation.apply(BASE).policy.verify_band is None


def test_disabling_retries_sets_one_attempt():
    ablation = next(a for a in standard_ablations(("bm25",)) if a.name == "no_retries")
    assert ablation.apply(BASE).policy.max_attempts == 1


def test_halving_the_budget_halves_max_candidates():
    ablation = next(a for a in standard_ablations(("bm25",)) if a.name == "half_budget")
    assert ablation.apply(BASE).selector_policy.max_candidates == 15


def test_ablations_never_mutate_the_base_config():
    for ablation in standard_ablations(("bm25", "dense")):
        ablation.apply(BASE)
    assert BASE.retriever_names == ("bm25", "dense")
    assert BASE.policy.max_attempts == 4


async def test_ablate_reports_a_row_per_ablation_plus_the_baseline():
    report = await ablate(factory, BASE, SOURCES, GOLD)
    names = [row.name for row in report.rows]
    assert names[0] == "baseline"
    assert len(names) == 1 + len(standard_ablations(BASE.retriever_names))


async def test_ablate_reports_a_delta_against_the_baseline():
    report = await ablate(factory, BASE, SOURCES, GOLD)
    baseline_row = report.rows[0]
    assert baseline_row.delta == 0.0


async def test_dropping_every_retriever_shows_a_negative_delta():
    from xwalk.evaluate.ablate import Ablation

    kill = Ablation(
        name="no_retrieval", description="drop all retrievers",
        apply=lambda c: MatcherConfig(**{**c.__dict__, "retriever_names": ()}),
    )
    report = await ablate(factory, BASE, SOURCES, GOLD, ablations=[kill])
    assert report.rows[1].delta < 0


async def test_ablate_writes_a_report_when_given_a_path(tmp_path):
    import json

    await ablate(factory, BASE, SOURCES, GOLD, out=tmp_path / "ablation.json")
    json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))


async def test_ablate_rows_carry_cost_so_a_cheaper_ablation_is_visible():
    report = await ablate(factory, BASE, SOURCES, GOLD)
    assert all(row.mean_llm_calls is not None for row in report.rows)
```

Create `tests/test_compare.py`:

```python
import pytest

from tests.test_ceiling import GOLD, attempt, cand, result
from xwalk.evaluate.compare import compare_runs, compare_runs_dict, summarise_run
from xwalk.ledger import Ledger


@pytest.fixture
async def ledger(tmp_path):
    led = Ledger.open(tmp_path / "l.sqlite")
    await led.put_result(result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})]))
    await led.put_result(result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]))
    yield led
    led.close()


async def test_summarise_run_carries_the_label(ledger):
    summary = summarise_run(ledger, "fp1", GOLD, label="gpt-4o-mini")
    assert summary.label == "gpt-4o-mini"


async def test_summarise_run_reports_precision_coverage_and_cost(ledger):
    summary = summarise_run(ledger, "fp1", GOLD, label="a")
    assert summary.accepted_precision == 0.5
    assert summary.automatic_coverage is not None
    assert summary.mean_llm_calls is not None


async def test_compare_runs_renders_one_row_per_run(ledger):
    a = summarise_run(ledger, "fp1", GOLD, label="model-a")
    b = summarise_run(ledger, "fp1", GOLD, label="model-b")
    table = compare_runs([a, b])
    assert table.count("model-a") == 1 and table.count("model-b") == 1


async def test_compare_runs_includes_a_header(ledger):
    table = compare_runs([summarise_run(ledger, "fp1", GOLD, label="a")])
    assert "precision" in table.lower() and "coverage" in table.lower()


async def test_compare_runs_marks_the_best_precision(ledger):
    from xwalk.evaluate.compare import RunSummary

    a = summarise_run(ledger, "fp1", GOLD, label="a")
    b = RunSummary(**{**a.__dict__, "label": "b", "accepted_precision": 0.9})
    table = compare_runs([a, b])
    best_line = next(line for line in table.splitlines() if line.strip().startswith("*"))
    assert "b" in best_line


async def test_compare_runs_dict_is_json_safe(ledger):
    import json

    json.dumps(compare_runs_dict([summarise_run(ledger, "fp1", GOLD, label="a")]))


def test_compare_runs_with_no_summaries_does_not_crash():
    assert "no runs" in compare_runs([]).lower()
```

- [ ] **Step 2: Write `src/xwalk/evaluate/ablate.py`**

```python
"""Ablation: flip one flag, report the delta.

The standard set includes halving the selector budget precisely because the ceiling
decomposition separates budget misses from retrieval misses — an ablation that confirms
a diagnosis is worth more than one that only measures a component.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import evaluate_results
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import Record
from xwalk.stages.select import SelectorPolicy


@dataclass(frozen=True)
class MatcherConfig:
    name: str
    retriever_names: tuple[str, ...]
    policy: MatchPolicy
    selector_policy: SelectorPolicy


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    apply: Callable[[MatcherConfig], MatcherConfig]


def standard_ablations(retriever_names: Sequence[str]) -> list[Ablation]:
    ablations: list[Ablation] = []

    for name in retriever_names:
        def drop(config: MatcherConfig, *, dropped: str = name) -> MatcherConfig:
            return replace(
                config,
                name=f"no_{dropped}",
                retriever_names=tuple(n for n in config.retriever_names if n != dropped),
            )

        ablations.append(
            Ablation(name=f"no_{name}", description=f"drop the {name} retriever", apply=drop)
        )

    ablations.append(
        Ablation(
            name="no_verifier",
            description="never buy a second opinion",
            apply=lambda c: replace(
                c, name="no_verifier", policy=replace(c.policy, verify_band=None)
            ),
        )
    )
    ablations.append(
        Ablation(
            name="no_retries",
            description="one attempt only; no reformulation",
            apply=lambda c: replace(
                c, name="no_retries", policy=replace(c.policy, max_attempts=1)
            ),
        )
    )
    ablations.append(
        Ablation(
            name="half_budget",
            description="halve the selector budget",
            apply=lambda c: replace(
                c,
                name="half_budget",
                selector_policy=replace(
                    c.selector_policy,
                    max_candidates=max(1, c.selector_policy.max_candidates // 2),
                ),
            ),
        )
    )
    return ablations


@dataclass(frozen=True)
class AblationRow:
    name: str
    description: str
    accepted_precision: float | None
    automatic_coverage: float | None
    review_rate: float | None
    mean_llm_calls: float | None
    delta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "accepted_precision": self.accepted_precision,
            "automatic_coverage": self.automatic_coverage,
            "review_rate": self.review_rate,
            "mean_llm_calls": self.mean_llm_calls,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class AblationReport:
    rows: tuple[AblationRow, ...]
    objective: str

    def as_dict(self) -> dict[str, Any]:
        return {"objective": self.objective, "rows": [r.as_dict() for r in self.rows]}


async def ablate(
    matcher_factory: Callable[[MatcherConfig], Matcher],
    base: MatcherConfig,
    records: Sequence[Record],
    gold: GoldSet,
    *,
    ablations: Sequence[Ablation] | None = None,
    objective: str = "accepted_precision",
    out: str | Path | None = None,
) -> AblationReport:
    chosen = list(ablations if ablations is not None else standard_ablations(base.retriever_names))

    async def run(config: MatcherConfig) -> tuple[Any, float | None]:
        matcher = matcher_factory(config)
        results = [await matcher.match(record) for record in records]
        report = evaluate_results(results, gold)
        return report, getattr(report, objective, None)

    baseline_report, baseline_score = await run(base)
    rows = [
        AblationRow(
            name="baseline",
            description="all components enabled",
            accepted_precision=baseline_report.accepted_precision,
            automatic_coverage=baseline_report.automatic_coverage,
            review_rate=baseline_report.review_rate,
            mean_llm_calls=baseline_report.mean_llm_calls,
            delta=0.0,
        )
    ]

    for ablation in chosen:
        report, score = await run(ablation.apply(base))
        delta = 0.0
        if score is not None and baseline_score is not None:
            delta = score - baseline_score
        elif score is None and baseline_score is not None:
            delta = -baseline_score
        rows.append(
            AblationRow(
                name=ablation.name,
                description=ablation.description,
                accepted_precision=report.accepted_precision,
                automatic_coverage=report.automatic_coverage,
                review_rate=report.review_rate,
                mean_llm_calls=report.mean_llm_calls,
                delta=delta,
            )
        )

    result = AblationReport(rows=tuple(rows), objective=objective)
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result
```

- [ ] **Step 3: Write `src/xwalk/evaluate/compare.py`**

```python
"""Comparing runs. This is how you actually 'select an LLM'."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from xwalk.evaluate.ceiling import CeilingBucket, ceiling_report
from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import evaluate_results
from xwalk.ledger import Ledger

_COLUMNS = (
    ("label", 24),
    ("precision", 10),
    ("coverage", 10),
    ("review", 8),
    ("errors", 8),
    ("calls", 7),
    ("tokens", 9),
    ("secs", 7),
    ("misjudged", 10),
)


@dataclass(frozen=True)
class RunSummary:
    label: str
    run_fingerprint: str
    labelled: int
    accepted_precision: float | None
    automatic_coverage: float | None
    review_rate: float | None
    error_rate: float | None
    mean_llm_calls: float | None
    mean_tokens: float | None
    mean_seconds: float | None
    misjudged: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "run_fingerprint": self.run_fingerprint,
            "labelled": self.labelled,
            "accepted_precision": self.accepted_precision,
            "automatic_coverage": self.automatic_coverage,
            "review_rate": self.review_rate,
            "error_rate": self.error_rate,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_tokens": self.mean_tokens,
            "mean_seconds": self.mean_seconds,
            "misjudged": self.misjudged,
        }


def summarise_run(
    ledger: Ledger, run_fingerprint: str, gold: GoldSet, *, label: str
) -> RunSummary:
    results = list(ledger.iter_results(run_fingerprint))
    metrics = evaluate_results(results, gold)
    ceiling = ceiling_report(results, gold)
    return RunSummary(
        label=label,
        run_fingerprint=run_fingerprint,
        labelled=metrics.labelled,
        accepted_precision=metrics.accepted_precision,
        automatic_coverage=metrics.automatic_coverage,
        review_rate=metrics.review_rate,
        error_rate=metrics.error_rate,
        mean_llm_calls=metrics.mean_llm_calls,
        mean_tokens=metrics.mean_tokens,
        mean_seconds=metrics.mean_seconds,
        misjudged=ceiling.buckets.get(CeilingBucket.MISJUDGED, 0),
    )


def _cell(value: Any, width: int) -> str:
    if value is None:
        text = "n/a"
    elif isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return text[:width].ljust(width)


def compare_runs(summaries: Sequence[RunSummary]) -> str:
    if not summaries:
        return "no runs to compare"

    best = max(
        (s for s in summaries if s.accepted_precision is not None),
        key=lambda s: s.accepted_precision or 0.0,
        default=None,
    )

    header = "  " + " ".join(name.ljust(width) for name, width in _COLUMNS)
    lines = [header, "  " + "-" * (len(header) - 2)]
    for summary in summaries:
        marker = "* " if best is not None and summary is best else "  "
        lines.append(
            marker
            + " ".join(
                _cell(value, width)
                for value, (_, width) in zip(
                    (
                        summary.label,
                        summary.accepted_precision,
                        summary.automatic_coverage,
                        summary.review_rate,
                        summary.error_rate,
                        summary.mean_llm_calls,
                        summary.mean_tokens,
                        summary.mean_seconds,
                        summary.misjudged,
                    ),
                    _COLUMNS,
                )
            )
        )
    return "\n".join(lines)


def compare_runs_dict(summaries: Sequence[RunSummary]) -> list[dict[str, Any]]:
    return [s.as_dict() for s in summaries]
```

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_ablate.py tests/test_compare.py -v   # expect 19 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/evaluate tests/
git commit -m "feat: component ablation and run comparison"
```

---

## Task 6: The job spec

**Files:**
- Create: `src/xwalk/config.py`
- Test: `tests/test_config.py`
- Create: `tests/fixtures/job_tiny.yaml`

**Interfaces:**
- Produces: pydantic models `SourceSpec`, `TargetSpec`, `TemplateSpec`, `RetrieverSpec`, `LLMSpec`, `PolicySpec`, `JobSpec`; `load_job(path) -> JobSpec`; `JobSpec.build_templates()`, `.build_target_records()`, `.build_source_records()`, `.build_store()`, `.build_retrievers(store_records, templates, index_dir)`, `.build_llm()`, `.build_prompts()`, `.build_matcher(...)`, `.run_fingerprint(...)`.

**Design note:** a config file is *serialized constructor arguments*, nothing more. Every field maps to a parameter the Phase 1 README passes by hand. No config field ever gates behaviour that the SDK cannot express, and API keys are read from named environment variables — never stored in the file.

- [ ] **Step 1: Create `tests/fixtures/job_tiny.yaml`**

```yaml
name: tiny
templates:
  query: "{{ mention }}"
  context: "{{ context_left }} [{{ mention }}] {{ context_right }}"
  doc: "{{ label }} {{ synonyms | join(' ') }}"
  candidate: "ID: {{ id }} Label: {{ label }}"

target:
  kind: csv
  path: tests/fixtures/targets_tiny.csv
  id_column: id
  multivalue_columns: [synonyms]

source:
  kind: csv
  path: tests/fixtures/sources_tiny.csv
  id_column: mention_id

retrievers:
  - kind: bm25
    name: bm25
    limit: 20

llm:
  kind: openai_compat
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  profile: openai
  temperature: 0.0

prompts:
  slots: examples/chemistry/slots.yaml

policy:
  max_attempts: 4
  accept_at: 0.6
  review_floor: 0.4
  verify_band: [0.6, 0.8]
  audit_rate: 0.0
  concurrency: 8

selector:
  max_candidates: 30
  max_candidate_tokens: 8000
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest
import yaml

from tests.conftest import FIXTURES
from xwalk.config import JobSpec, load_job

JOB = FIXTURES / "job_tiny.yaml"


def test_loads_a_job_file():
    assert load_job(JOB).name == "tiny"


def test_builds_the_template_set():
    templates = load_job(JOB).build_templates()
    from xwalk.records import Record

    assert templates.render_query(Record(id="s1", fields={"mention": "glucose"})) == "glucose"


def test_builds_target_records_with_multivalue_columns():
    records = list(load_job(JOB).build_target_records())
    assert records[0].fields["synonyms"] == ["dextrose", "grape sugar"]


def test_builds_source_records():
    assert len(list(load_job(JOB).build_source_records())) == 4


def test_builds_an_in_memory_store():
    store = load_job(JOB).build_store()
    assert len(store) == 5


def test_builds_a_bm25_retriever(tmp_path):
    job = load_job(JOB)
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path
    )
    assert [r.name for r in retrievers] == ["bm25"]


def test_builds_prompts_from_the_slots_file():
    prompts = load_job(JOB).build_prompts()
    assert prompts.slots.entity_noun == "chemical entity mention"


def test_builds_the_policy_with_the_declared_values():
    policy = load_job(JOB).build_policy()
    assert policy.accept_at == 0.6 and policy.verify_band == (0.6, 0.8)


def test_the_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = load_job(JOB).build_llm()
    assert llm.model == "gpt-4o-mini"


def test_a_missing_api_key_env_var_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_job(JOB).build_llm()


def test_an_api_key_written_inline_is_rejected(tmp_path):
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    data["llm"]["api_key"] = "sk-oops"
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_job(path)


def test_an_unknown_source_kind_is_rejected(tmp_path):
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    data["source"]["kind"] = "carrier_pigeon"
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="carrier_pigeon"):
        load_job(path)


def test_an_unknown_retriever_kind_is_rejected(tmp_path):
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    data["retrievers"] = [{"kind": "telepathy"}]
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="telepathy"):
        load_job(path)


def test_an_empty_retriever_list_is_rejected(tmp_path):
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    data["retrievers"] = []
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="retriever"):
        load_job(path)


def test_relative_paths_resolve_against_the_job_file(tmp_path):
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    (tmp_path / "targets.csv").write_text(
        (FIXTURES / "targets_tiny.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    data["target"]["path"] = "targets.csv"
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    job = load_job(path)
    assert len(list(job.build_target_records())) == 5


def test_the_run_fingerprint_is_reproducible(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    job = load_job(JOB)
    targets = list(job.build_target_records())
    templates = job.build_templates()
    a = job.run_fingerprint(
        store=job.build_store(),
        retrievers=job.build_retrievers(targets, templates, tmp_path / "a"),
        llm=job.build_llm(),
    )
    b = job.run_fingerprint(
        store=job.build_store(),
        retrievers=job.build_retrievers(targets, templates, tmp_path / "b"),
        llm=job.build_llm(),
    )
    assert a == b


def test_changing_the_job_changes_the_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    job = load_job(JOB)
    other = job.model_copy(
        update={"policy": job.policy.model_copy(update={"accept_at": 0.9})}
    )
    targets = list(job.build_target_records())
    templates = job.build_templates()
    args = dict(
        store=job.build_store(),
        retrievers=job.build_retrievers(targets, templates, tmp_path / "a"),
        llm=job.build_llm(),
    )
    assert job.run_fingerprint(**args) != other.run_fingerprint(**args)


def test_a_job_spec_can_be_constructed_in_python_without_a_file():
    """The SDK stays primary; the file is serialized constructor arguments."""
    spec = JobSpec.model_validate(yaml.safe_load(JOB.read_text(encoding="utf-8")))
    assert spec.name == "tiny"
```

- [ ] **Step 3: Write `src/xwalk/config.py`**

```python
"""A job spec is serialized constructor arguments — nothing more.

Every field maps to a parameter the README passes by hand. No config field gates
behaviour the SDK cannot express, and credentials are named environment variables,
never values in the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import yaml
from pydantic import BaseModel, Field, model_validator

from xwalk.batch import build_run_fingerprint
from xwalk.llm.base import LLMClient
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet, load_slots
from xwalk.records import Record
from xwalk.retrieval.base import Retriever
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.sources.tabular import csv_source, jsonl_source
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.stores.base import TargetStore
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet


class TemplateSpec(BaseModel):
    query: str
    context: str = ""
    doc: str
    candidate: str


class RecordSpec(BaseModel):
    kind: Literal["csv", "tsv", "jsonl", "obo", "owl", "sql"]
    path: str | None = None
    id_column: str = "id"
    id_field: str | None = None
    multivalue_columns: list[str] = Field(default_factory=list)
    multivalue_sep: str = "|"
    # sql
    url: str | None = None
    query: str | None = None
    # ontology
    id_prefix: str | None = None
    include_obsolete: bool = False

    def build(self, base_dir: Path) -> Iterator[Record]:
        if self.kind in ("csv", "tsv"):
            if self.path is None:
                raise ValueError(f"{self.kind} source needs a path")
            return csv_source(
                base_dir / self.path,
                id_column=self.id_column,
                delimiter="\t" if self.kind == "tsv" else ",",
                multivalue_columns=self.multivalue_columns,
                multivalue_sep=self.multivalue_sep,
            )
        if self.kind == "jsonl":
            if self.path is None:
                raise ValueError("jsonl source needs a path")
            return jsonl_source(base_dir / self.path, id_field=self.id_field or self.id_column)
        if self.kind == "obo":
            from xwalk.sources.ontology import obo_source

            if self.path is None:
                raise ValueError("obo source needs a path")
            return obo_source(
                base_dir / self.path,
                id_prefix=self.id_prefix,
                include_obsolete=self.include_obsolete,
            )
        if self.kind == "owl":
            from xwalk.sources.ontology import owl_source

            if self.path is None:
                raise ValueError("owl source needs a path")
            return owl_source(
                base_dir / self.path,
                id_prefix=self.id_prefix,
                include_obsolete=self.include_obsolete,
            )
        from xwalk.sources.sql import sql_source

        if not (self.url and self.query):
            raise ValueError("sql source needs url and query")
        return sql_source(
            self.url,
            self.query,
            id_column=self.id_column,
            multivalue_columns=self.multivalue_columns,
            multivalue_sep=self.multivalue_sep,
        )


class RetrieverSpec(BaseModel):
    kind: Literal["bm25", "dense"]
    name: str | None = None
    limit: int = 20
    # dense only
    model: str | None = None
    device: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""

    @model_validator(mode="after")
    def _check(self) -> RetrieverSpec:
        if self.kind == "dense" and not self.model:
            raise ValueError("a dense retriever needs a model name")
        return self


class LLMSpec(BaseModel):
    kind: Literal["openai_compat", "litellm"] = "openai_compat"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None  # present only so it can be rejected
    profile: str = "unknown"
    temperature: float = 0.0
    max_tokens: int = 1024

    @model_validator(mode="after")
    def _no_inline_secrets(self) -> LLMSpec:
        if self.api_key is not None:
            raise ValueError(
                "api_key must not appear in a job file; use api_key_env with the name of "
                "an environment variable"
            )
        if self.kind == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat needs a base_url")
        return self


class PolicySpec(BaseModel):
    max_attempts: int = 4
    accept_at: float = 0.6
    review_floor: float = 0.4
    verify_band: tuple[float, float] | None = (0.6, 0.8)
    audit_rate: float = 0.0
    concurrency: int = 32
    legacy_id_resolution: bool = False
    retriever_timeout: float = 60.0


class SelectorSpec(BaseModel):
    max_candidates: int = 30
    max_candidate_tokens: int = 8_000


class PromptSpec(BaseModel):
    slots: str


class JobSpec(BaseModel):
    name: str
    templates: TemplateSpec
    target: RecordSpec
    source: RecordSpec
    retrievers: list[RetrieverSpec] = Field(min_length=1)
    llm: LLMSpec
    prompts: PromptSpec
    policy: PolicySpec = PolicySpec()
    selector: SelectorSpec = SelectorSpec()

    base_dir: Path = Path(".")

    @model_validator(mode="after")
    def _need_a_retriever(self) -> JobSpec:
        if not self.retrievers:
            raise ValueError("at least one retriever is required")
        return self

    # --- builders ---------------------------------------------------------------

    def build_templates(self) -> TemplateSet:
        return TemplateSet(**self.templates.model_dump())

    def build_target_records(self) -> Iterator[Record]:
        return self.target.build(self.base_dir)

    def build_source_records(self) -> Iterator[Record]:
        return self.source.build(self.base_dir)

    def build_store(self) -> MemoryStore:
        return MemoryStore.from_source(self.build_target_records())

    def build_retrievers(
        self,
        records: Sequence[Record],
        templates: TemplateSet,
        index_dir: str | Path,
    ) -> list[Retriever]:
        index_dir = Path(index_dir)
        built: list[Retriever] = []
        for spec in self.retrievers:
            target = index_dir / (spec.name or spec.kind)
            if spec.kind == "bm25":
                built.append(
                    BM25Retriever.build(
                        records, templates, target,
                        name=spec.name or "bm25",
                        exact_fields=spec.exact_fields or ("label", "synonyms"),
                    )
                )
            else:
                from xwalk.retrieval.dense import DenseRetriever, SentenceTransformerEncoder

                encoder = SentenceTransformerEncoder(
                    spec.model or "",
                    device=spec.device,
                    query_prefix=spec.query_prefix,
                    doc_prefix=spec.doc_prefix,
                )
                built.append(
                    DenseRetriever.build(records, templates, target, encoder, name=spec.name)
                )
        return built

    def build_llm(self) -> LLMClient:
        api_key = None
        if self.llm.api_key_env:
            api_key = os.environ.get(self.llm.api_key_env)
            if not api_key:
                raise ValueError(
                    f"environment variable {self.llm.api_key_env} is not set; "
                    f"export it or change llm.api_key_env in the job file"
                )
        if self.llm.kind == "openai_compat":
            return OpenAICompatClient(
                base_url=self.llm.base_url or "",
                model=self.llm.model,
                api_key=api_key,
                profile=self.llm.profile,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
        from xwalk.llm.litellm import LiteLLMClient

        return LiteLLMClient(
            self.llm.model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )

    def build_prompts(self) -> PromptSet:
        return PromptSet.from_slots(load_slots(self.base_dir / self.prompts.slots))

    def build_policy(self) -> MatchPolicy:
        return MatchPolicy(**self.policy.model_dump())

    def build_selector_policy(self) -> SelectorPolicy:
        return SelectorPolicy(**self.selector.model_dump())

    def run_fingerprint(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        llm: LLMClient,
    ) -> str:
        return build_run_fingerprint(
            templates=self.build_templates(),
            prompts=self.build_prompts(),
            store=store,
            retrievers=retrievers,
            llm=llm,
            policy=self.build_policy(),
            selector_policy=self.build_selector_policy(),
        )

    def build_matcher(
        self,
        *,
        store: TargetStore,
        retrievers: Sequence[Retriever],
        llm: LLMClient,
    ) -> Matcher:
        templates = self.build_templates()
        prompts = self.build_prompts()
        policy = self.build_policy()
        return Matcher(
            templates=templates,
            retrievers=list(retrievers),
            store=store,
            selector=Selector(
                llm, prompts, templates,
                policy=self.build_selector_policy(),
                legacy_id_resolution=policy.legacy_id_resolution,
            ),
            scorer=Scorer(llm, prompts, templates, review_floor=policy.review_floor),
            verifier=Verifier(llm, prompts, templates),
            rewriter=QueryRewriter(llm, prompts, templates),
            policy=policy,
            run_fingerprint=self.run_fingerprint(
                store=store, retrievers=retrievers, llm=llm
            ),
            retriever_limit=max(spec.limit for spec in self.retrievers),
        )


def load_job(path: str | Path) -> JobSpec:
    path = Path(path)
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = JobSpec.model_validate(data)
    return spec.model_copy(update={"base_dir": path.parent.resolve()})
```

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_config.py -v   # expect 18 passed
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/config.py tests/
git commit -m "feat: YAML job spec as serialized constructor arguments"
```

---

## Task 7: The CLI

**Files:**
- Create: `src/xwalk/cli/__init__.py`, `src/xwalk/cli/main.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `main(argv: Sequence[str] | None = None) -> int`; subcommands `index`, `match`, `eval`, `compare`, `ablate`, `prompts draft`, `prompts optimize`, `review export`, `review apply`.

**Design note:** the CLI is a shell. Every subcommand parses arguments, calls one library function, prints, and returns an exit code. Nothing worth testing lives here — the tests assert wiring and exit codes only.

Exit codes: `0` success, `1` a run completed but has a non-empty review bucket or a failed metric threshold, `2` usage error, `3` a runtime failure (missing key, unreadable file).

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import json

import pytest

from tests.conftest import FIXTURES
from xwalk.cli.main import main

JOB = str(FIXTURES / "job_tiny.yaml")


def test_no_arguments_prints_usage_and_returns_two(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_version_prints_the_version(capsys):
    assert main(["--version"]) == 0
    from xwalk import __version__

    assert __version__ in capsys.readouterr().out


def test_an_unknown_subcommand_returns_two(capsys):
    assert main(["telepathy"]) == 2


def test_index_builds_the_retriever_indexes(tmp_path):
    assert main(["index", "--job", JOB, "--out", str(tmp_path / "idx")]) == 0
    assert (tmp_path / "idx" / "bm25" / "xwalk_meta.json").exists()


def test_index_reports_the_document_count(tmp_path, capsys):
    main(["index", "--job", JOB, "--out", str(tmp_path / "idx")])
    assert "5" in capsys.readouterr().out


def test_match_without_an_api_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["match", "--job", JOB, "--out", str(tmp_path / "run")]) == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_match_with_a_nonexistent_job_fails_cleanly(tmp_path, capsys):
    assert main(["match", "--job", "nope.yaml", "--out", str(tmp_path)]) == 3
    assert "nope.yaml" in capsys.readouterr().err


def test_eval_prints_a_report(tmp_path, capsys):
    from tests.test_ceiling import GOLD, attempt, cand, result  # noqa: F401
    import asyncio

    from xwalk.ledger import Ledger

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    asyncio.run(ledger.put_result(result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])))
    ledger.put_manifest("fp1", {"run_fingerprint": "fp1"})
    ledger.close()
    (run_dir / "manifest.json").write_text(json.dumps({"run_fingerprint": "fp1"}),
                                           encoding="utf-8")
    gold = tmp_path / "gold.csv"
    gold.write_text("source_id,gold_ids\ns1,T1\n", encoding="utf-8")

    assert main(["eval", "--run", str(run_dir), "--gold", str(gold)]) == 0
    assert "Where to spend effort" in capsys.readouterr().out


def test_eval_writes_a_report_file_when_asked(tmp_path):
    # reuse the setup above via a helper in the real implementation
    pytest.skip("covered by test_eval_prints_a_report plus write_report's own tests")


def test_review_export_writes_a_csv(tmp_path):
    import asyncio

    from tests.test_serde import sample_result
    from xwalk.ledger import Ledger
    from xwalk.records import DecisionReason, MatchResult, MatchStatus

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    base = sample_result()
    asyncio.run(
        ledger.put_result(
            MatchResult(**{**base.__dict__, "status": MatchStatus.NEEDS_REVIEW,
                           "reason": DecisionReason.BELOW_ACCEPT_THRESHOLD})
        )
    )
    ledger.put_manifest("fp1", {"run_fingerprint": "fp1"})
    ledger.close()
    (run_dir / "manifest.json").write_text(json.dumps({"run_fingerprint": "fp1"}),
                                           encoding="utf-8")

    out = tmp_path / "review.csv"
    assert main(["review", "export", "--run", str(run_dir), "--out", str(out)]) == 0
    assert "result_key" in out.read_text(encoding="utf-8")


def test_every_subcommand_appears_in_the_help(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for name in ("index", "match", "eval", "compare", "ablate", "prompts", "review"):
        assert name in out


def test_help_for_a_subcommand_lists_its_flags(capsys):
    with pytest.raises(SystemExit):
        main(["match", "--help"])
    assert "--resume" in capsys.readouterr().out
```

- [ ] **Step 2: Write `src/xwalk/cli/main.py`**

```python
"""The CLI: a thin shell over the SDK.

Every subcommand parses arguments, calls one library function, prints, and returns an
exit code. Nothing worth testing lives here.

Exit codes:
  0  success
  1  the run completed but something needs attention (non-empty review bucket)
  2  usage error
  3  runtime failure (missing key, unreadable file, unsupported platform)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from xwalk import __version__

EXIT_OK = 0
EXIT_ATTENTION = 1
EXIT_USAGE = 2
EXIT_RUNTIME = 3


def _run_fingerprint_of(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["run_fingerprint"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xwalk", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    index = sub.add_parser("index", help="build retriever indexes for a job")
    index.add_argument("--job", required=True)
    index.add_argument("--out", required=True, help="index directory")

    match = sub.add_parser("match", help="run a matching job")
    match.add_argument("--job", required=True)
    match.add_argument("--out", required=True, help="run directory")
    match.add_argument("--index", default=None, help="index directory (default: <out>/index)")
    match.add_argument("--resume", action="store_true", default=True)
    match.add_argument("--no-resume", dest="resume", action="store_false")
    match.add_argument("--limit", type=int, default=None, help="process only the first N records")

    ev = sub.add_parser("eval", help="evaluate a completed run against gold labels")
    ev.add_argument("--run", required=True)
    ev.add_argument("--gold", required=True)
    ev.add_argument("--out", default=None, help="write <out>.json and <out>.txt")

    comp = sub.add_parser("compare", help="compare completed runs")
    comp.add_argument("--gold", required=True)
    comp.add_argument("--run", action="append", required=True,
                      help="LABEL=PATH, repeatable")

    abl = sub.add_parser("ablate", help="re-run with each component disabled")
    abl.add_argument("--job", required=True)
    abl.add_argument("--gold", required=True)
    abl.add_argument("--out", required=True)

    prompts = sub.add_parser("prompts", help="draft or optimise prompt slots")
    prompt_sub = prompts.add_subparsers(dest="prompts_command")
    draft = prompt_sub.add_parser("draft")
    draft.add_argument("--job", required=True)
    draft.add_argument("--describe", required=True)
    draft.add_argument("--out", required=True)
    optimise = prompt_sub.add_parser("optimize")
    optimise.add_argument("--job", required=True)
    optimise.add_argument("--gold", required=True)
    optimise.add_argument("--out", required=True)
    optimise.add_argument("--role", default="selector")
    optimise.add_argument("--rounds", type=int, default=4)
    optimise.add_argument("--max-calls", type=int, default=None)

    review = sub.add_parser("review", help="export or apply human review")
    review_sub = review.add_subparsers(dest="review_command")
    export = review_sub.add_parser("export")
    export.add_argument("--run", required=True)
    export.add_argument("--out", required=True)
    apply_ = review_sub.add_parser("apply")
    apply_.add_argument("--run", required=True)
    apply_.add_argument("--reviewed", required=True)
    apply_.add_argument("--job", required=True)

    return parser


def _cmd_index(args: argparse.Namespace) -> int:
    from xwalk.config import load_job

    job = load_job(args.job)
    templates = job.build_templates()
    records = list(job.build_target_records())
    retrievers = job.build_retrievers(records, templates, args.out)
    print(f"indexed {len(records)} target records into {len(retrievers)} retriever(s)")
    for retriever in retrievers:
        print(f"  {retriever.name}: {retriever.fingerprint}")
    return EXIT_OK


def _cmd_match(args: argparse.Namespace) -> int:
    from xwalk.batch import run_batch
    from xwalk.config import load_job

    job = load_job(args.job)
    index_dir = Path(args.index or Path(args.out) / "index")
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, templates, index_dir)
    llm = job.build_llm()
    matcher = job.build_matcher(store=store, retrievers=retrievers, llm=llm)

    records = list(job.build_source_records())
    if args.limit is not None:
        records = records[: args.limit]

    report = asyncio.run(
        run_batch(
            matcher, records, out=args.out, resume=args.resume,
            manifest_extra={"job": job.name, "model": job.llm.model},
        )
    )
    print(f"matched {report.total} records into {report.out_dir}")
    for status, count in sorted(report.by_status().items(), key=lambda kv: kv[0].value):
        print(f"  {status.value:<14}: {count}")
    duplicates = report.duplicate_targets()
    if duplicates:
        print(f"  duplicate targets: {len(duplicates)} (see manifest.json)")
    review_count = len(report.needs_review())
    if review_count:
        print(f"\n{review_count} rows need review: xwalk review export --run {args.out}")
        return EXIT_ATTENTION
    return EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.evaluate.report import evaluate, render_report, write_report
    from xwalk.ledger import Ledger

    run_dir = Path(args.run)
    gold = load_gold_csv(args.gold)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        report = evaluate(ledger, _run_fingerprint_of(run_dir), gold)
    finally:
        ledger.close()
    print(render_report(report))
    if args.out:
        write_report(report, args.out)
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    from xwalk.evaluate.compare import compare_runs, summarise_run
    from xwalk.evaluate.gold import load_gold_csv
    from xwalk.ledger import Ledger

    gold = load_gold_csv(args.gold)
    summaries = []
    for entry in args.run:
        if "=" not in entry:
            print(f"--run expects LABEL=PATH, got {entry!r}", file=sys.stderr)
            return EXIT_USAGE
        label, _, path = entry.partition("=")
        run_dir = Path(path)
        ledger = Ledger.open(run_dir / "ledger.sqlite")
        try:
            summaries.append(
                summarise_run(ledger, _run_fingerprint_of(run_dir), gold, label=label)
            )
        finally:
            ledger.close()
    print(compare_runs(summaries))
    return EXIT_OK


def _cmd_ablate(args: argparse.Namespace) -> int:
    from xwalk.config import load_job
    from xwalk.evaluate.ablate import MatcherConfig, ablate
    from xwalk.evaluate.gold import load_gold_csv

    job = load_job(args.job)
    gold = load_gold_csv(args.gold)
    templates = job.build_templates()
    targets = list(job.build_target_records())
    store = job.build_store()
    llm = job.build_llm()
    index_dir = Path(args.out) / "index"

    def factory(config: MatcherConfig):
        retrievers = [
            r
            for r in job.build_retrievers(targets, templates, index_dir)
            if r.name in config.retriever_names
        ]
        spec = job.model_copy(
            update={
                "policy": job.policy.model_copy(update=config.policy.__dict__),
                "selector": job.selector.model_copy(
                    update={
                        "max_candidates": config.selector_policy.max_candidates,
                        "max_candidate_tokens": config.selector_policy.max_candidate_tokens,
                    }
                ),
            }
        )
        return spec.build_matcher(store=store, retrievers=retrievers, llm=llm)

    base = MatcherConfig(
        name="baseline",
        retriever_names=tuple(s.name or s.kind for s in job.retrievers),
        policy=job.build_policy(),
        selector_policy=job.build_selector_policy(),
    )
    report = asyncio.run(
        ablate(
            factory, base, list(job.build_source_records()), gold,
            out=Path(args.out) / "ablation.json",
        )
    )
    for row in report.rows:
        print(f"  {row.name:<16} {row.description:<34} delta {row.delta:+.3f}")
    return EXIT_OK


def _cmd_prompts(args: argparse.Namespace) -> int:
    from xwalk.config import load_job

    job = load_job(args.job)

    if args.prompts_command == "draft":
        from xwalk.prompts.author import draft_slots, slots_diff, write_slots
        from xwalk.prompts.contract import load_slots

        sources = list(job.build_source_records())[:8]
        targets = list(job.build_target_records())[:8]
        existing = None
        slots_path = job.base_dir / job.prompts.slots
        if slots_path.exists():
            existing = load_slots(slots_path)
        draft = asyncio.run(
            draft_slots(
                job.build_llm(), description=args.describe,
                source_samples=sources, target_samples=targets, existing=existing,
            )
        )
        for warning in draft.warnings:
            print(f"warning: {warning}")
        if existing is not None:
            print(slots_diff(existing, draft.slots) or "(no changes)")
        write_slots(draft.slots, args.out)
        print(f"wrote {args.out}")
        return EXIT_OK

    if args.prompts_command == "optimize":
        from xwalk.evaluate.failures import PromptRole
        from xwalk.evaluate.gold import load_gold_csv
        from xwalk.prompts.contract import load_slots
        from xwalk.prompts.optimize import OptimizeConfig, optimize_prompt

        gold = load_gold_csv(args.gold)
        templates = job.build_templates()
        targets = list(job.build_target_records())
        store = job.build_store()
        llm = job.build_llm()
        retrievers = job.build_retrievers(targets, templates, Path(args.out) / "index")

        def factory(prompts):
            matcher = job.build_matcher(store=store, retrievers=retrievers, llm=llm)
            return matcher  # prompts are wired via job.build_prompts(); see note below

        report = asyncio.run(
            optimize_prompt(
                matcher_factory=factory,
                source_records=list(job.build_source_records()),
                gold=gold,
                initial=load_slots(job.base_dir / job.prompts.slots),
                optimiser_llm=llm,
                config=OptimizeConfig(
                    role=PromptRole(args.role), rounds=args.rounds, max_calls=args.max_calls
                ),
                work_dir=args.out,
                progress=print,
            )
        )
        print(f"stopped: {report.stopped_because}")
        if report.test_report is not None:
            print(f"test accepted precision: {report.test_report.accepted_precision}")
        return EXIT_OK

    print("prompts needs a subcommand: draft or optimize", file=sys.stderr)
    return EXIT_USAGE


def _cmd_review(args: argparse.Namespace) -> int:
    from xwalk.ledger import Ledger
    from xwalk.review import apply_review, export_review, read_review

    run_dir = Path(args.run)
    ledger = Ledger.open(run_dir / "ledger.sqlite")
    try:
        run_fp = _run_fingerprint_of(run_dir)
        if args.review_command == "export":
            count = export_review(ledger, run_fp, args.out)
            print(f"exported {count} rows to {args.out}")
            return EXIT_OK
        if args.review_command == "apply":
            from xwalk.config import load_job

            job = load_job(args.job)
            report = apply_review(
                ledger, read_review(args.reviewed),
                target_store_fingerprint=job.build_store().fingerprint,
            )
            print(f"applied {report.applied} decisions")
            return EXIT_OK
    finally:
        ledger.close()

    print("review needs a subcommand: export or apply", file=sys.stderr)
    return EXIT_USAGE


_DISPATCH = {
    "index": _cmd_index,
    "match": _cmd_match,
    "eval": _cmd_eval,
    "compare": _cmd_compare,
    "ablate": _cmd_ablate,
    "prompts": _cmd_prompts,
    "review": _cmd_review,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(__version__)
        return EXIT_OK
    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        return handler(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001 - a CLI must not dump a traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Fix the `prompts optimize` wiring**

The `factory` above ignores the `prompts` argument, which silently disables optimisation — every round would run the original slots. Fix it by threading prompts through `build_matcher`: add a `prompts: PromptSet | None = None` parameter to `JobSpec.build_matcher` that overrides `self.build_prompts()` when supplied, and have the factory pass it. Add a test in `tests/test_config.py`:

```python
def test_build_matcher_accepts_overridden_prompts(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from xwalk.prompts.contract import PromptSet, PromptSlots

    job = load_job(JOB)
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, job.build_templates(), tmp_path)
    other = PromptSet.from_slots(
        PromptSlots(entity_noun="x", target_noun="y", domain_brief="z",
                    rubric=[{"score": 1.0, "name": "a", "when": "b"},
                            {"score": 0.4, "name": "c", "when": "d"}])
    )
    matcher = job.build_matcher(store=store, retrievers=retrievers, llm=job.build_llm(),
                                prompts=other)
    assert matcher.run_fingerprint != job.run_fingerprint(
        store=store, retrievers=retrievers, llm=job.build_llm()
    )
```

- [ ] **Step 4: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_cli.py tests/test_config.py -v
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add src/xwalk/cli src/xwalk/config.py tests/
git commit -m "feat: CLI shell over the SDK"
```

---

## Task 8: Port the four datasets as examples

**Files:**
- Create: `examples/chebi/{job.yaml,slots.yaml,README.md,sample/}`
- Create: `examples/ncbi_disease/{job.yaml,slots.yaml,gold_normalize.py,README.md,sample/}`
- Create: `examples/nlm_gene/{job.yaml,slots.yaml,README.md,sample/}`
- Create: `examples/cafeteria_fcd/{job.yaml,slots.yaml,README.md,sample/}`
- Test: `tests/test_examples.py`

**Interfaces:** none — these are configuration and data, and they act as regression tests for the whole surface.

**What each dataset proves:**

| Dataset | What it exercises |
|---|---|
| CRAFT ChEBI | OWL loader, the simple one-field source case |
| NCBI Disease | user-supplied gold alias expansion (the old `expand_ctd_eval_ids`) |
| NLM-Gene | TSV loader, row filtering, multi-field disambiguation, **numeric IDs — the case that motivates opaque keys** |
| CafeteriaFCD | a second OWL domain, showing prompt slots are the only domain-specific part |

- [ ] **Step 1: Extract the sample slices from the paper repo**

Each `sample/` holds at most 200 target records and 50 source records. **No file over 1 MB.** For each dataset, write a one-off extraction script under `examples/<name>/extract.py` documenting exactly which rows were taken and from which paper-repo path, then delete the script's output paths from `.gitignore` exceptions if needed. Verify:

```bash
find examples -type f -size +1M    # must print nothing
du -sh examples                    # expect well under 20 MB
```

- [ ] **Step 2: Write `examples/nlm_gene/` — the one that matters most**

`examples/nlm_gene/job.yaml`:

```yaml
name: nlm_gene
templates:
  query: "{{ mention }}"
  context: "{{ context_left }} [{{ mention }}] {{ context_right }}"
  doc: "{{ label }} {{ synonyms | join(' ') }} {{ definition }} {{ organism }}"
  candidate: |
    ID: {{ id }}
    Symbol: {{ label }}
    Organism: {{ organism }}
    {% if synonyms %}Synonyms: {{ synonyms | join('; ') }}{% endif %}
    {% if definition %}Description: {{ definition }}{% endif %}

target:
  kind: tsv
  path: sample/gene_info_sample.tsv
  id_column: id
  multivalue_columns: [synonyms]

source:
  kind: jsonl
  path: sample/mentions.jsonl
  id_field: mention_id

retrievers:
  - kind: bm25
    name: bm25
    limit: 30

llm:
  kind: openai_compat
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  profile: openai

prompts:
  slots: slots.yaml

policy:
  max_attempts: 3
  accept_at: 0.6
  review_floor: 0.4
  verify_band: [0.6, 0.8]
  concurrency: 8

selector:
  max_candidates: 25
```

`examples/nlm_gene/slots.yaml`:

```yaml
entity_noun: gene or protein mention
target_noun: NCBI Gene record
domain_brief: gene nomenclature across model organisms, where the same symbol names
  different genes in different species
rubric:
  - score: 1.0
    name: Certain
    when: symbol matches exactly and the organism is confirmed by the context
    example: "human TP53 -> NCBIGene:7157 (Homo sapiens TP53)"
  - score: 0.9
    name: High
    when: an official symbol or alias matches and only one organism is plausible
    example: "Trp53 -> the mouse gene, since Trp53 is mouse-specific nomenclature"
  - score: 0.6
    name: Plausible
    when: the symbol matches but the organism is not stated anywhere in the context
    example: "bare 'p53' with no species mentioned"
  - score: 0.3
    name: Speculative
    when: only a description or partial name matches, not the symbol
    example: "'the tumour suppressor' -> TP53"
hard_rules:
  - A gene symbol in one organism is a DIFFERENT entity from the same symbol in another
    organism. Never match across species without contextual evidence.
  - Gene family members differing only by a trailing number are distinct entities.
disambiguation_steps: |
  ## Step 1: Determine the organism
  Scan the context for a species name, a cell line, a tissue, or an organism-specific
  nomenclature convention (all-caps for human, initial-caps for mouse).

  ## Step 2: Filter candidates by that organism
  Discard candidates from other organisms before comparing symbols.

  ## Step 3: Match the symbol
  Prefer the official symbol over an alias when both are available.
```

`examples/nlm_gene/README.md`:

```markdown
# NLM-Gene example

The dataset that motivates opaque candidate keys. NCBI gene IDs are **numeric**
(`NCBIGene:7157`), and the paper repo's ID resolver ends in a branch that treats a bare
integer as a 1-based candidate rank. A hallucinated ID not in the candidate set falls
through that branch and silently becomes a different, real mapping.

In xwalk the model answers with `C01`, never with an identifier, and any answer that is
not a key issued for that attempt resolves to `UNRESOLVED_OUTPUT` and routes to review.

Also exercises: TSV loading with multivalue columns, multi-field disambiguation
(organism plus symbol), and a `context` template that matters — a bare symbol is
ambiguous across species, so scoring against the retrieval query alone would evaluate
the wrong thing.

## Run

```bash
export OPENAI_API_KEY=...
xwalk index --job examples/nlm_gene/job.yaml --out runs/nlm_gene/index
xwalk match --job examples/nlm_gene/job.yaml --out runs/nlm_gene
xwalk eval  --run runs/nlm_gene --gold examples/nlm_gene/sample/gold.csv
```
```

- [ ] **Step 3: Write the remaining three examples**

`examples/chebi/` — `target.kind: owl`, `path: sample/chebi_sample.owl`, one-field source, slots copied from `examples/chemistry/slots.yaml`. README states it exercises the OWL loader and the simplest source shape.

`examples/ncbi_disease/` — `target.kind: tsv` over the CTD disease sample. Add `examples/ncbi_disease/gold_normalize.py`:

```python
"""Gold ID normalization and alias expansion for NCBI Disease / CTD.

This is what replaced the paper repo's `normalize_ctd_prediction` and
`expand_ctd_eval_ids` hooks. It lives here, in the example, because it is knowledge
about *this dataset* — not about matching.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

MESH_SHORT = re.compile(r"^[CD]\d+$")


def normalize(raw: str) -> str | None:
    rid = raw.strip()
    if not rid:
        return None
    upper = rid.upper()
    if upper.startswith("MESH:"):
        return "MESH:" + rid.split(":", 1)[1]
    if upper.startswith("OMIM:"):
        return "OMIM:" + rid.split(":", 1)[1]
    if MESH_SHORT.match(upper):
        return f"MESH:{upper}"
    return rid


@lru_cache(maxsize=None)
def _alias_map(tsv_path: str) -> dict[str, frozenset[str]]:
    aliases: dict[str, set[str]] = {}
    with Path(tsv_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            canonical = normalize(row[1])
            if not canonical:
                continue
            aliases.setdefault(canonical, {canonical}).add(canonical)
            for alt in row[2].split("|"):
                alt_id = normalize(alt)
                if alt_id:
                    aliases.setdefault(alt_id, set()).add(canonical)
    return {k: frozenset(v) for k, v in aliases.items()}


def make_expander(tsv_path: str):
    """Return an `expand` callable for load_gold_csv."""
    aliases = _alias_map(tsv_path)

    def expand(ids: frozenset[str]) -> frozenset[str]:
        out: set[str] = set(ids)
        for i in ids:
            out |= aliases.get(i, frozenset())
        return frozenset(out)

    return expand
```

`examples/cafeteria_fcd/` — a second OWL domain with a food-composition slots file. Its README states plainly: the only file that differs from `examples/chebi/` in kind is `slots.yaml`.

- [ ] **Step 4: Write the regression test**

Create `tests/test_examples.py`:

```python
from pathlib import Path

import pytest

from xwalk.config import load_job
from xwalk.prompts.contract import PromptSet, load_slots, validate_contract

EXAMPLES = sorted(p for p in Path("examples").iterdir() if (p / "job.yaml").exists())


def test_there_are_examples():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_job_file_loads(example):
    assert load_job(example / "job.yaml").name


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_templates_compile(example):
    load_job(example / "job.yaml").build_templates()


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_slots_satisfy_the_prompt_contract(example):
    job = load_job(example / "job.yaml")
    validate_contract(PromptSet.from_slots(load_slots(job.base_dir / job.prompts.slots)))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_sample_target_records_load(example):
    job = load_job(example / "job.yaml")
    if job.target.kind in ("owl",):
        pytest.importorskip("rdflib")
    records = list(job.build_target_records())
    assert records and all(r.id for r in records)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_sample_source_records_load(example):
    job = load_job(example / "job.yaml")
    assert list(job.build_source_records())


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_doc_template_renders_non_empty_text_for_every_target(example):
    """An empty rendered doc means that record can never be retrieved."""
    job = load_job(example / "job.yaml")
    if job.target.kind in ("owl",):
        pytest.importorskip("rdflib")
    templates = job.build_templates()
    empty = [r.id for r in job.build_target_records() if not templates.render_doc(r)]
    assert not empty, f"{len(empty)} target records render an empty doc: {empty[:5]}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_the_query_template_renders_non_empty_text_for_every_source(example):
    job = load_job(example / "job.yaml")
    templates = job.build_templates()
    empty = [r.id for r in job.build_source_records() if not templates.render_query(r)]
    assert not empty, f"{len(empty)} source records render an empty query: {empty[:5]}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_bm25_indexes_the_sample_and_finds_something(example, tmp_path):
    import asyncio

    from xwalk.retrieval.base import SearchRequest
    from xwalk.retrieval.bm25 import BM25Retriever

    job = load_job(example / "job.yaml")
    if job.target.kind in ("owl",):
        pytest.importorskip("rdflib")
    templates = job.build_templates()
    targets = list(job.build_target_records())
    retriever = BM25Retriever.build(
        targets, templates, tmp_path / "idx", exact_fields=("label", "synonyms")
    )
    source = next(iter(job.build_source_records()))
    hits = asyncio.run(
        retriever.search(SearchRequest(text=templates.render_query(source), limit=5))
    )
    assert isinstance(hits, list)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_no_example_file_exceeds_one_megabyte(example):
    large = [p for p in example.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000]
    assert not large, f"files over 1 MB: {large}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_no_job_file_contains_an_inline_api_key(example):
    text = (example / "job.yaml").read_text(encoding="utf-8")
    assert "api_key:" not in text


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_example_has_a_readme(example):
    assert (example / "README.md").exists()


def test_the_nlm_gene_example_uses_numeric_target_ids():
    """The case opaque keys exist for. If this stops holding, the example lost its point."""
    job = load_job(Path("examples/nlm_gene/job.yaml"))
    ids = [r.id for r in job.build_target_records()]
    assert any(i.split(":")[-1].isdigit() for i in ids)
```

- [ ] **Step 5: Run the tests, lint, commit**

```bash
python -m pytest tests/test_examples.py -v
find examples -type f -size +1M    # must print nothing
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
git add examples tests/test_examples.py
git commit -m "feat: port the four paper datasets as working examples"
```

---

## Task 9: CI, packaging, and the platform matrix

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/platforms.md`
- Modify: `pyproject.toml` (classifiers, urls), `README.md` (install matrix)
- Test: `tests/test_packaging.py`

**Interfaces:** none.

**Why the matrix is explicit:** the spec fixes a Tantivy support matrix and forbids a silent fallback, because a switch between BM25 engines changes ranking behaviour and quietly undermines reproducibility. CI must therefore *prove* the declared platforms work, and the docs must tell an unsupported-platform user what to do instead — explicitly, never automatically.

- [ ] **Step 1: Write the failing packaging test**

Create `tests/test_packaging.py`:

```python
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

HEAVY = ("torch", "sentence-transformers", "faiss", "rdflib", "sqlalchemy", "litellm",
         "numpy", "scikit-learn")


def _base_dependencies() -> list[str]:
    block = re.search(r"^dependencies = \[(.*?)^\]", PYPROJECT, re.S | re.M)
    assert block, "could not find the base dependencies block"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_no_heavy_dependency_is_in_the_base_install():
    base = " ".join(_base_dependencies()).lower()
    for name in HEAVY:
        assert name not in base, f"{name} must live behind an extra"


def test_every_declared_extra_exists():
    for extra in ("dense", "ontology", "sql", "litellm", "all", "dev"):
        assert f"{extra} = [" in PYPROJECT


def test_the_python_floor_is_declared():
    assert 'requires-python = ">=3.10"' in PYPROJECT


def test_the_cli_entry_point_is_declared():
    assert 'xwalk = "xwalk.cli.main:main"' in PYPROJECT


def test_the_package_never_calls_itself_crosswalk():
    hits = []
    for path in (ROOT / "src").rglob("*.py"):
        if "crosswalk" in path.read_text(encoding="utf-8").lower():
            hits.append(path)
    assert not hits, f"'crosswalk' appears in {hits}"


def test_prompt_skeletons_are_packaged():
    from xwalk.prompts.contract import BASE_DIR

    assert {p.name for p in BASE_DIR.glob("*.j2")} == {
        "select.j2", "score.j2", "verify.j2", "rewrite.j2"
    }


def test_py_typed_marker_is_present():
    assert (ROOT / "src" / "xwalk" / "py.typed").exists()


def test_the_cli_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "xwalk.cli.main", "--version"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0
    assert result.stdout.strip()


def test_the_platform_document_lists_the_supported_matrix():
    text = (ROOT / "docs" / "platforms.md").read_text(encoding="utf-8")
    for entry in ("manylinux", "aarch64", "macOS", "Windows"):
        assert entry in text


def test_the_platform_document_forbids_automatic_fallback():
    text = (ROOT / "docs" / "platforms.md").read_text(encoding="utf-8").lower()
    assert "never automatic" in text or "not automatic" in text
```

- [ ] **Step 2: Write `docs/platforms.md`**

```markdown
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

If `pip install tantivy` fails, you have two explicit options:

1. **Bring your own retriever.** Implement the `Retriever` protocol against whatever
   search backend you already run (Elasticsearch, OpenSearch, Postgres full-text,
   Vespa). It is three members: `name`, `fingerprint`, and `async search(SearchRequest)`.
   `TargetStore` stays separate, so the backend never needs your records in memory.

2. **Use dense retrieval only.** Install `xwalk[dense]` and configure a single dense
   retriever. Expect different behaviour on exact-string matches, which is precisely why
   this is a decision you make rather than one xwalk makes for you.

Whichever you choose, record it: the retriever's `fingerprint` appears in the run
manifest, so results produced on different retrieval stacks are never silently mixed.
```

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy

  test:
    name: ${{ matrix.os }} / py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-14, windows-latest]
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev]"
      - name: Tantivy install smoke test
        run: python -c "import tantivy; print('tantivy', getattr(tantivy, '__version__', 'ok'))"
      - name: Tantivy ranking smoke test
        run: python -m pytest tests/test_bm25.py -q
      - name: Base test suite
        run: python -m pytest -q -m "not integration and not dense and not ontology and not sql"
      - name: Assert the base install stays light
        shell: bash
        run: |
          pip list --format=freeze | tr 'A-Z' 'a-z' > installed.txt
          for pkg in torch sentence-transformers faiss-cpu rdflib sqlalchemy litellm; do
            if grep -q "^${pkg}==" installed.txt; then
              echo "::error::${pkg} leaked into the base install"; exit 1
            fi
          done

  aarch64:
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/test_bm25.py -q

  extras:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        extra: [ontology, sql]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev,${{ matrix.extra }}]"
      - run: python -m pytest -q -m "${{ matrix.extra }}"

  dense:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev,dense]"
      - run: python -m pytest -q -m dense

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build
      - run: python -m build
      - name: Assert the wheel ships the prompt skeletons
        run: |
          python - <<'PY'
          import glob, zipfile
          wheel = glob.glob("dist/*.whl")[0]
          names = zipfile.ZipFile(wheel).namelist()
          expected = {f"xwalk/prompts/base/{n}.j2" for n in
                      ("select", "score", "verify", "rewrite")}
          missing = expected - set(names)
          assert not missing, f"wheel is missing {missing}"
          assert "xwalk/py.typed" in names
          print("wheel contents OK")
          PY
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

- [ ] **Step 4: Extend `pyproject.toml` and the README**

Add to `[project]`:

```toml
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
keywords = ["record-linkage", "entity-matching", "ontology", "rag", "llm", "crosswalk"]

[project.urls]
Homepage = "https://github.com/<owner>/xwalk"
Documentation = "https://github.com/<owner>/xwalk#readme"
Issues = "https://github.com/<owner>/xwalk/issues"
```

Add to `README.md`, near the top:

````markdown
## Install

```bash
pip install xwalk                 # BM25 + any OpenAI-compatible endpoint. No torch.
pip install 'xwalk[dense]'        # + sentence-transformers, torch, faiss-cpu
pip install 'xwalk[ontology]'     # + rdflib, for OWL sources
pip install 'xwalk[sql]'          # + SQLAlchemy, for database sources
pip install 'xwalk[all]'
```

Matching two CSVs with an API-hosted model installs none of the heavy stack.
See [docs/platforms.md](docs/platforms.md) for the supported platform matrix — xwalk
never substitutes a different BM25 engine silently, because that would change ranking.
````

- [ ] **Step 5: Run the tests, lint, type-check, commit**

```bash
python -m pytest tests/test_packaging.py -v
python -m pytest -q -m "not integration and not dense and not ontology and not sql"
python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy
python -m build && ls dist/
git add .github docs/platforms.md pyproject.toml README.md tests/test_packaging.py
git commit -m "chore: CI matrix, packaging metadata, and the platform support document"
```

---

## Definition of done for Phase 3

- [ ] `python -m pytest -q -m "not integration and not dense and not ontology and not sql"` — green on a base install.
- [ ] `python -m pytest -q -m ontology` / `-m sql` / `-m dense` — green with each extra installed.
- [ ] `python -m ruff check src tests && python -m ruff format --check src tests && python -m mypy` — clean.
- [ ] `pip install .` in a fresh venv → `pip list` contains none of torch, sentence-transformers, faiss, rdflib, sqlalchemy, litellm, numpy.
- [ ] `xwalk --help` lists index, match, eval, compare, ablate, prompts, review.
- [ ] All four examples pass `tests/test_examples.py`; `find examples -type f -size +1M` prints nothing.
- [ ] The built wheel contains all four `.j2` skeletons and `py.typed`.
- [ ] CI is green on every matrix row, including aarch64.
- [ ] `grep -ri crosswalk src examples docs` — zero hits.

## Plan self-review notes

Spec coverage for Phase 3's two build-order items:

| Spec item | Task |
|---|---|
| Dense retrieval, `[dense]` extra | 2 |
| Ontology (OWL/OBO) source, `[ontology]` extra | 3 |
| SQL source, `[sql]` extra | 4 |
| `llm/litellm.py`, `[litellm]` extra | 4 |
| `evaluate/ablate.py` | 5 |
| `evaluate/compare.py` | 5 |
| `config.py` job spec | 6 |
| Full CLI surface | 7 |
| Port the four datasets as examples | 8 |
| Tantivy support matrix, CI install smoke tests, documented alternate | 9 |

Two deviations from the spec's module layout, both deliberate:

1. **`sources/ontology.py` also handles OBO, with no dependency.** The spec listed only
   `owl / obo via rdflib`. rdflib does not parse OBO format, and a sixty-line stanza
   parser is cheaper than converting the format first. OWL still uses rdflib behind the
   extra.
2. **`DenseRetriever` falls back to exact dot-product search when FAISS is absent.**
   This is a *speed* fallback with identical semantics on normalised vectors
   (`IndexFlatIP` is exact), not a ranking change — so it does not violate the
   no-silent-substitution rule that governs the BM25 engine. It exists so the dense tests
   run with a toy encoder and no torch.

One item deliberately **not** built: the paper's exact numbers are not reproduced, per
the spec's non-goals. `examples/` uses committed sample slices; full data stays in the
paper repo, which remains frozen as the archive.

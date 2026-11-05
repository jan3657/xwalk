"""The NER site's Neon backend and its loader (docs/ner-site/NEON.md), offline.

`FakeNeon` is an in-memory stand-in for the database that answers each tagged statement
of `xwalk.retrieval.postgres` with the same semantics as the SQL (exact hits first, then
token-overlap "BM25"; inner product max-pooled per record), so the loader and the site
can be tested end to end without a server, a key or a network.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ner_site import indexes as indexes_mod
from ner_site.indexes import BM25_NAME, DENSE_NAME, open_neon_indexes
from ner_site.mapping import MapStats, map_entities
from ner_site.ontologies import FULL_ROUTES, ROUTES
from ner_site.schema import Entity
from ner_site.service import Settings, SettingsError

from xwalk.fingerprint import hash_value
from xwalk.records import Record
from xwalk.retrieval.postgres import PgVectorRetriever, PostgresBM25Retriever, PostgresError, tag_of

TOOL = Path(__file__).resolve().parents[1] / "ner-site" / "tools" / "load_neon.py"
SAMPLE_SLUGS = tuple(ROUTES.values())
FAKE_URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"
FAKE_KEY = "sk-or-SENTINEL-not-a-real-key-0000"


class ToyEncoder:
    """26-d letter histogram, unit length; counts the texts it encoded."""

    name = "toy"
    dimension = 26

    def __init__(self, fail_after: int | None = None) -> None:
        self.texts = 0
        self.calls = 0
        self.fail_after = fail_after

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            from xwalk.retrieval.embeddings_api import EmbeddingAPIError

            raise EmbeddingAPIError("provider down")
        self.texts += len(texts)
        out = []
        for text in texts:
            v = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class FakeNeon:
    """Semantic stand-in for the tagged statements. Thread-safe enough for the loader's
    pool (dict writes under the GIL; the tests check totals, not interleavings)."""

    def __init__(self) -> None:
        self.meta: dict[tuple[str, str], dict[str, Any]] = {}
        self.terms: dict[tuple[str, str], dict[str, Any]] = {}
        self.vectors: dict[tuple[str, str, int], tuple[str, str, list[float]]] = {}
        self.dimension: int | None = None
        self.fail: dict[str, Exception] = {}
        self.tags: list[str] = []

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        tag = tag_of(sql) or ""
        self.tags.append(tag)
        if tag in self.fail:
            raise self.fail[tag]
        p = list(params)
        handler = getattr(self, f"_{tag}", None)
        return handler(sql, p) if handler else []

    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]:
        return [self.execute(s, p) for s, p in statements]

    # --- writes
    def _schema(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        found = re.search(r"halfvec\((\d+)\)", sql)
        if found and self.dimension is None:
            self.dimension = int(found.group(1))
        return []

    def _vector_dimension(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        return [] if self.dimension is None else [{"dimension": self.dimension}]

    def _meta_write(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        ont, kind, name, status, comps, fp, limit, rc, vc, dim = p
        self.meta[(ont, kind)] = {
            "ontology": ont, "kind": kind, "name": name, "status": status,
            "components": json.loads(comps), "fingerprint": fp, "default_limit": limit,
            "record_count": rc, "vector_count": vc, "dimension": dim,
        }  # fmt: skip
        return []

    def _terms_upsert(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        for row in json.loads(p[1]):
            self.terms[(p[0], row["id"])] = row
        return []

    def _vectors_present(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        ont, ids, ords, texts = p
        return [
            {"record_id": i, "ord": o}
            for i, o, t in zip(ids, ords, texts, strict=True)
            if (ont, i, o) in self.vectors and self.vectors[(ont, i, o)][1] == t
        ]

    def _vectors_upsert(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        ont, ids, ords, kinds, texts, literals = p
        for i, o, k, t, e in zip(ids, ords, kinds, texts, literals, strict=True):
            self.vectors[(ont, i, o)] = (k, t, [float(x) for x in e[1:-1].split(",")])
        return []

    def _terms_prune(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        keep = set(p[1])
        gone = [k for k in self.terms if k[0] == p[0] and k[1] not in keep]
        for k in gone:
            del self.terms[k]
        return [{"n": len(gone)}]

    def _vectors_prune(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        keep = set(zip(p[1], p[2], strict=True))
        gone = [k for k in self.vectors if k[0] == p[0] and (k[1], k[2]) not in keep]
        for k in gone:
            del self.vectors[k]
        return [{"n": len(gone)}]

    def _terms_drop(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        self.terms = {k: v for k, v in self.terms.items() if k[0] != p[0]}
        return []

    def _vectors_drop(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        self.vectors = {k: v for k, v in self.vectors.items() if k[0] != p[0]}
        return []

    def _meta_drop(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        self.meta = {k: v for k, v in self.meta.items() if k[0] != p[0]}
        return []

    def _sizes(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        return [{"database": 7_000_000.0, "terms": 0.0, "term_vectors": 0.0, "index_meta": 0.0}]

    # --- reads
    def _meta_read(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        wanted = p[0]
        return [dict(m) for (o, _), m in self.meta.items() if wanted is None or o in wanted]

    def _store_get(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        return [
            {"id": i, "fields": self.terms[(p[0], i)]["fields"]}
            for i in p[1]
            if (p[0], i) in self.terms
        ]

    def _bm25_search(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        ont, text, exact, limit = p
        query = set(_tokens(text))
        rows = []
        for (o, rid), term in self.terms.items():
            if o != ont:
                continue
            score = -float(len(query & set(_tokens(term["doc"]))))
            if exact and exact in term["exact"]:
                rows.append({"id": rid, "exact": 1, "score": score, "fields": term["fields"]})
            if score < 0:
                rows.append({"id": rid, "exact": 0, "score": score, "fields": term["fields"]})
        rows.sort(key=lambda r: (-r["exact"], r["score"], r["id"]))
        exact_rows = [r for r in rows if r["exact"]][:limit]
        text_rows = [r for r in rows if not r["exact"]][:limit]
        return exact_rows + text_rows

    def _pgvector_search(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        ont, literal, fetch, limit = p
        q = [float(x) for x in literal[1:-1].split(",")]
        scored = sorted(
            (-sum(a * b for a, b in zip(q, v, strict=True)), rid)
            for (o, rid, _), (_, _, v) in self.vectors.items()
            if o == ont
        )[:fetch]
        best: dict[str, float] = {}
        for d, rid in scored:
            best.setdefault(rid, d)
        ranked = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:limit]
        return [
            {"id": rid, "score": -d, "fields": self.terms.get((ont, rid), {}).get("fields")}
            for rid, d in ranked
        ]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("load_neon_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _env(**extra: str) -> dict[str, str]:
    return {"DATABASE_URL": FAKE_URL, "OPENROUTER_API_KEY": FAKE_KEY, **extra}


def _load(fake: FakeNeon, encoder: ToyEncoder, *args: str, lines: list[str] | None = None) -> int:
    out = lines if lines is not None else []
    code: int = _tool().main(
        ["--dimensions", "26", "--batch", "50", "--concurrency", "2", *args],
        env=_env(),
        executor_factory=lambda env: fake,
        encoder_factory=lambda key, dims: encoder,
        out=out.append,
    )
    return code


@pytest.fixture(autouse=True)
def _fresh_memo() -> Any:
    indexes_mod.reset()
    yield
    indexes_mod.reset()


@pytest.fixture(scope="module")
def loaded() -> FakeNeon:
    fake = FakeNeon()
    assert _load(fake, ToyEncoder()) == 0
    return fake


# --- loader ------------------------------------------------------------------------


def test_dry_run_counts_the_samples_and_touches_nothing() -> None:
    lines: list[str] = []
    calls: list[str] = []
    code = _tool().main(
        ["--dry-run"],
        env={},
        executor_factory=lambda env: calls.append("db") or FakeNeon(),  # type: ignore[func-returns-value]
        encoder_factory=lambda key, dims: calls.append("enc") or ToyEncoder(),  # type: ignore[func-returns-value]
        out=lines.append,
    )
    assert code == 0 and calls == []
    total = next(line for line in lines if line.startswith("total:"))
    assert "800 records, 3,132 vectors" in total
    assert "~82,930 tokens" in total  # 165,860 characters at 2.0 a token
    assert "halfvec(1024)" in total
    assert any("dry run" in line for line in lines)


def test_loading_the_samples_marks_every_index_ready(loaded: FakeNeon) -> None:
    for slug in SAMPLE_SLUGS:
        for kind in ("terms", "bm25", "dense"):
            meta = loaded.meta[(slug, kind)]
            assert meta["status"] == "ready"
            assert meta["fingerprint"] == hash_value(meta["components"])
    assert sum(1 for k in loaded.terms if k[0] in SAMPLE_SLUGS) == 800
    assert len(loaded.vectors) == 3132
    assert loaded.meta[("ncbi-gene-sample", "dense")]["name"] == DENSE_NAME
    assert loaded.meta[("ncbi-gene-sample", "dense")]["dimension"] == 26


def test_a_rerun_embeds_nothing(loaded: FakeNeon) -> None:
    encoder = ToyEncoder()
    lines: list[str] = []
    assert _load(loaded, encoder, lines=lines) == 0
    assert encoder.texts == 0
    assert any("already stored 3,132" in line or "already stored" in line for line in lines)


def test_an_interrupted_load_stays_incomplete_and_resumes_where_it_stopped() -> None:
    fake = FakeNeon()
    lines: list[str] = []
    assert _load(fake, ToyEncoder(fail_after=3), "--only", "ncbi-gene", lines=lines) == 1
    assert fake.meta[("ncbi-gene-sample", "dense")]["status"] == "loading"
    assert any("rerun to resume" in line for line in lines)
    stored = len(fake.vectors)
    assert 0 < stored < 1000
    second = ToyEncoder()
    assert _load(fake, second, "--only", "ncbi-gene") == 0
    total = len(fake.vectors)
    assert second.texts == total - stored
    assert fake.meta[("ncbi-gene-sample", "dense")]["status"] == "ready"


def test_a_load_prunes_records_that_left_the_source() -> None:
    fake = FakeNeon()
    assert _load(fake, ToyEncoder(), "--only", "foodon") == 0
    assert _load(fake, ToyEncoder(), "--only", "foodon", "--limit", "10") == 0
    assert sum(1 for k in fake.terms if k[0] == "foodon-sample") == 10
    assert fake.meta[("foodon-sample", "bm25")]["record_count"] == 10


def test_load_needs_the_database_url_and_the_key() -> None:
    lines: list[str] = []
    assert _tool().main([], env={"OPENROUTER_API_KEY": FAKE_KEY}, out=lines.append) == 2
    assert "DATABASE_URL is not set" in lines[-1]
    lines.clear()
    code = _tool().main(
        ["--only", "chebi"],
        env={"DATABASE_URL": FAKE_URL},
        executor_factory=lambda env: FakeNeon(),
        out=lines.append,
    )
    assert code == 2 and "OPENROUTER_API_KEY is not set" in lines[-1]
    assert not any("SENTINELpw" in line for line in lines)


def test_a_full_load_over_the_size_guard_is_refused_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    records = [Record(f"CHEBI:{i}", {"label": f"thing {i}", "synonyms": [], "definition": ""})
               for i in range(50)]  # fmt: skip
    monkeypatch.setattr(tool, "full_records", lambda kind, cache: iter(records))
    fake = FakeNeon()
    lines: list[str] = []
    code = tool.main(
        ["--source", "full", "--only", "chebi", "--max-bytes", "1000000"],
        env=_env(),
        executor_factory=lambda env: fake,
        encoder_factory=lambda key, dims: ToyEncoder(),
        out=lines.append,
    )
    assert code == 3
    assert "refusing" in lines[-1] and "--max-bytes" in lines[-1]
    assert fake.tags == ["sizes"]
    raised = tool.main(
        ["--source", "full", "--only", "chebi", "--dimensions", "26", "--max-bytes", "99000000"],
        env=_env(),
        executor_factory=lambda env: fake,
        encoder_factory=lambda key, dims: ToyEncoder(),
        out=lines.append,
    )
    assert raised == 0
    assert fake.meta[("chebi", "bm25")]["status"] == "ready"


def test_ctd_and_gene_parsers(tmp_path: Path) -> None:
    tool = _tool()
    ctd = tmp_path / "CTD_diseases.tsv.gz"
    with gzip.open(ctd, "wt", encoding="utf-8") as handle:
        handle.write("# comment\n# DiseaseName\tDiseaseID\t...\n")
        handle.write("Alkaptonuria\tMESH:D000474\t\tA rare disorder.\t\t\t\tAKU|Homogentisuria\t\n")
    (rec,) = list(tool.ctd_records(ctd))
    assert rec.id == "MESH:D000474"
    assert rec.fields == {
        "label": "Alkaptonuria",
        "synonyms": ["AKU", "Homogentisuria"],
        "definition": "A rare disorder.",
    }
    gene = tmp_path / "Homo_sapiens.gene_info.gz"
    with gzip.open(gene, "wt", encoding="utf-8") as handle:
        handle.write(
            "#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchr\tmap\tdescription\n"
        )
        handle.write("9606\t100\tADA\t-\tADA1\tx\t20\tq13\tadenosine deaminase\n")
        handle.write("10090\t11486\tAda\t-\t-\tx\t2\tH3\t-\n")
    first, second = list(tool.gene_records([gene]))
    assert first == Record(
        "NCBIGene:100",
        {
            "label": "ADA",
            "synonyms": ["ADA1"],
            "definition": "adenosine deaminase",
            "organism": "Homo sapiens",
            "tax_id": "9606",
        },
    )
    assert second.fields["synonyms"] == [] and second.fields["definition"] == ""
    assert second.fields["organism"] == "Mus musculus"


def test_estimates_scale_with_the_dimension() -> None:
    tool = _tool()
    from ner_site.ontologies import load_terms

    records = load_terms("ctd-disease-sample")
    small = tool.estimate("x", records, 512)
    large = tool.estimate("x", records, 1024)
    assert small.vectors == large.vectors
    per_dim = tool.VECTOR_BYTES_PER_DIM * 512
    assert large.vector_bytes - small.vector_bytes == pytest.approx(large.vectors * per_dim, 1e-3)
    assert tool.vector_bytes(1024) == pytest.approx(5677, abs=5)
    assert tool.vector_bytes(512) == pytest.approx(2615, abs=5)


# --- site --------------------------------------------------------------------------


def test_settings_read_the_backend_and_keep_the_database_url_secret() -> None:
    assert Settings.from_env({}).backend == "bundled"
    settings = Settings.from_env({"NER_SITE_BACKEND": "neon", "DATABASE_URL": FAKE_URL})
    assert settings.backend == "neon" and settings.database_url == FAKE_URL
    assert settings.neon_schema == "xwalk" and settings.neon_dimensions == 1024
    assert "SENTINELpw" not in repr(settings)
    assert FAKE_URL in settings.secrets() and "npg_SENTINELpw0123" in settings.secrets()
    view = settings.index_settings()
    assert view.backend == "neon" and "SENTINELpw" not in repr(view)
    for bad in (
        {"NER_SITE_BACKEND": "mongo"},
        {"NER_SITE_BACKEND": "neon", "NER_SITE_NEON_SCHEMA": "x;drop"},
        {"NER_SITE_BACKEND": "neon", "NER_SITE_NEON_DIMENSIONS": "0"},
        {"NER_SITE_NEON_CATALOG": "full"},
    ):
        with pytest.raises(SettingsError):
            Settings.from_env(bad)


def _open(fake: FakeNeon, **kw: Any) -> Any:
    defaults: dict[str, Any] = {
        "executor": fake,
        "api_key": None,
        "schema": "xwalk",
        "dimensions": 26,
        "catalog": "samples",
        "encoder_factory": ToyEncoder,
    }
    return open_neon_indexes(**{**defaults, **kw})


def test_neon_indexes_open_ok_with_the_site_names(loaded: FakeNeon) -> None:
    opened = _open(loaded)
    assert list(opened) == list(SAMPLE_SLUGS)
    for index in opened.values():
        assert index.status["bm25"].status == "ok" and index.status["dense"].status == "ok"
        bm25, dense = index.retrievers
        assert isinstance(bm25, PostgresBM25Retriever) and bm25.name == BM25_NAME
        assert isinstance(dense, PgVectorRetriever) and dense.name == DENSE_NAME


def test_neon_mapping_matches_through_both_retrievers(loaded: FakeNeon) -> None:
    opened = _open(loaded)
    entities = [
        Entity(id="e1", text="NAT2", start=0, end=4, type="gene"),
        Entity(id="e2", text="Alkaptonuria", start=5, end=17, type="disease"),
    ]
    stats = MapStats()
    results = asyncio.run(map_entities(entities, opened, top_k=3, stats=stats))
    assert results[0].matches[0].id == "NCBIGene:10"
    assert results[0].matches[0].label == "NAT2"
    assert set(results[0].matches[0].retrievers) == {BM25_NAME, DENSE_NAME}
    assert results[1].matches[0].id == "MESH:D000474"
    assert stats.embedding_requests == 1 and stats.dense_entities == ["e1", "e2"]


def test_neon_without_a_key_runs_bm25_only(loaded: FakeNeon) -> None:
    opened = _open(loaded, encoder_factory=None)
    for index in opened.values():
        assert index.status["bm25"].status == "ok"
        assert index.status["dense"].status == "unavailable"
        assert "OPENROUTER_API_KEY is not set" in (index.status["dense"].reason or "")


def test_neon_reasons_for_missing_incomplete_and_stale_indexes(loaded: FakeNeon) -> None:
    fake = FakeNeon()
    fake.meta = {k: dict(v) for k, v in loaded.meta.items()}
    fake.terms, fake.vectors = loaded.terms, loaded.vectors
    del fake.meta[("chebi-sample", "bm25")]
    fake.meta[("foodon-sample", "dense")]["status"] = "loading"
    stale = fake.meta[("ctd-disease-sample", "bm25")]
    stale["components"] = {**stale["components"], "doc_template": "{{ label }}"}
    stale["fingerprint"] = hash_value(stale["components"])
    opened = _open(fake)
    chebi = opened["chebi-sample"].status["bm25"]
    assert chebi.status == "unavailable"
    assert chebi.reason is not None and chebi.reason.startswith("no neon index for chebi-sample")
    assert "load_neon.py" in chebi.reason
    foodon = opened["foodon-sample"].status["dense"].reason or ""
    assert "neon index for foodon-sample is incomplete" in foodon
    ctd = opened["ctd-disease-sample"].status["bm25"].reason or ""
    assert "neon index for ctd-disease-sample is stale (differs in: doc_template)" in ctd
    assert opened["ncbi-gene-sample"].status["bm25"].status == "ok"


def test_neon_dimension_mismatch_is_stale_not_a_crash(loaded: FakeNeon) -> None:
    opened = _open(loaded, dimensions=1024, encoder_factory=None, api_key=FAKE_KEY)
    reason = opened["chebi-sample"].status["dense"].reason or ""
    assert "neon index for chebi-sample is stale" in reason and "encoder.dimension" in reason


def test_neon_unreachable_database_is_reported_and_not_memoised(
    loaded: FakeNeon, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = FakeNeon()
    broken.fail["meta_read"] = PostgresError("could not reach Neon at ep-x: ConnectError")
    made: list[str] = []

    def factory(url: str) -> Any:
        made.append(url)
        return broken if len(made) == 1 else loaded

    monkeypatch.setattr(indexes_mod, "make_executor", factory)
    settings = Settings.from_env(
        {"NER_SITE_BACKEND": "neon", "DATABASE_URL": FAKE_URL, "NER_SITE_NEON_DIMENSIONS": "26"}
    )
    first = indexes_mod.get_indexes(settings.index_settings())
    reason = first["chebi-sample"].status["bm25"].reason or ""
    assert "neon database could not be queried" in reason
    assert first["chebi-sample"].retrievers == []
    second = indexes_mod.get_indexes(settings.index_settings())
    assert second["chebi-sample"].status["bm25"].status == "ok"
    assert indexes_mod.get_indexes(settings.index_settings()) is second


def test_neon_without_a_database_url_says_so() -> None:
    settings = Settings.from_env({"NER_SITE_BACKEND": "neon"})
    opened = indexes_mod.get_indexes(settings.index_settings())
    for index in opened.values():
        for kind in ("bm25", "dense"):
            assert index.status[kind].reason == "DATABASE_URL is not set (NER_SITE_BACKEND=neon)"


def test_health_reports_the_backend_and_no_secret(
    loaded: FakeNeon, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ner_site import http as http_mod

    monkeypatch.setattr(indexes_mod, "make_executor", lambda url: loaded)
    settings = Settings.from_env(
        {
            "NER_SITE_BACKEND": "neon",
            "DATABASE_URL": FAKE_URL,
            "NER_SITE_NEON_DIMENSIONS": "26",
            "NER_SITE_CACHE_DIR": str(tmp_path),
        }
    )
    status, _, body = http_mod.handle("GET", "/api/health", {}, b"", settings=settings)
    data = json.loads(body)
    assert status == 200
    assert data["backend"] == "neon"
    assert data["retrieval"]["bm25"] == {"status": "ok"}
    assert data["retrieval"]["dense"]["status"] == "unavailable"  # no key
    assert [o["slug"] for o in data["ontologies"]] == list(SAMPLE_SLUGS)
    assert b"SENTINELpw" not in body
    bundled = Settings.from_env({"NER_SITE_CACHE_DIR": str(tmp_path)})
    _, _, plain = http_mod.handle("GET", "/api/health", {}, b"", settings=bundled)
    assert json.loads(plain)["backend"] == "bundled"


def test_the_full_catalog_routes_to_the_full_slugs_and_counts_from_the_database() -> None:
    tool = _tool()
    fake = FakeNeon()
    words = ("alpha", "beta", "gamma", "delta", "epsilon")
    records = [Record(f"CHEBI:{i}", {"label": w, "synonyms": [], "definition": ""})
               for i, w in enumerate(words)]  # fmt: skip
    tool_main = tool.main
    import unittest.mock as mock

    with mock.patch.object(tool, "full_records", lambda kind, cache: iter(records)):
        assert (
            tool_main(
                ["--source", "full", "--only", "chebi", "--dimensions", "26"],
                env=_env(),
                executor_factory=lambda env: fake,
                encoder_factory=lambda key, dims: ToyEncoder(),
                out=lambda line: None,
            )
            == 0
        )
    opened = _open(fake, catalog="full")
    assert list(opened) == list(FULL_ROUTES.values())
    assert opened["chebi"].status["bm25"].status == "ok"
    assert opened["chebi"].status["dense"].status == "ok"
    assert opened["foodon"].status["bm25"].reason is not None
    entity = Entity(id="e1", text="delta", start=0, end=5, type="chemical")
    (result,) = asyncio.run(map_entities([entity], opened, top_k=1, routes=FULL_ROUTES))
    assert result.ontology == "chebi" and result.matches[0].id == "CHEBI:3"


def test_the_full_catalog_does_not_extract_genes() -> None:
    """NCBI Gene is not loaded in full: genes are neither routed nor asked for."""
    full = Settings.from_env(
        {"NER_SITE_BACKEND": "neon", "NER_SITE_NEON_CATALOG": "full", "DATABASE_URL": FAKE_URL}
    )
    assert "gene" not in full.routes
    assert sorted(full.routes) == ["chemical", "disease", "food"]
    assert "- gene:" not in full.default_prompt
    for kept in ("- chemical:", "- disease:", "- food:", "- other:"):
        assert kept in full.default_prompt
    samples = Settings.from_env({"NER_SITE_BACKEND": "neon", "DATABASE_URL": FAKE_URL})
    assert "- gene:" in samples.default_prompt and samples.routes["gene"] == "ncbi-gene-sample"
    # The loader can still load the full gene catalog under its own slug.
    assert _tool().slug_for("ncbi-gene", "full") == "ncbi-gene"
    assert set(FULL_ROUTES.values()) == {"chebi", "ctd-disease", "foodon"}


# --- live --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL is not set")
def test_live_neon_sample_indexes_answer_bm25() -> None:
    """Reads the sample load in schema `xwalk` (made by `load_neon.py`); writes nothing."""
    executor = indexes_mod.make_executor(os.environ["DATABASE_URL"])
    opened = open_neon_indexes(
        executor=executor,
        api_key=None,
        schema="xwalk",
        dimensions=1024,
        catalog="samples",
    )
    for index in opened.values():
        assert index.status["bm25"].status == "ok", index.status["bm25"].reason
    entity = Entity(id="e1", text="NAT2", start=0, end=4, type="gene")
    (result,) = asyncio.run(map_entities([entity], opened, top_k=3))
    assert result.matches[0].id == "NCBIGene:10"


def test_drop_removes_the_selected_ontologies() -> None:
    fake = FakeNeon()
    assert _load(fake, ToyEncoder(), "--only", "foodon") == 0
    lines: list[str] = []
    assert _load(fake, ToyEncoder(), "--only", "foodon", "--drop", lines=lines) == 0
    assert any("foodon-sample: dropped" in line for line in lines)
    assert "terms_drop" in fake.tags and "meta_drop" in fake.tags and "vacuum" in fake.tags
    assert not any(k[0] == "foodon-sample" for k in [*fake.terms, *fake.meta, *fake.vectors])


def test_the_loader_builds_indexes_in_one_long_request_without_retries() -> None:
    """A retried CREATE INDEX restarts the build and cancels the last one: ChEBI's HNSW
    build outlived the 120 s request four times on 4 November 2026."""
    tool = _tool()
    build = tool._index_executor({"DATABASE_URL": FAKE_URL})
    assert build._max_retries == 0 and build._client.timeout.read >= 3600
    rows = tool._executor({"DATABASE_URL": FAKE_URL})
    assert rows._client.timeout.read == 120.0

"""The NER demo site's mapping step (docs/ner-site/PLAN.md section 5, WS-B).

Offline: BM25 is built in `tmp_path`; the dense index comes from a 26-d letter-histogram
encoder, so no key, network or model weights are needed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ner_site import indexes as indexes_mod
from ner_site.indexes import (
    BM25_NAME,
    DENSE_NAME,
    OntologyIndex,
    get_indexes,
    open_indexes,
    summarise_status,
)
from ner_site.mapping import OTHER_NOTE, MapStats, map_entities
from ner_site.ontologies import ROUTES, TEMPLATES, catalog, load_terms, route, term_url
from ner_site.schema import Entity, EntityResult, OntologySummary, RetrieverStatus

from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.dense import DenseRetriever
from xwalk.stores.memory import MemoryStore

SLUGS = ("chebi-sample", "ctd-disease-sample", "ncbi-gene-sample", "foodon-sample")
TOOL = Path(__file__).resolve().parents[1] / "ner-site" / "tools" / "build_indexes.py"


class ToyEncoder:
    """26-d letter histogram, unit length. Counts its calls."""

    name = "toy"
    dimension = 26

    def __init__(self) -> None:
        self.calls: list[tuple[bool, list[str]]] = []

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.calls.append((is_query, list(texts)))
        out = []
        for text in texts:
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / norm for v in vector])
        return out


class PrefixedToyEncoder(ToyEncoder):
    """Same name and dimension, different vector meaning: an index built with
    `ToyEncoder` is stale for it."""

    settings = {"query_prefix": "query: "}


class FailIfCalled:
    """Records every call: `pytest.fail` inside a gathered task would only become a
    degradation note, so the test asserts on `calls` instead."""

    name = "must-not-run"
    fingerprint = "x"
    default_limit = 5

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, request: Any) -> Sequence[RetrievalHit]:
        self.calls.append(request.text)
        pytest.fail(f"retriever called for {request.text!r}")


class Raising:
    name = "broken"
    fingerprint = "x"
    default_limit = 5

    async def search(self, request: Any) -> Sequence[RetrievalHit]:
        raise RuntimeError("backend down")


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ner_site_build_indexes", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entity(eid: str, text: str, etype: str) -> Entity:
    return Entity(id=eid, text=text, start=0, end=len(text), type=etype)  # type: ignore[arg-type]


def _map(entities: list[Entity], idx: Any, **kw: Any) -> list[EntityResult]:
    return asyncio.run(map_entities(entities, idx, **kw))


@pytest.fixture(autouse=True)
def _fresh_memo() -> Any:
    indexes_mod.reset()
    yield
    indexes_mod.reset()


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dense") / "indexes"
    _tool().build_all(out, ToyEncoder())
    return out


# --- ontologies --------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_load_terms_gives_200_records_per_sample(slug: str) -> None:
    terms = load_terms(slug)
    assert len(terms) == 200
    assert all(isinstance(t, Record) and "id" not in t.fields for t in terms)
    assert all(isinstance(t.fields.get("label"), str) for t in terms)


def test_load_terms_refuses_an_unknown_slug() -> None:
    with pytest.raises(KeyError):
        load_terms("../etc/passwd")


def test_catalog_lists_the_four_routed_samples() -> None:
    entries = [OntologySummary.model_validate(e) for e in catalog()]
    assert {e.slug for e in entries} == set(SLUGS)
    assert {e.type for e in entries} == {"chemical", "disease", "gene", "food"}
    assert all(e.count == len(load_terms(e.slug)) for e in entries)
    assert all(ROUTES[e.type] == e.slug for e in entries)


def test_route() -> None:
    assert route("chemical") == "chebi-sample"
    assert route("disease") == "ctd-disease-sample"
    assert route("gene") == "ncbi-gene-sample"
    assert route("food") == "foodon-sample"
    assert route("other") is None
    assert route("planet") is None


@pytest.mark.parametrize(
    ("record_id", "url"),
    [
        ("CHEBI:15365", "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365"),
        ("MESH:D003920", "https://meshb.nlm.nih.gov/record/ui?ui=D003920"),
        ("MESH:C000720973", "https://meshb.nlm.nih.gov/record/ui?ui=C000720973"),
        ("OMIM:601665", "https://omim.org/entry/601665"),
        ("NCBIGene:672", "https://www.ncbi.nlm.nih.gov/gene/672"),
        ("FOODON:00001013", "http://purl.obolibrary.org/obo/FOODON_00001013"),
        ("CHEBI:abc", None),
        ("MESH:D003920&x=1", None),
        ("NCBIGene:1/../2", None),
        ("HP:0000001", None),
        ("chebi:1", None),
        ("", None),
    ],
)
def test_term_url(record_id: str, url: str | None) -> None:
    assert term_url(record_id) == url


def test_every_sample_id_has_a_url() -> None:
    for slug in SLUGS:
        assert all(term_url(t.id) for t in load_terms(slug)), slug


def test_doc_template_never_renders_none_or_splits_a_string_synonym() -> None:
    sparse = Record("X:1", {"label": "aspirin", "synonyms": None, "definition": None})
    assert TEMPLATES.render_doc(sparse) == "aspirin"
    bare = Record("X:2", {"label": "aspirin"})
    assert TEMPLATES.render_doc(bare) == "aspirin"
    single = Record("X:3", {"label": "aspirin", "synonyms": "ASA"})
    assert TEMPLATES.render_doc(single) == "aspirin ASA"
    for slug in SLUGS:
        for term in load_terms(slug):
            assert "None" not in TEMPLATES.render_doc(term).split(), term.id


def test_gene_docs_carry_the_organism() -> None:
    gene = next(t for t in load_terms("ncbi-gene-sample") if t.fields["label"] == "A1BG")
    doc = TEMPLATES.render_doc(gene)
    assert doc.startswith("A1BG A1B ABG")
    assert "Homo sapiens" in doc


# --- indexes -----------------------------------------------------------------------


def test_open_indexes_builds_bm25_and_reports_dense_unavailable_without_a_dir(
    tmp_path: Path,
) -> None:
    opened = open_indexes(cache_dir=tmp_path, dense_dir=None, api_key=None)
    assert list(opened) == list(SLUGS)
    for slug, index in opened.items():
        assert isinstance(index, OntologyIndex) and index.slug == slug
        assert [r.name for r in index.retrievers] == [BM25_NAME]
        assert index.status["bm25"] == RetrieverStatus(status="ok")
        assert index.status["dense"].status == "unavailable"
        assert "NER_SITE_DENSE_DIR" in (index.status["dense"].reason or "")
        assert (tmp_path / slug / "bm25" / "xwalk_meta.json").exists()
        assert len(index.store) == 200


def test_missing_dense_index_names_the_path_and_the_tool(tmp_path: Path) -> None:
    opened = open_indexes(cache_dir=tmp_path / "c", dense_dir=tmp_path / "indexes", api_key="k")
    reason = opened["chebi-sample"].status["dense"].reason or ""
    assert reason == ("no index at indexes/chebi-sample/dense; run ner-site/tools/build_indexes.py")
    assert str(tmp_path) not in reason


def test_missing_key_makes_dense_unavailable(tmp_path: Path, dense_dir: Path) -> None:
    opened = open_indexes(cache_dir=tmp_path, dense_dir=dense_dir, api_key=None)
    status = opened["foodon-sample"].status["dense"]
    assert status.status == "unavailable"
    assert "OPENROUTER_API_KEY" in (status.reason or "")
    assert [r.name for r in opened["foodon-sample"].retrievers] == [BM25_NAME]


def test_a_built_dense_index_opens_ok_under_its_stable_name(
    tmp_path: Path, dense_dir: Path
) -> None:
    opened = open_indexes(
        cache_dir=tmp_path, dense_dir=dense_dir, api_key=None, encoder_factory=ToyEncoder
    )
    for index in opened.values():
        assert index.status["dense"] == RetrieverStatus(status="ok")
        assert [r.name for r in index.retrievers] == [BM25_NAME, DENSE_NAME]
        assert isinstance(index.retrievers[1], DenseRetriever)
    assert DENSE_NAME == "dense-qwen3"


def test_the_dense_name_is_fixed_on_open_whatever_the_index_recorded(tmp_path: Path) -> None:
    out = tmp_path / "indexes"
    for slug in SLUGS:  # built with xwalk's default name, "dense:toy"
        DenseRetriever.build(
            load_terms(slug),
            TEMPLATES,
            out / slug / "dense",
            ToyEncoder(),
            name_fields=("label", "synonyms"),
        )
    opened = open_indexes(
        cache_dir=tmp_path / "c", dense_dir=out, api_key=None, encoder_factory=ToyEncoder
    )
    assert [r.name for r in opened["chebi-sample"].retrievers] == [BM25_NAME, "dense-qwen3"]


def test_a_stale_dense_index_is_unavailable_with_a_reason(tmp_path: Path, dense_dir: Path) -> None:
    opened = open_indexes(
        cache_dir=tmp_path, dense_dir=dense_dir, api_key="k", encoder_factory=PrefixedToyEncoder
    )
    status = opened["chebi-sample"].status["dense"]
    assert status.status == "unavailable"
    reason = status.reason or ""
    assert "stale" in reason and "encoder.query_prefix" in reason
    assert "build_indexes.py" in reason


def test_a_corrupt_dense_index_is_unavailable_not_an_exception(
    tmp_path: Path, dense_dir: Path
) -> None:
    broken = tmp_path / "indexes"
    (broken / "chebi-sample" / "dense").mkdir(parents=True)
    (broken / "chebi-sample" / "dense" / "xwalk_meta.json").write_text("{not json")
    opened = open_indexes(
        cache_dir=tmp_path / "c", dense_dir=broken, api_key="k", encoder_factory=ToyEncoder
    )
    status = opened["chebi-sample"].status["dense"]
    assert status.status == "unavailable"
    assert "could not be opened" in (status.reason or "")


def test_a_warm_bm25_index_is_reused_and_a_mismatched_one_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_indexes(cache_dir=tmp_path, dense_dir=None, api_key=None)
    builds: list[str] = []
    real_build = BM25Retriever.build.__func__  # type: ignore[attr-defined]

    def counting_build(cls: type, records: Any, templates: Any, index_dir: Any, **kw: Any) -> Any:
        builds.append(Path(index_dir).parent.name)
        return real_build(cls, records, templates, index_dir, **kw)

    monkeypatch.setattr(BM25Retriever, "build", classmethod(counting_build))
    open_indexes(cache_dir=tmp_path, dense_dir=None, api_key=None)
    assert builds == []

    meta_path = tmp_path / "foodon-sample" / "bm25" / "xwalk_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["components"]["records"] = "something else"
    meta_path.write_text(json.dumps(meta))
    reopened = open_indexes(cache_dir=tmp_path, dense_dir=None, api_key=None)
    assert builds == ["foodon-sample"]
    assert reopened["foodon-sample"].status["bm25"].status == "ok"


def test_get_indexes_builds_once_when_two_threads_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[str] = []
    real_build = BM25Retriever.build.__func__  # type: ignore[attr-defined]

    def slow_build(cls: type, records: Any, templates: Any, index_dir: Any, **kw: Any) -> Any:
        builds.append(Path(index_dir).parent.name)
        return real_build(cls, records, templates, index_dir, **kw)

    monkeypatch.setattr(BM25Retriever, "build", classmethod(slow_build))
    settings = SimpleNamespace(cache_dir=tmp_path, dense_dir=None, api_key=None)
    results: list[Any] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        results.append(get_indexes(settings))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 2 and results[0] is results[1]
    assert sorted(builds) == sorted(SLUGS)


def test_summarise_status() -> None:
    ok = RetrieverStatus(status="ok")

    def index(slug: str, dense: RetrieverStatus) -> OntologyIndex:
        return OntologyIndex(slug, MemoryStore.from_source([]), [], {"bm25": ok, "dense": dense})

    all_ok = {s: index(s, ok) for s in SLUGS}
    assert summarise_status(all_ok) == {"bm25": ok, "dense": ok}

    gone = RetrieverStatus(status="unavailable", reason="no index at x")
    mixed = {s: index(s, gone if s == "ncbi-gene-sample" else ok) for s in SLUGS}
    summary = summarise_status(mixed)
    assert summary["bm25"] == ok
    assert summary["dense"] == RetrieverStatus(
        status="unavailable", reason="ncbi-gene-sample: no index at x"
    )
    assert summarise_status({}) == {
        "bm25": RetrieverStatus(status="unavailable", reason="no ontology indexes are loaded"),
        "dense": RetrieverStatus(status="unavailable", reason="no ontology indexes are loaded"),
    }


# --- mapping -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fused(tmp_path_factory: pytest.TempPathFactory, dense_dir: Path) -> Any:
    return open_indexes(
        cache_dir=tmp_path_factory.mktemp("bm25"),
        dense_dir=dense_dir,
        api_key=None,
        encoder_factory=ToyEncoder,
    )


def test_fused_matches_name_both_retrievers_and_keep_store_ids(fused: Any) -> None:
    stats = MapStats()
    entities = [
        _entity("e1", "voriconazole", "chemical"),
        _entity("e2", "fever", "other"),
        _entity("e3", "A1BG", "gene"),
    ]
    results = _map(entities, fused, top_k=3, stats=stats)
    assert [r.id for r in results] == ["e1", "e2", "e3"]
    first = results[0]
    assert first.ontology == "chebi-sample" and first.notes == []
    top = first.matches[0]
    assert top.label == "voriconazole"
    assert top.retrievers[BM25_NAME] == 1 and DENSE_NAME in top.retrievers
    assert top.url == f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={top.id}"
    assert top.id in fused["chebi-sample"].store
    assert results[2].matches[0].id == "NCBIGene:1"
    assert stats.embedding_requests == 2
    assert stats.warnings == []


def test_other_gets_no_ontology_and_no_retriever_call() -> None:
    retriever = FailIfCalled()
    trap = {
        slug: OntologyIndex(
            slug,
            MemoryStore.from_source([]),
            [retriever],
            {"bm25": RetrieverStatus(status="ok"), "dense": RetrieverStatus(status="ok")},
        )
        for slug in SLUGS
    }
    stats = MapStats()
    results = _map([_entity("e1", "fever", "other")], trap, top_k=5, stats=stats)
    assert retriever.calls == []
    assert results == [
        EntityResult(
            id="e1",
            text="fever",
            start=0,
            end=5,
            type="other",
            ontology=None,
            matches=[],
            notes=[OTHER_NOTE],
        )
    ]
    assert OTHER_NOTE == "no ontology for type 'other'"
    assert stats.embedding_requests == 0


def test_a_raising_retriever_yields_notes_not_an_exception(fused: Any) -> None:
    chebi = fused["chebi-sample"]
    degraded = {
        **fused,
        "chebi-sample": OntologyIndex(
            "chebi-sample", chebi.store, [chebi.retrievers[0], Raising()], chebi.status
        ),
        "foodon-sample": OntologyIndex(
            "foodon-sample", fused["foodon-sample"].store, [Raising()], chebi.status
        ),
    }
    stats = MapStats()
    results = _map(
        [_entity("e1", "voriconazole", "chemical"), _entity("e2", "cheese", "food")],
        degraded,
        top_k=5,
        stats=stats,
    )
    assert results[0].matches[0].label == "voriconazole"
    assert results[0].notes == ["broken: backend down"]
    assert results[1].matches == []
    assert results[1].notes == ["broken: backend down", "every retriever failed; no matches"]
    assert stats.warnings == ["broken: backend down (e1, e2)"]


def test_top_k_is_honoured(fused: Any) -> None:
    for k in (1, 4):
        results = _map([_entity("e1", "acid", "chemical")], fused, top_k=k)
        assert len(results[0].matches) == k
    with pytest.raises(ValueError):
        _map([_entity("e1", "acid", "chemical")], fused, top_k=0)


def test_a_missing_ontology_index_is_a_note(fused: Any) -> None:
    partial = {k: v for k, v in fused.items() if k != "foodon-sample"}
    results = _map([_entity("e1", "cheese", "food")], partial, top_k=5)
    assert results[0].ontology == "foodon-sample"
    assert results[0].matches == []
    assert results[0].notes == ["ontology 'foodon-sample' is not loaded"]


# --- tools/build_indexes.py --------------------------------------------------------


def test_build_indexes_with_the_histogram_encoder(tmp_path: Path) -> None:
    tool = _tool()
    encoder = ToyEncoder()
    reports = tool.build_all(tmp_path / "out", encoder, only=["foodon-sample"])
    assert [r.slug for r in reports] == ["foodon-sample"]
    report = reports[0]
    assert report.records == 200 and report.vectors > 200
    assert report.requests is None and report.cost is None  # the toy encoder reports none
    meta = json.loads(
        (tmp_path / "out" / "foodon-sample" / "dense" / "xwalk_meta.json").read_text()
    )
    assert meta["name"] == DENSE_NAME
    assert meta["components"]["name_fields"] == ["label", "synonyms"]
    assert not (tmp_path / "out" / "chebi-sample").exists()
    assert "unknown" in tool.format_report(report)


def test_build_indexes_refuses_to_run_without_a_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()
    code = tool.main(["--out", str(tmp_path)], env={})
    assert code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_build_indexes_dry_run_makes_no_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()

    def no_encoder(api_key: str) -> Any:
        pytest.fail("dry run created an encoder")

    code = tool.main(["--out", str(tmp_path), "--dry-run"], env={}, encoder_factory=no_encoder)
    assert code == 0
    out = capsys.readouterr().out
    for slug in SLUGS:
        assert slug in out
    assert "200 records" in out
    assert not any(tmp_path.iterdir())


def test_build_indexes_main_uses_the_key_and_prints_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()
    seen: list[str] = []

    def factory(api_key: str) -> ToyEncoder:
        seen.append(api_key)
        return ToyEncoder()

    code = tool.main(
        ["--out", str(tmp_path), "--only", "ncbi-gene-sample"],
        env={"OPENROUTER_API_KEY": "sk-test"},
        encoder_factory=factory,
    )
    assert code == 0 and seen == ["sk-test"]
    out = capsys.readouterr().out
    assert "ncbi-gene-sample" in out and "unknown" in out
    assert "sk-test" not in out
    assert (tmp_path / "ncbi-gene-sample" / "dense" / "vectors.f32").exists()


def test_build_indexes_rejects_an_unknown_slug(tmp_path: Path) -> None:
    tool = _tool()
    with pytest.raises(SystemExit) as exc:
        tool.main(["--out", str(tmp_path), "--only", "nope"], env={"OPENROUTER_API_KEY": "k"})
    assert exc.value.code == 2

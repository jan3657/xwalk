"""`xwalk ui`, bring-your-own-data: uploads, file inspection, the ontology library,
projects, quick mapping and session credentials.

Offline throughout: matching runs use the scripted stand-in, clustering the lexical one,
downloads come from a local HTTP server, and no endpoint is contacted.
"""

from __future__ import annotations

import gzip
import http.server
import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from xwalk import ops
from xwalk.prompts.contract import load_slots
from xwalk.ui import files, projects
from xwalk.ui.api import ApiError, Download, Workspace, dispatch
from xwalk.ui.library import CATALOG, Library, LibraryError, combined_terms, download

PRODUCTS = (
    "﻿Product Code;Product Name;Category\n"
    "P1;Cheddar cheese;dairy\n"
    "P2;cream cheese spread;dairy\n"
    "P3;Wilson disease;clinical\n"
)
CATALOGUE = (
    "code\tname\taliases\nC-01\tCheese, cheddar\tcheddar|mature cheddar\nC-02\tCream cheese\t\n"
)
TOY_OBO = (
    "format-version: 1.2\n\n"
    '[Term]\nid: TOY:0001\nname: dark chocolate\nsynonym: "plain chocolate" EXACT []\n'
    'def: "Chocolate with no milk." []\n\n'
    "[Term]\nid: TOY:0002\nname: milk chocolate\n\n"
    "[Term]\nid: TOY:0003\nname: old chocolate\nis_obsolete: true\n"
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root, max_calls_cap=20, max_upload_mb=1)


def _upload(ws: Workspace, name: str, text: str | bytes) -> str:
    body = text.encode("utf-8") if isinstance(text, str) else text
    result = dispatch(
        ws, "POST", "/api/upload", {"name": name, "_body": io.BytesIO(body), "_length": len(body)}
    )
    assert isinstance(result, dict)
    return str(result["path"])


def _call(ws: Workspace, method: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    result = dispatch(ws, method, path, params)
    assert isinstance(result, dict)
    return result


# --- files ------------------------------------------------------------------------------


def test_safe_filenames_and_unique_paths(tmp_path: Path) -> None:
    assert files.safe_filename("../../etc/passwd") == "passwd"
    assert files.safe_filename("my foods (v2).csv") == "my_foods_v2_.csv"
    assert files.safe_filename("..") == "upload"
    assert files.safe_filename(".hidden.csv") == "hidden.csv"
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "b.obo.gz").write_text("x")
    assert files.unique_path(tmp_path, "a.csv").name == "a-2.csv"
    assert files.unique_path(tmp_path, "b.obo.gz").name == "b-2.obo.gz"


def test_formats_come_from_the_extension() -> None:
    assert files.detect_format(Path("x.TSV")) == "tsv"
    assert files.detect_format(Path("x.obo.gz")) == "obo"
    assert files.detect_format(Path("x.rdf")) == "owl"
    with pytest.raises(files.FileProblem, match="save the sheet as CSV"):
        files.detect_format(Path("book.xlsx"))
    with pytest.raises(files.FileProblem, match="unknown file type"):
        files.detect_format(Path("notes.docx"))


def test_inspecting_a_semicolon_csv_with_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "products.csv"
    path.write_text(PRODUCTS, encoding="utf-8")
    seen = files.inspect_file(path, "products.csv")
    assert seen.columns == ["Product Code", "Product Name", "Category"]
    assert seen.rows == 3 and not seen.rows_capped
    assert seen.suggest["id_column"] == "Product Code"
    assert seen.suggest["text_column"] == "Product Name"


def test_inspecting_suggests_synonyms_and_their_separator(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.tsv"
    path.write_text(CATALOGUE, encoding="utf-8")
    suggest = files.inspect_file(path, "c.tsv").suggest
    assert suggest["id_column"] == "code" and suggest["label_column"] == "name"
    assert suggest["synonyms_column"] == "aliases" and suggest["synonyms_sep"] == "|"


def test_inspecting_an_ontology(tmp_path: Path) -> None:
    path = tmp_path / "toy.obo"
    path.write_text(TOY_OBO, encoding="utf-8")
    seen = files.inspect_file(path, "toy.obo")
    assert seen.format == "obo" and seen.rows == 2  # the obsolete term is left out
    assert seen.sample[0]["label"] == "dark chocolate"


def test_a_column_id_with_repeats_is_not_suggested(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,name\n1,a\n1,b\n", encoding="utf-8")
    assert files.inspect_file(path, "x.csv").suggest["id_column"] is None


def test_source_records_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "products.csv"
    path.write_text(PRODUCTS, encoding="utf-8")
    records = list(
        files.source_records(
            path,
            "csv",
            id_column="Product Code",
            text_column="Product Name",
            context_columns=["Category"],
        )
    )
    assert [r.id for r in records] == ["P1", "P2", "P3"]
    assert records[0].fields == {"text": "Cheddar cheese", "context": "dairy"}
    numbered = list(files.source_records(path, "csv", id_column=None, text_column="Category"))
    assert [r.id for r in numbered] == ["r1", "r2", "r3"]
    assert numbered[0].fields["product_name"] == "Cheddar cheese"


def test_source_records_refuse_repeats_and_unknown_columns(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,name\n1,a\n1,b\n", encoding="utf-8")
    with pytest.raises(files.FileProblem, match="appears twice"):
        list(files.source_records(path, "csv", id_column="id", text_column="name"))
    with pytest.raises(files.FileProblem, match="no column 'label'"):
        list(files.source_records(path, "csv", id_column=None, text_column="label"))


def test_target_records_split_synonyms(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.tsv"
    path.write_text(CATALOGUE, encoding="utf-8")
    records = list(
        files.target_records(
            path, "tsv", id_column="code", label_column="name", synonyms_column="aliases"
        )
    )
    assert records[0].fields["synonyms"] == ["cheddar", "mature cheddar"]
    assert records[1].fields["synonyms"] == []


def test_text_and_non_utf8_files(tmp_path: Path) -> None:
    text = tmp_path / "terms.txt"
    text.write_text("cheddar\n\n  brie \n", encoding="utf-8")
    assert [
        r.fields["text"]
        for r in files.source_records(text, "txt", id_column=None, text_column=None)
    ] == ["cheddar", "brie"]
    latin = tmp_path / "latin.csv"
    latin.write_bytes("name\ncaf\xe9\n".encode("latin-1"))
    with pytest.raises(files.FileProblem, match="UTF-8"):
        list(files.iter_rows(latin, "csv"))


def test_pasted_terms() -> None:
    records = files.terms_from_text("glucose\n\nglucose\nsugar\tin the coffee\n")
    assert [(r.id, r.fields["text"], r.fields["context"]) for r in records] == [
        ("q1", "glucose", ""),
        ("q2", "sugar", "in the coffee"),
    ]


# --- the library --------------------------------------------------------------------------


def test_the_built_in_samples_are_parsed_and_labelled(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    builtin = library.builtin()
    assert {o.slug for o in builtin} == {
        "chebi-sample",
        "foodon-sample",
        "ctd-disease-sample",
        "ncbi-gene-sample",
    }
    for entry in builtin:
        terms = list(library.terms(entry.slug))
        assert len(terms) == entry.count == 200 and entry.sample
        assert all(t.fields["label"] for t in terms)
        assert isinstance(terms[0].fields["synonyms"], list)


def test_importing_and_deleting(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    path = tmp_path / "toy.obo"
    path.write_text(TOY_OBO, encoding="utf-8")
    entry = library.add("Toy chocolate", files.read_ontology(path, "obo"), source="toy.obo")
    assert entry.slug == "toy-chocolate" and entry.count == 2 and entry.kind == "imported"
    again = library.add("Toy chocolate", files.read_ontology(path, "obo"), source="toy.obo")
    assert again.slug == "toy-chocolate-2"
    library.delete("toy-chocolate-2")
    assert [o.slug for o in library.imported()] == ["toy-chocolate"]
    with pytest.raises(LibraryError, match="shared"):
        library.delete("chebi-sample")
    with pytest.raises(LibraryError):
        library.get("nope")


def test_an_import_with_repeated_ids_leaves_nothing(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    from xwalk.records import Record

    records = [Record("a", {"label": "x"}), Record("a", {"label": "y"})]
    with pytest.raises(files.FileProblem, match="twice"):
        library.add("bad", records, source="test")
    assert library.imported() == []
    assert not (tmp_path / "lib" / "bad").exists()


def test_combined_terms_tag_their_ontology_and_report_overlaps(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    records, warnings = combined_terms(library, ["foodon-sample", "foodon-sample"])
    assert len(records) == 200 and warnings == []
    assert records[0].fields["ontology"] == "FoodOn (sample)"
    path = tmp_path / "copy.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": r.id, "label": "copy"}) + "\n"
            for r in list(library.terms("foodon-sample"))[:3]
        ),
        encoding="utf-8",
    )
    library.add("copy", files.read_jsonl(path), source="copy")
    merged, warnings = combined_terms(library, ["foodon-sample", "copy"])
    assert len(merged) == 200 and "3 term id(s)" in warnings[0]
    with pytest.raises(LibraryError):
        combined_terms(library, [])


def test_the_catalog_names_obo_purls() -> None:
    assert len({c["id"] for c in CATALOG}) == len(CATALOG)
    assert all(c["url"].startswith("http://purl.obolibrary.org/obo/") for c in CATALOG)


@pytest.fixture
def served(tmp_path: Path) -> Iterator[str]:
    """A local HTTP server for `tmp_path/www` (plain and gzipped ontologies)."""
    www = tmp_path / "www"
    www.mkdir()
    (www / "toy.obo").write_bytes(TOY_OBO.encode("utf-8"))
    (www / "toy-gz.obo").write_bytes(gzip.compress(TOY_OBO.encode("utf-8")))

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(www), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_download_plain_and_gzipped(served: str, tmp_path: Path) -> None:
    seen: list[int] = []
    plain = download(f"{served}/toy.obo", tmp_path / "a.obo", progress=lambda n, _: seen.append(n))
    assert plain.read_text(encoding="utf-8") == TOY_OBO and seen[-1] == len(TOY_OBO)
    unzipped = download(f"{served}/toy-gz.obo", tmp_path / "b.obo")
    assert unzipped.read_text(encoding="utf-8") == TOY_OBO
    with pytest.raises(files.FileProblem, match="404"):
        download(f"{served}/missing.obo", tmp_path / "c.obo")
    with pytest.raises(files.FileProblem, match="download limit"):
        download(f"{served}/toy.obo", tmp_path / "d.obo", max_bytes=10)
    assert not (tmp_path / "d.obo").exists() and not (tmp_path / "d.obo.download").exists()
    with pytest.raises(files.FileProblem, match="http"):
        download("file:///etc/passwd", tmp_path / "e.obo")


# --- projects -----------------------------------------------------------------------------


def test_the_generated_slots_and_llm_block() -> None:
    slots = projects.slots_document({"entity_noun": "food name"})
    assert slots["entity_noun"] == "food name"
    assert slots["target_noun"] == projects.DEFAULT_DESCRIPTIONS["target_noun"]
    block = projects.llm_block({"preset": "ollama", "model": "qwen3"})
    assert block["model"] == "qwen3" and "api_key_env" not in block
    openai = projects.llm_block(None)
    assert openai["api_key_env"] == "OPENAI_API_KEY" and "api_key" not in openai
    with pytest.raises(files.FileProblem):
        projects.llm_block({"preset": "custom"})
    with pytest.raises(files.FileProblem):
        projects.llm_block({"preset": "mystery"})


def test_a_map_project_is_a_valid_job_that_runs(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    summary = projects.create_map_project(
        tmp_path,
        library,
        name="My foods",
        sources=files.terms_from_text("cheddar cheese\nhepatolenticular degeneration"),
        targets=projects.TargetChoice(libraries=["foodon-sample", "ctd-disease-sample"]),
        descriptions={"entity_noun": "product name"},
        policy={"accept_at": 0.8, "review_floor": 0.3},
    )
    job = Path(summary["job_path"])
    assert job == tmp_path / "projects" / "my-foods" / "job.yaml"
    assert summary["sources"] == 2 and summary["targets"] == 400
    assert "api_key:" not in job.read_text(encoding="utf-8")
    assert load_slots(job.parent / "slots.yaml").entity_noun == "product name"
    check = ops.validate(job, check_credentials=False)
    assert check.exit_code == ops.EXIT_OK, [e.message for e in check.errors]

    from xwalk.ui import offline

    result = ops.run(job, job.parent / "runs" / "a", llm=offline.match_llm())
    assert result.counts["matched"] == 2

    lookup = projects.lookup(
        job, projects.project_sources(job.parent), index_dir=tmp_path / "idx", top_k=2
    )
    first = lookup["rows"][1]["candidates"][0]
    assert first["id"] == "MESH:D006527" and first["exact"] is True
    assert first["ontology"] == "CTD diseases / MeSH (sample)"


def test_projects_refuse_bad_thresholds_and_empty_inputs(tmp_path: Path) -> None:
    library = Library(tmp_path / "lib")
    choice = projects.TargetChoice(libraries=["foodon-sample"])
    with pytest.raises(files.FileProblem, match="thresholds"):
        projects.create_map_project(
            tmp_path,
            library,
            name="x",
            sources=files.terms_from_text("a"),
            targets=choice,
            policy={"accept_at": 0.3, "review_floor": 0.5},
        )
    with pytest.raises(files.FileProblem, match="no source records"):
        projects.create_map_project(tmp_path, library, name="x", sources=[], targets=choice)
    assert not (tmp_path / "projects" / "x").exists()


def test_a_cluster_project_validates_and_runs(tmp_path: Path) -> None:
    summary = projects.create_cluster_project(
        tmp_path,
        name="dupes",
        records=files.terms_from_text("dark chocolate\nchocolate, dark\nquinoa"),
        relation="Two records are equivalent when they name the same food.",
    )
    job = Path(summary["job_path"])
    assert ops.validate(job, check_credentials=False).exit_code == ops.EXIT_OK
    from xwalk.ui import offline

    result = ops.cluster(job, job.parent / "runs" / "a", llm=offline.cluster_llm())
    assert result.counts["clusters"] == 2


# --- the API ----------------------------------------------------------------------------


def test_uploads_are_stored_named_safely_and_bounded(ws: Workspace) -> None:
    path = _upload(ws, "../My Foods.csv", PRODUCTS)
    assert path == "uploads/My_Foods.csv"
    assert _upload(ws, "My Foods.csv", PRODUCTS) == "uploads/My_Foods-2.csv"
    listing = _call(ws, "GET", "/api/files", {})
    assert {f["name"] for f in listing["files"]} == {"My_Foods.csv", "My_Foods-2.csv"}
    with pytest.raises(ApiError) as caught:
        _upload(ws, "big.csv", b"x" * (1024 * 1024 + 1))
    assert caught.value.status == 413
    with pytest.raises(ApiError) as caught:
        _upload(ws, "sheet.xlsx", b"PK")
    assert caught.value.status == 422
    with pytest.raises(ApiError) as caught:
        dispatch(
            ws, "POST", "/api/upload", {"name": "x.csv", "_body": io.BytesIO(b"ab"), "_length": 5}
        )
    assert caught.value.code == "bad_request"
    assert not any(p.name.endswith(".partial") for p in (ws.root / "uploads").iterdir())


def test_only_uploads_can_be_deleted(ws: Workspace) -> None:
    path = _upload(ws, "a.csv", PRODUCTS)
    (ws.root / "keep.csv").write_text("x", encoding="utf-8")
    with pytest.raises(ApiError) as caught:
        _call(ws, "POST", "/api/files/delete", {"path": "keep.csv"})
    assert caught.value.status == 403
    _call(ws, "POST", "/api/files/delete", {"path": path})
    assert not (ws.root / path).exists() and (ws.root / "keep.csv").exists()


def test_library_endpoints(ws: Workspace) -> None:
    obo = _upload(ws, "toy.obo", TOY_OBO)
    seen = _call(ws, "GET", "/api/inspect", {"path": obo})
    assert seen["format"] == "obo"
    imported = _call(ws, "POST", "/api/library/import", {"path": obo, "name": "Toy"})
    assert imported["ontology"]["slug"] == "toy" and imported["ontology"]["count"] == 2
    slugs = [o["slug"] for o in _call(ws, "GET", "/api/library", {})["ontologies"]]
    assert slugs[-1] == "toy" and "chebi-sample" in slugs

    found = _call(
        ws, "POST", "/api/library/search", {"libraries": ["toy"], "query": "plain chocolate"}
    )
    assert found["candidates"][0]["id"] == "TOY:0001" and found["candidates"][0]["exact"]
    page = _call(
        ws, "GET", "/api/library/terms", {"slug": "foodon-sample", "filter": "cheese", "limit": 2}
    )
    assert len(page["rows"]) == 2 and page["total"] > 2 and page["next_offset"] == 2

    tsv = _upload(ws, "catalogue.tsv", CATALOGUE)
    table = _call(
        ws,
        "POST",
        "/api/library/import",
        {"path": tsv, "id_column": "code", "label_column": "name", "synonyms_column": "aliases"},
    )
    assert table["ontology"]["count"] == 2
    _call(ws, "POST", "/api/library/delete", {"slug": "toy"})
    with pytest.raises(ApiError) as caught:
        _call(ws, "POST", "/api/library/delete", {"slug": "chebi-sample"})
    assert caught.value.status == 422
    with pytest.raises(ApiError) as caught:
        _call(ws, "POST", "/api/library/import", {"path": tsv, "label_column": "nope"})
    assert caught.value.status == 422 and "no column" in caught.value.message


def test_library_download_runs_as_a_task(ws: Workspace, served: str) -> None:
    view = _call(
        ws, "POST", "/api/library/download", {"url": f"{served}/toy-gz.obo", "name": "Toy remote"}
    )
    done = ws.tasks.wait(view["id"])
    assert done.result is not None and done.result["status"] == "ok", done.result
    assert done.result["counts"] == {"terms": 2}
    assert "toy-remote" in [o["slug"] for o in _call(ws, "GET", "/api/library", {})["ontologies"]]
    assert [e["status"] for e in done.events][-1] == "imported"
    failed = _call(ws, "POST", "/api/library/download", {"url": f"{served}/missing.obo"})
    result = ws.tasks.wait(failed["id"]).result
    assert result is not None and result["errors"][0]["code"] == "download_failed"
    with pytest.raises(ApiError):
        _call(ws, "POST", "/api/library/download", {"url": f"{served}/notes.docx"})
    with pytest.raises(ApiError):
        _call(ws, "POST", "/api/library/download", {"catalog_id": "nope"})
    catalog = _call(ws, "GET", "/api/library/catalog", {})["catalog"]
    assert len(catalog) == len(CATALOG) and not any(c["imported"] for c in catalog)


def test_a_project_from_an_upload_onto_ontologies(ws: Workspace) -> None:
    path = _upload(ws, "products.csv", PRODUCTS)
    project = _call(
        ws,
        "POST",
        "/api/projects",
        {
            "mode": "map",
            "name": "products",
            "source": {
                "path": path,
                "id_column": "Product Code",
                "text_column": "Product Name",
                "context_columns": ["Category"],
            },
            "target": {"libraries": ["foodon-sample", "ctd-disease-sample"]},
        },
    )
    assert project["job_path"] == "projects/products/job.yaml"
    assert project["sources"] == 3 and project["targets"] == 400
    task = ws.start_task({"job": project["job_path"], "out": project["suggested_out"]})
    assert ws.tasks.wait(task["id"]).result["counts"]["matched"] == 3  # type: ignore[index]

    page = _call(ws, "GET", "/api/mapping", {"dir": project["suggested_out"]})
    wilson = next(r for r in page["data"]["rows"] if r["source_id"] == "P3")
    assert wilson["source_text"] == "Wilson disease"
    assert wilson["target_label"] == "Hepatolenticular Degeneration"
    assert wilson["target_ontology"] == "CTD diseases / MeSH (sample)"
    assert (
        _call(ws, "GET", "/api/run", {"dir": project["suggested_out"]})["answered_by"] == "offline"
    )

    sheet = dispatch(ws, "GET", "/api/mapping.csv", {"dir": project["suggested_out"]})
    assert isinstance(sheet, Download)
    assert sheet.body.decode("utf-8").splitlines()[0].startswith("source_id,source_text,status")
    job = dispatch(ws, "GET", "/api/job-file", {"path": project["job_path"]})
    assert isinstance(job, Download) and b"retrievers:" in job.body


def test_a_project_onto_an_uploaded_file(ws: Workspace) -> None:
    source = _upload(ws, "products.csv", PRODUCTS)
    target = _upload(ws, "catalogue.tsv", CATALOGUE)
    project = _call(
        ws,
        "POST",
        "/api/projects",
        {
            "source": {"path": source, "text_column": "Product Name"},
            "target": {
                "path": target,
                "id_column": "code",
                "label_column": "name",
                "synonyms_column": "aliases",
                "synonyms_sep": "|",
            },
        },
    )
    assert project["targets"] == 2
    targets = [
        json.loads(line)
        for line in (ws.root / project["dir"] / "targets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert targets[0]["id"] == "C-01" and targets[0]["label"] == "Cheese, cheddar"
    assert targets[0]["synonyms"] == ["cheddar", "mature cheddar"]
    assert targets[1]["synonyms"] == []


def test_project_requests_are_checked(ws: Workspace) -> None:
    path = _upload(ws, "products.csv", PRODUCTS)
    bad = [
        {"mode": "merge", "source": {"text": "a"}},
        {"source": {"text": "   "}, "target": {"libraries": ["foodon-sample"]}},
        {"source": {"text": "a"}, "target": {"libraries": ["nope"]}},
        {
            "source": {"path": path, "text_column": "Nope"},
            "target": {"libraries": ["foodon-sample"]},
        },
        {"source": {"text": "a"}, "target": "foodon"},
        {"source": {"path": "../x.csv"}, "target": {"libraries": ["foodon-sample"]}},
    ]
    for params in bad:
        with pytest.raises(ApiError):
            _call(ws, "POST", "/api/projects", params)


def test_a_cluster_project_from_pasted_text(ws: Workspace) -> None:
    project = _call(
        ws,
        "POST",
        "/api/projects",
        {"mode": "cluster", "source": {"text": "dark chocolate\nchocolate, dark\nquinoa"}},
    )
    task = ws.start_task({"job": project["job_path"], "out": project["suggested_out"]})
    result = ws.tasks.wait(task["id"]).result
    assert result is not None and result["counts"]["clusters"] == 2


def test_quick_map_candidates_and_runs(ws: Workspace) -> None:
    found = _call(
        ws,
        "POST",
        "/api/quickmap",
        {
            "terms": "cheddar\nhepatolenticular degeneration\nunobtainium",
            "target": {"libraries": ["foodon-sample", "ctd-disease-sample"]},
            "top_k": 3,
        },
    )
    rows = {r["text"]: r for r in found["rows"]}
    assert rows["hepatolenticular degeneration"]["candidates"][0]["exact"] is True
    assert rows["unobtainium"]["candidates"] == []
    assert len(rows["cheddar"]["candidates"]) <= 3
    # The lookup index is built once and reused.
    again = _call(
        ws,
        "POST",
        "/api/quickmap",
        {"terms": "brie", "target": {"libraries": ["foodon-sample", "ctd-disease-sample"]}},
    )
    assert found["indexes"] == {"bm25": "built"} and again["indexes"] == {"bm25": "opened"}

    started = _call(
        ws,
        "POST",
        "/api/quickmap",
        {"terms": "cheddar\nbrie", "target": {"libraries": ["foodon-sample"]}, "mode": "offline"},
    )
    result = ws.tasks.wait(started["task"]["id"]).result
    assert result is not None and result["counts"]["total"] == 2
    with pytest.raises(ApiError) as caught:
        _call(
            ws,
            "POST",
            "/api/quickmap",
            {"terms": "x", "target": {"libraries": ["foodon-sample"]}, "mode": "endpoint"},
        )
    assert caught.value.code == "missing_parameter"


def test_reviewing_through_a_spreadsheet(ws: Workspace) -> None:
    from xwalk.llm.fake import FakeLLM

    project = _call(
        ws,
        "POST",
        "/api/projects",
        {
            "source": {"text": "cheddar cheese\ncream cheese"},
            "target": {"libraries": ["foodon-sample"]},
        },
    )
    run_dir = ws.root / project["suggested_out"]
    reply = json.dumps(
        {"chosen_key": "C01", "confidence_score": 0.5, "decision": "support", "queries": []}
    )
    ops.run(ws.root / project["job_path"], run_dir, llm=FakeLLM(handler=lambda _: reply))
    sheet = dispatch(ws, "GET", "/api/review-sheet", {"dir": project["suggested_out"]})
    assert isinstance(sheet, Download)
    lines = sheet.body.decode("utf-8").splitlines()
    header, first = lines[0].split(","), lines[1].split(",")
    filled = dict(zip(header, first, strict=True))
    filled.update(decision="accept", reviewer="jan")
    upload = _upload(
        ws, "reviewed.csv", ",".join(header) + "\n" + ",".join(filled[h] for h in header)
    )
    applied = _call(
        ws,
        "POST",
        "/api/review-upload",
        {"dir": project["suggested_out"], "path": upload, "job": project["job_path"]},
    )
    assert applied["status"] == "ok" and applied["counts"]["applied"] == 1


def test_session_credentials(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    with pytest.raises(ApiError):
        _call(ws, "POST", "/api/credentials", {"env": "PATH", "value": "x"})
    with pytest.raises(ApiError) as caught:
        _call(ws, "POST", "/api/credentials", {"env": "ANTHROPIC_API_KEY", "value": ""})
    assert caught.value.code == "not_from_page"
    monkeypatch.setenv("OPENAI_API_KEY", "")  # registered for cleanup by monkeypatch
    _call(ws, "POST", "/api/credentials", {"env": "OPENAI_API_KEY", "value": " sk-test "})
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    listing = {c["env"]: c for c in _call(ws, "GET", "/api/credentials", {})["credentials"]}
    assert listing["OPENAI_API_KEY"] == {"env": "OPENAI_API_KEY", "set": True, "from_page": True}
    assert listing["ANTHROPIC_API_KEY"]["from_page"] is False
    assert "sk-test" not in json.dumps(listing)
    _call(ws, "POST", "/api/credentials", {"env": "OPENAI_API_KEY", "value": ""})
    assert "OPENAI_API_KEY" not in os.environ
    assert not any(
        "sk-test" in p.read_text(errors="ignore") for p in ws.root.rglob("*") if p.is_file()
    )


# --- HTTP ---------------------------------------------------------------------------------


def test_uploading_over_http(tmp_path: Path) -> None:
    import urllib.error
    import urllib.request

    from xwalk.ui import make_server

    root = tmp_path / "ws"
    root.mkdir()
    server = make_server(str(root), port=0, token="t", max_upload_mb=1)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:

        def post(name: str, body: bytes, token: str = "t") -> tuple[int, dict[str, Any]]:
            request = urllib.request.Request(
                f"{server.url}api/upload?name={name}",
                data=body,
                method="POST",
                headers={"X-Xwalk-Token": token, "Content-Type": "application/octet-stream"},
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        status, body = post("terms.txt", b"cheddar\nbrie\n")
        assert status == 200 and body == {
            "path": "uploads/terms.txt",
            "name": "terms.txt",
            "size": 13,
        }
        assert (root / "uploads" / "terms.txt").read_bytes() == b"cheddar\nbrie\n"
        assert post("x.txt", b"a", token="wrong")[0] == 403
        status, body = post("big.txt", b"x" * (1024 * 1024 + 5))
        assert status == 413 and body["error"]["code"] == "too_large"
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{server.url}api/job-file?path=uploads/terms.txt", timeout=10)
        assert caught.value.code == 403
    finally:
        server.shutdown()
        server.server_close()

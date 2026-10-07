"""The UI's endpoints for people who bring data: uploads, the ontology library, project
building, quick mapping and session credentials. Routed by `xwalk.ui.api.dispatch`.

Every handler takes the `Workspace` and the request parameters, like those in
`xwalk.ui.api`. File problems the person can fix (a missing column, a duplicate id, an
unreadable file) answer `422` with a message that says what to change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

from xwalk import ops
from xwalk._extras import MissingExtra
from xwalk.records import Record
from xwalk.ui import files, projects
from xwalk.ui.api import (
    UI_DIR,
    ApiError,
    Download,
    Params,
    Response,
    Task,
    Workspace,
    _envelope,
    _int,
    _str,
)
from xwalk.ui.library import CATALOG, Library, LibraryError, download

LOOKUPS_DIR = "lookups"
MAX_TERMS_TEXT = 2 * 1024 * 1024
MAX_PAGE = 200
_CHUNK = 1024 * 1024
MAX_DRAIN = 64 * 1024 * 1024
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*_(API_KEY|KEY|TOKEN)$")


def library(ws: Workspace) -> Library:
    return Library(ws.root / UI_DIR / "library")


def _problem(exc: BaseException) -> ApiError:
    return ApiError(422, "file_problem", str(exc))


def _guarded(work: Callable[[], Response]) -> Response:
    """Run `work`, turning file and library problems into 422 answers."""
    try:
        return work()
    except (files.FileProblem, LibraryError, MissingExtra) as exc:
        raise _problem(exc) from None
    except ValueError as exc:  # a loader's own message about the file (bad OBO, ...)
        raise _problem(exc) from None


# --- uploads ---------------------------------------------------------------------------


def upload(ws: Workspace, params: Params) -> Response:
    """Store a raw request body (`_body`, `_length`) as `uploads/<name>`."""
    name = files.safe_filename(_str(params, "name"))
    body: IO[bytes] | None = params.get("_body")
    length = params.get("_length")
    if body is None or not isinstance(length, int):
        raise ApiError(400, "bad_request", "send the file as the request body")
    if length > ws.max_upload_bytes:
        if length <= ws.max_upload_bytes + MAX_DRAIN:
            # Read and drop the body, so the client receives this answer instead of a
            # broken connection. A far larger body is not worth reading: it is cut off.
            while length > 0:
                length -= len(body.read(min(_CHUNK, length)) or b"x" * length)
        raise ApiError(
            413, "too_large", f"the file is larger than {ws.max_upload_bytes // 1024**2} MB"
        )
    try:
        files.detect_format(Path(name))
    except files.FileProblem as exc:
        raise _problem(exc) from None
    directory = ws.root / files.UPLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    dest = files.unique_path(directory, name)
    partial = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:6]}.partial")
    remaining = length
    try:
        with partial.open("wb") as handle:
            while remaining > 0:
                chunk = body.read(min(_CHUNK, remaining))
                if not chunk:
                    raise ApiError(400, "bad_request", "the upload ended early")
                handle.write(chunk)
                remaining -= len(chunk)
        partial.replace(dest)
    finally:
        if partial.exists():
            partial.unlink()
    return {"path": ws.rel(dest), "name": dest.name, "size": length}


def _upload_kind(path: Path) -> str | None:
    try:
        return files.detect_format(path)
    except files.FileProblem:
        return None


def list_files(ws: Workspace, params: Params) -> Response:
    """Uploaded files, newest first."""
    directory = ws.root / files.UPLOAD_DIR
    found: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_file() and not path.name.startswith("."):
                stat = path.stat()
                found.append(
                    {
                        "path": ws.rel(path),
                        "name": path.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "format": _upload_kind(path),
                    }
                )
    found.sort(key=lambda f: f["modified"], reverse=True)
    return {"files": found, "max_upload_mb": ws.max_upload_bytes // 1024**2}


def delete_file(ws: Workspace, params: Params) -> Response:
    path = ws.path(_str(params, "path"), must_exist=True)
    if path.parent != ws.root / files.UPLOAD_DIR or not path.is_file():
        raise ApiError(403, "not_an_upload", "only files in uploads/ can be deleted here")
    path.unlink()
    return {"deleted": ws.rel(path)}


def inspect(ws: Workspace, params: Params) -> Response:
    path = ws.path(_str(params, "path"), must_exist=True)
    if not path.is_file():
        raise ApiError(400, "not_a_file", f"{ws.rel(path)} is not a file")
    return _guarded(lambda: files.inspect_file(path, ws.rel(path)).to_dict())


# --- reading choices -----------------------------------------------------------------


def _source_choice(ws: Workspace, spec: Any) -> tuple[list[Record], dict[str, Any]]:
    """Source records from `{text}` (pasted lines) or `{path, columns...}`."""
    if not isinstance(spec, Mapping):
        raise ApiError(400, "invalid_parameter", "source must be an object")
    if spec.get("text") is not None:
        text = str(spec["text"])
        if len(text) > MAX_TERMS_TEXT:
            raise ApiError(413, "too_large", "paste at most 2 MB of terms; upload a file instead")
        records = files.terms_from_text(text)
        if not records:
            raise ApiError(400, "missing_parameter", "type at least one term")
        return records, {"source": "pasted text", "terms": len(records)}
    path = ws.path(_str(spec, "path"), must_exist=True)

    def read() -> list[Record]:
        fmt = files.detect_format(path)
        return list(
            files.source_records(
                path,
                fmt,
                id_column=spec.get("id_column") or None,
                text_column=spec.get("text_column") or None,
                context_columns=[c for c in spec.get("context_columns") or [] if c],
            )
        )

    result = _guarded(lambda: {"records": read()})
    assert isinstance(result, dict)
    return result["records"], {"source": ws.rel(path), "columns": _columns(spec)}


def _columns(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spec.items() if k.endswith(("_column", "_columns", "_sep")) and v}


def _target_choice(ws: Workspace, spec: Any) -> tuple[projects.TargetChoice, dict[str, Any]]:
    """Targets from `{libraries: [...]}` or an uploaded file `{path, columns...}`."""
    if not isinstance(spec, Mapping):
        raise ApiError(400, "invalid_parameter", "target must be an object")
    slugs = spec.get("libraries")
    if slugs:
        if not isinstance(slugs, list) or not all(isinstance(s, str) for s in slugs):
            raise ApiError(400, "invalid_parameter", "libraries must be a list of names")
        lib = library(ws)
        names = []
        for slug in slugs:
            try:
                names.append(lib.get(slug).name)
            except LibraryError as exc:
                raise ApiError(404, "not_found", str(exc)) from None
        return projects.TargetChoice(libraries=list(slugs), description=", ".join(names)), {
            "target": {"libraries": slugs}
        }
    path = ws.path(_str(spec, "path"), must_exist=True)

    def read() -> list[Record]:
        fmt = files.detect_format(path)
        return list(
            files.target_records(
                path,
                fmt,
                id_column=spec.get("id_column") or None,
                label_column=spec.get("label_column") or None,
                synonyms_column=spec.get("synonyms_column") or None,
                synonyms_sep=spec.get("synonyms_sep") or "|",
                definition_column=spec.get("definition_column") or None,
            )
        )

    result = _guarded(lambda: {"records": read()})
    assert isinstance(result, dict)
    choice = projects.TargetChoice(records=result["records"], description=ws.rel(path))
    return choice, {"target": {"path": ws.rel(path), "columns": _columns(spec)}}


def _project_view(ws: Workspace, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{k: v for k, v in summary.items() if not isinstance(v, Path)},
        "dir": ws.rel(summary["dir"]),
        "job_path": ws.rel(summary["job_path"]),
        "suggested_out": ws.rel(Path(summary["dir"]) / "runs" / "ui-1"),
    }


# --- the ontology library ----------------------------------------------------------------


def library_list(ws: Workspace, params: Params) -> Response:
    return {"ontologies": [o.to_dict() for o in library(ws).all()]}


def library_catalog(ws: Workspace, params: Params) -> Response:
    imported = {o.source for o in library(ws).imported()}
    return {
        "catalog": [{**entry, "imported": entry["url"] in imported} for entry in CATALOG],
        "allow_download": True,
    }


def library_import(ws: Workspace, params: Params) -> Response:
    """Parse an uploaded or workspace file into a new library entry."""
    path = ws.path(_str(params, "path"), must_exist=True)
    name = _str(params, "name", required=False) or path.stem
    description = _str(params, "description", required=False)

    def work() -> Response:
        fmt = files.detect_format(path)
        records = files.target_records(
            path,
            fmt,
            id_column=params.get("id_column") or None,
            label_column=params.get("label_column") or None,
            synonyms_column=params.get("synonyms_column") or None,
            synonyms_sep=params.get("synonyms_sep") or "|",
            definition_column=params.get("definition_column") or None,
        )
        entry = library(ws).add(name, records, source=ws.rel(path), description=description)
        return {"ontology": entry.to_dict()}

    return _guarded(work)


def library_download(ws: Workspace, params: Params) -> Response:
    """Download an ontology (a catalog id or a URL) and import it, as a background task."""
    catalog_id = _str(params, "catalog_id", required=False)
    entry = next((c for c in CATALOG if c["id"] == catalog_id), None) if catalog_id else None
    if catalog_id and entry is None:
        raise ApiError(404, "not_found", f"no catalog entry {catalog_id!r}")
    url = entry["url"] if entry else _str(params, "url")
    name = _str(params, "name", required=False) or (entry["name"] if entry else Path(url).stem)
    try:
        fmt = files.detect_format(Path(url.split("?")[0]))
    except files.FileProblem:
        fmt = _str(params, "format", required=False) or ""
    if fmt not in ("obo", "owl", "csv", "tsv", "jsonl"):
        raise ApiError(
            400, "invalid_parameter", "the URL must end in .obo, .owl, .csv, .tsv or .jsonl"
        )
    task = Task(
        id=uuid.uuid4().hex[:12],
        kind="download",
        job=url,
        out=name,
        out_path=str(ws.root / UI_DIR / "downloads"),
        model="none",
        max_calls=None,
        limit=None,
    )

    def fetch_and_import() -> ops.OpResult:
        scratch = ws.root / UI_DIR / "downloads"
        dest = scratch / f"{uuid.uuid4().hex[:8]}.{fmt}"
        last = [0.0]

        def progress(received: int, total: int | None) -> None:
            now = time.monotonic()
            if now - last[0] > 0.5 or (total and received >= total):
                last[0] = now
                task.add_event({"status": "downloading", "bytes": received, "total": total})

        try:
            download(url, dest, progress=progress)
            task.add_event({"status": "parsing"})
            records = files.target_records(
                dest, fmt, id_column="id", label_column="label", synonyms_column="synonyms"
            )
            stored = library(ws).add(
                name,
                records,
                source=url,
                description=entry["domain"] if entry else "",
                homepage=url,
            )
        finally:
            if dest.exists():
                dest.unlink()
        task.add_event({"status": "imported", "count": stored.count})
        return ops.OpResult(
            "library download", counts={"terms": stored.count}, data={"ontology": stored.to_dict()}
        )

    async def work() -> ops.OpResult:
        import asyncio

        try:
            return await asyncio.to_thread(fetch_and_import)
        except (files.FileProblem, LibraryError, MissingExtra, ValueError) as exc:
            return ops.OpResult(
                "library download",
                exit_code=ops.EXIT_RUNTIME,
                errors=[ops.OpMessage("download_failed", str(exc))],
            )

    return ws.tasks.start(task, "library download", work).view()


def library_delete(ws: Workspace, params: Params) -> Response:
    slug = _str(params, "slug")

    def work() -> Response:
        library(ws).delete(slug)
        return {"deleted": slug}

    return _guarded(work)


def library_terms(ws: Workspace, params: Params) -> Response:
    """One page of an entry's terms, optionally filtered by a substring of label/synonyms."""
    slug = _str(params, "slug")
    offset = _int(params, "offset", default=0, low=0) or 0
    limit = _int(params, "limit", default=50, low=1, high=MAX_PAGE) or 50
    needle = _str(params, "filter", required=False).lower()

    def work() -> Response:
        lib = library(ws)
        entry = lib.get(slug)
        rows: list[dict[str, Any]] = []
        total = 0
        for record in lib.terms(slug):
            if needle:
                names = [record.id, record.fields.get("label", "")]
                names += list(record.fields.get("synonyms") or [])
                if not any(needle in str(n).lower() for n in names):
                    continue
            if offset <= total < offset + limit:
                rows.append({"id": record.id, **record.fields})
            total += 1
        end = offset + len(rows)
        return {
            "ontology": entry.to_dict(),
            "rows": rows,
            "total": total,
            "offset": offset,
            "limit": limit,
            "next_offset": end if end < total else None,
        }

    return _guarded(work)


# --- lookups (retrieval only) ----------------------------------------------------------


def _lookup_project(ws: Workspace, choice: projects.TargetChoice, key: Mapping[str, Any]) -> Path:
    """A hidden project holding one target set, reused across lookups (job path)."""
    lib = library(ws)
    stamp: dict[str, Any] = dict(key)
    if choice.libraries:
        stamp["entries"] = [
            (o.slug, o.count, o.created) for o in lib.all() if o.slug in choice.libraries
        ]
    digest = hashlib.sha256(json.dumps(stamp, sort_keys=True, default=str).encode()).hexdigest()
    directory = ws.root / UI_DIR / LOOKUPS_DIR / digest[:16]
    job = directory / "projects"
    found = sorted(job.glob("*/job.yaml")) if job.is_dir() else []
    if found:
        return found[0]
    summary = projects.create_map_project(
        directory,
        lib,
        name="lookup",
        sources=[Record("q1", {"text": "lookup", "context": ""})],
        targets=choice,
        model={"preset": "openai"},
    )
    return Path(summary["job_path"])


def _lookup(ws: Workspace, target: Any, queries: list[Record], top_k: int) -> dict[str, Any]:
    choice, meta = _target_choice(ws, target)
    key: dict[str, Any] = dict(meta)
    if choice.records is not None and isinstance(target, Mapping):
        source = ws.path(str(target.get("path")))
        key["mtime"] = source.stat().st_mtime
        key["size"] = source.stat().st_size

    def work() -> Response:
        job = _lookup_project(ws, choice, key)
        index_dir = job.parent / "index"
        result = projects.lookup(job, queries, index_dir=index_dir, top_k=top_k)
        result["target_description"] = choice.description
        return result

    response = _guarded(work)
    assert isinstance(response, dict)
    return response


def library_search(ws: Workspace, params: Params) -> Response:
    """Search one or more library entries for a query (retrieval only)."""
    slugs = params.get("libraries")
    query = _str(params, "query")
    top_k = _int(params, "top_k", default=10, low=1, high=projects.MAX_TOP_K) or 10
    result = _lookup(ws, {"libraries": slugs}, [Record("q1", {"text": query})], top_k)
    return {"query": query, "candidates": result["rows"][0]["candidates"], **_meta(result)}


def _meta(result: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "rows"}


# --- projects and quick mapping ---------------------------------------------------------


def create_project(ws: Workspace, params: Params) -> Response:
    """Write a project (job directory) for mapping or clustering and return it."""
    mode = _str(params, "mode", default="map", required=False)
    if mode not in ("map", "cluster"):
        raise ApiError(400, "invalid_parameter", "mode must be 'map' or 'cluster'")
    name = _str(params, "name", required=False) or (
        "clustering" if mode == "cluster" else "mapping"
    )
    sources, source_meta = _source_choice(ws, params.get("source"))
    model = params.get("model") if isinstance(params.get("model"), Mapping) else None
    if mode == "cluster":
        relation = _str(params, "relation", required=False)
        summary = _guarded(
            lambda: projects.create_cluster_project(
                ws.root,
                name=name,
                records=sources,
                relation=relation,
                model=model,
                provenance=source_meta,
            )
        )
    else:
        target, target_meta = _target_choice(ws, params.get("target"))
        descriptions = params.get("descriptions")
        policy = params.get("policy")
        summary = _guarded(
            lambda: projects.create_map_project(
                ws.root,
                library(ws),
                name=name,
                sources=sources,
                targets=target,
                descriptions=descriptions if isinstance(descriptions, Mapping) else None,
                model=model,
                policy=policy if isinstance(policy, Mapping) else None,
                provenance={**source_meta, **target_meta},
            )
        )
    assert isinstance(summary, dict)
    return _project_view(ws, summary)


def quick_map(ws: Workspace, params: Params) -> Response:
    """Map pasted terms: candidates only (no model), or a run with a model as a task."""
    mode = _str(params, "mode", default="candidates", required=False)
    if mode not in ("candidates", "offline", "endpoint"):
        raise ApiError(400, "invalid_parameter", "mode must be candidates, offline or endpoint")
    terms = files.terms_from_text(_str(params, "terms"))
    if not terms:
        raise ApiError(400, "missing_parameter", "type at least one term")
    if len(terms) > projects.MAX_LOOKUP_TERMS:
        raise ApiError(413, "too_large", f"at most {projects.MAX_LOOKUP_TERMS} terms at once")
    target = params.get("target")
    if mode == "candidates":
        top_k = _int(params, "top_k", default=5, low=1, high=projects.MAX_TOP_K) or 5
        return {"mode": mode, **_lookup(ws, target, terms, top_k)}
    project = create_project(
        ws,
        {
            "mode": "map",
            "name": _str(params, "name", required=False)
            or f"quick-{time.strftime('%Y%m%d-%H%M%S')}",
            "source": {"text": "\n".join(str(t.fields["text"]) for t in terms)},
            "target": target,
            "descriptions": params.get("descriptions"),
            "model": params.get("model"),
            "policy": params.get("policy"),
        },
    )
    assert isinstance(project, dict)
    task = ws.start_task(
        {
            "job": project["job_path"],
            "out": project["suggested_out"],
            "model": "endpoint" if mode == "endpoint" else "offline",
            "max_calls": params.get("max_calls"),
        }
    )
    return {"mode": mode, "project": project, "task": task}


def mapping(ws: Workspace, params: Params) -> Response:
    """A page of a run's results with each source's text and each target's label.

    Works for any matching run whose job is known to the UI; labels come from the job's
    own collections. Use `/api/results` for the plain rows.
    """
    run_dir = ws.path(_str(params, "dir"), must_exist=True)
    page = ws.results(params)
    assert isinstance(page, dict)
    if page.get("status") == "error":
        return page
    job_rel = ws._run_jobs().get(ws.rel(run_dir))
    labels: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    if job_rel:
        rows = page["data"]["rows"]
        wanted_targets = {r["matched_id"] for r in rows if r.get("matched_id")}
        wanted_sources = {r["source_id"] for r in rows}
        try:
            spec = ops.load_valid_job(ws.path(job_rel), operation="mapping")
            templates = spec.build_templates()
            for record in spec.build_target_records():
                if record.id in wanted_targets:
                    labels[record.id] = {
                        "label": record.fields.get("label") or templates.render_candidate(record),
                        "ontology": record.fields.get("ontology"),
                    }
            for record in spec.build_source_records():
                if record.id in wanted_sources:
                    texts[record.id] = templates.render_query(record)
        except Exception:  # labels are a convenience; the rows stand without them
            labels, texts = {}, {}
    for row in page["data"]["rows"]:
        row["source_text"] = texts.get(row["source_id"])
        target = labels.get(row.get("matched_id") or "")
        row["target_label"] = target["label"] if target else None
        row["target_ontology"] = target["ontology"] if target else None
    page["data"]["job_path"] = job_rel
    return page


def mapping_csv(ws: Workspace, params: Params) -> Response:
    """The whole current mapping with source texts and target labels, as CSV."""
    import csv
    import io

    first = mapping(ws, {**params, "offset": 0, "limit": ops.MAX_PAGE_SIZE})
    assert isinstance(first, dict)
    if first.get("status") == "error":
        raise ApiError(422, "export_failed", "; ".join(e["message"] for e in first["errors"]))
    rows = list(first["data"]["rows"])
    offset = first["data"]["next_offset"]
    while offset is not None:
        page = mapping(ws, {**params, "offset": offset, "limit": ops.MAX_PAGE_SIZE})
        assert isinstance(page, dict)
        rows += page["data"]["rows"]
        offset = page["data"]["next_offset"]
    out = io.StringIO()
    columns = [
        "source_id",
        "source_text",
        "status",
        "matched_id",
        "target_label",
        "target_ontology",
        "confidence",
        "reason",
    ]
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    name = Path(_str(params, "dir")).name
    return Download(f"{name}-mapping.csv", "text/csv; charset=utf-8", out.getvalue().encode())


def job_file(ws: Workspace, params: Params) -> Response:
    """Download a project's job file (to run it with the CLI)."""
    path = ws.path(_str(params, "path"), must_exist=True)
    if path.suffix not in (".yaml", ".yml"):
        raise ApiError(400, "invalid_parameter", "not a job file")
    return Download(path.name, "application/yaml; charset=utf-8", path.read_bytes())


def review_upload(ws: Workspace, params: Params) -> Response:
    """Apply a filled-in review worksheet (CSV) from the workspace."""
    run_dir = ws.path(_str(params, "dir"), must_exist=True)
    sheet = ws.path(_str(params, "path"), must_exist=True)
    job = ws.path(_str(params, "job"), must_exist=True)
    return _envelope("review apply", lambda: ops.review_apply(run_dir, sheet, job))


def review_sheet(ws: Workspace, params: Params) -> Response:
    """Download the review worksheet to fill in a spreadsheet."""
    run_dir = ws.path(_str(params, "dir"), must_exist=True)
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "review.csv"
        try:
            ops.review_export(run_dir, out)
        except Exception as exc:
            result = ops.failure_result("review export", exc)
            message = "; ".join(e.message for e in result.errors)
            raise ApiError(422, "export_failed", message) from None
        return Download(f"{run_dir.name}-review.csv", "text/csv; charset=utf-8", out.read_bytes())


# --- presets and session credentials ---------------------------------------------------


def presets(ws: Workspace, params: Params) -> Response:
    return {
        "presets": projects.PRESETS,
        "descriptions": projects.DEFAULT_DESCRIPTIONS,
    }


def credentials(ws: Workspace, params: Params) -> Response:
    names = {str(p["api_key_env"]) for p in projects.PRESETS.values() if p["api_key_env"]}
    names |= ws.session_credentials
    return {
        "credentials": [
            {
                "env": name,
                "set": bool(os.environ.get(name)),
                "from_page": name in ws.session_credentials,
            }
            for name in sorted(names)
        ]
    }


def set_credential(ws: Workspace, params: Params) -> Response:
    """Set (or with an empty value, clear) an API key for this server process only.

    The value goes into this process's environment, where a job's `api_key_env` reads
    it. It is never written to disk and never sent back to the page.
    """
    name = _str(params, "env")
    if not _ENV_NAME.match(name):
        raise ApiError(
            400,
            "invalid_parameter",
            "the variable name must be upper case and end in _API_KEY, _KEY or _TOKEN",
        )
    value = params.get("value")
    if not isinstance(value, str):
        raise ApiError(400, "invalid_parameter", "value must be a string")
    if value.strip():
        os.environ[name] = value.strip()
        ws.session_credentials.add(name)
    else:
        if name not in ws.session_credentials:
            raise ApiError(
                403, "not_from_page", f"{name} was not set from this page; it is left alone"
            )
        os.environ.pop(name, None)
        ws.session_credentials.discard(name)
    return {"env": name, "set": bool(os.environ.get(name))}


Handler = Callable[[Workspace, Params], Response]

ROUTES: dict[tuple[str, str], Handler] = {
    ("POST", "/api/upload"): upload,
    ("GET", "/api/files"): list_files,
    ("POST", "/api/files/delete"): delete_file,
    ("GET", "/api/inspect"): inspect,
    ("GET", "/api/library"): library_list,
    ("GET", "/api/library/catalog"): library_catalog,
    ("POST", "/api/library/import"): library_import,
    ("POST", "/api/library/download"): library_download,
    ("POST", "/api/library/delete"): library_delete,
    ("GET", "/api/library/terms"): library_terms,
    ("POST", "/api/library/search"): library_search,
    ("POST", "/api/projects"): create_project,
    ("POST", "/api/quickmap"): quick_map,
    ("GET", "/api/mapping"): mapping,
    ("GET", "/api/mapping.csv"): mapping_csv,
    ("GET", "/api/job-file"): job_file,
    ("GET", "/api/review-sheet"): review_sheet,
    ("POST", "/api/review-upload"): review_upload,
    ("GET", "/api/presets"): presets,
    ("GET", "/api/credentials"): credentials,
    ("POST", "/api/credentials"): set_credential,
}

# Plain links (downloads) cannot carry the token header; these GETs accept it in the URL.
DOWNLOAD_ROUTES = ("/api/export", "/api/mapping.csv", "/api/job-file", "/api/review-sheet")

__all__ = ["DOWNLOAD_ROUTES", "ROUTES", "library"]

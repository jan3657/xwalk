#!/usr/bin/env python
"""Download and parse full ontologies into a library directory for `xwalk ui --library`.

Each ontology is downloaded once, parsed into `<out>/<slug>/terms.jsonl` (id, label,
synonyms, definition) and, with `--cache`, given a ready search index, so a hosted
explorer offers it parsed and searchable from the first visit. Run it while building the
image (see `deploy/Dockerfile`):

    python scripts/build_ontology_library.py --out /opt/xwalk/library --cache /opt/xwalk/cache
    python scripts/build_ontology_library.py --out lib --only hp mondo
    python scripts/build_ontology_library.py --out lib --url "HPO=https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo"

Entries already in `<out>` are kept (pass `--refresh` to rebuild them). The default set is
the catalog in `xwalk.ui.library.CATALOG` (OBO Foundry permanent URLs).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from xwalk.ui import files, projects
from xwalk.ui.library import CATALOG, Library, download, slugify


def _build(
    library: Library, slug: str, name: str, url: str, description: str, refresh: bool
) -> str:
    slug = slugify(slug)
    existing = {o.slug: o for o in library.imported()}
    if slug in existing and not refresh:
        print(f"{slug}: kept ({existing[slug].count} terms)", flush=True)
        return slug
    if slug in existing:
        library.delete(slug)
    started = time.monotonic()
    fmt = files.detect_format(Path(url.split("?")[0]))
    with tempfile.TemporaryDirectory() as scratch:
        path = download(url, Path(scratch) / f"download.{fmt}")
        records = files.target_records(
            path, fmt, id_column="id", label_column="label", synonyms_column="synonyms"
        )
        entry = library.add(
            name, records, source=url, description=description, homepage=url, slug=slug
        )
    print(f"{entry.slug}: {entry.count} terms in {time.monotonic() - started:.0f} s", flush=True)
    return entry.slug


def _index(library: Library, slug: str, cache: Path) -> None:
    started = time.monotonic()
    choice = projects.TargetChoice(libraries=[slug])
    job = projects.lookup_job(library, choice, projects.library_lookup_key([slug]), cache)
    from xwalk.records import Record

    projects.lookup(
        job, [Record("q1", {"text": "warm up"})], index_dir=job.parent / "index", top_k=1
    )
    print(f"{slug}: index ready in {time.monotonic() - started:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="the library directory to fill")
    parser.add_argument("--cache", help="also build each ontology's search index here")
    parser.add_argument("--only", nargs="*", help="catalog ids to build (default: all)")
    parser.add_argument("--skip", nargs="*", default=[], help="catalog ids to leave out")
    parser.add_argument(
        "--url", action="append", default=[], help="NAME=URL of another ontology (repeatable)"
    )
    parser.add_argument("--refresh", action="store_true", help="rebuild entries already there")
    parser.add_argument(
        "--keep-going", action="store_true", help="report a failed ontology and continue"
    )
    args = parser.parse_args(argv)

    jobs: list[tuple[str, str, str, str]] = []
    wanted = set(args.only) if args.only is not None else {c["id"] for c in CATALOG}
    unknown = wanted - {c["id"] for c in CATALOG}
    if unknown:
        parser.error(f"unknown catalog ids: {sorted(unknown)}")
    for entry in CATALOG:
        if entry["id"] in wanted and entry["id"] not in args.skip:
            jobs.append((entry["id"], entry["name"], entry["url"], entry["domain"]))
    for item in args.url:
        name, sep, url = item.partition("=")
        if not sep or not url:
            parser.error(f"--url needs NAME=URL, got {item!r}")
        jobs.append((name, name, url, ""))

    library = Library(Path(args.out))
    failures = 0
    for slug_hint, name, url, description in jobs:
        try:
            slug = _build(library, slug_hint, name, url, description, args.refresh)
            if args.cache:
                _index(library, slug, Path(args.cache))
        except Exception as exc:
            failures += 1
            print(f"{name}: FAILED: {exc}", file=sys.stderr, flush=True)
            if not args.keep_going:
                return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

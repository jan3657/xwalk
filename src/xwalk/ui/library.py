"""The ontology library: parsed term collections ready to map onto.

Every entry is one JSONL file of terms (`id`, `label`, `synonyms`, `definition`, plus
any other field) and its metadata. Parsing happens once, when an ontology is imported;
mapping, browsing and search read the JSONL afterwards, so an OWL import needs rdflib
only at import time.

- **Built-in** entries ship with xwalk (`xwalk/ui/ontologies/`): the small slices the
  examples carry (200 terms each), clearly labelled as samples.
- **Imported** entries live in the workspace (`.xwalk-ui/library/<slug>/`): an OBO, OWL,
  CSV, TSV or JSONL file the user uploaded, or an ontology downloaded from a URL (the
  catalog lists OBO Foundry ontologies by their permanent URLs). Downloads are https
  only, to hosts that resolve to public addresses, with each redirect checked again;
  `allow_local` (`xwalk ui --allow-local-downloads`) lifts both for intranet mirrors.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from xwalk.records import Record
from xwalk.ui.files import FileProblem, read_jsonl, write_jsonl

TERMS_FILE = "terms.jsonl"
META_FILE = "meta.json"
DEFAULT_MAX_DOWNLOAD = 2 * 1024**3
MAX_REDIRECTS = 5
_SLUG = re.compile(r"[^a-z0-9]+")

# OBO Foundry ontologies by permanent URL (https://purl.obolibrary.org/obo/<id>.obo). Sizes
# change with every release; the UI does not quote them.
CATALOG: list[dict[str, str]] = [
    {
        "id": "hp",
        "name": "Human Phenotype Ontology (HPO)",
        "domain": "phenotypes, clinical signs",
        "url": "https://purl.obolibrary.org/obo/hp.obo",
    },
    {
        "id": "mondo",
        "name": "Mondo Disease Ontology",
        "domain": "diseases",
        "url": "https://purl.obolibrary.org/obo/mondo.obo",
    },
    {
        "id": "doid",
        "name": "Human Disease Ontology (DOID)",
        "domain": "diseases",
        "url": "https://purl.obolibrary.org/obo/doid.obo",
    },
    {
        "id": "go-basic",
        "name": "Gene Ontology (basic)",
        "domain": "biological processes, functions, cellular components",
        "url": "https://purl.obolibrary.org/obo/go/go-basic.obo",
    },
    {
        "id": "chebi",
        "name": "ChEBI",
        "domain": "chemical entities (large: hundreds of MB)",
        "url": "https://purl.obolibrary.org/obo/chebi.obo",
    },
    {
        "id": "uberon",
        "name": "Uberon anatomy",
        "domain": "anatomy across species",
        "url": "https://purl.obolibrary.org/obo/uberon.obo",
    },
    {
        "id": "cl",
        "name": "Cell Ontology (CL)",
        "domain": "cell types",
        "url": "https://purl.obolibrary.org/obo/cl.obo",
    },
    {
        "id": "pato",
        "name": "Phenotype And Trait Ontology (PATO)",
        "domain": "qualities and traits",
        "url": "https://purl.obolibrary.org/obo/pato.obo",
    },
    {
        "id": "uo",
        "name": "Units of Measurement Ontology (UO)",
        "domain": "units",
        "url": "https://purl.obolibrary.org/obo/uo.obo",
    },
    {
        "id": "envo",
        "name": "Environment Ontology (ENVO)",
        "domain": "environments, habitats, materials",
        "url": "https://purl.obolibrary.org/obo/envo.obo",
    },
]


class LibraryError(ValueError):
    """A library request that cannot be met (unknown entry, a built-in to delete, ...)."""


def slugify(name: str) -> str:
    slug = _SLUG.sub("-", name.lower()).strip("-")[:48]
    return slug or "ontology"


@dataclass
class Ontology:
    slug: str
    name: str
    kind: str  # hosted, builtin or imported
    count: int
    path: Path
    description: str = ""
    source: str = ""
    homepage: str = ""
    licence: str = ""
    created: float | None = None
    sample: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "kind": self.kind,
            "count": self.count,
            "description": self.description,
            "source": self.source,
            "homepage": self.homepage,
            "licence": self.licence,
            "created": self.created,
            "sample": self.sample,
            "read_only": self.kind != "imported",
        }


def _builtin_dir() -> Path:
    return Path(str(resources.files("xwalk.ui") / "ontologies"))


class Library:
    """Entries from three places: `shared` directories (read-only, pre-parsed by the
    operator, kind ``hosted``), the samples shipped with xwalk (``builtin``) and the
    workspace's own `directory` (``imported``)."""

    def __init__(self, directory: Path, *, shared: Sequence[Path] = ()) -> None:
        self.directory = directory
        self.shared = tuple(shared)

    def builtin(self) -> list[Ontology]:
        index = json.loads((_builtin_dir() / "index.json").read_text(encoding="utf-8"))
        return [
            Ontology(
                slug=str(entry["slug"]),
                name=str(entry["name"]),
                kind="builtin",
                count=int(entry["count"]),
                path=_builtin_dir() / f"{entry['slug']}.jsonl",
                description=str(entry.get("description", "")),
                source=str(entry.get("source", "")),
                homepage=str(entry.get("homepage", "")),
                licence=str(entry.get("licence", "")),
                sample=True,
            )
            for entry in index
        ]

    def imported(self) -> list[Ontology]:
        return _read_entries(self.directory, "imported")

    def hosted(self) -> list[Ontology]:
        found: list[Ontology] = []
        for directory in self.shared:
            found += _read_entries(directory, "hosted")
        return found

    def all(self) -> list[Ontology]:
        return self.hosted() + self.builtin() + self.imported()

    def get(self, slug: str) -> Ontology:
        for entry in self.all():
            if entry.slug == slug:
                return entry
        raise LibraryError(f"no ontology {slug!r} in the library")

    def terms(self, slug: str) -> Iterator[Record]:
        return read_jsonl(self.get(slug).path)

    def add(
        self,
        name: str,
        records: Iterable[Record],
        *,
        source: str,
        description: str = "",
        homepage: str = "",
        slug: str | None = None,
    ) -> Ontology:
        """Store parsed terms under a new slug. Ids must be unique; empty labels are
        dropped (a term nobody can name cannot be matched)."""
        slug = self._free_slug(slugify(slug or name))
        directory = self.directory / slug
        seen: set[str] = set()

        def checked() -> Iterator[Record]:
            for record in records:
                if record.id in seen:
                    raise FileProblem(f"term id {record.id!r} appears twice")
                seen.add(record.id)
                if str(record.fields.get("label") or "").strip():
                    yield record

        try:
            count = write_jsonl(checked(), directory / TERMS_FILE)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        if count == 0:
            shutil.rmtree(directory, ignore_errors=True)
            raise FileProblem("no terms with a label were found")
        meta = {
            "name": name,
            "count": count,
            "source": source,
            "description": description,
            "homepage": homepage,
            "created": time.time(),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return self.get(slug)

    def delete(self, slug: str) -> None:
        entry = self.get(slug)
        if entry.kind != "imported":
            raise LibraryError(f"{entry.name} is shared and cannot be deleted")
        shutil.rmtree(self.directory / slug)

    def _free_slug(self, base: str) -> str:
        taken = {entry.slug for entry in self.all()}
        if base not in taken:
            return base
        n = 2
        while f"{base}-{n}" in taken:
            n += 1
        return f"{base}-{n}"


def _read_entries(directory: Path, kind: str) -> list[Ontology]:
    found: list[Ontology] = []
    if not directory.is_dir():
        return found
    for entry in sorted(directory.iterdir()):
        meta_path = entry / META_FILE
        if not (meta_path.is_file() and (entry / TERMS_FILE).is_file()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        found.append(
            Ontology(
                slug=entry.name,
                name=str(meta.get("name") or entry.name),
                kind=kind,
                count=int(meta.get("count") or 0),
                path=entry / TERMS_FILE,
                description=str(meta.get("description", "")),
                source=str(meta.get("source", "")),
                homepage=str(meta.get("homepage", "")),
                licence=str(meta.get("licence", "")),
                created=meta.get("created"),
            )
        )
    return found


def combined_terms(library: Library, slugs: list[str]) -> tuple[list[Record], list[str]]:
    """The terms of several entries as one collection, each tagged with `ontology`.

    Returns `(records, warnings)`. An id that two entries share keeps its first entry's
    term, and the overlap is reported, so a target id always names one record.
    """
    if not slugs:
        raise LibraryError("choose at least one ontology")
    records: list[Record] = []
    seen: dict[str, str] = {}
    warnings: list[str] = []
    for slug in dict.fromkeys(slugs):
        entry = library.get(slug)
        repeated = 0
        for record in library.terms(slug):
            if record.id in seen:
                repeated += 1
                continue
            seen[record.id] = slug
            records.append(Record(record.id, {**record.fields, "ontology": entry.name}))
        if repeated:
            warnings.append(f"{entry.name}: {repeated} term id(s) already in an earlier choice")
    return records, warnings


Progress = Callable[[int, int | None], None]


def _download_client(url: str, timeout: float, allow_local: bool) -> Any:
    """An httpx client for one hop of a download, after checking where `url` points.

    Without `allow_local`: https only, every address the host resolves to must be
    public, and the connection goes to those addresses (no second lookup, so DNS
    rebinding cannot swap in a private one). With it: http or https, anywhere."""
    from urllib.parse import urlsplit

    import httpx

    from xwalk.ui import egress

    scheme = urlsplit(url).scheme.lower()
    if allow_local:
        if scheme not in ("http", "https"):
            raise FileProblem(f"{url}: only http:// and https:// URLs can be downloaded")
        return httpx.Client(timeout=timeout, follow_redirects=False)
    if scheme != "https":
        raise FileProblem(
            f"{url}: only https:// URLs can be downloaded (start `xwalk ui` with "
            "--allow-local-downloads for http or a local mirror)"
        )
    host = egress.wire_host(url)
    if not host:
        raise FileProblem(f"{url}: no host to download from")
    try:
        addresses = egress.resolve(host, urlsplit(url).port or 443)
    except (OSError, ValueError):
        raise FileProblem(f"{url}: cannot resolve {host}") from None
    if not addresses or not all(egress.is_public(a) for a in addresses):
        raise FileProblem(
            f"{url}: {host} resolves to a non-public address (start `xwalk ui` with "
            "--allow-local-downloads to fetch from your own network)"
        )
    return httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        transport=egress.PinnedTransport(host, addresses),
    )


def download(
    url: str,
    dest: Path,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD,
    progress: Progress | None = None,
    timeout: float = 60.0,
    allow_local: bool = False,
    max_redirects: int = MAX_REDIRECTS,
) -> Path:
    """Stream `url` to `dest` (decompressing a `.gz` body into `dest`); returns `dest`.

    Only https to hosts that resolve to public addresses, unless `allow_local` (then
    http and any address too). At most `max_redirects` redirects, each target checked
    like the first URL; more than `max_bytes` is refused.
    """
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = dest.with_name(dest.name + ".download")
    received = 0
    try:
        current = url
        for _ in range(max_redirects + 1):
            with (
                _download_client(current, timeout, allow_local) as client,
                client.stream("GET", current) as response,
            ):
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FileProblem(f"{current}: a redirect without a Location")
                    current = str(response.url.join(location))
                    continue
                if response.status_code >= 400:
                    raise FileProblem(f"{url}: the server answered HTTP {response.status_code}")
                total = response.headers.get("content-length")
                expected = int(total) if total and total.isdigit() else None
                with raw.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        received += len(chunk)
                        if received > max_bytes:
                            raise FileProblem(
                                f"{url}: larger than the {max_bytes // 1024**2} MB download limit"
                            )
                        handle.write(chunk)
                        if progress is not None:
                            progress(received, expected)
                break
        else:
            raise FileProblem(f"{url}: more than {max_redirects} redirects")
        with raw.open("rb") as handle:
            gzipped = handle.read(2) == b"\x1f\x8b"
        if gzipped:
            with gzip.open(raw, "rb") as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
            raw.unlink()
        else:
            raw.replace(dest)
    except httpx.HTTPError as exc:
        raise FileProblem(f"{url}: download failed: {exc}") from None
    finally:
        if raw.exists():
            raw.unlink()
    return dest


__all__ = [
    "CATALOG",
    "Library",
    "LibraryError",
    "Ontology",
    "combined_terms",
    "download",
    "slugify",
]

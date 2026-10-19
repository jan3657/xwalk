"""Confine paths to a set of root directories (the MCP server's `--root`).

An MCP client is an agent that may follow instructions planted in the data it reads, so
the server treats every path it is given as untrusted. `Roots.resolve` resolves a path
(relative paths against the working directory, symlinks followed, `..` collapsed) and
refuses it unless the result lies inside one of the roots. `Roots.check_job` applies the
same rule to the files a job refers to, before anything in the job is compiled or read.

Directories xwalk writes or reads beneath (`out`, `index_dir`, `run`) are also refused
when they already contain a symlink leading out of the roots.

The check is made when a tool is called; xwalk then opens the resolved path. A process
that can swap a directory for a symlink between the two can still escape -- the roots
guard against an agent naming paths, not against a hostile local user.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["PathRefused", "Roots"]


class PathRefused(ValueError):
    """A path outside every allowed root, or one that cannot be checked."""

    code = "path_outside_roots"


class Roots:
    """The directories a server may read and write: the working directory by default."""

    def __init__(self, roots: Iterable[str | Path] | None = None) -> None:
        given = [Path.cwd()] if roots is None else [Path(r) for r in roots]
        if not given:
            raise ValueError("at least one root directory is required")
        resolved: list[Path] = []
        for root in given:
            path = root.resolve()
            if not path.is_dir():
                raise ValueError(f"root {str(root)!r} is not a directory")
            if path not in resolved:
                resolved.append(path)
        self.paths: tuple[Path, ...] = tuple(resolved)

    def describe(self) -> str:
        return ", ".join(str(p) for p in self.paths)

    def _inside(self, path: Path) -> bool:
        """`path` (already resolved) is a root or lies beneath one.

        The textual check settles almost every call. On a case-insensitive filesystem a
        resolved path keeps the caller's casing, so `/Proj/x` and the root `/proj` differ
        as text; then the existing ancestors are compared with the roots by identity
        (`samefile`: device and inode), which accepts the same directory under another
        spelling and nothing else.
        """
        if any(path == root or path.is_relative_to(root) for root in self.paths):
            return True
        for ancestor in (path, *path.parents):
            try:
                if not ancestor.exists():
                    continue
                if any(ancestor.samefile(root) for root in self.paths):
                    return True
            except OSError:
                continue
        return False

    def _tree_links(self, top: Path, *, what: str, value: str | Path) -> None:
        """Refuse a directory holding a symlink (at any depth) that leaves the roots.

        xwalk writes and reads beneath `out`, `index_dir` and `run`; a pre-existing link
        such as `index/bm25 -> /elsewhere` would carry those writes out of the roots.
        Links that stay inside are followed; directory identities break cycles.
        """
        if not top.is_dir():
            return
        pending = [top]
        visited: set[tuple[int, int]] = set()
        while pending:
            directory = pending.pop()
            try:
                stat = directory.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in visited:
                    continue
                visited.add(identity)
                for entry in directory.iterdir():
                    target = entry
                    if entry.is_symlink():
                        try:
                            target = entry.resolve()
                        except (OSError, RuntimeError):
                            raise PathRefused(
                                f"{what}: {str(value)!r} contains an unusable symlink "
                                f"{str(entry)!r}"
                            ) from None
                        if not self._inside(target):
                            raise PathRefused(
                                f"{what}: {str(value)!r} contains a symlink {str(entry)!r} "
                                f"that leads outside the allowed roots ({self.describe()})"
                            )
                    if target.is_dir():
                        pending.append(target)
            except OSError as exc:
                raise PathRefused(
                    f"{what}: {str(value)!r} cannot check directory {str(directory)!r} ({exc})"
                ) from None

    def resolve(self, value: str | Path, *, what: str, tree: bool = False) -> Path:
        """`value` resolved inside a root, or `PathRefused` naming the argument `what`.

        `tree=True` (a directory xwalk will write or read beneath) also refuses one that
        contains a symlink leading outside the roots."""
        try:
            resolved = Path(value).resolve()  # relative: against the working directory
        except (OSError, ValueError, RuntimeError) as exc:  # NUL byte, symlink loop
            raise PathRefused(f"{what}: {str(value)!r} is not a usable path ({exc})") from None
        if not self._inside(resolved):
            raise PathRefused(
                f"{what}: {str(value)!r} resolves outside the allowed roots "
                f"({self.describe()}); start the server with --root to allow another "
                "directory"
            )
        if tree:
            self._tree_links(resolved, what=what, value=value)
        return resolved

    def check_job(self, job: str | Path) -> None:
        """Refuse a job file that refers to files outside the roots.

        Checked: `source.path`, `target.path`, `prompts.slots`, a dense retriever's
        `model` when it names a local path, and a `sql` collection's database (a sqlite
        file inside the roots; network databases are refused). A job file that does not
        load is left alone: the operation reports it as invalid without compiling it.
        """
        refused: list[str] = []
        for where, value in _job_references(Path(job)):
            try:
                self.resolve(value, what=where)
            except PathRefused as exc:
                refused.append(str(exc))
        if refused:
            raise PathRefused(f"job {str(job)!r} refers outside the roots: " + "; ".join(refused))


def _job_references(job: Path) -> list[tuple[str, Path | str]]:
    """Every local path a job refers to, as `(where in the job file, path)`."""
    from xwalk.cluster.job import is_cluster_job, load_cluster_job
    from xwalk.config import JobValidationError, load_job

    spec: Any
    try:
        spec = load_cluster_job(job) if is_cluster_job(job) else load_job(job)
    except JobValidationError:
        return []
    base: Path = spec.base_dir
    found: list[tuple[str, Path | str]] = []
    collections: list[tuple[str, Any]] = [("source", spec.source)]
    if getattr(spec, "target", None) is not None:
        collections.insert(0, ("target", spec.target))
    for role, collection in collections:
        if collection.path is not None:
            found.append((f"{role}.path", base / collection.path))
        if collection.kind == "sql" and collection.url:
            found += _sqlite_reference(f"{role}.url", collection.url)
    prompts = getattr(spec, "prompts", None)
    if prompts is not None:
        found.append(("prompts.slots", base / prompts.slots))
    retrievers: Sequence[tuple[str, Any]] = [
        (f"retrievers.{i}", r) for i, r in enumerate(getattr(spec, "retrievers", []))
    ]
    if getattr(spec, "dense", None) is not None:
        retrievers = [*retrievers, ("dense", spec.dense)]
    for where, retriever in retrievers:
        model = getattr(retriever, "model", None)
        # sentence-transformers loads a local directory when the name exists as a path
        # (relative to the working directory); anything else is a hub id.
        if model and (os.path.isabs(model) or os.path.exists(model)):
            found.append((f"{where}.model", model))
    return found


def _sqlite_reference(where: str, url: str) -> list[tuple[str, Path | str]]:
    """The sqlite file a SQLAlchemy URL opens, or `PathRefused` for anything else."""
    scheme, sep, rest = url.partition("://")
    if not sep or scheme.split("+", 1)[0].lower() != "sqlite":
        raise PathRefused(
            f"{where}: the MCP server reads no network database ({scheme or url!r}); "
            "only a sqlite file inside the roots. Run this job with the CLI"
        )
    database, _, query = rest.partition("?")
    if "uri=true" in query.lower():
        raise PathRefused(f"{where}: sqlite URI filenames (uri=true) are not accepted")
    database = database[1:] if database.startswith("/") else database
    if database in ("", ":memory:"):
        return []
    return [(where, database)]  # SQLAlchemy opens a relative file from the working directory

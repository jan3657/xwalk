"""Benchmark data: manifests, digest checks, and constructed variants.

A manifest (``benchmarks/manifests/*.yaml``) pins the frozen files of one pilot dataset
by sha256 and declares its variants. `materialise` writes each variant into a work
directory as plain files every method reads the same way:

- ``targets.jsonl`` -- the catalog (same fields the job's loader produced)
- ``mentions.jsonl`` -- the source records, unchanged
- ``gold.csv`` -- gold ids, already normalised/expanded and restricted to the catalog
- ``job.yaml`` -- the original job pointed at the files above
- ``construction.json`` -- what the construction removed and why

Nothing here calls a model or reads the network.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xwalk.config import load_job
from xwalk.evaluate import GoldSet, Partition, Partitioner, load_gold_csv
from xwalk.records import Record

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


class DataRevisionError(RuntimeError):
    """A frozen file no longer has the digest its manifest pins."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(name_or_path: str | Path) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = MANIFEST_DIR / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a manifest is a mapping")
    data["_path"] = str(path)
    return data


def verify_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Check every pinned file. Returns the digests (the data revision)."""
    found: dict[str, str] = {}
    for rel, expected in manifest["sha256"].items():
        actual = sha256_file(REPO_ROOT / rel)
        if actual != expected:
            raise DataRevisionError(
                f"{rel}: sha256 {actual} does not match the manifest's {expected}; "
                f"record a new data revision in {manifest['_path']}"
            )
        found[rel] = actual
    return found


@dataclass
class Variant:
    """One materialised dataset variant."""

    name: str
    dataset: str
    directory: Path
    job_path: Path
    targets: list[Record]
    sources: list[Record]
    gold: GoldSet
    # Gold before any construction: what clustering uses (it involves no catalog).
    base_gold: GoldSet
    split: dict[str, str]
    query_field: str
    construction: dict[str, Any] = field(default_factory=dict)

    def ids_in_split(self, split: str) -> list[str]:
        if split == "all":
            return [s.id for s in self.sources]
        return [s.id for s in self.sources if self.split[s.id] == split]


def _expander(manifest: Mapping[str, Any]) -> tuple[Any, Any]:
    module_name = manifest.get("gold_expander")
    if not module_name:
        return None, None
    module = importlib.import_module(module_name)
    targets = REPO_ROOT / manifest["frozen_files"]["targets"]
    return module.normalize, module.make_expander(str(targets))


def _load_base(manifest: Mapping[str, Any]) -> tuple[list[Record], list[Record], GoldSet]:
    files = manifest["frozen_files"]
    spec = load_job(REPO_ROOT / files["job"])
    targets = list(spec.build_target_records())
    sources = list(spec.build_source_records())
    normalize, expand = _expander(manifest)
    gold = load_gold_csv(REPO_ROOT / files["gold"], normalize=normalize, expand=expand)
    return targets, sources, gold


def _hash_order(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{salt}\x00{i}".encode()).hexdigest())


def remove_gold_targets(
    targets: list[Record],
    gold: GoldSet,
    *,
    target_nomatch: int,
    max_nomatch: int,
    salt: str,
) -> tuple[list[Record], dict[str, frozenset[str]], dict[str, Any]]:
    """Construct no-match cases by deleting whole gold sets from the catalog.

    Returns (remaining targets, new gold labels, construction log). Every new label is
    the old one restricted to the remaining catalog, so a mention whose gold vanished
    entirely is now a correct no-match case.
    """
    catalog = {t.id for t in targets}
    removed: set[str] = set()
    chosen: list[str] = []
    skipped: list[str] = []

    def labels_after(gone: set[str]) -> dict[str, frozenset[str]]:
        return {sid: frozenset((ids & catalog) - gone) for sid, ids in gold.labels.items()}

    def nomatch_count(labels: Mapping[str, frozenset[str]]) -> int:
        return sum(1 for ids in labels.values() if not ids)

    for source_id in _hash_order(list(gold.labels), salt):
        current = labels_after(removed)
        if nomatch_count(current) >= target_nomatch:
            break
        if not current[source_id]:
            continue
        trial = removed | set(current[source_id])
        if nomatch_count(labels_after(trial)) > max_nomatch:
            skipped.append(source_id)
            continue
        removed = trial
        chosen.append(source_id)

    labels = labels_after(removed)
    before = labels_after(set())
    log = {
        "method": "remove_gold_targets",
        "salt": salt,
        "target_nomatch": target_nomatch,
        "max_nomatch": max_nomatch,
        "seed_mentions": chosen,
        "skipped_mentions": skipped,
        "removed_target_ids": sorted(removed),
        "nomatch_mentions": sorted(s for s, ids in labels.items() if not ids),
        "gold_reduced_mentions": sorted(s for s, ids in labels.items() if ids and ids != before[s]),
        "targets_before": len(targets),
        "targets_after": len(targets) - len(removed),
    }
    return [t for t in targets if t.id not in removed], labels, log


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_gold(path: Path, labels: Mapping[str, frozenset[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "gold_ids"])
        for source_id, ids in labels.items():
            writer.writerow([source_id, "|".join(sorted(ids))])


def _write_job(manifest: Mapping[str, Any], directory: Path) -> Path:
    files = manifest["frozen_files"]
    job = yaml.safe_load((REPO_ROOT / files["job"]).read_text(encoding="utf-8"))
    job["name"] = directory.name
    job["target"] = {"kind": "jsonl", "path": "targets.jsonl", "id_field": "id"}
    job["source"] = {"kind": "jsonl", "path": "mentions.jsonl", "id_field": "id"}
    job["prompts"]["slots"] = str((REPO_ROOT / files["slots"]).resolve())
    path = directory / "job.yaml"
    path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    return path


def materialise(
    manifest: Mapping[str, Any],
    work_dir: Path,
    *,
    variants: list[str] | None = None,
    limit: int | None = None,
) -> list[Variant]:
    """Write every (or the named) variant of `manifest` under `work_dir`.

    `limit` keeps only the first N source records (quick checks; the catalog and the
    no-match construction are unchanged, so a limited run is a subset of the full one).
    """
    verify_digests(manifest)
    targets, sources, gold = _load_base(manifest)
    if limit is not None:
        sources = sources[:limit]
    catalog = {t.id for t in targets}
    partitioner = Partitioner(
        fractions=tuple(manifest["splits"]["fractions"]),
        salt=manifest["splits"]["salt"],
    )
    split = {s.id: partitioner.assign(s.id).value for s in sources}
    out: list[Variant] = []
    for entry in manifest["variants"]:
        if variants is not None and entry["name"] not in variants:
            continue
        construction: dict[str, Any] = {"method": "none"}
        variant_targets = targets
        labels = {sid: frozenset(ids & catalog) for sid, ids in gold.labels.items()}
        if entry["construction"] == "remove_gold_targets":
            variant_targets, labels, construction = remove_gold_targets(
                targets,
                gold,
                target_nomatch=int(entry["target_nomatch"]),
                max_nomatch=int(entry["max_nomatch"]),
                salt=str(entry["salt"]),
            )
        directory = work_dir / entry["name"]
        directory.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            directory / "targets.jsonl", [{"id": t.id, **t.fields} for t in variant_targets]
        )
        _write_jsonl(directory / "mentions.jsonl", [{"id": s.id, **s.fields} for s in sources])
        kept = {s.id for s in sources}
        labels = {sid: ids for sid, ids in labels.items() if sid in kept}
        _write_gold(directory / "gold.csv", labels)
        construction["gold_outside_catalog"] = sorted(
            sid for sid, ids in gold.labels.items() if ids and not ids & catalog
        )
        (directory / "construction.json").write_text(
            json.dumps(construction, indent=2, sort_keys=True), encoding="utf-8"
        )
        out.append(
            Variant(
                name=entry["name"],
                dataset=manifest["name"],
                directory=directory,
                job_path=_write_job(manifest, directory),
                targets=variant_targets,
                sources=sources,
                gold=GoldSet(labels),
                base_gold=gold,
                split=split,
                query_field=manifest.get("query_field", "mention"),
                construction=construction,
            )
        )
    return out


def split_counts(variant: Variant) -> dict[str, int]:
    counts = {p.value: 0 for p in Partition}
    for value in variant.split.values():
        counts[value] += 1
    return counts

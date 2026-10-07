"""Projects: job directories the UI writes for people who bring files, not job files.

A project is an ordinary xwalk job directory under `<workspace>/projects/<slug>/`:

    job.yaml        a normal job file (runs with `xwalk match --job ...` too)
    slots.yaml      the prompt vocabulary, from the form's three descriptions
    sources.jsonl   the records to match (or to cluster), normalized
    targets.jsonl   the records to match onto (library terms or an uploaded file)
    project.json    how it was made: the choices, the original files, warnings

`lookup` answers the retrieval-only question -- "which targets would be shown for these
terms?" -- with the job's own retrievers and fusion and no model call.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xwalk import ops
from xwalk.records import Record
from xwalk.retrieval.base import SearchRequest
from xwalk.retrieval.fusion import reciprocal_rank_fusion
from xwalk.ui.files import FileProblem, read_jsonl, write_jsonl
from xwalk.ui.library import Library, combined_terms, slugify

PROJECTS_DIR = "projects"
MAX_LOOKUP_TERMS = 5_000
MAX_TOP_K = 50

PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "profile": "openai",
    },
    "anthropic": {
        "label": "Anthropic (OpenAI-compatible endpoint)",
        "model": "claude-sonnet-5-5",
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
        "profile": "anthropic-compat",
    },
    "ollama": {
        "label": "Ollama on this machine",
        "model": "llama3.1",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "profile": "unknown",
    },
    "custom": {
        "label": "Other OpenAI-compatible endpoint",
        "model": "",
        "base_url": "",
        "api_key_env": None,
        "profile": "unknown",
    },
}

DEFAULT_DESCRIPTIONS = {
    "entity_noun": "term",
    "target_noun": "reference term",
    "domain_brief": "general terminology, where one concept can appear under different "
    "names, spellings, abbreviations and levels of detail",
}

_TEMPLATES = {
    "query": "{{ text }}",
    "context": "{% if context %}{{ context }}{% else %}{{ text }}{% endif %}",
    "doc": "{{ label }} {{ synonyms | join(' ') }}",
    "candidate": (
        "ID: {{ id }}\nLabel: {{ label }}"
        "{% if synonyms %}\nSynonyms: {{ synonyms | join('; ') }}{% endif %}"
        "{% if definition %}\nDefinition: {{ definition }}{% endif %}"
        "{% if ontology %}\nOntology: {{ ontology }}{% endif %}"
    ),
}


def _rubric() -> list[dict[str, Any]]:
    return [
        {
            "score": 1.0,
            "name": "Certain",
            "when": "the term is the candidate's label or one of its synonyms, ignoring case "
            "and punctuation",
        },
        {
            "score": 0.8,
            "name": "Confident",
            "when": "the same concept under another spelling, an abbreviation, a plural or "
            "a common variant name",
        },
        {
            "score": 0.5,
            "name": "Plausible",
            "when": "the candidate is a slightly broader or narrower concept, or the term is "
            "ambiguous and the context does not settle it",
        },
        {
            "score": 0.2,
            "name": "Speculative",
            "when": "related only by a broad category or a shared word",
        },
    ]


def slots_document(descriptions: Mapping[str, str]) -> dict[str, Any]:
    """The slots file for a generated job: the three descriptions and a generic rubric."""
    values = {
        key: (str(descriptions.get(key) or "").strip() or default)
        for key, default in DEFAULT_DESCRIPTIONS.items()
    }
    return {
        **values,
        "rubric": _rubric(),
        "hard_rules": [
            "Choose a candidate only when it denotes the same concept as the term. A merely "
            "related, broader or narrower concept is not the same concept.",
            "If no candidate denotes the same concept, abstain: a wrong mapping is worse "
            "than no mapping.",
        ],
        "disambiguation_steps": "Read the context, if any, before choosing. When the term "
        "carries a qualifier (a variant, a subtype, a unit, an organism), that qualifier "
        "must match.",
    }


def llm_block(model: Mapping[str, Any] | None) -> dict[str, Any]:
    """The job's `llm:` block from a preset plus overrides. A key is never written: the
    block names the environment variable that holds it."""
    choice = dict(model or {})
    preset_name = str(choice.get("preset") or "openai")
    if preset_name not in PRESETS:
        raise FileProblem(f"unknown model preset {preset_name!r}; choose from {sorted(PRESETS)}")
    preset = PRESETS[preset_name]
    block: dict[str, Any] = {"kind": "openai_compat"}
    for key in ("model", "base_url", "api_key_env", "profile"):
        value = choice.get(key)
        value = value.strip() if isinstance(value, str) else value
        block[key] = value if value not in (None, "") else preset[key]
    if not block["model"]:
        raise FileProblem("name the model to use (for example gpt-4o-mini)")
    if not block["base_url"]:
        raise FileProblem("give the endpoint's base URL")
    if not block["api_key_env"]:
        del block["api_key_env"]
    return block


@dataclass
class TargetChoice:
    """Where the targets come from: library entries, or records already read."""

    libraries: list[str] = field(default_factory=list)
    records: list[Record] | None = None
    description: str = ""


def _policy(options: Mapping[str, Any]) -> dict[str, Any]:
    accept_at = float(options.get("accept_at", 0.7))
    review_floor = float(options.get("review_floor", 0.4))
    if not 0.0 <= review_floor <= accept_at <= 1.0:
        raise FileProblem("thresholds must satisfy 0 <= review floor <= accept at <= 1")
    return {
        "max_attempts": 3,
        "accept_at": accept_at,
        "review_floor": review_floor,
        "verify_band": [accept_at, min(1.0, round(accept_at + 0.2, 4))],
        "concurrency": 8,
    }


def _project_dir(root: Path, name: str) -> Path:
    base = root / PROJECTS_DIR
    slug = slugify(name)
    candidate = base / slug
    n = 2
    while candidate.exists():
        candidate = base / f"{slug}-{n}"
        n += 1
    return candidate


class _Dumper(yaml.SafeDumper):
    """Multi-line strings (templates) as `|` blocks, so the job file reads as written."""


def _str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


def _write_yaml(path: Path, data: Mapping[str, Any], header: str) -> None:
    text = yaml.dump(dict(data), Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=88)
    path.write_text(header + text, encoding="utf-8")


def create_map_project(
    root: Path,
    library: Library,
    *,
    name: str,
    sources: Iterable[Record],
    targets: TargetChoice,
    descriptions: Mapping[str, str] | None = None,
    model: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a matching project and return its summary (`job_path`, counts, warnings)."""
    llm = llm_block(model)
    policy_block = _policy(policy or {})
    warnings: list[str] = []
    if targets.records is not None:
        target_records = targets.records
    else:
        target_records, warnings = combined_terms(library, targets.libraries)
    directory = _project_dir(root, name)
    try:
        directory.mkdir(parents=True)
        n_sources = write_jsonl(sources, directory / "sources.jsonl")
        if n_sources == 0:
            raise FileProblem("there are no source records with text to match")
        n_targets = write_jsonl(target_records, directory / "targets.jsonl")
        if n_targets == 0:
            raise FileProblem("there are no target records with a label")
        _write_yaml(directory / "slots.yaml", slots_document(descriptions or {}), _SLOTS_HEADER)
        job = {
            "name": name,
            "templates": dict(_TEMPLATES),
            "target": {"kind": "jsonl", "path": "targets.jsonl"},
            "source": {"kind": "jsonl", "path": "sources.jsonl"},
            "retrievers": [
                {"kind": "bm25", "name": "bm25", "limit": 30, "exact_fields": ["label", "synonyms"]}
            ],
            "llm": llm,
            "prompts": {"slots": "slots.yaml"},
            "policy": policy_block,
            "selector": {"max_candidates": 20},
        }
        _write_yaml(directory / "job.yaml", job, _JOB_HEADER)
    except BaseException:
        _remove_tree(directory)
        raise
    summary = {
        "kind": "match",
        "name": name,
        "dir": directory,
        "job_path": directory / "job.yaml",
        "sources": n_sources,
        "targets": n_targets,
        "libraries": targets.libraries,
        "target_description": targets.description,
        "warnings": warnings,
        "created": time.time(),
        "provenance": dict(provenance or {}),
    }
    _write_project_json(directory, summary)
    return summary


def create_cluster_project(
    root: Path,
    *,
    name: str,
    records: Iterable[Record],
    relation: str = "",
    model: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a clustering project (`kind: cluster`) over one collection's `text`."""
    llm = llm_block(model)
    directory = _project_dir(root, name)
    try:
        directory.mkdir(parents=True)
        count = write_jsonl(records, directory / "records.jsonl")
        if count == 0:
            raise FileProblem("there are no records with text to cluster")
        job: dict[str, Any] = {
            "kind": "cluster",
            "name": name,
            "source": {"kind": "jsonl", "path": "records.jsonl"},
            "templates": {"query": "{{ text }}"},
            "llm": llm,
        }
        if relation.strip():
            job["relation"] = relation.strip()
        _write_yaml(directory / "job.yaml", job, _CLUSTER_HEADER)
    except BaseException:
        _remove_tree(directory)
        raise
    summary = {
        "kind": "cluster",
        "name": name,
        "dir": directory,
        "job_path": directory / "job.yaml",
        "sources": count,
        "warnings": [],
        "created": time.time(),
        "provenance": dict(provenance or {}),
    }
    _write_project_json(directory, summary)
    return summary


def _write_project_json(directory: Path, summary: Mapping[str, Any]) -> None:
    data = {k: (str(v) if isinstance(v, Path) else v) for k, v in summary.items()}
    (directory / "project.json").write_text(
        json.dumps(data, indent=1, default=str), encoding="utf-8"
    )


def _remove_tree(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


_JOB_HEADER = """\
# Written by xwalk ui. A normal job file: run it from the command line with
#   xwalk validate --job job.yaml
#   xwalk match --job job.yaml --out runs/a --max-calls 200
# Credentials are read from the environment variable named in llm.api_key_env.
"""
_SLOTS_HEADER = """\
# Written by xwalk ui: the vocabulary the prompts use. Edit the nouns, the brief and the
# rules to describe your domain (docs/reference/prompts.md).
"""
_CLUSTER_HEADER = """\
# Written by xwalk ui. Clustering is experimental (docs/guide/clustering.md).
#   xwalk cluster --job job.yaml --out runs/a --max-calls 200
"""


# --- retrieval-only lookup --------------------------------------------------------------


async def lookup_async(
    job_path: Path,
    queries: Sequence[Record],
    *,
    index_dir: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """The fused top-k targets for each query record's `text`, with no model call."""
    if not 1 <= top_k <= MAX_TOP_K:
        raise FileProblem(f"top_k must be between 1 and {MAX_TOP_K}")
    if len(queries) > MAX_LOOKUP_TERMS:
        raise FileProblem(f"at most {MAX_LOOKUP_TERMS} terms per lookup")
    spec = ops.load_valid_job(job_path, operation="lookup")
    templates = spec.build_templates()
    targets = list(spec.build_target_records())
    from xwalk.stores.memory import MemoryStore

    store = MemoryStore.from_source(targets)
    plans = ops.plan_indexes(spec, targets, templates, index_dir)
    retrievers, actions = ops.prepare_indexes(plans, targets, templates, rebuild=False)

    async def one(query: Record) -> dict[str, Any]:
        text = str(query.fields.get("text") or "")
        groups = []
        for retriever in retrievers:
            depth = max(top_k, getattr(retriever, "default_limit", None) or top_k)
            groups.append(await retriever.search(SearchRequest(text=text, limit=depth)))
        fused = reciprocal_rank_fusion(groups, store)[:top_k]
        return {
            "id": query.id,
            "text": text,
            "candidates": [
                {
                    "rank": rank,
                    "id": c.id,
                    "label": c.record.fields.get("label", ""),
                    "synonyms": list(c.record.fields.get("synonyms") or [])[:6],
                    "ontology": c.record.fields.get("ontology"),
                    "fused_score": c.fused_score,
                    "retrievers": {hit.retriever: hit.rank for hit in c.evidence},
                    "exact": _exact(text, c.record),
                }
                for rank, c in enumerate(fused, start=1)
            ],
        }

    rows = await asyncio.gather(*(one(q) for q in queries))
    return {"rows": list(rows), "targets": len(targets), "indexes": actions, "top_k": top_k}


def lookup(
    job_path: Path, queries: Sequence[Record], *, index_dir: Path, top_k: int = 5
) -> dict[str, Any]:
    return asyncio.run(lookup_async(job_path, queries, index_dir=index_dir, top_k=top_k))


def _exact(text: str, record: Record) -> bool:
    """Whether the text equals the label or a synonym, ignoring case and spacing."""
    wanted = " ".join(text.lower().split())
    names = [record.fields.get("label", ""), *(record.fields.get("synonyms") or [])]
    return any(" ".join(str(n).lower().split()) == wanted for n in names)


def project_sources(directory: Path) -> list[Record]:
    return list(read_jsonl(directory / "sources.jsonl"))


__all__ = [
    "DEFAULT_DESCRIPTIONS",
    "MAX_LOOKUP_TERMS",
    "MAX_TOP_K",
    "PRESETS",
    "PROJECTS_DIR",
    "TargetChoice",
    "create_cluster_project",
    "create_map_project",
    "llm_block",
    "lookup",
    "lookup_async",
    "slots_document",
]

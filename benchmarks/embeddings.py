"""Retrieval-only bake-off of embedding models on the in-repo gold sets.

For every (dataset, model) it builds a dense index over the dataset's target catalog,
embeds every labelled source query and reports where the gold target lands: recall@k and
MRR for the dense retriever alone, for the job's own BM25 alone, and for both fused by
reciprocal rank (what a hybrid job would see). No LLM and no decider is called.

    python -m benchmarks.embeddings --dry-run                 # presets, data, cost estimate
    python -m benchmarks.embeddings                           # every preset, every dataset
    python -m benchmarks.embeddings --model nemotron-3.5-8b --model qwen3-8b
    python -m benchmarks.embeddings --job my/job.yaml --gold my/gold.csv --model voyage-4-lite

Remote presets go through OpenRouter (`--base-url`, key in `--api-key-env`, default
OPENROUTER_API_KEY). Free (`:free`) models need the account's privacy setting that allows
free-model providers, or OpenRouter answers 404. Indexes are cached under
`<out>/index` (gitignored), so a rerun pays only for the query embeddings. A preset that fails is
recorded with its error and the run goes on. Prices come from OpenRouter's public model
list; the cost column is what the endpoint reported, "unknown" when it said nothing.

The pilot catalogs hold about 200 targets each, which makes deep cut-offs easy: read
recall@1, recall@5 and MRR, and confirm a winner on a full catalog before making it a
default.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from xwalk.config import JobSpec, load_job
from xwalk.evaluate.gold import GoldSet, load_gold_csv
from xwalk.records import Record
from xwalk.retrieval.base import IndexMismatchError, Retriever
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.dense import DenseRetriever, Encoder, encoder_identity, index_components
from xwalk.retrieval.embeddings_api import EmbeddingUsage, OpenAIEmbeddingsEncoder
from xwalk.retrieve import retrieve
from xwalk.stores.memory import MemoryStore
from xwalk.templates import TemplateSet

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENROUTER = "https://openrouter.ai/api/v1"
DEFAULT_KS = (1, 5, 10, 20, 50)
DEPTH = 50  # dense retrieval depth; recall is reported up to it
CONCURRENCY = 8

# Qwen3-Embedding's documented query format: an instruction, then "Query:" and the text.
QWEN_QUERY = "Instruct: Given a term, retrieve the ontology concept it names\nQuery:"


@dataclass(frozen=True)
class Preset:
    """One embedding configuration. `model` starting with "local:" is a
    sentence-transformers model (needs xwalk[dense]); anything else is remote."""

    name: str
    model: str
    note: str = ""
    query_prefix: str = ""
    doc_prefix: str = ""
    query_input_type: str | None = None
    doc_input_type: str | None = None
    dimensions: int | None = None  # None: the model's native length, found with one call
    send_dimensions: bool = False
    truncate: bool = False
    batch_size: int = 64  # texts per request; some free endpoints fail on larger batches

    @property
    def local(self) -> bool:
        return self.model.startswith("local:")


# Input conventions per model card. Where a model's card names query/document roles but
# it is unknown whether OpenRouter forwards `input_type` for it, there are two presets:
# prefixes in the text, and `-it` with OpenRouter's documented input_type values.
PRESETS: dict[str, Preset] = {
    p.name: p
    for p in [
        Preset(
            "nemotron-3.5-8b",
            "nvidia/nemotron-3.5-embed-8b:free",
            "NVIDIA, Oct 2026, free; 4096-d, truncatable",
            query_prefix="query: ",
            doc_prefix="passage: ",
            batch_size=1,  # batches fail intermittently with HTTP 500 (2026-11-03)
        ),
        Preset(
            "nemotron-3.5-8b-it",
            "nvidia/nemotron-3.5-embed-8b:free",
            "as above, roles sent as input_type",
            query_input_type="search_query",
            doc_input_type="search_document",
            batch_size=1,  # batches fail intermittently with HTTP 500 (2026-11-03)
        ),
        Preset(
            "nemotron-3.5-1b",
            "nvidia/nemotron-3.5-embed-1b:free",
            "NVIDIA, Oct 2026, free; distilled from the 8B",
            query_prefix="query: ",
            doc_prefix="passage: ",
            batch_size=1,  # batches fail intermittently with HTTP 500 (2026-11-03)
        ),
        Preset(
            "nemotron-3.5-1b-it",
            "nvidia/nemotron-3.5-embed-1b:free",
            "as above, roles sent as input_type",
            query_input_type="search_query",
            doc_input_type="search_document",
            batch_size=1,  # batches fail intermittently with HTTP 500 (2026-11-03)
        ),
        Preset(
            "nemotron-3-1b",
            "nvidia/nemotron-3-embed-1b:free",
            "NVIDIA, Jul 2026, free; text-only",
            query_prefix="query: ",
            doc_prefix="passage: ",
        ),
        Preset(
            "qwen3-8b",
            "qwen/qwen3-embedding-8b",
            "$0.01/M, open weights, 4096-d",
            query_prefix=QWEN_QUERY,
        ),
        Preset(
            "qwen3-4b",
            "qwen/qwen3-embedding-4b",
            "$0.02/M, open weights, 2560-d",
            query_prefix=QWEN_QUERY,
        ),
        Preset("voyage-4-lite", "voyageai/voyage-4-lite", "$0.02/M, 1024-d"),
        Preset(
            "voyage-4-lite-it",
            "voyageai/voyage-4-lite",
            "as above, roles sent as input_type",
            query_input_type="search_query",
            doc_input_type="search_document",
        ),
        Preset("voyage-4", "voyageai/voyage-4", "$0.06/M, 1024-d"),
        Preset("voyage-4-large", "voyageai/voyage-4-large", "$0.12/M, 1024-d"),
        Preset("openai-3-small", "openai/text-embedding-3-small", "$0.02/M, 1536-d"),
        Preset("openai-3-large", "openai/text-embedding-3-large", "$0.13/M, 3072-d"),
        Preset("gemini-2", "google/gemini-embedding-2", "$0.20/M, 3072-d"),
        Preset("pplx-0.6b", "perplexity/pplx-embed-v1-0.6b", "$0.004/M, 1024-d"),
        Preset("pplx-4b", "perplexity/pplx-embed-v1-4b", "$0.03/M, 2560-d"),
        Preset("bge-m3", "baai/bge-m3", "$0.01/M, 1024-d, multilingual"),
        Preset(
            "e5-large",
            "intfloat/multilingual-e5-large",
            "$0.01/M, 1024-d, 512-token context",
            query_prefix="query: ",
            doc_prefix="passage: ",
        ),
        Preset("lfm-350m", "liquid/lfm-2.5-embedding-350m:free", "Liquid AI, free, 1024-d"),
        Preset(
            "local-e5-small",
            "local:intfloat/multilingual-e5-small",
            "the examples' local model (needs xwalk[dense])",
            query_prefix="query: ",
            doc_prefix="passage: ",
        ),
    ]
}


@dataclass(frozen=True)
class Dataset:
    name: str
    job: Path
    gold: Path
    expander: str | None = None  # module with normalize() and make_expander(targets)


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset("chebi", Path("examples/chebi/job.yaml"), Path("examples/chebi/sample/gold.csv")),
        Dataset(
            "ncbi_disease",
            Path("examples/ncbi_disease/job.yaml"),
            Path("examples/ncbi_disease/sample/gold.csv"),
            expander="examples.ncbi_disease.gold_normalize",
        ),
        Dataset(
            "nlm_gene",
            Path("examples/nlm_gene/job.yaml"),
            Path("examples/nlm_gene/sample/gold.csv"),
        ),
        Dataset(
            "cafeteria_fcd",
            Path("examples/cafeteria_fcd/job.yaml"),
            Path("examples/cafeteria_fcd/sample/gold.csv"),
        ),
    ]
}


# --- one dataset, loaded once --------------------------------------------------------


@dataclass
class Loaded:
    name: str
    job: JobSpec
    templates: TemplateSet
    targets: list[Record]
    store: MemoryStore
    labelled: list[tuple[Record, frozenset[str]]]
    queries: list[list[str]]

    @property
    def doc_texts(self) -> list[str]:
        return [self.templates.render_doc(t) for t in self.targets]


def load_dataset(dataset: Dataset) -> Loaded:
    job_path = dataset.job if dataset.job.is_absolute() else REPO_ROOT / dataset.job
    gold_path = dataset.gold if dataset.gold.is_absolute() else REPO_ROOT / dataset.gold
    job = load_job(job_path)
    targets = list(job.build_target_records())
    normalize = expand = None
    if dataset.expander:
        module = importlib.import_module(dataset.expander)
        normalize = module.normalize
        target_path = job.base_dir / (job.target.path or "")
        expand = module.make_expander(str(target_path))
    gold: GoldSet = load_gold_csv(gold_path, normalize=normalize, expand=expand)
    labelled = [
        (source, wanted) for source in job.build_source_records() if (wanted := gold.get(source.id))
    ]
    if not labelled:
        raise ValueError(f"{dataset.name}: no source record has a gold id")
    templates = job.build_templates()
    return Loaded(
        name=dataset.name,
        job=job,
        templates=templates,
        targets=targets,
        store=MemoryStore.from_source(targets),
        labelled=labelled,
        queries=[templates.render_queries(source) for source, _ in labelled],
    )


# --- metrics -------------------------------------------------------------------------


def rank_metrics(
    ranked: Sequence[Sequence[str]], wanted: Sequence[frozenset[str]], ks: Sequence[int]
) -> dict[str, float]:
    """recall@k (a gold id in the top k) and MRR (of the first gold id, within DEPTH)."""
    hits = {k: 0 for k in ks}
    reciprocal = 0.0
    for ids, gold in zip(ranked, wanted, strict=True):
        first = next((i for i, rid in enumerate(ids[:DEPTH], start=1) if rid in gold), None)
        if first is not None:
            reciprocal += 1.0 / first
        for k in ks:
            hits[k] += first is not None and first <= k
    n = len(wanted)
    out = {f"recall@{k}": hits[k] / n for k in ks}
    out["mrr"] = reciprocal / n
    return out


async def ranked_ids(
    loaded: Loaded, retrievers: Sequence[Retriever], timeout: float = 120.0
) -> tuple[list[list[str]], int]:
    """The fused ids for every labelled source, and how many records lost a retriever."""
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(source: Record, queries: list[str]) -> tuple[list[str], bool]:
        async with semaphore:
            fused, notes, failed = await retrieve(
                queries, source, retrievers, loaded.store, timeout=timeout
            )
        return [c.id for c in fused], bool(notes) or failed

    outcomes = await asyncio.gather(
        *(one(s, q) for (s, _), q in zip(loaded.labelled, loaded.queries, strict=True))
    )
    return [ids for ids, _ in outcomes], sum(degraded for _, degraded in outcomes)


# --- encoders ------------------------------------------------------------------------


class PrefetchedQueries:
    """Wraps an encoder so every query vector comes from one batched call.

    Identity, document encoding and the vectors themselves are the wrapped encoder's:
    the query texts and settings are exactly what `DenseRetriever.search` would send,
    only grouped into batches instead of one request per query.
    """

    def __init__(self, inner: Encoder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def settings(self) -> dict[str, Any]:
        return dict(getattr(self._inner, "settings", None) or {})

    def prefetch(self, texts: Sequence[str]) -> None:
        wanted = sorted({t for t in texts if t.strip() and t not in self._cache})
        if wanted:
            self._cache.update(zip(wanted, self._inner.encode(wanted, is_query=True), strict=True))

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        if is_query and all(t in self._cache for t in texts):
            return [self._cache[t] for t in texts]
        return self._inner.encode(texts, is_query=is_query)


def native_dimension(
    preset: Preset, *, base_url: str, api_key: str | None, transport: Any = None
) -> int:
    """The model's own vector length, from one call. Only the bake-off probes; a job
    declares its `dimensions`."""
    probe = OpenAIEmbeddingsEncoder(
        base_url,
        preset.model,
        dimensions=1,
        api_key=api_key,
        send_dimensions=False,
        doc_prefix=preset.doc_prefix,
        doc_input_type=preset.doc_input_type,
        transport=transport,
    )
    try:
        return probe.probe_dimension()
    finally:
        probe.close()


def build_encoder(
    preset: Preset,
    *,
    base_url: str,
    api_key: str | None,
    transport: Any = None,
) -> Encoder:
    if preset.local:
        from xwalk.retrieval.dense import SentenceTransformerEncoder

        return SentenceTransformerEncoder(
            preset.model.removeprefix("local:"),
            query_prefix=preset.query_prefix,
            doc_prefix=preset.doc_prefix,
        )
    dimensions = preset.dimensions or native_dimension(
        preset, base_url=base_url, api_key=api_key, transport=transport
    )
    return OpenAIEmbeddingsEncoder(
        base_url,
        preset.model,
        dimensions=dimensions,
        api_key=api_key,
        send_dimensions=preset.send_dimensions,
        truncate=preset.truncate,
        query_prefix=preset.query_prefix,
        doc_prefix=preset.doc_prefix,
        query_input_type=preset.query_input_type,
        doc_input_type=preset.doc_input_type,
        batch_size=preset.batch_size,
        transport=transport,
    )


# --- one run -------------------------------------------------------------------------


@dataclass
class Row:
    dataset: str
    method: str
    model: str
    labelled: int
    alone: dict[str, float] | None = None
    fused: dict[str, float] | None = None
    dimension: int | None = None
    index_action: str | None = None
    degraded_records: int = 0
    requests: int | None = None
    prompt_tokens: int | None = None
    cost: float | None = None
    seconds: float | None = None
    error: str | None = None
    note: str = ""


def _bm25(loaded: Loaded, index_root: Path) -> list[Retriever]:
    built: list[Retriever] = []
    for spec in loaded.job.retrievers:
        if spec.kind != "bm25":
            continue
        built.append(
            BM25Retriever.build(
                loaded.targets,
                loaded.templates,
                index_root / loaded.name / f"bm25-{spec.index_name}",
                name=spec.name or "bm25",
                exact_fields=spec.effective_exact_fields,
                default_limit=spec.limit,
                analyzer=spec.analyzer,
                fuzzy_distance=spec.fuzzy_distance,
            )
        )
    return built


def _dense_index(loaded: Loaded, encoder: Encoder, directory: Path) -> tuple[DenseRetriever, str]:
    expected = index_components(loaded.targets, loaded.templates, encoder)
    if (directory / "xwalk_meta.json").is_file():
        try:
            return (
                DenseRetriever.open(directory, encoder, default_limit=DEPTH, expected=expected),
                "opened",
            )
        except (IndexMismatchError, ValueError):
            pass
    built = DenseRetriever.build(
        loaded.targets, loaded.templates, directory, encoder, default_limit=DEPTH
    )
    return built, "built"


def _usage(encoder: Encoder) -> EmbeddingUsage | None:
    usage = getattr(encoder, "usage", None)
    return usage if isinstance(usage, EmbeddingUsage) else None


def run_preset(
    loaded: Loaded,
    preset: Preset,
    encoder: Encoder,
    bm25: Sequence[Retriever],
    index_root: Path,
    ks: Sequence[int],
) -> Row:
    row = Row(loaded.name, preset.name, preset.model, len(loaded.labelled), note=preset.note)
    inner = encoder
    wrapped = PrefetchedQueries(encoder)
    started = time.monotonic()
    before = _usage(inner)
    try:
        identity = encoder_identity(inner)
        row.dimension = int(identity["dimension"])
        directory = index_root / loaded.name / preset.name
        dense, row.index_action = _dense_index(loaded, wrapped, directory)
        wrapped.prefetch([q for queries in loaded.queries for q in queries])
        wanted = [gold for _, gold in loaded.labelled]
        alone, degraded = asyncio.run(ranked_ids(loaded, [dense]))
        row.alone = rank_metrics(alone, wanted, ks)
        if bm25:
            together, degraded_fused = asyncio.run(ranked_ids(loaded, [*bm25, dense]))
            row.fused = rank_metrics(together, wanted, ks)
            degraded = max(degraded, degraded_fused)
        row.degraded_records = degraded
    except Exception as exc:  # one failing preset must not end the bake-off
        row.error = f"{type(exc).__name__}: {exc}"[:600]
    row.seconds = round(time.monotonic() - started, 2)
    after = _usage(inner)
    if before is not None and after is not None:
        row.requests = after.requests - before.requests
        row.prompt_tokens = (
            None
            if after.prompt_tokens is None or before.prompt_tokens is None
            else after.prompt_tokens - before.prompt_tokens
        )
        row.cost = None if after.cost is None or before.cost is None else after.cost - before.cost
    return row


def bm25_row(loaded: Loaded, bm25: Sequence[Retriever], ks: Sequence[int]) -> Row:
    row = Row(loaded.name, "bm25", "tantivy", len(loaded.labelled), note="the job's own BM25")
    ids, row.degraded_records = asyncio.run(ranked_ids(loaded, bm25))
    row.alone = rank_metrics(ids, [gold for _, gold in loaded.labelled], ks)
    return row


# --- cost estimate -------------------------------------------------------------------


def prices(transport: Any = None) -> dict[str, float]:
    """USD per input token for every OpenRouter embedding model (public, no key)."""
    try:
        with httpx.Client(transport=transport, timeout=30.0) as client:
            response = client.get(f"{OPENROUTER}/embeddings/models")
        data = response.json().get("data", [])
    except (httpx.HTTPError, ValueError, AttributeError):
        return {}
    out: dict[str, float] = {}
    for model in data:
        try:
            out[str(model["id"])] = float(model["pricing"]["prompt"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def estimate_tokens(loaded: Sequence[Loaded], preset: Preset) -> int:
    """A rough count (four characters per token) of what one preset embeds."""
    chars = 0
    for data in loaded:
        chars += sum(len(preset.doc_prefix) + len(t) for t in data.doc_texts)
        chars += sum(len(preset.query_prefix) + len(q) for qs in data.queries for q in qs)
    return math.ceil(chars / 4)


# --- report --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _mean(values: Sequence[float | None]) -> float | None:
    """The mean, or None unless every dataset produced a value."""
    if not values or any(v is None for v in values):
        return None
    known = [v for v in values if v is not None]
    return sum(known) / len(known)


def _metric(row: Row, which: str, key: str) -> float | None:
    found = row.alone if which == "alone" else row.fused
    return None if found is None else found.get(key)


def _total(values: Sequence[float | None]) -> float | None:
    """A sum that is unknown if any part is unknown."""
    return None if any(v is None for v in values) else sum(v or 0 for v in values)


def render_markdown(rows: Sequence[Row], ks: Sequence[int], meta: Mapping[str, Any]) -> str:
    datasets = list(dict.fromkeys(r.dataset for r in rows))
    methods = list(dict.fromkeys(r.method for r in rows))
    k = 5 if 5 in ks else ks[0]
    lines = [
        f"# Embedding bake-off ({meta['date']})",
        "",
        f"Datasets: {', '.join(datasets)}. Endpoint: {meta['base_url']}. Dense depth "
        f"{DEPTH}. Every number below is from this run; none was edited by hand.",
        "",
        "## Mean over datasets",
        "",
        "Alone: the dense retriever by itself (for `bm25`, the job's BM25 by itself). "
        "Fused: the job's BM25 and the dense retriever by reciprocal rank fusion, as a "
        "hybrid job retrieves. A dash is a mean the preset did not finish on every dataset.",
        "",
        f"| method | model | MRR | R@1 | R@{k} | fused MRR | fused R@{k} | tokens | cost $ | "
        "failed datasets |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked: list[tuple[float, str]] = []
    for method in methods:
        mine = [r for r in rows if r.method == method]
        remote = method != "bm25" and not mine[0].model.startswith("local:")
        tokens = _total([r.prompt_tokens for r in mine]) if remote else None
        cost = _total([r.cost for r in mine]) if remote else None
        mrr = _mean([_metric(r, "alone", "mrr") for r in mine])
        line = " | ".join(
            [
                f"| {method}",
                f"`{mine[0].model}`",
                _fmt(mrr),
                _fmt(_mean([_metric(r, "alone", "recall@1") for r in mine])),
                _fmt(_mean([_metric(r, "alone", f"recall@{k}") for r in mine])),
                _fmt(_mean([_metric(r, "fused", "mrr") for r in mine])),
                _fmt(_mean([_metric(r, "fused", f"recall@{k}") for r in mine])),
                ("unknown" if tokens is None else str(int(tokens))) if remote else "—",
                ("unknown" if cost is None else f"{cost:.6f}") if remote else "—",
                f"{sum(r.error is not None for r in mine) or ''} |",
            ]
        )
        ranked.append((-1.0 if mrr is None else mrr, line))
    lines += [line for _, line in sorted(ranked, key=lambda pair: -pair[0])]
    lines.append("")
    for dataset in datasets:
        lines += [
            f"## {dataset}",
            "",
            "| method | dim | "
            + " | ".join(f"R@{c}" for c in ks)
            + " | MRR | fused MRR | index | seconds | error |",
            "|---|---|" + "---|" * len(ks) + "---|---|---|---|---|",
        ]
        for r in (r for r in rows if r.dataset == dataset):
            cells = [_fmt(_metric(r, "alone", f"recall@{c}")) for c in ks]
            error = (r.error or "").replace("|", "/").replace("\n", " ")[:200]
            lines.append(
                f"| {r.method} | {r.dimension or '—'} | {' | '.join(cells)} | "
                f"{_fmt(_metric(r, 'alone', 'mrr'))} | {_fmt(_metric(r, 'fused', 'mrr'))} | "
                f"{r.index_action or '—'} | {_fmt(r.seconds, 1)} | {error} |"
            )
        lines.append("")
    return "\n".join(lines)


# --- entry point ---------------------------------------------------------------------


@dataclass
class Options:
    presets: list[Preset]
    datasets: list[Dataset]
    out: Path
    base_url: str = OPENROUTER
    api_key_env: str = "OPENROUTER_API_KEY"
    ks: tuple[int, ...] = DEFAULT_KS
    max_cost: float | None = 1.0
    transport: Any = None
    log: Callable[[str], None] = field(default=lambda line: print(line, file=sys.stderr))


def load_all(datasets: Sequence[Dataset], log: Callable[[str], None]) -> list[Loaded]:
    """Every dataset that loads. One needing a missing extra (OWL needs
    xwalk[ontology]) is skipped with the reason."""
    from xwalk._extras import MissingExtra

    loaded: list[Loaded] = []
    for dataset in datasets:
        try:
            loaded.append(load_dataset(dataset))
        except MissingExtra as exc:
            log(f"skipping {dataset.name}: {str(exc).splitlines()[0]}")
    if not loaded:
        raise SystemExit("no dataset could be loaded")
    return loaded


def run(options: Options) -> list[Row]:
    loaded = load_all(options.datasets, options.log)
    api_key = os.environ.get(options.api_key_env) or None
    remote = [p for p in options.presets if not p.local]
    if remote and api_key is None:
        raise SystemExit(f"{options.api_key_env} is not set; export it or pass --api-key-env")
    if options.max_cost is not None and remote:
        table = prices(options.transport)
        estimate = sum(estimate_tokens(loaded, p) * table.get(p.model, 0.0) for p in remote)
        unpriced = [p.model for p in remote if p.model not in table]
        options.log(
            f"estimated cost ${estimate:.4f}"
            + (f" (no price for {', '.join(unpriced)})" if unpriced else "")
        )
        if estimate > options.max_cost:
            raise SystemExit(
                f"estimated ${estimate:.4f} exceeds --max-cost {options.max_cost}; "
                "raise it or pick fewer presets"
            )

    index_root = options.out / "index"  # gitignored: vectors are never committed
    rows: list[Row] = []
    bm25 = {data.name: _bm25(data, index_root) for data in loaded}
    for data in loaded:
        if bm25[data.name]:
            rows.append(bm25_row(data, bm25[data.name], options.ks))
    for preset in options.presets:
        try:
            encoder = build_encoder(
                preset,
                base_url=options.base_url,
                api_key=api_key,
                transport=options.transport,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:600]
            options.log(f"{preset.name}: could not start ({message})")
            rows += [
                Row(d.name, preset.name, preset.model, len(d.labelled), error=message)
                for d in loaded
            ]
            continue
        for data in loaded:
            row = run_preset(data, preset, encoder, bm25[data.name], index_root, options.ks)
            status = row.error or (
                f"MRR {_fmt(row.alone['mrr'] if row.alone else None)}, "
                f"fused MRR {_fmt(row.fused['mrr'] if row.fused else None)}"
            )
            options.log(f"{preset.name} on {data.name}: {status}")
            rows.append(row)
    return rows


def write(rows: Sequence[Row], options: Options) -> tuple[Path, Path]:
    options.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": date.today().isoformat(),
        "base_url": options.base_url,
        "ks": list(options.ks),
        "depth": DEPTH,
        "presets": {p.name: asdict(p) for p in options.presets},
        "datasets": {d.name: {"job": str(d.job), "gold": str(d.gold)} for d in options.datasets},
    }
    results = options.out / "results.json"
    results.write_text(
        json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = options.out / "results.md"
    report.write_text(render_markdown(rows, options.ks, meta) + "\n", encoding="utf-8")
    return results, report


def _parse(argv: Sequence[str] | None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.embeddings",
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", action="append", help="preset name; repeatable (default all)")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--job", type=Path, help="a job file of your own (with --gold)")
    parser.add_argument("--gold", type=Path, help="its gold CSV (source_id,gold_ids)")
    parser.add_argument("--out", type=Path, help="results directory")
    parser.add_argument("--base-url", default=OPENROUTER)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--k", type=int, action="append", help="recall cut-off; repeatable")
    parser.add_argument(
        "--max-cost", type=float, default=1.0, help="refuse to start above this estimate (USD)"
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would run; no calls")
    return parser.parse_args(argv), parser


def main(argv: Sequence[str] | None = None) -> int:
    args, parser = _parse(argv)
    if (args.job is None) != (args.gold is None):
        parser.error("--job and --gold go together")
    unknown = [m for m in args.model or [] if m not in PRESETS]
    if unknown:
        parser.error(f"unknown preset {', '.join(unknown)}; known: {', '.join(PRESETS)}")
    presets = [PRESETS[m] for m in args.model] if args.model else list(PRESETS.values())
    if args.job is not None:
        datasets = [Dataset(args.job.stem, args.job.resolve(), args.gold.resolve())]
    else:
        datasets = [DATASETS[d] for d in args.dataset] if args.dataset else list(DATASETS.values())
    ks = tuple(sorted(set(args.k))) if args.k else DEFAULT_KS
    if any(k < 1 or k > DEPTH for k in ks):
        parser.error(f"every --k must be between 1 and {DEPTH}")
    out = args.out or REPO_ROOT / "benchmarks" / "results" / f"embeddings-{date.today()}"
    options = Options(presets, datasets, out, args.base_url, args.api_key_env, ks, args.max_cost)

    if args.dry_run:
        loaded = load_all(datasets, options.log)
        table = prices()
        for d in loaded:
            print(f"dataset {d.name}: {len(d.targets)} targets, {len(d.labelled)} labelled")
        for p in presets:
            tokens = estimate_tokens(loaded, p)
            price = table.get(p.model)
            cost = "local" if p.local else "unknown" if price is None else f"${tokens * price:.5f}"
            print(f"  {p.name:20} {p.model:40} ~{tokens} tokens  {cost}  {p.note}")
        return 0

    rows = run(options)
    results, report = write(rows, options)
    print(report.read_text(encoding="utf-8"))
    print(f"wrote {results} and {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

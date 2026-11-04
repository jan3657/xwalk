"""Load the NER demo's ontologies into Postgres (Neon) for `NER_SITE_BACKEND=neon`.

    python ner-site/tools/load_neon.py --dry-run                 # samples: counts, sizes, cost
    python ner-site/tools/load_neon.py                           # load the four samples
    python ner-site/tools/load_neon.py --source full --dry-run   # downloads, then estimates
    python ner-site/tools/load_neon.py --source full --only ncbi-gene --limit 2000
    python ner-site/tools/load_neon.py --benchmark-upload 5000   # synthetic vectors, then dropped

Needs `DATABASE_URL` (a Neon connection string; queries go over HTTPS) and, unless
`--dry-run`, `OPENROUTER_API_KEY` for the document embeddings (the default Qwen3 encoder,
`--dimensions` wide, stored as `halfvec`). Writes records, BM25 text and vectors through
`xwalk.retrieval.postgres.PostgresIndexWriter`, with the site's own doc template, exact
fields and per-name vectors, so the stored identity is what `ner_site.indexes` expects.

Resumable: every write is an upsert; a rerun re-embeds only vectors that are not stored
yet with the same text (a crash loses at most the batches in flight), and an ontology
is marked ready only after its last write. A `--source full` load refuses to start when
its estimated size plus the database's current size exceeds `--max-bytes` (400 MB, the
Neon free plan's 0.5 GB with headroom) unless the limit is raised explicitly.

Downloaded files are data: they are kept in `--cache-dir`, parsed, never executed.
Plan: docs/ner-site/NEON.md.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import math
import os
import random
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SITE_ROOT = Path(__file__).resolve().parents[1]
if str(SITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SITE_ROOT))

import httpx  # noqa: E402
from ner_site.indexes import (  # noqa: E402
    BM25_NAME,
    DEFAULT_LIMIT,
    DENSE_NAME,
    EXACT_FIELDS,
    NAME_FIELDS,
    NEON_SCHEMA,
)
from ner_site.ontologies import FULL_CATALOGS, ROUTES, TEMPLATES, load_terms  # noqa: E402

from xwalk.records import Record  # noqa: E402
from xwalk.retrieval.dense import Encoder  # noqa: E402
from xwalk.retrieval.embeddings_api import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    EmbeddingAPIError,
    EmbeddingUsage,
    default_embeddings_encoder,
)
from xwalk.retrieval.postgres import (  # noqa: E402
    NeonHTTPExecutor,
    PostgresError,
    PostgresIndexWriter,
    SQLExecutor,
    VectorRow,
    bm25_components,
    pgvector_components,
    terms_components,
    vector_literal,
)

DEFAULT_CACHE = Path(
    "/tmp/claude-0/-home-user-xwalk/b81f059f-c412-5c1f-9791-4c9353b2141e/scratchpad/onto-cache"
)
DEFAULT_DIMENSIONS = 1024
DEFAULT_MAX_BYTES = 400_000_000
DATABASE_ENV = "DATABASE_URL"
KINDS = ("chebi", "ctd-disease", "ncbi-gene", "foodon")
# Entity type of each kind (ROUTES / FULL_CATALOGS are keyed by type).
TYPES = {"chebi": "chemical", "ctd-disease": "disease", "ncbi-gene": "gene", "foodon": "food"}

# Size model, measured on Neon on 2026-11-04 (docs/ner-site/NEON.md, "Sizes"). A halfvec
# row of 1024 dimensions does not fit Postgres's 2 KB inline limit, so it goes to TOAST,
# and the HNSW graph stores the vector again: 6,000 synthetic vectors took 5,677 bytes
# each at 1024 dimensions and 2,615 at 512 (heap, TOAST, keys and HNSW), a line of
# 5.98 bytes a dimension minus 446. The four samples' terms took 2,125 bytes each
# (123.6 doc characters on average; row, tsvector, exact forms, BM25 and GIN postings,
# and four partial indexes' fixed pages). Rough by design: it guards a quota.
VECTOR_BYTES_PER_DIM = 5.98
VECTOR_BYTES_OFFSET = -446.0
TERM_OVERHEAD_BYTES = 1300
TERM_TEXT_FACTOR = 6.6
# Qwen3-Embedding 8B through OpenRouter as reported on 2026-11-04: the samples' 3,132 texts
# (165,860 characters) were 65,316 tokens for $0.000653 (2.54 characters a token), a
# 2,000-gene slice's 10,964 texts 94,773 tokens for $0.000948 (1.95: symbols tokenise
# poorly). So $0.01 a million tokens, and 2.0 characters a token errs high. Estimates
# only; the real spend is printed as reported.
CHARS_PER_TOKEN = 2.0
DOLLARS_PER_MTOK = 0.01


def vector_bytes(dimensions: int) -> float:
    """Stored bytes of one vector (heap, TOAST, keys, HNSW), from the measured line;
    never less than its two halfvec copies."""
    return max(VECTOR_BYTES_PER_DIM * dimensions + VECTOR_BYTES_OFFSET, 4.0 * dimensions)


SOURCES: dict[str, list[str]] = {
    "ctd-disease": ["https://ctdbase.org/reports/CTD_diseases.tsv.gz"],
    "chebi": ["https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo.gz"],
    "foodon": ["https://raw.githubusercontent.com/FoodOntology/foodon/master/foodon.owl"],
    "ncbi-gene": [
        f"https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/{species}.gene_info.gz"
        for species in ("Homo_sapiens", "Mus_musculus", "Rattus_norvegicus")
    ],
}
ORGANISMS = {"9606": "Homo sapiens", "10090": "Mus musculus", "10116": "Rattus norvegicus"}


def slug_for(kind: str, source: str) -> str:
    return (ROUTES if source == "samples" else FULL_CATALOGS)[TYPES[kind]]


# --- sources -----------------------------------------------------------------------


def download(url: str, cache_dir: Path, *, client: httpx.Client | None = None) -> Path:
    """`url` into `cache_dir` (kept: a second run reuses it). Written to a temporary
    name first, so an interrupted download is never mistaken for a complete one."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / url.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size > 0:
        return target
    partial = target.with_name(target.name + ".part")
    own = client is None
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
    finally:
        if own:
            http.close()
    partial.replace(target)
    return target


def _gunzip(path: Path) -> Path:
    plain = path.with_suffix("")
    if not (plain.exists() and plain.stat().st_size > 0):
        partial = plain.with_name(plain.name + ".part")
        with gzip.open(path, "rb") as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        partial.replace(plain)
    return plain


def _split(value: str) -> list[str]:
    if value.strip() in {"", "-"}:
        return []
    return list(dict.fromkeys(s.strip() for s in value.split("|") if s.strip()))


def ctd_records(path: Path) -> Iterator[Record]:
    """CTD_diseases.tsv(.gz): DiseaseName, DiseaseID, AltDiseaseIDs, Definition, ...,
    Synonyms (`|`-separated). Comment lines start with `#`."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for row in rows:
            if len(row) < 8 or not row[1].strip():
                continue
            yield Record(
                row[1].strip(),
                {"label": row[0].strip(), "synonyms": _split(row[7]), "definition": row[3].strip()},
            )


def gene_records(paths: Sequence[Path]) -> Iterator[Record]:
    """NCBI gene_info(.gz): #tax_id, GeneID, Symbol, LocusTag, Synonyms (`|`, `-` for
    none), dbXrefs, chromosome, map_location, description, ..."""
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 9:
                    continue
                tax_id = cols[0].strip()
                description = cols[8].strip()
                fields: dict[str, Any] = {
                    "label": cols[2].strip(),
                    "synonyms": _split(cols[4]),
                    "definition": "" if description == "-" else description,
                    "organism": ORGANISMS.get(tax_id, tax_id),
                    "tax_id": tax_id,
                }
                yield Record(f"NCBIGene:{cols[1].strip()}", fields)


def _plain(record: Record) -> Record:
    """Only the fields the samples carry (an OBO or OWL record also has parents)."""
    fields = record.fields
    return Record(
        record.id,
        {
            "label": str(fields.get("label") or ""),
            "synonyms": list(dict.fromkeys(str(s) for s in fields.get("synonyms") or [])),
            "definition": str(fields.get("definition") or ""),
        },
    )


def full_records(kind: str, cache_dir: Path) -> Iterator[Record]:
    paths = [download(url, cache_dir) for url in SOURCES[kind]]
    if kind == "ctd-disease":
        yield from ctd_records(paths[0])
    elif kind == "ncbi-gene":
        yield from gene_records(paths)
    elif kind == "chebi":
        from xwalk.sources.ontology import obo_source

        for record in obo_source(_gunzip(paths[0]), id_prefix="CHEBI:"):
            yield _plain(record)
    elif kind == "foodon":
        from xwalk.sources.ontology import owl_source

        for record in owl_source(paths[0], id_prefix="FOODON:"):
            yield _plain(record)
    else:  # pragma: no cover - argparse restricts it
        raise KeyError(kind)


def records_for(kind: str, source: str, cache_dir: Path, limit: int | None) -> list[Record]:
    if source == "samples":
        records = load_terms(slug_for(kind, source))
    else:
        records = []
        seen: set[str] = set()
        for record in full_records(kind, cache_dir):
            if record.id in seen:
                continue
            seen.add(record.id)
            records.append(record)
    return records[:limit] if limit is not None else records


# --- estimates ---------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    slug: str
    records: int
    vectors: int
    text_chars: int  # embedded text (docs and names)
    term_bytes: int
    vector_bytes: int

    @property
    def bytes(self) -> int:
        return self.term_bytes + self.vector_bytes

    @property
    def tokens(self) -> int:
        return math.ceil(self.text_chars / CHARS_PER_TOKEN)

    @property
    def cost(self) -> float:
        return self.tokens / 1e6 * DOLLARS_PER_MTOK


def estimate(slug: str, records: Sequence[Record], dimensions: int) -> Estimate:
    rows = PostgresIndexWriter.vector_rows(records, TEMPLATES, name_fields=NAME_FIELDS)
    text_chars = sum(len(r.text) for r in rows)
    doc_chars = sum(len(r.text) for r in rows if r.kind == "doc")
    term_bytes = int(len(records) * TERM_OVERHEAD_BYTES + doc_chars * TERM_TEXT_FACTOR)
    per_vector = vector_bytes(dimensions)
    stored = int(sum(per_vector + len(r.text) + len(r.record_id) for r in rows))
    return Estimate(slug, len(records), len(rows), text_chars, term_bytes, stored)


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def format_estimate(e: Estimate) -> str:
    return (
        f"{e.slug}: {e.records:,} records, {e.vectors:,} vectors, "
        f"~{e.tokens:,} tokens (~${e.cost:.4f}), ~{_mb(e.bytes)} "
        f"(terms {_mb(e.term_bytes)}, vectors {_mb(e.vector_bytes)})"
    )


# --- loading -----------------------------------------------------------------------


@dataclass
class Progress:
    """Counters shared by the worker threads; `vector_bytes`/`upload_s` measure the
    vector upserts alone (request bytes, seconds summed over requests)."""

    embedded: int = 0
    skipped: int = 0
    vector_bytes: int = 0
    upload_s: float = 0.0
    embed_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _payload_bytes(rows: Sequence[VectorRow], vectors: Sequence[Sequence[float]]) -> int:
    return sum(
        len(vector_literal(v)) + len(r.text) + len(r.record_id) + 12
        for r, v in zip(rows, vectors, strict=True)
    )


def load_one(
    writer: PostgresIndexWriter,
    slug: str,
    records: Sequence[Record],
    encoder: Encoder,
    *,
    batch: int,
    concurrency: int,
    progress: Progress,
    out: Callable[[str], None],
    index_writer: PostgresIndexWriter | None = None,
) -> None:
    """Load one ontology: meta `loading`, terms, missing vectors, prune, indexes, meta
    `ready`. Safe to rerun at any point. `index_writer` builds the indexes (one long
    request; `writer` when not given)."""
    terms = terms_components(records)
    bm25 = bm25_components(records, TEMPLATES, exact_fields=EXACT_FIELDS)
    dense = pgvector_components(records, TEMPLATES, encoder, name_fields=NAME_FIELDS)
    for kind, components, name in (
        ("terms", terms, "terms"),
        ("bm25", bm25, BM25_NAME),
        ("dense", dense, DENSE_NAME),
    ):
        writer.write_meta(slug, kind, components, name=name, status="loading")

    for start in range(0, len(records), 1000):
        writer.upsert_terms(
            slug, records[start : start + 1000], TEMPLATES, exact_fields=EXACT_FIELDS
        )
    out(f"{slug}: {len(records):,} terms written")

    rows = writer.vector_rows(records, TEMPLATES, name_fields=NAME_FIELDS)
    started = time.monotonic()
    embedded_before, skipped_before = progress.embedded, progress.skipped

    def embed_and_store(chunk: list[VectorRow]) -> int:
        t0 = time.monotonic()
        vectors = encoder.encode([r.text for r in chunk], is_query=False)
        t1 = time.monotonic()
        writer.upsert_vectors(slug, chunk, vectors)
        t2 = time.monotonic()
        with progress.lock:
            progress.embedded += len(chunk)
            progress.embed_s += t1 - t0
            progress.upload_s += t2 - t1
            progress.vector_bytes += _payload_bytes(chunk, vectors)
        return len(chunk)

    window = max(batch * concurrency * 4, 2000)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="load") as pool:
        for start in range(0, len(rows), window):
            part = rows[start : start + window]
            present = writer.present_vectors(slug, part)
            missing = [r for r in part if (r.record_id, r.ord) not in present]
            progress.skipped += len(part) - len(missing)
            futures: list[Future[int]] = [
                pool.submit(embed_and_store, missing[i : i + batch])
                for i in range(0, len(missing), batch)
            ]
            for future in futures:
                future.result()
            done = start + len(part)
            embedded = progress.embedded - embedded_before
            skipped = progress.skipped - skipped_before
            rate = embedded / max(time.monotonic() - started, 1e-9)
            out(
                f"{slug}: vectors {done:,}/{len(rows):,} (embedded {embedded:,}, "
                f"already stored {skipped:,}; {rate:,.0f}/s)"
            )

    deleted_terms, deleted_vectors = writer.prune(
        slug, [r.id for r in records], [(r.record_id, r.ord) for r in rows]
    )
    if deleted_terms or deleted_vectors:
        out(f"{slug}: pruned {deleted_terms} stale terms, {deleted_vectors} stale vectors")
    (index_writer or writer).create_indexes(slug)
    writer.analyze()
    count = len(records)
    writer.write_meta(slug, "terms", terms, name="terms", status="ready", record_count=count)
    writer.write_meta(
        slug,
        "bm25",
        bm25,
        name=BM25_NAME,
        status="ready",
        default_limit=DEFAULT_LIMIT,
        record_count=count,
    )
    writer.write_meta(
        slug,
        "dense",
        dense,
        name=DENSE_NAME,
        status="ready",
        default_limit=DEFAULT_LIMIT,
        record_count=count,
        vector_count=len(rows),
        dimension=encoder.dimension,
    )
    out(f"{slug}: ready ({count:,} records, {len(rows):,} vectors)")


def _show(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6f}" if isinstance(value, float) else f"{value:,}"


def format_usage(usage: EmbeddingUsage | None) -> str:
    if usage is None:
        return "embeddings: usage not reported by the encoder"
    return (
        f"embeddings: requests {_show(usage.requests)}, texts {_show(usage.texts)}, "
        f"tokens {_show(usage.prompt_tokens)}, cost {_show(usage.cost)} (as reported)"
    )


def benchmark_upload(
    executor: SQLExecutor,
    *,
    schema: str,
    count: int,
    dimensions: int,
    batch: int,
    concurrency: int,
    out: Callable[[str], None],
) -> float:
    """Upload `count` random unit vectors into a throwaway schema, drop it, and return
    MB/s of request payload. No embedding call; nothing is left behind."""
    writer = PostgresIndexWriter(executor, schema=schema, dimension=dimensions)
    rng = random.Random(0)
    try:
        writer.create_schema()
        rows = [VectorRow(f"BENCH:{i // 3}", i % 3, "name", f"text {i}") for i in range(count)]

        def unit() -> list[float]:
            v = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
            n = math.sqrt(sum(x * x for x in v))
            return [x / n for x in v]

        vectors = [unit() for _ in rows]
        payload = _payload_bytes(rows, vectors)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            jobs = [
                pool.submit(
                    writer.upsert_vectors, "bench", rows[i : i + batch], vectors[i : i + batch]
                )
                for i in range(0, count, batch)
            ]
            for job in jobs:
                job.result()
        seconds = time.monotonic() - started
        rate = payload / 1e6 / seconds
        out(
            f"upload benchmark: {count:,} vectors x {dimensions} dims, batch {batch}, "
            f"concurrency {concurrency}: {_mb(payload)} in {seconds:.1f} s = {rate:.2f} MB/s, "
            f"{count / seconds:,.0f} vectors/s"
        )
        before = writer.sizes().get("term_vectors", 0)
        built = time.monotonic()
        writer.create_indexes("bench")
        index_s = time.monotonic() - built
        after = writer.sizes().get("term_vectors", 0)
        out(
            f"stored: {before / count:,.0f} bytes a vector before the HNSW index, "
            f"{after / count:,.0f} with it; HNSW build {index_s:.1f} s "
            f"({count / max(index_s, 1e-9):,.0f} vectors/s)"
        )
        return rate
    finally:
        writer.drop_schema()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="load_neon.py", description="Load the NER demo's ontologies into Neon Postgres."
    )
    parser.add_argument("--source", choices=("samples", "full"), default="samples")
    parser.add_argument(
        "--only",
        action="append",
        choices=KINDS,
        metavar="KIND",
        help=f"load only this ontology (repeatable; one of {', '.join(KINDS)})",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N records per ontology")
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--schema", default=NEON_SCHEMA)
    parser.add_argument("--batch", type=int, default=128, help="texts per embeddings request")
    parser.add_argument("--concurrency", type=int, default=4, help="batches in flight")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="refuse a --source full load whose estimate plus the current database size "
        "exceeds this (default 400 MB: the Neon free plan)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="counts, size, token and cost estimates; no database write, no embedding call",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="remove the selected ontologies (rows, indexes, identities) instead of loading",
    )
    parser.add_argument(
        "--benchmark-upload",
        type=int,
        default=0,
        metavar="N",
        help="upload N random vectors into a throwaway schema, report MB/s, drop it",
    )
    return parser


def _executor(env: Mapping[str, str]) -> SQLExecutor:
    return NeonHTTPExecutor.from_env(env, name=DATABASE_ENV, timeout=120.0, max_retries=3)


# An index build is one long statement: give it an hour and never retry it, because a
# retried CREATE INDEX starts over and the abandoned request cancels the build (ChEBI's
# HNSW index outlived the 120 s request four times on 4 November 2026).
INDEX_TIMEOUT_S = 3600.0


def _index_executor(env: Mapping[str, str]) -> SQLExecutor:
    return NeonHTTPExecutor.from_env(env, name=DATABASE_ENV, timeout=INDEX_TIMEOUT_S, max_retries=0)


def _encoder(api_key: str, dimensions: int) -> Encoder:
    return default_embeddings_encoder(api_key=api_key, dimensions=dimensions)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    executor_factory: Callable[[Mapping[str, str]], SQLExecutor] | None = None,
    encoder_factory: Callable[[str, int], Encoder] | None = None,
    out: Callable[[str], None] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    env = env if env is not None else os.environ
    say = out or (lambda line: print(line, flush=True))
    kinds = [k for k in KINDS if not args.only or k in args.only]
    if args.limit is not None and args.limit < 1:
        say("error: --limit must be at least 1")
        return 2
    if not 1 <= args.dimensions <= 4000 or args.batch < 1 or args.concurrency < 1:
        say("error: --dimensions must be 1..4000, --batch and --concurrency at least 1")
        return 2

    database_url = (env.get(DATABASE_ENV) or "").strip()
    need_db = not args.dry_run or args.benchmark_upload
    if need_db and not database_url:
        say(f"error: {DATABASE_ENV} is not set (use --dry-run for estimates without it)")
        return 2

    if args.benchmark_upload:
        executor = (executor_factory or _executor)(env)
        benchmark_upload(
            executor,
            schema=f"{args.schema}_bench",
            count=args.benchmark_upload,
            dimensions=args.dimensions,
            batch=args.batch,
            concurrency=args.concurrency,
            out=say,
        )
        return 0

    if args.drop:
        if not database_url:
            say(f"error: {DATABASE_ENV} is not set")
            return 2
        executor = (executor_factory or _executor)(env)
        writer = PostgresIndexWriter(executor, schema=args.schema, dimension=args.dimensions)
        for kind in kinds:
            slug = slug_for(kind, args.source)
            writer.drop_ontology(slug)
            say(f"{slug}: dropped")
        writer.vacuum()
        with contextlib.suppress(PostgresError):
            say("sizes: " + ", ".join(f"{k} {_mb(v)}" for k, v in writer.sizes().items()))
        return 0

    loaded: list[tuple[str, list[Record], Estimate]] = []
    for kind in kinds:
        slug = slug_for(kind, args.source)
        say(f"{slug}: reading {'bundled sample' if args.source == 'samples' else 'sources'}")
        records = records_for(kind, args.source, args.cache_dir, args.limit)
        loaded.append((slug, records, estimate(slug, records, args.dimensions)))
    total = Estimate(
        "total",
        sum(e.records for _, _, e in loaded),
        sum(e.vectors for _, _, e in loaded),
        sum(e.text_chars for _, _, e in loaded),
        sum(e.term_bytes for _, _, e in loaded),
        sum(e.vector_bytes for _, _, e in loaded),
    )
    for _, _, e in loaded:
        say(format_estimate(e))
    say(format_estimate(total) + f" at halfvec({args.dimensions})")
    if args.dry_run:
        say("dry run: nothing written, nothing embedded")
        return 0

    api_key = (env.get(DEFAULT_API_KEY_ENV) or "").strip()
    if not api_key:
        say(f"error: {DEFAULT_API_KEY_ENV} is not set; document embeddings need it")
        return 2

    executor = (executor_factory or _executor)(env)
    writer = PostgresIndexWriter(executor, schema=args.schema, dimension=args.dimensions)
    # Tests inject one executor for everything; a real load builds indexes on its own.
    index_writer = (
        None
        if executor_factory is not None
        else PostgresIndexWriter(
            _index_executor(env), schema=args.schema, dimension=args.dimensions
        )
    )
    try:
        current = writer.sizes().get("database", 0)
    except PostgresError as exc:
        say(f"error: the database could not be queried: {exc}")
        return 1
    if args.source == "full" and current + total.bytes > args.max_bytes:
        say(
            f"error: refusing: estimated {_mb(total.bytes)} plus the database's current "
            f"{_mb(current)} exceeds --max-bytes {_mb(args.max_bytes)}; raise --max-bytes "
            "explicitly once the plan allows it, or load fewer records (--only, --limit)"
        )
        return 3

    encoder = (encoder_factory or _encoder)(api_key, args.dimensions)
    progress = Progress()
    started = time.monotonic()
    code = 0
    try:
        writer.create_schema()
        for slug, records, _ in loaded:
            load_one(
                writer,
                slug,
                records,
                encoder,
                batch=args.batch,
                concurrency=args.concurrency,
                progress=progress,
                out=say,
                index_writer=index_writer,
            )
    except (EmbeddingAPIError, PostgresError, ValueError) as exc:
        say(f"error: {type(exc).__name__}: {exc} (rerun to resume)")
        code = 1
    finally:
        usage = getattr(encoder, "usage", None)
        say(format_usage(usage if isinstance(usage, EmbeddingUsage) else None))
        stats = getattr(executor, "stats", None)
        if stats is not None:
            say(
                f"database: {stats.requests:,} requests, {_mb(stats.bytes_sent)} sent, "
                f"{_mb(stats.bytes_received)} received"
            )
        if progress.upload_s > 0:
            say(
                f"vector upload: {_mb(progress.vector_bytes)} in {progress.upload_s:.1f} "
                f"request-seconds ({progress.vector_bytes / 1e6 / progress.upload_s:.2f} MB/s "
                f"per request; {args.concurrency} in flight); embedding "
                f"{progress.embed_s:.1f} request-seconds"
            )
        say(f"elapsed: {time.monotonic() - started:.1f} s")
        with contextlib.suppress(PostgresError):
            say("sizes: " + ", ".join(f"{k} {_mb(v)}" for k, v in writer.sizes().items()))
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

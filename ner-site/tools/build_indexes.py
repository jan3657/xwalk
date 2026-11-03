"""Build the committed dense indexes for the NER demo site.

    OPENROUTER_API_KEY=... python ner-site/tools/build_indexes.py [--out DIR] [--only SLUG]
    python ner-site/tools/build_indexes.py --dry-run     # counts only; no key, no calls

Writes ``<out>/<slug>/dense/`` (vectors.f32, record_ids.json, xwalk_meta.json) with the
default Qwen3 encoder through OpenRouter, the same doc template and name fields that
``ner_site.indexes`` opens them with, and the retriever name ``dense-qwen3``. Prints what
each ontology cost as the provider reported it ("unknown" when it reported nothing).
Plan: docs/ner-site/PLAN.md section 5.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]
if str(SITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SITE_ROOT))

from ner_site.indexes import DEFAULT_LIMIT, DENSE_NAME, NAME_FIELDS  # noqa: E402
from ner_site.ontologies import ROUTES, TEMPLATES, load_terms  # noqa: E402

from xwalk.retrieval.dense import DenseRetriever, Encoder, record_names  # noqa: E402
from xwalk.retrieval.embeddings_api import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    EmbeddingAPIError,
    EmbeddingUsage,
    default_embeddings_encoder,
)

DEFAULT_OUT = SITE_ROOT / "indexes"
SLUGS: tuple[str, ...] = tuple(ROUTES.values())


@dataclass(frozen=True)
class BuildReport:
    """One ontology's build. Usage fields are deltas of the encoder's own usage over this
    build; None means the encoder (or the provider) did not report it."""

    slug: str
    records: int
    vectors: int  # one per record (the doc) plus one per distinct label/synonym
    requests: int | None
    texts: int | None
    prompt_tokens: int | None
    cost: float | None


def counts(slug: str) -> tuple[int, int]:
    """(records, vectors) for one ontology, without encoding anything."""
    records = load_terms(slug)
    names = sum(len(record_names(r, NAME_FIELDS)) for r in records)
    return len(records), len(records) + names


def _usage(encoder: Encoder) -> EmbeddingUsage | None:
    usage = getattr(encoder, "usage", None)
    return usage if isinstance(usage, EmbeddingUsage) else None


def _delta(after: int | float | None, before: int | float | None) -> int | float | None:
    if after is None or before is None:
        return None
    return after - before


def build_one(out: Path, slug: str, encoder: Encoder) -> BuildReport:
    records = load_terms(slug)
    before = _usage(encoder)
    DenseRetriever.build(
        records,
        TEMPLATES,
        out / slug / "dense",
        encoder,
        name=DENSE_NAME,
        default_limit=DEFAULT_LIMIT,
        name_fields=NAME_FIELDS,
    )
    after = _usage(encoder)
    n_records, n_vectors = counts(slug)
    if before is None or after is None:
        return BuildReport(slug, n_records, n_vectors, None, None, None, None)
    requests = _delta(after.requests, before.requests)
    texts = _delta(after.texts, before.texts)
    tokens = _delta(after.prompt_tokens, before.prompt_tokens)
    return BuildReport(
        slug,
        n_records,
        n_vectors,
        None if requests is None else int(requests),
        None if texts is None else int(texts),
        None if tokens is None else int(tokens),
        _delta(after.cost, before.cost),
    )


def build_all(
    out: Path, encoder: Encoder, *, only: Sequence[str] | None = None
) -> list[BuildReport]:
    """Build every routed ontology's dense index (or those in `only`) into `out`."""
    slugs = [s for s in SLUGS if not only or s in only]
    return [build_one(Path(out), slug, encoder) for slug in slugs]


def _show(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def format_report(report: BuildReport) -> str:
    return (
        f"{report.slug}: {report.records} records, {report.vectors} vectors; "
        f"requests {_show(report.requests)}, texts {_show(report.texts)}, "
        f"tokens {_show(report.prompt_tokens)}, cost {_show(report.cost)} (as reported)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_indexes.py",
        description="Build the NER demo site's dense indexes with the default Qwen3 encoder.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=SLUGS,
        metavar="SLUG",
        help=f"build only this ontology (repeatable; one of {', '.join(SLUGS)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print record and vector counts; no key needed, nothing is sent or written",
    )
    return parser


def _default_factory(api_key: str) -> Encoder:
    return default_embeddings_encoder(api_key=api_key)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    encoder_factory: Callable[[str], Encoder] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    env = env if env is not None else os.environ
    only: list[str] | None = args.only
    slugs = [s for s in SLUGS if not only or s in only]

    if args.dry_run:
        total = 0
        for slug in slugs:
            n_records, n_vectors = counts(slug)
            total += n_vectors
            print(f"{slug}: {n_records} records, {n_vectors} vectors to embed")
        print(f"total: {total} texts; dry run, nothing sent")
        return 0

    api_key = env.get(DEFAULT_API_KEY_ENV, "").strip()
    if not api_key:
        print(
            f"error: {DEFAULT_API_KEY_ENV} is not set; the dense indexes are built with the "
            "default Qwen3 encoder through OpenRouter and need the key "
            "(use --dry-run to see the counts without it)",
            file=sys.stderr,
        )
        return 2

    encoder = (encoder_factory or _default_factory)(api_key)
    try:
        for slug in slugs:
            print(format_report(build_one(Path(args.out), slug, encoder)), flush=True)
    except EmbeddingAPIError as exc:
        print(f"error: embeddings request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

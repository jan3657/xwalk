"""Retrieval quality of the NER site's own indexes on the pilot gold sets.

The site's four sample ontologies are the pilots' target catalogs (CHEBI, CTD diseases,
NCBI Gene, FoodOn; about 200 targets each), so the pilot gold sets measure exactly what
the site retrieves. For each backend (``bundled``: Tantivy BM25 and the committed dense
index; ``neon``: lakebase BM25 and pgvector HNSW in Postgres) every labelled mention is
searched the way ``ner_site.mapping`` searches it (the site's query template over the
bare mention, each retriever at the site's depth of 20), and BM25 alone, dense alone and
both fused by reciprocal rank (``xwalk.retrieve``) are scored: recall@k and MRR of the
first gold id. No LLM is called; each backend costs one query embedding per mention.

    python -m benchmarks.site_retrieval --backend bundled --backend neon \\
        --out benchmarks/results/site-retrieval-2026-11-04

``neon`` needs DATABASE_URL (queried over Neon's HTTPS endpoint) and both need
OPENROUTER_API_KEY for the query embeddings. ``--catalog full`` searches the full
catalogs instead (once ``ner-site/tools/load_neon.py --source full`` has loaded them);
the gold ids are the same, the distractors are not (pass ``--schema`` and
``--dimensions`` for the full load; NLM Gene is skipped there, genes not being routed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from benchmarks.embeddings import DATASETS, load_dataset, rank_metrics
from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest
from xwalk.retrieve import retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ner-site"))

from ner_site import indexes as site_indexes  # noqa: E402
from ner_site.ontologies import FULL_ROUTES, ROUTES, TEMPLATES  # noqa: E402

KS = (1, 5, 10, 20)
TYPE_OF = {"chebi": "chemical", "ncbi_disease": "disease", "nlm_gene": "gene"}
TYPE_OF["cafeteria_fcd"] = "food"
CONCURRENCY = 8


class _Fixed:
    """A retriever that answers with hits already fetched, so fusion costs no request."""

    def __init__(self, name: str, hits: Sequence[RetrievalHit]) -> None:
        self._name = name
        self._hits = list(hits)

    @property
    def name(self) -> str:
        return self._name

    @property
    def fingerprint(self) -> str:
        return self._name

    @property
    def default_limit(self) -> int:
        return len(self._hits) or 1

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return self._hits


def open_backend(
    backend: str, catalog: str, schema: str, dimensions: int
) -> Mapping[str, site_indexes.OntologyIndex]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if backend == "bundled":
        cache = Path(os.environ.get("NER_SITE_CACHE_DIR", "/tmp/xwalk-ner-site-eval"))
        return site_indexes.open_indexes(
            cache_dir=cache, dense_dir=REPO_ROOT / "ner-site" / "indexes", api_key=api_key
        )
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("--backend neon needs DATABASE_URL")
    return site_indexes.open_neon_indexes(
        executor=site_indexes.make_executor(url),
        api_key=api_key,
        schema=schema,
        dimensions=dimensions,
        catalog="full" if catalog == "full" else "samples",
    )


async def evaluate(
    index: site_indexes.OntologyIndex, queries: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """Ranked ids per retriever and fused, for each (source id, mention)."""
    by_kind = {
        ("bm25" if r.name == site_indexes.BM25_NAME else "dense"): r for r in index.retrievers
    }
    semaphore = asyncio.Semaphore(CONCURRENCY)
    timings: dict[str, list[float]] = {kind: [] for kind in by_kind}
    errors: dict[str, int] = {kind: 0 for kind in by_kind}

    async def one(source_id: str, mention: str) -> dict[str, list[str]]:
        source = Record(source_id, {"mention": mention})
        query = TEMPLATES.render_query(source)
        out: dict[str, list[str]] = {}
        fixed = []
        async with semaphore:
            for kind, retriever in by_kind.items():
                started = time.perf_counter()
                try:
                    hits = await retriever.search(
                        SearchRequest(
                            text=query, limit=retriever.default_limit, source_record=source
                        )
                    )
                except Exception:  # noqa: BLE001 - counted; the mention scores as a miss
                    errors[kind] += 1
                    hits = []
                timings[kind].append(time.perf_counter() - started)
                out[kind] = [h.record_id for h in hits]
                fixed.append(_Fixed(retriever.name, hits))
        if len(fixed) == 2:
            candidates, _, _ = await retrieve([query], source, fixed, index.store, timeout=60.0)
            out["fused"] = [c.id for c in candidates]
        return out

    ranked = await asyncio.gather(*(one(sid, m) for sid, m in queries))
    return {"ranked": ranked, "timings": timings, "errors": errors}


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", action="append", choices=("bundled", "neon"))
    parser.add_argument("--catalog", choices=("samples", "full"), default="samples")
    parser.add_argument("--schema", default=site_indexes.NEON_SCHEMA, help="neon schema")
    parser.add_argument(
        "--dimensions", type=int, default=site_indexes.NEON_DIMENSIONS, help="neon vectors"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    backends = args.backend or ["bundled", "neon"]
    if args.catalog == "full" and "bundled" in backends:
        parser.error("--catalog full needs --backend neon only")
    routes = FULL_ROUTES if args.catalog == "full" else ROUTES

    loaded = {name: load_dataset(DATASETS[name]) for name in TYPE_OF}
    results: dict[str, Any] = {
        "date": date.today().isoformat(),
        "catalog": args.catalog,
        "ks": list(KS),
        "depth": site_indexes.DEFAULT_LIMIT,
        "neon": {"schema": args.schema, "dimensions": args.dimensions},
        "backends": {},
    }
    for backend in backends:
        opened = open_backend(backend, args.catalog, args.schema, args.dimensions)
        per: dict[str, Any] = {}
        for name, data in loaded.items():
            if TYPE_OF[name] not in routes:  # genes are not routed with the full catalogs
                continue
            slug = routes[TYPE_OF[name]]
            index = opened[slug]
            status = {k: s.model_dump() for k, s in index.status.items()}
            queries = [(source.id, str(source.fields["mention"])) for source, _ in data.labelled]
            wanted = [gold for _, gold in data.labelled]
            outcome = asyncio.run(evaluate(index, queries))
            metrics = {}
            for kind in ("bm25", "dense", "fused"):
                lists = [r.get(kind) for r in outcome["ranked"]]
                if all(lst is not None for lst in lists):
                    metrics[kind] = rank_metrics(lists, wanted, KS)
            per[name] = {
                "ontology": slug,
                "mentions": len(queries),
                "status": status,
                "metrics": metrics,
                "errors": outcome["errors"],
                "median_search_s": {k: _median(v) for k, v in outcome["timings"].items()},
            }
            print(f"{backend:8} {name:14} {json.dumps(metrics.get('fused') or metrics)}")
        results["backends"][backend] = per

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "results.md").write_text(render_markdown(results))
    print(f"wrote {args.out}/results.json and results.md")
    return 0


def render_markdown(results: Mapping[str, Any]) -> str:
    ks = results["ks"]
    head = "| backend | dataset | retriever | " + " | ".join(f"R@{k}" for k in ks) + " | MRR |"
    lines = [
        f"# Site retrieval on the pilot gold sets ({results['date']}, catalog "
        f"{results['catalog']}, depth {results['depth']})",
        "",
        head,
        "|" + "---|" * (len(ks) + 4),
    ]
    means: dict[tuple[str, str], list[float]] = {}
    for backend, per in results["backends"].items():
        for name, row in per.items():
            for kind, m in row["metrics"].items():
                cells = " | ".join(f"{m[f'recall@{k}']:.2f}" for k in ks)
                lines.append(f"| {backend} | {name} | {kind} | {cells} | {m['mrr']:.3f} |")
                means.setdefault((backend, kind), []).append(m["mrr"])
    lines += ["", "| backend | retriever | mean MRR | datasets |", "|---|---|---|---|"]
    for (backend, kind), values in means.items():
        lines.append(f"| {backend} | {kind} | {sum(values) / len(values):.3f} | {len(values)} |")
    lines += ["", "Median search latency (seconds, from the machine that ran this):", ""]
    for backend, per in results["backends"].items():
        for name, row in per.items():
            timing = ", ".join(
                f"{k} {v:.3f}" for k, v in row["median_search_s"].items() if v is not None
            )
            errors = ", ".join(f"{k} {v}" for k, v in row["errors"].items() if v)
            lines.append(
                f"- {backend} {name}: {timing}" + (f"; errors: {errors}" if errors else "")
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

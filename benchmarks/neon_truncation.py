"""Does truncating the stored Qwen3 vectors cost retrieval quality on a full catalog?

Embeds each labelled mention of a pilot gold set once at 1024 dimensions (the site's
query encoder and template), then ranks a fully loaded ontology in Neon by exact search
at each dimension: the stored ``halfvec(1024)`` vectors and the queries are cut to the
first D components and renormalised (Matryoshka truncation, done in SQL with
``subvector`` and ``l2_normalize``), records scored by their best vector as
``PgVectorRetriever`` does. Exact search, so the HNSW approximation is not part of the
comparison. Costs one query embedding per mention.

    python -m benchmarks.neon_truncation --dataset ncbi_disease --ontology ctd-disease \\
        --dims 1024 768 512 --out benchmarks/results/neon-truncation-2026-11-04
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from benchmarks.embeddings import DATASETS, load_dataset, rank_metrics
from xwalk.records import Record
from xwalk.retrieval.embeddings_api import default_embeddings_encoder
from xwalk.retrieval.postgres import NeonHTTPExecutor, vector_literal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ner-site"))

from ner_site.indexes import DEFAULT_LIMIT, QUERY_ENCODER_SETTINGS  # noqa: E402
from ner_site.ontologies import TEMPLATES  # noqa: E402

KS = (1, 5, 10, 20)
SQL = (
    "select record_id from ("
    " select v.record_id, min(l2_normalize(subvector(v.embedding, 1, {d}))"
    " <#> l2_normalize(subvector($2::halfvec({stored}), 1, {d}))) as dist"
    " from {schema}.term_vectors v where v.ontology = $1 group by v.record_id"
    ") r order by dist, record_id limit $3"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--schema", default="xwalk")
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 768, 512])
    parser.add_argument("--stored", type=int, default=1024, help="the loaded dimension")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = load_dataset(DATASETS[args.dataset])
    mentions = [str(source.fields["mention"]) for source, _ in data.labelled]
    wanted = [gold for _, gold in data.labelled]
    encoder = default_embeddings_encoder(
        api_key=os.environ["OPENROUTER_API_KEY"], **QUERY_ENCODER_SETTINGS
    )
    queries = [
        TEMPLATES.render_query(Record(f"q{i}", {"mention": m})) for i, m in enumerate(mentions)
    ]
    vectors = [v[: args.stored] for v in encoder.encode(queries, is_query=True)]
    executor = NeonHTTPExecutor(os.environ["DATABASE_URL"], timeout=300, max_retries=2)
    count = executor.execute(
        f"select count(*) as n from {args.schema}.term_vectors where ontology = $1",
        [args.ontology],
    )[0]
    results: dict[str, object] = {
        "date": date.today().isoformat(),
        "dataset": args.dataset,
        "ontology": args.ontology,
        "vectors": int(count["n"]),
        "mentions": len(mentions),
        "depth": DEFAULT_LIMIT,
        "embedding_usage": str(getattr(encoder, "usage", "unknown")),
        "dims": {},
    }
    for d in args.dims:
        started = time.monotonic()
        ranked = [
            [
                row["record_id"]
                for row in executor.execute(
                    SQL.format(d=int(d), stored=int(args.stored), schema=args.schema),
                    [args.ontology, vector_literal(vec), DEFAULT_LIMIT],
                )
            ]
            for vec in vectors
        ]
        metrics = rank_metrics(ranked, wanted, KS)
        metrics["seconds"] = round(time.monotonic() - started, 1)
        results["dims"][str(d)] = metrics  # type: ignore[index]
        print(d, json.dumps(metrics))
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}--{args.ontology}.json"
    (args.out / name).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

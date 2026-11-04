"""Fusion alternatives on cached bake-off indexes (retrieval only, no LLM)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks import embeddings as bake
from xwalk.retrieval.base import SearchRequest

ROOT = Path(__file__).resolve().parents[1] / "paid" / "index"
KEY = os.environ["OPENROUTER_API_KEY"]
DEPTH = 50


def rrf(lists, weights, k=60, depths=None):
    scores = {}
    for i, (ranked, w) in enumerate(zip(lists, weights, strict=False)):
        d = depths[i] if depths else len(ranked)
        for rank, rid in enumerate(ranked[:d], start=1):
            scores[rid] = scores.get(rid, 0.0) + w / (k + rank)
    return [r for r, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))]


def combsum(scored_lists):
    """Min-max normalise each list's raw scores, then sum."""
    scores = {}
    for scored in scored_lists:
        if not scored:
            continue
        lo, hi = min(s for _, s in scored), max(s for _, s in scored)
        for rid, s in scored:
            scores[rid] = scores.get(rid, 0.0) + ((s - lo) / (hi - lo) if hi > lo else 1.0)
    return [r for r, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))]


def mrr(ranked, gold):
    total = 0.0
    for ids, g in zip(ranked, gold, strict=False):
        for i, rid in enumerate(ids[:DEPTH], 1):
            if rid in g:
                total += 1 / i
                break
    return total / len(gold)


async def lists_for(loaded, retriever):
    out = []
    for (source, _), queries in zip(loaded.labelled, loaded.queries, strict=False):
        hits = await retriever.search(
            SearchRequest(text=queries[0], limit=DEPTH, source_record=source)
        )
        out.append([(h.record_id, h.raw_score) for h in hits])
    return out


def main():
    presets = sys.argv[1:] or ["qwen3-8b", "gemini-2"]
    report = {}
    for name in bake.DATASETS:
        loaded = bake.load_dataset(bake.DATASETS[name])
        assert all(len(q) == 1 for q in loaded.queries)
        bm25 = bake._bm25(loaded, ROOT)
        gold = [g for _, g in loaded.labelled]
        b = asyncio.run(lists_for(loaded, bm25[0]))
        for preset_name in presets:
            preset = bake.PRESETS[preset_name]
            encoder = bake.build_encoder(preset, base_url=bake.OPENROUTER, api_key=KEY)
            wrapped = bake.PrefetchedQueries(encoder)
            dense, action = bake._dense_index(loaded, wrapped, ROOT / name / preset_name)
            assert action == "opened", action
            wrapped.prefetch([q for qs in loaded.queries for q in qs])
            d = asyncio.run(lists_for(loaded, dense))
            ids_b = [[r for r, _ in x] for x in b]
            ids_d = [[r for r, _ in x] for x in d]
            row = {
                "bm25": mrr(ids_b, gold),
                "dense": mrr(ids_d, gold),
                "rrf": mrr([rrf([x, y], [1, 1]) for x, y in zip(ids_b, ids_d, strict=False)], gold),
            }
            for w in (2, 3, 5):
                row[f"rrf_dense_x{w}"] = mrr(
                    [rrf([x, y], [1, w]) for x, y in zip(ids_b, ids_d, strict=False)], gold
                )
            for depth in (1, 3, 5, 10):
                row[f"rrf_bm25_top{depth}"] = mrr(
                    [
                        rrf([x, y], [1, 1], depths=[depth, DEPTH])
                        for x, y in zip(ids_b, ids_d, strict=False)
                    ],
                    gold,
                )
            row["combsum"] = mrr([combsum([x, y]) for x, y in zip(b, d, strict=False)], gold)
            # how often does bm25 even contain gold within its list?
            row["bm25_has_gold"] = sum(
                any(r in g for r in x) for x, g in zip(ids_b, gold, strict=False)
            ) / len(gold)
            report[f"{name}/{preset_name}"] = row
            print(name, preset_name, {k: round(v, 3) for k, v in row.items()}, flush=True)
    Path(sys.argv[0]).with_suffix(".json").write_text(json.dumps(report, indent=1))


main()

"""Per-name vectors (max-pooled per record) and context-bearing queries, retrieval only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks import embeddings as bake
from xwalk.retrieval.embeddings_api import OpenAIEmbeddingsEncoder

KEY = os.environ["OPENROUTER_API_KEY"]
DEPTH = 50
INSTR_PLAIN = bake.QWEN_QUERY
INSTR_CTX = (
    "Instruct: Given a term and the sentence it occurs in, "
    "retrieve the ontology concept the term names\nQuery:"
)

MODELS = {
    "qwen3-8b": dict(
        model="qwen/qwen3-embedding-8b", dims=4096, q_plain=INSTR_PLAIN, q_ctx=INSTR_CTX, it=None
    ),
    "gemini-2": dict(model="google/gemini-embedding-2", dims=3072, q_plain="", q_ctx="", it=None),
}


def ctx(source, width=100):
    f = source.fields
    left = (f.get("context_left") or "")[-width:]
    right = (f.get("context_right") or "")[:width]
    return f"{f['mention']} (in: …{left}[{f['mention']}]{right}…)"


def metrics(score_matrix, ids, gold):
    out = {"mrr": 0.0, "r@1": 0, "r@10": 0, "r@20": 0}
    for row, g in zip(score_matrix, gold, strict=False):
        order = np.argsort(-row, kind="stable")[:DEPTH]
        ranked = [ids[i] for i in order]
        first = next((i for i, r in enumerate(ranked, 1) if r in g), None)
        if first:
            out["mrr"] += 1 / first
            for k in (1, 10, 20):
                out[f"r@{k}"] += first <= k
    n = len(gold)
    return {k: round(v / n, 3) for k, v in out.items()}


def main():
    report = {}
    total_cost = 0.0
    for mname, cfg in MODELS.items():
        enc = OpenAIEmbeddingsEncoder(
            bake.OPENROUTER,
            cfg["model"],
            dimensions=cfg["dims"],
            api_key=KEY,
            send_dimensions=False,
            batch_size=64,
        )
        for dname, ds in bake.DATASETS.items():
            L = bake.load_dataset(ds)
            ids = [t.id for t in L.targets]
            gold = [g for _, g in L.labelled]
            docs = np.asarray(enc.encode(L.doc_texts), dtype="float32")
            name_rows, owner = [], []
            for i, t in enumerate(L.targets):
                names = [t.fields.get("label") or ""] + list(t.fields.get("synonyms") or [])
                for n in dict.fromkeys(x for x in names if x and x.strip()):
                    name_rows.append(n)
                    owner.append(i)
            names = np.asarray(enc.encode(name_rows), dtype="float32")
            owner = np.asarray(owner)
            mentions = [s.fields["mention"] for s, _ in L.labelled]
            q_plain = np.asarray(
                enc.encode([cfg["q_plain"] + m for m in mentions], is_query=True), dtype="float32"
            )
            q_ctx = np.asarray(
                enc.encode([cfg["q_ctx"] + ctx(s) for s, _ in L.labelled], is_query=True),
                dtype="float32",
            )

            def pooled(q, include_doc, names=names, ids=ids, owner=owner, docs=docs):
                s = q @ names.T
                best = np.full((len(q), len(ids)), -1.0, dtype="float32")
                for j in range(len(ids)):
                    cols = owner == j
                    if cols.any():
                        best[:, j] = s[:, cols].max(axis=1)
                return np.maximum(best, q @ docs.T) if include_doc else best

            variants = {
                "doc / mention (baseline)": q_plain @ docs.T,
                "names / mention": pooled(q_plain, False),
                "names+doc / mention": pooled(q_plain, True),
                "doc / mention+context": q_ctx @ docs.T,
                "names+doc / mention+context": pooled(q_ctx, True),
            }
            for vname, mat in variants.items():
                m = metrics(mat, ids, gold)
                report[f"{mname} | {dname} | {vname}"] = m
                print(f"{mname:9} {dname:14} {vname:30} {m}", flush=True)
        total_cost += enc.usage.cost or 0.0
    print("reported cost", round(total_cost, 5))
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=1))


main()

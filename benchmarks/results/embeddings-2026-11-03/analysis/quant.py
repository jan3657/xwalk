"""Dense MRR of the per-name Qwen3 index under storage formats pgvector offers."""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks import embeddings as bake
from xwalk.retrieval.embeddings_api import default_embeddings_encoder

ROOT = Path(__file__).resolve().parents[1] / "names" / "index"
enc = default_embeddings_encoder(api_key=os.environ["OPENROUTER_API_KEY"], max_retries=2)


def unit(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def mrr(scores, owner, ids, gold, depth=50):
    total = 0.0
    r1 = 0
    for row, g in zip(scores, gold, strict=False):
        best = {}
        for j in np.argsort(-row, kind="stable"):
            rid = ids[owner[j]]
            if rid not in best:
                best[rid] = len(best) + 1
                if rid in g:
                    total += 1 / best[rid]
                    r1 += best[rid] == 1
                    break
            if len(best) >= depth:
                break
    return total / len(gold), r1 / len(gold)


res = {}
for name in bake.DATASETS:
    L = bake.load_dataset(bake.DATASETS[name])
    d = ROOT / name / "qwen3-8b-1024-names"
    rows = json.loads((d / "record_ids.json").read_text())
    V = np.fromfile(d / "vectors.f32", dtype="<f4").reshape(len(rows), -1)
    uniq = list(dict.fromkeys(rows))
    pos = {r: i for i, r in enumerate(uniq)}
    owner = np.array([pos[r] for r in rows])
    gold = [g for _, g in L.labelled]
    Q = np.asarray(enc.encode([q[0] for q in L.queries], is_query=True), dtype="float32")
    variants = {}
    variants["f32-1024"] = Q @ V.T
    variants["f16-1024"] = (Q.astype(np.float16).astype(np.float32)) @ (
        V.astype(np.float16).astype(np.float32)
    ).T
    V512, Q512 = unit(V[:, :512]), unit(Q[:, :512])
    variants["f16-512 (truncated)"] = (
        Q512.astype(np.float16).astype(np.float32) @ V512.astype(np.float16).astype(np.float32).T
    )
    V256, Q256 = unit(V[:, :256]), unit(Q[:, :256])
    variants["f16-256 (truncated)"] = Q256 @ V256.T
    # binary 1024 (sign bits), hamming -> top 100 candidates, rerank with f16
    B, QB = V > 0, Q > 0
    ham = -(QB[:, None, :] != B[None, :, :]).sum(-1).astype(np.float32)
    rer = np.full_like(ham, -1e9)
    full = variants["f16-1024"]
    for i in range(len(Q)):
        top = np.argsort(-ham[i], kind="stable")[:400]
        rer[i, top] = full[i, top]
    variants["binary-1024 + f16 rerank (top 400 vectors)"] = rer
    variants["binary-1024 only"] = ham
    for k, m in variants.items():
        res.setdefault(k, {})[name] = mrr(m, owner, uniq, gold)
    print(name, {k: round(v[name][0], 3) for k, v in res.items()}, flush=True)
print()
for k, v in res.items():
    m = [v[n][0] for n in bake.DATASETS]
    three = [v[n][0] for n in list(bake.DATASETS)[1:]]
    print(
        f"{k:42} mean MRR {sum(m) / 4:.3f}  (excluding saturated chebi {sum(three) / 3:.3f})  "
        + "  ".join(f"{n}={v[n][0]:.3f}" for n in bake.DATASETS)
    )
print("embedding usage", enc.usage)
out = {k: {n: {"mrr": v[n][0], "r@1": v[n][1]} for n in v} for k, v in res.items()}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")

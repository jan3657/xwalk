"""LinkTransformer comparison adapter -- PENDING, never run in this repository.

LinkTransformer (Arora and Dell, 2023; https://github.com/dell-research-harvard/linktransformer)
links two tables by nearest neighbours in a sentence-embedding space. Running it needs
`pip install linktransformer` (pandas, torch, sentence-transformers) and a model
download, which this benchmark environment does not do. This module is the adapter the
comparison would use; it is not imported by `benchmarks.run`.

Common supported task: source-to-catalog matching on a materialised variant, comparing
the mention text with the target label (LinkTransformer links on one text column per
side; the synonyms xwalk's BM25 also indexes are joined into that column). It always
returns its nearest target, so it cannot abstain unless a score threshold is applied:
`accept_at` is that threshold and must be fixed before looking at test results.

The call below follows LinkTransformer's documented `merge` API as of its 0.1 releases;
it has not been executed here, so treat it as unverified until a real run.
"""

from __future__ import annotations

from typing import Any

from benchmarks.data import Variant
from benchmarks.methods import MethodOutput, _names, _query
from benchmarks.metrics import Prediction

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def linktransformer_match(
    variant: Variant, *, model: str = DEFAULT_MODEL, accept_at: float = 0.8
) -> MethodOutput:  # pragma: no cover - needs linktransformer and a model download
    try:
        import linktransformer as lt  # type: ignore[import-not-found,unused-ignore]
        import pandas as pd  # type: ignore[import-untyped,import-not-found,unused-ignore]
    except ImportError as exc:
        raise RuntimeError(
            "the LinkTransformer comparison needs `pip install linktransformer` and a "
            "model download; it is pending in this benchmark"
        ) from exc

    left = pd.DataFrame(
        {
            "source_id": [s.id for s in variant.sources],
            "text": [_query(variant, s) for s in variant.sources],
        }
    )
    right = pd.DataFrame(
        {
            "target_id": [t.id for t in variant.targets],
            "text": ["; ".join(_names(t)) for t in variant.targets],
        }
    )
    merged: Any = lt.merge(left, right, merge_type="1:m", on="text", model=model)
    predictions = []
    for row in merged.itertuples():
        score = float(row.score)
        status = "matched" if score >= accept_at else "unmatched"
        predictions.append(
            Prediction(row.source_id, status, row.target_id if status == "matched" else None, score)
        )
    return MethodOutput(predictions, config={"model": model, "accept_at": accept_at})

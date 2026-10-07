"""Retrieval defaults for the explorer and its offline ontology builder."""

from __future__ import annotations

from typing import Any

DEFAULT_DENSE_MODELS = (
    ("dense-general", "sentence-transformers/all-MiniLM-L6-v2"),
    ("dense-biomedical", "pritamdeka/S-PubMedBert-MS-MARCO"),
)


def heads(*, dense: bool = False, device: str | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {"kind": "bm25", "name": "bm25", "limit": 30, "exact_fields": ["label", "synonyms"]}
    ]
    if dense:
        for name, model in DEFAULT_DENSE_MODELS:
            result.append(
                {
                    "kind": "dense",
                    "name": name,
                    "model": model,
                    "limit": 30,
                    **({"device": device} if device else {}),
                }
            )
    return result

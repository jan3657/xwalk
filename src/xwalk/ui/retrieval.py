"""Retrieval defaults for the explorer and its offline ontology builder.

`xwalk ui --dense` (or `--dense local`) adds two local sentence-transformers heads and
needs xwalk[dense]. `--dense api` adds the default API encoder instead (Qwen3-Embedding
8B through OpenRouter, chosen by the November 2026 embedding bake-off) with per-name
vectors over each term's label and synonyms; it needs only an OPENROUTER_API_KEY.
"""

from __future__ import annotations

from typing import Any

from xwalk.retrieval.embeddings_api import DEFAULT_API_KEY_ENV

DENSE_MODES = ("off", "local", "api")

DEFAULT_DENSE_MODELS = (
    ("dense-general", "sentence-transformers/all-MiniLM-L6-v2"),
    ("dense-biomedical", "pritamdeka/S-PubMedBert-MS-MARCO"),
)

API_HEAD_NAME = "dense-qwen3"


def dense_mode(dense: bool | str) -> str:
    """`True` is the original `--dense` (local models); `False` is off."""
    if dense is True:
        return "local"
    if dense is False:
        return "off"
    if dense not in DENSE_MODES:
        raise ValueError(f"unknown dense mode {dense!r}; choose from {', '.join(DENSE_MODES)}")
    return dense


def heads(*, dense: bool | str = False, device: str | None = None) -> list[dict[str, Any]]:
    mode = dense_mode(dense)
    result: list[dict[str, Any]] = [
        {"kind": "bm25", "name": "bm25", "limit": 30, "exact_fields": ["label", "synonyms"]}
    ]
    if mode == "local":
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
    elif mode == "api":
        # No model: the job gets the default API encoder, every setting of it recorded
        # in the generated job's index identity.
        result.append(
            {
                "kind": "dense",
                "name": API_HEAD_NAME,
                "limit": 30,
                "encoder": "openai_embeddings",
                "name_fields": ["label", "synonyms"],
            }
        )
    return result


def credential_names(retrieval_heads: Any) -> set[str]:
    """The key variables the configured heads read (the API head's, when present)."""
    names: set[str] = set()
    for head in retrieval_heads:
        if head.get("encoder") == "openai_embeddings":
            names.add(str(head.get("api_key_env") or DEFAULT_API_KEY_ENV))
    return names

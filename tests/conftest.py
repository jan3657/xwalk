from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def e5_retriever(limit: int) -> dict[str, Any]:
    """The multilingual E5 dense block the decider jobs carry beside BM25 (A2).

    Needs xwalk[dense] and the model in the HF cache: `_env.sh` runs offline, so pull it
    once with HF_HUB_OFFLINE=0 via `SentenceTransformerEncoder("intfloat/multilingual-e5-small")`.
    """
    return {
        "kind": "dense",
        "name": "e5",
        "model": "intfloat/multilingual-e5-small",
        "limit": limit,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    }


@pytest.fixture
def targets_csv() -> Path:
    return FIXTURES / "targets_tiny.csv"


@pytest.fixture
def sources_csv() -> Path:
    return FIXTURES / "sources_tiny.csv"


@pytest.fixture
def targets_eval_csv() -> Path:
    """~50-row target collection.

    `targets_tiny.csv` stays at 5 rows because Phase 1's ranking assertions are
    verified against it. This one exists so evaluation work has a population:
    near-misses (anomers, related sugars, the methylxanthine family) make the
    retrieval-ceiling decomposition mean something.
    """
    return FIXTURES / "targets_eval.csv"

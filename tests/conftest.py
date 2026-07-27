from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


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

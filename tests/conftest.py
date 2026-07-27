from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def targets_csv() -> Path:
    return FIXTURES / "targets_tiny.csv"


@pytest.fixture
def sources_csv() -> Path:
    return FIXTURES / "sources_tiny.csv"

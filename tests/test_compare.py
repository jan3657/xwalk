import pytest

from tests.test_ceiling import GOLD, attempt, cand, result
from xwalk.evaluate.compare import compare_runs, compare_runs_dict, summarise_run
from xwalk.ledger import Ledger


@pytest.fixture
async def ledger(tmp_path):
    led = Ledger.open(tmp_path / "l.sqlite")
    await led.put_result(result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})]))
    await led.put_result(result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]))
    yield led
    led.close()


async def test_summarise_run_carries_the_label(ledger):
    assert summarise_run(ledger, "fp1", GOLD, label="gpt-4o-mini").label == "gpt-4o-mini"


async def test_summarise_run_reports_precision_coverage_and_cost(ledger):
    summary = summarise_run(ledger, "fp1", GOLD, label="a")
    assert summary.accepted_precision == 0.5
    assert summary.automatic_coverage is not None
    assert summary.mean_llm_calls is not None


async def test_summarise_run_carries_the_misjudged_count(ledger):
    """Precision alone cannot tell you whether a worse model is worse at judging or
    was simply handed a worse candidate list."""
    assert summarise_run(ledger, "fp1", GOLD, label="a").misjudged == 0


async def test_compare_runs_renders_one_row_per_run(ledger):
    a = summarise_run(ledger, "fp1", GOLD, label="model-a")
    b = summarise_run(ledger, "fp1", GOLD, label="model-b")
    table = compare_runs([a, b])
    assert table.count("model-a") == 1 and table.count("model-b") == 1


async def test_compare_runs_includes_a_header(ledger):
    table = compare_runs([summarise_run(ledger, "fp1", GOLD, label="a")])
    assert "precision" in table.lower() and "coverage" in table.lower()


async def test_compare_runs_marks_the_best_precision(ledger):
    from xwalk.evaluate.compare import RunSummary

    a = summarise_run(ledger, "fp1", GOLD, label="alpha")
    b = RunSummary(**{**a.__dict__, "label": "bravo", "accepted_precision": 0.9})
    table = compare_runs([a, b])
    best_line = next(line for line in table.splitlines() if line.strip().startswith("*"))
    assert "bravo" in best_line


async def test_compare_runs_marks_nothing_when_no_run_has_a_precision(ledger):
    from xwalk.evaluate.compare import RunSummary

    a = summarise_run(ledger, "fp1", GOLD, label="a")
    blank = RunSummary(**{**a.__dict__, "accepted_precision": None})
    assert not any(line.strip().startswith("*") for line in compare_runs([blank]).splitlines())


async def test_compare_runs_dict_is_json_safe(ledger):
    import json

    json.dumps(compare_runs_dict([summarise_run(ledger, "fp1", GOLD, label="a")]))


def test_compare_runs_with_no_summaries_does_not_crash():
    assert "no runs" in compare_runs([]).lower()

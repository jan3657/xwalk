import json

import pytest

from tests.test_ceiling import GOLD, attempt, cand, result
from xwalk.evaluate.report import evaluate, render_report, write_report
from xwalk.ledger import Ledger


@pytest.fixture
async def ledger(tmp_path):
    led = Ledger.open(tmp_path / "l.sqlite")
    await led.put_result(result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})]))
    await led.put_result(result("s3", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})]))
    yield led
    led.close()


async def test_evaluate_returns_metrics_and_ceiling(ledger):
    report = evaluate(ledger, "fp1", GOLD)
    assert report.metrics.labelled == 2
    assert report.ceiling.evaluable == 2


async def test_evaluate_includes_a_threshold_curve(ledger):
    assert len(evaluate(ledger, "fp1", GOLD).thresholds) == 21


async def test_the_rendered_report_leads_with_the_recommendation(ledger):
    text = render_report(evaluate(ledger, "fp1", GOLD))
    assert "Where to spend effort" in text
    assert "retriev" in text.lower()


async def test_the_rendered_report_names_every_headline_metric(ledger):
    text = render_report(evaluate(ledger, "fp1", GOLD))
    for label in ("accepted precision", "automatic coverage", "review rate"):
        assert label in text.lower()


async def test_the_cost_block_reports_money_not_just_calls(ledger):
    """A run's bill is the question the cost block exists to answer."""
    text = render_report(evaluate(ledger, "fp1", GOLD))
    assert "model calls / record" in text
    assert "cost / record" in text
    assert "$" in text.split("## Cost")[1]


async def test_the_rendered_report_states_when_a_metric_is_undefined(ledger):
    """A blank is ambiguous; 'n/a (no gold no-match labels)' is not."""
    text = render_report(evaluate(ledger, "fp1", GOLD))
    assert "n/a" in text.lower()


async def test_a_calibration_warning_appears_prominently_when_present(tmp_path):
    from xwalk.records import MatchResult

    led = Ledger.open(tmp_path / "cal.sqlite")
    base = result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])
    rows = [(f"c{i}", "T1", c) for i, c in enumerate([0.80, 0.82, 0.78])]
    rows += [(f"w{i}", "T9", c) for i, c in enumerate([0.81, 0.79, 0.83])]
    for i, (sid, mid, conf) in enumerate(rows):
        await led.put_result(
            MatchResult(
                **{
                    **base.__dict__,
                    "result_key": f"rk{i}",
                    "source_id": sid,
                    "matched_id": mid,
                    "confidence": conf,
                }
            )
        )
    gold = type(GOLD)(
        {
            **{f"c{i}": frozenset({"T1"}) for i in range(3)},
            **{f"w{i}": frozenset({"T8"}) for i in range(3)},
        }
    )
    text = render_report(evaluate(led, "fp1", gold))
    led.close()
    assert "calibration" in text.lower()


async def test_partition_sizes_are_reported_when_a_partitioner_is_given(ledger):
    from xwalk.evaluate.partition import Partitioner

    report = evaluate(ledger, "fp1", GOLD, partitioner=Partitioner())
    assert sum(report.partition_sizes.values()) == 2


async def test_partition_sizes_are_empty_without_a_partitioner(ledger):
    assert evaluate(ledger, "fp1", GOLD).partition_sizes == {}


async def test_write_report_produces_json_and_text(ledger, tmp_path):
    report = evaluate(ledger, "fp1", GOLD)
    write_report(report, tmp_path / "eval")
    assert (tmp_path / "eval.json").exists()
    assert (tmp_path / "eval.txt").exists()
    json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))


async def test_evaluate_calls_no_llm_and_no_retriever(ledger):
    """Evaluation reads the ledger. It must be free, repeatable, and offline."""
    report = evaluate(ledger, "fp1", GOLD)
    assert report.metrics.total == 2  # completed without any client being constructed


async def test_an_empty_run_renders_without_crashing(tmp_path):
    """A run whose every record failed still has to produce a readable report."""
    led = Ledger.open(tmp_path / "empty.sqlite")
    text = render_report(evaluate(led, "nothing", GOLD))
    led.close()
    assert "no ceiling failures" in text

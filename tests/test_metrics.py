from xwalk.evaluate.gold import GoldSet
from xwalk.evaluate.metrics import calibration_warning, evaluate_results, threshold_curve
from xwalk.records import DecisionReason, MatchResult, MatchStatus, Record, Usage

BELOW_ACCEPT = DecisionReason.BELOW_ACCEPT_THRESHOLD


def result(
    source_id,
    matched_id,
    status=MatchStatus.MATCHED,
    reason=DecisionReason.ACCEPT_THRESHOLD,
    confidence=0.9,
    calls=2,
    seconds=1.0,
):
    return MatchResult(
        result_key=f"rk-{source_id}",
        source_id=source_id,
        source_hash="h",
        matched_id=matched_id,
        matched_record=None if matched_id is None else Record(id=matched_id, fields={}),
        confidence=confidence,
        status=status,
        reason=reason,
        explanation="",
        candidates=(),
        attempts=(),
        usage=Usage(prompt_tokens=100, completion_tokens=20, calls=calls),
        elapsed_seconds=seconds,
        run_fingerprint="fp1",
    )


GOLD = GoldSet(
    {
        "s1": frozenset({"T1"}),
        "s2": frozenset({"T2"}),
        "s3": frozenset({"T3"}),
        "s4": frozenset(),  # correct answer is no match
    }
)


def _calibration_gold():
    return GoldSet(
        {
            **{f"c{i}": frozenset({"T1"}) for i in range(3)},
            **{f"w{i}": frozenset({"T9"}) for i in range(3)},
        }
    )


# --- accepted precision and coverage ---------------------------------------------


def test_accepted_precision_counts_only_matched_results():
    results = [result("s1", "T1"), result("s2", "TX")]
    assert evaluate_results(results, GOLD).accepted_precision == 0.5


def test_a_review_row_does_not_count_against_accepted_precision():
    results = [
        result("s1", "T1"),
        result("s2", "TX", status=MatchStatus.NEEDS_REVIEW, reason=BELOW_ACCEPT),
    ]
    assert evaluate_results(results, GOLD).accepted_precision == 1.0


def test_automatic_coverage_is_matched_over_labelled():
    results = [
        result("s1", "T1"),
        result(
            "s2", None, status=MatchStatus.NEEDS_REVIEW, reason=DecisionReason.UNRESOLVED_OUTPUT
        ),
    ]
    assert evaluate_results(results, GOLD).automatic_coverage == 0.5


def test_unlabelled_results_are_excluded_entirely():
    report = evaluate_results([result("s1", "T1"), result("s99", "TX")], GOLD)
    assert report.labelled == 1
    assert report.accepted_precision == 1.0


def test_multi_gold_counts_any_listed_id_as_correct():
    gold = GoldSet({"s1": frozenset({"T1", "T1-alias"})})
    assert evaluate_results([result("s1", "T1-alias")], gold).accepted_precision == 1.0


# --- rates ------------------------------------------------------------------------


def test_review_rate():
    results = [
        result("s1", "T1"),
        result("s2", "T2", status=MatchStatus.NEEDS_REVIEW, reason=BELOW_ACCEPT),
    ]
    assert evaluate_results(results, GOLD).review_rate == 0.5


def test_error_rate_counts_failed_results():
    results = [
        result("s1", None, status=MatchStatus.FAILED, reason=DecisionReason.PROVIDER_FAILURE),
        result("s2", "T2"),
    ]
    assert evaluate_results(results, GOLD).error_rate == 0.5


def test_unresolved_rate_is_reported_separately_from_review_rate():
    results = [
        result(
            "s1", None, status=MatchStatus.NEEDS_REVIEW, reason=DecisionReason.UNRESOLVED_OUTPUT
        ),
        result("s2", "T2", status=MatchStatus.NEEDS_REVIEW, reason=BELOW_ACCEPT),
    ]
    report = evaluate_results(results, GOLD)
    assert report.review_rate == 1.0 and report.unresolved_rate == 0.5


def test_recall_at_any_status_ignores_the_threshold():
    """The ceiling automation could reach with perfectly tuned thresholds."""
    results = [
        result("s1", "T1", status=MatchStatus.NEEDS_REVIEW, reason=BELOW_ACCEPT),
        result("s2", "TX"),
    ]
    report = evaluate_results(results, GOLD)
    assert report.recall_at_any_status == 0.5
    assert report.automatic_coverage == 0.5


# --- no-match ---------------------------------------------------------------------


def test_no_match_precision_and_recall():
    results = [
        result("s4", None, status=MatchStatus.UNMATCHED, reason=DecisionReason.SELECTOR_ABSTAINED),
        result("s1", None, status=MatchStatus.UNMATCHED, reason=DecisionReason.SELECTOR_ABSTAINED),
    ]
    report = evaluate_results(results, GOLD)
    assert report.no_match_precision == 0.5  # s4 right, s1 wrong
    assert report.no_match_recall == 1.0  # the only gold no-match was found


def test_no_match_metrics_are_none_when_no_such_labels_exist():
    gold = GoldSet({"s1": frozenset({"T1"})})
    assert evaluate_results([result("s1", "T1")], gold).no_match_recall is None


def test_a_failed_result_is_not_counted_as_a_no_match_prediction():
    """A provider 500 is not evidence of a non-match."""
    results = [
        result("s4", None, status=MatchStatus.FAILED, reason=DecisionReason.PROVIDER_FAILURE)
    ]
    assert evaluate_results(results, GOLD).no_match_precision is None


# --- duplicates, cost, latency ----------------------------------------------------


def test_duplicate_target_conflicts_are_counted():
    results = [result("s1", "T1"), result("s2", "T1")]
    assert evaluate_results(results, GOLD).duplicate_target_conflicts == 1


def test_cost_and_latency_are_per_completed_record():
    results = [
        result("s1", "T1", calls=2, seconds=1.0),
        result("s2", "T2", calls=4, seconds=3.0),
        result(
            "s3",
            None,
            status=MatchStatus.FAILED,
            reason=DecisionReason.PROVIDER_FAILURE,
            calls=1,
            seconds=10.0,
        ),
    ]
    report = evaluate_results(results, GOLD)
    assert report.mean_llm_calls == 3.0  # failures excluded
    assert report.mean_seconds == 2.0


def test_an_empty_result_set_does_not_divide_by_zero():
    report = evaluate_results([], GOLD)
    assert report.labelled == 0
    assert report.accepted_precision is None


# --- threshold curve and calibration ----------------------------------------------


def test_threshold_curve_covers_zero_to_one():
    points = threshold_curve([result("s1", "T1", confidence=0.9)], GOLD, steps=11)
    assert points[0].threshold == 0.0 and points[-1].threshold == 1.0


def test_raising_the_threshold_never_increases_coverage():
    results = [
        result("s1", "T1", confidence=0.95),
        result("s2", "TX", confidence=0.45),
    ]
    coverages = [p.coverage for p in threshold_curve(results, GOLD, steps=11)]
    assert coverages == sorted(coverages, reverse=True)


def test_a_higher_threshold_improves_precision_on_separated_confidences():
    results = [
        result("s1", "T1", confidence=0.95),
        result("s2", "TX", confidence=0.45),
    ]
    points = {round(p.threshold, 2): p for p in threshold_curve(results, GOLD, steps=11)}
    assert points[0.0].precision == 0.5
    assert points[0.9].precision == 1.0


def test_calibration_warning_fires_when_bands_overlap():
    """Correct and incorrect confidences that look the same mean the score is useless
    as a threshold, and every threshold recommendation from it is noise."""
    correct = [result(f"c{i}", "T1", confidence=c) for i, c in enumerate([0.80, 0.82, 0.78])]
    wrong = [result(f"w{i}", "TX", confidence=c) for i, c in enumerate([0.81, 0.79, 0.83])]
    assert calibration_warning(correct + wrong, _calibration_gold()) is not None


def test_no_calibration_warning_when_bands_separate():
    correct = [result(f"c{i}", "T1", confidence=c) for i, c in enumerate([0.95, 0.92, 0.97])]
    wrong = [result(f"w{i}", "TX", confidence=c) for i, c in enumerate([0.20, 0.15, 0.25])]
    assert calibration_warning(correct + wrong, _calibration_gold()) is None


def test_calibration_warning_needs_three_of_each_class():
    """Two correct and one incorrect is not evidence of anything."""
    results = [
        result("s1", "T1", confidence=0.95),
        result("s2", "TX", confidence=0.20),
        result("s3", "T3", confidence=0.92),
    ]
    assert calibration_warning(results, GOLD) is None


def test_calibration_warning_is_none_without_enough_data():
    assert calibration_warning([result("s1", "T1")], GOLD) is None


# --- from a ledger ------------------------------------------------------------------


async def test_evaluate_run_reads_the_ledger(tmp_path):
    from xwalk.evaluate.metrics import evaluate_run
    from xwalk.ledger import Ledger

    ledger = Ledger.open(tmp_path / "l.sqlite")
    await ledger.put_result(result("s1", "T1"))
    await ledger.put_result(result("s2", "TX"))
    report = evaluate_run(ledger, "fp1", GOLD)
    ledger.close()
    assert report.accepted_precision == 0.5


def test_report_as_dict_is_json_safe():
    import json

    json.dumps(evaluate_results([result("s1", "T1")], GOLD).as_dict())

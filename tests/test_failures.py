from tests.test_ceiling import GOLD, attempt, cand, result
from xwalk.evaluate.failures import PromptRole, render_failure, select_failures
from xwalk.records import DecisionReason, MatchResult, MatchStatus


def with_status(r, status, reason, confidence=0.9):
    return MatchResult(
        **{**r.__dict__, "status": status, "reason": reason, "confidence": confidence}
    )


MISJUDGED = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9", "C02": "T1"})])
NEVER = result("s1", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})])
TRUNCATED = result("s1", "T9", [attempt(0, [cand("T9"), cand("T1")], {"C01": "T9"}, truncated=1)])
CORRECT = result("s1", "T1", [attempt(0, [cand("T1")], {"C01": "T1"})])


# --- selector ---------------------------------------------------------------------


def test_selector_takes_misjudged_cases():
    cases = select_failures([MISJUDGED], GOLD, PromptRole.SELECTOR)
    assert [c.source_id for c in cases] == ["s1"]


def test_selector_ignores_never_retrieved_cases():
    """The model never saw the right answer; there is nothing for it to learn."""
    assert select_failures([NEVER], GOLD, PromptRole.SELECTOR) == []


def test_selector_ignores_truncated_cases():
    """A budget miss is not a prompt problem."""
    assert select_failures([TRUNCATED], GOLD, PromptRole.SELECTOR) == []


def test_selector_ignores_correct_results():
    assert select_failures([CORRECT], GOLD, PromptRole.SELECTOR) == []


# --- scorer -----------------------------------------------------------------------


def test_scorer_takes_incorrect_automatic_accepts():
    accepted_wrong = with_status(MISJUDGED, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD)
    cases = select_failures([accepted_wrong], GOLD, PromptRole.SCORER)
    assert [c.source_id for c in cases] == ["s1"]


def test_scorer_takes_a_wrong_accept_even_when_gold_was_never_retrieved():
    """The canonical scorer failure.

    When nothing correct was retrieved, the right behaviour is to abstain; a confident
    accept instead is exactly what the rubric exists to prevent. Excluding these would
    blind the optimiser to abstention behaviour. The "the model never saw the right
    answer" argument applies to the selector, which can only choose among what it was
    shown -- not to the scorer, whose job includes rejecting the whole slate.
    """
    assert len(select_failures([NEVER], GOLD, PromptRole.SCORER)) == 1


def test_scorer_takes_unnecessary_abstentions_where_gold_was_presented():
    abstained = MatchResult(
        **{
            **result("s1", None, [attempt(0, [cand("T1")], {"C01": "T1"})]).__dict__,
            "status": MatchStatus.UNMATCHED,
            "reason": DecisionReason.SELECTOR_ABSTAINED,
        }
    )
    assert [c.source_id for c in select_failures([abstained], GOLD, PromptRole.SCORER)] == ["s1"]


def test_scorer_takes_correct_matches_pushed_below_the_floor():
    below = with_status(
        CORRECT, MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR, confidence=0.2
    )
    assert len(select_failures([below], GOLD, PromptRole.SCORER)) == 1


def test_scorer_ignores_a_correct_confident_accept():
    good = with_status(CORRECT, MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD)
    assert select_failures([good], GOLD, PromptRole.SCORER) == []


# --- rewriter ---------------------------------------------------------------------


def test_rewriter_takes_cases_recovered_by_a_later_attempt():
    recovered = result(
        "s1",
        "T1",
        [attempt(0, [cand("T9")], {"C01": "T9"}), attempt(1, [cand("T1")], {"C01": "T1"})],
    )
    cases = select_failures([recovered], GOLD, PromptRole.REWRITER)
    assert [c.source_id for c in cases] == ["s1"]


def test_rewriter_takes_cases_never_recovered():
    assert len(select_failures([NEVER], GOLD, PromptRole.REWRITER)) == 1


def test_rewriter_ignores_cases_found_on_the_first_attempt():
    assert select_failures([CORRECT], GOLD, PromptRole.REWRITER) == []


# --- doc template -----------------------------------------------------------------


def test_doc_template_takes_only_never_retrieved_cases():
    cases = select_failures([NEVER, MISJUDGED, TRUNCATED], GOLD, PromptRole.DOC_TEMPLATE)
    assert [c.bucket.value for c in cases] == ["never_retrieved"]


# --- shared behaviour -------------------------------------------------------------


def test_unlabelled_results_are_never_selected():
    unlabelled = result("s99", "T9", [attempt(0, [cand("T9")], {"C01": "T9"})])
    for role in PromptRole:
        assert select_failures([unlabelled], GOLD, role) == []


def test_no_match_gold_labels_are_never_selected_for_retrieval_roles():
    no_match = result("s2", None, [attempt(0, [], {})])
    assert select_failures([no_match], GOLD, PromptRole.DOC_TEMPLATE) == []


def _many():
    return [
        MatchResult(**{**MISJUDGED.__dict__, "source_id": f"x{i}", "result_key": f"rk{i}"})
        for i in range(10)
    ]


def _gold_many():
    return type(GOLD)({f"x{i}": frozenset({"T1"}) for i in range(10)})


def test_the_limit_is_honoured():
    assert len(select_failures(_many(), _gold_many(), PromptRole.SELECTOR, limit=3)) == 3


def test_selection_is_deterministic_under_a_limit():
    many, gold_many = _many(), _gold_many()
    first = select_failures(many, gold_many, PromptRole.SELECTOR, limit=3)
    second = select_failures(list(reversed(many)), gold_many, PromptRole.SELECTOR, limit=3)
    assert [c.source_id for c in first] == [c.source_id for c in second]


def test_render_failure_shows_gold_chosen_and_what_was_presented():
    [case] = select_failures([MISJUDGED], GOLD, PromptRole.SELECTOR)
    text = render_failure(case)
    assert "T1" in text and "T9" in text


def test_render_failure_never_contains_a_python_repr():
    [case] = select_failures([MISJUDGED], GOLD, PromptRole.SELECTOR)
    assert "frozenset" not in render_failure(case)

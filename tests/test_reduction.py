"""Decision reduction and retrieval health (engine revision batch 3, R4 and R5).

R4: `derive_status` agrees with `is_acceptable`. The loop stops on an acceptable attempt;
the reducer must then report that attempt as the match, not a higher-scoring attempt a
verifier rejected or could not check. When no attempt is acceptable the existing
fallback ordering is unchanged (oracle: `_old_derive_status` below, the reducer as it
was before this batch).

R5: incomplete evidence never settles as a no-match. Precedence, highest first:

| Evidence                                                    | Status / reason                 |
|-------------------------------------------------------------|---------------------------------|
| any fatal provider failure                                  | failed / fatal_provider_failure |
| a completed acceptable attempt                              | matched / accept_threshold      |
| every core attempt failed                                   | failed / last failure reason    |
| the existing fallback produces a review                     | needs_review, existing reason   |
| otherwise negative, with a core failure or degraded search  | failed / latest blocking reason |
| clean scored negative                                       | unmatched / below_review_floor  |
| clean explicit selector abstention                          | unmatched / selector_abstained  |
| clean retrieval with no candidates                          | unmatched / no_candidates       |
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence

import pytest

from tests.test_matcher import (
    PROMPTS,
    SOURCE,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    rewrite_reply,
    score_reply,
    select_reply,
    verify_reply,
)
from tests.test_policy import attempt, matched
from xwalk.batch import run_batch, run_fingerprint_components
from xwalk.llm.base import LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy, derive_status, is_acceptable
from xwalk.records import Attempt, DecisionReason, MatchStatus
from xwalk.serde import result_from_dict, result_to_dict
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.keying import Resolution
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

POLICY = MatchPolicy()
BAND = MatchPolicy(verify_band=(0.6, 0.8), max_attempts=3)


def matcher(llm, *retrievers, policy=None, keep=True, run_fp="fp1"):
    policy = policy or MatchPolicy()
    return Matcher(
        templates=TEMPLATES,
        retrievers=list(retrievers),
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES, review_floor=policy.review_floor),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=policy,
        run_fingerprint=run_fp,
        keep_candidates_in_trace=keep,
    )


def dead(name="dense"):
    return ScriptedRetriever({}, name=name, fail=True)


# --- R4: the probes from the review (P1, P1b, P2) --------------------------------


async def test_p1_a_verified_lower_retry_beats_a_verifier_rejected_higher_score():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.75, better_candidate_keys=["C02"]),
            verify_reply("disagree", preferred_key="C02"),
            score_reply(0.65),  # the scorer's lead, re-scored without a new search
            verify_reply("support"),
        ]
    )
    retriever = ScriptedRetriever({"glucose": ["T1", "T2"]})
    result = await matcher(llm, retriever, policy=BAND).match(SOURCE)
    assert len(result.attempts) == 2 and is_acceptable(result.attempts[1], BAND)
    assert (result.status, result.reason) == (MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD)
    assert result.matched_id == "T2" and result.confidence == 0.65


async def test_p1b_an_audited_disagreement_does_not_hide_a_verified_retry():
    policy = MatchPolicy(verify_band=(0.6, 0.8), audit_rate=1.0, max_attempts=3)
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.95, better_candidate_keys=["C02"]),
            verify_reply("disagree", preferred_key="C02"),
            score_reply(0.9),
            verify_reply("support"),
        ]
    )
    result = await matcher(llm, ScriptedRetriever({"glucose": ["T1", "T2"]}), policy=policy).match(
        SOURCE
    )
    assert (result.status, result.matched_id, result.confidence) == (
        MatchStatus.MATCHED,
        "T2",
        0.9,
    )


async def test_p2_a_verified_retry_beats_a_higher_unverified_score():
    llm = FakeLLM(
        [
            select_reply("C01"),
            score_reply(0.75),
            LLMRetryableError("503"),  # the gating verifier
            select_reply("C01"),
            score_reply(0.70),
            verify_reply("support"),
        ]
    )
    result = await matcher(llm, ScriptedRetriever({"glucose": ["T1", "T2"]}), policy=BAND).match(
        SOURCE
    )
    assert (result.status, result.reason) == (MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD)
    assert result.matched_id == "T1" and result.confidence == 0.70
    assert result.attempts[0].verifier_decision == "error"


def test_an_unchallenged_lower_score_does_not_beat_an_unverified_higher_one():
    """The fallback is unchanged when nothing is acceptable: an unverified 0.7 is still
    the review proposal, not an unchallenged 0.3 (the full re-ranking tuple was rejected)."""
    unverified = matched(0.7, verifier_decision="error", reason=DecisionReason.PROVIDER_FAILURE)
    low = matched(0.3, index=1, chosen_id="T2")
    status, reason, best = derive_status([unverified, low], POLICY)
    assert (status, reason, best) == (
        MatchStatus.NEEDS_REVIEW,
        DecisionReason.PROVIDER_FAILURE,
        unverified,
    )


def test_an_attempt_without_a_target_is_never_acceptable():
    assert not is_acceptable(matched(0.9, chosen_id=None), POLICY)


# --- R4: property test against the old reducer -----------------------------------

_EXACT = Resolution.EXACT_KEY.value
_FAILURES = (
    DecisionReason.RETRIEVER_FAILURE,
    DecisionReason.PROVIDER_FAILURE,
    DecisionReason.FATAL_PROVIDER_FAILURE,
)


def _old_derive_status(
    attempts: Sequence[Attempt], policy: MatchPolicy
) -> tuple[MatchStatus, DecisionReason, Attempt | None]:
    """`policy.derive_status` as it was before batch 3, kept verbatim as the oracle."""
    from xwalk.policy import is_valid_score

    def failure(a: Attempt) -> bool:
        return a.reason in _FAILURES

    if not attempts:
        return MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE, None
    if any(a.reason is DecisionReason.FATAL_PROVIDER_FAILURE for a in attempts):
        return MatchStatus.FAILED, DecisionReason.FATAL_PROVIDER_FAILURE, None
    if all(failure(a) for a in attempts):
        return MatchStatus.FAILED, attempts[-1].reason or DecisionReason.PROVIDER_FAILURE, None
    scored = [a for a in attempts if a.chosen_id is not None and is_valid_score(a.primary_score)]
    if scored:
        best = min(scored, key=lambda a: (-(a.primary_score or 0.0), failure(a), a.index))
        if best.verifier_decision in ("disagree", "no_match"):
            return MatchStatus.NEEDS_REVIEW, DecisionReason.VERIFIER_DISAGREEMENT, best
        if best.verifier_decision not in (None, "support"):
            return MatchStatus.NEEDS_REVIEW, DecisionReason.PROVIDER_FAILURE, best
        if best.resolution != _EXACT:
            return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT, best
        score = best.primary_score or 0.0
        if score >= policy.accept_at:
            return MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD, best
        if score >= policy.review_floor:
            return MatchStatus.NEEDS_REVIEW, DecisionReason.BELOW_ACCEPT_THRESHOLD, best
        return MatchStatus.UNMATCHED, DecisionReason.BELOW_REVIEW_FLOOR, best
    if any(a.resolution == Resolution.UNRESOLVED.value for a in attempts):
        last = next(a for a in reversed(attempts) if a.resolution == Resolution.UNRESOLVED.value)
        return MatchStatus.NEEDS_REVIEW, DecisionReason.UNRESOLVED_OUTPUT, last
    if any(a.candidate_count > 0 for a in attempts):
        last = next(a for a in reversed(attempts) if a.candidate_count > 0)
        return MatchStatus.UNMATCHED, DecisionReason.SELECTOR_ABSTAINED, last
    return MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES, attempts[-1]


def _random_attempt(rng: random.Random, index: int) -> Attempt:
    """An attempt of a shape the loop can produce, plus a few it should never produce."""
    kind = rng.choice(
        ["scored", "scored", "scored", "abstain", "unresolved", "none", "fail", "noisy"]
    )
    degraded = ("dense: index unavailable",) if rng.random() < 0.2 else ()
    if kind == "none":
        reason = rng.choice([DecisionReason.NO_CANDIDATES, DecisionReason.RETRIEVER_FAILURE])
        return attempt(index=index, reason=reason, retrieval_failures=degraded)
    if kind == "fail":
        return attempt(
            index=index,
            candidate_count=2,
            chosen_id=rng.choice([None, "T1"]),
            reason=rng.choice([DecisionReason.PROVIDER_FAILURE] * 6 + [_FAILURES[2]]),
            retrieval_failures=degraded,
        )
    if kind == "abstain":
        return attempt(index=index, candidate_count=2, retrieval_failures=degraded)
    if kind == "unresolved":
        return attempt(
            index=index,
            candidate_count=2,
            chosen_id=rng.choice([None, "T1"]),
            resolution=Resolution.UNRESOLVED.value,
            retrieval_failures=degraded,
        )
    score = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0])
    verdict = rng.choice([None, None, "support", "disagree", "no_match", "error"])
    reason = DecisionReason.PROVIDER_FAILURE if verdict == "error" else None
    resolution = _EXACT if rng.random() < 0.9 else Resolution.LEGACY_RANK.value
    if kind == "noisy":  # never produced by the loop: a failed attempt with a clean verdict
        reason = rng.choice(list(DecisionReason))
        verdict = rng.choice([None, "support"])
    return matched(
        score,
        index=index,
        chosen_id=rng.choice(["T1", "T2", "T3"]),
        verifier_decision=verdict,
        reason=reason,
        resolution=resolution,
        retrieval_failures=degraded,
    )


def _latest_blocking(attempts: Sequence[Attempt]) -> DecisionReason | None:
    for a in reversed(attempts):
        if a.reason in (DecisionReason.PROVIDER_FAILURE, DecisionReason.RETRIEVER_FAILURE):
            return a.reason
        if a.retrieval_failures:
            return DecisionReason.RETRIEVER_FAILURE
    return None


@pytest.mark.parametrize("seed", range(8))
def test_reduction_property_against_the_old_reducer(seed):
    rng = random.Random(seed)
    for _ in range(2500):
        attempts = [_random_attempt(rng, i) for i in range(rng.randint(1, 5))]
        policy = rng.choice([POLICY, BAND, MatchPolicy(accept_at=0.8, review_floor=0.5)])
        status, reason, best = derive_status(attempts, policy)
        old = _old_derive_status(attempts, policy)
        acceptable = [a for a in attempts if is_acceptable(a, policy)]
        fatal = any(a.reason is DecisionReason.FATAL_PROVIDER_FAILURE for a in attempts)

        if fatal:
            assert (status, reason, best) == old
        elif acceptable:
            top = min(acceptable, key=lambda a: (-(a.primary_score or 0.0), a.index))
            assert (status, reason, best) == (
                MatchStatus.MATCHED,
                DecisionReason.ACCEPT_THRESHOLD,
                top,
            )
        elif old[0] is MatchStatus.MATCHED:
            # Only reachable with a failed attempt carrying a clean verdict, which the
            # loop never builds; it must not become an automatic match.
            assert (status, reason, best) == (
                MatchStatus.NEEDS_REVIEW,
                DecisionReason.PROVIDER_FAILURE,
                old[2],
            )
        elif old[0] is MatchStatus.UNMATCHED and _latest_blocking(attempts) is not None:
            assert (status, reason, best) == (MatchStatus.FAILED, _latest_blocking(attempts), None)
        else:
            assert (status, reason, best) == old
        assert (status is MatchStatus.MATCHED) == (bool(acceptable) and not fatal)


# --- R5: the precedence table, one row at a time -------------------------------


def test_row1_a_fatal_failure_beats_an_acceptable_attempt():
    status, reason, _ = derive_status(
        [matched(0.95), attempt(index=1, reason=DecisionReason.FATAL_PROVIDER_FAILURE)], POLICY
    )
    assert (status, reason) == (MatchStatus.FAILED, DecisionReason.FATAL_PROVIDER_FAILURE)


def test_row2_an_acceptable_attempt_beats_failures_and_degraded_search():
    failed = attempt(reason=DecisionReason.PROVIDER_FAILURE, candidate_count=2)
    good = matched(0.9, index=1, retrieval_failures=("dense: down",))
    status, reason, best = derive_status([failed, good], POLICY)
    assert (status, reason, best) == (MatchStatus.MATCHED, DecisionReason.ACCEPT_THRESHOLD, good)


def test_row3_every_core_attempt_failed_reports_the_last_reason():
    status, reason, best = derive_status(
        [
            attempt(reason=DecisionReason.PROVIDER_FAILURE),
            attempt(index=1, reason=DecisionReason.RETRIEVER_FAILURE),
        ],
        POLICY,
    )
    assert (status, reason, best) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE, None)


@pytest.mark.parametrize(
    ("review", "expected"),
    [
        (matched(0.5), DecisionReason.BELOW_ACCEPT_THRESHOLD),
        (matched(0.9, verifier_decision="disagree"), DecisionReason.VERIFIER_DISAGREEMENT),
        (
            matched(0.7, verifier_decision="error", reason=DecisionReason.PROVIDER_FAILURE),
            DecisionReason.PROVIDER_FAILURE,
        ),
        (
            attempt(candidate_count=2, resolution=Resolution.UNRESOLVED.value),
            DecisionReason.UNRESOLVED_OUTPUT,
        ),
    ],
)
def test_row4_a_review_survives_failures_and_degraded_search(review, expected):
    degraded = dataclasses.replace(review, retrieval_failures=("dense: down",))
    later_failure = attempt(index=1, reason=DecisionReason.PROVIDER_FAILURE, candidate_count=2)
    clean_abstention = attempt(index=2, candidate_count=2)
    status, reason, best = derive_status([degraded, later_failure, clean_abstention], POLICY)
    assert (status, reason, best) == (MatchStatus.NEEDS_REVIEW, expected, degraded)


@pytest.mark.parametrize(
    "negative",
    [matched(0.2), attempt(candidate_count=2), attempt(reason=DecisionReason.NO_CANDIDATES)],
    ids=["below_floor", "abstained", "no_candidates"],
)
def test_row5_a_negative_with_a_core_failure_is_failed(negative):
    failure = attempt(index=1, reason=DecisionReason.PROVIDER_FAILURE, candidate_count=2)
    status, reason, best = derive_status([negative, failure], POLICY)
    assert (status, reason, best) == (MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE, None)


@pytest.mark.parametrize(
    "negative",
    [matched(0.2), attempt(candidate_count=2), attempt(reason=DecisionReason.NO_CANDIDATES)],
    ids=["below_floor", "abstained", "no_candidates"],
)
def test_row5_a_negative_from_degraded_search_is_a_retriever_failure(negative):
    degraded = dataclasses.replace(negative, retrieval_failures=("dense: index unavailable",))
    status, reason, best = derive_status([degraded], POLICY)
    assert (status, reason, best) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE, None)


def test_row5_follows_the_latest_blocking_failure():
    provider = attempt(reason=DecisionReason.PROVIDER_FAILURE, candidate_count=2)
    degraded = attempt(index=1, candidate_count=2, retrieval_failures=("dense: down",))
    assert derive_status([provider, degraded], POLICY)[1] is DecisionReason.RETRIEVER_FAILURE
    provider = dataclasses.replace(provider, index=1)
    degraded = dataclasses.replace(degraded, index=0)
    assert derive_status([degraded, provider], POLICY)[1] is DecisionReason.PROVIDER_FAILURE


def test_row5_within_one_attempt_the_provider_failure_is_primary():
    both = attempt(
        candidate_count=2,
        reason=DecisionReason.PROVIDER_FAILURE,
        retrieval_failures=("dense: down",),
    )
    clean_negative = attempt(index=1, candidate_count=2)
    status, reason, _ = derive_status([both, clean_negative], POLICY)
    assert (status, reason) == (MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE)


def test_row5_a_later_clean_negative_does_not_erase_an_earlier_failure():
    failure = attempt(reason=DecisionReason.PROVIDER_FAILURE, candidate_count=2)
    status, _, _ = derive_status([failure, matched(0.1, index=1)], POLICY)
    assert status is MatchStatus.FAILED


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [
        ([matched(0.2)], DecisionReason.BELOW_REVIEW_FLOOR),
        ([attempt(candidate_count=2)], DecisionReason.SELECTOR_ABSTAINED),
        ([attempt(reason=DecisionReason.NO_CANDIDATES)], DecisionReason.NO_CANDIDATES),
        ([attempt(candidate_count=2), matched(0.2, index=1)], DecisionReason.BELOW_REVIEW_FLOOR),
        (
            [attempt(reason=DecisionReason.NO_CANDIDATES), attempt(index=1, candidate_count=2)],
            DecisionReason.SELECTOR_ABSTAINED,
        ),
    ],
    ids=["scored", "abstained", "no_candidates", "abstain_then_scored", "empty_then_abstain"],
)
def test_rows6_to_8_clean_negatives_settle_as_unmatched(attempts, expected):
    status, reason, _ = derive_status(attempts, POLICY)
    assert (status, reason) == (MatchStatus.UNMATCHED, expected)


# --- R5 through the loop ----------------------------------------------------------


async def test_a_partial_outage_with_no_surviving_hits_is_a_retriever_failure():
    llm = FakeLLM([])
    empty = ScriptedRetriever({}, name="bm25")
    result = await matcher(llm, empty, dead(), policy=MatchPolicy(max_attempts=1)).match(SOURCE)
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert result.attempts[0].reason is DecisionReason.RETRIEVER_FAILURE
    assert result.attempts[0].retrieval_failures == ("dense: index unavailable",)
    assert llm.requests == []  # no rewriter call for an outage


async def test_a_partial_outage_retries_the_same_query_and_can_recover():
    calls = {"n": 0}

    class Flaky(ScriptedRetriever):
        async def search(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("blip")
            return await super().search(request)

    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    empty = ScriptedRetriever({}, name="bm25")
    flaky = Flaky({"glucose": ["T1"]}, name="dense")
    result = await matcher(llm, empty, flaky, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert flaky.queries == ["glucose"]  # the first call raised before recording
    assert empty.queries == ["glucose", "glucose"]  # same query, not a rewrite
    assert (result.status, result.matched_id) == (MatchStatus.MATCHED, "T1")


async def test_a_partial_outage_with_surviving_hits_can_still_match():
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    result = await matcher(llm, good, dead()).match(SOURCE)
    assert (result.status, result.matched_id) == (MatchStatus.MATCHED, "T1")
    assert result.attempts[0].retrieval_failures == ("dense: index unavailable",)
    assert result.attempts[0].reason is None


async def test_a_partial_outage_with_surviving_hits_can_still_review():
    llm = FakeLLM([select_reply("C01"), score_reply(0.5)])
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    result = await matcher(llm, good, dead(), policy=MatchPolicy(max_attempts=1)).match(SOURCE)
    assert (result.status, result.reason) == (
        MatchStatus.NEEDS_REVIEW,
        DecisionReason.BELOW_ACCEPT_THRESHOLD,
    )


@pytest.mark.parametrize(
    "script",
    [[select_reply(None)], [select_reply("C01"), score_reply(0.1)]],
    ids=["abstained", "below_floor"],
)
async def test_a_negative_from_a_partial_outage_does_not_settle(script):
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    result = await matcher(FakeLLM(script), good, dead(), policy=MatchPolicy(max_attempts=1)).match(
        SOURCE
    )
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert result.matched_id is None


async def test_empty_search_then_a_scorer_outage_is_failed_not_abstained():
    """Advisor finding A#4: there was no selector abstention to report."""
    llm = FakeLLM([rewrite_reply("dextrose"), select_reply("C01"), LLMRetryableError("offline")])
    retriever = ScriptedRetriever({"dextrose": ["T1"]})
    result = await matcher(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE)
    assert result.matched_id is None


async def test_an_abstention_then_a_provider_failure_is_failed():
    llm = FakeLLM([select_reply(None), rewrite_reply("dextrose"), LLMRetryableError("503")])
    retriever = ScriptedRetriever({"glucose": ["T1"], "dextrose": ["T2"]})
    result = await matcher(llm, retriever, policy=MatchPolicy(max_attempts=2)).match(SOURCE)
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE)


async def test_a_recoverable_rewriter_failure_is_not_a_core_failure():
    llm = FakeLLM([select_reply(None), LLMRetryableError("503")])
    result = await matcher(llm, ScriptedRetriever({"glucose": ["T1"]})).match(SOURCE)
    assert (result.status, result.reason) == (
        MatchStatus.UNMATCHED,
        DecisionReason.SELECTOR_ABSTAINED,
    )
    assert "rewriter" in (result.attempts[0].error or "")


async def test_trace_settings_do_not_change_degraded_decisions():
    outcomes = []
    for keep in (True, False):
        good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
        llm = FakeLLM([select_reply("C01"), score_reply(0.1)])
        policy = MatchPolicy(max_attempts=1)
        result = await matcher(llm, good, dead(), policy=policy, keep=keep).match(SOURCE)
        outcomes.append(result)
    on, off = outcomes
    assert (on.status, on.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert (off.status, off.reason, off.matched_id) == (on.status, on.reason, on.matched_id)
    assert off.attempts[0].candidates == () and on.attempts[0].candidates
    assert off.attempts[0].retrieval_failures == on.attempts[0].retrieval_failures


# --- resume ---------------------------------------------------------------------


async def test_a_failed_row_from_degraded_search_is_retried_on_resume(tmp_path):
    out = tmp_path / "run"

    def run(*retrievers, script):
        return run_batch(
            matcher(FakeLLM(script), *retrievers, policy=MatchPolicy(max_attempts=1)),
            [SOURCE],
            out=out,
            resume=True,
        )

    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    first = await run(good, dead(), script=[select_reply(None)])
    assert first.by_status() == {MatchStatus.FAILED: 1}

    healthy = ScriptedRetriever({"glucose": ["T1"]}, name="dense")
    second = await run(good, healthy, script=[select_reply("C01"), score_reply(0.95)])
    assert healthy.queries == ["glucose"]  # the failed row was recomputed
    assert second.by_status() == {MatchStatus.MATCHED: 1}


# --- storage and identity -------------------------------------------------------


async def test_retrieval_failures_round_trip_and_old_blobs_read_as_clean():
    good = ScriptedRetriever({"glucose": ["T1"]}, name="bm25")
    llm = FakeLLM([select_reply("C01"), score_reply(0.95)])
    result = await matcher(llm, good, dead()).match(SOURCE)
    data = result_to_dict(result)
    assert data["attempts"][0]["retrieval_failures"] == ["dense: index unavailable"]
    assert result_from_dict(data).attempts[0].retrieval_failures == ("dense: index unavailable",)
    del data["attempts"][0]["retrieval_failures"]
    assert result_from_dict(data).attempts[0].retrieval_failures == ()


def test_the_status_semantics_version_is_part_of_run_identity():
    from xwalk.policy import STATUS_SEMANTICS_VERSION

    components = run_fingerprint_components(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever({})],
        llm=FakeLLM([]),
        policy=POLICY,
        selector_policy=SelectorPolicy(),
    )
    assert components["policy"]["status_semantics"] == STATUS_SEMANTICS_VERSION >= 2


def test_the_decider_status_semantics_version_is_part_of_run_identity():
    from tests.test_decide_matcher import QUESTIONS
    from tests.test_decide_matcher import STORE as DSTORE
    from tests.test_decide_matcher import TEMPLATES as DTEMPLATES
    from xwalk.batch import decision_run_fingerprint_components
    from xwalk.decide.fake import FakeDecider
    from xwalk.decide.policy import DecisionPolicy

    components = decision_run_fingerprint_components(
        templates=DTEMPLATES,
        questions=QUESTIONS,
        store=DSTORE,
        retrievers=[ScriptedRetriever({})],
        decider=FakeDecider(),
        policy=DecisionPolicy(),
    )
    from xwalk.decide.policy import STATUS_SEMANTICS_VERSION as DECIDE_VERSION

    assert components["policy"]["status_semantics"] == DECIDE_VERSION >= 2


# --- the decider path -------------------------------------------------------------


class _SometimesDown:
    """A decider-path retriever that fails for the queries in `down` ("*": all)."""

    def __init__(self, by_query, *, name="bm25", down=()):
        self._by_query, self._name, self._down = by_query, name, set(down)
        self.queries: list[str] = []

    @property
    def name(self):
        return self._name

    @property
    def fingerprint(self):
        return "s"

    @property
    def default_limit(self):
        return 10

    async def search(self, request):
        from xwalk.records import RetrievalHit
        from xwalk.retrieval.base import RetrieverError

        self.queries.append(request.text)
        if request.text in self._down or "*" in self._down:
            raise RetrieverError("index unavailable")
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(self._by_query.get(request.text, []), start=1)
        ]


def _decider(*retrievers, decider=None, rewriter=None):
    from tests.test_decide_matcher import _matcher

    return _matcher(decider, retrievers=list(retrievers), rewriter=rewriter)


async def test_decider_partial_outage_with_no_surviving_hits_is_a_retriever_failure():
    from tests.test_decide_matcher import SOURCE as DSOURCE

    empty = _SometimesDown({}, name="bm25")
    down = _SometimesDown({}, name="dense", down={"*"})
    result = await _decider(empty, down).match(DSOURCE)
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert len(result.attempts[0].retrieval_failures) == 2  # one per query


async def test_decider_partial_outage_with_surviving_hits_can_still_match():
    from tests.test_decide_matcher import SOURCE as DSOURCE

    good = _SometimesDown({"glucose": ["T1", "T2"]}, name="bm25")
    down = _SometimesDown({}, name="dense", down={"*"})
    result = await _decider(good, down).match(DSOURCE)
    assert (result.status, result.matched_id) == (MatchStatus.MATCHED, "T1")
    assert result.attempts[0].retrieval_failures


async def test_decider_negative_from_a_partial_outage_does_not_settle():
    from tests.test_decide_matcher import MISS

    # The surviving retriever only finds fructose: below the screen floor.
    good = _SometimesDown({"blood sugar": ["T2"]}, name="bm25")
    down = _SometimesDown({}, name="dense", down={"*"})
    result = await _decider(good, down).match(MISS)
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert result.matched_id is None


async def test_decider_rewritten_retrieval_failure_is_kept_in_the_final_status():
    from tests.test_decide_matcher import MISS, TWO_QUERIES, _rewriter
    from xwalk.llm.fake import FakeLLM

    # First pass healthy but a miss; every search of the rewritten queries fails.
    retriever = _SometimesDown({"blood sugar": ["T2"]}, down={"glucose", "dextrose"})
    llm = FakeLLM([TWO_QUERIES])
    result = await _decider(retriever, rewriter=_rewriter(llm)).match(MISS)
    assert retriever.queries == ["blood sugar", "bs", "glucose", "dextrose"]
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert result.attempts[0].retrieval_failures == (
        "bm25: index unavailable",
        "bm25: index unavailable",
    )


async def test_decider_provider_failure_is_primary_and_keeps_retrieval_diagnostics():
    from tests.test_decide_matcher import SOURCE as DSOURCE
    from xwalk.decide.base import DecisionRetryableError
    from xwalk.decide.fake import FakeDecider

    def broken(state, questions):
        raise DecisionRetryableError("503")

    good = _SometimesDown({"glucose": ["T1", "T2"]}, name="bm25")
    down = _SometimesDown({}, name="dense", down={"*"})
    result = await _decider(good, down, decider=FakeDecider(handler=broken)).match(DSOURCE)
    assert (result.status, result.reason) == (MatchStatus.FAILED, DecisionReason.PROVIDER_FAILURE)
    assert result.attempts[0].retrieval_failures


async def test_decider_clean_negative_still_settles():
    from tests.test_decide_matcher import MISS

    result = await _decider(_SometimesDown({"blood sugar": ["T2"]})).match(MISS)
    assert (result.status, result.reason) == (
        MatchStatus.UNMATCHED,
        DecisionReason.BELOW_REVIEW_FLOOR,
    )
    assert result.attempts[0].retrieval_failures == ()


async def test_decider_trace_settings_do_not_change_degraded_decisions():
    from tests.test_decide_matcher import MISS, _matcher

    outcomes = []
    for keep in (True, False):
        good = _SometimesDown({"blood sugar": ["T2"]}, name="bm25")
        down = _SometimesDown({}, name="dense", down={"*"})
        m = _matcher(retrievers=[good, down])
        m._keep_candidates = keep
        outcomes.append(await m.match(MISS))
    on, off = outcomes
    assert (on.status, on.reason) == (MatchStatus.FAILED, DecisionReason.RETRIEVER_FAILURE)
    assert (off.status, off.reason) == (on.status, on.reason)
    assert off.attempts[0].retrieval_failures == on.attempts[0].retrieval_failures


def test_decider_derive_status_fails_a_degraded_negative():
    from xwalk.decide.policy import DecisionPolicy, Signals
    from xwalk.decide.policy import derive_status as decide_status

    policy = DecisionPolicy()
    clean = Signals(candidate_count=0, retrieval_failed=False)
    assert decide_status(clean, policy) == (MatchStatus.UNMATCHED, DecisionReason.NO_CANDIDATES)
    degraded = Signals(candidate_count=0, retrieval_failed=False, retrieval_degraded=True)
    assert decide_status(degraded, policy) == (
        MatchStatus.FAILED,
        DecisionReason.RETRIEVER_FAILURE,
    )
    low = Signals(candidate_count=3, retrieval_failed=False, screen_best=0.05)
    assert decide_status(dataclasses.replace(low, retrieval_degraded=True), policy) == (
        MatchStatus.FAILED,
        DecisionReason.RETRIEVER_FAILURE,
    )
    review = Signals(
        candidate_count=3,
        retrieval_failed=False,
        retrieval_degraded=True,
        screen_best=0.9,
        screen_chosen=0.9,
        resolution=Resolution.EXACT_KEY.value,
        p_choice=0.9,
    )
    assert decide_status(review, policy) == (
        MatchStatus.NEEDS_REVIEW,
        DecisionReason.BELOW_ACCEPT_THRESHOLD,
    )

"""Flat equivalence clustering: outcomes, policy, determinism, history, bounds.

Every test drives the real engine, prompts and store with `Oracle` (tests/cluster_helpers),
a scripted judge that answers from ground truth. No model is called.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import pytest

from tests.cluster_helpers import (
    CHOCOLATE,
    TEMPLATES,
    Oracle,
    blocks,
    cluster,
    concepts,
    member_rows,
    partition,
    records,
    sections,
    settings,
)
from xwalk.cluster import ClusterStore
from xwalk.cluster.engine import ClusterEngine
from xwalk.cluster.prompts import SYSTEM, ClusterPrompts
from xwalk.cluster.run import order_sources
from xwalk.llm.base import LLMFatalError, LLMRetryableError

EXPECTED_CHOCOLATE = {
    frozenset({"c1"}),
    frozenset({"c2", "c3", "c4"}),
    frozenset({"c5"}),
    frozenset({"c6", "c7"}),
    frozenset({"c8"}),
}


def _store(run: Path) -> ClusterStore:
    return ClusterStore.open(run / "cluster.sqlite")


# --- outcomes -------------------------------------------------------------------------


async def test_synonyms_cocluster_related_concepts_stay_apart_and_novel_records_stand_alone(
    tmp_path,
):
    oracle = Oracle(concepts(CHOCOLATE))
    report = await cluster(records(CHOCOLATE), tmp_path / "run", oracle)

    assert report.run_state == "complete" and report.stop_reason == "converged"
    # "chocolate, dark" / "dark chocolate" / "plain chocolate" are one concept; the broader
    # "chocolate" and the related "milk chocolate" stay separate; "heart attack" and
    # "myocardial infarction" share no token and still meet (expansion scans the pool).
    assert partition(tmp_path / "run") == EXPECTED_CHOCOLATE
    rows = member_rows(tmp_path / "run")
    assert {sid for sid, r in rows.items() if r["outcome"] == "singleton"} == {"c1", "c5", "c8"}
    assert rows["c7"]["outcome"] == "assigned" and rows["c7"]["reason"] == "verified"
    assert report.counts == {
        "assigned": 5,
        "singleton": 3,
        "needs_review": 0,
        "failed": 0,
        "pending": 0,
        "total": 8,
        "clusters": 5,
        "decisions": report.counts["decisions"],
    }


async def test_every_source_is_accounted_for_once_and_accepted_clusters_partition(tmp_path):
    def flaky(name, request):
        # "plain chocolate" gets an unscored verification: deferred, never accepted
        if name == "verify-assignment" and "plain chocolate" in sections(request.user)["Record"]:
            return json.dumps({"decision": "equivalent", "confidence_score": 0.55})
        return None

    oracle = Oracle(concepts(CHOCOLATE), override=flaky)
    report = await cluster(records(CHOCOLATE), tmp_path / "run", oracle)

    rows = member_rows(tmp_path / "run")
    assert sorted(rows) == sorted(CHOCOLATE)
    assert rows["c4"]["outcome"] == "needs_review" and rows["c4"]["cluster_id"] == ""
    assert rows["c4"]["reason"] == "below_accept_threshold"
    assert rows["c4"]["proposal_cluster_id"] == rows["c2"]["cluster_id"]
    members = [m for group in partition(tmp_path / "run") for m in group]
    assert len(members) == len(set(members))
    assert set(members) == {s for s, r in rows.items() if r["outcome"] in ("assigned", "singleton")}
    with (tmp_path / "run" / "unresolved.csv").open(encoding="utf-8") as handle:
        assert [r["source_id"] for r in csv.DictReader(handle)] == ["c4"]
    with (tmp_path / "run" / "clusters.csv").open(encoding="utf-8") as handle:
        statuses = {r["member_ids"]: r["status"] for r in csv.DictReader(handle)}
    assert statuses["c1"] == "singleton"  # never described as a verified equivalence
    assert report.counts["needs_review"] == 1


async def test_a_contradictory_chain_does_not_force_a_transitive_cluster(tmp_path):
    # A~B and B~C were each judged equivalent; A~C was not. Selection is naive (any
    # member matches), so the engine is offered the chain and must refuse to close it.
    table = {"a": ("alpha", "x"), "b": ("alpha beta", "x"), "c": ("beta", "x")}
    oracle = Oracle(
        concepts(table),
        pairs=[("alpha", "alpha beta"), ("alpha beta", "beta")],
        naive_select=True,
    )
    await cluster(records(table), tmp_path / "run", oracle)

    groups = partition(tmp_path / "run")
    assert frozenset({"a", "b", "c"}) not in groups
    assert not any({"a", "c"} <= group for group in groups)
    store = _store(tmp_path / "run")
    try:
        kinds = [(d["subject_id"], d["kind"], d["result"]) for d in store.decisions()]
        merges = store.merges()
    finally:
        store.close()
    # "beta" was proposed the {alpha, alpha beta} cluster and rejected by verification
    assert (
        "c",
        "verify",
        {"decision": "not_equivalent", "cluster_id": groups_id(tmp_path, "a")},
    ) in [(s, k, r) for s, k, r in kinds]
    assert all(m["outcome"] != "applied" for m in merges)


def groups_id(run_dir: Path, source_id: str) -> str:
    return member_rows(run_dir / "run")[source_id]["cluster_id"]


async def test_novelty_needs_the_configured_expansion_first(tmp_path):
    # Without scanning, "myocardial infarction" never sees the heart-attack cluster.
    # With mint_requires=pool_exhausted it may not mint: a lexical miss is not novelty.
    oracle = Oracle(concepts(CHOCOLATE))
    config = settings(scan_below=0, mint_requires="pool_exhausted", max_refine_iterations=0)
    await cluster(records(CHOCOLATE), tmp_path / "run", oracle, settings=config)
    rows = member_rows(tmp_path / "run")
    assert rows["c7"]["outcome"] == "needs_review"
    assert rows["c7"]["reason"] == "expansion_incomplete:retrieval_exhausted"
    assert rows["c1"]["reason"] == "minted:empty_pool"
    novelty_subjects = [
        sections(r.user)["Record"] for r in oracle.llm.requests if "verify-novelty" in r.user
    ]
    assert "myocardial infarction" not in novelty_subjects


async def test_mint_provenance_records_the_search_that_preceded_it(tmp_path):
    oracle = Oracle(concepts(CHOCOLATE))
    config = settings(scan_below=0, mint_requires="retrieval_exhausted", max_refine_iterations=0)
    await cluster(records(CHOCOLATE), tmp_path / "run", oracle, settings=config)
    rows = member_rows(tmp_path / "run")
    # with no scan the heart-attack synonym mints its own cluster, labelled honestly
    assert rows["c7"]["reason"] == "minted:retrieval_exhausted"
    with (tmp_path / "run" / "clusters.csv").open(encoding="utf-8") as handle:
        provenance = {r["seed_id"]: r["mint_provenance"] for r in csv.DictReader(handle)}
    assert provenance["c1"] == "empty_pool" and provenance["c2"] == "pool_exhausted"


async def test_unparseable_and_failing_calls_fail_the_source_not_the_run(tmp_path):
    def broken(name, request):
        record = sections(request.user).get("Record", "")
        if "quinoa" in record:
            raise LLMRetryableError("upstream 503")
        if "milk chocolate" in record and name == "select-cluster":
            return "I think it is C01"
        return None

    oracle = Oracle(concepts(CHOCOLATE), override=broken)
    report = await cluster(records(CHOCOLATE), tmp_path / "run", oracle)
    rows = member_rows(tmp_path / "run")
    assert rows["c8"]["outcome"] == "failed" and rows["c8"]["reason"] == "provider_failure"
    assert rows["c5"]["outcome"] == "failed" and rows["c5"]["reason"] == "unparseable_output"
    assert report.run_state == "failed"
    assert {e["source_id"] for e in report.errors} == {"c5", "c8"}
    # retried in every refinement iteration, then reported, never silently assigned
    assert oracle.calls["select-cluster"] >= 3


async def test_a_fatal_provider_error_aborts_at_a_step_boundary(tmp_path):
    def fatal(name, request):
        if "heart attack" in sections(request.user).get("Record", ""):
            raise LLMFatalError("401 unauthorized")
        return None

    report = await cluster(
        records(CHOCOLATE), tmp_path / "run", Oracle(concepts(CHOCOLATE), override=fatal)
    )
    assert report.run_state == "aborted"
    assert report.errors[0]["code"] == "fatal_provider_failure"
    rows = member_rows(tmp_path / "run")
    assert rows["c6"]["outcome"] == "pending" and rows["c8"]["outcome"] == "pending"
    assert rows["c3"]["outcome"] == "assigned"


# --- merges and refinement ------------------------------------------------------------


def _wrongly_novel(*labels: str):
    """A judge that wrongly calls these records novel, creating duplicate clusters."""

    def override(name, request):
        subject = sections(request.user).get("Record")
        if subject not in labels:
            return None
        if name == "verify-novelty":
            return json.dumps({"novel": True, "confidence_score": 0.95, "explanation": "x"})
        if name == "select-cluster":
            return json.dumps({"chosen_key": None, "confidence_score": 0.9, "explanation": "x"})
        return None

    return override


async def test_duplicate_clusters_merge_by_a_verified_cluster_decision(tmp_path):
    oracle = Oracle(concepts(CHOCOLATE), override=_wrongly_novel("dark chocolate"))
    await cluster(records(CHOCOLATE), tmp_path / "run", oracle, settings=settings(reassign=False))

    assert partition(tmp_path / "run") == EXPECTED_CHOCOLATE
    store = _store(tmp_path / "run")
    try:
        applied = [m for m in store.merges() if m["outcome"] == "applied"]
        latest = store.latest_revisions()
    finally:
        store.close()
    assert len(applied) == 1
    loser = (
        applied[0]["left_id"]
        if applied[0]["winner_id"] != applied[0]["left_id"]
        else applied[0]["right_id"]
    )
    assert (
        latest[loser]["live"] is False and latest[loser]["merged_into"] == applied[0]["winner_id"]
    )
    rows = member_rows(tmp_path / "run")
    # the smaller duplicate's members moved by the merge decision, not by a new select
    assert rows["c2"]["cluster_id"] == rows["c3"]["cluster_id"] == applied[0]["winner_id"]
    assert "merged" in {rows["c2"]["reason"], rows["c3"]["reason"], rows["c4"]["reason"]}


async def test_a_cluster_takes_at_most_one_merge_per_iteration(tmp_path):
    # three duplicate clusters of one concept: no chain of merges inside one iteration;
    # the third joins in the next iteration, judged against the merged representation.
    table = {
        "d1": ("chocolate, dark", "dark"),
        "d2": ("dark chocolate", "dark"),
        "d3": ("dark chocolate bar", "dark"),
    }
    oracle = Oracle(
        concepts(table), override=_wrongly_novel("dark chocolate", "dark chocolate bar")
    )
    report = await cluster(
        records(table), tmp_path / "run", oracle, settings=settings(max_refine_iterations=3)
    )

    assert partition(tmp_path / "run") == {frozenset({"d1", "d2", "d3"})}
    store = _store(tmp_path / "run")
    try:
        applied = [m for m in store.merges() if m["outcome"] == "applied"]
        merge_decisions = {d["decision_id"]: d for d in store.decisions() if d["kind"] == "merge"}
    finally:
        store.close()
    assert [m["iteration"] for m in applied] == [1, 2]
    second = merge_decisions[applied[1]["decision_id"]]
    shown_sizes = [len(blocks(second["prompt"].split("## Cluster A", 1)[1])[i][1]) for i in (0, 1)]
    assert sorted(shown_sizes) == [1, 2]  # the merged cluster, shown with its members
    assert report.stop_reason == "converged"


async def test_reassignment_injects_the_incumbent_when_the_shown_list_is_full(tmp_path):
    # "Alpha." and "ALPHA!" are homonyms of "alpha" (different concepts) with shorter
    # documents, so they outrank alpha's own cluster; with two slots the incumbent must
    # replace the lowest-ranked challenger rather than disappear.
    table = {
        "a1": ("alpha", "A"),
        "a2": ("omega", "A"),
        "b1": ("Alpha.", "B"),
        "c1": ("ALPHA!", "C"),
    }
    oracle = Oracle(concepts(table))
    await cluster(records(table), tmp_path / "run", oracle, settings=settings(shown_limit=2))
    store = _store(tmp_path / "run")
    try:
        reassign = [
            d
            for d in store.decisions()
            if d["subject_id"] == "a1" and d["result"].get("incumbent") is not None
        ]
    finally:
        store.close()
    assert reassign, "a1 was reconsidered"
    decision = reassign[0]
    incumbent = decision["result"]["incumbent"]
    assert decision["result"]["incumbent_injected"] is True
    assert len(decision["shown"]) == 2
    assert decision["shown"][-1] == {
        "key": "C02",
        "cluster_id": incumbent,
        "revision": decision["shown"][-1]["revision"],
        "exclude": "a1",
    }
    assert [h["cluster_id"] for h in decision["retrieved"]][:2] != [
        s["cluster_id"] for s in decision["shown"]
    ]
    assert decision["result"]["chosen"] == incumbent  # kept
    assert partition(tmp_path / "run") == {
        frozenset({"a1", "a2"}),
        frozenset({"b1"}),
        frozenset({"c1"}),
    }


def _flipper():
    """Moves 'red fruit salad' to the other cluster at every reassignment. Until it has
    joined P (stream), the honest oracle answers."""
    where: dict[str, str | None] = {"current": None}

    def override(name, request):
        parts = sections(request.user)
        if parts.get("Record") != "red fruit salad":
            return None
        if where["current"] is None:
            if name == "verify-assignment":
                proposed = blocks(parts["Proposed cluster"])[0][1]
                where["current"] = "Q" if any("salad" in m for m in proposed) else "P"
            return None
        if name == "select-cluster" and "Candidate clusters" in parts:
            for key, members in blocks(parts["Candidate clusters"]):
                in_q = any("salad" in m for m in members)
                if (where["current"] == "P") == in_q:
                    return json.dumps(
                        {"chosen_key": key, "confidence_score": 0.9, "explanation": "x"}
                    )
        if name == "compare-clusters":
            where["current"] = "Q" if where["current"] == "P" else "P"
            return json.dumps(
                {"preferred": "alternative", "confidence_score": 0.95, "explanation": "x"}
            )
        return None

    return override


FRUIT = {
    "p1": ("red fruit", "P"),
    "p2": ("red berry", "P"),
    "q1": ("fruit salad", "Q"),
    "q2": ("berry salad", "Q"),
    "x": ("red fruit salad", "P"),
}
# "red fruit salad" is judged equivalent to every record (an ambiguous label), the two
# groups are not equivalent to each other: x fits either cluster and neither merge.
FRUIT_PAIRS = [
    ("red fruit", "red berry"),
    ("fruit salad", "berry salad"),
    *[("red fruit salad", label) for label, _ in FRUIT.values() if label != "red fruit salad"],
]


async def test_oscillation_stops_refinement_and_is_reported(tmp_path):
    oracle = Oracle(concepts(FRUIT), pairs=FRUIT_PAIRS, override=_flipper())
    report = await cluster(
        records(FRUIT), tmp_path / "run", oracle, settings=settings(max_refine_iterations=5)
    )
    assert report.stop_reason == "oscillation"
    assert report.last_revision == 2 and report.selected_revision == 0
    assert (
        member_rows(tmp_path / "run")["x"]["cluster_id"]
        == member_rows(tmp_path / "run")["p1"]["cluster_id"]
    )


async def test_exports_use_the_selected_revision_not_the_last_transient_state(tmp_path):
    oracle = Oracle(concepts(FRUIT), pairs=FRUIT_PAIRS, override=_flipper())
    report = await cluster(
        records(FRUIT), tmp_path / "run", oracle, settings=settings(max_refine_iterations=1)
    )
    assert report.stop_reason == "max_iterations"
    assert report.last_revision == 1 and report.selected_revision == 0
    assert report.exported_revision == 0
    rows = member_rows(tmp_path / "run")
    assert rows["x"]["cluster_id"] == rows["p1"]["cluster_id"]  # revision 0
    store = _store(tmp_path / "run")
    try:
        current = store.current_assignments()["x"]["cluster_id"]  # the transient state
    finally:
        store.close()
    assert current == rows["q1"]["cluster_id"] != rows["x"]["cluster_id"]


async def test_refinement_converges_and_never_mints(tmp_path):
    oracle = Oracle(concepts(CHOCOLATE), override=_wrongly_novel("dark chocolate"))
    report = await cluster(records(CHOCOLATE), tmp_path / "run", oracle)
    store = _store(tmp_path / "run")
    try:
        # cluster_revisions columns: cluster_id, revision, ..., step_key (index 9)
        first_revisions = [row for row in store.dump("cluster_revisions") if row[1] == 1]
        steps = [row[1] for row in store.dump("steps")]
    finally:
        store.close()
    assert {row[9].split("/")[0] for row in first_revisions} <= {"stream", "retry"}
    assert any(step.startswith("reassign/") for step in steps)
    assert any(step.startswith("consolidate/") for step in steps)
    assert report.stop_reason == "converged"


# --- determinism and order -----------------------------------------------------------


def _dump(run: Path) -> dict[str, list[tuple]]:
    store = _store(run)
    try:
        from xwalk.cluster.store import DETERMINISTIC_TABLES

        return {table: store.dump(table) for table in DETERMINISTIC_TABLES}
    finally:
        store.close()


def _exports(run: Path) -> dict[str, str]:
    names = ("members.csv", "clusters.csv", "unresolved.csv", "decisions.jsonl")
    return {name: (run / name).read_text(encoding="utf-8") for name in names}


async def test_fixed_inputs_responses_and_order_give_identical_ids_and_results(tmp_path):
    first = await cluster(records(CHOCOLATE), tmp_path / "a", Oracle(concepts(CHOCOLATE)))
    second = await cluster(records(CHOCOLATE), tmp_path / "b", Oracle(concepts(CHOCOLATE)))
    assert first.run_fingerprint == second.run_fingerprint
    assert _dump(tmp_path / "a") == _dump(tmp_path / "b")
    assert _exports(tmp_path / "a") == _exports(tmp_path / "b")


async def test_label_order_makes_input_permutation_irrelevant(tmp_path):
    shuffled = records(CHOCOLATE)
    random.Random(3).shuffle(shuffled)
    first = await cluster(records(CHOCOLATE), tmp_path / "a", Oracle(concepts(CHOCOLATE)))
    second = await cluster(shuffled, tmp_path / "b", Oracle(concepts(CHOCOLATE)))
    assert first.run_fingerprint == second.run_fingerprint
    assert _exports(tmp_path / "a") == _exports(tmp_path / "b")


async def test_input_order_is_part_of_identity_and_can_change_results(tmp_path):
    """Documented behaviour: with `order: input` a permutation is a different run. Clear
    synonyms still meet, but cluster ids follow the seed, and a contradictory chain
    resolves differently depending on which record arrives first."""
    clear = records(CHOCOLATE)
    reversed_clear = list(reversed(clear))
    config = settings(order="input")
    a = await cluster(clear, tmp_path / "a", Oracle(concepts(CHOCOLATE)), settings=config)
    b = await cluster(reversed_clear, tmp_path / "b", Oracle(concepts(CHOCOLATE)), settings=config)
    assert a.run_fingerprint != b.run_fingerprint
    assert a.components["snapshot"] != b.components["snapshot"]
    assert partition(tmp_path / "a") == partition(tmp_path / "b") == EXPECTED_CHOCOLATE
    assert (
        member_rows(tmp_path / "a")["c2"]["cluster_id"]
        != member_rows(tmp_path / "b")["c2"]["cluster_id"]
    )

    table = {"a": ("alpha", "x"), "b": ("alpha beta", "x"), "c": ("beta", "x")}
    pairs = [("alpha", "alpha beta"), ("alpha beta", "beta")]
    chain = records(table)
    await cluster(
        chain,
        tmp_path / "c1",
        Oracle(concepts(table), pairs=pairs, naive_select=True),
        settings=config,
    )
    await cluster(
        [chain[2], chain[1], chain[0]],
        tmp_path / "c2",
        Oracle(concepts(table), pairs=pairs, naive_select=True),
        settings=config,
    )
    # whichever end of the chain arrives first keeps the middle; the other end is left
    # for review rather than forced into either cluster
    assert partition(tmp_path / "c1") == {frozenset({"a", "b"})}
    assert member_rows(tmp_path / "c1")["c"]["outcome"] == "needs_review"
    assert partition(tmp_path / "c2") == {frozenset({"c", "b"})}
    assert member_rows(tmp_path / "c2")["a"]["outcome"] == "needs_review"


def test_label_order_is_casefolded_whitespace_normalised_then_by_id():
    from xwalk.records import Record

    items = [
        Record(id="2", fields={"label": "Beta"}),
        Record(id="1", fields={"label": "beta "}),
        Record(id="0", fields={"label": "  ALPHA"}),
    ]
    assert [r.id for r in order_sources(items, TEMPLATES)] == ["0", "1", "2"]
    assert [r.id for r in order_sources(items, TEMPLATES, "input")] == ["2", "1", "0"]


# --- interruption and resume ---------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 6, 14, 21, 25])
async def test_resume_after_a_call_limit_reproduces_a_clean_run(tmp_path, limit):
    """The call limit stops the run mid-step at different phases (stream, refinement);
    a resumed run must end exactly where an uninterrupted one does."""
    override = _wrongly_novel("dark chocolate")  # gives consolidation work too
    clean = await cluster(
        records(CHOCOLATE), tmp_path / "clean", Oracle(concepts(CHOCOLATE), override=override)
    )
    total = clean.usage.calls
    assert total > 25

    stopped = await cluster(
        records(CHOCOLATE),
        tmp_path / "resumed",
        Oracle(concepts(CHOCOLATE), override=override),
        max_calls=limit,
    )
    assert stopped.run_state == "aborted"
    assert stopped.errors[0]["code"] == "call_limit_reached"
    assert stopped.usage.calls == limit
    resumed = await cluster(
        records(CHOCOLATE), tmp_path / "resumed", Oracle(concepts(CHOCOLATE), override=override)
    )
    assert resumed.run_state == "complete"
    assert _dump(tmp_path / "clean") == _dump(tmp_path / "resumed")
    assert _exports(tmp_path / "clean") == _exports(tmp_path / "resumed")


async def test_resume_after_keyboard_interrupt_reproduces_a_clean_run(tmp_path):
    clean = await cluster(records(CHOCOLATE), tmp_path / "clean", Oracle(concepts(CHOCOLATE)))
    count = {"n": 0}

    def interrupt(name, request):
        count["n"] += 1
        if count["n"] == 9:
            raise KeyboardInterrupt
        return None

    with pytest.raises(KeyboardInterrupt):
        await cluster(
            records(CHOCOLATE), tmp_path / "run", Oracle(concepts(CHOCOLATE), override=interrupt)
        )
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_state"] == "interrupted"
    assert manifest["counts"]["pending"] > 0
    resumed = await cluster(records(CHOCOLATE), tmp_path / "run", Oracle(concepts(CHOCOLATE)))
    assert resumed.run_fingerprint == clean.run_fingerprint
    assert _dump(tmp_path / "clean") == _dump(tmp_path / "run")


async def test_a_complete_run_repeated_makes_no_calls_and_identical_exports(tmp_path):
    await cluster(records(CHOCOLATE), tmp_path / "run", Oracle(concepts(CHOCOLATE)))
    before = _exports(tmp_path / "run")
    again = Oracle(concepts(CHOCOLATE))
    report = await cluster(records(CHOCOLATE), tmp_path / "run", again)
    assert report.usage.calls == 0 and not again.llm.requests
    assert _exports(tmp_path / "run") == before


# --- bounds -------------------------------------------------------------------------


async def test_call_and_candidate_bounds_hold_under_refinement(tmp_path):
    config = settings(
        shown_limit=2, member_evidence=2, max_expansion_pages=1, max_refine_iterations=3
    )
    oracle = Oracle(
        concepts(CHOCOLATE), override=_wrongly_novel("dark chocolate", "plain chocolate")
    )
    report = await cluster(records(CHOCOLATE), tmp_path / "run", oracle, settings=config)

    store = _store(tmp_path / "run")
    try:
        decisions = list(store.decisions())
    finally:
        store.close()
    per_step: dict[str, int] = {}
    for d in decisions:
        per_step[d["step_key"]] = per_step.get(d["step_key"], 0) + d["usage"]["calls"]
        if d["kind"] in ("select", "merge_select", "novelty"):
            keyed = [s for s in d["shown"] if s["key"] is not None]
            assert len(keyed) <= 2
        for _, members in blocks(d["prompt"] or ""):
            assert len(members) <= 2  # bounded member evidence
    for key, calls in per_step.items():
        bound = config.max_calls_per_member_decision if key.startswith(("stream", "retry")) else 2
        assert calls <= bound, key
    # every dispatched call is in a decision, and the client agrees
    assert sum(per_step.values()) == report.usage.calls == len(oracle.llm.requests)
    assert any(k.startswith("consolidate/") for k in per_step)


async def test_the_call_limit_is_never_exceeded(tmp_path):
    oracle = Oracle(concepts(CHOCOLATE))
    report = await cluster(records(CHOCOLATE), tmp_path / "run", oracle, max_calls=5)
    assert len(oracle.llm.requests) == 5 == report.usage.calls


# --- reconstructable history --------------------------------------------------------


async def test_every_decision_prompt_can_be_rebuilt_from_stored_revisions(tmp_path):
    oracle = Oracle(concepts(CHOCOLATE), override=_wrongly_novel("dark chocolate"))
    await cluster(records(CHOCOLATE), tmp_path / "run", oracle)

    store = _store(tmp_path / "run")
    try:
        sources = {sid: fields for _, sid, _, fields in store.sources()}
        decisions = list(store.decisions())
        engine = ClusterEngine(
            store=store,
            llm=oracle.llm,
            templates=TEMPLATES,
            settings=settings(),
            prompts=ClusterPrompts(),
            run_fingerprint="x",
        )
        engine.load()  # only for its renderer and source records
        checked = 0
        for d in decisions:
            if d["prompt"] is None:
                assert d["kind"] == "mint"
                continue
            assert d["system"] == SYSTEM
            for entry in d["shown"]:
                revision = store.revision(entry["cluster_id"], entry["revision"])
                assert revision["representation"] == engine.block(revision["members"])
                rebuilt = engine.block(
                    revision["members"], key=entry["key"], exclude=entry["exclude"]
                )
                assert rebuilt in d["prompt"], (d["kind"], entry)
                checked += 1
            for hit in d["retrieved"]:
                store.revision(hit["cluster_id"], hit["revision"])  # exists
            # the subject is stored too: a record's own fields render the Record section
            if "## Record" in d["prompt"]:
                subject = sections(d["prompt"])["Record"]
                assert subject == TEMPLATES.render_context(
                    __import__("xwalk").Record(id=d["subject_id"], fields=sources[d["subject_id"]])
                )
        assert checked > 20
    finally:
        store.close()


async def test_cluster_ids_are_stable_hashes_of_run_and_seed(tmp_path):
    from xwalk.cluster.engine import cluster_id_for

    report = await cluster(records(CHOCOLATE), tmp_path / "run", Oracle(concepts(CHOCOLATE)))
    with (tmp_path / "run" / "clusters.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            assert row["cluster_id"] == cluster_id_for(report.run_fingerprint, row["seed_id"])
            assert re.fullmatch(r"K[0-9a-f]{16}", row["cluster_id"])


async def test_the_pool_index_is_updated_per_change_never_rebuilt_per_mint(tmp_path):
    table = {
        f"s{c}_{v}": (label, f"C{c}")
        for c in range(30)
        for v, label in enumerate((f"item{c} kind", f"item{c} kind variant"))
    }
    report = await cluster(records(table), tmp_path / "run", Oracle(concepts(table)))
    assert report.counts["clusters"] == 30
    store = _store(tmp_path / "run")
    try:
        revisions = len(store.dump("cluster_revisions"))
    finally:
        store.close()
    # one document update per cluster revision at most; a rebuild on every mint would
    # re-add every live cluster each time (~30 * 31 / 2 updates here)
    assert report.pool_updates <= revisions == 60
    # the shared token "kind" retrieves every cluster, so late records mint only after
    # the configured expansion pages (default mint_requires=bounded), and say so
    rows = member_rows(tmp_path / "run")
    assert rows["s9_0"]["reason"] == "minted:bounded"


async def test_a_stricter_mint_requirement_defers_instead(tmp_path):
    table = {
        f"s{c}_{v}": (label, f"C{c}")
        for c in range(30)
        for v, label in enumerate((f"item{c} kind", f"item{c} kind variant"))
    }
    config = settings(mint_requires="retrieval_exhausted", max_refine_iterations=0)
    await cluster(records(table), tmp_path / "run", Oracle(concepts(table)), settings=config)
    rows = member_rows(tmp_path / "run")
    assert rows["s9_0"]["outcome"] == "needs_review"
    assert rows["s9_0"]["reason"] == "expansion_incomplete:bounded"

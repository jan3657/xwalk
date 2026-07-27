import csv
import json

import pytest

from tests.test_matcher import (
    PROMPTS,
    STORE,
    TEMPLATES,
    ScriptedRetriever,
    score_reply,
    select_reply,
)
from xwalk.batch import build_run_fingerprint, export_mapping_csv, run_batch
from xwalk.llm.fake import FakeLLM
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.records import MatchStatus, Record
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.select import Selector, SelectorPolicy

SOURCES = [
    Record(id="s1", fields={"mention": "glucose"}),
    Record(id="s2", fields={"mention": "fructose"}),
]


def make_matcher(llm, retriever, *, run_fp="fp1", policy=None):
    policy = policy or MatchPolicy()
    return Matcher(
        templates=TEMPLATES,
        retrievers=[retriever],
        store=STORE,
        selector=Selector(llm, PROMPTS, TEMPLATES),
        scorer=Scorer(llm, PROMPTS, TEMPLATES),
        verifier=Verifier(llm, PROMPTS, TEMPLATES),
        rewriter=QueryRewriter(llm, PROMPTS, TEMPLATES),
        policy=policy,
        run_fingerprint=run_fp,
    )


def two_good_matches() -> FakeLLM:
    def handler(request):
        if "## Candidates" in request.user:
            return select_reply("C01")
        return score_reply(0.95)

    return FakeLLM(handler=handler)


RETRIEVER = {"glucose": ["T1"], "fructose": ["T2"]}


# --- fingerprint -----------------------------------------------------------------


def test_fingerprint_changes_with_the_policy():
    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        selector_policy=SelectorPolicy(),
    )
    a = build_run_fingerprint(policy=MatchPolicy(accept_at=0.6), **kwargs)
    b = build_run_fingerprint(policy=MatchPolicy(accept_at=0.7), **kwargs)
    assert a != b


def test_fingerprint_changes_with_the_selector_budget():
    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        policy=MatchPolicy(),
    )
    a = build_run_fingerprint(selector_policy=SelectorPolicy(max_candidates=30), **kwargs)
    b = build_run_fingerprint(selector_policy=SelectorPolicy(max_candidates=10), **kwargs)
    assert a != b


def test_fingerprint_changes_with_the_target_snapshot():
    from xwalk.stores.memory import MemoryStore

    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        policy=MatchPolicy(),
        selector_policy=SelectorPolicy(),
    )
    other = MemoryStore.from_source([Record(id="T1", fields={"label": "changed"})])
    assert build_run_fingerprint(store=STORE, **kwargs) != build_run_fingerprint(
        store=other, **kwargs
    )


def test_fingerprint_is_stable_across_identical_configurations():
    kwargs = dict(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever(RETRIEVER)],
        llm=FakeLLM(["x"]),
        policy=MatchPolicy(),
        selector_policy=SelectorPolicy(),
    )
    assert build_run_fingerprint(**kwargs) == build_run_fingerprint(**kwargs)


# --- running ----------------------------------------------------------------------


async def test_every_source_record_produces_a_result(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert report.total == 2


async def test_results_are_grouped_by_status(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert report.by_status() == {MatchStatus.MATCHED: 2}


async def test_usage_is_aggregated(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert report.usage.calls == 4  # two records, select + score each


async def test_the_ledger_file_is_created(tmp_path):
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert (tmp_path / "run" / "ledger.sqlite").exists()


async def test_a_progress_callback_fires_once_per_record(tmp_path):
    seen = []
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    await run_batch(matcher, SOURCES, out=tmp_path / "run", progress=seen.append)
    assert len(seen) == 2


async def test_concurrency_is_bounded_by_the_policy(tmp_path):
    import asyncio

    live = {"now": 0, "max": 0}

    class CountingRetriever(ScriptedRetriever):
        async def search(self, request):
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            await asyncio.sleep(0.01)
            try:
                return await super().search(request)
            finally:
                live["now"] -= 1

    many = [Record(id=f"s{i}", fields={"mention": "glucose"}) for i in range(20)]
    matcher = make_matcher(
        two_good_matches(), CountingRetriever(RETRIEVER), policy=MatchPolicy(concurrency=3)
    )
    await run_batch(matcher, many, out=tmp_path / "run")
    assert live["max"] <= 3


# --- resume -----------------------------------------------------------------------


async def test_resume_skips_records_already_completed(tmp_path):
    out = tmp_path / "run"
    first = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), first), SOURCES, out=out)

    second = ScriptedRetriever(RETRIEVER)
    report = await run_batch(
        make_matcher(two_good_matches(), second), SOURCES, out=out, resume=True
    )
    assert second.queries == []  # nothing re-run
    assert report.total == 2


async def test_resume_false_reruns_everything(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    second = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), second), SOURCES, out=out, resume=False)
    assert len(second.queries) == 2


async def test_a_changed_source_record_is_reprocessed_on_resume(tmp_path):
    """Same id, different content — the prior result is not about this record."""
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    changed = [Record(id="s1", fields={"mention": "glucose", "note": "new"}), SOURCES[1]]
    second = ScriptedRetriever(RETRIEVER)
    await run_batch(make_matcher(two_good_matches(), second), changed, out=out, resume=True)
    assert second.queries == ["glucose"]


async def test_a_changed_run_fingerprint_forces_a_fresh_run(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER), run_fp="fp1"),
        SOURCES,
        out=out,
    )
    second = ScriptedRetriever(RETRIEVER)
    await run_batch(
        make_matcher(two_good_matches(), second, run_fp="fp2"), SOURCES, out=out, resume=True
    )
    assert len(second.queries) == 2


async def test_completed_work_survives_a_crash_mid_run(tmp_path):
    """The point of committing each result as it finishes."""
    out = tmp_path / "run"
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated crash")
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.95)

    matcher = make_matcher(
        FakeLLM(handler=handler), ScriptedRetriever(RETRIEVER), policy=MatchPolicy(concurrency=1)
    )
    with pytest.raises(RuntimeError):
        await run_batch(matcher, SOURCES, out=out)

    resumed = ScriptedRetriever(RETRIEVER)
    report = await run_batch(
        make_matcher(two_good_matches(), resumed), SOURCES, out=out, resume=True
    )
    assert report.total == 2
    assert len(resumed.queries) == 1  # only the unfinished record re-ran


# --- reporting --------------------------------------------------------------------


async def test_duplicate_targets_are_reported_not_resolved(tmp_path):
    """Reporting is not solving. The library surfaces the conflict and does nothing."""
    both_to_t1 = [
        Record(id="s1", fields={"mention": "glucose"}),
        Record(id="s2", fields={"mention": "glucose"}),
    ]
    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, both_to_t1, out=tmp_path / "run")
    assert report.duplicate_targets() == {"T1": ["s1", "s2"]}
    assert report.by_status() == {MatchStatus.MATCHED: 2}  # both still matched


async def test_needs_review_lists_the_review_bucket(tmp_path):
    def handler(request):
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.5)

    matcher = make_matcher(FakeLLM(handler=handler), ScriptedRetriever(RETRIEVER))
    report = await run_batch(matcher, SOURCES, out=tmp_path / "run")
    assert {r.source_id for r in report.needs_review()} == {"s1", "s2"}


# --- exports ----------------------------------------------------------------------


async def test_mapping_csv_has_one_row_per_source_record(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    rows = list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))
    assert len(rows) == 2


async def test_mapping_csv_columns_are_the_documented_set(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    row = next(iter(csv.DictReader((out / "mapping.csv").open(encoding="utf-8"))))
    assert list(row) == [
        "source_id",
        "matched_id",
        "confidence",
        "status",
        "reason",
        "explanation",
        "attempts",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
    ]


async def test_results_jsonl_has_one_object_per_line(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    lines = (out / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["source_id"] == "s1"


async def test_the_manifest_records_the_run_fingerprint_and_components(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_fingerprint"] == "fp1"
    assert "target_fingerprint" in manifest
    assert "counts" in manifest


async def test_the_manifest_never_contains_an_api_key(tmp_path):
    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)),
        SOURCES,
        out=out,
        manifest_extra={"model": "fake"},
    )
    text = (out / "manifest.json").read_text(encoding="utf-8").lower()
    assert "api_key" not in text and "authorization" not in text


async def test_exports_are_regenerable_from_the_ledger_alone(tmp_path):
    from xwalk.ledger import Ledger

    out = tmp_path / "run"
    await run_batch(
        make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )
    (out / "mapping.csv").unlink()
    ledger = Ledger.open(out / "ledger.sqlite")
    export_mapping_csv(ledger, "fp1", out / "mapping.csv")
    ledger.close()
    assert len(list(csv.DictReader((out / "mapping.csv").open(encoding="utf-8")))) == 2


async def test_mapping_csv_can_be_written_from_the_adjudicated_view(tmp_path):
    from xwalk.ledger import Ledger
    from xwalk.review import ReviewDecision, ReviewRow, apply_review

    out = tmp_path / "run"

    def handler(request):
        return select_reply("C01") if "## Candidates" in request.user else score_reply(0.5)

    await run_batch(
        make_matcher(FakeLLM(handler=handler), ScriptedRetriever(RETRIEVER)), SOURCES, out=out
    )

    ledger = Ledger.open(out / "ledger.sqlite")
    result = next(iter(ledger.iter_results("fp1")))
    apply_review(
        ledger,
        [
            ReviewRow(
                result_key=result.result_key,
                run_fingerprint="fp1",
                source_id=result.source_id,
                source_hash=result.source_hash,
                proposed_target_id=result.matched_id,
                decision=ReviewDecision.ACCEPT,
                corrected_target_id=None,
                reviewer="jan",
                review_note="",
                reviewed_at="2026-07-26T10:00:00Z",
            )
        ],
        target_store_fingerprint=STORE.fingerprint,
    )
    export_mapping_csv(ledger, "fp1", out / "adjudicated.csv", use_review=True)
    ledger.close()

    rows = {
        r["source_id"]: r for r in csv.DictReader((out / "adjudicated.csv").open(encoding="utf-8"))
    }
    assert rows[result.source_id]["status"] == "matched"


# --- sync facade ------------------------------------------------------------------


def test_run_batch_sync_works_outside_an_event_loop(tmp_path):
    from xwalk.batch import run_batch_sync

    matcher = make_matcher(two_good_matches(), ScriptedRetriever(RETRIEVER))
    report = run_batch_sync(matcher, SOURCES, out=tmp_path / "run")
    assert report.total == 2


async def test_two_csvs_in_a_mapping_table_out(tmp_path, targets_csv, sources_csv):
    from xwalk.retrieval.bm25 import BM25Retriever
    from xwalk.sources.tabular import csv_source
    from xwalk.stores.memory import MemoryStore
    from xwalk.templates import TemplateSet

    templates = TemplateSet(
        query="{{ mention }}",
        context="{{ context_left }} [{{ mention }}] {{ context_right }}",
        doc="{{ label }} {{ synonyms | join(' ') }}",
        candidate="ID: {{ id }} Label: {{ label }}",
    )
    targets = list(csv_source(targets_csv, id_column="id", multivalue_columns=["synonyms"]))
    store = MemoryStore.from_source(targets)
    retriever = BM25Retriever.build(
        targets, templates, tmp_path / "idx", exact_fields=("label", "synonyms")
    )

    def handler(request):
        if "## Candidates" in request.user:
            return select_reply("C01") if "[C01]" in request.user else select_reply(None)
        return score_reply(0.95)

    llm = FakeLLM(handler=handler)
    matcher = Matcher(
        templates=templates,
        retrievers=[retriever],
        store=store,
        selector=Selector(llm, PROMPTS, templates),
        scorer=Scorer(llm, PROMPTS, templates),
        verifier=Verifier(llm, PROMPTS, templates),
        rewriter=QueryRewriter(llm, PROMPTS, templates),
        policy=MatchPolicy(max_attempts=1),
        run_fingerprint="e2e",
    )
    report = await run_batch(
        matcher, csv_source(sources_csv, id_column="mention_id"), out=tmp_path / "run"
    )

    assert report.total == 4
    rows = {
        r["source_id"]: r
        for r in csv.DictReader((tmp_path / "run" / "mapping.csv").open(encoding="utf-8"))
    }
    assert rows["s1"]["matched_id"] == "CHEBI:17234"  # glucose
    assert rows["s2"]["matched_id"] == "CHEBI:17234"  # dextrose, via synonym
    assert rows["s4"]["status"] in ("unmatched", "needs_review")  # unobtainium

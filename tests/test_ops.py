"""The operations layer (task 03): validate, index, run, inspect, explain, export, init.

Each test drives `xwalk.ops` directly with FakeLLM; the CLI tests check only formatting.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from tests.conftest import FIXTURES
from tests.test_matcher import score_reply, select_reply
from xwalk import ops
from xwalk.config import JobSpec, load_job
from xwalk.llm.fake import FakeLLM

JOB = FIXTURES / "job_tiny.yaml"
BASE = yaml.safe_load(JOB.read_text(encoding="utf-8"))


def _llm(score: float = 0.95) -> FakeLLM:
    def handler(request):
        if "## Candidates" not in request.user:
            return score_reply(score)
        return select_reply("C01") if "[C01]" in request.user else select_reply(None)

    return FakeLLM(handler=handler)


def _job_file(tmp_path: Path, edit=None, *, name: str = "job.yaml") -> Path:
    """A copy of the tiny job next to copies of its data, optionally edited."""
    data = copy.deepcopy(BASE)
    data["prompts"]["slots"] = str((FIXTURES / data["prompts"]["slots"]).resolve())
    for role in ("target", "source"):
        source = FIXTURES / data[role]["path"]
        copied = tmp_path / source.name
        if not copied.exists():
            copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if edit is not None:
        edit(data)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- validate -----------------------------------------------------------------------


def test_validate_accepts_the_fixture_and_reports_the_credential_by_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print-me")
    result = ops.validate(JOB)
    assert result.exit_code == 0, result.errors
    assert result.counts == {"targets": 5, "sources": 4}
    assert result.data["credentials"] == {"OPENAI_API_KEY": "set"}
    assert "sk-do-not-print-me" not in json.dumps(result.envelope())


def test_validate_makes_no_model_call_and_builds_no_client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validate must not build or call an LLM client")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(JobSpec, "build_llm", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    result = ops.validate(JOB)
    assert result.exit_code == 0
    assert result.data["paid_calls"] == 0


def test_validate_reports_a_missing_credential(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = ops.validate(JOB)
    assert result.exit_code == 2
    assert [e.code for e in result.errors] == ["credential_missing"]
    assert ops.validate(JOB, check_credentials=False).exit_code == 0


def test_a_misspelled_key_fails_preflight(tmp_path):
    path = _job_file(tmp_path, lambda d: d["policy"].__setitem__("acept_at", 0.9))
    result = ops.validate(path, check_credentials=False)
    assert result.exit_code == 2
    assert result.errors[0].code == "unknown_field"
    assert "policy.acept_at" in result.errors[0].message
    assert "accept_at" in result.errors[0].message


def test_validate_rejects_duplicate_source_ids(tmp_path):
    path = _job_file(tmp_path)
    sources = tmp_path / "sources_tiny.csv"
    sources.write_text(
        sources.read_text(encoding="utf-8") + "s1,glucose again,,\n", encoding="utf-8"
    )
    result = ops.validate(path, check_credentials=False)
    assert result.exit_code == 2
    assert [e.code for e in result.errors] == ["duplicate_id"]
    assert "'s1'" in result.errors[0].message


def test_validate_names_a_missing_file(tmp_path):
    path = _job_file(tmp_path, lambda d: d["target"].__setitem__("path", "gone.csv"))
    result = ops.validate(path, check_credentials=False)
    assert [e.code for e in result.errors] == ["missing_file"]


def test_validate_names_an_unavailable_optional_extra(tmp_path, monkeypatch):
    real = __import__("importlib.util").util.find_spec

    def find_spec(name, *args):
        return None if name == "sentence_transformers" else real(name, *args)

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    path = _job_file(
        tmp_path, lambda d: d["retrievers"].append({"kind": "dense", "model": "some/model"})
    )
    result = ops.validate(path, check_credentials=False)
    assert result.exit_code == 2
    assert [e.code for e in result.errors] == ["missing_extra"]
    assert "xwalk[dense]" in result.errors[0].message


# --- index identity (CONTRACTS.md section 7) ----------------------------------------


class CountingEncoder:
    name = "toy"
    dimension = 26

    def __init__(self, **settings):
        self.settings = {
            "revision": "unknown",
            "normalize": True,
            "query_prefix": "",
            "doc_prefix": "",
            "max_seq_length": 128,
            **settings,
        }
        self.encoded = 0

    def encode(self, texts, *, is_query=False):
        self.encoded += len(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def _dense_job(tmp_path: Path, **retriever) -> Path:
    spec = {"kind": "dense", "name": "dense", "model": "toy", **retriever}
    return _job_file(tmp_path, lambda d: d.__setitem__("retrievers", [spec]))


def test_an_unchanged_index_is_opened_without_calling_the_encoder(tmp_path):
    encoder = CountingEncoder()
    job = _dense_job(tmp_path)
    first = ops.index(job, tmp_path / "idx", encoder_factory=lambda spec: encoder)
    assert first.data["retrievers"][0]["action"] == "built"
    built_calls = encoder.encoded
    assert built_calls == 5

    second = ops.index(job, tmp_path / "idx", encoder_factory=lambda spec: encoder)
    assert second.data["retrievers"][0]["action"] == "opened"
    assert encoder.encoded == built_calls  # not one more text encoded


@pytest.mark.parametrize(
    "setting",
    [{"query_prefix": "query: "}, {"normalize": False}, {"revision": "abc123"}],
)
def test_a_changed_encoder_setting_is_refused_then_rebuilt_on_request(tmp_path, setting):
    job = _dense_job(tmp_path)
    ops.index(job, tmp_path / "idx", encoder_factory=lambda spec: CountingEncoder())

    changed = CountingEncoder(**setting)
    with pytest.raises(ops.OpError) as info:
        ops.index(job, tmp_path / "idx", encoder_factory=lambda spec: changed)
    assert info.value.result.exit_code == 3
    key = f"encoder.{next(iter(setting))}"
    assert key in info.value.result.data["differences"]["dense"]
    assert changed.encoded == 0  # a refusal touches nothing

    rebuilt = ops.index(job, tmp_path / "idx", rebuild=True, encoder_factory=lambda s: changed)
    assert rebuilt.data["retrievers"][0]["action"] == "rebuilt"


def test_changed_targets_make_a_bm25_index_incompatible(tmp_path):
    job = _job_file(tmp_path)
    ops.index(job, tmp_path / "idx")
    targets = tmp_path / "targets_tiny.csv"
    targets.write_text(
        targets.read_text(encoding="utf-8") + "CHEBI:1,maltose,,A sugar.\n", encoding="utf-8"
    )
    with pytest.raises(ops.OpError) as info:
        ops.index(job, tmp_path / "idx")
    differences = info.value.result.data["differences"]["bm25"]
    assert {"records", "record_count"} <= set(differences)


def test_an_index_built_before_component_metadata_is_refused(tmp_path):
    job = _job_file(tmp_path)
    ops.index(job, tmp_path / "idx")
    meta_path = tmp_path / "idx" / "bm25" / "xwalk_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["components"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ops.OpError) as info:
        ops.index(job, tmp_path / "idx")
    assert info.value.code == "index_mismatch"


# --- run ----------------------------------------------------------------------------


def test_a_repeated_match_opens_the_index_instead_of_rebuilding_it(tmp_path, monkeypatch):
    job = _job_file(tmp_path)
    out = tmp_path / "run"
    assert ops.run(job, out, llm=_llm()).data["indexes"] == {"bm25": "built"}

    from xwalk.retrieval.bm25 import BM25Retriever

    def no_build(*args, **kwargs):
        raise AssertionError("a compatible index must not be rebuilt")

    monkeypatch.setattr(BM25Retriever, "build", no_build)
    assert ops.run(job, out, llm=_llm()).data["indexes"] == {"bm25": "opened"}


def test_match_reads_the_targets_once(tmp_path, monkeypatch):
    reads = []
    original = JobSpec.build_target_records

    def counting(self):
        reads.append(1)
        return original(self)

    monkeypatch.setattr(JobSpec, "build_target_records", counting)
    ops.run(_job_file(tmp_path), tmp_path / "run", llm=_llm())
    assert len(reads) == 1


def test_an_unchanged_complete_run_makes_no_calls_and_identical_exports(tmp_path):
    job = _job_file(tmp_path)
    out = tmp_path / "run"
    first = ops.run(job, out, llm=_llm())
    mapping = (out / "mapping.csv").read_bytes()
    second_llm = _llm()
    second = ops.run(job, out, llm=second_llm)
    assert second_llm.requests == []
    assert second.usage is not None and second.usage["calls"] == 0
    assert (out / "mapping.csv").read_bytes() == mapping
    assert second.run == first.run


def test_a_different_run_is_refused_and_nothing_is_overwritten(tmp_path):
    out = tmp_path / "run"
    ops.run(_job_file(tmp_path), out, llm=_llm())
    before = (out / "mapping.csv").read_bytes()

    changed = _job_file(
        tmp_path, lambda d: d["policy"].__setitem__("accept_at", 0.7), name="changed.yaml"
    )
    llm = _llm()
    with pytest.raises(ops.OpError) as info:
        ops.run(changed, out, llm=llm)
    result = info.value.result
    assert result.exit_code == 3
    assert info.value.code == "run_fingerprint_mismatch"
    assert "policy.accept_at" in result.data["differences"]
    assert "policy.accept_at" in result.errors[0].message
    assert llm.requests == []
    assert (out / "mapping.csv").read_bytes() == before


def test_a_non_empty_foreign_directory_is_refused(tmp_path):
    out = tmp_path / "mine"
    out.mkdir()
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ops.OpError) as info:
        ops.run(_job_file(tmp_path), out, llm=_llm())
    assert info.value.code == "out_not_a_run"
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_the_call_limit_aborts_and_a_resume_finishes(tmp_path):
    job = _job_file(tmp_path)
    out = tmp_path / "run"
    limited = _llm()
    result = ops.run(job, out, llm=limited, max_calls=3)
    assert len(limited.requests) <= 3
    assert result.exit_code == 3
    assert result.run is not None and result.run["run_state"] == "aborted"
    assert result.errors[0].code == "call_limit_reached"
    assert result.usage is not None and result.usage["calls"] <= 3

    resumed = ops.run(job, out, llm=_llm())
    assert resumed.run is not None and resumed.run["run_state"] == "complete"
    assert resumed.counts["total"] == 4


@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_the_call_limit_is_reported_once_however_many_records_hit_it(tmp_path, cap):
    """Several in-flight records refused by the same spent budget are one run-level
    stop, not one error per record (review finding, 0.2.0rc1)."""
    result = ops.run(_job_file(tmp_path), tmp_path / "run", llm=_llm(), max_calls=cap)
    limits = [e for e in result.errors if e.code == "call_limit_reached"]
    assert len(limits) == 1
    assert limits[0].source_id is None
    assert limits[0].message.startswith(f"call limit of {cap} reached")


def test_exit_codes_follow_the_run(tmp_path):
    job = _job_file(tmp_path)
    assert ops.run(job, tmp_path / "a", llm=_llm()).exit_code == 0
    assert ops.run(job, tmp_path / "b", llm=_llm(score=0.5)).exit_code == 1
    partial = ops.run(job, tmp_path / "c", llm=_llm(), limit=2)
    assert partial.exit_code == 1 and partial.counts["pending"] == 2


# --- inspect / explain / export ------------------------------------------------------


@pytest.fixture
def finished_run(tmp_path):
    out = tmp_path / "run"
    ops.run(_job_file(tmp_path), out, llm=_llm(score=0.5))  # every match needs review
    return out


def test_inspect_names_the_run_version_state_and_counts(finished_run):
    result = ops.inspect(finished_run)
    manifest = json.loads((finished_run / "manifest.json").read_text(encoding="utf-8"))
    assert result.run == {
        "dir": str(finished_run),
        "run_fingerprint": manifest["run_fingerprint"],
        "run_state": "complete",
    }
    assert result.counts["needs_review"] == 3 and result.counts["total"] == 4
    assert result.data["job"] == "tiny"
    assert result.data["library_version"] == manifest["library_version"]
    assert result.data["fingerprint_components"]["policy"]["accept_at"] == 0.6
    assert result.exit_code == 1  # the run needs review


def test_explain_identifies_source_decision_and_attempts(finished_run):
    result = ops.explain(finished_run, "s2")
    data = result.data
    assert data["source_id"] == "s2"
    assert data["decision"]["matched_id"] == "CHEBI:17234"
    assert data["decision"]["status"] == "needs_review"
    assert data["attempts"] and data["attempts"][0]["query"] == "dextrose"
    assert data["attempts"][0]["top_candidates"][0] == "CHEBI:17234"
    assert result.run is not None and result.run["run_fingerprint"]
    assert data["library_version"]


def test_explain_an_unknown_source_is_a_usage_error(finished_run):
    with pytest.raises(ops.OpError) as info:
        ops.explain(finished_run, "nope")
    assert info.value.result.exit_code == 2


def _review(run_dir: Path, tmp_path: Path, decision: str, corrected: str = "") -> None:
    worksheet = tmp_path / "review.csv"
    ops.review_export(run_dir, worksheet)
    rows = list(csv.DictReader(worksheet.open(encoding="utf-8")))
    row = next(r for r in rows if r["source_id"] == "s2")
    row.update(decision=decision, corrected_target_id=corrected, reviewer="jan")
    with worksheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerow(row)
    assert ops.review_apply(run_dir, worksheet, _job_file(tmp_path)).counts["applied"] == 1


def test_the_reviewed_export_applies_the_overlay_and_keeps_the_model_decision(
    finished_run, tmp_path
):
    _review(finished_run, tmp_path, "replace", "CHEBI:15903")
    result = ops.export(finished_run, "reviewed", tmp_path / "reviewed.csv")
    assert result.counts == {"rows": 4, "reviewed": 1}
    rows = {r["source_id"]: r for r in csv.DictReader((tmp_path / "reviewed.csv").open())}
    assert rows["s2"]["final_matched_id"] == "CHEBI:15903"
    assert rows["s2"]["final_status"] == "matched"
    assert rows["s2"]["review_decision"] == "replace"
    assert rows["s2"]["model_matched_id"] == "CHEBI:17234"
    assert rows["s2"]["model_status"] == "needs_review"
    assert rows["s1"]["review_decision"] == ""

    raw = ops.export(finished_run, "raw", tmp_path / "raw.csv")
    raw_rows = {r["source_id"]: r for r in csv.DictReader((tmp_path / "raw.csv").open())}
    assert raw.counts == {"rows": 4}
    assert raw_rows["s2"]["matched_id"] == "CHEBI:17234"  # the ledger was not rewritten
    assert ops.explain(finished_run, "s2").data["review"]["final_matched_id"] == "CHEBI:15903"


def test_the_history_export_includes_superseded_results_and_reviews(finished_run, tmp_path):
    _review(finished_run, tmp_path, "accept")
    ops.run(_job_file(tmp_path), finished_run, llm=_llm(score=0.5), resume=False)
    result = ops.export(finished_run, "history", tmp_path / "history.jsonl")
    lines = [json.loads(x) for x in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert result.counts["rows"] == 8 == len(lines)
    assert sum(1 for line in lines if line["current"]) == 4
    reviews = (tmp_path / "history.reviews.jsonl").read_text().splitlines()
    assert [json.loads(r)["state"] for r in reviews] == ["stale"]


def test_an_unknown_export_view_is_a_usage_error(finished_run, tmp_path):
    with pytest.raises(ops.OpError) as info:
        ops.export(finished_run, "cooked", tmp_path / "x.csv")
    assert info.value.result.exit_code == 2


# --- init and the high-level route --------------------------------------------------


def test_init_copies_a_valid_bundled_job_and_refuses_to_overwrite(tmp_path):
    result = ops.init(tmp_path / "qs")
    assert set(result.data["files"]) >= {"job.yaml", "slots.yaml", "targets.csv", "sources.csv"}
    assert ops.validate(tmp_path / "qs" / "job.yaml", check_credentials=False).exit_code == 0
    with pytest.raises(ops.OpError) as info:
        ops.init(tmp_path / "qs")
    assert info.value.result.exit_code == 3


def test_the_bundled_job_is_the_one_the_examples_document():
    job = load_job(ops.bundled_example() / "job.yaml")
    assert job.name == "quickstart"


def test_the_python_quickstart_runs_offline(tmp_path, capsys):
    from examples.quickstart import main

    assert main(str(tmp_path / "run")) == 0
    rows = list(csv.DictReader((tmp_path / "run" / "reviewed.csv").open(encoding="utf-8")))
    assert [r["source_id"] for r in rows] == ["s1", "s2", "s3", "s4"]
    assert "CHEBI:17234" in capsys.readouterr().out


def test_the_python_quickstart_is_short():
    """The acceptance bar: a common job in roughly 10-15 readable lines."""
    import inspect

    from examples import quickstart

    body = inspect.getsource(quickstart.main).splitlines()
    assert len(body) <= 15

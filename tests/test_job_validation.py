"""Strict job validation (CONTRACTS.md section 8).

Before task 03 every spec accepted unknown keys, so `accept_att: 0.9` silently ran with
the default `accept_at` of 0.6. Each case below names the bad key or value.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from tests.conftest import FIXTURES
from xwalk.config import JobValidationError, load_job, parse_job

BASE = yaml.safe_load((FIXTURES / "job_tiny.yaml").read_text(encoding="utf-8"))


def _job(edit):
    data = copy.deepcopy(BASE)
    edit(data)
    return data


def _issues(data):
    with pytest.raises(JobValidationError) as info:
        parse_job(data)
    return [(i.code, i.loc, i.message) for i in info.value.issues]


def test_the_fixture_job_is_valid():
    assert parse_job(copy.deepcopy(BASE)).name == "tiny"


def test_a_misspelled_policy_key_fails_and_suggests_the_real_one():
    issues = _issues(_job(lambda d: d["policy"].__setitem__("accept_att", 0.9)))
    assert issues == [
        ("unknown_field", "policy.accept_att", "unknown field; did you mean 'accept_at'?")
    ]


def test_a_misspelled_top_level_key_fails():
    issues = _issues(_job(lambda d: d.__setitem__("retreivers", [])))
    assert issues[0][:2] == ("unknown_field", "retreivers")
    assert "retrievers" in issues[0][2]


def test_an_unknown_key_inside_a_list_item_is_located():
    issues = _issues(_job(lambda d: d["retrievers"][0].__setitem__("limt", 5)))
    assert issues[0][:2] == ("unknown_field", "retrievers.0.limt")
    assert "'limit'" in issues[0][2]


def test_an_invalid_selector_value_is_rejected():
    issues = _issues(_job(lambda d: d["selector"].__setitem__("max_candidates", 0)))
    assert issues[0][:2] == ("invalid_value", "selector.max_candidates")


def test_a_missing_required_field_is_named():
    issues = _issues(_job(lambda d: d.pop("llm")))
    # Since the decider path, the message also names the alternative block.
    assert issues == [
        (
            "missing_field",
            "llm",
            "required field is missing; a job needs exactly one of `llm:` or `decider:`",
        )
    ]


@pytest.mark.parametrize(
    ("edit", "loc"),
    [
        (lambda d: d["policy"].__setitem__("accept_at", 1.5), "policy.accept_at"),
        (lambda d: d["policy"].__setitem__("review_floor", 0.9), "policy"),
        (lambda d: d["policy"].__setitem__("verify_band", [0.9, 0.2]), "policy"),
        (lambda d: d["policy"].__setitem__("concurrency", 0), "policy.concurrency"),
        (lambda d: d["retrievers"][0].__setitem__("limit", 0), "retrievers.0.limit"),
        (lambda d: d["llm"].__setitem__("max_tokens", 0), "llm.max_tokens"),
    ],
)
def test_out_of_range_values_are_rejected(edit, loc):
    issues = _issues(_job(edit))
    assert issues[0][0] == "invalid_value"
    assert issues[0][1] == loc


def test_duplicate_retriever_names_are_rejected():
    def edit(d):
        d["retrievers"].append({"kind": "bm25"})  # unnamed bm25 -> index dir "bm25" again

    issues = _issues(_job(edit))
    assert "share the name 'bm25'" in issues[0][2]


@pytest.mark.parametrize("name", ["../x", "a/b", "a\\b", "..", ".", "/abs", " pad", "x y"])
def test_a_retriever_name_must_be_a_safe_index_directory_name(name):
    """The name is the index subdirectory: it may not leave the index directory."""

    def edit(d):
        d["retrievers"][0]["name"] = name

    issues = _issues(_job(edit))
    assert issues[0][1].startswith("retrievers.0")
    assert "letters, digits" in issues[0][2]


@pytest.mark.parametrize("name", ["bm25", "dense-mini", "labels_v2", "bm25.v2", "B2"])
def test_ordinary_retriever_names_are_accepted(name):
    parse_job(_job(lambda d: d["retrievers"][0].__setitem__("name", name)))


def test_a_field_that_does_not_apply_to_the_kind_is_rejected():
    issues = _issues(_job(lambda d: d["retrievers"][0].__setitem__("query_prefix", "q: ")))
    assert "query_prefix does not apply to kind 'bm25'" in issues[0][2]

    issues = _issues(_job(lambda d: d["source"].__setitem__("url", "sqlite://")))
    assert "url does not apply to kind 'csv'" in issues[0][2]


def test_a_file_collection_without_a_path_is_rejected():
    issues = _issues(_job(lambda d: d["target"].pop("path")))
    assert "needs a path" in issues[0][2]


def test_base_dir_cannot_be_set_from_the_file():
    issues = _issues(_job(lambda d: d.__setitem__("base_dir", "/etc")))
    assert issues[0][:2] == ("unknown_field", "base_dir")


def test_load_job_names_a_missing_file(tmp_path):
    with pytest.raises(JobValidationError) as info:
        load_job(tmp_path / "nope.yaml")
    assert info.value.issues[0].code == "job_not_found"
    assert "nope.yaml" in str(info.value)


def test_load_job_reports_malformed_yaml(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(JobValidationError) as info:
        load_job(path)
    assert info.value.issues[0].code == "job_yaml_invalid"


def test_a_key_given_twice_is_rejected_not_silently_overridden(tmp_path):
    """PyYAML keeps the last of two equal keys; a strict job must refuse the file."""
    text = (FIXTURES / "job_tiny.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert "policy" in data
    path = tmp_path / "job.yaml"
    path.write_text(text + "\npolicy:\n  accept_at: 0.1\n", encoding="utf-8")
    with pytest.raises(JobValidationError) as info:
        load_job(path)
    issue = info.value.issues[0]
    assert issue.code == "job_yaml_invalid"
    assert "duplicate key 'policy'" in issue.message


def test_a_nested_key_given_twice_is_rejected(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text("name: x\npolicy:\n  accept_at: 0.9\n  accept_at: 0.1\n", encoding="utf-8")
    with pytest.raises(JobValidationError) as info:
        load_job(path)
    assert "duplicate key 'accept_at' (first given on line 3)" in info.value.issues[0].message


def test_a_merge_key_may_still_be_overridden(tmp_path):
    from xwalk.config import load_job_yaml

    text = "base: &b {accept_at: 0.9, review_floor: 0.4}\npolicy:\n  <<: *b\n  accept_at: 0.7\n"
    data = load_job_yaml(tmp_path / "job.yaml", text)
    assert data["policy"] == {"accept_at": 0.7, "review_floor": 0.4}


def test_a_yaml_error_does_not_echo_the_file_content(tmp_path):
    """Any path can be named (an MCP client can pass one); the error gives a position,
    not a quotation of the file."""
    path = tmp_path / "creds.ini"
    path.write_text("[default]\nsecret_key = AKIASECRETVALUE\n", encoding="utf-8")
    with pytest.raises(JobValidationError) as info:
        load_job(path)
    message = info.value.issues[0].message
    assert "AKIASECRETVALUE" not in str(info.value)
    assert "line 2" in message


def test_every_issue_is_reported_not_only_the_first():
    def edit(d):
        d["policy"]["accept_att"] = 0.9
        d["selector"]["max_candidatez"] = 3

    locs = {loc for _, loc, _ in _issues(_job(edit))}
    assert locs == {"policy.accept_att", "selector.max_candidatez"}

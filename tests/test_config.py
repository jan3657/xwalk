import pytest
import yaml

from tests.conftest import FIXTURES
from xwalk.config import JobSpec, load_job

JOB = FIXTURES / "job_tiny.yaml"


def _job_copy(tmp_path, mutate):
    data = yaml.safe_load(JOB.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_a_job_file():
    assert load_job(JOB).name == "tiny"


def test_builds_the_template_set():
    from xwalk.records import Record

    templates = load_job(JOB).build_templates()
    assert templates.render_query(Record(id="s1", fields={"mention": "glucose"})) == "glucose"


def test_builds_target_records_with_multivalue_columns():
    records = list(load_job(JOB).build_target_records())
    assert records[0].fields["synonyms"] == ["dextrose", "grape sugar"]


def test_builds_source_records():
    assert len(list(load_job(JOB).build_source_records())) == 4


def test_builds_an_in_memory_store():
    assert len(load_job(JOB).build_store()) == 5


def test_builds_a_bm25_retriever(tmp_path):
    job = load_job(JOB)
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path
    )
    assert [r.name for r in retrievers] == ["bm25"]


def test_the_retriever_limit_reaches_the_retriever(tmp_path):
    """`limit` in the job file is retrieval depth, and the matcher reads it off the
    retriever. A spec field that never reaches an object is a lie in the config."""
    job = load_job(JOB)
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path
    )
    assert retrievers[0].default_limit == 20


def test_builds_prompts_from_the_slots_file():
    assert load_job(JOB).build_prompts().slots.entity_noun == "chemical entity mention"


def test_builds_the_policy_with_the_declared_values():
    policy = load_job(JOB).build_policy()
    assert policy.accept_at == 0.6 and policy.verify_band == (0.6, 0.8)


def test_the_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert load_job(JOB).build_llm().model == "gpt-4o-mini"


def test_a_missing_api_key_env_var_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_job(JOB).build_llm()


def test_an_api_key_written_inline_is_rejected(tmp_path):
    path = _job_copy(tmp_path, lambda d: d["llm"].__setitem__("api_key", "sk-oops"))
    with pytest.raises(ValueError, match="api_key"):
        load_job(path)


def test_an_unknown_source_kind_is_rejected(tmp_path):
    path = _job_copy(tmp_path, lambda d: d["source"].__setitem__("kind", "carrier_pigeon"))
    with pytest.raises(ValueError, match="carrier_pigeon"):
        load_job(path)


def test_an_unknown_retriever_kind_is_rejected(tmp_path):
    path = _job_copy(tmp_path, lambda d: d.__setitem__("retrievers", [{"kind": "telepathy"}]))
    with pytest.raises(ValueError, match="telepathy"):
        load_job(path)


def test_an_empty_retriever_list_is_rejected(tmp_path):
    path = _job_copy(tmp_path, lambda d: d.__setitem__("retrievers", []))
    with pytest.raises(ValueError, match="retriever"):
        load_job(path)


def test_a_dense_retriever_without_a_model_is_rejected(tmp_path):
    path = _job_copy(tmp_path, lambda d: d.__setitem__("retrievers", [{"kind": "dense"}]))
    with pytest.raises(ValueError, match="model"):
        load_job(path)


def test_relative_paths_resolve_against_the_job_file(tmp_path):
    (tmp_path / "targets.csv").write_text(
        (FIXTURES / "targets_tiny.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )

    def mutate(data):
        data["target"]["path"] = "targets.csv"
        data["source"]["path"] = str((FIXTURES / "sources_tiny.csv").resolve())
        data["prompts"]["slots"] = str(
            (FIXTURES / ".." / "..").resolve() / "examples/chemistry/slots.yaml"
        )

    path = _job_copy(tmp_path, mutate)
    assert len(list(load_job(path).build_target_records())) == 5


def test_the_run_fingerprint_is_reproducible(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    job = load_job(JOB)
    targets = list(job.build_target_records())
    templates = job.build_templates()
    a = job.run_fingerprint(
        store=job.build_store(),
        retrievers=job.build_retrievers(targets, templates, tmp_path / "a"),
        llm=job.build_llm(),
    )
    b = job.run_fingerprint(
        store=job.build_store(),
        retrievers=job.build_retrievers(targets, templates, tmp_path / "b"),
        llm=job.build_llm(),
    )
    assert a == b


def test_changing_the_job_changes_the_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    job = load_job(JOB)
    other = job.model_copy(update={"policy": job.policy.model_copy(update={"accept_at": 0.9})})
    targets = list(job.build_target_records())
    templates = job.build_templates()
    args = dict(
        store=job.build_store(),
        retrievers=job.build_retrievers(targets, templates, tmp_path / "a"),
        llm=job.build_llm(),
    )
    assert job.run_fingerprint(**args) != other.run_fingerprint(**args)


def test_a_job_spec_can_be_constructed_in_python_without_a_file():
    """The SDK stays primary; the file is serialized constructor arguments."""
    spec = JobSpec.model_validate(yaml.safe_load(JOB.read_text(encoding="utf-8")))
    assert spec.name == "tiny"


def test_build_matcher_accepts_overridden_prompts(monkeypatch, tmp_path):
    """Without this the prompt optimiser silently re-runs the original slots every
    round and measures nothing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from xwalk.prompts.contract import PromptSet, PromptSlots

    job = load_job(JOB)
    targets = list(job.build_target_records())
    store = job.build_store()
    retrievers = job.build_retrievers(targets, job.build_templates(), tmp_path)
    other = PromptSet.from_slots(
        PromptSlots(
            entity_noun="x",
            target_noun="y",
            domain_brief="z",
            rubric=[
                {"score": 1.0, "name": "a", "when": "b"},
                {"score": 0.4, "name": "c", "when": "d"},
            ],
        )
    )
    matcher = job.build_matcher(
        store=store, retrievers=retrievers, llm=job.build_llm(), prompts=other
    )
    assert matcher.run_fingerprint != job.run_fingerprint(
        store=store, retrievers=retrievers, llm=job.build_llm()
    )


def test_templates_accept_a_queries_list(tmp_path):
    from xwalk.records import Record

    def mutate(data):
        del data["templates"]["query"]
        data["templates"]["queries"] = ["{{ mention }}", "{{ context_left }}"]

    templates = load_job(_job_copy(tmp_path, mutate)).build_templates()
    record = Record(id="s", fields={"mention": "glucose", "context_left": "blood"})
    assert templates.render_query(record) == "glucose"
    assert templates.render_queries(record) == ["glucose", "blood"]


def test_templates_need_a_query_or_queries(tmp_path):
    def mutate(data):
        del data["templates"]["query"]

    with pytest.raises(ValueError, match="query"):
        load_job(_job_copy(tmp_path, mutate))

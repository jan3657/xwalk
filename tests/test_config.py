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


@pytest.mark.parametrize("distance", [-1, 3])
def test_a_fuzzy_distance_outside_0_to_2_is_rejected(tmp_path, distance):
    """Tantivy raises on every fuzzy search above distance 2; refuse it at load time."""
    retriever = {"kind": "bm25", "fuzzy_distance": distance}
    path = _job_copy(tmp_path, lambda d: d.__setitem__("retrievers", [retriever]))
    with pytest.raises(ValueError, match="fuzzy_distance"):
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

    templates = load_job(_jev_job(tmp_path, mutate)).build_templates()
    record = Record(id="s", fields={"mention": "glucose", "context_left": "blood"})
    assert templates.render_query(record) == "glucose"
    assert templates.render_queries(record) == ["glucose", "blood"]


def test_several_queries_are_refused_on_an_llm_job(tmp_path):
    """0.2 strict validation: the LLM matcher would never search the extra queries."""

    def mutate(data):
        del data["templates"]["query"]
        data["templates"]["queries"] = ["{{ mention }}", "{{ context_left }}"]

    with pytest.raises(ValueError, match="decider jobs"):
        load_job(_job_copy(tmp_path, mutate))


def test_templates_need_a_query_or_queries(tmp_path):
    def mutate(data):
        del data["templates"]["query"]

    with pytest.raises(ValueError, match="query"):
        load_job(_job_copy(tmp_path, mutate))


def _jev_job(tmp_path, extra_mutate=None):
    def mutate(data):
        del data["llm"]
        del data["selector"]
        data["decider"] = {
            "kind": "jev",
            "model": "~typesafe/jev-latest",
            "base_url": "https://openrouter.ai/api/alpha/decisions",
            "api_key_env": "XWALK_TEST_API_KEY",
        }
        data["policy"] = {"accept_at": 0.9, "chunk_size": 25, "concurrency": 4}
        # The copy lives in tmp_path, so the job's relative paths have to be re-anchored
        # on the fixture directory or the builders below have nothing to read.
        data["target"]["path"] = str(FIXTURES / "targets_tiny.csv")
        data["source"]["path"] = str(FIXTURES / "sources_tiny.csv")
        data["prompts"]["slots"] = str((FIXTURES / data["prompts"]["slots"]).resolve())
        if extra_mutate:
            extra_mutate(data)

    return _job_copy(tmp_path, mutate)


def test_a_decider_job_loads_and_routes_policy(tmp_path):
    job = load_job(_jev_job(tmp_path))
    assert job.llm is None and job.decider is not None
    assert job.decider.model == "~typesafe/jev-latest"
    policy = job.build_decision_policy()
    assert (policy.accept_at, policy.chunk_size, policy.concurrency) == (0.9, 25, 4)
    assert policy.screen_floor == 0.30  # untouched default


def test_llm_and_decider_are_mutually_exclusive(tmp_path):
    def keep_llm(data):
        data["llm"] = {
            "kind": "openai_compat",
            "model": "m",
            "base_url": "https://x",
            "api_key_env": "K",
        }

    with pytest.raises(ValueError, match="exactly one"):
        load_job(_jev_job(tmp_path, keep_llm))


def test_neither_llm_nor_decider_is_an_error(tmp_path):
    def drop(data):
        del data["decider"]

    with pytest.raises(ValueError, match="exactly one"):
        load_job(_jev_job(tmp_path, drop))


def test_decider_rejects_inline_keys(tmp_path):
    def inline(data):
        data["decider"]["api_key"] = "sk-live"

    with pytest.raises(ValueError, match="api_key_env"):
        load_job(_jev_job(tmp_path, inline))


def test_build_decider_needs_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("XWALK_TEST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XWALK_TEST_API_KEY"):
        load_job(_jev_job(tmp_path)).build_decider()


def test_build_decision_matcher(tmp_path, monkeypatch):
    from xwalk.decide.fake import FakeDecider
    from xwalk.decide.matcher import DecisionMatcher

    job = load_job(_jev_job(tmp_path))
    store = job.build_store()
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path / "idx"
    )
    matcher = job.build_decision_matcher(store=store, retrievers=retrievers, decider=FakeDecider())
    assert isinstance(matcher, DecisionMatcher)
    assert matcher.policy.concurrency == 4
    assert len(matcher.run_fingerprint) == 16


def test_decision_fingerprint_moves_with_policy_and_model(tmp_path):
    from xwalk.decide.fake import FakeDecider

    job = load_job(_jev_job(tmp_path))
    store = job.build_store()
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path / "idx"
    )
    a = job.decision_run_fingerprint(
        store=store, retrievers=retrievers, decider=FakeDecider(model="a")
    )
    b = job.decision_run_fingerprint(
        store=store, retrievers=retrievers, decider=FakeDecider(model="b")
    )
    c = load_job(
        _jev_job(tmp_path, lambda d: d["policy"].update({"accept_at": 0.7}))
    ).decision_run_fingerprint(store=store, retrievers=retrievers, decider=FakeDecider(model="a"))
    assert a != b and a != c


def test_decision_policy_spec_defaults_match_the_policy():
    """Two declarations of the same defaults; a silent drift would ship two policies."""
    from dataclasses import asdict

    from xwalk.config import DecisionPolicySpec
    from xwalk.decide.policy import DecisionPolicy

    assert DecisionPolicySpec().model_dump() == asdict(DecisionPolicy())


_REWRITE = {
    "llm": {
        "kind": "openai_compat",
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "XWALK_TEST_API_KEY",
    }
}


def _with_rewrite(data, **extra):
    data["decider"]["rewrite"] = {**_REWRITE, **extra}


def test_decider_rewrite_parses(tmp_path):
    job = load_job(_jev_job(tmp_path, lambda d: _with_rewrite(d, max_queries=2)))
    assert job.decider is not None and job.decider.rewrite is not None
    assert job.decider.rewrite.llm.model == "qwen/qwen3-next-80b-a3b-instruct"
    assert job.decider.rewrite.max_queries == 2


def test_decider_rewrite_defaults(tmp_path):
    assert load_job(_jev_job(tmp_path)).decider.rewrite is None
    job = load_job(_jev_job(tmp_path, _with_rewrite))
    assert job.decider.rewrite.max_queries == 3


@pytest.mark.parametrize("bad", [0, 6])
def test_decider_rewrite_bounds_max_queries(tmp_path, bad):
    with pytest.raises(ValueError, match="max_queries"):
        load_job(_jev_job(tmp_path, lambda d: _with_rewrite(d, max_queries=bad)))


def test_decider_rewrite_rejects_inline_keys(tmp_path):
    def inline(data):
        _with_rewrite(data)
        data["decider"]["rewrite"]["llm"] = {**_REWRITE["llm"], "api_key": "sk-live"}

    with pytest.raises(ValueError, match="api_key_env"):
        load_job(_jev_job(tmp_path, inline))


def test_build_rewrite_llm_needs_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("XWALK_TEST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="decider.rewrite.llm.api_key_env"):
        load_job(_jev_job(tmp_path, _with_rewrite)).build_rewrite_llm()


def test_build_llm_keeps_its_error_wording(monkeypatch):
    monkeypatch.delenv("XWALK_TEST_API_KEY", raising=False)
    job = load_job(JOB)
    job = job.model_copy(
        update={"llm": job.llm.model_copy(update={"api_key_env": "XWALK_TEST_API_KEY"})}
    )
    with pytest.raises(ValueError, match=r"change llm\.api_key_env in the job file"):
        job.build_llm()


def test_build_decision_matcher_wires_the_rewriter(tmp_path, monkeypatch):
    from xwalk.decide.fake import FakeDecider
    from xwalk.llm.fake import FakeLLM

    job = load_job(_jev_job(tmp_path))
    store = job.build_store()
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path / "idx"
    )
    plain = job.build_decision_matcher(store=store, retrievers=retrievers, decider=FakeDecider())
    assert plain.rewriter is None
    with pytest.raises(ValueError, match="rewrite"):
        job.build_decision_matcher(
            store=store, retrievers=retrievers, decider=FakeDecider(), rewrite_llm=FakeLLM([])
        )

    rewriting = load_job(_jev_job(tmp_path, _with_rewrite)).build_decision_matcher(
        store=store, retrievers=retrievers, decider=FakeDecider(), rewrite_llm=FakeLLM([])
    )
    assert rewriting.rewriter is not None
    assert rewriting.run_fingerprint != plain.run_fingerprint


def test_decision_fingerprint_moves_with_the_rewrite_block(tmp_path):
    from xwalk.decide.fake import FakeDecider

    store_job = load_job(_jev_job(tmp_path))
    store = store_job.build_store()
    retrievers = store_job.build_retrievers(
        list(store_job.build_target_records()), store_job.build_templates(), tmp_path / "idx"
    )

    def fp(mutate):
        return load_job(_jev_job(tmp_path, mutate)).decision_run_fingerprint(
            store=store, retrievers=retrievers, decider=FakeDecider(model="a")
        )

    none = fp(lambda d: None)
    three = fp(_with_rewrite)
    two = fp(lambda d: _with_rewrite(d, max_queries=2))
    assert len({none, three, two}) == 3

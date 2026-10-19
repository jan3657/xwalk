"""Run identity follows the requests a run actually sends (CONTRACTS.md sections 6 and 9).

Every case compares captured outgoing request bodies, not hashes alone: two
configurations whose bodies differ must have different run fingerprints, and two whose
bodies are identical (a rotated key, a different concurrency, a budget or cache
wrapper) must share one. Offline throughout: httpx MockTransport and FakeDecider.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from tests.test_decide_matcher import QUESTIONS, SLOTS, Scripted
from tests.test_decide_matcher import STORE as DECIDE_STORE
from tests.test_decide_matcher import TEMPLATES as DECIDE_TEMPLATES
from tests.test_matcher import PROMPTS, SOURCE, STORE, TEMPLATES, ScriptedRetriever
from xwalk import ops
from xwalk.batch import (
    build_decision_run_fingerprint,
    build_run_fingerprint,
    run_batch,
    run_fingerprint_components,
)
from xwalk.decide.budget import BudgetedDecider
from xwalk.decide.cache import CachingDecider
from xwalk.decide.fake import FakeDecider
from xwalk.decide.jev import JevClient
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.ledger import Ledger
from xwalk.llm.base import LLMCapabilities
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.llm.cache import CachingLLM
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.matcher import Matcher
from xwalk.policy import MatchPolicy
from xwalk.prompts.contract import PromptSet
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.choose import Chooser
from xwalk.stages.gate import Scorer, Verifier
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.rewrite import QueryRewriter
from xwalk.stages.screen import Screener
from xwalk.stages.select import Selector, SelectorPolicy

FIXTURES = Path(__file__).parent / "fixtures"
# One reply every LLM stage can read: a selection, a score, a verdict and queries.
REPLY = json.dumps(
    {
        "chosen_key": "C01",
        "confidence_score": 0.7,
        "explanation": "",
        "decision": "support",
        "queries": ["dextrose"],
    }
)


class Capture:
    """Builds clients whose every outgoing request body lands in `bodies`."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.jev_bodies: list[dict[str, Any]] = []

    def _llm_handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": REPLY}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    def client(self, **kwargs: Any) -> OpenAICompatClient:
        settings: dict[str, Any] = {
            "base_url": "https://llm.test/v1",
            "model": "m",
            "api_key": "sk-one",
            "profile": "unknown",
            "max_retries": 0,
        }
        settings.update(kwargs)
        return OpenAICompatClient(transport=httpx.MockTransport(self._llm_handler), **settings)

    def _jev_handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.jev_bodies.append(body)
        answers: dict[str, Any] = {}
        # Low everywhere: every screen misses, so a rewriter is always consulted.
        for name, question in body["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.05}
            elif question["type"] == "choice":
                keys = list(question["criteria"])
                answers[name] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {k: 1 / len(keys) for k in keys},
                    "confidence": 0.5,
                }
            else:
                levels = len(question["criteria"])
                answers[name] = {
                    "type": "score",
                    "score": 0,
                    "probabilities": {str(i): 1 / levels for i in range(levels)},
                    "confidence": 0.5,
                }
        return httpx.Response(
            200, json={"model": "jev", "answers": answers, "usage": {"input_tokens": 1}}
        )

    def jev(self, **kwargs: Any) -> JevClient:
        settings: dict[str, Any] = {"api_key": "sk-one", "max_retries": 0}
        settings.update(kwargs)
        return JevClient(
            "https://openrouter.ai/api/alpha/decisions",
            "~typesafe/jev-latest",
            transport=httpx.MockTransport(self._jev_handler),
            **settings,
        )


# --- the Matcher path ---------------------------------------------------------------


def default_stages(
    llm: Any, *, prompts: Any = PROMPTS, templates: Any = TEMPLATES
) -> dict[str, Any]:
    """The four LLM stages as a job file builds them, for the required builder arguments."""
    return {
        "selector": Selector(llm, prompts, templates),
        "scorer": Scorer(llm, prompts, templates),
        "verifier": Verifier(llm, prompts, templates),
        "rewriter": QueryRewriter(llm, prompts, templates),
    }


def default_decision_stages(
    decider: Any, *, questions: Any = QUESTIONS, templates: Any = DECIDE_TEMPLATES
) -> dict[str, Any]:
    """The decider stages for a default `DecisionPolicy`, and no rewriter."""
    return {
        "screener": Screener(decider, questions, templates),
        "chooser": Chooser(decider, questions, templates),
        "gate": PropertyGate(decider, questions, templates),
        "rewriter": None,
    }


def _config(
    llm: Any,
    *,
    selector: dict[str, Any] | None = None,
    scorer: dict[str, Any] | None = None,
    verifier: dict[str, Any] | None = None,
    rewriter: dict[str, Any] | None = None,
    verifier_llm: Any = None,
    policy: MatchPolicy | None = None,
) -> dict[str, Any]:
    """Stages built the way a Python-API user builds them."""
    policy = policy or MatchPolicy()
    return {
        "llm": llm,
        "policy": policy,
        "selector": Selector(llm, PROMPTS, TEMPLATES, **(selector or {})),
        "scorer": Scorer(
            llm, PROMPTS, TEMPLATES, **{"review_floor": policy.review_floor, **(scorer or {})}
        ),
        "verifier": Verifier(verifier_llm or llm, PROMPTS, TEMPLATES, **(verifier or {})),
        "rewriter": QueryRewriter(llm, PROMPTS, TEMPLATES, **(rewriter or {})),
    }


def _fingerprint(config: dict[str, Any]) -> str:
    return build_run_fingerprint(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever({})],
        llm=config["llm"],
        policy=config["policy"],
        selector_policy=SelectorPolicy(),
        selector=config["selector"],
        scorer=config["scorer"],
        verifier=config["verifier"],
        rewriter=config["rewriter"],
    )


CANDIDATES = [
    Candidate(
        record=STORE.get(target_id),
        fused_score=1.0 / rank,
        evidence=(RetrievalHit(record_id=target_id, retriever="bm25", raw_score=1.0, rank=rank),),
    )
    for rank, target_id in enumerate(["T1", "T2"], start=1)
]


async def _stage_bodies(capture: Capture, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Call every LLM stage once and return what was sent."""
    start = len(capture.bodies)
    selection = await config["selector"].select(SOURCE, "blood [glucose] levels", CANDIDATES)
    keyed = selection.keyed
    await config["scorer"].score(SOURCE, "ctx", keyed, "C01")
    await config["verifier"].verify(SOURCE, "ctx", keyed, "C01")
    await config["rewriter"].rewrite(SOURCE, "ctx", ["glucose"], keyed)
    return capture.bodies[start:]


Variant = Callable[[Capture, Path], dict[str, Any]]

DIFFERENT_REQUESTS: dict[str, Variant] = {
    "a scorer max_tokens override": lambda c, _: _config(c.client(), scorer={"max_tokens": 256}),
    "a selector temperature override": lambda c, _: _config(
        c.client(), selector={"temperature": 0.7}
    ),
    "a verifier max_tokens override": lambda c, _: _config(c.client(), verifier={"max_tokens": 64}),
    "a rewriter temperature override": lambda c, _: _config(
        c.client(), rewriter={"temperature": 0.9}
    ),
    "a selector system message": lambda c, _: _config(
        c.client(), selector={"system": "Answer with one JSON object."}
    ),
    "a scorer review floor unlike the policy's": lambda c, _: _config(
        c.client(), scorer={"review_floor": 0.2}
    ),
    "a rewriter max_queries": lambda c, _: _config(c.client(), rewriter={"max_queries": 4}),
    "a verifier on another model": lambda c, _: _config(
        c.client(), verifier_llm=c.client(model="other")
    ),
    "a capability override that sends a schema": lambda c, _: _config(
        c.client(capabilities=LLMCapabilities(json_schema=True))
    ),
}

SAME_REQUESTS: dict[str, Variant] = {
    "a rotated API key": lambda c, _: _config(c.client(api_key="sk-two")),
    "explicit overrides equal to the client's values": lambda c, _: _config(
        c.client(),
        scorer={"temperature": 0.0},
        selector={"max_tokens": 1024},
        rewriter={"temperature": 0.0, "max_tokens": 1024},
    ),
    "another concurrency": lambda c, _: _config(c.client(), policy=MatchPolicy(concurrency=1)),
    "budget and cache wrappers": lambda c, tmp: _config(
        CachingLLM(BudgetedLLM(c.client(), CallBudget(100)), Ledger.open(tmp / "cache.sqlite"))
    ),
    "a capability override that changes no request": lambda c, _: _config(
        c.client(capabilities=LLMCapabilities(usage_reporting=True, native_retry_after=True))
    ),
    "a strict flag without a schema capability": lambda c, _: _config(
        c.client(capabilities=LLMCapabilities(strict_schema=True))
    ),
    "a longer timeout and more retries": lambda c, _: _config(c.client(timeout=5.0, max_retries=3)),
}


@pytest.mark.parametrize("name", list(DIFFERENT_REQUESTS))
async def test_different_requests_get_different_run_fingerprints(name, tmp_path):
    capture = Capture()
    base = _config(capture.client())
    variant = DIFFERENT_REQUESTS[name](capture, tmp_path)
    assert await _stage_bodies(capture, base) != await _stage_bodies(capture, variant)
    assert _fingerprint(base) != _fingerprint(variant)


@pytest.mark.parametrize("name", list(SAME_REQUESTS))
async def test_identical_requests_keep_the_run_fingerprint(name, tmp_path):
    capture = Capture()
    base = _config(capture.client())
    variant = SAME_REQUESTS[name](capture, tmp_path)
    assert await _stage_bodies(capture, base) == await _stage_bodies(capture, variant)
    assert _fingerprint(base) == _fingerprint(variant)


def test_default_stages_add_no_identity_component():
    """Stages built with no overrides, as a job file builds them, add nothing: such runs
    keep the fingerprint they had before stages were hashed."""
    capture = Capture()
    config = _config(capture.client())
    common = {
        "templates": TEMPLATES,
        "prompts": PROMPTS,
        "store": STORE,
        "retrievers": [ScriptedRetriever({})],
        "llm": config["llm"],
        "policy": config["policy"],
        "selector_policy": SelectorPolicy(),
    }
    with_stages = run_fingerprint_components(
        **common,
        selector=config["selector"],
        scorer=config["scorer"],
        verifier=config["verifier"],
        rewriter=config["rewriter"],
    )
    assert with_stages == run_fingerprint_components(**common, **default_stages(config["llm"]))
    assert "stages" not in with_stages


def test_a_stage_override_names_the_stage_and_setting_without_secrets():
    capture = Capture()
    config = _config(capture.client(api_key="sk-secret-value"), scorer={"max_tokens": 256})
    components = run_fingerprint_components(
        templates=TEMPLATES,
        prompts=PROMPTS,
        store=STORE,
        retrievers=[ScriptedRetriever({})],
        llm=config["llm"],
        policy=config["policy"],
        selector_policy=SelectorPolicy(),
        selector=config["selector"],
        scorer=config["scorer"],
        verifier=config["verifier"],
        rewriter=config["rewriter"],
    )
    assert components["stages"] == {"scorer": {"max_tokens": 256}}
    assert "sk-secret-value" not in json.dumps(components)


def test_capabilities_enter_the_client_fingerprint_only_when_they_change_requests():
    capture = Capture()
    profile = capture.client(profile="openai")
    assert (
        profile.fingerprint
        == capture.client(
            profile="openai",
            capabilities=LLMCapabilities(
                json_schema=True, strict_schema=True, seed=True, usage_reporting=False
            ),
        ).fingerprint
    )
    assert (
        profile.fingerprint
        != capture.client(
            profile="openai", capabilities=LLMCapabilities(json_schema=True, seed=True)
        ).fingerprint
    )
    google = capture.client(profile="google-compat")
    assert (
        google.fingerprint
        != capture.client(
            profile="google-compat",
            capabilities=LLMCapabilities(json_schema=True, strict_schema=True),
        ).fingerprint
    )
    plain = FakeLLM(["x"])
    assert plain.fingerprint == FakeLLM(["x"], capabilities=LLMCapabilities()).fingerprint
    assert (
        plain.fingerprint
        != FakeLLM(["x"], capabilities=LLMCapabilities(json_schema=True)).fingerprint
    )


async def test_a_python_api_run_never_reuses_results_from_other_requests(tmp_path):
    """`run_batch` lets runs share a ledger; a changed stage must be a new run, not a
    resume, while a rotated key and wrappers resume with no call."""

    def matcher(config: dict[str, Any]) -> Matcher:
        return Matcher(
            templates=TEMPLATES,
            retrievers=[ScriptedRetriever({"glucose": ["T1"]})],
            store=STORE,
            selector=config["selector"],
            scorer=config["scorer"],
            verifier=config["verifier"],
            rewriter=config["rewriter"],
            policy=config["policy"],
            run_fingerprint=_fingerprint(config),
        )

    out = tmp_path / "run"
    capture = Capture()
    first = await run_batch(matcher(_config(capture.client())), [SOURCE], out=out)
    assert capture.bodies

    same = Capture()
    wrapped = BudgetedLLM(same.client(api_key="sk-two"), CallBudget(10))
    resumed = await run_batch(matcher(_config(wrapped)), [SOURCE], out=out)
    assert same.bodies == []
    assert resumed.run_fingerprint == first.run_fingerprint

    changed = Capture()
    other = await run_batch(
        matcher(_config(changed.client(), scorer={"max_tokens": 256})), [SOURCE], out=out
    )
    assert other.run_fingerprint != first.run_fingerprint
    assert any(body["max_tokens"] == 256 for body in changed.bodies)


# --- the job path through ops (refusal before any call) -----------------------------


def _job(tmp_path: Path, name: str, edit: Callable[[dict[str, Any]], None] | None = None) -> Path:
    data = yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))
    for block in ("target", "source"):
        data[block]["path"] = str((FIXTURES / data[block]["path"]).resolve())
    data["prompts"]["slots"] = str((FIXTURES / data["prompts"]["slots"]).resolve())
    if edit is not None:
        edit(data)
    path = tmp_path / f"job-{len(list(tmp_path.glob('job-*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_a_matcher_run_with_other_requests_is_refused_before_any_call(tmp_path):
    job = _job(tmp_path, "job_tiny.yaml")
    out = tmp_path / "run"
    first = Capture()
    ops.run(job, out, llm=first.client())
    assert first.bodies

    rotated = Capture()
    resumed = ops.run(job, out, llm=rotated.client(api_key="sk-rotated"))
    assert resumed.run is not None and resumed.usage is not None
    assert resumed.usage["calls"] == 0 and rotated.bodies == []

    schema = Capture()
    with pytest.raises(ops.OpError) as info:
        ops.run(job, out, llm=schema.client(capabilities=LLMCapabilities(json_schema=True)))
    assert info.value.code == "run_fingerprint_mismatch"
    assert "llm" in info.value.result.data["differences"]
    assert schema.bodies == []


_REWRITE = {
    "llm": {
        "kind": "openai_compat",
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "XWALK_TEST_API_KEY",
    },
    "max_queries": 2,
}


def test_a_decider_run_with_another_rewrite_client_is_refused_before_any_call(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XWALK_TEST_API_KEY", "sk-env")
    job = _job(tmp_path, "job_tiny_jev.yaml", lambda d: d["decider"].update(rewrite=_REWRITE))
    out = tmp_path / "run"

    def run(capture: Capture, decider: FakeDecider, **client: Any) -> ops.OpResult:
        return ops.run(
            job,
            out,
            decider=decider,
            rewrite_llm_factory=lambda spec: capture.client(**client),
        )

    run(Capture(), FakeDecider())

    rotated, decider = Capture(), FakeDecider()
    resumed = run(rotated, decider, api_key="sk-rotated")
    assert resumed.usage is not None and resumed.usage["calls"] == 0
    assert rotated.bodies == [] and decider.calls == []

    schema, decider = Capture(), FakeDecider()
    with pytest.raises(ops.OpError) as info:
        run(schema, decider, capabilities=LLMCapabilities(json_schema=True))
    assert info.value.code == "run_fingerprint_mismatch"
    assert "rewrite" in info.value.result.data["differences"]
    assert schema.bodies == [] and decider.calls == []


def test_job_built_clients_keep_the_job_fingerprints(tmp_path, monkeypatch):
    """A job file cannot express a stage override or a capability override, so a run
    built from one sends what it always sent and keeps its identity."""
    from xwalk.config import load_job

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("XWALK_TEST_API_KEY", "sk-env")
    job = load_job(_job(tmp_path, "job_tiny.yaml"))
    store = job.build_store()
    retrievers = job.build_retrievers(
        list(job.build_target_records()), job.build_templates(), tmp_path / "idx"
    )
    components = job.run_fingerprint_components(
        store=store, retrievers=retrievers, llm=job.build_llm()
    )
    assert "stages" not in components

    jev = load_job(
        _job(tmp_path, "job_tiny_jev.yaml", lambda d: d["decider"].update(rewrite=_REWRITE))
    )
    decider = FakeDecider()
    legacy = jev.decision_run_fingerprint(store=store, retrievers=retrievers, decider=decider)
    built = jev.decision_run_fingerprint(
        store=store, retrievers=retrievers, decider=decider, rewrite_llm=jev.build_rewrite_llm()
    )
    assert built == legacy
    matcher = jev.build_decision_matcher(store=store, retrievers=retrievers, decider=decider)
    assert matcher.run_fingerprint == legacy


# --- the DecisionMatcher path -------------------------------------------------------

DECIDE_POLICY = DecisionPolicy(
    accept_at=0.5,
    screen_floor=0.1,
    shortlist_floor=0.0,
    property_floor=0.0,
    rubric_floor=0.0,
    chunk_size=2,
)
DECIDE_PROMPTS = PromptSet.from_slots(SLOTS)
MISS = Record(id="s2", fields={"mention": "blood sugar", "alias": "bs", "note": ""})


def _decide_config(
    decider: Any,
    rewrite_llm: Any,
    *,
    screener: dict[str, Any] | None = None,
    rewriter: dict[str, Any] | None = None,
    policy: DecisionPolicy = DECIDE_POLICY,
) -> dict[str, Any]:
    return {
        "decider": decider,
        "policy": policy,
        "screener": Screener(
            decider,
            QUESTIONS,
            DECIDE_TEMPLATES,
            **{
                "chunk_size": policy.chunk_size,
                "shortlist_size": policy.shortlist_size,
                "shortlist_floor": policy.shortlist_floor,
                **(screener or {}),
            },
        ),
        "chooser": Chooser(decider, QUESTIONS, DECIDE_TEMPLATES),
        "gate": PropertyGate(decider, QUESTIONS, DECIDE_TEMPLATES),
        "rewriter": QueryRewriter(
            rewrite_llm, DECIDE_PROMPTS, DECIDE_TEMPLATES, **{"max_queries": 3, **(rewriter or {})}
        ),
    }


def _decide_fingerprint(config: dict[str, Any]) -> str:
    return build_decision_run_fingerprint(
        templates=DECIDE_TEMPLATES,
        questions=QUESTIONS,
        store=DECIDE_STORE,
        retrievers=[Scripted({})],
        decider=config["decider"],
        policy=config["policy"],
        screener=config["screener"],
        chooser=config["chooser"],
        gate=config["gate"],
        rewriter=config["rewriter"],
    )


async def _decide_bodies(capture: Capture, config: dict[str, Any]) -> list[Any]:
    """Match one record that misses, so the screen and the rewrite both send requests."""
    llm_start, jev_start = len(capture.bodies), len(capture.jev_bodies)
    matcher = DecisionMatcher(
        templates=DECIDE_TEMPLATES,
        retrievers=[Scripted({"blood sugar": ["T1", "T2", "T3"], "dextrose": ["T3"]})],
        store=DECIDE_STORE,
        screener=config["screener"],
        chooser=config["chooser"],
        gate=config["gate"],
        policy=config["policy"],
        run_fingerprint="fp",
        rewriter=config["rewriter"],
    )
    await matcher.match(MISS)
    return [capture.jev_bodies[jev_start:], capture.bodies[llm_start:]]


DIFFERENT_DECISIONS: dict[str, Variant] = {
    "a screener chunk size unlike the policy's": lambda c, _: _decide_config(
        c.jev(), c.client(), screener={"chunk_size": 1}
    ),
    "a rewriter temperature override": lambda c, _: _decide_config(
        c.jev(), c.client(), rewriter={"temperature": 0.8}
    ),
    "a rewriter max_queries": lambda c, _: _decide_config(
        c.jev(), c.client(), rewriter={"max_queries": 1}
    ),
    "a rewrite client that sends a schema": lambda c, _: _decide_config(
        c.jev(), c.client(capabilities=LLMCapabilities(json_schema=True))
    ),
    "a rewrite client on another model": lambda c, _: _decide_config(
        c.jev(), c.client(model="other")
    ),
}


def _wrapped_decider(capture: Capture, tmp: Path) -> dict[str, Any]:
    budget = CallBudget(1000)
    decider = CachingDecider(
        BudgetedDecider(capture.jev(), budget.share()), Ledger.open(tmp / "decisions.sqlite")
    )
    return _decide_config(decider, BudgetedLLM(capture.client(), budget.share()))


SAME_DECISIONS: dict[str, Variant] = {
    "rotated decider and rewrite keys": lambda c, _: _decide_config(
        c.jev(api_key="sk-two"), c.client(api_key="sk-two")
    ),
    "another concurrency and timeout": lambda c, _: _decide_config(
        c.jev(timeout=5.0),
        c.client(),
        policy=dataclasses.replace(DECIDE_POLICY, concurrency=1),
    ),
    "budget and cache wrappers": _wrapped_decider,
    "explicit rewriter overrides equal to the client's": lambda c, _: _decide_config(
        c.jev(), c.client(), rewriter={"temperature": 0.0, "max_tokens": 1024}
    ),
}


@pytest.mark.parametrize("name", list(DIFFERENT_DECISIONS))
async def test_different_decider_requests_get_different_run_fingerprints(name, tmp_path):
    capture = Capture()
    base = _decide_config(capture.jev(), capture.client())
    variant = DIFFERENT_DECISIONS[name](capture, tmp_path)
    assert await _decide_bodies(capture, base) != await _decide_bodies(capture, variant)
    assert _decide_fingerprint(base) != _decide_fingerprint(variant)


@pytest.mark.parametrize("name", list(SAME_DECISIONS))
async def test_identical_decider_requests_keep_the_run_fingerprint(name, tmp_path):
    capture = Capture()
    base = _decide_config(capture.jev(), capture.client())
    variant = SAME_DECISIONS[name](capture, tmp_path)
    assert await _decide_bodies(capture, base) == await _decide_bodies(capture, variant)
    assert _decide_fingerprint(base) == _decide_fingerprint(variant)


def test_default_decider_stages_add_no_identity_component():
    capture = Capture()
    config = _decide_config(capture.jev(), capture.client())
    common = {
        "templates": DECIDE_TEMPLATES,
        "questions": QUESTIONS,
        "store": DECIDE_STORE,
        "retrievers": [Scripted({})],
        "decider": config["decider"],
        "policy": config["policy"],
    }
    from xwalk.batch import decision_run_fingerprint_components

    stages = {k: config[k] for k in ("screener", "chooser", "gate")}
    staged = decision_run_fingerprint_components(**common, **stages, rewriter=None)
    assert "stages" not in staged and "rewrite" not in staged
    with_rewriter = decision_run_fingerprint_components(
        **common, **stages, rewriter=config["rewriter"]
    )
    assert "rewrite" in with_rewriter and "sk-one" not in json.dumps(with_rewriter)


# --- review follow-ups ----------------------------------------------------------------


def _jev_rewrite_job(tmp_path: Path, **llm: Any) -> Any:
    from xwalk.config import load_job

    rewrite = {**_REWRITE, "llm": {**_REWRITE["llm"], **llm}}
    return load_job(
        _job(tmp_path, "job_tiny_jev.yaml", lambda d: d["decider"].update(rewrite=rewrite))
    )


def test_the_rewrite_fingerprint_hashes_no_credential_in_its_base_url(tmp_path, monkeypatch):
    """Userinfo and key parameters in `decider.rewrite.llm.base_url` are credentials:
    they must never reach a hash input, and rotating them must not change identity."""
    import xwalk.config as config

    inputs: list[str] = []
    real = config.hash_value

    def recording(value: Any) -> str:
        inputs.append(json.dumps(value, default=str))
        return real(value)

    monkeypatch.setattr(config, "hash_value", recording)

    def fingerprint(url: str) -> str:
        job = _jev_rewrite_job(tmp_path, base_url=url)
        return job.decision_run_fingerprint(
            store=DECIDE_STORE, retrievers=[Scripted({})], decider=FakeDecider()
        )

    one = fingerprint("https://user:pw-one@llm.example/v1?api_key=key-one&region=eu")
    two = fingerprint("https://user:pw-two@llm.example/v1?api_key=key-two&region=eu")
    other = fingerprint("https://llm.example/v1?region=us")
    assert one == two != other
    assert inputs
    assert not any(s in text for text in inputs for s in ("pw-one", "key-one", "pw-two"))


def test_the_fingerprint_builders_require_every_stage():
    """An omitted stage used to count as the default, so a stage built with
    `max_tokens=256` but not passed resumed the default run with zero calls."""
    capture = Capture()
    config = _config(capture.client(), scorer={"max_tokens": 256})
    old_signature: dict[str, Any] = {
        "templates": TEMPLATES,
        "prompts": PROMPTS,
        "store": STORE,
        "retrievers": [ScriptedRetriever({})],
        "llm": config["llm"],
        "policy": config["policy"],
        "selector_policy": SelectorPolicy(),
    }
    with pytest.raises(TypeError, match="scorer"):
        build_run_fingerprint(**old_signature)
    with pytest.raises(TypeError, match="scorer"):
        run_fingerprint_components(**old_signature)

    decide = _decide_config(capture.jev(), capture.client())
    old_decision: dict[str, Any] = {
        "templates": DECIDE_TEMPLATES,
        "questions": QUESTIONS,
        "store": DECIDE_STORE,
        "retrievers": [Scripted({})],
        "decider": decide["decider"],
        "policy": decide["policy"],
    }
    with pytest.raises(TypeError, match="screener"):
        build_decision_run_fingerprint(**old_decision)
    stages = {k: decide[k] for k in ("screener", "chooser", "gate")}
    with pytest.raises(TypeError, match="rewriter"):
        build_decision_run_fingerprint(**old_decision, **stages)
    # "No rewriter" is said explicitly, and is not the same run as one with a rewriter.
    without = build_decision_run_fingerprint(**old_decision, **stages, rewriter=None)
    assert without != _decide_fingerprint(decide)


@pytest.mark.parametrize("kind", ["openai_compat", "litellm"])
def test_job_rewrite_identity_is_computed_without_building_a_client(tmp_path, monkeypatch, kind):
    """Comparing an injected rewrite client with the job's block must not construct the
    job's client (an HTTP client can fail to build, e.g. on a bad SSL_CERT_FILE)."""
    import xwalk.llm.litellm as litellm_module
    import xwalk.llm.openai_compat as openai_module

    monkeypatch.setenv("XWALK_TEST_API_KEY", "sk-env")
    job = _jev_rewrite_job(tmp_path, kind=kind, temperature=0.4)
    common: dict[str, Any] = {"store": DECIDE_STORE, "retrievers": [Scripted({})]}
    built = job.build_rewrite_llm()  # what the job builds, before construction is forbidden
    expected = job.decision_run_fingerprint(**common, decider=FakeDecider(), rewrite_llm=built)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("identity must not construct a client")

    monkeypatch.setattr(openai_module.OpenAICompatClient, "__init__", forbidden)
    monkeypatch.setattr(litellm_module.LiteLLMClient, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", forbidden)
    legacy = job.decision_run_fingerprint(**common, decider=FakeDecider())
    again = job.decision_run_fingerprint(**common, decider=FakeDecider(), rewrite_llm=built)
    injected = job.decision_run_fingerprint(
        **common, decider=FakeDecider(), rewrite_llm=FakeLLM(["x"])
    )
    assert expected == again == legacy != injected


# --- LiteLLM: captured completion kwargs ----------------------------------------------


class LiteLLMCapture:
    """LiteLLM clients whose `acompletion` kwargs land in `bodies`; no litellm import."""

    def __init__(self, monkeypatch: Any) -> None:
        import xwalk.llm.litellm as litellm_module

        self.bodies: list[dict[str, Any]] = []
        bodies = self.bodies

        class Module:
            @staticmethod
            async def acompletion(**kwargs: Any) -> Any:
                bodies.append(kwargs)
                return {
                    "model": "m",
                    "choices": [{"message": {"content": REPLY}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        monkeypatch.setattr(litellm_module, "require", lambda *a, **k: Module)

    def client(self, **kwargs: Any) -> Any:
        from xwalk.llm.litellm import LiteLLMClient

        settings: dict[str, Any] = {"api_key": "sk-one", "api_base": "https://proxy.test/v1"}
        settings.update(kwargs)
        return LiteLLMClient("openai/m", **settings)


def _without_key(bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in body.items() if k != "api_key"} for body in bodies]


LITELLM_DIFFERENT: dict[str, Callable[[LiteLLMCapture], dict[str, Any]]] = {
    "a schema capability": lambda c: _config(
        c.client(seed=7, capabilities=LLMCapabilities(json_schema=True))
    ),
    "a seed capability with a seed": lambda c: _config(
        c.client(seed=7, capabilities=LLMCapabilities(seed=True))
    ),
    "a stage max_tokens override": lambda c: _config(c.client(seed=7), scorer={"max_tokens": 256}),
    "a stage temperature override": lambda c: _config(
        c.client(seed=7), selector={"temperature": 0.5}
    ),
}


@pytest.mark.parametrize("name", list(LITELLM_DIFFERENT))
async def test_litellm_requests_that_differ_get_different_fingerprints(name, monkeypatch):
    capture = LiteLLMCapture(monkeypatch)
    base = _config(capture.client(seed=7))
    variant = LITELLM_DIFFERENT[name](capture)
    before = _without_key(await _stage_bodies(capture, base))  # type: ignore[arg-type]
    after = _without_key(await _stage_bodies(capture, variant))  # type: ignore[arg-type]
    assert before != after
    assert _fingerprint(base) != _fingerprint(variant)


async def test_litellm_strictness_is_identity_only_with_a_schema(monkeypatch):
    capture = LiteLLMCapture(monkeypatch)
    loose = _config(capture.client(capabilities=LLMCapabilities(json_schema=True)))
    strict = _config(
        capture.client(capabilities=LLMCapabilities(json_schema=True, strict_schema=True))
    )
    assert await _stage_bodies(capture, loose) != await _stage_bodies(capture, strict)  # type: ignore[arg-type]
    assert _fingerprint(loose) != _fingerprint(strict)

    plain = _config(capture.client())
    strict_alone = _config(capture.client(capabilities=LLMCapabilities(strict_schema=True)))
    assert await _stage_bodies(capture, plain) == await _stage_bodies(capture, strict_alone)  # type: ignore[arg-type]
    assert _fingerprint(plain) == _fingerprint(strict_alone)


async def test_litellm_credential_rotation_keeps_the_fingerprint(monkeypatch):
    capture = LiteLLMCapture(monkeypatch)
    one = _config(capture.client(api_key="sk-one"))
    two = _config(capture.client(api_key="sk-two"))
    first = await _stage_bodies(capture, one)  # type: ignore[arg-type]
    second = await _stage_bodies(capture, two)  # type: ignore[arg-type]
    assert [b["api_key"] for b in first + second] == ["sk-one"] * 4 + ["sk-two"] * 4
    assert _without_key(first) == _without_key(second)
    assert _fingerprint(one) == _fingerprint(two)


@pytest.mark.parametrize(
    "url",
    [
        "http://jev.example/decide?deployment=a",
        "https://jev.example/decide?deployment=b",
        "https://jev.example/decide?deployment=a&flag=",
    ],
)
async def test_jev_full_endpoint_changes_run_identity(url):
    one = JevClient("https://jev.example/decide?deployment=a", "jev")
    two = JevClient(url, "jev")
    try:
        assert _decide_fingerprint(_decide_config(one, FakeLLM([]))) != _decide_fingerprint(
            _decide_config(two, FakeLLM([]))
        )
    finally:
        await one.aclose()
        await two.aclose()


async def test_jev_identity_never_hashes_url_credentials(monkeypatch):
    import xwalk.decide.jev as jev
    from xwalk.fingerprint import hash_value

    inputs = []
    monkeypatch.setattr(jev, "hash_value", lambda value: inputs.append(value) or hash_value(value))
    one = JevClient("https://alice:secret@jev.example/decide?api_key=hidden&deployment=a", "jev")
    two = JevClient("https://bob:rotated@jev.example/decide?token=changed&deployment=a", "jev")
    try:
        assert one.fingerprint == two.fingerprint
        assert (
            inputs
            == [
                {
                    "adapter": "jev",
                    "adapter_version": 2,
                    "endpoint": "https://jev.example/decide?deployment=a",
                    "model": "jev",
                }
            ]
            * 2
        )
    finally:
        await one.aclose()
        await two.aclose()

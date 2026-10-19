"""Templates from job files, slots files and skeleton directories are untrusted input.

Security review finding F1: the record templates ran in a plain Jinja environment, so a
job file could reach `''.__class__.__mro__[1].__subclasses__()` and from there any
Python object -- and `xwalk validate` alone compiled them. Every Jinja environment now
sandboxes, and the bundled templates render exactly as they did before.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment

from xwalk import ops
from xwalk.cluster.job import is_cluster_job, load_cluster_job
from xwalk.config import load_job
from xwalk.prompts.contract import (
    BASE_DIR,
    ContractError,
    PromptSet,
    PromptSlots,
    load_slots,
    validate_contract,
)
from xwalk.records import Record
from xwalk.templates import TemplateError, TemplateSet, _EmptyUndefined, _tidy

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

SSTI = "{{ ''.__class__.__mro__[1].__subclasses__() | length }}"
SAFE = dict(query="{{ mention }}", context="{{ mention }}", doc="{{ label }}", candidate="{{ id }}")

# Payloads the compile-time check refuses outright: a constant dunder in attribute or
# subscript position never reads a record field.
STATIC_PAYLOADS = [
    SSTI,
    "{{ ''.__class__ }}",
    "{{ mention.__class__.__init__.__globals__ }}",
    "{{ mention['__class__'] }}",
    "{% if mention %}{{ ().__class__.__base__.__subclasses__() }}{% endif %}",
    "{% for x in [1] %}{{ x.__class__ }}{% endfor %}",
]

# Payloads that only the sandbox can see, because the attribute name is computed.
DYNAMIC_PAYLOADS = [
    "{{ mention | attr('__class__') }}",
    "{{ mention | attr('__cla' ~ 'ss__') | string }}",
    "{{ mention[key] }}",
    "{{ '{0.__class__}'.format(mention) }}",
]


def _record(**fields: Any) -> Record:
    return Record(id="s1", fields={"mention": "glucose", "key": "__class__", **fields})


# --- record templates ---------------------------------------------------------------


@pytest.mark.parametrize("slot", ["query", "context", "doc", "candidate"])
@pytest.mark.parametrize("payload", STATIC_PAYLOADS)
def test_a_dunder_traversal_is_refused_when_the_templates_are_built(slot, payload):
    with pytest.raises(TemplateError, match=f"{slot} template.*unsafe"):
        TemplateSet(**{**SAFE, slot: payload})


def test_an_extra_query_template_is_checked_too():
    with pytest.raises(TemplateError, match=r"queries\[1\] template.*unsafe"):
        TemplateSet(**SAFE, queries=("{{ mention }}", SSTI))


@pytest.mark.parametrize("payload", DYNAMIC_PAYLOADS)
def test_a_computed_attribute_is_refused_by_the_sandbox_at_render(payload):
    ts = TemplateSet(**{**SAFE, "query": payload})
    with pytest.raises(TemplateError, match="query template failed on record 's1'"):
        ts.render_query(_record())


def test_the_sandbox_refusal_is_not_hidden_by_the_empty_undefined():
    # `_EmptyUndefined` renders a missing field as "". An attribute the sandbox refuses
    # comes back as an undefined too; it must raise, not render as an empty slot.
    ts = TemplateSet(**{**SAFE, "query": "{{ mention | attr('__class__') }}"})
    with pytest.raises(TemplateError, match="unsafe|access to attribute"):
        ts.render_query(_record())


def test_a_template_cannot_mutate_a_record_field():
    ts = TemplateSet(**{**SAFE, "query": "{{ synonyms.append('x') }}{{ synonyms | length }}"})
    record = _record(synonyms=["a"])
    with pytest.raises(TemplateError):
        ts.render_query(record)
    assert record.fields["synonyms"] == ["a"]


def test_underscore_fields_of_a_mapping_still_render():
    # Only dunders are refused statically; a JSON field such as `_id` is data.
    ts = TemplateSet(**{**SAFE, "query": "{{ meta._id }} {{ meta['_id'] }}"})
    assert ts.render_query(_record(meta={"_id": "x1"})) == "x1 x1"


# --- validate refuses a hostile job file --------------------------------------------


def _hostile_job(tmp_path: Path, source: Path, **templates: Any) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["templates"].update(templates)
    for section in ("target", "source"):
        if section in data and "path" in data[section]:
            data[section]["path"] = str((source.parent / data[section]["path"]).resolve())
    if "prompts" in data:
        data["prompts"]["slots"] = str((source.parent / data["prompts"]["slots"]).resolve())
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("fixture", ["job_tiny.yaml", "job_tiny_jev.yaml"])
def test_validate_reports_an_ssti_template_as_an_invalid_job(tmp_path, fixture):
    path = _hostile_job(tmp_path, FIXTURES / fixture, context=SSTI)
    result = ops.validate(path, check_credentials=False)
    assert result.exit_code != 0
    codes = {e.code: e.message for e in result.errors}
    assert "template_invalid" in codes
    assert "unsafe" in codes["template_invalid"]


def test_validate_reports_a_cluster_ssti_template_as_an_invalid_job(tmp_path):
    path = _hostile_job(tmp_path, FIXTURES / "cluster_tiny" / "job.yaml", query=SSTI)
    result = ops.validate(path, check_credentials=False)
    assert result.exit_code != 0
    assert any(e.code == "template_invalid" and "unsafe" in e.message for e in result.errors)


def test_validate_reports_a_dynamic_payload_while_scanning_records(tmp_path):
    path = _hostile_job(
        tmp_path, FIXTURES / "job_tiny.yaml", query="{{ mention | attr('__class__') }}"
    )
    result = ops.validate(path, check_credentials=False)
    assert result.exit_code != 0
    assert any("query template failed" in e.message for e in result.errors)


# --- prompt slots and skeletons -----------------------------------------------------


def _slots(**overrides: Any) -> PromptSlots:
    data: dict[str, Any] = {
        "entity_noun": "mention",
        "target_noun": "term",
        "domain_brief": "chemistry",
        "rubric": [
            {"score": 1.0, "name": "Certain", "when": "exact"},
            {"score": 0.0, "name": "None", "when": "different"},
        ],
    }
    data.update(overrides)
    return PromptSlots.model_validate(data)


def test_a_slot_value_is_data_and_is_never_evaluated():
    slots = _slots(domain_brief=SSTI, hard_rules=[SSTI], disambiguation_steps=SSTI)
    prompts = PromptSet.from_slots(slots)
    validate_contract(prompts)
    text = prompts.render_select(source_fields={"m": SSTI}, context=SSTI, candidate_block="[C01]")
    assert text.count(SSTI) == 5


def test_a_hostile_skeleton_is_refused_as_a_contract_error(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name in ("select", "score", "verify", "rewrite"):
        (base / f"{name}.j2").write_text((BASE_DIR / f"{name}.j2").read_text(), encoding="utf-8")
    hostile = "{{ slots.__class__.__init__.__globals__ }}\n" + (BASE_DIR / "select.j2").read_text()
    (base / "select.j2").write_text(hostile, encoding="utf-8")
    prompts = PromptSet.from_slots(_slots(), base_dir=base)
    with pytest.raises(ContractError, match="select skeleton.*unsafe"):
        validate_contract(prompts)


def test_a_skeleton_computing_an_attribute_is_refused_at_render(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name in ("select", "score", "verify", "rewrite"):
        (base / f"{name}.j2").write_text((BASE_DIR / f"{name}.j2").read_text(), encoding="utf-8")
    hostile = "{{ slots | attr('__class__') }}\n" + (BASE_DIR / "select.j2").read_text()
    (base / "select.j2").write_text(hostile, encoding="utf-8")
    prompts = PromptSet.from_slots(_slots(), base_dir=base)
    with pytest.raises(ContractError, match="select skeleton"):
        validate_contract(prompts)


# --- bundled templates render exactly as an unsandboxed environment does ---------------

_JOBS = sorted(
    set(ROOT.glob("examples/**/job*.yaml"))
    | set(FIXTURES.glob("**/job*.yaml"))
    | {ROOT / "src" / "xwalk" / "resources" / "quickstart" / "job.yaml"}
)
_SYNTHETIC = [
    Record(
        "x1",
        {
            "text": "glucose",
            "mention": "glucose",
            "label": "Glucose",
            "synonyms": ["dextrose", "D-glucose"],
            "definition": "a sugar",
            "context": "",
            "context_left": "blood ",
            "context_right": " level",
            "aliases": None,
            "ontology": "CHEBI",
        },
    ),
    Record("x2", {"text": "x  y", "label": "L", "synonyms": [], "definition": None}),
    Record("x3", {}),
]


def _reference(source: str, record: Record) -> str:
    """How the pre-sandbox code rendered one template: a plain environment."""
    env = Environment(undefined=_EmptyUndefined, keep_trailing_newline=False)
    variables = {k: v for k, v in record.fields.items() if v is not None}
    variables["id"] = record.id
    return _tidy(env.from_string(source).render(**variables))


def _records(builder: Any) -> list[Record]:
    try:
        return list(itertools.islice(builder(), 200))
    except Exception:  # noqa: BLE001 - data too large to ship, or an extra not installed
        return []


def _template_sources(ts: TemplateSet) -> dict[str, str]:
    sources = {name: getattr(ts, name) for name in ("query", "context", "doc", "candidate")}
    sources.update({f"queries.{i}": q for i, q in enumerate(ts.queries)})
    return sources


@pytest.mark.parametrize("job", _JOBS, ids=lambda p: str(p.relative_to(ROOT)))
def test_bundled_job_templates_render_as_before(job):
    if is_cluster_job(job):
        spec: Any = load_cluster_job(job)
        records = _records(spec.build_source_records)
    else:
        spec = load_job(job)
        records = _records(spec.build_source_records) + _records(spec.build_target_records)
    ts = spec.build_templates()
    rendered = 0
    for record in records + _SYNTHETIC:
        for name, source in _template_sources(ts).items():
            assert ts._render(name, record) == _reference(source, record), (name, record.id)
            rendered += 1
    assert rendered


def test_the_workbench_and_chemistry_templates_render_as_before():
    from xwalk.ui import projects

    chemistry = yaml.safe_load((ROOT / "examples" / "chemistry" / "templates.yaml").read_text())
    for templates in (projects._TEMPLATES, chemistry):
        ts = TemplateSet(**{k: templates[k] for k in ("query", "context", "doc", "candidate")})
        for record in _SYNTHETIC:
            for name, source in _template_sources(ts).items():
                assert ts._render(name, record) == _reference(source, record)


@pytest.mark.parametrize(
    "slots_path",
    sorted(ROOT.glob("examples/**/slots*.yaml"))
    + [ROOT / "src" / "xwalk" / "resources" / "quickstart" / "slots.yaml"],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_bundled_prompts_render_as_before(slots_path):
    from jinja2 import FileSystemLoader, StrictUndefined

    reference_env = Environment(
        loader=FileSystemLoader(str(BASE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    prompts = PromptSet.from_slots(load_slots(slots_path))
    fields = {"mention": "glucose", "organism": "Homo sapiens", "synonyms": ["a", "b"]}
    variables: dict[str, dict[str, Any]] = {
        "select": dict(source_fields=fields, context="ctx", candidate_block="[C01] x"),
        "score": dict(
            source_fields=fields,
            context="ctx",
            chosen_block="[C01] x",
            other_candidates="[C02] y",
            review_floor=0.4,
        ),
        "verify": dict(
            source_fields=fields, context="ctx", chosen_block="[C01] x", other_candidates=""
        ),
        "rewrite": dict(
            source_fields=fields,
            context="ctx",
            previous_queries=["a", "b"],
            best_candidates="",
            max_queries=3,
        ),
    }
    for name, kwargs in variables.items():
        actual = getattr(prompts, f"render_{name}")(**kwargs)
        expected = (
            reference_env.from_string(prompts.skeletons[name])
            .render(slots=prompts.slots, **kwargs)
            .strip()
        )
        assert actual == expected, name

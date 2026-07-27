import pytest

from xwalk.records import Record
from xwalk.templates import TemplateError, TemplateSet

TINY = dict(
    query="{{ mention }}",
    context=(
        "{% if context_left %}{{ context_left }} [{{ mention }}] {{ context_right }}{% endif %}"
    ),
    doc="{{ label }} {{ synonyms | join(' ') }}",
    candidate="ID: {{ id }}\nLabel: {{ label }}",
)


def test_renders_a_single_field():
    ts = TemplateSet(**TINY)
    assert ts.render_query(Record(id="s1", fields={"mention": "glucose"})) == "glucose"


def test_missing_field_renders_empty_not_an_error():
    ts = TemplateSet(**TINY)
    assert ts.render_context(Record(id="s1", fields={"mention": "glucose"})) == ""


def test_record_id_is_available_as_id():
    ts = TemplateSet(**TINY)
    out = ts.render_candidate(Record(id="CHEBI:17234", fields={"label": "glucose"}))
    assert out == "ID: CHEBI:17234\nLabel: glucose"


def test_list_fields_work_with_filters():
    ts = TemplateSet(**TINY)
    record = Record(id="t1", fields={"label": "glucose", "synonyms": ["dextrose", "grape sugar"]})
    assert ts.render_doc(record) == "glucose dextrose grape sugar"


def test_whitespace_is_collapsed_and_trimmed():
    ts = TemplateSet(query="  {{ a }}   {{ b }}  ", context="", doc="", candidate="")
    assert ts.render_query(Record(id="s1", fields={"a": "x", "b": "y"})) == "x y"


def test_blank_lines_from_skipped_conditionals_are_removed():
    ts = TemplateSet(
        query="",
        context="",
        doc="",
        candidate="ID: {{ id }}\n{% if defn %}Def: {{ defn }}\n{% endif %}Label: {{ label }}",
    )
    out = ts.render_candidate(Record(id="t1", fields={"label": "glucose"}))
    assert out == "ID: t1\nLabel: glucose"


def test_syntax_error_raises_at_construction_not_at_render():
    with pytest.raises(TemplateError, match="query"):
        TemplateSet(query="{{ unclosed ", context="", doc="", candidate="")


def test_fingerprint_changes_when_any_template_changes():
    a = TemplateSet(**TINY)
    b = TemplateSet(**{**TINY, "doc": "{{ label }}"})
    assert a.fingerprint != b.fingerprint


def test_fingerprint_is_stable_for_identical_templates():
    assert TemplateSet(**TINY).fingerprint == TemplateSet(**TINY).fingerprint


def test_id_field_in_fields_does_not_shadow_record_id():
    ts = TemplateSet(query="{{ id }}", context="", doc="", candidate="")
    record = Record(id="real", fields={"id": "spoofed"})
    assert ts.render_query(record) == "real"

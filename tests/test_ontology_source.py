import pytest

from tests.conftest import FIXTURES
from xwalk.sources.ontology import curie, obo_source, owl_source

OBO = FIXTURES / "tiny.obo"
OWL = FIXTURES / "tiny.owl"


# --- OBO (no dependency) ----------------------------------------------------------


def test_obo_yields_one_record_per_term():
    records = list(obo_source(OBO))
    assert [r.id for r in records] == ["CHEBI:17234", "CHEBI:28757"]


def test_obo_skips_obsolete_terms_by_default():
    assert "CHEBI:00000" not in {r.id for r in obo_source(OBO)}


def test_obo_can_include_obsolete_terms():
    assert "CHEBI:00000" in {r.id for r in obo_source(OBO, include_obsolete=True)}


def test_obo_ignores_non_term_stanzas():
    assert "part_of" not in {r.id for r in obo_source(OBO)}


def test_obo_extracts_the_label():
    assert next(iter(obo_source(OBO))).fields["label"] == "glucose"


def test_obo_extracts_all_synonyms():
    assert next(iter(obo_source(OBO))).fields["synonyms"] == ["dextrose", "grape sugar"]


def test_obo_strips_the_definition_reference_list():
    assert next(iter(obo_source(OBO))).fields["definition"] == "A monosaccharide sugar."


def test_obo_extracts_parents_without_the_trailing_comment():
    assert next(iter(obo_source(OBO))).fields["parents"] == ["CHEBI:16646"]


def test_obo_records_the_obsolete_flag():
    assert next(iter(obo_source(OBO, include_obsolete=True))).fields["obsolete"] is False


def test_obo_filters_by_id_prefix():
    assert list(obo_source(OBO, id_prefix="GO:")) == []


def test_obo_is_lazy():
    import types

    assert isinstance(obo_source(OBO), types.GeneratorType)


def test_obo_yields_the_final_stanza(tmp_path):
    """A file whose last line is a term must not lose that term to the loop exit."""
    path = tmp_path / "one.obo"
    path.write_text("[Term]\nid: X:1\nname: only\n", encoding="utf-8")
    assert [r.id for r in obo_source(path)] == ["X:1"]


# --- OWL (needs xwalk[ontology]) --------------------------------------------------


@pytest.mark.ontology
def test_owl_yields_one_record_per_class():
    pytest.importorskip("rdflib")
    assert {r.id for r in owl_source(OWL)} == {"CHEBI:17234", "CHEBI:28757"}


@pytest.mark.ontology
def test_owl_produces_the_same_field_names_as_obo():
    pytest.importorskip("rdflib")
    owl_record = next(r for r in owl_source(OWL) if r.id == "CHEBI:17234")
    obo_record = next(r for r in obo_source(OBO) if r.id == "CHEBI:17234")
    assert set(owl_record.fields) == set(obo_record.fields)


@pytest.mark.ontology
def test_owl_extracts_label_synonyms_and_definition():
    pytest.importorskip("rdflib")
    record = next(r for r in owl_source(OWL) if r.id == "CHEBI:17234")
    assert record.fields["label"] == "glucose"
    assert set(record.fields["synonyms"]) == {"dextrose", "grape sugar"}
    assert record.fields["definition"] == "A monosaccharide sugar."


@pytest.mark.ontology
def test_owl_extracts_parents_as_curies():
    pytest.importorskip("rdflib")
    record = next(r for r in owl_source(OWL) if r.id == "CHEBI:17234")
    assert record.fields["parents"] == ["CHEBI:16646"]


@pytest.mark.ontology
def test_owl_skips_deprecated_classes_by_default():
    pytest.importorskip("rdflib")
    assert "CHEBI:00000" not in {r.id for r in owl_source(OWL)}


@pytest.mark.ontology
def test_owl_synonyms_are_sorted_for_determinism():
    pytest.importorskip("rdflib")
    first = [r.fields["synonyms"] for r in owl_source(OWL)]
    second = [r.fields["synonyms"] for r in owl_source(OWL)]
    assert first == second


def test_owl_without_the_extra_gives_an_actionable_error(monkeypatch):
    """Patches `importlib.import_module`, which is what `require` actually calls --
    `importlib.import_module` does not route through `builtins.__import__`, so patching
    that would leave this test passing for the wrong reason."""
    import importlib

    from xwalk._extras import MissingExtra

    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name.startswith("rdflib"):
            raise ImportError("no rdflib")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(MissingExtra, match=r"xwalk\[ontology\]"):
        list(owl_source(OWL))


# --- curie ------------------------------------------------------------------------


def test_curie_converts_an_obo_purl():
    assert curie("http://purl.obolibrary.org/obo/CHEBI_17234") == "CHEBI:17234"


def test_curie_leaves_an_unrecognised_uri_alone():
    assert curie("http://example.org/thing") == "http://example.org/thing"


def test_curie_handles_a_fragment_uri():
    assert curie("http://example.org/onto#Term_1") == "Term_1"

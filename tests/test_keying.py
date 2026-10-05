import pytest

from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stages.keying import Resolution, assign_keys, resolve_key
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(query="", context="", doc="", candidate="ID: {{ id }}\nLabel: {{ label }}")


def candidate(record_id: str, label: str, score: float = 0.5) -> Candidate:
    return Candidate(
        record=Record(id=record_id, fields={"label": label}),
        fused_score=score,
        evidence=(RetrievalHit(record_id=record_id, retriever="bm25", raw_score=1.0, rank=1),),
    )


@pytest.fixture
def keyed():
    return assign_keys(
        [candidate("NCBIGene:3", "A2MP1"), candidate("NCBIGene:7", "AANAT")], TEMPLATES
    )


def test_keys_are_one_based_and_zero_padded(keyed):
    assert keyed.order == ("C01", "C02")


def test_key_width_grows_past_ninety_nine():
    many = [candidate(f"T{i}", f"L{i}") for i in range(120)]
    keyed = assign_keys(many, TEMPLATES)
    assert keyed.order[0] == "C001" and keyed.order[-1] == "C120"


def test_rendered_block_shows_the_key_before_the_record(keyed):
    assert keyed.rendered.startswith("[C01] ID: NCBIGene:3")


def test_rendered_block_contains_every_candidate(keyed):
    assert "[C02]" in keyed.rendered and "AANAT" in keyed.rendered


def test_issued_maps_keys_to_record_ids(keyed):
    assert keyed.issued == {"C01": "NCBIGene:3", "C02": "NCBIGene:7"}


def test_empty_candidate_list_produces_no_keys():
    keyed = assign_keys([], TEMPLATES)
    assert keyed.order == () and keyed.rendered == ""


# --- resolution: the happy paths -------------------------------------------------


def test_resolves_an_issued_key(keyed):
    choice = resolve_key("C01", keyed)
    assert choice.record_id == "NCBIGene:3"
    assert choice.resolution is Resolution.EXACT_KEY


def test_tolerates_surrounding_whitespace_and_brackets(keyed):
    for raw in [" C01 ", "[C01]", "\nC01\n", '"C01"']:
        assert resolve_key(raw, keyed).record_id == "NCBIGene:3"


def test_key_case_is_normalised(keyed):
    """Safe here and only here: the key space is issued by us, so c01 and C01 cannot
    denote two different records. Identifier case-folding is a different matter."""
    assert resolve_key("c01", keyed).record_id == "NCBIGene:3"


@pytest.mark.parametrize("raw", [None, "null", "NULL", "none", "", "  ", "no_match", "NO MATCH"])
def test_abstention_is_recognised(raw, keyed):
    choice = resolve_key(raw, keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.ABSTAIN


# --- resolution: the adversarial cases opaque keys exist to prevent ---------------


def test_a_hallucinated_numeric_id_never_becomes_a_rank(keyed):
    """The exact failure mode of the paper repo's resolver: '3' must not mean C03,
    and must not become NCBIGene:3 either."""
    choice = resolve_key("3", keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_a_raw_target_id_does_not_resolve(keyed):
    choice = resolve_key("NCBIGene:3", keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_an_id_differing_only_by_case_does_not_resolve(keyed):
    assert resolve_key("ncbigene:3", keyed).resolution is Resolution.UNRESOLVED


def test_a_key_not_issued_this_attempt_does_not_resolve(keyed):
    assert resolve_key("C09", keyed).resolution is Resolution.UNRESOLVED


def test_a_prefix_stripped_id_does_not_resolve(keyed):
    assert resolve_key("3", keyed).record_id is None


def test_an_id_from_another_namespace_does_not_resolve(keyed):
    """The spec's third adversarial case. `CHEBI:3` shares its local part with the
    candidate `NCBIGene:3`, which is exactly the collision CURIE-prefix stripping
    creates. Strict mode must not see them as the same entity."""
    choice = resolve_key("CHEBI:3", keyed)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_legacy_mode_also_refuses_a_foreign_namespace_suffix_collision(keyed):
    """Legacy mode keeps a suffix map for reproducing prior work. It must key on the
    bare suffix only, never accept a *different* namespace that happens to end in it —
    otherwise `CHEBI:3` silently becomes the real record `NCBIGene:3`."""
    choice = resolve_key("CHEBI:3", keyed, legacy=True)
    assert choice.record_id is None
    assert choice.resolution is Resolution.UNRESOLVED


def test_a_key_issued_by_a_previous_attempt_does_not_resolve():
    """Keys are issued per attempt. A five-candidate attempt issues C05; if the next
    attempt has two candidates, C05 must not resolve against it."""
    wide = assign_keys([candidate(f"T{i}", f"L{i}") for i in range(5)], TEMPLATES)
    narrow = assign_keys([candidate("T0", "L0"), candidate("T1", "L1")], TEMPLATES)
    assert resolve_key("C05", wide).record_id == "T4"
    assert resolve_key("C05", narrow).resolution is Resolution.UNRESOLVED


def test_prose_does_not_resolve(keyed):
    assert resolve_key("I think it is C01, the first one", keyed).resolution is (
        Resolution.UNRESOLVED
    )


def test_the_raw_answer_is_always_preserved_for_the_trace(keyed):
    assert resolve_key("banana", keyed).raw == "banana"


# --- legacy mode -----------------------------------------------------------------


def test_legacy_mode_accepts_an_exact_record_id(keyed):
    choice = resolve_key("NCBIGene:3", keyed, legacy=True)
    assert choice.record_id == "NCBIGene:3"
    assert choice.resolution is Resolution.LEGACY_EXACT_ID


def test_legacy_mode_accepts_a_case_insensitive_id_but_flags_it(keyed):
    choice = resolve_key("ncbigene:3", keyed, legacy=True)
    assert choice.record_id == "NCBIGene:3"
    assert choice.resolution is Resolution.LEGACY_FUZZY


def test_legacy_mode_accepts_a_bare_rank_but_flags_it(keyed):
    choice = resolve_key("2", keyed, legacy=True)
    assert choice.record_id == "NCBIGene:7"
    assert choice.resolution is Resolution.LEGACY_RANK


def test_legacy_mode_still_prefers_an_exact_key(keyed):
    assert resolve_key("C02", keyed, legacy=True).resolution is Resolution.EXACT_KEY


def test_legacy_mode_still_fails_on_a_genuinely_unknown_id(keyed):
    assert resolve_key("CHEBI:99999", keyed, legacy=True).resolution is Resolution.UNRESOLVED


def test_legacy_rank_out_of_range_does_not_resolve(keyed):
    assert resolve_key("99", keyed, legacy=True).resolution is Resolution.UNRESOLVED


def test_every_non_exact_resolution_is_distinguishable_from_exact(keyed):
    """Task 14 relies on this: anything other than EXACT_KEY or ABSTAIN forces review."""
    fuzzy = {Resolution.LEGACY_EXACT_ID, Resolution.LEGACY_FUZZY, Resolution.LEGACY_RANK}
    assert Resolution.EXACT_KEY not in fuzzy and Resolution.ABSTAIN not in fuzzy


def test_each_candidate_keeps_its_own_block_even_with_blank_lines():
    class Paragraphs(TemplateSet):
        def render_candidate(self, record):
            return f"{record.fields['label']}\n\nmore about {record.fields['label']}"

    keyed = assign_keys(
        [candidate("T1", "first"), candidate("T2", "second")],
        Paragraphs(query="", context="", doc="", candidate=""),
    )
    assert keyed.blocks["C01"] == "[C01] first\n\nmore about first"
    assert keyed.render_except("C01") == "[C02] second\n\nmore about second"
    assert keyed.rendered == keyed.blocks["C01"] + "\n\n" + keyed.blocks["C02"]

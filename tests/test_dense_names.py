"""Per-name vectors: `DenseRetriever` with `name_fields`.

Each distinct value of the named fields gets its own vector beside the whole-document
vector, and a record scores by its best vector. A record with many synonyms is then no
longer diluted into one averaged vector.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from xwalk.records import Record
from xwalk.retrieval.base import IndexMismatchError, SearchRequest
from xwalk.retrieval.dense import DenseRetriever, index_components
from xwalk.templates import TemplateSet

TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="",
    doc="{{ label }} {{ synonyms | join(' ') }}",
    candidate="ID: {{ id }}",
)

# T1's document vector is dominated by the noise synonyms; its synonym "glucose" is an
# exact match for the query. T2 is merely close to "glucose" as a whole document.
RECORDS = [
    Record(
        id="T1",
        fields={
            "label": "dextrose",
            "synonyms": ["zzzzzzzzzz", "wwwwwwwwwwww", "glucose", "yyyyyyyyyy"],
        },
    ),
    Record(id="T2", fields={"label": "glucoses", "synonyms": []}),
    Record(id="T3", fields={"label": "automobile", "synonyms": ["car"]}),
]
NAMES = ("label", "synonyms")


class ToyEncoder:
    name = "toy"
    dimension = 26

    def __init__(self) -> None:
        self.calls: list[tuple[bool, list[str]]] = []

    def encode(self, texts, *, is_query: bool = False):
        self.calls.append((is_query, list(texts)))
        out = []
        for text in texts:
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / norm for v in vector])
        return out


class PrefixedEncoder(ToyEncoder):
    settings = {"doc_prefix": "passage: "}


def _search(retriever, text="glucose", limit=3):
    return asyncio.run(retriever.search(SearchRequest(text=text, limit=limit)))


def test_without_names_the_diluted_record_loses(tmp_path):
    """The premise: one vector per record ranks the merely-similar record first."""
    retriever = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())
    assert _search(retriever)[0].record_id == "T2"


def test_a_matching_name_ranks_its_record_first(tmp_path):
    retriever = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    hits = _search(retriever)
    assert hits[0].record_id == "T1"
    assert hits[0].raw_score == pytest.approx(1.0)


def test_each_record_appears_once_with_contiguous_ranks(tmp_path):
    retriever = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    hits = _search(retriever, limit=10)
    assert sorted(h.record_id for h in hits) == ["T1", "T2", "T3"]
    assert [h.rank for h in hits] == [1, 2, 3]


def test_the_limit_counts_records_not_vectors(tmp_path):
    retriever = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    assert len(_search(retriever, limit=2)) == 2


def test_the_whole_document_vector_is_kept(tmp_path):
    """A query matching only the document as a whole still finds it."""
    retriever = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    doc = TEMPLATES.render_doc(RECORDS[2])
    assert _search(retriever, text=doc)[0].raw_score == pytest.approx(1.0)


def test_names_are_encoded_as_documents_once_each(tmp_path):
    encoder = ToyEncoder()
    records = [Record(id="A", fields={"label": "milk", "synonyms": ["milk", "  ", None, "lait"]})]
    DenseRetriever.build(records, TEMPLATES, tmp_path / "idx", encoder, name_fields=NAMES)
    sent = [t for is_query, batch in encoder.calls if not is_query for t in batch]
    # the whole doc as the template renders it, then each distinct non-blank name once
    assert sent == [TEMPLATES.render_doc(records[0]), "milk", "lait"]


def test_a_scalar_field_and_a_missing_field_are_handled(tmp_path):
    records = [
        Record(id="A", fields={"label": "milk"}),
        Record(id="B", fields={"label": "water", "synonyms": "aqua"}),
    ]
    templates = TemplateSet(query="{{ mention }}", context="", doc="{{ label }}", candidate="")
    retriever = DenseRetriever.build(
        records, templates, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    assert _search(retriever, text="aqua")[0].record_id == "B"


def test_a_record_with_no_names_keeps_its_document_vector(tmp_path):
    records = [Record(id="A", fields={"label": None, "synonyms": []})]
    templates = TemplateSet(query="{{ mention }}", context="", doc="{{ id }} soup", candidate="")
    retriever = DenseRetriever.build(
        records, templates, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    assert [h.record_id for h in _search(retriever, text="soup")] == ["A"]


def test_the_identity_records_the_name_fields_only_when_set():
    plain = index_components(RECORDS, TEMPLATES, ToyEncoder())
    assert "name_fields" not in plain and plain["format_version"] == 1
    named = index_components(RECORDS, TEMPLATES, ToyEncoder(), name_fields=NAMES)
    assert named["name_fields"] == ["label", "synonyms"]
    assert named["format_version"] == 2
    other = index_components(RECORDS, TEMPLATES, ToyEncoder(), name_fields=("label",))
    assert other != named


def test_reopening_embeds_nothing_and_searches_the_same(tmp_path):
    built = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    encoder = ToyEncoder()
    expected = index_components(RECORDS, TEMPLATES, encoder, name_fields=NAMES)
    reopened = DenseRetriever.open(tmp_path / "idx", encoder, expected=expected)
    assert not encoder.calls
    assert reopened.fingerprint == built.fingerprint
    assert [h.record_id for h in _search(reopened)] == [h.record_id for h in _search(built)]


def test_a_different_name_fields_setting_is_refused_on_open(tmp_path):
    DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES)
    expected = index_components(RECORDS, TEMPLATES, ToyEncoder())
    with pytest.raises(IndexMismatchError) as info:
        DenseRetriever.open(tmp_path / "idx", ToyEncoder(), expected=expected)
    assert "name_fields" in info.value.differences


def test_an_index_without_names_is_unchanged_on_disk(tmp_path):
    """Existing jobs keep their index identity and files."""
    retriever = DenseRetriever.build(RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder())
    rows = (tmp_path / "idx" / "vectors.f32").stat().st_size // (26 * 4)
    assert rows == len(RECORDS)
    assert (
        retriever.fingerprint
        == DenseRetriever.build(
            RECORDS, TEMPLATES, tmp_path / "again", ToyEncoder(), name_fields=None
        ).fingerprint
    )


@pytest.mark.parametrize("backend", ["numpy", "python"])
def test_every_search_backend_agrees(tmp_path, backend):
    retriever = DenseRetriever.build(
        RECORDS, TEMPLATES, tmp_path / "idx", ToyEncoder(), name_fields=NAMES
    )
    if backend == "python":
        retriever._faiss_index = None
        retriever._numpy_vectors = None
    elif retriever._numpy_vectors is None:
        pytest.skip("numpy is not installed")
    hits = _search(retriever, limit=10)
    assert [h.record_id for h in hits][0] == "T1"
    assert sorted(h.record_id for h in hits) == ["T1", "T2", "T3"]


def test_names_use_the_document_prefix(tmp_path):
    """Names are documents: an encoder applies its document-side settings to them."""
    encoder = PrefixedEncoder()
    DenseRetriever.build(RECORDS[:1], TEMPLATES, tmp_path / "idx", encoder, name_fields=NAMES)
    assert all(not is_query for is_query, _ in encoder.calls)

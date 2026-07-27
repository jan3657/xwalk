import pytest

from xwalk.records import Record
from xwalk.sources.tabular import csv_source
from xwalk.stores.memory import MemoryStore


def test_from_source_indexes_every_record(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    assert len(store) == 5


def test_get_returns_the_record(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    assert store.get("CHEBI:17234").fields["label"] == "glucose"


def test_get_raises_keyerror_for_an_unknown_id(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    with pytest.raises(KeyError):
        store.get("CHEBI:99999")


def test_get_many_preserves_request_order(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    got = store.get_many(["CHEBI:17992", "CHEBI:17234"])
    assert [r.id for r in got] == ["CHEBI:17992", "CHEBI:17234"]


def test_get_many_skips_unknown_ids_without_raising(targets_csv):
    """A retriever backed by a stale index can name a since-deleted record."""
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    got = store.get_many(["CHEBI:17234", "GONE:1"])
    assert [r.id for r in got] == ["CHEBI:17234"]


def test_duplicate_ids_are_rejected():
    dupes = [Record(id="a", fields={"n": 1}), Record(id="a", fields={"n": 2})]
    with pytest.raises(ValueError, match="duplicate"):
        MemoryStore.from_source(dupes)


def test_fingerprint_is_order_independent():
    a = MemoryStore.from_source([Record(id="a", fields={}), Record(id="b", fields={})])
    b = MemoryStore.from_source([Record(id="b", fields={}), Record(id="a", fields={})])
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_content_changes():
    a = MemoryStore.from_source([Record(id="a", fields={"n": 1})])
    b = MemoryStore.from_source([Record(id="a", fields={"n": 2})])
    assert a.fingerprint != b.fingerprint


def test_iterating_the_store_yields_records(targets_csv):
    store = MemoryStore.from_source(csv_source(targets_csv, id_column="id"))
    assert {r.id for r in store} == {
        "CHEBI:17234",
        "CHEBI:28757",
        "CHEBI:17992",
        "CHEBI:17716",
        "CHEBI:15903",
    }


def test_memory_store_satisfies_the_target_store_protocol():
    from xwalk.stores.base import TargetStore

    store = MemoryStore.from_source([Record(id="a", fields={})])
    assert isinstance(store, TargetStore)

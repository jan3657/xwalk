import asyncio

from xwalk.records import Record, RetrievalHit
from xwalk.retrieval.base import RetrieverError, SearchRequest
from xwalk.retrieve import retrieve
from xwalk.stores.memory import MemoryStore

STORE = MemoryStore.from_source(
    [Record(id="T1", fields={"label": "glucose"}), Record(id="T2", fields={"label": "fructose"})]
)
SOURCE = Record(id="s1", fields={"mention": "glucose"})


class Scripted:
    def __init__(self, by_query, *, name="r", fail=False, hang=False):
        self._by_query, self._name, self._fail, self._hang = by_query, name, fail, hang
        self.queries: list[str] = []

    @property
    def name(self):
        return self._name

    @property
    def fingerprint(self):
        return "s"

    @property
    def default_limit(self):
        return 10

    async def search(self, request: SearchRequest):
        self.queries.append(request.text)
        if self._fail:
            raise RetrieverError("down")
        if self._hang:
            await asyncio.sleep(3600)
        return [
            RetrievalHit(record_id=rid, retriever=self._name, raw_score=1.0, rank=i)
            for i, rid in enumerate(self._by_query.get(request.text, []), start=1)
        ]


async def test_every_query_is_searched_and_fused():
    r = Scripted({"glucose": ["T1"], "dextrose": ["T2", "T1"]})
    candidates, notes, all_failed = await retrieve(
        ["glucose", "dextrose"], SOURCE, [r], STORE, timeout=1.0
    )
    assert r.queries == ["glucose", "dextrose"]
    assert [c.id for c in candidates] == ["T1", "T2"]
    assert notes == [] and all_failed is False


async def test_a_record_found_by_two_queries_outranks_one_found_once():
    r = Scripted({"a": ["T2"], "b": ["T2"], "c": ["T1"]})
    candidates, _, _ = await retrieve(["a", "b", "c"], SOURCE, [r], STORE, timeout=1.0)
    assert candidates[0].id == "T2"


async def test_a_failing_retriever_degrades_with_a_note():
    ok = Scripted({"q": ["T1"]}, name="ok")
    bad = Scripted({}, name="bad", fail=True)
    candidates, notes, all_failed = await retrieve(["q"], SOURCE, [ok, bad], STORE, timeout=1.0)
    assert [c.id for c in candidates] == ["T1"]
    assert notes == ["bad: down"] and all_failed is False


async def test_everything_failing_is_reported_as_such():
    bad = Scripted({}, fail=True)
    candidates, notes, all_failed = await retrieve(["q", "r"], SOURCE, [bad], STORE, timeout=1.0)
    assert candidates == [] and len(notes) == 2 and all_failed is True


async def test_a_hanging_retriever_times_out():
    slow = Scripted({}, name="slow", hang=True)
    _, notes, all_failed = await retrieve(["q"], SOURCE, [slow], STORE, timeout=0.05)
    assert notes == ["slow: timed out after 0.05s"] and all_failed is True

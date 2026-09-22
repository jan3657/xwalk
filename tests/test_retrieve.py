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


async def test_two_queries_both_vote_but_the_evidence_names_the_retriever_once():
    """The per-query fusion tag buys the second vote; it must not reach the trace."""
    r = Scripted({"a": ["T2", "T1"], "b": ["T1"]})
    candidates, _, _ = await retrieve(["a", "b"], SOURCE, [r], STORE, timeout=1.0)
    by_id = {c.id: c for c in candidates}
    assert by_id["T1"].fused_score == 1 / 62 + 1 / 61  # rank 2 on "a", rank 1 on "b"
    assert by_id["T1"].fused_score > by_id["T2"].fused_score
    assert [(h.retriever, h.rank) for h in by_id["T1"].evidence] == [("r", 1)]


async def test_multi_query_evidence_keeps_one_untagged_entry_per_retriever():
    one = Scripted({"a": ["T1"], "b": ["T1"]}, name="zeta")
    two = Scripted({"a": ["T1"], "b": ["T1"]}, name="alpha")
    candidates, _, _ = await retrieve(["a", "b"], SOURCE, [one, two], STORE, timeout=1.0)
    assert [h.retriever for h in candidates[0].evidence] == ["alpha", "zeta"]


async def test_a_single_query_leaves_the_evidence_untouched():
    r = Scripted({"q": ["T1"]})
    candidates, _, _ = await retrieve(["q"], SOURCE, [r], STORE, timeout=1.0)
    assert [(h.retriever, h.rank, h.raw_score) for h in candidates[0].evidence] == [("r", 1, 1.0)]

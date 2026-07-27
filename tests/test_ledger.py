import asyncio

import pytest

from tests.test_serde import sample_result
from xwalk.ledger import Ledger
from xwalk.records import MatchStatus


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "run.sqlite")
    yield led
    led.close()


async def test_put_then_get_round_trips(ledger):
    result = sample_result()
    await ledger.put_result(result)
    assert ledger.get_result(result.result_key) == result


def test_get_returns_none_for_an_unknown_key(ledger):
    assert ledger.get_result("nope") is None


async def test_has_result_is_true_after_a_put(ledger):
    result = sample_result()
    assert not ledger.has_result(result.result_key)
    await ledger.put_result(result)
    assert ledger.has_result(result.result_key)


async def test_putting_the_same_key_twice_replaces_rather_than_duplicates(ledger):
    result = sample_result()
    await ledger.put_result(result)
    await ledger.put_result(result)
    assert ledger.count(result.run_fingerprint) == 1


async def test_iter_results_is_scoped_to_a_run_fingerprint(ledger):
    from xwalk.records import MatchResult

    first = sample_result()
    second = MatchResult(**{**first.__dict__, "result_key": "rk2", "run_fingerprint": "fp2"})
    await ledger.put_result(first)
    await ledger.put_result(second)
    assert [r.result_key for r in ledger.iter_results("fp1")] == ["rk1"]
    assert ledger.count("fp2") == 1


async def test_iter_results_is_ordered_by_source_id(ledger):
    from xwalk.records import MatchResult

    first = sample_result()
    for sid in ["s3", "s1", "s2"]:
        await ledger.put_result(
            MatchResult(**{**first.__dict__, "result_key": f"rk-{sid}", "source_id": sid})
        )
    assert [r.source_id for r in ledger.iter_results("fp1")] == ["s1", "s2", "s3"]


async def test_status_is_queryable_without_deserialising_every_blob(ledger):
    await ledger.put_result(sample_result())
    assert ledger.count_by_status("fp1") == {MatchStatus.MATCHED: 1}


# --- cache ----------------------------------------------------------------------


async def test_cache_round_trips(ledger):
    await ledger.put_cached("ck1", '{"a": 1}')
    assert ledger.get_cached("ck1") == '{"a": 1}'


def test_cache_miss_returns_none(ledger):
    assert ledger.get_cached("nope") is None


# --- manifest -------------------------------------------------------------------


async def test_manifest_round_trips(ledger):
    ledger.put_manifest("fp1", {"model": "gpt-4o", "target": "t1"})
    assert ledger.get_manifest("fp1") == {"model": "gpt-4o", "target": "t1"}


def test_manifest_miss_returns_none(ledger):
    assert ledger.get_manifest("nope") is None


# --- durability -----------------------------------------------------------------


async def test_results_survive_reopening_the_file(tmp_path):
    path = tmp_path / "run.sqlite"
    led = Ledger.open(path)
    await led.put_result(sample_result())
    led.close()

    reopened = Ledger.open(path)
    assert reopened.has_result("rk1")
    reopened.close()


async def test_wal_mode_is_enabled(ledger):
    assert ledger.journal_mode.lower() == "wal"


async def test_concurrent_writes_all_land(ledger):
    from xwalk.records import MatchResult

    base = sample_result()
    results = [
        MatchResult(**{**base.__dict__, "result_key": f"rk{i}", "source_id": f"s{i}"})
        for i in range(50)
    ]
    await asyncio.gather(*(ledger.put_result(r) for r in results))
    assert ledger.count("fp1") == 50


async def test_a_crash_mid_run_leaves_completed_results_readable(tmp_path):
    """Simulate a kill: write some results, drop the object without closing, reopen."""
    from xwalk.records import MatchResult

    path = tmp_path / "run.sqlite"
    led = Ledger.open(path)
    base = sample_result()
    for i in range(5):
        await led.put_result(
            MatchResult(**{**base.__dict__, "result_key": f"rk{i}", "source_id": f"s{i}"})
        )
    del led  # no close(), no flush

    reopened = Ledger.open(path)
    assert reopened.count("fp1") == 5
    reopened.close()

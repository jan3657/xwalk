import pytest

from xwalk.evaluate.partition import (
    Partition,
    Partitioner,
    ids_in,
    load_partition_file,
    partition_of,
    write_partition_file,
)

IDS = [f"s{i}" for i in range(1000)]


def test_assignment_is_deterministic():
    p = Partitioner()
    assert [p.assign(i) for i in IDS] == [Partitioner().assign(i) for i in IDS]


def test_a_different_salt_produces_a_different_split():
    a = Partitioner(salt="one")
    b = Partitioner(salt="two")
    assert [a.assign(i) for i in IDS] != [b.assign(i) for i in IDS]


def test_every_id_lands_in_exactly_one_partition():
    split = Partitioner().split(IDS)
    assert sum(len(v) for v in split.values()) == len(IDS)
    assert len(set().union(*(set(v) for v in split.values()))) == len(IDS)


def test_fractions_are_approximately_honoured():
    split = Partitioner(fractions=(0.5, 0.25, 0.25)).split(IDS)
    assert 0.45 < len(split[Partition.PROMPT_TRAIN]) / 1000 < 0.55
    assert 0.20 < len(split[Partition.VALIDATION]) / 1000 < 0.30
    assert 0.20 < len(split[Partition.TEST]) / 1000 < 0.30


def test_custom_fractions_are_honoured():
    split = Partitioner(fractions=(0.8, 0.1, 0.1)).split(IDS)
    assert 0.75 < len(split[Partition.PROMPT_TRAIN]) / 1000 < 0.85


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        Partitioner(fractions=(0.5, 0.5, 0.5))


def test_a_zero_fraction_is_rejected():
    """An empty validation set silently disables round selection."""
    with pytest.raises(ValueError, match="positive"):
        Partitioner(fractions=(0.8, 0.2, 0.0))


def test_split_output_is_sorted_for_reproducible_reporting():
    split = Partitioner().split(["s3", "s1", "s2"] * 10)
    for ids in split.values():
        assert ids == sorted(set(ids))


def test_adding_records_does_not_move_existing_ones():
    """Growing the labelled set must not reshuffle the test partition."""
    p = Partitioner()
    before = {i: p.assign(i) for i in IDS[:500]}
    after = {i: p.assign(i) for i in IDS}
    assert all(after[i] is before[i] for i in before)


def test_a_partition_file_round_trips(tmp_path):
    assignment = {i: Partitioner().assign(i) for i in IDS[:20]}
    path = tmp_path / "partitions.csv"
    write_partition_file(assignment, path)
    assert load_partition_file(path) == assignment


def test_an_explicit_partition_file_overrides_hashing(tmp_path):
    path = tmp_path / "partitions.csv"
    path.write_text("source_id,partition\ns1,test\ns2,validation\n", encoding="utf-8")
    loaded = load_partition_file(path)
    assert loaded["s1"] is Partition.TEST
    assert partition_of("s1", Partitioner(), loaded) is Partition.TEST


def test_an_unknown_partition_name_is_rejected(tmp_path):
    path = tmp_path / "partitions.csv"
    path.write_text("source_id,partition\ns1,dev\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dev"):
        load_partition_file(path)


def test_partition_of_falls_back_to_hashing_without_an_override():
    p = Partitioner()
    assert partition_of("s1", p, {"other": Partition.TEST}) is p.assign("s1")


def test_ids_in_returns_only_that_partition_sorted():
    p = Partitioner()
    got = ids_in(IDS[:100], Partition.TEST, p)
    assert got == sorted(got)
    assert all(p.assign(i) is Partition.TEST for i in got)

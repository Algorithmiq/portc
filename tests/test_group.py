"""Test the device group: work splitting, collectives and failure handling."""

import threading

import numpy as np
import pytest

from portc._group import DeviceGroup, GroupAbortedError, blocks


@pytest.mark.parametrize("n", [0, 1, 7, 8, 2000])
@pytest.mark.parametrize("size", [1, 2, 3, 8])
def test_blocks_partition_in_rank_order(n, size):
    parts = blocks(n, size)

    assert len(parts) == size
    assert [i for lo, hi in parts for i in range(lo, hi)] == list(range(n))
    sizes = [hi - lo for lo, hi in parts]
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)


def test_a_group_of_one_runs_inline():
    group = DeviceGroup("cpu", [0])
    local = np.arange(6.0).reshape(3, 2)

    def fn(rank):
        assert threading.current_thread() is threading.main_thread()
        assert group.allgather(rank, local) is local
        assert group.gather(rank, local) is local
        assert group.scatter(rank, local) is local
        assert group.alltoall(rank, [local]) is local
        return rank

    assert group.run(fn) == [0]


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_collectives(size, dtype):
    group = DeviceGroup("cpu", size)
    # Uneven blocks: rank r holds r + 1 rows.
    parts = [
        (np.arange((r + 1) * 6) + 100 * r).reshape(r + 1, 2, 3) for r in range(size)
    ]
    parts = [p.astype(dtype) for p in parts]
    expected = np.concatenate(parts)

    def fn(rank):
        gathered = group.allgather(rank, parts[rank])
        on_root = group.gather(rank, parts[rank])
        mine = group.scatter(rank, expected if rank == 0 else None)
        # A second round reuses the slots of the first.
        again = group.allgather(rank, parts[rank])
        return gathered, on_root, mine, again

    results = group.run(fn)

    for rank, (gathered, on_root, mine, again) in enumerate(results):
        np.testing.assert_array_equal(gathered, expected)
        np.testing.assert_array_equal(again, expected)
        assert (on_root is not None) == (rank == 0)
        lo, hi = blocks(expected.shape[0], size)[rank]
        np.testing.assert_array_equal(mine, expected[lo:hi])
    np.testing.assert_array_equal(results[0][1], expected)


@pytest.mark.parametrize("size", [2, 3])
def test_alltoall_concatenates_what_each_rank_sent(size):
    group = DeviceGroup("cpu", size)
    # Rank r sends rank s an (r + 1, s + 2) block filled with 10 r + s.
    sends = [
        [np.full((r + 1, s + 2), 10.0 * r + s) for s in range(size)]
        for r in range(size)
    ]

    results = group.run(lambda rank: group.alltoall(rank, sends[rank]))

    for rank, received in enumerate(results):
        expected = np.concatenate([sends[r][rank] for r in range(size)])
        np.testing.assert_array_equal(received, expected)


def test_many_collectives_in_a_row():
    group = DeviceGroup("cpu", [0, 1, 2])

    def fn(rank):
        total = np.zeros(3)
        for step in range(200):
            total += group.allgather(rank, np.array([rank + step], dtype=float))
        return total

    for total in group.run(fn):
        np.testing.assert_array_equal(total, [19900, 20100, 20300])


def test_a_failing_rank_releases_the_others():
    group = DeviceGroup("cpu", [0, 1, 2])

    def fn(rank):
        if rank == 1:
            msg = "boom on rank 1"
            raise ValueError(msg)
        # Never completes: rank 1 does not join.
        return group.allgather(rank, np.zeros(1))

    with pytest.raises(ValueError, match="boom on rank 1"):
        group.run(fn)
    assert group.aborted.is_set()


def test_abort_wakes_registered_waiters():
    group = DeviceGroup("cpu", [0, 1])
    released = threading.Event()
    group.on_abort(released.set)

    def fn(rank):
        if rank == 0:
            msg = "boom"
            raise RuntimeError(msg)
        released.wait()
        raise GroupAbortedError

    with pytest.raises(RuntimeError, match="boom"):
        group.run(fn)

"""Test the environment store on every tier, on the host backend."""

import errno

import numpy as np
import pytest

import portc._store as store_module
from portc._plan import Plan, SitePlan
from portc._store import EnvironmentStore
from portc.utils import NullStream

CHI = 10
SHAPES = [(CHI, 1), (CHI, 3, 2), (CHI, 4, 2), (CHI, 2, 1)]


def make_plan(tier, *, env=4, sketch=3):
    """Environments of sites 1-3 on ``tier``, written in batches of ``env``."""
    sites = [SitePlan(env, 0, 5, "device")]
    sites += [SitePlan(env, sketch, 5, tier) for _ in range(2)]
    sites += [SitePlan(0, sketch, 5, tier)]
    return Plan(tuple(sites), 1, 0, 0, 0)


def ranges(n, batch):
    return [(lo, min(lo + batch, n)) for lo in range(0, n, batch)]


def fill(store, rng, batch=4):
    """Put random environments for sites 1-3 and return them."""
    envs = {}
    for j in (1, 2, 3):
        envs[j] = rng.normal(size=SHAPES[j]) + 1j * rng.normal(size=SHAPES[j])
        for lo, hi in ranges(CHI, batch):
            store.put(j, lo, hi, envs[j][lo:hi])
    return envs


def open_store(tier, tmp_path, **batches):
    return EnvironmentStore(
        make_plan(tier, **batches),
        SHAPES,
        np.complex128,
        tmp_path,
        copy_stream=NullStream(),
    )


@pytest.mark.parametrize("tier", ["device", "host", "disk"])
def test_round_trip_in_other_batches(tier, tmp_path):
    rng = np.random.default_rng(0)
    with open_store(tier, tmp_path) as store:
        envs = fill(store, rng)
        for j in (3, 2, 1):
            batches = ranges(CHI, 3)  # read in batches of 3, written in batches of 4
            store.prefetch(j, batches)
            got = np.concatenate([store.get(j, lo, hi) for lo, hi in batches])
            np.testing.assert_array_equal(got, envs[j])
            store.drop(j)


def test_get_without_prefetch(tmp_path):
    rng = np.random.default_rng(1)
    with open_store("disk", tmp_path) as store:
        envs = fill(store, rng)

        np.testing.assert_array_equal(store.get(2, 7, 10), envs[2][7:10])


def test_get_out_of_order_raises(tmp_path):
    rng = np.random.default_rng(2)
    with open_store("host", tmp_path) as store:
        fill(store, rng)
        store.prefetch(1, [(0, 3), (3, 6)])

        with pytest.raises(RuntimeError, match="out of order"):
            store.get(1, 3, 6)


def test_disk_tier_files_are_removed(tmp_path):
    rng = np.random.default_rng(3)
    with open_store("disk", tmp_path) as store:
        fill(store, rng)
        (scratch,) = tmp_path.iterdir()
        store.prefetch(3, [(0, CHI)])
        store.get(3, 0, CHI)
        store.drop(3)
        assert sorted(p.name for p in scratch.iterdir()) == [
            "env-0001.bin",
            "env-0002.bin",
        ]
    assert list(tmp_path.iterdir()) == []


def fill_then_fail(tmp_path):
    with open_store("disk", tmp_path) as store:
        fill(store, np.random.default_rng(4))
        raise KeyError


def test_scratch_is_removed_after_an_error(tmp_path):
    with pytest.raises(KeyError):
        fill_then_fail(tmp_path)

    assert list(tmp_path.iterdir()) == []


def fill_then_read(tmp_path):
    with open_store("disk", tmp_path) as store:
        fill(store, np.random.default_rng(5))
        store.get(1, 0, 3)


def test_writer_errors_are_raised(monkeypatch, tmp_path):
    def full_disk(*_args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(store_module, "_write_all", full_disk)

    with pytest.raises(OSError, match=r"env-0001\.bin failed \(No space left"):
        fill_then_read(tmp_path)

    assert list(tmp_path.iterdir()) == []

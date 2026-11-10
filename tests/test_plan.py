"""Test the memory planner: sizes, budgets, batches and tiers."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import portc._plan as plan_module
from portc import Resources
from portc._group import DeviceGroup
from portc._plan import (
    GEMM_MULTIPLE,
    Budgets,
    make_plan,
    parse_size,
    resolve_budgets,
)

GB = 10**9
HOST = DeviceGroup("cpu")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        ("36GB", 36 * 10**9),
        ("36GiB", 36 * 2**30),
        ("1.5 kB", 1500),
        ("512", 512),
        ("2mib", 2 * 2**20),
    ],
)
def test_parse_size(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", [-1, "36 GBs", "lots", ""])
def test_parse_size_rejects_values(value):
    with pytest.raises(ValueError, match="Expected"):
        parse_size(value)


@pytest.mark.parametrize("value", [True, 1.5, None])
def test_parse_size_rejects_types(value):
    with pytest.raises(TypeError, match="Expected"):
        parse_size(value)


def test_resources_validate_at_construction():
    with pytest.raises(ValueError, match="size string"):
        Resources(gpu_memory="plenty")


def test_resolve_budgets_explicit_on_host(tmp_path):
    budgets = resolve_budgets(Resources(host_memory="2GB", scratch_dir=tmp_path), HOST)

    assert budgets.unified
    assert budgets.host == budgets.device == 2 * GB
    assert budgets.scratch_dir == tmp_path
    assert budgets.disk > 0


def test_resolve_budgets_detects_host_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(plan_module, "host_memory_available", lambda: 10 * GB)

    budgets = resolve_budgets(Resources(scratch_dir=tmp_path / "not" / "yet"), HOST)

    assert budgets.host == 9 * GB


@pytest.fixture
def fake_gpus(monkeypatch):
    """Fake devices for the planner; returns a dict of ``id -> (free, total)``."""
    memory = {0: (30 * GB, 40 * GB)}
    current = [0]

    @contextmanager
    def use_device(_xp, device):
        previous, current[0] = current[0], device
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(plan_module, "is_host", lambda _xp: False)
    monkeypatch.setattr(plan_module, "use_device", use_device)
    monkeypatch.setattr(plan_module, "device_memory", lambda _xp: memory[current[0]])
    return memory


@pytest.mark.usefixtures("fake_gpus")
def test_resolve_budgets_detects_device_memory(tmp_path):
    budgets = resolve_budgets(Resources(host_memory="1GB", scratch_dir=tmp_path), HOST)

    assert not budgets.unified
    assert budgets.device == 26 * GB  # minus max(10% of 40 GB, 1 GiB)
    assert budgets.device_cap == 30 * GB  # the pool may grow into the margin


@pytest.mark.usefixtures("fake_gpus")
def test_resolve_budgets_caps_explicit_device_memory_with_margin(tmp_path):
    budgets = resolve_budgets(
        Resources(gpu_memory="10GB", host_memory="1GB", scratch_dir=tmp_path), HOST
    )

    assert budgets.device == 10 * GB
    assert budgets.device_cap == 14 * GB


def test_resolve_budgets_takes_the_smallest_device(fake_gpus, tmp_path):
    fake_gpus[1] = (20 * GB, 40 * GB)

    budgets = resolve_budgets(
        Resources(host_memory="1GB", scratch_dir=tmp_path), DeviceGroup("cpu", 2)
    )

    assert budgets.device == 16 * GB
    assert budgets.device_cap == 20 * GB


def test_resolve_budgets_splits_the_host_among_simulated_devices(tmp_path):
    budgets = resolve_budgets(
        Resources(host_memory="8GB", scratch_dir=tmp_path), DeviceGroup("cpu", 4)
    )

    assert budgets.host == 8 * GB
    assert budgets.device == 2 * GB


def test_resolve_budgets_leaves_the_host_uncapped(tmp_path):
    budgets = resolve_budgets(Resources(host_memory="1GB", scratch_dir=tmp_path), HOST)

    assert budgets.device_cap is None


def test_resolve_budgets_detects_disk(monkeypatch, tmp_path):
    usage = type("Usage", (), {"free": 100 * GB})
    monkeypatch.setattr(plan_module.shutil, "disk_usage", lambda _path: usage)

    budgets = resolve_budgets(Resources(host_memory="1GB", scratch_dir=tmp_path), HOST)

    assert budgets.disk == 95 * GB


def mpo_stack_shapes(n_sites, bonds, phys=4):
    """Padded shapes of a stack of MPOs, one bond dimension per layer."""

    def shape(j, bond):
        return (1 if j == 0 else bond, 1 if j == n_sites - 1 else bond, phys, phys)

    return [tuple(shape(j, bond) for bond in bonds) for j in range(n_sites)]


def site_bytes(shapes, itemsize=16):
    return [sum(int(np.prod(s)) * itemsize for s in site) for site in shapes]


def budgets(device, host=None, disk=10**15, *, unified=False):
    return Budgets(
        device, device if host is None else host, disk, Path("/scratch"), unified
    )


def test_small_problem_is_one_batch_on_the_device():
    shapes = mpo_stack_shapes(6, [2, 3, 2])

    plan = make_plan(shapes, site_bytes(shapes), 64, np.complex128, budgets(GB))

    assert plan.prefetch == 1
    assert all(site.tier == "device" for site in plan.sites)
    assert plan.sites[0].env_batch == 64
    assert plan.sites[-1].env_batch == 0
    assert plan.sites[0].sketch_batch == 0
    assert all(site.sketch_batch == 64 for site in plan.sites[1:])
    assert all(site.project_batch == 64 for site in plan.sites)
    assert plan.disk_bytes == 0


def test_tight_budget_shrinks_batches_to_gemm_multiples():
    shapes = mpo_stack_shapes(6, [4, 4, 64, 4])
    chi = 512
    loose = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))

    tight = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(loose.device_peak // 4)
    )

    batches = [s.sketch_batch for s in tight.sites[1:]]
    assert min(batches) < chi
    assert all(b % GEMM_MULTIPLE == 0 for b in batches if b >= GEMM_MULTIPLE)
    assert tight.device_peak <= loose.device_peak // 4


def test_tiers_go_newest_first():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    env = chi * 4 * 4 * 64 * 4 * 16  # one bulk environment, complex128
    roomy = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))

    seen = set()
    for extra in range(8):
        plan = make_plan(
            shapes,
            site_bytes(shapes),
            chi,
            np.complex128,
            budgets(roomy.device_peak - extra * env, host=roomy.host_peak + 7 * env),
        )
        tiers = [site.tier for site in plan.sites[1:]]
        # Oldest sites on the slowest tier: disk, then host, then device.
        assert tiers == sorted(tiers, key=["disk", "host", "device"].index)
        assert plan.disk_bytes == env * tiers.count("disk")
        seen.update(tiers)
    assert seen == {"device", "host", "disk"}


def test_unified_memory_has_no_host_tier():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    roomy = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB, unified=True)
    )

    plan = make_plan(
        shapes,
        site_bytes(shapes),
        chi,
        np.complex128,
        budgets(roomy.device_peak // 2, unified=True),
    )

    tiers = {site.tier for site in plan.sites[1:]}
    assert "host" not in tiers
    assert "disk" in tiers


def test_infeasible_site_names_the_site():
    shapes = mpo_stack_shapes(6, [4, 4, 64, 4])

    with pytest.raises(MemoryError, match=r"Site \d+: the \w+ step needs \d+ bytes"):
        make_plan(shapes, site_bytes(shapes), 256, np.complex128, budgets(10**6))


def test_environments_must_fit_on_disk():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    roomy = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))
    work = roomy.device_peak - 7 * chi * 4 * 4 * 64 * 4 * 16

    with pytest.raises(MemoryError, match="bytes on disk"):
        make_plan(
            shapes,
            site_bytes(shapes),
            chi,
            np.complex128,
            budgets(work, host=roomy.host_peak, disk=1000),
        )


def test_prefetch_is_dropped_before_giving_up():
    shapes = mpo_stack_shapes(6, [4, 4, 64, 4])
    chi = 64
    cores = max(site_bytes(shapes))
    with_prefetch = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB)
    )
    # Remove about one site of cores from the smallest budget that fits one column.
    minimum = with_prefetch.device_peak
    lo, hi = 0, minimum
    while lo < hi:  # the smallest budget that plans with prefetching
        mid = (lo + hi) // 2
        try:
            make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(mid))
            hi = mid
        except MemoryError:
            lo = mid + 1

    plan = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(lo))
    assert plan.prefetch in {0, 1}
    if plan.prefetch == 1:
        dropped = make_plan(
            shapes, site_bytes(shapes), chi, np.complex128, budgets(lo - cores // 2)
        )
        assert dropped.prefetch == 0


def test_devices_split_the_environments_and_the_batches():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    env = chi * 4 * 4 * 64 * 4 * 16  # one bulk environment on one device
    one = make_plan(shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB))
    # A device budget that holds half of one device's environments on four devices
    # holds all of them.
    tight = one.device_peak - 3 * env

    single = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(tight, host=10 * GB)
    )
    four = make_plan(
        shapes,
        site_bytes(shapes),
        chi,
        np.complex128,
        budgets(tight, host=10 * GB),
        devices=4,
    )

    assert "host" in {s.tier for s in single.sites}
    assert {s.tier for s in four.sites} == {"device"}
    for site in four.sites[1:]:
        assert site.sketch_batch <= chi // 4
        assert site.project_batch <= chi // 4
    assert four.sites[0].env_batch <= chi // 4
    # The first site runs on rank 0 alone, on every row.
    assert four.sites[0].project_batch == chi


def test_devices_share_host_memory_and_disk():
    shapes = mpo_stack_shapes(8, [4, 4, 64, 4])
    chi = 256
    env = chi * 4 * 4 * 64 * 4 * 16 // 2  # one bulk environment on each of 2 devices
    roomy = make_plan(
        shapes, site_bytes(shapes), chi, np.complex128, budgets(10 * GB), devices=2
    )
    device = roomy.device_peak - 6 * env
    # Each device stages four batches of 128 columns, one environment each.
    staging = 4 * env
    host = roomy.host_peak + 2 * staging + 4 * env

    plan = make_plan(
        shapes,
        site_bytes(shapes),
        chi,
        np.complex128,
        budgets(device, host=host),
        devices=2,
    )

    tiers = [s.tier for s in plan.sites[1:]]
    # Host room for four environments in all: two per device.
    assert tiers.count("host") == 2
    assert plan.disk_bytes == 2 * env * tiers.count("disk")
    assert plan.host_peak <= host

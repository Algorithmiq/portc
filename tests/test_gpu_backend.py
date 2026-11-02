"""GPU (cupy) backend tests.

Skipped automatically when cupy or a CUDA/ROCm device is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
import quimb.tensor as qtn

import src_method._sweep as sweep_module
from src_method import Resources, apply, compress, src
from src_method._group import DeviceGroup
from src_method._kernels import SiteKernels
from src_method._plan import (
    GPU_MARGIN_FRACTION,
    GPU_MARGIN_MIN,
    Budgets,
    Plan,
    make_plan,
)
from src_method._sites import padded_shapes, site_bytes
from src_method.utils import thin_svd
from src_method.utils.linalg import DEVICE_SVD_MIN

cupy = pytest.importorskip("cupy")


def as_mps(arrays: list[np.ndarray]) -> qtn.MatrixProductState:
    """Wrap a list of site arrays returned by src_method into a quimb MPS."""
    return qtn.MatrixProductState(arrays)


def as_mpo(arrays: list[np.ndarray]) -> qtn.MatrixProductOperator:
    """Wrap a list of site arrays returned by src_method into a quimb MPO."""
    return qtn.MatrixProductOperator(arrays)


@pytest.fixture(autouse=True)
def _require_device() -> None:
    """Skip the whole module when no GPU runtime is reachable."""
    try:
        cupy.cuda.runtime.getDeviceCount()
    except (cupy.cuda.runtime.CUDARuntimeError, RuntimeError) as err:
        pytest.skip(f"No usable GPU device: {err}")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_apply_mpo_mps_gpu_matches_cpu(device: str) -> None:
    """``apply`` MPO @ MPS should produce equivalent compressed states on both devices."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H = qtn.MPO_rand(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128
    )

    out = as_mps(
        apply(
            H.arrays,
            psi.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=0,
            device=device,
        )
    )
    ref = H.apply(psi, compress=False)

    np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_apply_mpo_mpo_gpu_matches_cpu(device: str) -> None:
    """``apply`` MPO @ MPO should produce equivalent compressed MPOs on both devices."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=np.complex128
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128
    )

    out = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=0,
            device=device,
        )
    )
    ref = H1.apply(H2, compress=False)

    np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_compress_mpo_gpu_matches_cpu(device: str) -> None:
    """``compress`` on an MPO should match the CPU result on GPU."""
    n_sites, phys_dim, chi_out = 5, 2, 8
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)
    B = qtn.MPO_rand(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=np.complex128) / 1e8
    C = A + B

    out = as_mpo(
        compress(C.arrays, chi_out=chi_out, dtype=np.complex128, seed=0, device=device)
    )

    np.testing.assert_allclose(C.distance(out), 0.0, atol=1e-6)


def test_apply_mpo_mpo_cpu_gpu_equivalent() -> None:
    """CPU and GPU ``apply`` MPO @ MPO must produce equivalent MPOs."""
    n_sites, phys_dim, chi_out = 8, 4, 32
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)
    H2 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)

    cpu_out = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=42,
            device="cpu",
        )
    )
    gpu_out = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=42,
            device="gpu",
        )
    )

    # Gauge freedom means individual tensors can differ; compare the contracted MPOs.
    # atol=1e-6 accounts for floating-point accumulation across different execution orders.
    np.testing.assert_allclose(cpu_out.distance(gpu_out), 0.0, atol=1e-6)


def test_compress_mpo_cpu_gpu_equivalent() -> None:
    """CPU and GPU ``compress`` must produce equivalent MPOs."""
    n_sites, phys_dim, chi_out = 8, 4, 32
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=np.complex128)

    cpu_out = as_mpo(
        compress(A.arrays, chi_out=chi_out, dtype=np.complex128, seed=42, device="cpu")
    )
    gpu_out = as_mpo(
        compress(A.arrays, chi_out=chi_out, dtype=np.complex128, seed=42, device="gpu")
    )

    np.testing.assert_allclose(cpu_out.distance(gpu_out), 0.0, atol=1e-6)


def test_invalid_device_raises() -> None:
    """An unrecognised ``device`` must raise ``ValueError``."""
    H = qtn.MPO_rand(5, bond_dim=4, phys_dim=2, dtype=np.complex128)
    psi = qtn.MPS_rand_state(5, bond_dim=4, phys_dim=2, dtype=np.complex128)
    with pytest.raises(ValueError, match="Unknown device"):
        apply(H.arrays, psi.arrays, chi_out=4, device="tpu")


@pytest.mark.parametrize("oversample", [None, 1.0, 2.0])
@pytest.mark.parametrize("kind", ["mps", "mpo"])
def test_oversampled_truncation_matches_cpu(
    oversample: float | None, kind: str
) -> None:
    """The final truncation on the GPU agrees with the CPU one."""
    make_train = qtn.MPS_rand_state if kind == "mps" else qtn.MPO_rand
    train = make_train(5, bond_dim=4, dtype=np.complex128, seed=0)
    cpu = compress(train.arrays, 2, oversample=oversample, cutoff=0.1, seed=1)
    gpu = compress(
        train.arrays,
        2,
        oversample=oversample,
        cutoff=0.1,
        seed=1,
        device="gpu",
        resources=Resources(devices=[0]),
    )

    assert all(isinstance(site, np.ndarray) for site in gpu)
    np.testing.assert_allclose(
        type(train)(cpu).to_dense(), type(train)(gpu).to_dense(), atol=1e-10
    )


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_src_stack_gpu_matches_reference(device: str) -> None:
    """A depth-3 ``src`` stack should match the exact product on both devices."""
    n_sites, phys_dim, chi_out = 5, 2, 16
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=2, phys_dim=phys_dim, dtype=np.complex128, seed=1
    )
    H2 = qtn.MPO_rand(
        n_sites, bond_dim=2, phys_dim=phys_dim, dtype=np.complex128, seed=2
    )
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=3
    )

    out = as_mps(
        src(
            H1.arrays,
            H2.arrays,
            psi.arrays,
            chi_out=chi_out,
            dtype=np.complex128,
            seed=0,
            device=device,
        )
    )
    ref = H1.apply(H2.apply(psi, compress=False), compress=False)

    np.testing.assert_allclose(ref.distance(out), 0.0, atol=1e-6)


# ---------------------------------------
# --- Budgets, batching and spilling ---
# ---------------------------------------


def random_mpo_arrays(bonds: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    """Complex Gaussian MPO with the given bonds and physical legs of 2."""

    def site(*shape: int) -> np.ndarray:
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    lefts, rights = [None, *bonds], [*bonds, None]
    return [
        site(*(b for b in (lb, rb) if b is not None), 2, 2)
        for lb, rb in zip(lefts, rights)
    ]


def dense_mpo(train: list[np.ndarray]) -> np.ndarray:
    """Contract an MPO into a ``(U, D)`` matrix."""
    T = train[0]
    for W in train[1:-1]:
        T = np.einsum("aUD,abud->bUuDd", T, W)
        T = T.reshape(W.shape[1], T.shape[1] * T.shape[2], T.shape[3] * T.shape[4])
    T = np.einsum("aUD,aud->UuDd", T, train[-1])
    return T.reshape(T.shape[0] * T.shape[1], T.shape[2] * T.shape[3])


@pytest.fixture
def plans(monkeypatch: pytest.MonkeyPatch) -> list[Plan]:
    """Record the plan of every sweep."""
    recorded: list[Plan] = []

    def spy(*args: Any, **kwargs: Any) -> Plan:
        recorded.append(make_plan(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(sweep_module, "make_plan", spy)
    return recorded


def test_gpu_tiers_and_batches_match_cpu(tmp_path, plans: list[Plan]) -> None:
    """Every tier and small batches on the GPU give the operator of the CPU run."""
    rng = np.random.default_rng(7)
    stack = [random_mpo_arrays([4, 8, 8, 8, 4], rng) for _ in range(4)]
    chi = 64
    shapes, sizes = padded_shapes(stack), site_bytes(stack)
    roomy = make_plan(
        shapes,
        sizes,
        chi,
        np.complex128,
        Budgets(10**10, 10**10, 10**12, tmp_path, unified=False),
    )
    env = chi * 8**4 * 16  # one bulk environment
    tight = Resources(
        gpu_memory=roomy.device_peak // 2,
        host_memory=roomy.host_peak + 5 * env,
        scratch_dir=tmp_path,
    )

    cpu = src(*stack, chi_out=chi, seed=3, dtype=np.complex128)
    gpu = src(
        *stack, chi_out=chi, seed=3, dtype=np.complex128, device="gpu", resources=tight
    )

    assert {site.tier for site in plans[-1].sites} == {"device", "host", "disk"}
    assert min(site.sketch_batch for site in plans[-1].sites[1:]) < chi
    reference = dense_mpo(cpu)
    error = np.linalg.norm(dense_mpo(gpu) - reference) / np.linalg.norm(reference)
    assert error < 1e-10
    assert list(tmp_path.iterdir()) == []


def test_gpu_pool_limit_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap on cupy's pool is lifted after the sweep, also after an error."""
    rng = np.random.default_rng(8)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    pool = cupy.get_default_memory_pool()
    previous = pool.get_limit()

    src(*stack, chi_out=4, seed=0, device="gpu")
    assert pool.get_limit() == previous

    def boom(*_args: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(SiteKernels, "sketch", boom)
    with pytest.raises(RuntimeError, match="boom"):
        src(*stack, chi_out=4, seed=0, device="gpu")
    assert pool.get_limit() == previous


def test_gpu_pool_cap_leaves_room_for_fragmentation(tmp_path) -> None:
    """The pool is capped at the GPU budget plus the margin, not at the budget."""
    rng = np.random.default_rng(9)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    pool = cupy.get_default_memory_pool()
    limits: list[int] = []

    def spy(*_args: object) -> None:
        limits.append(pool.get_limit())
        msg = "spy"
        raise RuntimeError(msg)

    budget = 10**8
    used = pool.used_bytes()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SiteKernels, "env", spy)
        with pytest.raises(RuntimeError, match="spy"):
            src(
                *stack,
                chi_out=4,
                seed=0,
                device="gpu",
                resources=Resources(gpu_memory=budget, scratch_dir=tmp_path),
            )

    _, total = cupy.cuda.runtime.memGetInfo()
    margin = max(int(GPU_MARGIN_FRACTION * total), GPU_MARGIN_MIN)
    assert limits == [used + budget + margin]


# --------------------
# --- Several GPUs ---
# --------------------


@pytest.fixture
def two_gpus() -> None:
    """Skip unless at least two GPUs are visible."""
    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Needs at least two GPUs.")


@pytest.mark.usefixtures("two_gpus")
@pytest.mark.parametrize("uploads", ["split", "full"])
def test_two_gpus_match_one(
    tmp_path, plans: list[Plan], monkeypatch: pytest.MonkeyPatch, uploads: str
) -> None:
    """Two GPUs, with or without split uploads, give the operator of one.

    An odd ``chi`` gives uneven blocks, and the budgets every tier.
    """
    rng = np.random.default_rng(7)
    stack = [random_mpo_arrays([4, 8, 8, 8, 4], rng) for _ in range(4)]
    chi = 63
    if uploads == "full":
        monkeypatch.setattr(
            "src_method._group.enable_peer_access", lambda _xp, _devices: False
        )
    shapes, sizes = padded_shapes(stack), site_bytes(stack)
    roomy = make_plan(
        shapes,
        sizes,
        chi,
        np.complex128,
        Budgets(10**10, 10**10, 10**12, tmp_path, unified=False),
        devices=2,
    )
    env = (chi // 2 + 1) * 8**4 * 16  # one bulk environment on one GPU
    # Checked with make_plan: one environment per GPU on the device, two in host
    # memory, the oldest two on disk.
    tight = Resources(
        gpu_memory=roomy.device_peak - 2 * env,
        host_memory=roomy.host_peak + 12 * env,
        scratch_dir=tmp_path,
        devices=2,
    )

    one = src(*stack, chi_out=chi, seed=3, dtype=np.complex128, device="gpu")
    two = src(
        *stack, chi_out=chi, seed=3, dtype=np.complex128, device="gpu", resources=tight
    )

    assert {site.tier for site in plans[-1].sites} == {"device", "host", "disk"}
    reference = dense_mpo(one)
    error = np.linalg.norm(dense_mpo(two) - reference) / np.linalg.norm(reference)
    assert error < 1e-10
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("two_gpus")
@pytest.mark.parametrize("cutoff", [0.0, 1e-3])
def test_two_gpus_factor_by_cholesky_qr(
    cutoff: float, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The distributed CholeskyQR on two GPUs gives the operator of one GPU."""
    rng = np.random.default_rng(11)
    stack = [random_mpo_arrays([4, 8, 8, 8, 8, 4], rng) for _ in range(3)]
    one = src(*stack, chi_out=40, cutoff=cutoff, seed=2, device="gpu")

    monkeypatch.setattr(sweep_module, "DISTRIBUTED_QR_MIN", 0)
    with caplog.at_level(logging.DEBUG, logger="src_method"):
        two = src(
            *stack,
            chi_out=40,
            cutoff=cutoff,
            seed=2,
            device="gpu",
            resources=Resources(devices=2),
        )

    assert "CholeskyQR at" in caplog.text
    assert [t.shape for t in two] == [t.shape for t in one]
    reference = dense_mpo(one)
    error = np.linalg.norm(dense_mpo(two) - reference) / np.linalg.norm(reference)
    assert error < 1e-10
    for core in two[1:]:
        matrix = core.reshape(core.shape[0], -1)
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(core.shape[0]), atol=1e-13
        )


@pytest.mark.usefixtures("two_gpus")
def test_two_gpus_take_device_inputs() -> None:
    """Cores given on one GPU reach the other through the staging buffer."""
    rng = np.random.default_rng(10)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    with cupy.cuda.Device(1):
        on_device = [[cupy.asarray(t) for t in train] for train in stack]

    host = src(*stack, chi_out=8, seed=0, device="gpu")
    both = src(
        *on_device, chi_out=8, seed=0, device="gpu", resources=Resources(devices=2)
    )

    reference = dense_mpo(host)
    assert (
        np.linalg.norm(dense_mpo(both) - reference) / np.linalg.norm(reference) < 1e-10
    )


@pytest.mark.usefixtures("two_gpus")
@pytest.mark.usefixtures("two_gpus")
def test_one_gpu_takes_inputs_from_another() -> None:
    """Cores given on GPU 1 are staged to GPU 0 when the sweep runs there alone."""
    rng = np.random.default_rng(11)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    with cupy.cuda.Device(1):
        on_device = [[cupy.asarray(t) for t in train] for train in stack]

    host = src(*stack, chi_out=8, seed=0, device="gpu")
    other = src(
        *on_device, chi_out=8, seed=0, device="gpu", resources=Resources(devices=[0])
    )

    reference = dense_mpo(host)
    assert (
        np.linalg.norm(dense_mpo(other) - reference) / np.linalg.norm(reference) < 1e-10
    )


@pytest.mark.usefixtures("two_gpus")
def test_two_gpus_raise_the_first_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error on one GPU stops both, restores the pool limits and is raised."""
    rng = np.random.default_rng(11)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]
    limits = []
    for d in range(2):
        with cupy.cuda.Device(d):
            limits.append(cupy.get_default_memory_pool().get_limit())

    def boom(self: SiteKernels, *args: Any) -> Any:
        if cupy.cuda.runtime.getDevice() == 1:
            msg = "boom on GPU 1"
            raise RuntimeError(msg)
        return original(self, *args)

    original = SiteKernels.sketch
    monkeypatch.setattr(SiteKernels, "sketch", boom)
    with pytest.raises(RuntimeError, match="boom on GPU 1"):
        src(*stack, chi_out=4, seed=0, device="gpu", resources=Resources(devices=2))
    for d in range(2):
        with cupy.cuda.Device(d):
            assert cupy.get_default_memory_pool().get_limit() == limits[d]


@pytest.mark.usefixtures("two_gpus")
def test_too_many_devices_raise() -> None:
    count = cupy.cuda.runtime.getDeviceCount()
    rng = np.random.default_rng(12)
    stack = [random_mpo_arrays([2, 3, 3, 2], rng) for _ in range(2)]

    with pytest.raises(ValueError, match="visible"):
        src(*stack, chi_out=4, device="gpu", resources=Resources(devices=count + 1))


def test_group_runs_on_a_non_blocking_stream() -> None:
    """Ranks leave the null stream, keep one stream per device, and see prior work."""
    group = DeviceGroup("gpu", [0])
    data = cupy.zeros(1 << 24, dtype=cupy.float32)
    data += 1  # queued on the null stream before the run

    def fn(_rank: int) -> tuple[int, float]:
        stream = cupy.cuda.get_current_stream()
        assert stream != cupy.cuda.Stream.null
        return stream.ptr, float((data * 2).sum())

    ((first, total),) = group.run(fn)
    ((second, _),) = group.run(fn)

    assert total == 2 * data.size
    assert first == second
    assert cupy.cuda.get_current_stream() == cupy.cuda.Stream.null


@pytest.mark.usefixtures("two_gpus")
def test_two_gpus_run_on_their_own_streams() -> None:
    """Each rank of a group runs on a non-blocking stream of its own device."""
    group = DeviceGroup("gpu", 2)

    def fn(rank: int) -> tuple[int, int]:
        stream = cupy.cuda.get_current_stream()
        assert stream != cupy.cuda.Stream.null
        return rank, stream.device_id

    assert group.run(fn) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("n", [DEVICE_SVD_MIN // 2, DEVICE_SVD_MIN + 8])
def test_thin_svd_matches_host_on_either_side_of_the_threshold(n: int) -> None:
    """Below the threshold the SVD runs on the host, above it on the device."""
    rng = np.random.default_rng(3)
    host = rng.standard_normal((n + 5, n)) + 1j * rng.standard_normal((n + 5, n))
    U, S, Vh = thin_svd(cupy.asarray(host))
    assert isinstance(U, cupy.ndarray)
    assert isinstance(Vh, cupy.ndarray)
    assert isinstance(S, np.ndarray)
    np.testing.assert_allclose(S, np.linalg.svd(host, compute_uv=False), rtol=1e-10)
    rebuilt = cupy.asnumpy((U * cupy.asarray(S)) @ Vh)
    np.testing.assert_allclose(rebuilt, host, atol=1e-10 * np.abs(host).max())

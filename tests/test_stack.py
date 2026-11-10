"""Test `src` over stacks of tensor trains, and the sweep behind it."""

from __future__ import annotations

import logging
import threading
from functools import reduce

import numpy as np
import opt_einsum as oe
import pytest

import portc._sweep as sweep_module
import portc.stack as stack_module
from portc import Resources, apply, compress, src
from portc._kernels import SiteKernels
from portc._plan import make_plan
from portc._sweep import sweep
from portc._tensor_train import pad

# -------------
# --- Utils ---
# -------------


def random_mpo(bonds, rng, *, up=2, down=2):
    """Complex MPO with the given (possibly jagged) bonds, ``len(bonds) + 1`` sites."""

    def site(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    lefts, rights = [None, *bonds], [*bonds, None]
    return [
        site(*(b for b in (lb, rb) if b is not None), up, down)
        for lb, rb in zip(lefts, rights)
    ]


def random_mps(bonds, rng, *, phys=2):
    """Complex MPS with the given (possibly jagged) bonds."""
    return [t[..., 0] for t in random_mpo(bonds, rng, up=phys, down=1)]


def identity_mpo(n_sites, phys=2):
    eye = np.eye(phys, dtype=complex)
    return [eye[None]] + [eye[None, None]] * (n_sites - 2) + [eye[None]]


def dense(train):
    """Contract a train into a ``(U, D)`` matrix; an MPS gives a column ``(U, 1)``."""
    if train[0].ndim == 2:
        train = [t[..., None] for t in train]
    T = train[0]
    for W in train[1:-1]:
        T = np.einsum("aUD,abud->bUuDd", T, W)
        T = T.reshape(W.shape[1], T.shape[1] * T.shape[2], T.shape[3] * T.shape[4])
    T = np.einsum("aUD,aud->UuDd", T, train[-1])
    return T.reshape(T.shape[0] * T.shape[1], T.shape[2] * T.shape[3])


def dense_stack(*trains):
    """Dense product of a stack; a leading MPS is a row vector, not conjugated."""
    mats = [dense(t) for t in trains]
    if len(trains) > 1 and trains[0][0].ndim == 2:
        mats[0] = mats[0].T
    return reduce(np.matmul, mats)


def rel_error(train, reference):
    got = dense(train).ravel()
    return np.linalg.norm(got - reference.ravel()) / np.linalg.norm(reference)


N_SITES = 5
CHI_EXACT = 64  # above every exact bond of the stacks below


def make_stack(spec, rng, n_sites=N_SITES):
    """Build a stack from a spec such as ``"A B psi"`` or ``"phi A B"``.

    Names ``psi`` and ``phi`` are MPSs, anything else an MPO; bonds are jagged.
    """
    mps_bonds, mpo_bonds = [3, 2, 4, 3][: n_sites - 1], [2, 3, 1, 2][: n_sites - 1]
    return [
        random_mps(mps_bonds, rng)
        if name in {"psi", "phi"}
        else random_mpo(mpo_bonds, rng)
        for name in spec.split()
    ]


KET_STACKS = ["A", "psi", "A psi", "A B", "A B psi", "A B C", "A B C psi"]
BRA_STACKS = ["phi A", "phi A B"]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ----------------------
# --- Generic sweep ---
# ----------------------


@pytest.mark.parametrize("spec", KET_STACKS)
def test_sweep_is_exact_without_truncation(spec, rng):
    stack = make_stack(spec, rng)
    kind = "mps" if stack[-1][0].ndim == 2 else "mpo"

    out = sweep(stack, kind, CHI_EXACT, np.random.default_rng(0), dtype=np.float64)

    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_sweep_rectangular_physical_legs(rng):
    A = random_mpo([2, 3, 2, 2], rng, up=2, down=3)
    B = random_mpo([3, 2, 2, 1], rng, up=3, down=2)
    psi = random_mps([2, 2, 3, 2], rng, phys=2)

    out = sweep(
        [A, B, psi], "mps", CHI_EXACT, np.random.default_rng(0), dtype=np.float64
    )

    assert rel_error(out, dense_stack(A, B, psi)) < 1e-10


def test_sweep_does_not_mutate_inputs(rng):
    stack = make_stack("A B psi", rng)
    before = [[t.copy() for t in train] for train in stack]

    sweep(stack, "mps", 4, np.random.default_rng(0), dtype=np.float64)

    for train, saved in zip(stack, before):
        for t, s in zip(train, saved):
            np.testing.assert_array_equal(t, s)


# --------------------
# --- Public src ----
# --------------------


@pytest.mark.parametrize("spec", KET_STACKS + BRA_STACKS)
def test_src_is_exact_without_truncation(spec, rng):
    stack = make_stack(spec, rng)

    out = src(*stack, chi_out=CHI_EXACT, dtype=np.complex128)

    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_src_bra_is_not_conjugated(rng):
    phi, A = make_stack("phi A", rng)

    out = dense(src(phi, A, chi_out=CHI_EXACT)).ravel()

    plain = (dense(phi).T @ dense(A)).ravel()
    conjugated = (dense(phi).conj().T @ dense(A)).ravel()
    np.testing.assert_allclose(out, plain, atol=1e-10)
    assert np.linalg.norm(out - conjugated) > 1e-2 * np.linalg.norm(plain)


def test_src_identity_layers(rng):
    A = random_mpo([2, 3, 3, 2], rng)
    eye = identity_mpo(N_SITES)

    out = src(eye, eye, A, chi_out=3)

    assert rel_error(out, dense(A)) < 1e-10


def test_src_truncates_to_chi_out(rng):
    stack = make_stack("A B C", rng)

    out = src(*stack, chi_out=5, seed=0)

    assert max(t.shape[0] for t in out[1:]) <= 5


def cut_lower_bound(reference, n_sites, chi):
    """Largest best-rank-``chi`` tail over all cuts of an MPS-shaped reference.

    No train of bond dimension ``chi`` can have a smaller relative error.
    """
    vec = reference.ravel()
    tails = (
        np.linalg.svd(vec.reshape(2**cut, -1), compute_uv=False)[chi:]
        for cut in range(1, n_sites)
    )
    return max(np.linalg.norm(tail) for tail in tails) / np.linalg.norm(vec)


def test_src_truncation_is_near_optimal():
    """Under truncation the error stays within a small factor of the optimum.

    Single instances have a heavy tail (up to ~10x the bound over 200 seeds), so
    the assertion is on the median over fixed instances, which sits near 2x.
    """
    chi = 2
    ratios = []
    for seed in range(10):
        stack = make_stack("A B psi", np.random.default_rng(seed))
        reference = dense_stack(*stack)
        error = rel_error(src(*stack, chi_out=chi, seed=seed), reference)
        bound = cut_lower_bound(reference, N_SITES, chi)
        assert error >= bound * (1 - 1e-10)
        ratios.append(error / bound)

    assert np.median(ratios) <= 4


@pytest.mark.parametrize("spec", ["A B C", "A B psi", "phi A B"])
def test_src_output_is_right_canonical(spec, rng):
    out = src(*make_stack(spec, rng), chi_out=3, oversample=1.0, seed=0)

    for site in out[1:]:
        flat = site.reshape(site.shape[0], -1)
        np.testing.assert_allclose(
            flat @ flat.conj().T, np.eye(site.shape[0]), atol=1e-10
        )


@pytest.mark.parametrize("spec", KET_STACKS + BRA_STACKS)
@pytest.mark.parametrize("oversample", [None, 1.5, 2.0])
def test_oversampled_output_is_left_canonical(spec, oversample, rng):
    out = src(*make_stack(spec, rng), chi_out=3, oversample=oversample, seed=0)

    for site in pad(out)[:-1]:
        flat = site.transpose(0, 2, 3, 1).reshape(-1, site.shape[1])
        np.testing.assert_allclose(
            flat.conj().T @ flat, np.eye(site.shape[1]), atol=1e-10
        )
        assert site.shape[1] <= 3


@pytest.mark.parametrize("spec", KET_STACKS)
@pytest.mark.parametrize("cutoff", [0.0, 0.1])
def test_oversample_one_matches_legacy_sweep(spec, cutoff, rng):
    stack = make_stack(spec, rng)
    kind = "mps" if stack[-1][0].ndim == 2 else "mpo"
    expected = sweep(
        stack,
        kind,
        3,
        np.random.default_rng(0),
        cutoff=cutoff,
        dtype=np.complex128,
    )
    out = src(*stack, chi_out=3, oversample=1.0, cutoff=cutoff, seed=0)

    assert_same_trains(out, expected)


@pytest.mark.parametrize(
    ("chi", "oversample", "expected"),
    [
        (3, None, 13),
        (21, None, 32),
        (3, 1.0, 3),
        (3, np.float32(1.5), 5),
        (3, 2.0, 6),
    ],
)
def test_oversampling_sets_sweep_size(chi, oversample, expected, rng, monkeypatch):
    calls = []
    original = stack_module.sweep

    def spy(*args, **kwargs):
        calls.append((args[2], kwargs["truncate_to"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(stack_module, "sweep", spy)
    src(*make_stack("A B psi", rng), chi_out=chi, oversample=oversample, seed=0)

    assert calls == [(expected, chi if expected > chi else None)]


def test_default_does_not_oversample(rng, monkeypatch):
    calls = []
    original = stack_module.sweep

    def spy(*args, **kwargs):
        calls.append((args[2], kwargs["truncate_to"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(stack_module, "sweep", spy)
    src(*make_stack("A B psi", rng), chi_out=3, seed=0)

    assert calls == [(3, None)]


@pytest.mark.parametrize(
    "oversample", [0.0, -1.0, 0.99, np.nan, np.inf, -np.inf, True, "2", 2j]
)
@pytest.mark.parametrize("spec", ["A psi", "A"])
def test_invalid_oversample_raises(spec, oversample, rng):
    trains = make_stack(spec, rng)
    function = apply if len(trains) == 2 else compress

    with pytest.raises(ValueError, match="oversample"):
        function(*trains, chi_out=3, oversample=oversample)


@pytest.mark.parametrize("spec", ["A B C", "A B psi", "phi A B"])
def test_oversampling_improves_accuracy(spec):
    errors = []
    for seed in range(5):
        trains = make_stack(spec, np.random.default_rng(seed))
        reference = dense_stack(*trains)
        errors.append(
            [
                rel_error(
                    src(*trains, chi_out=2, oversample=factor, seed=seed),
                    reference,
                )
                for factor in (1.0, 2.0, None)
            ]
        )

    legacy, doubled, recommended = np.median(errors, axis=0)
    assert doubled < legacy
    assert recommended < legacy


@pytest.mark.parametrize("cutoff", [0.0, 0.1])
def test_oversampling_matches_exact_truncation_of_sketch(cutoff, rng):
    qtn = pytest.importorskip("quimb.tensor")
    trains = make_stack("A B C", rng)
    intermediate = src(*trains, chi_out=8, oversample=1.0, seed=0)
    reference = qtn.MatrixProductOperator(intermediate)
    reference.compress(form="left", max_bond=3, cutoff=cutoff, cutoff_mode="rel")

    out = src(*trains, chi_out=3, oversample=8 / 3, cutoff=cutoff, seed=0)

    assert rel_error(out, reference.to_dense()) < 1e-10


@pytest.mark.parametrize("kind", ["mps", "mpo"])
@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e8])
def test_oversampled_cutoff_uses_schmidt_values(kind, scale):
    sites = [
        np.diag([scale, 0.05 * scale]),
        np.einsum("ab,ac->abc", np.eye(2), np.eye(2)),
        np.eye(2),
    ]
    if kind == "mpo":
        sites = [site[..., None] for site in sites]
    before = [site.copy() for site in sites]

    out = compress(sites, chi_out=2, cutoff=0.1, oversample=None, seed=0)

    assert all(site.shape[0] == 1 for site in out[1:])
    reference = np.zeros_like(dense(sites))
    reference[0, 0] = scale
    np.testing.assert_allclose(dense(out), reference, atol=1e-12 * scale)
    assert_same_trains(sites, before)


@pytest.mark.parametrize("kind", ["mps", "mpo"])
def test_oversampled_zero_train(kind):
    sites = [np.zeros((2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2))]
    if kind == "mpo":
        sites = [site[..., None] for site in sites]

    out = compress(sites, chi_out=2, cutoff=0.1, oversample=None, seed=0)

    assert all(site.shape[0] >= 1 for site in out[1:])
    np.testing.assert_array_equal(dense(out), dense(sites))


@pytest.mark.parametrize("kind", ["mps", "mpo"])
def test_oversampled_output_does_not_retain_discarded_columns(kind, rng):
    train = (
        random_mps([4, 4, 4], rng, phys=32)
        if kind == "mps"
        else random_mpo([4, 4, 4], rng, up=4, down=8)
    )

    out = compress(train, chi_out=1, oversample=None, seed=0)

    for site in out[:-1]:
        storage = site
        while isinstance(storage.base, np.ndarray):
            storage = storage.base
        assert storage.nbytes == site.nbytes


def test_src_does_not_mutate_bra_stack(rng):
    stack = make_stack("phi A B", rng)
    before = [[t.copy() for t in train] for train in stack]

    src(*stack, chi_out=3, seed=0)

    for train, saved in zip(stack, before):
        for t, s in zip(train, saved):
            np.testing.assert_array_equal(t, s)


def test_src_cutoff_depth_three(rng):
    eye = identity_mpo(N_SITES)
    psi = random_mps([2, 4, 4, 2], rng)

    out = src(eye, eye, psi, chi_out=16, cutoff=1e-10, seed=0)
    full = src(eye, eye, psi, chi_out=16, seed=0)

    assert rel_error(out, dense(psi)) < 1e-8
    assert all(o.shape[0] <= f.shape[0] for o, f in zip(out[1:], full[1:]))
    assert max(t.shape[0] for t in out[1:]) <= 4


def test_src_seed_is_deterministic(rng):
    stack = make_stack("A B psi", rng)

    first = src(*stack, chi_out=4, seed=7)
    second = src(*stack, chi_out=4, seed=7)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("spec", ["A B psi", "A B C", "phi A B"])
@pytest.mark.parametrize("oversample", [None, 1.0, 2.0])
def test_src_two_site_stack_falls_back(spec, oversample, rng, caplog):
    stack = make_stack(spec, rng, n_sites=2)

    out = src(*stack, chi_out=CHI_EXACT, oversample=oversample)

    assert "Defaulting" in caplog.text
    assert rel_error(out, dense_stack(*stack)) < 1e-10


def test_src_single_site_raises_before_warning(caplog):
    mpo = [np.ones((1, 2, 2))]
    mps = [np.ones((1, 2))]

    with pytest.raises(ValueError, match="two-site tensor train"):
        src(mpo, mpo, mps, chi_out=4)

    assert "Defaulting" not in caplog.text


def test_src_validates_the_stack(rng):
    with pytest.raises(TypeError, match="MPS may come first"):
        src(*make_stack("psi A psi", rng), chi_out=4)


# -------------------------
# --- Contraction paths ---
# -------------------------


@pytest.mark.parametrize(
    "optimize", ["auto", "greedy", oe.DynamicProgramming(minimize="size")]
)
@pytest.mark.parametrize("spec", ["A B psi", "A B C", "phi A B"])
def test_optimizer_does_not_change_the_result(spec, optimize, rng):
    stack = make_stack(spec, rng)

    default = src(*stack, chi_out=4, seed=7)
    other = src(*stack, chi_out=4, seed=7, optimize=optimize)

    assert rel_error(other, dense(default)) < 1e-10


def test_registered_optimizer_is_used(rng, monkeypatch):
    calls = []

    def counting_greedy(*args, **kwargs):
        calls.append(1)
        return oe.paths.greedy(*args, **kwargs)

    # What opt_einsum.paths.register_path_fn does, undone after the test.
    monkeypatch.setitem(oe.paths._PATH_OPTIONS, "src-test-greedy", counting_greedy)
    stack = make_stack("A B psi", rng)

    out = src(*stack, chi_out=CHI_EXACT, optimize="src-test-greedy")

    assert calls
    assert rel_error(out, dense_stack(*stack)) < 1e-10


@pytest.mark.parametrize(
    ("optimize", "error"),
    [("fastest", ValueError), (True, TypeError), ([(0, 1), (0, 1)], TypeError)],
)
@pytest.mark.parametrize("entry", ["src", "apply", "compress"])
def test_invalid_optimizer_raises(entry, optimize, error, rng):
    A, psi = make_stack("A psi", rng)
    call = {
        "src": lambda: src(A, psi, chi_out=4, optimize=optimize),
        "apply": lambda: apply(A, psi, 4, optimize=optimize),
        "compress": lambda: compress(psi, 4, optimize=optimize),
    }[entry]

    with pytest.raises(error, match="optimize"):
        call()


# ----------------
# --- Wrappers ---
# ----------------


def assert_same_trains(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def assert_cutoff_trims(train, untrimmed):
    """Guard for the wrapper tests: the cutoff must change the result."""
    assert [t.shape for t in train] != [t.shape for t in untrimmed]


@pytest.mark.parametrize("oversample", [None, 1.0, 1.5])
@pytest.mark.parametrize("cutoff", [0.0, 0.5])
@pytest.mark.parametrize("spec", ["A psi", "A B"])
def test_apply_matches_src(spec, cutoff, oversample, rng):
    left, right = make_stack(spec, rng)

    via_apply = apply(
        left,
        right,
        4,
        cutoff=cutoff,
        oversample=oversample,
        dtype=np.complex128,
        seed=3,
    )
    via_src = src(
        left,
        right,
        chi_out=4,
        cutoff=cutoff,
        oversample=oversample,
        dtype=np.complex128,
        seed=3,
    )

    assert_same_trains(via_apply, via_src)
    if cutoff:
        assert_cutoff_trims(via_src, src(left, right, chi_out=4, seed=3))


@pytest.mark.parametrize("oversample", [None, 1.0, 1.5])
@pytest.mark.parametrize("cutoff", [0.0, 0.5])
@pytest.mark.parametrize("spec", ["psi", "A"])
def test_compress_matches_src(spec, cutoff, oversample, rng):
    (train,) = make_stack(spec, rng)

    via_compress = compress(train, 4, cutoff=cutoff, oversample=oversample, seed=3)
    via_src = src(train, chi_out=4, cutoff=cutoff, oversample=oversample, seed=3)

    assert_same_trains(via_compress, via_src)
    if cutoff:
        assert_cutoff_trims(via_src, src(train, chi_out=4, seed=3))


def test_apply_rejects_bra(rng):
    phi, A = make_stack("phi A", rng)

    with pytest.raises(TypeError, match="expected an MPO on the left"):
        apply(phi, A, 4)


# -----------------------------------------
# --- Check for performance regressions ---
# -----------------------------------------


@pytest.mark.perf
def test_benchmark_src_stack_depth3(benchmark):
    """Benchmarks a depth-3 stack: two near-identity MPOs applied to an MPS."""
    qtn = pytest.importorskip("quimb.tensor")
    n_sites, phys_dim, chi_out = 10, 2, 32
    dtype = np.complex128
    layers = [
        qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=dtype)
        + 1e-8
        * qtn.MPO_rand(n_sites, bond_dim=3, phys_dim=phys_dim, dtype=dtype, seed=seed)
        for seed in (1, 2)
    ]
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=dtype, seed=3
    )

    out = benchmark(
        src, *(t.arrays for t in layers), psi.arrays, chi_out=chi_out, dtype=dtype
    )

    # Still has to be correct
    ref = layers[0].apply(layers[1].apply(psi, compress=False), compress=False)
    np.testing.assert_allclose(
        ref.distance(qtn.MatrixProductState(out)), 0.0, atol=1e-6
    )


# ----------------------------
# --- Budgets and spilling ---
# ----------------------------


@pytest.fixture
def plans(monkeypatch):
    """Record the plan of every sweep."""
    recorded = []

    def spy(*args, **kwargs):
        recorded.append(make_plan(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(sweep_module, "make_plan", spy)
    return recorded


def test_tiny_budget_batches_and_spills(rng, tmp_path, plans):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]
    tight = Resources(host_memory="1MB", scratch_dir=tmp_path)

    default = src(*stack, chi_out=16, seed=3, dtype=np.complex128)
    spilled = src(*stack, chi_out=16, seed=3, dtype=np.complex128, resources=tight)

    assert all(site.tier == "device" for site in plans[0].sites)
    assert "disk" in {site.tier for site in plans[1].sites}
    assert min(site.sketch_batch for site in plans[1].sites[1:]) < 16
    # Rounding differs, and the output cores of an ill-conditioned sketch with it,
    # but not the operator they represent.
    assert rel_error(spilled, dense(default)) < 1e-10
    assert list(tmp_path.iterdir()) == []


def test_budget_too_small_raises(rng):
    stack = make_stack("A B psi", rng)

    with pytest.raises(MemoryError, match="working set exceeds the budget"):
        src(*stack, chi_out=8, resources=Resources(host_memory="1kB"))


def test_apply_passes_resources(rng):
    A, psi = make_stack("A psi", rng)

    with pytest.raises(MemoryError, match="working set exceeds the budget"):
        apply(A, psi, chi_out=8, resources=Resources(host_memory="1kB"))


def test_compress_passes_resources(rng):
    (A,) = make_stack("A", rng)

    with pytest.raises(MemoryError, match="working set exceeds the budget"):
        compress(A, chi_out=8, resources=Resources(host_memory="1kB"))


def _rows_capped(monkeypatch, cap):
    """Make every kernel run out of memory on batches of more than ``cap`` rows."""
    rows_of = {
        "env": lambda args: args[0].shape[0],
        "sketch": lambda args: args[0].shape[0],
        "project": lambda args: args[0].shape[0],
        "first": lambda args: args[1].shape[0],
    }
    for name, rows in rows_of.items():
        original = getattr(SiteKernels, name)

        def kernel(self, *args, original=original, rows=rows):
            if rows(args) > cap:
                msg = "simulated out of memory"
                raise MemoryError(msg)
            return original(self, *args)

        monkeypatch.setattr(SiteKernels, name, kernel)


@pytest.mark.parametrize("devices", [None, 2])
def test_out_of_memory_splits_the_batches(devices, rng, monkeypatch, caplog):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(3)]
    resources = Resources(devices=devices)
    reference = src(*stack, chi_out=16, seed=4, dtype=np.complex128)

    _rows_capped(monkeypatch, 4)
    with caplog.at_level(logging.DEBUG, logger="portc"):
        split = src(
            *stack, chi_out=16, seed=4, dtype=np.complex128, resources=resources
        )

    assert [t.shape for t in split] == [t.shape for t in reference]
    assert rel_error(split, dense(reference)) < 1e-10
    for kernel in ("env", "sketch", "project", "first"):
        assert f"Out of memory in a {kernel} batch" in caplog.text


def test_out_of_memory_on_one_row_raises(rng, monkeypatch):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(3)]

    _rows_capped(monkeypatch, 0)
    with pytest.raises(MemoryError, match="simulated out of memory"):
        src(*stack, chi_out=16, seed=4)


# -------------------------------
# --- Several (simulated) devices ---
# -------------------------------


@pytest.fixture(params=[False, True], ids=["householder", "cholesky"])
def cholesky(request, monkeypatch):
    """Factor the sketches of several devices by Householder QR or by CholeskyQR."""
    if request.param:
        monkeypatch.setattr(sweep_module, "DISTRIBUTED_QR_MIN", 0)
    return request.param


@pytest.mark.parametrize(
    ("devices", "chi"), [(2, 16), (3, 16), (3, 17), ([2, 0, 1], 16), (4, 3)]
)
@pytest.mark.usefixtures("cholesky")
def test_devices_match_one_device(devices, chi, rng):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]

    one = src(*stack, chi_out=chi, seed=5, dtype=np.complex128)
    many = src(
        *stack,
        chi_out=chi,
        seed=5,
        dtype=np.complex128,
        resources=Resources(devices=devices),
    )

    assert [t.shape for t in many] == [t.shape for t in one]
    assert rel_error(many, dense(one)) < 1e-10


@pytest.mark.parametrize("spec", ["psi", "A psi", "A B", "phi A B"])
def test_devices_are_exact_without_truncation(spec, rng):
    stack = make_stack(spec, rng)

    out = src(*stack, chi_out=CHI_EXACT, resources=Resources(devices=3))

    assert rel_error(out, dense_stack(*stack)) < 1e-10


@pytest.mark.usefixtures("cholesky")
def test_devices_with_cutoff_keep_the_bonds(rng):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(3)]

    one = src(*stack, chi_out=24, cutoff=0.05, seed=2)
    many = src(*stack, chi_out=24, cutoff=0.05, seed=2, resources=Resources(devices=2))

    assert [t.shape for t in many] == [t.shape for t in one]
    assert rel_error(many, dense(one)) < 1e-10


@pytest.mark.parametrize("cutoff", [0.0, 0.05])
def test_devices_factor_tall_sketches_by_cholesky_qr(cutoff, rng, monkeypatch, caplog):
    stack = [random_mpo([3, 4, 4, 4, 4, 3], rng) for _ in range(3)]
    one = src(*stack, chi_out=24, cutoff=cutoff, seed=6, dtype=np.complex128)

    monkeypatch.setattr(sweep_module, "DISTRIBUTED_QR_MIN", 0)
    with caplog.at_level(logging.DEBUG, logger="portc"):
        many = src(
            *stack,
            chi_out=24,
            cutoff=cutoff,
            seed=6,
            dtype=np.complex128,
            resources=Resources(devices=3),
        )

    assert "CholeskyQR at 4 sites" in caplog.text
    assert [t.shape for t in many] == [t.shape for t in one]
    assert rel_error(many, dense(one)) < 1e-10
    for core in many[1:]:
        matrix = core.reshape(core.shape[0], -1)
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(core.shape[0]), atol=1e-13
        )


def test_cholesky_qr_backs_off_after_repeated_fallbacks():
    backoff = sweep_module._Backoff()
    outcomes = iter([False, False, False, True, False, True])

    tried = []
    for _ in range(10):
        if backoff.attempt():
            tried.append(True)
            backoff.record(succeeded=next(outcomes))
        else:
            tried.append(False)

    # Fail, fail (skip 2), fail (skip 4), ...: only the second fallback in a row
    # starts skipping.
    assert tried[:6] == [True, True, False, False, True, False]
    assert (backoff.tried, backoff.fallbacks) == (4, 3)


def test_devices_spill_and_clean_up(rng, tmp_path, plans):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(4)]
    tight = Resources(host_memory="1MB", scratch_dir=tmp_path, devices=2)

    default = src(*stack, chi_out=16, seed=3, dtype=np.complex128)
    spilled = src(*stack, chi_out=16, seed=3, dtype=np.complex128, resources=tight)

    assert "disk" in {site.tier for site in plans[1].sites}
    assert rel_error(spilled, dense(default)) < 1e-10
    assert list(tmp_path.iterdir()) == []


def test_an_error_on_one_device_is_raised(rng, monkeypatch, tmp_path):
    stack = [random_mpo([3, 4, 4, 4, 3], rng) for _ in range(2)]
    calls, lock = [0], threading.Lock()
    original = SiteKernels.sketch

    def flaky(self, *args):
        with lock:
            calls[0] += 1
            fail = calls[0] == 2
        if fail:
            msg = "sketch failed"
            raise RuntimeError(msg)
        return original(self, *args)

    monkeypatch.setattr(SiteKernels, "sketch", flaky)
    resources = Resources(host_memory="2MB", scratch_dir=tmp_path, devices=3)
    with pytest.raises(RuntimeError, match="sketch failed"):
        src(*stack, chi_out=16, seed=0, resources=resources)
    assert list(tmp_path.iterdir()) == []


def test_devices_are_validated(rng):
    stack = make_stack("A B", rng)

    with pytest.raises(ValueError, match="positive device count"):
        Resources(devices=0)
    with pytest.raises(ValueError, match="distinct"):
        Resources(devices=[1, 1])
    with pytest.raises(TypeError, match="device count or a sequence"):
        Resources(devices="0,1")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="integer device ids"):
        Resources(devices=[0.0, 1.0])  # ty: ignore[invalid-argument-type]
    # Simulated devices have no upper limit.
    src(*stack, chi_out=4, resources=Resources(devices=[0, 7]))

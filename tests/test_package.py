"""Test the four flavors of the SRC algorithm.

1. MPO-MPS randomized contraction-compression.
2. MPO-MPO randomized contraction-compression.
3. MPO randomized compression.
4. MPS randomized compression.


"""

from __future__ import annotations

from importlib.metadata import distribution

import numpy as np
import pytest
import quimb.tensor as qtn

import portc
from portc import apply, compress
from portc.stack import src
from portc.utils import gaussian_sketch


def test_installed_package_metadata() -> None:
    installed = distribution("portc")

    assert installed.metadata["Name"] == "portc"
    assert installed.version == portc.__version__
    assert set(installed.metadata.get_all("Provides-Extra") or []) == {
        "gpu-nvidia",
        "gpu-rocm",
    }
    assert portc.apply is apply
    assert portc.compress is compress
    assert portc.src is src


# -------------
# --- Utils ---
# -------------

# ``portc`` operates on plain lists of site arrays.  quimb is used here only
# to build reference networks and to measure distances, so every call unwraps its
# inputs with ``.arrays`` and re-wraps the result with one of the helpers below.


def as_mps(arrays: list[np.ndarray]) -> qtn.MatrixProductState:
    """Wrap a list of site arrays returned by portc into a quimb MPS."""
    return qtn.MatrixProductState(arrays)


def as_mpo(arrays: list[np.ndarray]) -> qtn.MatrixProductOperator:
    """Wrap a list of site arrays returned by portc into a quimb MPO."""
    return qtn.MatrixProductOperator(arrays)


def random_mpo(
    bonds: list[int],
    *,
    phys: int = 2,
    dtype: type = np.complex128,
    rng: np.random.Generator | None = None,
) -> qtn.MatrixProductOperator:
    """Generate a random MPO with given bond dimensions and physical dimension.

    Useful for generating jagged MPOs.

    Args:
        bonds: List of bond dimensions for the MPO. The last site is implicit.
        phys: Physical dimension. Defaults to 2.
        dtype: Data type of the tensors. Defaults to np.complex128.
        rng: Random number generator. Defaults to a fresh unseeded generator.

    Returns:
        qtn.MatrixProductOperator: The generated random MPO.
    """
    # A default generator built here rather than in the signature: a default
    # argument is evaluated once, so callers would share generator state.
    if rng is None:
        rng = np.random.default_rng()

    # First site
    tensors = [rng.normal(size=(bonds[0], phys, phys)).astype(dtype)]

    # Bulk sites
    tensors.extend(
        rng.normal(size=(bonds[i - 1], bonds[i], phys, phys)).astype(dtype)
        for i in range(1, len(bonds))
    )
    # Last site
    tensors.append(rng.normal(size=(bonds[-1], phys, phys)).astype(dtype))

    return qtn.MatrixProductOperator(tensors)


# ----------------
# --- Fixtures ---
# ----------------


@pytest.fixture
def n_sites():
    return 5


@pytest.fixture
def n_sites_small():
    return 2


@pytest.fixture
def phys_dim():
    return 2


@pytest.fixture
def chi_out():
    return 10


@pytest.fixture
def array_type():
    return np.complex128


@pytest.fixture
def mpo_jagged_left(phys_dim, array_type):
    return random_mpo([1, 4, 1, 4, 1], phys=phys_dim, dtype=array_type) * 1e-8


@pytest.fixture
def mpo_jagged_right(phys_dim, array_type):
    return random_mpo([100, 400, 100, 400, 100], phys=phys_dim, dtype=array_type)


# --------------------------------------------
# --- Test MPO-MPS contraction-compression ---
# --------------------------------------------


def test_src_mpo_mps(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPS contraction-compression."""

    # Generate a random MPS and connect it to the identity MPO
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # SRC MPS should be identical to the original
    psi_compress = as_mps(
        apply(H.arrays, psi.arrays, chi_out=chi_out, dtype=array_type)
    )

    np.testing.assert_allclose(psi.distance(psi_compress), 0.0, atol=1e-6)


def test_src_mpo_mps_trims_terminal_bond() -> None:
    """MPO-MPS apply should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 2, 64
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=np.complex128)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=3
    )

    psi_src = apply(H.arrays, psi.arrays, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert psi_src[-1].shape[0] < chi_out
    assert psi_src[-1].shape[0] <= phys_dim


# --------------------------------------------
# --- Test MPO-MPO contraction-compression ---
# --------------------------------------------


def test_src_mpo_mpo_identity(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPO contraction-compression."""

    # Generate a random MPO and the identity MPO
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)

    # The compressed product should be identical to the original MPO
    H_compress = as_mpo(apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(H1.distance(H_compress), 0.0, atol=1e-6)


def test_src_mpo_mpo(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPO contraction-compression."""

    # Generate a random MPO and a perturbed identity MPO
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # Quimb's contraction
    H_ref = H1.apply(H2, compress=False)

    # The compressed product should be identical to the original MPO
    H_src = as_mpo(apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(H_ref.distance(H_src), 0.0, atol=1e-6)


def test_src_mpo_mpo_long_phys(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO-MPO contraction-compression."""
    phys_dim_long = 3 * phys_dim  # triggers transpose in isometry

    # Generate a random MPO and a perturbed identity MPO
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim_long, dtype=array_type
    )
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim_long, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim_long, dtype=array_type
    )

    # Quimb's contraction
    H_ref = H1.apply(H2, compress=False)

    # The compressed product should be identical to the original MPO
    H_src = as_mpo(apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(H_ref.distance(H_src), 0.0, atol=1e-6)


def test_src_mpo_mpo_jagged(mpo_jagged_left, mpo_jagged_right, array_type):
    """Tests the SRC MPO-MPO contraction-compression with jagged MPOs."""
    # Quimb's contraction
    H_ref = mpo_jagged_left.apply(mpo_jagged_right, compress=False)

    # The compressed product should be identical to the original MPO
    H_src = as_mpo(
        apply(
            mpo_jagged_left.arrays,
            mpo_jagged_right.arrays,
            chi_out=100,
            dtype=array_type,
        )
    )

    np.testing.assert_allclose(H_ref.distance(H_src), 0.0, atol=1e-6)


def test_src_mpo_mpo_trims_terminal_bond() -> None:
    """MPO-MPO apply should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 4, 128
    H1 = qtn.MPO_rand(
        n_sites, bond_dim=8, phys_dim=phys_dim, dtype=np.complex128, seed=1
    )
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=np.complex128
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=2
    )

    H_src = apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert H_src[-1].shape[0] < chi_out
    assert H_src[-1].shape[0] <= phys_dim**2


# --------------------------------
# ----- Test MPO compression -----
# --------------------------------


def test_src_mpo_compression(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPO compression."""

    # Add an almost-zero MPO B to a dense MPO A, inflating the bond dimension
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    B = qtn.MPO_rand(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type) / 1e8
    C = A + B

    # SRC MPO. It should be trivially compressed.
    D = as_mpo(compress(C.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_src_mpo_compression_trims_terminal_bond() -> None:
    """MPO compression should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 4, 128
    A = qtn.MPO_rand(
        n_sites, bond_dim=8, phys_dim=phys_dim, dtype=np.complex128, seed=4
    )
    B = (
        qtn.MPO_rand(
            n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=5
        )
        / 1e8
    )
    C = A + B

    D = compress(C.arrays, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert D[-1].shape[0] < chi_out
    assert D[-1].shape[0] <= phys_dim**2


# --------------------------------
# ----- Test MPS compression -----
# --------------------------------


def test_src_mps(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPS compression."""

    # Add an almost-zero MPS B to a dense MPS A, inflating the bond dimension
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    # SRC MPS. It should be trivially compressed.
    D = as_mps(compress(C.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_src_mps_compression_trims_terminal_bond() -> None:
    """MPS compression should not force the terminal bond to ``chi_out``."""
    n_sites, phys_dim, chi_out = 5, 2, 64
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=8, phys_dim=phys_dim, dtype=np.complex128, seed=6
    )
    B = (
        qtn.MPS_rand_state(
            n_sites, bond_dim=4, phys_dim=phys_dim, dtype=np.complex128, seed=7
        )
        / 1e8
    )
    C = A + B

    D = compress(C.arrays, chi_out=chi_out, dtype=np.complex128, seed=0)

    assert D[-1].shape[0] < chi_out
    assert D[-1].shape[0] <= phys_dim


def test_src_mps_small_chi(n_sites, phys_dim, chi_out, array_type):
    """Tests the SRC MPS compression with a smaller BD.

    Checks that the code works when wide matrices are involved.
    """
    chi_small = chi_out // 2

    # Add an almost-zero MPS B to a dense MPS A, inflating the bond dimension
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    # SRC MPS. It should be trivially compressed.
    D = as_mps(compress(C.arrays, chi_out=chi_small, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


# ---------------------------------------------------
# --- Test exact fallback for small (<3 site) TNs ---
# ---------------------------------------------------


def test_apply_small_system_dispatch(n_sites_small, phys_dim, chi_out, array_type):
    """Tests that apply falls back to the exact path for small systems."""

    # Generate a random MPS and connect it to the identity MPO
    H = qtn.MPO_identity(n_sites_small, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites_small, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # SRC MPS should be identical to the original
    psi_compress = as_mps(
        apply(H.arrays, psi.arrays, chi_out=chi_out, dtype=array_type)
    )

    np.testing.assert_allclose(psi.distance(psi_compress), 0.0, atol=1e-6)


def test_apply_small_system_mpo_mpo(n_sites_small, phys_dim, chi_out, array_type):
    """The exact fallback must also handle two-site MPO-MPO products."""
    H1 = qtn.MPO_rand(
        n_sites_small, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    H2 = qtn.MPO_identity(n_sites_small, phys_dim=phys_dim, dtype=array_type)

    H_src = as_mpo(apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(H1.distance(H_src), 0.0, atol=1e-6)


def test_compress_small_system_dispatch(n_sites_small, phys_dim, chi_out, array_type):
    """Tests that compress falls back to the exact path for small systems."""

    # Add an almost-zero MPS B to a dense MPS A, inflating the bond dimension
    A = qtn.MPS_rand_state(
        n_sites_small, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(
            n_sites_small, bond_dim=5, phys_dim=phys_dim, dtype=array_type
        )
        / 1e8
    )
    C = A + B

    # SRC MPS. It should be trivially compressed.
    D = as_mps(compress(C.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_compress_small_system_mpo(n_sites_small, phys_dim, chi_out, array_type):
    """The exact fallback must also handle two-site MPOs."""
    A = qtn.MPO_rand(
        n_sites_small, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPO_rand(n_sites_small, bond_dim=2, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    D = as_mpo(compress(C.arrays, chi_out=chi_out, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_single_site_train_raises(phys_dim, chi_out, array_type):
    """A single-site train is degenerate and must be rejected."""
    with pytest.raises(ValueError, match="two-site tensor train"):
        compress([np.ones((1, phys_dim), dtype=array_type)], chi_out=chi_out)


def test_single_site_train_raises_without_warning(
    phys_dim, chi_out, array_type, caplog
):
    """The degenerate train must raise before the fallback is announced."""
    mpo = [np.ones((1, phys_dim, phys_dim), dtype=array_type)]
    mps = [np.ones((1, phys_dim), dtype=array_type)]

    with pytest.raises(ValueError, match="two-site tensor train"):
        apply(mpo, mps, chi_out=chi_out)

    assert "Defaulting" not in caplog.text


def test_mismatched_site_counts_raise(phys_dim, chi_out, array_type):
    """A shorter left train must not silently truncate the right one."""
    H = qtn.MPO_identity(4, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(6, bond_dim=4, phys_dim=phys_dim, dtype=array_type)

    with pytest.raises(ValueError, match="same number of sites"):
        apply(H.arrays, psi.arrays, chi_out=chi_out, dtype=array_type)


# -------------------------------------------------
# --- Test adaptive bond truncation via cutoff  ---
# -------------------------------------------------


def test_cutoff_zero_matches_default_mpo_mpo(n_sites, phys_dim, chi_out, array_type):
    """cutoff=0 should produce the same result as no cutoff (default)."""
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    H_default = as_mpo(
        apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type, seed=42)
    )
    H_cutoff0 = as_mpo(
        apply(
            H1.arrays, H2.arrays, chi_out=chi_out, cutoff=0.0, dtype=array_type, seed=42
        )
    )

    np.testing.assert_allclose(H_default.distance(H_cutoff0), 0.0, atol=1e-6)


def test_cutoff_trims_bonds_mpo_mpo(n_sites, phys_dim, chi_out, array_type):
    """With cutoff>0, bonds with low effective rank should be trimmed."""
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)

    # Identity product: effective rank = chi_out, no truncation expected
    H_no_cut = as_mpo(
        apply(H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type, seed=42)
    )
    H_cut = as_mpo(
        apply(
            H1.arrays,
            H2.arrays,
            chi_out=chi_out,
            cutoff=1e-10,
            dtype=array_type,
            seed=42,
        )
    )

    # Result should still be accurate
    np.testing.assert_allclose(H1.distance(H_cut), 0.0, atol=1e-5)

    # Bonds should be <= those without cutoff
    for b_cut, b_nocut in zip(H_cut.bond_sizes(), H_no_cut.bond_sizes()):
        assert b_cut <= b_nocut


def test_cutoff_preserves_accuracy_mpo_mps(n_sites, phys_dim, chi_out, array_type):
    """cutoff should preserve accuracy for MPO-MPS."""
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    psi_cut = as_mps(
        apply(H.arrays, psi.arrays, chi_out=chi_out, cutoff=1e-10, dtype=array_type)
    )

    np.testing.assert_allclose(psi.distance(psi_cut), 0.0, atol=1e-6)


def test_cutoff_preserves_accuracy_compress_mpo(n_sites, phys_dim, chi_out, array_type):
    """cutoff should preserve accuracy for MPO compression."""
    A = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    B = qtn.MPO_rand(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type) / 1e8
    C = A + B

    D = as_mpo(compress(C.arrays, chi_out=chi_out, cutoff=1e-10, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


def test_cutoff_preserves_accuracy_compress_mps(n_sites, phys_dim, chi_out, array_type):
    """cutoff should preserve accuracy for MPS compression."""
    A = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    B = (
        qtn.MPS_rand_state(n_sites, bond_dim=5, phys_dim=phys_dim, dtype=array_type)
        / 1e8
    )
    C = A + B

    D = as_mps(compress(C.arrays, chi_out=chi_out, cutoff=1e-10, dtype=array_type))

    np.testing.assert_allclose(C.distance(D), 0.0, atol=1e-6)


# --------------------------------------------
# --- Test unsupported tensor combinations ---
# --------------------------------------------


def test_apply_unsupported_types(n_sites, phys_dim, chi_out, array_type):
    """Tests that apply raises TypeError for unsupported tensor combinations."""

    # Generate a random MPS and MPO
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )
    phi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # Unsupported combination: MPS-MPS
    with pytest.raises(TypeError):
        apply(psi.arrays, phi.arrays, chi_out=chi_out, dtype=array_type)


def test_compress_unsupported_type(n_sites, chi_out, array_type):
    """Tests that compress raises TypeError for unsupported tensor layouts."""

    # Rank-4 boundary tensors are neither an MPS nor an MPO (e.g. a PEPS row, or a
    # periodic MPO); the message points at the periodic case.
    peps_like = [np.zeros((2, 2, 2, 2), dtype=array_type)] * n_sites

    with pytest.raises(TypeError, match="periodic"):
        compress(peps_like, chi_out=chi_out, dtype=array_type)


def test_periodic_boundary_mps_raises(n_sites, chi_out, array_type):
    """A periodic MPS has rank 3 everywhere, so its first site looks like an MPO
    boundary; the interior rank gives it away.
    """
    periodic_mps = [np.zeros((2, 2, 2), dtype=array_type)] * n_sites

    with pytest.raises(ValueError, match="open boundary"):
        compress(periodic_mps, chi_out=chi_out, dtype=array_type)

    ok_mpo = qtn.MPO_identity(n_sites, phys_dim=2, dtype=array_type).arrays
    with pytest.raises(ValueError, match="open boundary"):
        apply(periodic_mps, ok_mpo, chi_out=chi_out, dtype=array_type)


def test_inconsistent_site_ranks_raise(n_sites, chi_out, array_type):
    """Both ends can be rank 3 and still disagree with an interior site."""
    mpo = list(qtn.MPO_identity(n_sites, phys_dim=2, dtype=array_type).arrays)
    mpo[-2] = np.zeros((mpo[-2].shape[0], mpo[-2].shape[1], 2), dtype=array_type)

    with pytest.raises(ValueError, match="open boundary"):
        compress(mpo, chi_out=chi_out, dtype=array_type)

    ok_mpo = qtn.MPO_identity(n_sites, phys_dim=2, dtype=array_type).arrays
    with pytest.raises(ValueError, match="open boundary"):
        apply(ok_mpo, mpo, chi_out=chi_out, dtype=array_type)


def test_mismatched_boundary_ranks_raise(chi_out, array_type):
    """The two ends of a train must have the same rank, also for two sites."""
    two_site = [
        np.zeros((2, 2, 2), dtype=array_type),
        np.zeros((2, 2), dtype=array_type),
    ]
    with pytest.raises(ValueError, match="open boundary"):
        compress(two_site, chi_out=chi_out, dtype=array_type)

    longer = [
        np.zeros((2, 2, 2), dtype=array_type),
        np.zeros((2, 2, 2, 2), dtype=array_type),
        np.zeros((2, 2, 2), dtype=array_type),
        np.zeros((2, 2, 2, 2), dtype=array_type),
    ]
    with pytest.raises(ValueError, match="open boundary"):
        compress(longer, chi_out=chi_out, dtype=array_type)


# -----------------------------------------------
# --- Test chi_out / cutoff argument validation --
# -----------------------------------------------


@pytest.mark.parametrize(
    ("chi_out", "error"),
    [(0, ValueError), (-3, ValueError), (2.5, TypeError), (True, TypeError)],
)
def test_invalid_chi_out_raises(n_sites, phys_dim, chi_out, error, array_type):
    """Non-positive chi_out raises ValueError, a non-integer one TypeError."""
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(n_sites, bond_dim=4, phys_dim=phys_dim, dtype=array_type)

    with pytest.raises(error, match="chi_out"):
        apply(H.arrays, psi.arrays, chi_out=chi_out, dtype=array_type)
    with pytest.raises(error, match="chi_out"):
        compress(H.arrays, chi_out=chi_out, dtype=array_type)


@pytest.mark.parametrize("cutoff", [2.0, -1.0])
def test_invalid_cutoff_raises(n_sites, phys_dim, chi_out, cutoff, array_type):
    """cutoff outside [0.0, 1.0) must raise ValueError, not silently misbehave."""
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    with pytest.raises(ValueError, match="cutoff"):
        apply(H.arrays, psi.arrays, chi_out=chi_out, cutoff=cutoff, dtype=array_type)
    with pytest.raises(ValueError, match="cutoff"):
        compress(H.arrays, chi_out=chi_out, cutoff=cutoff, dtype=array_type)


def test_numpy_integer_chi_out_accepted(n_sites, phys_dim, chi_out, array_type):
    """chi_out as a numpy integer (np.int64 etc.) must be accepted, not rejected.

    validate_chi_out deliberately checks `numbers.Integral` rather than
    `isinstance(chi_out, int)` so numpy integer types aren't rejected.
    """
    H = qtn.MPO_identity(n_sites, phys_dim=phys_dim, dtype=array_type)
    psi = qtn.MPS_rand_state(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    psi_compress = as_mps(
        apply(H.arrays, psi.arrays, chi_out=np.int64(chi_out), dtype=array_type)
    )
    np.testing.assert_allclose(psi.distance(psi_compress), 0.0, atol=1e-6)


# -----------------------------------------
# --- Check for performance regressions ---
# -----------------------------------------


@pytest.mark.perf
def test_benchmark_src_mpo_mpo(benchmark):
    """Benchmarks the SRC MPO-MPO contraction-compression."""
    # Problem size
    n_sites = 10
    phys_dim = 4
    chi_out = 40
    array_type = np.complex128

    # MPOs
    H1 = qtn.MPO_rand(n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type)
    H2 = qtn.MPO_identity(
        n_sites, phys_dim=phys_dim, dtype=array_type
    ) + 1e-8 * qtn.MPO_rand(
        n_sites, bond_dim=chi_out, phys_dim=phys_dim, dtype=array_type
    )

    # Benchmark the application
    result_mpo = benchmark(
        apply, H1.arrays, H2.arrays, chi_out=chi_out, dtype=array_type
    )

    # Still has to be correct
    np.testing.assert_allclose(H1.distance(as_mpo(result_mpo)), 0.0, atol=1e-6)


@pytest.mark.parametrize("make_train", [qtn.MPS_rand_state, qtn.MPO_rand])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_compress_precision(make_train, dtype) -> None:
    tensor = make_train(4, bond_dim=2, dtype=dtype, seed=12)

    result = compress(tensor.arrays, chi_out=2, seed=14)

    assert all(arr.dtype == dtype for arr in result)
    tolerance = 2e-5 if np.finfo(dtype).bits == 32 else 1e-12
    np.testing.assert_allclose(
        type(tensor)(result).to_dense(),
        tensor.to_dense(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("make_train", [qtn.MPS_rand_state, qtn.MPO_rand])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_apply_precision(make_train, dtype) -> None:
    tensor = make_train(4, bond_dim=2, dtype=dtype, seed=12)
    identity = qtn.MPO_identity(4, phys_dim=2, dtype=dtype)

    result = apply(identity.arrays, tensor.arrays, chi_out=2, seed=14)

    assert all(arr.dtype == dtype for arr in result)
    tolerance = 2e-5 if np.finfo(dtype).bits == 32 else 1e-12
    np.testing.assert_allclose(
        type(tensor)(result).to_dense(),
        tensor.to_dense(),
        rtol=tolerance,
        atol=tolerance,
    )


def test_stack_dtype_follows_inputs() -> None:
    mpo = qtn.MPO_rand(4, bond_dim=2, dtype=np.complex64, seed=1)
    result = src(mpo.arrays, mpo.arrays, chi_out=4, seed=0)
    assert all(arr.dtype == np.complex64 for arr in result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_gaussian_sketch(dtype) -> None:
    prng = np.random.default_rng(0)
    omega = gaussian_sketch(prng, (400, 400), dtype)

    assert omega.dtype == dtype
    assert omega.var() == pytest.approx(1.0, rel=0.05)
    if np.dtype(dtype).kind == "c":
        # Circularly symmetric: real and imaginary parts share the variance.
        assert omega.real.var() == pytest.approx(0.5, rel=0.05)
        assert omega.imag.var() == pytest.approx(0.5, rel=0.05)
        assert abs((omega**2).mean()) < 0.02


def test_gaussian_sketch_seed_is_precision_independent() -> None:
    single = gaussian_sketch(np.random.default_rng(3), (5, 5), np.complex64)
    double = gaussian_sketch(np.random.default_rng(3), (5, 5), np.complex128)
    np.testing.assert_allclose(single, double, rtol=1e-6)

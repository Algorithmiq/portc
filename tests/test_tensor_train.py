"""Test the stack helpers in `portc._tensor_train`."""

import numpy as np
import pytest

from portc._tensor_train import (
    SwappedLegs,
    exact_stack,
    normalize_stack,
    pad,
    pad_site,
    padded_shape,
    transpose_mpo,
    unpad,
)


def random_mpo(n_sites, bond, rng, *, up=2, down=2):
    return (
        [rng.normal(size=(bond, up, down))]
        + [rng.normal(size=(bond, bond, up, down)) for _ in range(n_sites - 2)]
        + [rng.normal(size=(bond, up, down))]
    )


def random_mps(n_sites, bond, rng, *, phys=2):
    return (
        [rng.normal(size=(bond, phys))]
        + [rng.normal(size=(bond, bond, phys)) for _ in range(n_sites - 2)]
        + [rng.normal(size=(bond, phys))]
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------
# --- pad / unpad -------
# ------------------------


@pytest.mark.parametrize("kind", ["mps", "mpo"])
def test_pad_shapes_and_round_trip(kind, rng):
    train = random_mps(4, 3, rng) if kind == "mps" else random_mpo(4, 3, rng)

    padded = pad(train)

    down = 1 if kind == "mps" else 2
    assert [t.shape for t in padded] == [
        (1, 3, 2, down),
        (3, 3, 2, down),
        (3, 3, 2, down),
        (3, 1, 2, down),
    ]
    for original, restored in zip(train, unpad(padded, kind)):
        np.testing.assert_array_equal(original, restored)


def test_pad_does_not_copy(rng):
    train = random_mpo(3, 2, rng)

    assert all(np.shares_memory(t, p) for t, p in zip(train, pad(train)))


def test_transpose_mpo_swaps_physical_legs(rng):
    train = random_mpo(3, 2, rng, up=2, down=3)

    transposed = transpose_mpo(train)

    assert [t.shape for t in transposed] == [(2, 3, 2), (2, 2, 3, 2), (2, 3, 2)]
    np.testing.assert_array_equal(transposed[1], train[1].transpose(0, 1, 3, 2))


# ----------------------------
# --- normalize_stack -------
# ----------------------------


def test_normalize_single_trains(rng):
    mps, mpo = random_mps(3, 2, rng), random_mpo(3, 2, rng)

    assert normalize_stack([mps]) == ([mps], "mps")
    assert normalize_stack([mpo]) == ([mpo], "mpo")


def test_normalize_ket_stack_is_unchanged(rng):
    A, B, psi = random_mpo(3, 2, rng), random_mpo(3, 2, rng), random_mps(3, 2, rng)

    assert normalize_stack([A, B]) == ([A, B], "mpo")
    assert normalize_stack([A, B, psi]) == ([A, B, psi], "mps")


def test_normalize_bra_stack_becomes_transposed_ket(rng):
    phi = random_mps(3, 2, rng, phys=2)
    A = random_mpo(3, 2, rng, up=2, down=3)
    B = random_mpo(3, 2, rng, up=3, down=4)

    layers, kind = normalize_stack([phi, A, B])

    assert kind == "mps"
    assert len(layers) == 3
    for got, want in zip(layers[0], transpose_mpo(B)):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(layers[1], transpose_mpo(A)):
        np.testing.assert_array_equal(got, want)
    assert layers[2] is phi


def test_normalize_empty_stack_raises():
    with pytest.raises(ValueError, match="at least one tensor train"):
        normalize_stack([])


def test_normalize_unknown_layout_raises(rng):
    peps_like = [np.zeros((2, 2, 2, 2))] * 3

    with pytest.raises(TypeError, match="layout for train 1"):
        normalize_stack([random_mpo(3, 2, rng), peps_like])


@pytest.mark.parametrize(
    "roles",
    [("mps", "mps"), ("mpo", "mps", "mpo"), ("mps", "mpo", "mps")],
)
def test_normalize_misplaced_mps_raises(roles, rng):
    trains = [
        random_mps(3, 2, rng) if r == "mps" else random_mpo(3, 2, rng) for r in roles
    ]

    with pytest.raises(TypeError, match="MPS may come first"):
        normalize_stack(trains)


def test_normalize_site_count_mismatch_raises(rng):
    with pytest.raises(ValueError, match="same number of sites"):
        normalize_stack([random_mpo(4, 2, rng), random_mps(6, 2, rng)])


def test_normalize_physical_dim_mismatch_raises(rng):
    A = random_mpo(3, 2, rng, up=2, down=3)
    psi = random_mps(3, 2, rng, phys=2)

    with pytest.raises(ValueError, match="site 0 between trains 0 and 1: 3 != 2"):
        normalize_stack([A, psi])


def test_normalize_bra_physical_dim_mismatch_raises(rng):
    phi = random_mps(3, 2, rng, phys=3)
    A = random_mpo(3, 2, rng, up=2, down=2)

    with pytest.raises(ValueError, match="between trains 0 and 1: 3 != 2"):
        normalize_stack([phi, A])


# ------------------------
# --- exact_stack -------
# ------------------------


def dense_two_site(train):
    """Two-site train -> (U, D) matrix; an MPS gives a column (U, 1)."""
    if train[0].ndim == 2:
        return np.einsum("au,av->uv", *train).reshape(-1, 1)
    first, last = train
    t = np.einsum("aij,akl->ikjl", first, last)
    return t.reshape(first.shape[1] * last.shape[1], -1)


@pytest.mark.parametrize("kind", ["mps", "mpo"])
def test_exact_stack_depth_three(kind, rng):
    A, B = random_mpo(2, 3, rng), random_mpo(2, 2, rng)
    last = random_mps(2, 2, rng) if kind == "mps" else random_mpo(2, 2, rng)

    out = exact_stack([A, B, last], chi_out=64, kind=kind)

    want = dense_two_site(A) @ dense_two_site(B) @ dense_two_site(last)
    np.testing.assert_allclose(dense_two_site(out), want, atol=1e-10)


# -------------------------------
# --- padded_shape / pad_site ---
# -------------------------------


@pytest.mark.parametrize("kind", ["mps", "mpo"])
@pytest.mark.parametrize("n_sites", [2, 3, 4])
def test_padded_shape_matches_pad(kind, n_sites, rng):
    train = (
        random_mps(n_sites, 3, rng) if kind == "mps" else random_mpo(n_sites, 3, rng)
    )
    last = n_sites - 1

    shapes = [padded_shape(t.shape, kind, i, last) for i, t in enumerate(train)]

    assert shapes == [t.shape for t in pad(train)]
    for i, site in enumerate(train):
        np.testing.assert_array_equal(pad_site(site, kind, i, last), pad(train)[i])


class LazySite:
    """A site with shape and dtype but no array methods, read by ``np.asarray``."""

    def __init__(self, data):
        self.data = data
        self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.data, dtype=dtype)


def test_transpose_mpo_of_lazy_sites(rng):
    train = random_mpo(3, 2, rng, up=2, down=3)

    transposed = transpose_mpo([LazySite(site) for site in train])

    assert all(isinstance(site, SwappedLegs) for site in transposed)
    assert [t.shape for t in transposed] == [(2, 3, 2), (2, 2, 3, 2), (2, 3, 2)]
    np.testing.assert_array_equal(np.asarray(transposed[1]), train[1].swapaxes(-2, -1))
    assert np.asarray(transposed[0], dtype=np.complex64).dtype == np.complex64

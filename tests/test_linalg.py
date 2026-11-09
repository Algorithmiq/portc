"""Test the decompositions: CholeskyQR over row blocks against Householder QR."""

import numpy as np
import pytest

from portc._group import DeviceGroup, blocks
from portc.utils import cholesky_qr, truncated_qr


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def conditioned(m, n, singular_values, rng):
    """A complex ``m x n`` matrix with the given singular values."""

    def unitary(k):
        z = rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))
        return np.linalg.qr(z)[0]

    U = unitary(n)
    V = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    return (U * singular_values) @ V.conj().T


def split_qr(Y, size, cutoff=0.0):
    """`cholesky_qr` of ``Y`` over ``size`` simulated devices; Q or ``None``."""
    group = DeviceGroup("cpu", size)
    rows = blocks(Y.shape[0], size)

    def fn(rank):
        lo, hi = rows[rank]

        def gram(part):
            parts = group.allgather(rank, part)
            return np.sum(parts.reshape(size, *part.shape), axis=0)

        return cholesky_qr(np.ascontiguousarray(Y[lo:hi].T), Y.shape[0], cutoff, gram)

    results = group.run(fn)
    if any(r is None for r in results):
        assert all(r is None for r in results)
        return None
    return np.concatenate([r.T for r in results])


def projector(Q):
    return Q @ Q.conj().T


@pytest.mark.parametrize("size", [1, 3])
@pytest.mark.parametrize("condition", [1.0, 1e4, 1e8, 1e12])
def test_cholesky_qr_spans_the_householder_range(size, condition, rng):
    S = np.logspace(0, -np.log10(condition), 40)
    Y = conditioned(160, 40, S, rng)

    Q = split_qr(Y, size)

    assert Q.shape == (160, 40)
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(40), atol=1e-13)
    residual = Y - projector(Q) @ Y
    assert np.linalg.norm(residual) < 1e-13 * np.linalg.norm(Y)
    Q_ref = truncated_qr(Y, 0.0)
    np.testing.assert_allclose(projector(Q), projector(Q_ref), atol=1e-6 * condition)


@pytest.mark.parametrize("size", [1, 2])
def test_cholesky_qr_truncates_like_truncated_qr(size, rng):
    S = np.concatenate([np.logspace(0, -2, 25), np.logspace(-6, -9, 15)])
    Y = conditioned(120, 40, S, rng)

    Q = split_qr(Y, size, cutoff=1e-4)

    Q_ref = truncated_qr(Y, 1e-4)
    assert Q.shape == Q_ref.shape == (120, 25)
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(25), atol=1e-13)
    np.testing.assert_allclose(projector(Q), projector(Q_ref), atol=1e-10)


@pytest.mark.parametrize("size", [1, 3])
def test_cholesky_qr_declines_an_exactly_singular_matrix(size, rng):
    Y = conditioned(160, 40, np.ones(40), rng)
    Y[:, 7] = 0

    assert split_qr(Y, size) is None


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("size", [1, 3])
def test_cholesky_qr_declines_or_factors_a_rank_deficient_matrix(size, seed):
    # Whether rounding leaves the null space resolvable depends on the BLAS; either
    # way the result must be exact: declined, or an orthonormal basis of Y.
    S = np.concatenate([np.ones(30), np.zeros(10)])
    Y = conditioned(160, 40, S, np.random.default_rng(seed))

    Q = split_qr(Y, size)

    if Q is not None:
        np.testing.assert_allclose(Q.conj().T @ Q, np.eye(40), atol=1e-13)
        residual = Y - projector(Q) @ Y
        assert np.linalg.norm(residual) < 1e-13 * np.linalg.norm(Y)

"""Linear algebra utilities (numpy / cupy compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from array_api_compat import array_namespace

from ._backend import is_host, solve_triangular, to_numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._backend import Array

# Smallest side for which a device SVD (cuSOLVER gesvd) beats numpy's host gesdd:
# one complex128 n x n SVD on an A100 against OpenBLAS takes 0.023 s against
# 0.015 s at n = 256, 0.058 s against 0.071 s at 512 and 0.63 s against 2.2 s at
# 2048. gesvd also converges on the ill-conditioned factors where gesdd fails.
DEVICE_SVD_MIN = 512

# A plain CholeskyQR pass leaves its output orthonormal to working precision once
# the Gram matrix it factors lies within this Frobenius distance of the identity:
# the condition number of its input is then at most sqrt(1.1 / 0.9).
CHOLESKY_QR_TOL = 0.1


def truncated_qr(matrix: Array, cutoff: float) -> Array:
    """QR with SVD-based rank truncation, returning only the isometry.

    Decomposes ``matrix = Q @ R``, then truncates via SVD on R,
    discarding singular values below ``cutoff * sigma_max``.

    When ``cutoff <= 0`` this falls back to a plain QR (no truncation).

    Args:
        matrix: Input matrix of shape (m, n), on any backend.
        cutoff: Relative singular-value threshold.  Singular values
            satisfying ``s < cutoff * s_max`` are discarded.

    Returns:
        The truncated isometry Q of shape ``(m, rank)``, on the device of
        ``matrix``.
    """
    xp = array_namespace(matrix)
    if cutoff <= 0:
        return xp.linalg.qr(matrix)[0]

    transpose = matrix.shape[0] < matrix.shape[1]
    Q, R = xp.linalg.qr(matrix.T if transpose else matrix)
    U, S, _ = thin_svd(R.T if transpose else R)
    rank = max(1, int((cutoff * S[0] <= S).sum()))
    U = U[:, :rank]
    return U if transpose else Q @ U


def cholesky_qr(
    block: Array, rows: int, cutoff: float, gram: Callable[[Array], Array]
) -> Array | None:
    """`truncated_qr` of a tall matrix split by rows, by shifted CholeskyQR.

    The matrix ``Y`` (``rows x n``, ``rows >= n``) is held in blocks of rows, each
    transposed, possibly on several devices: ``block`` is one of them. The blocks
    only exchange ``n x n`` Gram matrices, through ``gram``. This is shifted
    CholeskyQR3 (Fukaya et al., SIAM J. Sci. Comput. 42, A477 (2020)): a first,
    shifted pass makes ``Y`` well conditioned whenever its condition number is
    below about ``1 / eps``, and at most two plain passes then orthonormalize it.
    The second plain pass is skipped when the first already sees an orthonormal
    input.

    Args:
        block: Some rows of ``Y``, transposed: ``(n, rows_block)``.
        rows: The number of rows of ``Y``.
        cutoff: As in `truncated_qr`.
        gram: Sums the ``(n, n)`` partial Gram matrices of every block. Every
            block's call must return the same bits, so that they all take the same
            decisions.

    Returns:
        The same rows of the isometry ``Q``, transposed: ``(rank, rows_block)``.
        ``None`` if ``Y`` is too ill-conditioned for CholeskyQR, typically
        numerically rank-deficient; `truncated_qr` handles that case. Whether a
        rank-deficient ``Y`` is declined depends on the rounding; a returned
        isometry is accurate either way.
    """
    xp = array_namespace(block)
    n = block.shape[0]
    eye = xp.eye(n, dtype=block.dtype)
    factors = []
    for p in range(3):
        G = gram(xp.conj(block) @ block.T)
        G = (G + xp.conj(G.T)) / 2
        if p == 0:
            # trace(G) bounds ||Y||_2^2 from above, so the shift is safe.
            unit = xp.finfo(block.dtype).eps / 2
            trace = float(xp.real(xp.linalg.trace(G)))
            G = G + 11 * (rows * n + n * (n + 1)) * unit * trace * eye
            last = False
        else:
            last = float(xp.linalg.vector_norm(G - eye)) <= CHOLESKY_QR_TOL
            if p == 2 and not last:
                return None
        try:
            L = xp.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            return None
        # A failed factorization (CuPy does not raise) leaves a pivot <= 0 or NaN.
        if not float(xp.min(xp.real(xp.linalg.diagonal(L)))) > 0:
            return None
        # Y = Q L^H, so the blocks of Q^T are conj(L)^-1 times those of Y^T.
        block = solve_triangular(xp.conj(L), block, lower=True)
        factors.append(L)
        if last:
            break
    if cutoff <= 0:
        return block
    R = xp.conj(factors[0].T)
    for L in factors[1:]:
        R = xp.conj(L.T) @ R
    U, S, _ = thin_svd(R)
    rank = max(1, int((cutoff * S[0] <= S).sum()))
    return U[:, :rank].T @ block


def thin_svd(matrix: Array) -> tuple[Array, np.ndarray, Array]:
    """Thin SVD of a small square-ish factor, where it runs fastest.

    On the host below `DEVICE_SVD_MIN`, on the device of ``matrix`` above it.

    Args:
        matrix: Input matrix of shape (m, n), on any backend.

    Returns:
        ``U`` and ``Vh`` on the backend of ``matrix``, and the singular values
        ``S`` as a host array, for the rank decisions.
    """
    xp = array_namespace(matrix)
    if is_host(xp) or min(matrix.shape) < DEVICE_SVD_MIN:
        U, S, Vh = np.linalg.svd(to_numpy(matrix), full_matrices=False)
        return xp.asarray(U), S, xp.asarray(Vh)
    U, S, Vh = xp.linalg.svd(matrix, full_matrices=False)
    return U, to_numpy(S), Vh

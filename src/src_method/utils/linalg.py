"""Linear algebra utilities (numpy / cupy compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from array_api_compat import array_namespace

from ._backend import is_host, to_numpy

if TYPE_CHECKING:
    from ._backend import Array

# Smallest side for which a device SVD (cuSOLVER gesvd) beats numpy's host gesdd:
# one complex128 n x n SVD on an A100 against OpenBLAS takes 0.023 s against
# 0.015 s at n = 256, 0.058 s against 0.071 s at 512 and 0.63 s against 2.2 s at
# 2048. gesvd also converges on the ill-conditioned factors where gesdd fails.
DEVICE_SVD_MIN = 512


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

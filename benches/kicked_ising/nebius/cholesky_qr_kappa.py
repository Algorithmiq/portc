"""Acceptance and accuracy of `cholesky_qr` against the condition number, per dtype.

``Y = U diag(s) V^H`` with Haar-random ``U`` (``rows x n``) and ``V``, and ``s``
geometric from 1 to ``1 / kappa``. Calls `src_method.utils.linalg.cholesky_qr`
on the host with one block (the Gram exchange is the identity), cutoff 0, and
reports whether it declines and, if not, the orthogonality ``||Q^H Q - I||_F /
sqrt(n)`` and the residual ``||Y - Q Q^H Y||_F / ||Y||_F`` against ``Y`` in
complex128. See ``cholesky_qr.md``.

    python benches/kicked_ising/nebius/cholesky_qr_kappa.py 16384 4096
"""

from __future__ import annotations

import logging
import sys
from time import perf_counter

import numpy as np

from src_method.utils.linalg import cholesky_qr

logger = logging.getLogger(__name__)

KAPPAS = (1e1, 1e2, 1e3, 1e4, 1e6, 1e8, 1e12)


def main(rows: int, n: int) -> None:
    """Sweep ``KAPPAS`` in complex64 and complex128 for a ``rows x n`` sketch."""
    rng = np.random.default_rng(0)
    U = np.linalg.qr(
        rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))
    )[0]
    V = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    dtypes = (np.complex64, np.complex128)
    for dt in dtypes:
        # The first pass shifts the Gram matrix by this multiple of its trace.
        shift = 11 * (rows * n + n * (n + 1)) * np.finfo(dt).eps / 2
        logger.info("%s: shift / trace(G) = %.2e", np.dtype(dt).name, shift)
    for kappa in KAPPAS:
        Y128 = (U * np.geomspace(1, 1 / kappa, n)) @ V.conj().T
        for dt in dtypes:
            t0 = perf_counter()
            Qt = cholesky_qr(Y128.astype(dt).T.copy(), rows, 0.0, lambda g: g)
            seconds = perf_counter() - t0
            name = np.dtype(dt).name
            if Qt is None:
                logger.info("kappa %.0e %s: declined (%.1f s)", kappa, name, seconds)
                continue
            Q = Qt.T.astype(np.complex128)
            orth = np.linalg.norm(Q.conj().T @ Q - np.eye(n)) / np.sqrt(n)
            res = np.linalg.norm(Y128 - Q @ (Q.conj().T @ Y128)) / np.linalg.norm(Y128)
            logger.info(
                "kappa %.0e %s: accepted (%.1f s), orthogonality %.1e, residual %.1e",
                kappa,
                name,
                seconds,
                orth,
                res,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main(int(sys.argv[1]), int(sys.argv[2]))

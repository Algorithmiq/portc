"""Test the batched site kernels and the peak-memory walk."""

import numpy as np
import opt_einsum as oe
import pytest

from portc._kernels import SiteKernels, equations, peak_elements


def cores(rng, dtype, *, left=(3, 2), right=(4, 3), up=2, mid=3, down=2):
    """Two padded layers: (l, r, u, x) and (l, r, x, d)."""

    def draw(*shape):
        out = rng.normal(size=shape)
        if np.issubdtype(dtype, np.complexfloating):
            out = out + 1j * rng.normal(size=shape)
        return out.astype(dtype)

    return (
        draw(left[0], right[0], up, mid),
        draw(left[1], right[1], mid, down),
    )


def batched(fn, n, batch, axis):
    parts = [fn(lo, min(lo + batch, n)) for lo in range(0, n, batch)]
    return np.concatenate(parts, axis=axis)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("batch", [1, 3, 7])
def test_batched_kernels_match_unbatched(dtype, batch):
    rng = np.random.default_rng(0)
    k = SiteKernels(2)
    site = cores(rng, dtype)
    chi, eta = 7, 5
    env = rng.normal(size=(chi, 3, 2)).astype(dtype)
    omega = rng.normal(size=(chi, 2, 2)).astype(dtype)
    proj = rng.normal(size=(eta, 4, 3)).astype(dtype)
    out_core = rng.normal(size=(chi, eta, 2, 2)).astype(dtype)

    full_env = k.env(env, omega, site)
    full_sketch = k.sketch(env, site, proj)
    full_proj = k.project(out_core, site, proj)
    full_first = k.first(site, proj)

    np.testing.assert_allclose(
        batched(lambda lo, hi: k.env(env[lo:hi], omega[lo:hi], site), chi, batch, 0),
        full_env,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        batched(lambda lo, hi: k.sketch(env[lo:hi], site, proj), chi, batch, 3),
        full_sketch,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        batched(lambda lo, hi: k.project(out_core[lo:hi], site, proj), chi, batch, 0),
        full_proj,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        batched(lambda lo, hi: k.first(site, proj[lo:hi]), eta, batch, 2),
        full_first,
        rtol=1e-12,
    )


def test_equations_depth_one():
    eqs = equations(1)

    assert eqs.ltr == "ad,afg,defg->ae"
    assert eqs.first == "defg,be->dbfg"


def test_peak_elements_of_a_matrix_chain():
    # (2x3)(3x4)(4x5): the path contracts the first pair into a 2x4 intermediate,
    # holding its two operands (6 + 12) and the output (8), then the last pair,
    # holding that intermediate twice (as live and as operand), the 4x5 operand
    # and the 2x5 output: 8 + 8 + 20 + 10 = 46.
    shapes = ((2, 3), (3, 4), (4, 5))

    assert peak_elements("ab,bc,cd->ad", shapes) == 46


class FixedPath(oe.paths.PathOptimizer):
    """Always return the same contraction path."""

    def __init__(self, path):
        self.path = path

    def __call__(self, inputs, output, size_dict, memory_limit=None):  # noqa: ARG002
        return self.path


def test_peak_elements_follows_the_optimizer():
    # Contracting (3x4)(4x5) first holds its operands (12 + 20) and a 3x5
    # intermediate (15): 47, one more than the default path's 46.
    shapes = ((2, 3), (3, 4), (4, 5))

    assert peak_elements("ab,bc,cd->ad", shapes, FixedPath([(1, 2), (0, 1)])) == 47

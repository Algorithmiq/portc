"""Compression of a single MPS or MPO.

A thin wrapper around `portc.stack.src` for one-train stacks:

1. MPO randomized compression.
2. MPS randomized compression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._kernels import DEFAULT_OPTIMIZE
from .stack import src

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import DTypeLike
    from opt_einsum.paths import PathOptimizer

    from ._plan import Resources
    from ._tensor_train import Site


def compress(
    tensor: Sequence[Site],
    chi_out: int | np.integer,
    *,
    cutoff: float = 0.0,
    oversample: float | None = 1.0,
    dtype: DTypeLike | None = None,
    seed: int | None = None,
    device: str = "cpu",
    resources: Resources | None = None,
    optimize: str | PathOptimizer = DEFAULT_OPTIMIZE,
) -> list[np.ndarray]:
    """Applies the Successive Randomized Compression (SRC) algorithm.

    Equivalent to ``src(tensor, ...)``; see `portc.stack.src` for the
    conventions. The train type is inferred from the rank of the first site
    tensor:

      1. MPS: `tensor` is an MPS. Results in an MPS.
      2. MPO: `tensor` is an MPO. Results in an MPO.

    Args:
        tensor: The site arrays of the tensor network to compress (MPS or MPO).
        chi_out: The desired maximum bond dimension of the output tensor network.
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
            Discards singular values below ``cutoff * sigma_max``; see
            `portc.stack.src`. Set to 0.0 (default) to disable cutoff
            trimming. Ignored for two sites.
        oversample: Multiplicative sketch-size factor, finite and at least 1.0,
            or ``None`` for the paper's recommendation; see
            `portc.stack.src`. Defaults to 1.0, no oversampling.
        dtype: Data type of the random sketches. Defaults to the promoted
            floating dtype of the inputs; an explicit dtype can promote the result.
        seed: An optional seed for the random number generator.
        device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy).  Requires
            the optional ``cupy`` dependency for GPU execution.
        resources: Memory budgets and scratch space; see `portc.stack.src`.
        optimize: The opt_einsum path optimizer of the contractions, a name or a
            ``PathOptimizer``; see `portc.stack.src`. Defaults to ``"dp"``.

    Returns:
        The site arrays of the compressed tensor network (MPS or MPO), in
        right-canonical form, or left-canonical form when oversampling (three
        sites or more).

    Raises:
        TypeError: If ``chi_out`` is not an integer, the input tensor type is
            unsupported, or ``optimize`` is neither a name nor a ``PathOptimizer``.
        ValueError: If ``chi_out`` is not positive, if ``cutoff`` is not
            in ``[0.0, 1.0)``, if ``oversample`` is not finite and at least 1.0,
            if the train is not open-boundary, if a
            sub-three-site train is not exactly two sites, if ``optimize`` names no
            opt_einsum path optimizer, or if ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    return src(
        tensor,
        chi_out=chi_out,
        cutoff=cutoff,
        oversample=oversample,
        dtype=dtype,
        seed=seed,
        device=device,
        resources=resources,
        optimize=optimize,
    )

"""Successive Randomized Compression of a stack of tensor trains.

A stack ``T_1 . T_2 . ... . T_m`` is an ordered sequence of trains contracted
along their physical legs and compressed in a single SRC sweep. `apply` and
`compress` are the two-train and one-train special cases.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._kernels import DEFAULT_OPTIMIZE
from ._sweep import sweep
from ._tensor_train import (
    MIN_SRC_SITES,
    check_exact_supported,
    exact_stack,
    normalize_stack,
)
from ._validation import (
    sketch_size,
    validate_chi_out,
    validate_cutoff,
    validate_optimize,
)
from .utils import default_rng, get_xp, sketch_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import DTypeLike
    from opt_einsum.paths import PathOptimizer

    from ._plan import Resources
    from ._tensor_train import Site

logger = logging.getLogger(__name__)

LOG_WARN_SMALL = (
    "The current SRC implementation targets tensor networks with 3 or more sites. "
    "Defaulting to an exact SVD-based contraction-compression."
)


def src(
    *trains: Sequence[Site],
    chi_out: int | np.integer,
    cutoff: float = 0.0,
    oversample: float | None = 1.0,
    dtype: DTypeLike | None = None,
    seed: int | None = None,
    device: str = "cpu",
    resources: Resources | None = None,
    optimize: str | PathOptimizer = DEFAULT_OPTIMIZE,
) -> list[np.ndarray]:
    """Contract a stack of tensor trains and compress the result with SRC.

    The stack ``src(T_1, T_2, ..., T_m)`` is the product ``T_1 T_2 ... T_m`` in
    mathematical order: ``T_m`` acts first. Trains are plain lists of per-site
    arrays in the default `quimb` layout, their kind inferred from the rank of the
    first site tensor. Each contraction joins the ``d`` leg of a train with the
    ``u`` (or physical) leg of the next. Accepted stacks:

    - one MPS or one MPO: plain compression;
    - ``A_1, ..., A_k``: an MPO product, giving an MPO;
    - ``A_1, ..., A_k, psi``: an MPO product applied to a ket, giving an MPS;
    - ``phi, A_1, ..., A_k``: a bra times an MPO product, giving an MPS on the
      ``d`` leg of ``A_k``.

    A leading MPS is contracted without conjugation, as the row vector
    ``phi^T A_1 ... A_k``. For the physical bra ``<psi| A_1 ... A_k``, pass
    ``[t.conj() for t in psi]``; the result then pairs with a ket by plain
    contraction.

    The per-site cost grows with the sketch size squared times the product of the
    layer bond dimensions, so one sweep pays off for shallow stacks of thin layers
    (for example two or three Trotter layers). Apply anything else pairwise.

    Args:
        *trains: The site arrays of each train, in mathematical order. A site may
            also be any array-like with ``shape``, ``dtype``, ``ndim`` and
            ``np.asarray`` support (`SiteLike`), such as ``np.memmap`` or a zarr or
            HDF5 dataset; it is then read only when the sweep reaches it.
        chi_out: The desired maximum bond dimension of the output train.
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
            Discards singular values below ``cutoff * sigma_max``: on the
            sketch's QR factor during the right-to-left sweep or, when
            oversampling, in the final left-to-right SVD pass. Set to 0.0
            (default) to disable cutoff trimming. Ignored for two-site stacks,
            which are contracted and truncated to ``chi_out`` exactly.
        oversample: Multiplicative sketch-size factor, finite and at least 1.0.
            The sketch size is ``ceil(oversample * chi_out)``; ``None`` uses the
            paper's ``max(ceil(1.5 * chi_out), chi_out + 10)``. An oversampled
            sweep is followed by an exact SVD truncation to ``chi_out``, which
            improves accuracy at a higher cost. Defaults to 1.0, no
            oversampling. Ignored for two-site stacks.
        dtype: Data type of the random sketches. Defaults to the promoted
            floating dtype of the inputs, so single precision stays single and
            complex inputs get complex Ginibre sketches. An explicit dtype
            overrides this and can promote the result.
        seed: An optional seed for the random number generator.
        device: ``"cpu"`` (default, numpy) or ``"gpu"`` (cupy). Requires
            the optional ``cupy`` dependency for GPU execution.
        resources: Memory budgets and scratch space for the sweep (see
            `Resources`); every budget left unset is detected. Ignored for
            two-site stacks.
        optimize: The opt_einsum path optimizer of the sweep's contractions: any
            name opt_einsum knows (``"dp"``, ``"auto"``, ``"greedy"``, ... or one
            added with ``opt_einsum.paths.register_path_fn``) or an
            ``opt_einsum.paths.PathOptimizer`` instance. The memory plan sizes
            batches for the same paths. Defaults to ``"dp"``, the cheapest paths.
            Ignored for two-site stacks.

    Returns:
        The site arrays of the compressed train (MPS or MPO), as numpy arrays
        (host-side, whatever the ``device``), in right-canonical form, or
        left-canonical form when oversampling (three sites or more).

    Raises:
        TypeError: If ``chi_out`` is not an integer, if a train has an unrecognised
            layout or an MPS sits anywhere other than at one end of the stack,
            or if ``optimize`` is neither a name nor a ``PathOptimizer``.
        ValueError: If ``chi_out`` is not positive, if ``cutoff`` is not in
            ``[0.0, 1.0)``, if ``oversample`` is not finite and at least 1.0,
            if a train is not open-boundary, if the stack is empty,
            if the trains differ in length or in the physical dimensions they join,
            if a sub-three-site stack is not exactly two sites, if ``optimize``
            names no opt_einsum path optimizer, or if ``device`` is not
            recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    chi_out = validate_chi_out(chi_out)
    validate_cutoff(cutoff)
    validate_optimize(optimize)
    chi_sketch = sketch_size(chi_out, oversample)
    get_xp(device)  # Reject an unknown device, or a missing cupy, at once.
    prng = default_rng(seed)
    layers, kind = normalize_stack(trains)

    n_sites = len(layers[0])
    if n_sites < MIN_SRC_SITES:
        check_exact_supported(n_sites)
        logger.warning(LOG_WARN_SMALL)
        return exact_stack(layers, chi_out, kind)

    logger.debug(
        "Starting SRC: n_sites=%d, depth=%d, output=%s, device=%s",
        n_sites,
        len(layers),
        kind,
        device,
    )
    result = sweep(
        layers,
        kind,
        chi_sketch,
        prng,
        device=device,
        cutoff=cutoff,
        truncate_to=chi_out if chi_sketch > chi_out else None,
        dtype=sketch_dtype(dtype, *layers),
        resources=resources,
        optimize=optimize,
    )
    logger.debug("SRC complete")
    return result

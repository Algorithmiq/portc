"""The generic SRC sweep over a stack of tensor trains.

One kernel covers every stack in ket form: ``k`` MPOs, optionally followed by an
MPS. Sites are padded to bulk ``(l, r, u, d)`` views (see
`src_method._tensor_train.pad`), so a single set of einsum equations, generated
from the depth ``k``, serves the boundaries and the bulk alike.

The left-to-right sweep sketches the open physical legs with Gaussian ``omega``
tensors and accumulates the environments ``C``; the sketch index is shared by
every site (a Khatri-Rao sketch). The right-to-left sweep builds the output
through `truncated_qr` while carrying the projected environment ``S``.

Every contraction runs in batches sized by `src_method._plan.make_plan` to the
memory budgets. The cores are read one site at a time (`SiteSource`) and the
environments live on the device, in host memory or on disk (`EnvironmentStore`).
CuPy's pool can fragment beyond the margin the plan leaves for it; a batch that
then runs out of memory is computed in smaller pieces (`_SplittingKernels`).

With several devices (`DeviceGroup`), each owns a contiguous block of the sketch
columns. The left-to-right pass needs no communication: the columns are
independent. At every site of the right-to-left pass, the blocks of the sketch are
gathered on rank 0, which runs the QR and scatters blocks of rows of the output
core; each device projects its rows of the new ``S``, and the rows are gathered on
every device. The random draws, and hence the result, are those of one device.
"""

from __future__ import annotations

import logging
import threading
from math import ceil
from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np
from array_api_compat import array_namespace

from ._group import DeviceGroup, blocks
from ._kernels import DEFAULT_OPTIMIZE, SiteKernels
from ._plan import make_plan, resolve_budgets
from ._sites import SiteSource, padded_shapes, site_bytes, sweep_order
from ._store import EnvironmentStore
from ._tensor_train import unpad
from .utils import (
    device_pool_bytes,
    device_pool_limit,
    gaussian_sketch,
    new_stream,
    release_cached_blocks,
    thin_svd,
    to_numpy,
    truncated_qr,
    use_device,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import DTypeLike
    from opt_einsum.paths import PathOptimizer

    from ._plan import Budgets, Plan, Resources
    from ._tensor_train import Site, TrainKind
    from .utils import Array

logger = logging.getLogger(__name__)


def _ranges(n: int, batch: int) -> Iterator[tuple[int, int]]:
    """Split ``range(n)`` into consecutive ``(lo, hi)`` batches."""
    for lo in range(0, n, batch):
        yield lo, min(lo + batch, n)


class _SplittingKernels:
    """The site kernels, computed in smaller pieces after running out of memory.

    A batch that fails is split along its batch axis and its pieces joined, so
    the batches read from and written to the store stay those of the plan. The
    split of a kernel lasts for the rest of the sweep. One instance per device.

    Args:
        kernels: The kernels of the sweep.
        tag: Prefix of the log messages.
    """

    def __init__(self, kernels: SiteKernels, tag: str) -> None:
        self._kernels, self._tag = kernels, tag
        self._pieces: dict[str, int] = {}

    def env(self, env: Array, omega: Array, cores: Sequence[Array]) -> Array:
        """`SiteKernels.env`, split along the sketch columns."""
        return self._split(
            "env",
            env,
            lambda a, b: self._kernels.env(env[a:b], omega[a:b], cores),
            axis=0,
        )

    def sketch(self, env: Array, cores: Sequence[Array], proj: Array) -> Array:
        """`SiteKernels.sketch`, split along the sketch columns."""
        return self._split(
            "sketch",
            env,
            lambda a, b: self._kernels.sketch(env[a:b], cores, proj),
            axis=-1,
        )

    def project(self, eta: Array, cores: Sequence[Array], proj: Array) -> Array:
        """`SiteKernels.project`, split along the rows."""
        return self._split(
            "project",
            eta,
            lambda a, b: self._kernels.project(eta[a:b], cores, proj),
            axis=0,
        )

    def first(self, cores: Sequence[Array], proj: Array) -> Array:
        """`SiteKernels.first`, split along the rows."""
        return self._split(
            "first",
            proj,
            lambda a, b: self._kernels.first(cores, proj[a:b]),
            axis=-3,
        )

    def _split(
        self,
        kernel: str,
        batched: Array,
        compute: Callable[[int, int], Array],
        *,
        axis: int,
    ) -> Array:
        """Compute ``compute`` over the first axis of ``batched``, in pieces."""
        n = batched.shape[0]
        parts, start = [], 0
        while start < n:
            pieces = self._pieces.get(kernel, 1)
            stop = min(n, start + ceil(n / pieces))
            try:
                parts.append(compute(start, stop))
            except MemoryError:
                if stop - start == 1:
                    raise
                release_cached_blocks(array_namespace(batched))
                self._pieces[kernel] = 2 * pieces
                logger.debug(
                    "%sOut of memory in a %s batch of %d rows; %d pieces from now on",
                    self._tag,
                    kernel,
                    stop - start,
                    2 * pieces,
                )
                continue
            start = stop
        if len(parts) == 1:
            return parts[0]
        return array_namespace(batched).concat(parts, axis=axis)


def sweep(
    layers: Sequence[Sequence[Site]],
    kind: TrainKind,
    chi_out: int,
    prng: np.random.Generator,
    *,
    device: str = "cpu",
    cutoff: float = 0.0,
    truncate_to: int | None = None,
    dtype: DTypeLike,
    resources: Resources | None = None,
    optimize: str | PathOptimizer = DEFAULT_OPTIMIZE,
) -> list[np.ndarray]:
    """Contract and compress a stack in ket form with one SRC sweep.

    Args:
        layers: MPOs, optionally followed by one MPS, all with the same number
            (at least three) of sites and matching physical legs. Sites may be any
            array-likes with ``shape``, ``dtype``, ``ndim`` and ``np.asarray``
            support; each is read only when the sweep reaches it.
        kind: The kind of the contracted train.
        chi_out: The sketch size, which bounds the intermediate bond dimension.
        prng: The generator for the Gaussian sketches, always host-side so that a
            seed gives the same draws on every device.
        device: ``"cpu"`` or ``"gpu"``, as in `src_method.stack.src`.
        cutoff: Relative singular-value cutoff for adaptive bond truncation.
        truncate_to: If set, truncate the right-canonical intermediate to this
            bond dimension with a final left-to-right SVD pass, applying
            ``cutoff`` there instead of to the sketch.
        dtype: The data type of the sketches.
        resources: Memory budgets, scratch space and devices; detected when
            ``None``.
        optimize: The opt_einsum path optimizer of the kernels and of the plan.

    Returns:
        The site arrays of the compressed train as numpy arrays, right-canonical
        unless ``truncate_to`` is set, in which case they are left-canonical.
    """
    shapes = padded_shapes(layers)
    # The dtype of the environments and the output: the sketches promoted by the
    # cores, as the contractions would.
    work = np.result_type(dtype, *(site.dtype for layer in layers for site in layer))
    group = DeviceGroup(device, resources.devices if resources else None)
    budgets = resolve_budgets(resources, group)
    plan = make_plan(
        shapes,
        site_bytes(layers),
        chi_out,
        work,
        budgets,
        devices=group.size,
        optimize=optimize,
    )
    # Guarded: the per-site lists walk the whole plan, wasted work unless logged.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "SRC plan: devices=%s, prefetch=%s, device peak=%d B, host peak=%d B, "
            "disk=%d B, scratch=%s, tiers=%s, batches (env, sketch, project)=%s",
            list(group.devices),
            plan.prefetch,
            plan.device_peak,
            plan.host_peak,
            plan.disk_bytes,
            budgets.scratch_dir if plan.disk_bytes else None,
            [site.tier for site in plan.sites],
            [
                (site.env_batch, site.sketch_batch, site.project_batch)
                for site in plan.sites
            ],
        )
    sweep_ = _Sweep(
        group,
        plan,
        budgets,
        shapes,
        chi_out=chi_out,
        cutoff=cutoff if truncate_to is None else 0.0,
        sketches=_Sketches(prng, chi_out, dtype, group.size),
        work=work,
        depth=len(layers),
        optimize=optimize,
    )
    with SiteSource(
        layers, sweep_order(len(shapes)), depth=plan.prefetch, group=group
    ) as source:
        eta = group.run(lambda rank: sweep_.run(rank, source))[0]
    assert eta is not None  # noqa: S101  (rank 0 returns the cores)
    if truncate_to is not None:
        xp = group.xp
        with (
            use_device(xp, group.devices[0]),
            device_pool_limit(xp, budgets.device_cap),
        ):
            eta = _truncate(eta, truncate_to, cutoff, group)
    return [to_numpy(site) for site in unpad(eta, kind)]


def _truncate(
    sites: list[np.ndarray], chi_out: int, cutoff: float, group: DeviceGroup
) -> list[np.ndarray]:
    """Truncate a padded right-canonical train, staging one centre at a time."""
    xp = group.xp
    centre = xp.asarray(sites[0])
    for j in range(len(sites) - 1):
        left, right, up, down = centre.shape
        matrix = centre.transpose(0, 2, 3, 1).reshape(left * up * down, right)
        # A tall matrix is reduced by QR first, so the SVD sees a small square.
        Q, R = xp.linalg.qr(matrix) if matrix.shape[0] > right else (None, matrix)
        U, S, Vh = thin_svd(R)
        rank = min(chi_out, S.size)
        if cutoff > 0:
            rank = min(rank, max(1, int((cutoff * S[0] <= S).sum())))
        U = U[:, :rank]
        U = U if Q is None else Q @ U
        sites[j] = to_numpy(
            xp.ascontiguousarray(U.reshape(left, up, down, rank).transpose(0, 3, 1, 2))
        )
        centre = xp.tensordot(
            xp.asarray(S[:rank, None]) * Vh[:rank], xp.asarray(sites[j + 1]), axes=1
        )
    sites[-1] = to_numpy(centre)
    return sites


class _Sketches:
    """The Gaussian tensors of the left-to-right pass, shared by the devices.

    Every device asks for the sites in order, and the first one at a site draws its
    tensor, so the draws come in the order of a single device, whatever the
    timing. A tensor is dropped once every device took it.
    """

    def __init__(
        self, prng: np.random.Generator, chi: int, dtype: DTypeLike, ranks: int
    ) -> None:
        self._prng, self._chi, self._dtype, self._ranks = prng, chi, dtype, ranks
        self._drawn: dict[int, tuple[np.ndarray, list[int]]] = {}
        self._lock = threading.Lock()

    def get(self, j: int, up: int, down: int) -> np.ndarray:
        """Return the host tensor ``(chi, up, down)`` of site ``j``."""
        with self._lock:
            if j not in self._drawn:
                omega = gaussian_sketch(self._prng, (self._chi, up, down), self._dtype)
                self._drawn[j] = (omega, [self._ranks])
            omega, left = self._drawn[j]
            left[0] -= 1
            if left[0] == 0:
                del self._drawn[j]
        return omega


class _Sweep:
    """The per-device part of a sweep; `run` executes it on one rank."""

    def __init__(
        self,
        group: DeviceGroup,
        plan: Plan,
        budgets: Budgets,
        shapes: Sequence[tuple[tuple[int, ...], ...]],
        *,
        chi_out: int,
        cutoff: float,
        sketches: _Sketches,
        work: np.dtype,
        depth: int,
        optimize: str | PathOptimizer,
    ) -> None:
        self.group, self.plan, self.budgets, self.shapes = group, plan, budgets, shapes
        self.chi_out, self.cutoff, self.sketches = chi_out, cutoff, sketches
        self.work, self.depth, self.optimize = work, depth, optimize
        self.columns = blocks(chi_out, group.size)

    def _tag(self, rank: int) -> str:
        if self.group.size == 1:
            return ""
        return f"Device {self.group.devices[rank]}: "

    def run(self, rank: int, source: SiteSource) -> list[np.ndarray] | None:
        """Run both passes on ``rank``; rank 0 returns the output cores."""
        xp, tag = self.group.xp, self._tag(rank)
        lo, hi = self.columns[rank]
        env_shapes = [(hi - lo, *(s[0] for s in site)) for site in self.shapes]
        kernels = _SplittingKernels(SiteKernels(self.depth, self.optimize), tag)
        with (
            device_pool_limit(xp, self.budgets.device_cap),
            EnvironmentStore(
                self.plan,
                env_shapes,
                self.work,
                self.budgets.scratch_dir,
                copy_stream=new_stream(xp),
            ) as store,
        ):
            tms = perf_counter_ns()
            self._left_to_right(rank, kernels, source, store)
            logger.debug(
                "%sLeft-to-right sweep: %.3f s", tag, (perf_counter_ns() - tms) * 1e-9
            )
            tms = perf_counter_ns()
            eta = self._right_to_left(rank, kernels, source, store)
            logger.debug(
                "%sRight-to-left sweep: %.3f s", tag, (perf_counter_ns() - tms) * 1e-9
            )
            logger.debug(
                "%sSRC stalls: sites %.3f s, environments %.3f s",
                tag,
                source.stall_seconds[rank],
                store.stall_seconds,
            )
            logger.debug("%sDevice pool: %d B", tag, device_pool_bytes(xp))
        return eta

    def _left_to_right(
        self,
        rank: int,
        kernels: _SplittingKernels,
        source: SiteSource,
        store: EnvironmentStore,
    ) -> None:
        """Build this device's columns of the environments ``C_1 .. C_{n-1}``."""
        xp, plan = self.group.xp, self.plan
        col_lo, col_hi = self.columns[rank]
        n_local = col_hi - col_lo
        first_env = xp.ones((n_local,) + (1,) * self.depth, dtype=self.work)
        for j in range(len(source) - 1):
            cores = source.next(rank)
            up, down = cores[0].shape[2], cores[-1].shape[3]
            omega = xp.asarray(self.sketches.get(j, up, down)[col_lo:col_hi])
            batches = list(_ranges(n_local, plan.sites[j].env_batch))
            if j > 0:
                store.prefetch(j, batches)
            for lo, hi in batches:
                env = first_env[lo:hi] if j == 0 else store.get(j, lo, hi)
                store.put(j + 1, lo, hi, kernels.env(env, omega[lo:hi], cores))

    def _right_to_left(
        self,
        rank: int,
        kernels: _SplittingKernels,
        source: SiteSource,
        store: EnvironmentStore,
    ) -> list[np.ndarray] | None:
        """Build the output cores, host-side on rank 0, from the last site down."""
        xp, plan, group, dtype = self.group.xp, self.plan, self.group, self.work
        root = rank == 0
        col_lo, col_hi = self.columns[rank]
        n_local = col_hi - col_lo
        eta_reversed: list[np.ndarray] = []
        proj = xp.ones((1,) * (self.depth + 1), dtype=dtype)
        for j in range(len(source) - 1, 0, -1):
            cores = source.next(rank)
            site = plan.sites[j]
            batches = list(_ranges(n_local, site.sketch_batch))
            store.prefetch(j, batches)
            up, down = cores[0].shape[2], cores[-1].shape[3]
            # Sketch columns first, so that each device's block is contiguous.
            local = xp.empty((n_local, proj.shape[0], up, down), dtype=dtype)
            for lo, hi in batches:
                part = kernels.sketch(store.get(j, lo, hi), cores, proj)
                local[lo:hi] = xp.moveaxis(part, -1, 0)
            store.drop(j)
            sketch = group.gather(rank, local)
            del local
            eta_full = None
            if root:
                assert sketch is not None  # noqa: S101  (gathered on rank 0)
                rows = proj.shape[0] * up * down
                Q = truncated_qr(
                    xp.reshape(sketch, (self.chi_out, rows)).T, self.cutoff
                )
                del sketch
                eta_full = xp.reshape(
                    xp.ascontiguousarray(Q.T), (-1, proj.shape[0], up, down)
                )
                del Q
            eta_local = group.scatter(rank, eta_full)
            if eta_full is not None:
                eta_reversed.append(to_numpy(eta_full))
                del eta_full
            new_local = xp.empty(
                (eta_local.shape[0], *(c.shape[0] for c in cores)), dtype=dtype
            )
            for lo, hi in _ranges(eta_local.shape[0], site.project_batch):
                new_local[lo:hi] = kernels.project(
                    xp.conj(eta_local[lo:hi]), cores, proj
                )
            del proj, eta_local
            proj = group.allgather(rank, new_local)
            del new_local
        # Every device takes the first site, so that the source can release it.
        cores = source.next(rank)
        if not root:
            return None
        up, down = cores[0].shape[2], cores[-1].shape[3]
        first = xp.empty((proj.shape[0], up, down), dtype=dtype)
        for lo, hi in _ranges(proj.shape[0], plan.sites[0].project_batch):
            first[lo:hi] = xp.reshape(
                kernels.first(cores, proj[lo:hi]), (hi - lo, up, down)
            )
        return [to_numpy(first)[None], *reversed(eta_reversed)]

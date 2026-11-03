"""Heisenberg-picture MPO simulation of IBM's 127-qubit kicked Ising experiment.

The circuit of Kim et al., Nature 618, 500 (2023): every Trotter step applies
``R_X(theta_h)`` to all qubits, then ``R_ZZ(-pi/2)`` on every edge of the Eagle
heavy-hex coupling map, starting from ``|0...0>``. A Pauli observable is evolved
in the Heisenberg picture, ``O <- U_s^dag O U_s`` with the last gate first, as a
matrix product operator over the qubit labels (the "IBM" ordering of Kim et al.'s
Supplementary Information, used by Anand et al., arXiv:2306.17839). Each step's
``R_ZZ`` gates are split into sub-layers with at most one gate across any cut of
the chain, so each sub-layer MPO has bond 2; the ``R_X`` layer is folded, exactly
and site by site, into the last sub-layer of the step. Every group of ``k``
sub-layers is one sandwich ``U_k^dag ... U_1^dag O U_1 ... U_k`` compressed once.
The value is ``<0...0|O|0...0>``; ``2^-n Tr(O^dag O)`` is the truncation
diagnostic (1 under exact evolution, never renormalised).

Commands:

- ``geometry``: coupling map, ordering, sub-layers and their MPO bonds;
- ``check``: dense checks on 12- and 16-qubit fragments, stack order, Clifford
  points against stim, and the paper's Pauli strings;
- ``run``: Heisenberg evolutions for a list of configurations, one JSON line per
  Trotter step;
- ``report``: tables and plots.

``--dtype`` (any command) sets the precision of every array, src-method's and
quimb's alike; ``KI_PLATFORM`` names the output folder (``leonardo`` by default).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import resource
import socket
import subprocess
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any, Literal

import cyclopts
import numpy as np
from cyclopts import Parameter

import src_method
from src_method import Resources, src

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Kicked Ising Heisenberg MPO: src-method against quimb.")

HERE = Path(__file__).parent
OUT = HERE / os.environ.get("KI_PLATFORM", "leonardo")
REFERENCE = HERE / "reference"
N_QUBITS = 127
THETA_J = -np.pi / 2
# Without a cutoff SRC returns bond chi_out wherever it fits, whatever the rank of
# the operator, so the exact checks drop only numerically zero directions.
CHECK_CUTOFF = 1e-12

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": PX, "Y": PY, "Z": PZ}
DType = Literal["complex128", "complex64"]
# Set once per process by `--dtype`, before any array is built.
DTYPE: type[np.complexfloating] = np.complex128
DTYPE_TAG = {"complex128": "c128", "complex64": "c64"}


def set_dtype(name: DType) -> None:
    """Make every gate, observable and diagnostic of this process ``name``."""
    global DTYPE, I2, PX, PY, PZ  # noqa: PLW0603
    DTYPE = np.dtype(name).type
    I2, PX, PY, PZ = (m.astype(DTYPE) for m in (I2, PX, PY, PZ))
    PAULI.update(I=I2, X=PX, Y=PY, Z=PZ)
    if DTYPE is np.complex64:
        # Strict FP32 in cuBLAS and cuSOLVER, as in benches/large/nebius.
        os.environ["CUPY_TF32"] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def dtype_name() -> str:
    """The name of ``--dtype``, as recorded in every output."""
    return np.dtype(DTYPE).name


# Kim et al., Figs. 3b, 3c and 4b; the same strings in Begušić-Gray-Chan, Fig. 3.
OBSERVABLES = {
    "w10": {"X": (13, 29, 31), "Y": (9, 30), "Z": (8, 12, 17, 28, 32)},
    "w17": {
        "X": (37, 41, 52, 56, 57, 58, 62, 79),
        "Y": (75,),
        "Z": (38, 40, 42, 63, 72, 80, 90, 91),
    },
    "z62": {"Z": (62,)},
}
# Kim et al.: the weight-10 and weight-17 strings are U(pi/2)^5 Z_q U(pi/2)^5^dag.
STABILIZER_OF = {"w10": 13, "w17": 58}

SRC_METHODS = tuple(f"src-k{k}{os_}" for k in (1, 2, 3) for os_ in ("", "-os"))
QUIMB_METHODS = ("quimb-fit", "quimb-zipup", "quimb-src", "quimb-src-os")
METHODS = (*SRC_METHODS, *QUIMB_METHODS, "quimb-src-default", "quimb-src-os-default")

Train = list[Any]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@cache
def eagle_edges() -> tuple[tuple[int, int], ...]:
    """The 144 edges of the 127-qubit Eagle coupling map (ibm_kyiv), ``i < j``."""
    from qiskit_ibm_runtime.fake_provider import FakeKyiv  # noqa: PLC0415

    edges = {tuple(sorted(e)) for e in FakeKyiv().coupling_map.get_edges()}
    return tuple(sorted(edges))  # type: ignore[arg-type]


def sublayers(edges: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group gates so that at most one crosses any cut of the chain.

    Gate ``(i, j)`` occupies cuts ``i .. j-1``. First fit by left end colours an
    interval graph optimally, so the count is the largest number of gates over
    one cut.
    """
    groups: list[list[tuple[int, int]]] = []
    ends: list[int] = []
    for i, j in sorted(edges):
        for g, end in enumerate(ends):
            if end <= i:
                groups[g].append((i, j))
                ends[g] = j
                break
        else:
            groups.append([(i, j)])
            ends.append(j)
    return groups


def cut_load(n: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Number of gates over each cut of the chain."""
    load = [0] * (n - 1)
    for i, j in edges:
        for c in range(i, j):
            load[c] += 1
    return load


# ---------------------------------------------------------------------------
# Trains in the quimb layout: MPO (l, r, u, d), boundary sites without the
# outer bond. "Full" form pads the boundaries with unit bonds.
# ---------------------------------------------------------------------------


def full(train: Sequence[Any]) -> Train:
    """Pad the boundary sites with unit outer bonds."""
    return [train[0][None], *train[1:-1], train[-1][:, None]]


def trim(train: Sequence[Any]) -> Train:
    """Drop the unit outer bonds of a full-form train."""
    return [train[0][0], *train[1:-1], train[-1][:, 0]]


def bonds(train: Sequence[Any]) -> list[int]:
    """Bond dimensions of a quimb-layout train, left to right."""
    return [int(t.shape[0]) for t in train[1:]]


def rx(theta: float) -> np.ndarray:
    """``R_X(theta) = exp(-i theta X / 2)``."""
    return (np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * PX).astype(DTYPE)


def pauli_mpo(n: int, string: dict[str, Sequence[int]]) -> Train:
    """A Pauli string as a bond-1 MPO."""
    ops = [I2] * n
    for p, qubits in string.items():
        for q in qubits:
            ops[q] = PAULI[p]
    return trim([op[None, None].copy() for op in ops])


def layer_mpo(
    n: int, gates: Sequence[tuple[int, int]], theta_j: float, after: Any = None
) -> Train:
    """``prod R_ZZ(theta_j)`` over gates that never share a cut, bond 2.

    ``R_ZZ = sum_a L_a (x) R_a`` with ``L = (I, Z)``, ``R = (cos I, -i sin Z)``.
    ``after`` is a one-site unitary applied first in time to every site
    (the ``R_X`` layer), so the train is ``R_ZZ-layer . R_X-layer``.
    """
    owner: list[int | None] = [None] * (n - 1)
    for g, (i, j) in enumerate(gates):
        for c in range(i, j):
            owner[c] = g
    left = (I2, PZ)
    right = (np.cos(theta_j / 2) * I2, -1j * np.sin(theta_j / 2) * PZ)
    sites = []
    for s in range(n):
        lg = owner[s - 1] if s > 0 else None
        rg = owner[s] if s < n - 1 else None
        dl, dr = (1 if lg is None else 2), (1 if rg is None else 2)
        w = np.zeros((dl, dr, 2, 2), dtype=DTYPE)
        if lg is not None and lg == rg:
            w[0, 0] = w[1, 1] = I2
        else:
            for a in range(dl):
                for b in range(dr):
                    w[a, b] = (I2 if lg is None else right[a]) @ (
                        I2 if rg is None else left[b]
                    )
        if after is not None:
            w = np.einsum("lruc,cd->lrud", w, after)
        sites.append(w)
    return trim(sites)


def dagger(train: Sequence[Any]) -> Train:
    """Hermitian conjugate of an MPO: swap ``u`` and ``d`` and conjugate."""
    return [t.swapaxes(-1, -2).conj() for t in train]


def step_layers(
    n: int, edges: Sequence[tuple[int, int]], theta_h: float
) -> list[tuple[Train, Train]]:
    """``(U, U^dag)`` of every sub-layer of one step, in processing order.

    The step is ``U_s = ZZ . R_X`` (``R_X`` first in time), so
    ``U_s^dag O U_s = R_X^dag ZZ^dag O ZZ R_X``; the ZZ sub-layers commute, and
    ``R_X`` is folded into the last one processed.
    """
    groups = sublayers(edges)
    out = []
    for g, gates in enumerate(groups):
        u = layer_mpo(n, gates, THETA_J, rx(theta_h) if g == len(groups) - 1 else None)
        out.append((u, dagger(u)))
    return out


def chunks(layers: Sequence[Any], k: int) -> list[list[Any]]:
    """Consecutive groups of ``k`` sub-layers."""
    return [list(layers[i : i + k]) for i in range(0, len(layers), k)]


# ---------------------------------------------------------------------------
# Dense references (small fragments only)
# ---------------------------------------------------------------------------


def dense_operator(mpo: Sequence[Any]) -> np.ndarray:
    """Matrix of an MPO; rows on ``u``, site 0 most significant."""
    f = full([np.asarray(t) for t in mpo])
    m = np.moveaxis(f[0][0], 0, -1)  # (u, d, r)
    for w in f[1:]:
        t = np.einsum("abr,rsud->aubds", m, w)
        s = t.shape
        m = t.reshape(s[0] * s[1], s[2] * s[3], s[4])
    return m[..., 0]


def kron_site(op: np.ndarray, k: int, n: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2**k), op), np.eye(2 ** (n - k - 1)))


def dense_step(n: int, edges: Sequence[tuple[int, int]], theta_h: float) -> np.ndarray:
    """``U_s = prod R_ZZ . prod R_X`` as a dense matrix (diagonal ZZ part)."""
    z = np.array([1, -1])
    diag = np.ones(2**n, dtype=DTYPE)
    for i, j in edges:
        zz = np.kron(np.ones(2**i), np.kron(z, np.ones(2 ** (n - i - 1)))) * np.kron(
            np.ones(2**j), np.kron(z, np.ones(2 ** (n - j - 1)))
        )
        diag *= np.cos(THETA_J / 2) - 1j * np.sin(THETA_J / 2) * zz
    xlayer = np.array([[1.0]], dtype=DTYPE)
    for _ in range(n):
        xlayer = np.kron(xlayer, rx(theta_h))
    return diag[:, None] * xlayer


def host(train: Sequence[Any]) -> Train:
    """Host copies of NumPy or CuPy sites."""
    return [t.get() if hasattr(t, "get") else np.asarray(t) for t in train]


def zero_expectation(mpo: Sequence[Any]) -> complex:
    """``<0...0|O|0...0>``, on the host."""
    env = np.ones(1, dtype=DTYPE)
    for w in full([np.asarray(t) for t in mpo]):
        env = env @ w[:, :, 0, 0]
    return complex(env[0])


# ---------------------------------------------------------------------------
# GPU helpers: pools, flop counters, diagnostics
# ---------------------------------------------------------------------------


@cache
def _cupy() -> Any:  # noqa: ANN401
    import cupy  # noqa: PLC0415  (GPU only)

    return cupy


@cache
def _analysis_pool() -> Any:  # noqa: ANN401
    return _cupy().cuda.MemoryPool()


@contextmanager
def analysis_pool() -> Iterator[None]:
    """Run diagnostics on a separate pool, so the default one sees the methods."""
    with _cupy().cuda.using_allocator(_analysis_pool().malloc):
        yield


def pool_gb() -> float:
    """The largest default-pool size over the visible GPUs, in GB."""
    cp = _cupy()
    sizes = []
    for gpu in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(gpu):
            sizes.append(cp.get_default_memory_pool().total_bytes())
    return max(sizes) / 1e9


def free_pools() -> None:
    """Release the cached blocks of every GPU's default pool."""
    cp = _cupy()
    for gpu in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(gpu):
            cp.get_default_memory_pool().free_all_blocks()


def sync() -> None:
    cp = _cupy()
    for gpu in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(gpu):
            cp.cuda.Device().synchronize()


FLOPS = {"contract": 0.0, "decomp": 0.0}


def _real_factor(*arrays: Any) -> int:
    """Real flops per multiply-add: 8 complex, 2 real."""
    return 8 if any(a.dtype.kind == "c" for a in arrays) else 2


@cache
def install_flop_counters() -> None:
    """Count flops of every CuPy contraction and decomposition, for all methods.

    Contractions (``tensordot``, ``einsum``, ``matmul``) are counted exactly from
    their operand shapes, decompositions with nominal LAPACK counts (QR
    ``2mn^2 - 2n^3/3``, SVD ``4mn^2 + 22n^3``, eigh ``9n^3`` real flops, times 4
    for complex). Products written with the ``@`` operator are not seen.
    """
    cp = _cupy()
    tensordot, einsum, matmul = cp.tensordot, cp.einsum, cp.matmul
    qr, svd, eigh = cp.linalg.qr, cp.linalg.svd, cp.linalg.eigh

    def counted_tensordot(a: Any, b: Any, axes: Any = 2) -> Any:  # noqa: ANN401
        if isinstance(axes, int):
            ax_a = list(range(a.ndim - axes, a.ndim))
        else:
            ax_a = [axes[0]] if isinstance(axes[0], int) else list(axes[0])
        k = math.prod(a.shape[i] for i in ax_a)
        if k:
            FLOPS["contract"] += _real_factor(a, b) * (a.size // k) * b.size
        return tensordot(a, b, axes)

    def counted_einsum(eq: str, *ops: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if len(ops) >= 2 and "..." not in eq:
            sizes: dict[str, int] = {}
            for term, op in zip(
                eq.split("->", maxsplit=1)[0].split(","), ops, strict=True
            ):
                sizes.update(zip(term.strip(), op.shape, strict=True))
            FLOPS["contract"] += _real_factor(*ops) * math.prod(sizes.values())
        return einsum(eq, *ops, **kwargs)

    def counted_matmul(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        out = matmul(a, b, *args, **kwargs)
        if a.ndim >= 2 and b.ndim >= 2:
            FLOPS["contract"] += _real_factor(a, b) * out.size * a.shape[-1]
        return out

    def counted_qr(a: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        m, n = a.shape[-2:]
        hi, lo = max(m, n), min(m, n)
        batch = a.size // (m * n) if m * n else 0
        f = 4 if a.dtype.kind == "c" else 1
        FLOPS["decomp"] += f * batch * (2 * hi * lo**2 - 2 * lo**3 / 3)
        return qr(a, *args, **kwargs)

    def counted_svd(a: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        m, n = a.shape[-2:]
        hi, lo = max(m, n), min(m, n)
        batch = a.size // (m * n) if m * n else 0
        f = 4 if a.dtype.kind == "c" else 1
        FLOPS["decomp"] += f * batch * (4 * hi * lo**2 + 22 * lo**3)
        return svd(a, *args, **kwargs)

    def counted_eigh(a: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        n = a.shape[-1]
        batch = a.size // (n * n) if n else 0
        f = 4 if a.dtype.kind == "c" else 1
        FLOPS["decomp"] += f * batch * 9 * n**3
        return eigh(a, *args, **kwargs)

    cp.tensordot, cp.einsum, cp.matmul = (
        counted_tensordot,
        counted_einsum,
        counted_matmul,
    )
    cp.linalg.qr, cp.linalg.svd, cp.linalg.eigh = counted_qr, counted_svd, counted_eigh


def diagnostics(mpo: Sequence[Any]) -> tuple[complex, float]:
    """``<0|O|0>`` and ``2^-n Tr(O^dag O)``, on the GPU's analysis pool.

    Streams one site at a time, so the operator never has to fit on the device.
    """
    cp = _cupy()
    saved = dict(FLOPS)
    with analysis_pool():
        env = cp.ones(1, dtype=DTYPE)
        norm = cp.ones((1, 1), dtype=DTYPE)
        for w in full(list(mpo)):
            w = cp.asarray(w)
            env = env @ w[:, :, 0, 0]
            tmp = cp.tensordot(norm, w.conj(), axes=(0, 0))  # (a', r, u, d)
            norm = cp.tensordot(tmp, w, axes=([0, 2, 3], [0, 2, 3])) / 2  # (r, r')
            del w, tmp
        value = complex(env[0])
        frob = float(norm[0, 0].real)
        del env, norm
    _analysis_pool().free_all_blocks()
    FLOPS.update(saved)
    return value, frob


def provenance() -> dict[str, Any]:
    """Versions, job and GPU of this process."""
    import autoray  # noqa: PLC0415
    import opt_einsum  # noqa: PLC0415
    import qiskit  # noqa: PLC0415
    import qiskit_ibm_runtime  # noqa: PLC0415
    import quimb  # noqa: PLC0415
    import stim  # noqa: PLC0415

    cp = _cupy()
    return {
        "src_method": src_method.__version__,
        "quimb": quimb.__version__,
        "autoray": autoray.__version__,
        "opt_einsum": opt_einsum.__version__,
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "gpus_visible": cp.cuda.runtime.getDeviceCount(),
        "qiskit": qiskit.__version__,
        "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
        "stim": stim.__version__,
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=HERE,
            check=False,
        ).stdout.strip(),
        "dtype": dtype_name(),
        "env": {
            v: os.environ.get(v)
            for v in (
                "CUDA_VISIBLE_DEVICES",
                "CUPY_TF32",
                "NVIDIA_TF32_OVERRIDE",
                "CUPY_CUDA_PER_THREAD_DEFAULT_STREAM",
                "CUBLAS_EMULATE_DOUBLE_PRECISION",
                "CUBLAS_EMULATE_SINGLE_PRECISION",
                "CUBLAS_EMULATION_STRATEGY",
                "TMPDIR",
            )
        },
    }


# ---------------------------------------------------------------------------
# Pipelines: one compression per group of sub-layers
# ---------------------------------------------------------------------------


class SrcPipe:
    """src-method: ``src(U_k^dag, ..., U_1^dag, O, U_1, ..., U_k)`` per group.

    Trains go in and come back as host NumPy arrays, as the released API does
    whatever the device; the transfers are part of the timed work.
    """

    def __init__(
        self,
        chi: int,
        k: int,
        oversample: float | None,
        seed: int,
        devices: int = 1,
        cutoff: float = 0.0,
    ) -> None:
        self.chi, self.k, self.seed = chi, k, seed
        self.opts: dict[str, Any] = {
            "device": "gpu",
            "oversample": oversample,
            "cutoff": cutoff,
        }
        if devices > 1:
            self.opts["resources"] = Resources(devices=devices)
        self.calls = 0

    def state(self, arrays: Train) -> Train:
        return [np.asarray(a) for a in arrays]

    def group(self, O: Train, layers: Sequence[tuple[Train, Train]]) -> Train:
        stack = [ud for _, ud in reversed(layers)] + [O] + [u for u, _ in layers]
        self.calls += 1
        return src(
            *stack,
            chi_out=self.chi,
            seed=self.seed * 1_000_003 + self.calls,
            **self.opts,
        )

    def arrays(self, O: Train) -> Train:
        return O


@cache
def _quimb_on_device() -> None:
    """Make quimb 1.15's random tensors follow the network onto the GPU.

    ``src`` draws its sketch noise, and ``fit`` its initial guess, with numpy
    whatever the arrays of the network, which then fails inside CuPy's tensordot.
    The wrappers move those random tensors to the device; the distributions and
    shapes are quimb's own. ``src``'s real noise is cast to the real type of
    ``--dtype``, so that float64 noise does not promote a complex64 network.
    """
    from quimb.tensor.tn1d import compress as qc  # noqa: PLC0415

    cp = _cupy()
    noise, matching = qc._src_get_local_noise_tensors, qc.TN_matching

    def noise_on_device(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        tws = noise(*args, **kwargs)
        for t in tws:
            t.modify(data=cp.asarray(t.data, dtype=np.finfo(DTYPE).dtype))
        return tws

    def matching_on_device(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        tn = matching(*args, **kwargs)
        tn.apply_to_arrays(cp.asarray)
        return tn

    qc._src_get_local_noise_tensors = noise_on_device
    qc.TN_matching = matching_on_device


class QuimbPipe:
    """quimb on the GPU: the lazy sandwich ``U^dag O U`` of one sub-layer, then
    ``tensor_network_1d_compress`` with ``method``, once per sub-layer.

    Every method runs with quimb's defaults except ``cutoff``, which is the same
    value src-method gets. ``src-os`` is quimb's
    ``src-oversample`` (sketch ``max(1.5 chi, chi + 10)``, as src-method's
    ``oversample=None``).
    """

    k = 1

    def __init__(self, method: str, chi: int, seed: int, cutoff: float = 0.0) -> None:
        import quimb as qu  # noqa: PLC0415
        import quimb.tensor as qtn  # noqa: PLC0415

        _quimb_on_device()
        self.qtn, self.chi, self.cutoff = qtn, chi, cutoff
        # quimb's src builds its projectors with tensor_split, whose default
        # relative cutoff of 1e-10 the `cutoff` argument does not reach; it is set
        # to the cutoff every other method gets ("-default" keeps quimb's).
        self.default_split = method.endswith("-default")
        method = method.removesuffix("-default")
        self.method = {"src-os": "src-oversample"}.get(method, method)
        self.extra: dict[str, Any] = {}
        if self.method.startswith("src") and not self.default_split:
            self.extra["project_opts"] = {"cutoff": cutoff}
        qu.seed_rand(seed)
        _cupy().random.seed(seed)

    def state(self, arrays: Train) -> Train:
        cp = _cupy()
        return [cp.asarray(a) for a in arrays]

    def _mpo(self, arrays: Train, upper: str, lower: str) -> Any:  # noqa: ANN401
        return self.qtn.MatrixProductOperator(
            arrays, upper_ind_id=upper, lower_ind_id=lower, site_tag_id="I{}"
        )

    def group(self, O: Train, layers: Sequence[tuple[Train, Train]]) -> Train:
        ((u, ud),) = layers
        cp = _cupy()
        o = self._mpo(O, "k{}", "b{}")
        if self.method == "zipup":
            # quimb's zipup is not exact at full bond (the Chebyshev bench); a
            # left-canonical input makes it converge. The QR sweep is timed.
            o.left_canonize()
        tn = (
            self._mpo([cp.asarray(t) for t in ud], "y{}", "k{}")
            | o
            | self._mpo([cp.asarray(t) for t in u], "b{}", "x{}")
        )
        n = len(O)
        tags = [f"I{i}" for i in range(n)]
        out = self.qtn.tensor_network_1d_compress(
            tn,
            max_bond=self.chi,
            cutoff=self.cutoff,
            method=self.method,
            site_tags=tags,
            **self.extra,
        )
        sites = [out[t] for t in tags]
        arrays = []
        for i, t in enumerate(sites):
            inds = []
            if i > 0:
                inds += list(self.qtn.bonds(sites[i - 1], t))
            if i < n - 1:
                inds += list(self.qtn.bonds(t, sites[i + 1]))
            arrays.append(t.transpose(*inds, f"y{i}", f"x{i}").data)
        return arrays

    def arrays(self, O: Train) -> Train:
        return O


def pipeline(
    method: str, chi: int, seed: int, cutoff: float = 0.0
) -> SrcPipe | QuimbPipe:
    """``src-k{k}[-os][-g{N}]`` or ``quimb-{fit,zipup,src,src-os}``.

    ``cutoff`` is passed unchanged to every method: src-method drops singular
    values below ``cutoff * sigma_max`` of the sketch's QR factor (or in the final
    SVD when oversampling); quimb applies it in its own SVDs (``src`` ignores it).
    """
    if method.startswith("quimb-"):
        return QuimbPipe(method.removeprefix("quimb-"), chi, seed, cutoff)
    parts = method.split("-")
    k = int(parts[1].removeprefix("k"))
    oversample = None if "os" in parts else 1.0
    devices = next((int(p[1:]) for p in parts[2:] if p.startswith("g")), 1)
    return SrcPipe(chi, k, oversample, seed, devices, cutoff)


def evolve(
    pipe: SrcPipe | QuimbPipe,
    O: Train,
    layers: Sequence[tuple[Train, Train]],
    steps: int,
) -> Iterator[Train]:
    """Yield ``O`` after every Trotter step."""
    groups = chunks(layers, pipe.k)
    O = pipe.state(O)
    for _ in range(steps):
        for g in groups:
            O = pipe.group(O, g)
        yield O


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


@cache
def bgc_reference() -> dict[tuple[str, int], dict[int, float]]:
    """Begušić, Gray and Chan's published data, keyed by (observable, steps).

    ``exact.csv`` (exact tensor network contraction): weight-10 (4b) and
    weight-17 (4c) after 5 steps, ``<Z62>`` after 9 steps (5a). ``mix.csv``: their
    converged MIX tensor network ``<Z62>`` after 20 steps (5b, error < 0.01).
    Rows are ``theta_h = k pi / 32``, ``k = 0..16``.
    """
    out: dict[tuple[str, int], dict[int, float]] = {}
    cols = {
        "exact.csv": {"4b": ("w10", 5), "4c": ("w17", 5), "5a": ("z62", 9)},
        "mix.csv": {"5b": ("z62", 20)},
    }
    for name, mapping in cols.items():
        with (REFERENCE / "bgc" / name).open() as fh:
            for row in csv.DictReader(fh):
                k = round(float(row["theta_h"]) * 32 / np.pi)
                for col, key in mapping.items():
                    out.setdefault(key, {})[k] = float(row[col])
    return out


def reference_value(obs: str, steps: int, theta_k: float) -> tuple[float, str] | None:
    """The published reference and its source, if any."""
    if theta_k == int(theta_k):
        table = bgc_reference().get((obs, steps))
        if table is not None and int(theta_k) in table:
            src_name = "BGC MIX" if steps == 20 else "BGC exact"
            return table[int(theta_k)], src_name
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )


@app.command
def geometry(output: Path = OUT / "geometry.json") -> None:
    """Coupling map, ordering, sub-layers and their measured MPO bonds."""
    _setup_logging()
    edges = eagle_edges()
    groups = sublayers(edges)
    load = cut_load(N_QUBITS, edges)
    layers = step_layers(N_QUBITS, edges, 0.3)
    info = {
        "backend": "qiskit_ibm_runtime FakeKyiv (ibm_kyiv, Eagle r3)",
        "n_qubits": N_QUBITS,
        "n_edges": len(edges),
        "ordering": "chain site = qubit label (Kim et al. SI Fig. S9b 'IBM' order)",
        "edges": edges,
        "max_cut_load": max(load),
        "cut_load": load,
        "n_sublayers": len(groups),
        "sublayers": [
            {
                "gates": g,
                "n_gates": len(g),
                "U_max_bond": max(bonds(u)),
                "Udag_max_bond": max(bonds(ud)),
                "cuts_with_bond_2": sum(b == 2 for b in bonds(u)),
            }
            for g, (u, ud) in zip(groups, layers, strict=True)
        ],
        "longest_gate": max(j - i for i, j in edges),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=1))
    logger.info(
        "%d edges, max cut load %d, %d sub-layers of %s gates",
        len(edges),
        max(load),
        len(groups),
        [len(g) for g in groups],
    )


def _fragment(qubits: Sequence[int]) -> tuple[int, list[tuple[int, int]]]:
    """The induced Eagle subgraph on ``qubits``, relabelled in sorted order."""
    label = {q: i for i, q in enumerate(sorted(qubits))}
    edges = [
        (label[i], label[j]) for i, j in eagle_edges() if i in label and j in label
    ]
    return len(qubits), sorted(edges)


def _qiskit_expectation(
    n: int, edges: Sequence[tuple[int, int]], theta_h: float, steps: int, pauli: str
) -> float:
    """``<0|U^dag P U|0>`` with qiskit's own gates, for the convention check.

    ``pauli`` is written site 0 first; qiskit labels are little-endian.
    """
    from qiskit import QuantumCircuit  # noqa: PLC0415
    from qiskit.quantum_info import Pauli, Statevector  # noqa: PLC0415

    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n):
            qc.rx(theta_h, q)
        for i, j in edges:
            qc.rzz(THETA_J, i, j)
    return float(Statevector(qc).expectation_value(Pauli(pauli[::-1])).real)


def _check_dense(seed: int) -> dict[str, Any]:
    """Dense U^dag O U on the 12-qubit heavy hexagon, every method, full bond."""
    rng = np.random.default_rng(seed)
    # 1. Dense operator check: the 12-qubit heavy hexagon, all methods, full bond.
    ring = (0, 1, 2, 3, 4, 14, 15, 18, 19, 20, 21, 22)
    n, edges = _fragment(ring)
    theta = float(rng.uniform(0.2, 1.3))
    steps = 2
    string = "".join(rng.choice(list("IXYZ"), size=n))
    O0 = trim([PAULI[p][None, None].copy() for p in string])
    U = np.eye(2**n, dtype=DTYPE)
    for _ in range(steps):
        U = dense_step(n, edges, theta) @ U
    want = U.conj().T @ dense_operator(O0) @ U
    layers = step_layers(n, edges, theta)
    dense: dict[str, Any] = {
        "qubits": ring,
        "edges": edges,
        "n_sublayers": len(layers),
        "theta_h": theta,
        "steps": steps,
        "pauli": string,
        "qiskit_vs_dense": abs(
            _qiskit_expectation(n, edges, theta, steps, string) - np.real(want[0, 0])
        ),
        "rel_error": {},
    }
    chi_full = 4 ** (n // 2)
    for method in METHODS:
        pipe = pipeline(method, chi_full, seed, CHECK_CUTOFF)
        *_, O = evolve(pipe, O0, layers, steps)
        got = dense_operator(host(pipe.arrays(O)))
        err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        dense["rel_error"][method] = err
        logger.info("dense 12q %s: rel error %.2e", method, err)
    # Stack order: src(B^dag, A^dag, O, A, B) is B^dag A^dag O A B. The last two
    # sub-layers do not commute (R_X is folded into the last), so the reversed
    # stack must give a different operator.
    pair = layers[-2:]
    good = SrcPipe(chi_full, 2, 1.0, seed, cutoff=CHECK_CUTOFF).group(O0, pair)
    a, b = pair
    want_pair = (
        dense_operator(b[1])
        @ dense_operator(a[1])
        @ dense_operator(O0)
        @ dense_operator(a[0])
        @ dense_operator(b[0])
    )
    rev = src(
        a[1],
        b[1],
        O0,
        b[0],
        a[0],
        chi_out=chi_full,
        cutoff=CHECK_CUTOFF,
        device="gpu",
        seed=1,
    )
    dense["stack_order"] = {
        "src(Bd, Ad, O, A, B) vs Bd Ad O A B": float(
            np.linalg.norm(dense_operator(good) - want_pair) / np.linalg.norm(want_pair)
        ),
        "reversed stack vs Bd Ad O A B": float(
            np.linalg.norm(dense_operator(rev) - want_pair) / np.linalg.norm(want_pair)
        ),
    }
    logger.info("stack order: %s", dense["stack_order"])

    return dense


def _check_fragment(seed: int) -> dict[str, Any]:
    """``<0|O|0>`` on a 16-qubit fragment against qiskit, every method."""
    rng = np.random.default_rng(seed + 1)
    # 2. <0|O|0> on a 16-qubit fragment against qiskit, every method, full bond.
    frag = (0, 1, 2, 3, 4, 5, 6, 14, 15, 18, 19, 20, 21, 22, 23, 24)
    n16, edges16 = _fragment(frag)
    theta16 = float(rng.uniform(0.2, 1.3))
    string16 = "".join(rng.choice(list("IZ"), size=n16 - 2)) + "XY"
    O16 = trim([PAULI[p][None, None].copy() for p in string16])
    layers16 = step_layers(n16, edges16, theta16)
    want16 = _qiskit_expectation(n16, edges16, theta16, 3, string16)
    frag16: dict[str, Any] = {
        "qubits": frag,
        "theta_h": theta16,
        "steps": 3,
        "pauli": string16,
        "qiskit": want16,
        "chi": 4096,
        "abs_error": {},
        "max_bond": {},
    }
    for method in METHODS:
        pipe = pipeline(method, 4096, seed, CHECK_CUTOFF)
        *_, O = evolve(pipe, O16, layers16, 3)
        arrays = host(pipe.arrays(O))
        frag16["abs_error"][method] = abs(zero_expectation(arrays) - want16)
        frag16["max_bond"][method] = max(bonds(arrays))
        logger.info("16q %s: error %.2e", method, frag16["abs_error"][method])

    return frag16


@app.command
def check(
    *sections: str,
    cases: str = "",
    output: Path = OUT / "checks.json",
    seed: int = 7,
) -> None:
    """Dense fragments, stack order, Clifford points and the paper's strings.

    ``sections`` is any of ``dense``, ``fragment`` and ``clifford`` (all by
    default); ``cases`` restricts the Clifford cases, e.g. ``w10:5:16,w17:5:16``.
    Each section is merged into ``output`` as soon as it finishes.
    """
    _setup_logging()
    install_flop_counters()
    sections = sections or ("dense", "fragment", "clifford")
    results: dict[str, Any] = json.loads(output.read_text()) if output.exists() else {}
    results["provenance"] = provenance()

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=1, default=str))

    if "dense" in sections:
        results["dense_12q"] = _check_dense(seed)
        save()
    if "fragment" in sections:
        results["fragment_16q"] = _check_fragment(seed)
        save()
    if "clifford" in sections:
        wanted = [tuple(c.split(":")) for c in cases.split(",") if c]
        results.setdefault("clifford", {}).update(clifford_checks(seed, wanted))
        save()


def _stim_step(n: int, edges: Sequence[tuple[int, int]]) -> Any:  # noqa: ANN401
    """One Trotter step at theta_h = pi/2, up to global phase.

    ``R_X(pi/2) ~ SQRT_X`` and ``R_ZZ(-pi/2) = exp(i pi/4 ZZ) ~ SQRT_ZZ_DAG``.
    """
    import stim  # noqa: PLC0415

    c = stim.Circuit()
    c.append("SQRT_X", range(n))
    c.append("SQRT_ZZ_DAG", [q for e in edges for q in e])
    return c


def _pauli_text(string: dict[str, Sequence[int]], n: int) -> str:
    chars = ["_"] * n
    for p, qubits in string.items():
        for q in qubits:
            chars[q] = p
    return "".join(chars)


def clifford_checks(
    seed: int, wanted: Sequence[tuple[str, ...]] = ()
) -> dict[str, Any]:
    """Stim against the MPO pipelines at theta_h = 0 and pi/2."""
    import stim  # noqa: PLC0415

    edges = eagle_edges()
    out: dict[str, Any] = {}
    step = _stim_step(N_QUBITS, edges)

    def circuit(steps: int) -> Any:  # noqa: ANN401
        c = stim.Circuit()
        for _ in range(steps):
            c += step
        return c

    for obs, q in STABILIZER_OF.items():
        z = stim.PauliString(N_QUBITS)
        z[q] = "Z"
        evolved = z.after(circuit(5))  # U Z_q U^dag
        paper = stim.PauliString(_pauli_text(OBSERVABLES[obs], N_QUBITS))
        ex, ez = evolved.to_numpy()
        px, pz = paper.to_numpy()
        out[f"{obs}_is_U5_Z{q}_U5dag"] = {
            "stim": str(evolved),
            "paper": str(paper),
            "matches_up_to_sign": bool((ex == px).all() and (ez == pz).all()),
        }
        logger.info("%s: %s", obs, out[f"{obs}_is_U5_Z{q}_U5dag"])

    # <0|U^dag P U|0> is the sign of U^dag P U if it is Z-type, else 0.
    def stim_value(obs: str, steps: int) -> float:
        p = stim.PauliString(_pauli_text(OBSERVABLES[obs], N_QUBITS)).before(
            circuit(steps)
        )
        xs, _ = p.to_numpy()
        return 0.0 if xs.any() else float(p.sign.real)

    cases = [("z62", 20, 0), ("z62", 20, 16), ("w10", 5, 16), ("w17", 5, 16)]
    if wanted:
        cases = [c for c in cases if tuple(str(x) for x in c) in wanted]
    for obs, steps, theta_k in cases:
        theta = theta_k * np.pi / 32
        layers = step_layers(N_QUBITS, edges, theta)
        want = 1.0 if theta_k == 0 and obs == "z62" else stim_value(obs, steps)
        entry: dict[str, Any] = {"stim": want}
        for method in ("src-k1", "src-k1-os", "src-k2", "src-k3", *QUIMB_METHODS):
            pipe = pipeline(method, 16, seed, CHECK_CUTOFF)
            values, max_bond = [], 0
            for O in evolve(pipe, pauli_mpo(N_QUBITS, OBSERVABLES[obs]), layers, steps):
                arrays = pipe.arrays(O)
                values.append(diagnostics(arrays)[0].real)
                max_bond = max(max_bond, *bonds(arrays))
            entry[method] = {
                "error": abs(values[-1] - want),
                "max_step_error_theta0": (
                    max(abs(v - 1) for v in values) if theta_k == 0 else None
                ),
                "max_bond": max_bond,
            }
            logger.info(
                "clifford %s %d steps k=%d %s: error %.1e, bond %d",
                obs,
                steps,
                theta_k,
                method,
                entry[method]["error"],
                max_bond,
            )
        out[f"{obs}_{steps}steps_k{theta_k}"] = entry
    return out


def _run_path(cfg: str, cutoff: float, *, householder: bool = False) -> Path:
    obs, steps, theta_k, method, chi, seed = cfg.split(":")
    cut = f"_cut{cutoff:g}" if cutoff else ""
    tag = DTYPE_TAG[dtype_name()] + ("_householder" if householder else "")
    name = f"{obs}_n{steps}_k{theta_k}_{method}_chi{chi}{cut}_{tag}_s{seed}.jsonl"
    return OUT / "runs" / name


def _done(path: Path) -> bool:
    """Finished, out of memory, or stopped by the budget after a full job."""
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    return bool(lines) and (
        '"done"' in lines[-1]
        or '"oom"' in lines[-1]
        or '"failed"' in lines[-1]
        or '"fresh": true' in lines[-1]
    )


SVD_FALLBACKS = {"count": 0}


class _Records(logging.Handler):
    """Keep src-method's DEBUG messages (plan, tiers, sweep times)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def record_src_debug() -> _Records:
    """Collect src-method's DEBUG messages from now on."""
    records = _Records()
    src_logger = logging.getLogger("src_method")
    src_logger.setLevel(logging.DEBUG)
    src_logger.addHandler(records)
    return records


_CQR = re.compile(
    r"CholeskyQR at (\d+) sites, (\d+) of them too ill-conditioned for it; "
    r"(\d+) sites skipped"
)


def cqr_counts(messages: Sequence[str]) -> dict[str, int]:
    """Distributed CholeskyQR statistics summed over the compressions logged.

    Rank 0 of a multi-device sweep logs one line per compression, and only when
    it tried CholeskyQR (``chi_out >= DISTRIBUTED_QR_MIN``), so single-device
    runs count zeros.
    """
    out = {"cqr_tried": 0, "cqr_fallbacks": 0, "cqr_skipped": 0}
    for m in messages:
        if match := _CQR.search(m):
            for key, value in zip(out, match.groups(), strict=True):
                out[key] += int(value)
    return out


def host_peak_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def install_svd_fallback() -> None:
    """Retry a non-converging host SVD with LAPACK gesvd (bench-side patch).

    src-method's ``truncated_qr`` and its oversampled truncation call
    ``np.linalg.svd`` (gesdd) on the host; on the ill-conditioned R factors of
    this problem it can fail to converge. The fallback only acts where the
    released code raises, and every use is counted.
    """
    import scipy.linalg  # noqa: PLC0415

    svd = np.linalg.svd

    def svd_with_fallback(a: Any, full_matrices: bool = True, **kwargs: Any) -> Any:  # noqa: ANN401, FBT001, FBT002
        try:
            return svd(a, full_matrices=full_matrices, **kwargs)
        except np.linalg.LinAlgError:
            SVD_FALLBACKS["count"] += 1
            return scipy.linalg.svd(
                a, full_matrices=full_matrices, lapack_driver="gesvd", **kwargs
            )

    np.linalg.svd = svd_with_fallback


def _warmup(methods: set[str], cutoff: float) -> None:
    """One untimed Trotter step of ``<Z62>`` at chi 32 per method.

    The first call of a process pays CuPy kernel compilation, library start-up
    and path planning (about 20 s for src-method, 3 s for quimb), which would
    otherwise land on the first timed step.
    """
    layers = step_layers(N_QUBITS, eagle_edges(), 0.7)
    for method in sorted(methods):
        t0 = perf_counter()
        pipe = pipeline(method, 32, 12345, cutoff)
        for _ in evolve(pipe, pauli_mpo(N_QUBITS, OBSERVABLES["z62"]), layers, 2):
            pass
        sync()
        logger.info("warm-up %s: %.1f s", method, perf_counter() - t0)
    free_pools()


@app.command
def run(
    *configs: str,
    budget_minutes: float = 27.0,
    cutoff: float = 0.0,
    svd_fallback: bool = False,
    householder: bool = False,
) -> None:
    """Run ``obs:steps:theta_k:method:chi:seed`` configurations.

    ``theta_h = theta_k pi / 32``. One JSON line per Trotter step; a finished run
    ends with a ``done`` line and is skipped next time. A run that would cross
    the budget is stopped and marked, and rerun from scratch by a later job.
    ``householder`` disables the distributed CholeskyQR as in `step_probe`, and
    writes to a separate ``_householder`` file.
    """
    from src_method import _sweep  # noqa: PLC0415

    if householder:
        _sweep.DISTRIBUTED_QR_MIN = 2**62
    _setup_logging()
    install_flop_counters()
    if svd_fallback:
        install_svd_fallback()
    records = record_src_debug()
    t_start = perf_counter()
    meta = provenance()
    edges = eagle_edges()
    _warmup(
        {
            c.split(":")[3]
            for c in configs
            if not _done(_run_path(c, cutoff, householder=householder))
        },
        cutoff,
    )
    started = 0
    for cfg in configs:
        path = _run_path(cfg, cutoff, householder=householder)
        if _done(path):
            logger.info("%s: done already", cfg)
            continue
        siblings = path.parent.glob(path.name.rsplit("_s", 1)[0] + "_s*.jsonl")
        if any(_done(q) and '"fresh": true' in q.read_text() for q in siblings):
            logger.info("%s: another seed overran a whole job, skipped", cfg)
            continue
        if (perf_counter() - t_start) / 60 > budget_minutes:
            logger.info("budget reached before %s", cfg)
            break
        obs, steps_s, theta_s, method, chi_s, seed_s = cfg.split(":")
        steps, chi, seed, theta_k = (
            int(steps_s),
            int(chi_s),
            int(seed_s),
            float(theta_s),
        )
        theta = theta_k * np.pi / 32
        layers = step_layers(N_QUBITS, edges, theta)
        pipe = pipeline(method, chi, seed, cutoff)
        path.parent.mkdir(parents=True, exist_ok=True)
        ref = reference_value(obs, steps, theta_k)
        free_pools()
        sync()
        head = {
            "config": cfg,
            "obs": obs,
            "steps": steps,
            "theta_k": theta_k,
            "theta_h": theta,
            "method": method,
            "chi": chi,
            "seed": seed,
            "k": pipe.k,
            "n_sublayers": len(layers),
            "cutoff": cutoff,
            "dtype": dtype_name(),
            "svd_fallback": svd_fallback,
            "householder": householder,
            "reference": None if ref is None else {"value": ref[0], "source": ref[1]},
            "provenance": meta,
        }
        rows = []
        with path.open("w") as fh:
            fh.write(json.dumps({"meta": head}) + "\n")
            FLOPS.update(contract=0.0, decomp=0.0)
            SVD_FALLBACKS["count"] = 0
            t_run = perf_counter()
            fresh = started == 0  # the first run of the job had the whole budget
            started += 1
            t0 = perf_counter()
            stopped = None
            records.messages.clear()
            try:
                for n, O in enumerate(
                    evolve(pipe, pauli_mpo(N_QUBITS, OBSERVABLES[obs]), layers, steps),
                    start=1,
                ):
                    sync()
                    dt = perf_counter() - t0
                    flops = dict(FLOPS)
                    arrays = pipe.arrays(O)
                    value, frob = diagnostics(arrays)
                    row = {
                        "step": n,
                        "time": dt,
                        "value": value.real,
                        "value_imag": value.imag,
                        "frob": frob,
                        "max_bond": max(bonds(arrays)),
                        "pool_gb": pool_gb(),
                        "flops_contract": flops["contract"],
                        "flops_decomp": flops["decomp"],
                        "svd_fallbacks": SVD_FALLBACKS["count"],
                        "host_peak_gb": host_peak_gb(),
                        **cqr_counts(records.messages),
                    }
                    records.messages.clear()
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    if (perf_counter() - t_start) / 60 > budget_minutes and n < steps:
                        stopped = n
                        break
                    t0 = perf_counter()
            except (MemoryError, _cupy().cuda.memory.OutOfMemoryError) as exc:
                fh.write(
                    json.dumps(
                        {"oom": True, "step": len(rows) + 1, "error": str(exc)[:300]}
                    )
                    + "\n"
                )
                logger.info("%s: out of memory at step %d", cfg, len(rows) + 1)
                del pipe
                free_pools()
                continue
            except Exception as exc:  # noqa: BLE001  (record, keep the job going)
                fh.write(
                    json.dumps(
                        {
                            "failed": True,
                            "step": len(rows) + 1,
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        }
                    )
                    + "\n"
                )
                logger.info("%s: failed at step %d: %r", cfg, len(rows) + 1, exc)
                del pipe
                free_pools()
                continue
            if stopped is not None:
                fh.write(json.dumps({"stopped": stopped, "fresh": fresh}) + "\n")
                logger.info("%s: budget reached at step %d", cfg, stopped)
                break
            total = sum(r["time"] for r in rows)
            err = None if ref is None else abs(rows[-1]["value"] - ref[0])
            fh.write(
                json.dumps(
                    {
                        "done": True,
                        "total_time": total,
                        "wall_with_diagnostics": perf_counter() - t_run,
                        "error": err,
                        "peak_gb": max(r["pool_gb"] for r in rows),
                    }
                )
                + "\n"
            )
        logger.info(
            "%s: done, %.1f s, value %.6f, error %s, frob %.4f, bond %d, %.1f GB",
            cfg,
            total,
            rows[-1]["value"],
            "n/a" if err is None else f"{err:.2e}",
            rows[-1]["frob"],
            rows[-1]["max_bond"],
            max(r["pool_gb"] for r in rows),
        )


@app.command
def debug_svd(cfg: str, cutoff: float = 1e-12) -> None:
    """Rerun one configuration and dissect a failing host SVD of src-method.

    src-method's ``truncated_qr`` takes ``np.linalg.svd`` (LAPACK gesdd) of the
    sketch's R factor on the host. On failure the matrix is saved to
    ``leonardo/svd_failure.npy`` and checked for finiteness, conditioning and
    whether gesvd converges on it.
    """
    import scipy.linalg  # noqa: PLC0415

    _setup_logging()
    svd = np.linalg.svd
    report: dict[str, Any] = {"config": cfg, "cutoff": cutoff, "calls": 0}

    def guarded(a: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        report["calls"] += 1
        try:
            return svd(a, *args, **kwargs)
        except np.linalg.LinAlgError:
            np.save(OUT / "svd_failure.npy", a)
            s = scipy.linalg.svd(a, compute_uv=False, lapack_driver="gesvd")
            report.update(
                failed_call=report["calls"],
                shape=a.shape,
                dtype=str(a.dtype),
                finite=bool(np.isfinite(a).all()),
                max_abs=float(np.abs(a).max()),
                gesvd_converged=True,
                s_max=float(s[0]),
                s_min=float(s[-1]),
                n_below_cutoff=int((s < cutoff * s[0]).sum()),
            )
            logger.info("gesdd failed: %s", report)
            raise

    np.linalg.svd = guarded
    obs, steps, theta_k, method, chi, seed = cfg.split(":")
    layers = step_layers(N_QUBITS, eagle_edges(), float(theta_k) * np.pi / 32)
    pipe = pipeline(method, int(chi), int(seed), cutoff)
    try:
        for n, _ in enumerate(
            evolve(pipe, pauli_mpo(N_QUBITS, OBSERVABLES[obs]), layers, int(steps)), 1
        ):
            logger.info("step %d ok", n)
    except np.linalg.LinAlgError as exc:
        report["error"] = str(exc)
    (OUT / "svd_failure.json").write_text(json.dumps(report, indent=1, default=str))


def exact_sandwich(O: Train, u: Train, ud: Train) -> Train:
    """``U^dag O U`` exactly, bonds multiplied, as host arrays."""
    out = []
    for a, o, b in zip(full(ud), full(host(O)), full(u), strict=True):
        t = np.einsum("abuc,lrce,xyed->alxbryud", a, o, b)
        s_ = t.shape
        out.append(t.reshape(s_[0] * s_[1] * s_[2], s_[3] * s_[4] * s_[5], 2, 2))
    return trim(out)


def mpo_inner(a: Sequence[Any], b: Sequence[Any]) -> complex:
    """Hilbert-Schmidt ``Tr(A^dag B)``, on the GPU's analysis pool."""
    cp = _cupy()
    with analysis_pool():
        env = cp.ones((1, 1), dtype=DTYPE)
        for x, y in zip(full(list(a)), full(list(b)), strict=True):
            x, y = cp.asarray(x), cp.asarray(y)
            tmp = cp.tensordot(env, x.conj(), axes=(0, 0))  # (b, a', u, d)
            env = cp.tensordot(tmp, y, axes=([0, 2, 3], [0, 2, 3]))  # (a', b')
        value = complex(env[0, 0])
        del env, tmp
    _analysis_pool().free_all_blocks()
    return value


def svd_truncate(train: Sequence[Any], chi: int) -> Train:
    """Optimal (canonical SVD) truncation of an MPO to bond ``chi``, on the GPU."""
    cp = _cupy()
    with analysis_pool():
        sites = [cp.asarray(t) for t in full(list(train))]
        for j in range(len(sites) - 1):  # left-canonical by QR
            l_, r_, u_, d_ = sites[j].shape
            Q, R = cp.linalg.qr(sites[j].transpose(0, 2, 3, 1).reshape(-1, r_))
            sites[j] = Q.reshape(l_, u_, d_, -1).transpose(0, 3, 1, 2)
            sites[j + 1] = cp.tensordot(R, sites[j + 1], axes=(1, 0))
        for j in range(len(sites) - 1, 0, -1):  # truncate right to left
            l_, r_, u_, d_ = sites[j].shape
            U, S, Vh = cp.linalg.svd(sites[j].reshape(l_, -1), full_matrices=False)
            k = min(chi, S.size)
            sites[j] = Vh[:k].reshape(k, r_, u_, d_)
            sites[j - 1] = cp.tensordot(sites[j - 1], U[:, :k] * S[:k], axes=(1, 0))
            sites[j - 1] = sites[j - 1].transpose(0, 3, 1, 2)
        out = [t.get() for t in sites]
    _analysis_pool().free_all_blocks()
    return trim(out)


@app.command
def sandwich_test(
    problem: str = "z62:8:6",
    chi: int = 256,
    seeds: int = 5,
    sublayer: int = 0,
    cutoff: float = 1e-12,
    output: Path = OUT / "sandwich_test.json",
) -> None:
    """Truncation error of one compression of ``U^dag O U``, method by method.

    ``O`` is the observable of ``problem`` (``obs:steps:theta_k``) evolved by
    src-k1 at bond ``chi``; ``U`` is sub-layer ``sublayer`` of the next step. The
    exact sandwich (bond up to ``4 chi``) is compressed back to ``chi`` by each
    method, and the relative Frobenius error and the change of ``<0|.|0>`` are
    compared with the optimal SVD truncation.
    """
    _setup_logging()
    install_flop_counters()
    install_svd_fallback()
    obs, steps, theta_k = problem.split(":")
    layers = step_layers(N_QUBITS, eagle_edges(), float(theta_k) * np.pi / 32)
    pipe = pipeline("src-k1", chi, 0, cutoff)
    *_, O = evolve(pipe, pauli_mpo(N_QUBITS, OBSERVABLES[obs]), layers, int(steps))
    u, ud = layers[sublayer]
    exact = exact_sandwich(O, u, ud)
    norm2 = mpo_inner(exact, exact).real
    value = zero_expectation(exact)
    result: dict[str, Any] = {
        "problem": problem,
        "chi": chi,
        "sublayer": sublayer,
        "cutoff": cutoff,
        "dtype": dtype_name(),
        "input_bond": max(bonds(host(O))),
        "exact_bond": max(bonds(exact)),
        "exact_value": value.real,
        "methods": {},
    }

    def score(t: Train, seconds: float) -> dict[str, float]:
        t = host(t)
        err2 = norm2 + mpo_inner(t, t).real - 2 * mpo_inner(exact, t).real
        return {
            "rel_frob_error": float(np.sqrt(max(err2, 0.0) / norm2)),
            "value_error": abs(zero_expectation(t) - value),
            "bond": max(bonds(t)),
            "seconds": seconds,
        }

    t0 = perf_counter()
    result["methods"]["svd-optimal"] = [
        score(svd_truncate(exact, chi), perf_counter() - t0)
    ]
    logger.info("svd-optimal: %s", result["methods"]["svd-optimal"])
    for method in (
        "src-k1",
        "src-k1-os",
        "quimb-src",
        "quimb-src-os",
        "quimb-fit",
        "quimb-zipup",
    ):
        rows = []
        for seed in range(1 if method == "quimb-zipup" else seeds):
            p_ = pipeline(method, chi, 100 + seed, cutoff)
            sync()
            t0 = perf_counter()
            out = p_.group(p_.state(O), [(u, ud)])
            sync()
            rows.append(score(out, perf_counter() - t0))
        result["methods"][method] = rows
        logger.info(
            "%s: rel error %s, value error %s",
            method,
            [f"{r['rel_frob_error']:.3e}" for r in rows],
            [f"{r['value_error']:.1e}" for r in rows],
        )
    result["svd_fallbacks"] = SVD_FALLBACKS["count"]
    output.write_text(json.dumps(result, indent=1))


def random_operator(n: int, chi: int, seed: int) -> Train:
    """A random MPO with the saturated bond profile ``min(chi, 4^(distance to edge))``.

    Entries are complex Gaussian, scaled so each site has unit Frobenius norm per
    bond row; it stands in for a late-time operator, whose cost per step depends
    only on the bonds.
    """
    rng = np.random.default_rng(seed)
    cuts = [min(chi, 4 ** (c + 1), 4 ** (n - 1 - c)) for c in range(n - 1)]
    sites = []
    for j in range(n):
        left = 1 if j == 0 else cuts[j - 1]
        right = 1 if j == n - 1 else cuts[j]
        shape = (left, right, 2, 2)
        w = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        w *= 1 / np.sqrt(2 * 4 * left)
        sites.append(w.astype(DTYPE))
    return trim(sites)


@app.command
def step_probe(
    chi: int,
    method: str = "src-k1",
    theta_k: float = 8,
    cutoff: float = 1e-12,
    budget_minutes: float = 26.0,
    seed: int = 0,
    *,
    householder: bool = False,
) -> None:
    """Time one Trotter step of a saturated operator at bond ``chi``.

    The compressions of the step's sub-layer groups run in order on a random
    operator of bond ``chi`` (see `random_operator`), with src-method's DEBUG
    plan recorded (devices, memory tiers, batches, sweep times, CholeskyQR
    counts). Writes ``{platform}/probe/{method}_chi{chi}_cut{cutoff}_{dtype}.json``.
    ``householder`` raises ``src_method._sweep.DISTRIBUTED_QR_MIN`` beyond any
    sketch, so multi-device sweeps gather every sketch for a Householder QR on
    rank 0 (bench-side patch, for the A/B against the distributed CholeskyQR).
    """
    from src_method import _sweep  # noqa: PLC0415

    _setup_logging()
    install_flop_counters()
    install_svd_fallback()
    records = record_src_debug()
    if householder:
        _sweep.DISTRIBUTED_QR_MIN = 2**62
    t_start = perf_counter()
    _warmup({method}, cutoff)
    layers = step_layers(N_QUBITS, eagle_edges(), theta_k * np.pi / 32)
    pipe = pipeline(method, chi, seed, cutoff)
    t0 = perf_counter()
    O = pipe.state(random_operator(N_QUBITS, chi, seed))
    build = perf_counter() - t0
    qr = "_householder" if householder else ""
    tag = DTYPE_TAG[dtype_name()]
    out_path = OUT / "probe" / f"{method}_chi{chi}_cut{cutoff:g}_{tag}{qr}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "chi": chi,
        "method": method,
        "k": pipe.k,
        "theta_k": theta_k,
        "cutoff": cutoff,
        "dtype": dtype_name(),
        "distributed_qr_min": _sweep.DISTRIBUTED_QR_MIN,
        "input_bond": max(bonds(O)),
        "input_gb": sum(t.nbytes for t in O) / 1e9,
        "build_seconds": build,
        "provenance": provenance(),
        "groups": [],
        "complete": False,
    }

    def save() -> None:
        result["host_peak_gb"] = host_peak_gb()
        out_path.write_text(json.dumps(result, indent=1, default=str))

    groups = chunks(layers, pipe.k)
    for g, group in enumerate(groups):
        if (perf_counter() - t_start) / 60 > budget_minutes:
            break
        free_pools()
        FLOPS.update(contract=0.0, decomp=0.0)
        records.messages.clear()
        sync()
        t0 = perf_counter()
        try:
            O = pipe.group(O, group)
        except (MemoryError, _cupy().cuda.memory.OutOfMemoryError) as exc:
            result["oom"] = {"group": g, "error": str(exc)[:300]}
            save()
            logger.info("chi %d %s: out of memory in group %d", chi, method, g)
            return
        sync()
        dt = perf_counter() - t0
        flops = FLOPS["contract"] + FLOPS["decomp"]
        result["groups"].append(
            {
                "group": g,
                "seconds": dt,
                "tflop": flops / 1e12,
                "tflops_rate": flops / dt / 1e12,
                "pool_gb": pool_gb(),
                "out_bond": max(bonds(O)),
                "svd_fallbacks": SVD_FALLBACKS["count"],
                "host_peak_gb": host_peak_gb(),
                **cqr_counts(records.messages),
                "src_debug": [
                    m
                    for m in records.messages
                    if "plan" in m or "sweep" in m or "stalls" in m or "Cholesky" in m
                ],
            }
        )
        logger.info(
            "chi %d %s group %d/%d: %.1f s, %.1f TFLOP, %.2f TFLOP/s, %.1f GB pool",
            chi,
            method,
            g + 1,
            len(groups),
            dt,
            flops / 1e12,
            flops / dt / 1e12,
            pool_gb(),
        )
        save()
    result["complete"] = len(result["groups"]) == len(groups)
    result["step_seconds"] = sum(r["seconds"] for r in result["groups"])
    save()


@app.command
def svd_bench(
    sizes: tuple[int, ...] = (256, 512, 1024, 2048, 4096),
    repeats: int = 3,
    output: Path = OUT / "svd_bench.json",
) -> None:
    """Time one square complex SVD on the host against the GPU.

    The matrix stands in for src-method's R factor: complex Gaussian factors
    around singular values decaying geometrically from 1 to 1e-14. Host: numpy
    (gesdd) and scipy gesvd at 8 threads and at every core; GPU: CuPy's svd
    (cuSOLVER gesvd), Jacobi gesvdj, and eigh of R^dag R (whose squared
    condition number loses the small singular values). GPU times include the
    transfers a host-resident R would need.
    """
    import scipy.linalg  # noqa: PLC0415
    from cupyx import cusolver  # noqa: PLC0415
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    _setup_logging()
    cp = _cupy()
    rng = np.random.default_rng(0)
    cores = len(os.sched_getaffinity(0))
    result: dict[str, Any] = {"cores": cores, "provenance": provenance(), "sizes": {}}

    def timed(fn: Any) -> float:  # noqa: ANN401
        times = []
        for _ in range(repeats):
            sync()
            t0 = perf_counter()
            fn()
            sync()
            times.append(perf_counter() - t0)
        return float(median(times))

    for n in sizes:

        def unitary() -> np.ndarray:
            z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            return np.linalg.qr(z)[0]

        s_ = np.logspace(0, -14, n)
        R = (unitary() * s_) @ unitary()
        row: dict[str, float] = {}
        for threads in sorted({8, cores}):
            with threadpool_limits(threads):
                row[f"numpy gesdd, {threads} threads"] = timed(
                    lambda: np.linalg.svd(R, full_matrices=False)
                )
                row[f"scipy gesvd, {threads} threads"] = timed(
                    lambda: scipy.linalg.svd(
                        R, full_matrices=False, lapack_driver="gesvd"
                    )
                )
        row["cupy svd (gesvd)"] = timed(
            lambda: [x.get() for x in cp.linalg.svd(cp.asarray(R), full_matrices=False)]
        )
        row["cupyx gesvdj"] = timed(
            lambda: [
                x.get() for x in cusolver.gesvdj(cp.asarray(R), full_matrices=False)
            ]
        )

        def eigh_route() -> None:
            r = cp.asarray(R)
            w, v = cp.linalg.eigh(r.conj().T @ r)
            (r @ v).get()
            w.get()

        row["cupy eigh(R^dag R)"] = timed(eigh_route)
        result["sizes"][n] = row
        logger.info("n=%d: %s", n, {k: round(v, 3) for k, v in row.items()})
        output.write_text(json.dumps(result, indent=1, default=str))


def load_runs(folder: Path = OUT / "runs") -> list[dict[str, Any]]:
    """Every run file: its meta, step rows and ending (done, OOM or stopped)."""
    runs = []
    for path in sorted(folder.glob("*.jsonl")):
        run_: dict[str, Any] = {"rows": [], "status": "partial", "file": path.name}
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" in d:
                run_.update(d["meta"])
            elif "done" in d:
                run_.update(
                    status="done", **{k: v for k, v in d.items() if k != "done"}
                )
            elif "oom" in d:
                run_.update(status="oom", oom_step=d["step"])
            elif "failed" in d:
                run_.update(status="failed", fail_step=d["step"], fail=d["error"])
            elif "stopped" in d:
                run_["status"] = "stopped"
            else:
                run_["rows"].append(d)
        if "config" in run_:
            run_.setdefault("dtype", "complex128")
            if run_.get("cutoff"):
                run_["method"] += f" cut {run_['cutoff']:g}"
            if run_.get("svd_fallback"):
                run_["method"] += " +gesvd"
            if run_.get("householder"):
                run_["method"] += " householder"
            # complex128 keeps the Leonardo labels.
            run_["method"] += {"complex64": " c64"}.get(run_["dtype"], "")
            runs.append(run_)
    return runs


COLORS = {
    "src-k1": "#1f77b4",
    "src-k1-os": "#6baed6",
    "src-k2": "#2ca02c",
    "src-k2-os": "#98df8a",
    "src-k3": "#9467bd",
    "src-k3-os": "#c5b0d5",
    "quimb-fit": "#d62728",
    "quimb-zipup": "#ff7f0e",
    "quimb-src": "#8c564b",
    "quimb-src-os": "#e377c2",
}


def _methods(runs: Sequence[dict[str, Any]]) -> list[str]:
    """Methods present, in the order of METHODS, cutoff variants after each."""
    present = {r["method"] for r in runs}
    return sorted(present, key=lambda m: (METHODS.index(m.split()[0]), m))


def _fmt(x: float | None, spec: str = ".2e") -> str:
    return "-" if x is None else format(x, spec)


def _summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Seed statistics of one (problem, method, chi) cell."""
    done = [r for r in runs if r["status"] == "done"]
    rows = [r for r in runs if r["rows"]]
    cell: dict[str, Any] = {"n": len(done), "n_runs": len(runs)}
    if rows:
        cell["step_time"] = median(x["time"] for r in rows for x in r["rows"])
        cell["peak_gb"] = max(x["pool_gb"] for r in rows for x in r["rows"])
        flops = [
            (x["flops_contract"] + x["flops_decomp"], sum(y["time"] for y in r["rows"]))
            for r in rows
            for x in r["rows"][-1:]
        ]
        cell["tflop"] = float(np.mean([f for f, _ in flops])) / 1e12
        cell["tflops_rate"] = float(np.mean([f / t for f, t in flops if t])) / 1e12
        cell["decomp_share"] = float(
            np.mean(
                [
                    r["rows"][-1]["flops_decomp"]
                    / max(
                        r["rows"][-1]["flops_decomp"] + r["rows"][-1]["flops_contract"],
                        1,
                    )
                    for r in rows
                ]
            )
        )
    if done:
        cell["total"] = float(np.mean([r["total_time"] for r in done]))
        errs = [r["error"] for r in done if r.get("error") is not None]
        if errs:
            cell["err"] = float(np.mean(errs))
            cell["err_min"], cell["err_max"] = min(errs), max(errs)
        cell["frob"] = float(np.mean([r["rows"][-1]["frob"] for r in done]))
        cell["value"] = float(np.mean([r["rows"][-1]["value"] for r in done]))
        cell["bond"] = max(r["rows"][-1]["max_bond"] for r in done)
    if any(r["status"] == "failed" for r in runs):
        cell["failed"] = sum(r["status"] == "failed" for r in runs)
    if any(r["status"] == "oom" for r in runs):
        cell["oom"] = min(r["oom_step"] for r in runs if r["status"] == "oom")
    return cell


@app.command
def report(folder: Path = OUT) -> None:
    """Write ``summary.md`` and the plots from the run files."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    runs = load_runs(folder / "runs")
    lines = ["# Kicked Ising: Heisenberg MPO, src-method against quimb", ""]
    if runs:
        lines += [
            "## Provenance",
            "",
            "```json",
            json.dumps(runs[-1]["provenance"], indent=1),
            "```",
            "",
        ]
    geo = folder / "geometry.json"
    if geo.exists():
        g = json.loads(geo.read_text())
        lines += [
            "## Geometry",
            "",
            f"{g['backend']}: {g['n_qubits']} qubits, {g['n_edges']} edges. "
            f"Ordering: {g['ordering']}. The largest number of gates over one cut "
            f"is {g['max_cut_load']}, so {g['n_sublayers']} sub-layers with at most "
            "one gate per cut is optimal; Anand et al. used 13. Longest gate: "
            f"{g['longest_gate']} sites.",
            "",
            "| sub-layer | gates | U bond | U^dag bond | cuts at bond 2 |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {i} | {s['n_gates']} | {s['U_max_bond']} | {s['Udag_max_bond']} "
            f"| {s['cuts_with_bond_2']} |"
            for i, s in enumerate(g["sublayers"])
        ]
        lines.append("")
    chk = folder / "checks.json"
    if chk.exists():
        c = json.loads(chk.read_text())
        lines += _check_lines(c)

    lines += _failure_lines(runs, load_runs(folder / "runs_failed"))
    lines += _stage1_lines(runs)
    lines += _sandwich_lines(folder)
    lines += _probe_lines(folder)

    # One section per stage-2 problem: (obs, steps, theta_k).
    problems = sorted(
        {(r["obs"], r["steps"], r["theta_k"]) for r in runs} - _stage1_only(runs)
    )
    for obs, steps, theta_k in problems:
        sel = [
            r
            for r in runs
            if (r["obs"], r["steps"], r["theta_k"]) == (obs, steps, theta_k)
        ]
        ref = sel[0]["reference"]
        lines += [
            f"## {obs}, {steps} steps, theta_h = {theta_k:g} pi/32 = {theta_k * np.pi / 32:.4f}",
            "",
            "Reference: "
            + ("none" if ref is None else f"{ref['value']:.10g} ({ref['source']})"),
            "",
            "| method | chi | runs done | s / step | total s | abs error (min-max) "
            "| 2^-n Tr O^dag O | bond | peak GB | TFLOP | TFLOP/s | decomp share |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for method in _methods(sel):
            for chi in sorted({r["chi"] for r in sel if r["method"] == method}):
                cell = _summary(
                    [r for r in sel if r["method"] == method and r["chi"] == chi]
                )
                err = (
                    f"{_fmt(cell.get('err'))} ({_fmt(cell.get('err_min'))}-{_fmt(cell.get('err_max'))})"
                    if "err" in cell
                    else "-"
                )
                if "oom" in cell:
                    err += f" (OOM at step {cell['oom']})"
                if "failed" in cell:
                    err += f" ({cell['failed']} failed)"
                lines.append(
                    f"| {method} | {chi} | {cell['n']}/{cell['n_runs']} "
                    f"| {_fmt(cell.get('step_time'), '.3f')} | {_fmt(cell.get('total'), '.1f')} "
                    f"| {err} | {_fmt(cell.get('frob'), '.4f')} | {cell.get('bond', '-')} "
                    f"| {_fmt(cell.get('peak_gb'), '.1f')} | {_fmt(cell.get('tflop'), '.1f')} "
                    f"| {_fmt(cell.get('tflops_rate'), '.2f')} | {_fmt(cell.get('decomp_share'), '.2f')} |"
                )
        lines.append("")

    # Time against error, one panel per problem with a reference.
    with_ref = [
        p
        for p in problems
        if any(
            r["reference"] for r in runs if (r["obs"], r["steps"], r["theta_k"]) == p
        )
    ]
    if with_ref:
        cols = min(3, len(with_ref))
        rows_n = math.ceil(len(with_ref) / cols)
        fig, axes = plt.subplots(
            rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False
        )
        for ax, (obs, steps, theta_k) in zip(axes.flat, with_ref, strict=False):
            sel = [
                r
                for r in runs
                if (r["obs"], r["steps"], r["theta_k"]) == (obs, steps, theta_k)
            ]
            for method in _methods(sel):
                pts = []
                for chi in sorted({r["chi"] for r in sel if r["method"] == method}):
                    cell = _summary(
                        [r for r in sel if r["method"] == method and r["chi"] == chi]
                    )
                    if "err" in cell and "total" in cell:
                        pts.append(
                            (
                                cell["total"],
                                cell["err"],
                                cell["err_min"],
                                cell["err_max"],
                                chi,
                            )
                        )
                if not pts:
                    continue
                t, e, lo, hi, chis = map(np.array, zip(*pts, strict=True))
                ax.errorbar(
                    t,
                    np.maximum(e, 1e-12),
                    yerr=[e - lo, hi - e],
                    marker="o",
                    color=COLORS.get(method.split()[0]),
                    label=method,
                    capsize=2,
                    ls=":"
                    if " cut " in method
                    else "--"
                    if method.startswith("quimb")
                    else "-",
                )
                for ti, ei, c in zip(t, e, chis, strict=True):
                    ax.annotate(str(c), (ti, max(ei, 1e-12)), fontsize=6)
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="total time (s)",
                ylabel="abs error",
                title=f"{obs}, {steps} steps, theta_h = {theta_k:g} pi/32",
            )
            ax.grid(True, which="both", alpha=0.3)
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / "time_vs_error.png", dpi=130)
        lines += [
            "## Time against error",
            "",
            "![time vs error](time_vs_error.png)",
            "",
        ]

    conclusion = folder / "conclusion.md"
    if conclusion.exists():
        lines += [conclusion.read_text().rstrip(), ""]
    (folder / "summary.md").write_text("\n".join(lines))


STAGE2_THETAS = (6.0, 8.0, 10.0)


def _stage1_only(runs: Sequence[dict[str, Any]]) -> set[tuple[str, int, float]]:
    """Problems run only for stage 1: weight-10, and weight-17 off the stage-2 angles."""
    return {
        (r["obs"], r["steps"], r["theta_k"])
        for r in runs
        if r["obs"] == "w10"
        or (r["obs"] == "w17" and r["theta_k"] not in STAGE2_THETAS)
    }


def _failure_lines(
    runs: Sequence[dict[str, Any]], failed: Sequence[dict[str, Any]]
) -> list[str]:
    """Runs of the released src-method that stopped on a host SVD failure."""
    failed = [*failed, *(r for r in runs if r["status"] == "failed")]
    if not failed:
        return []
    released = [
        r
        for r in runs
        if r["method"].startswith("src-") and "+gesvd" not in r["method"]
    ]
    lines = [
        "## Host SVD failures of the released src-method",
        "",
        f"{len(failed)} of {len(released) + len(failed)} src-method runs without the "
        "fallback stopped with `LinAlgError: SVD did not converge` (numpy gesdd on the "
        "host, in `truncated_qr` or the oversampled truncation). Rerun with a gesvd "
        "fallback, they appear as `+gesvd` in the tables.",
        "",
        "| run | step | error |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['config']} | {r.get('fail_step', '-')} | {r.get('fail', '')[:60]} |"
        for r in sorted(failed, key=lambda r: r["config"])
    ]
    lines.append("")
    return lines


def _stage1_lines(runs: Sequence[dict[str, Any]]) -> list[str]:
    """Stage 1: 5-step weight-10 and weight-17 over the theta grid, src-k1 seed 0."""
    lines = ["## Stage 1: exactly verifiable regime (5 steps, src-k1, seed 0)", ""]
    for obs, dtype in (
        (o, d) for o in ("w10", "w17") for d in ("complex128", "complex64")
    ):
        sel = [
            r
            for r in runs
            if r["obs"] == obs
            and r["steps"] == 5
            and r["method"].split()[0] == "src-k1"  # noqa: PLR2004
            and r["seed"] == 0
            and r["dtype"] == dtype
        ]
        chis = sorted({r["chi"] for r in sel})
        if not sel:
            continue
        lines += [
            f"{obs}, {dtype}: absolute error against BGC exact; `F` marks a failed "
            "run (host SVD did not converge), `*` a run with the gesvd fallback.",
            "",
            "| theta_h / (pi/32) | exact | "
            + " | ".join(f"chi {c}" for c in chis)
            + " |",
            "|---" * (len(chis) + 2) + "|",
        ]
        for k in range(17):
            cells = []
            ref = None
            for c in chis:
                rr = [r for r in sel if r["theta_k"] == k and r["chi"] == c]
                if not rr:
                    cells.append("-")
                    continue
                r = rr[0]
                ref = ref or r["reference"]
                if r["status"] == "failed":
                    cells.append("F")
                elif r["status"] == "done" and r.get("error") is not None:
                    star = "*" if r.get("svd_fallback") else ""
                    cells.append(f"{r['error']:.1e}{star}")
                else:
                    cells.append(r["status"])
            exact = "-" if ref is None else f"{ref['value']:.6g}"
            lines.append(f"| {k} | {exact} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _sandwich_lines(folder: Path) -> list[str]:
    """One compression of an exact sandwich, method by method (sandwich_test)."""
    lines: list[str] = []
    for path in sorted(folder.glob("sandwich_test*.json")):
        d = json.loads(path.read_text())
        lines += [
            f"## One compression of U^dag O U ({d['problem']}, sub-layer {d['sublayer']}, "
            f"chi {d['chi']})",
            "",
            f"O is the observable evolved by src-k1 at bond {d['chi']} (input bond "
            f"{d['input_bond']}); the exact sandwich has bond {d['exact_bond']}. "
            "Relative Frobenius error of the compressed operator and change of "
            "<0|.|0>, mean (min-max) over seeds.",
            "",
            "| method | seeds | rel. Frobenius error | value error | seconds |",
            "|---|---|---|---|---|",
        ]
        for m, rows in d["methods"].items():
            e = [r["rel_frob_error"] for r in rows]
            v = [r["value_error"] for r in rows]
            t = [r["seconds"] for r in rows]
            lines.append(
                f"| {m} | {len(rows)} | {np.mean(e):.3f} ({min(e):.3f}-{max(e):.3f}) "
                f"| {np.mean(v):.1e} | {np.median(t):.1f} |"
            )
        lines.append("")
    return lines


def _probe_lines(folder: Path) -> list[str]:
    """One Trotter step of a saturated operator (step_probe)."""
    probes = [
        json.loads(p.read_text()) for p in sorted((folder / "probe").glob("*.json"))
    ]
    if not probes:
        return []
    lines = [
        "## Cost of one Trotter step at large chi (random saturated operator)",
        "",
        "| method | chi | cutoff | dtype | QR | groups done | step s (or partial) "
        "| TFLOP | TFLOP/s | GPU pool GB | host peak GB | input GB | tiers (first group) "
        "| CholeskyQR tried / fallback / skipped |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(probes, key=lambda d: (d["chi"], d["cutoff"], d["method"])):
        g = d["groups"]
        secs = sum(x["seconds"] for x in g)
        tflop = sum(x["tflop"] for x in g)
        tiers = "-"
        if g:
            plan = next((m for m in g[0]["src_debug"] if "tiers=" in m), "")
            if plan:
                raw = plan.split("tiers=")[1].split(", batches")[0]
                tiers = ", ".join(
                    f"{t}: {raw.count(t)}"
                    for t in ("device", "host", "disk")
                    if t in raw
                )
        status = (
            f"{len(g)}"
            + (" (OOM)" if "oom" in d else "")
            + ("" if d["complete"] else " partial")
        )
        qr = (
            "Householder on rank 0"
            if d.get("distributed_qr_min", 0) > 2**40
            else "library"
        )
        cqr = " / ".join(
            str(sum(x.get(k, 0) for x in g))
            for k in ("cqr_tried", "cqr_fallbacks", "cqr_skipped")
        )
        lines.append(
            f"| {d['method']} | {d['chi']} | {d['cutoff']:g} "
            f"| {d.get('dtype', 'complex128')} | {qr} | {status} | {secs:.1f} | {tflop:.1f} "
            f"| {tflop / secs if secs else 0:.2f} | {max((x['pool_gb'] for x in g), default=0):.1f} "
            f"| {d.get('host_peak_gb', 0):.0f} | {d['input_gb']:.1f} | {tiers} | {cqr} |"
        )
    lines.append("")
    return lines


def _check_lines(c: dict[str, Any]) -> list[str]:
    lines = ["## Checks", ""]
    d = c.get("dense_12q")
    if d:
        lines += [
            f"Dense operator, 12-qubit heavy hexagon (qubits {d['qubits']}), "
            f"{d['n_sublayers']} sub-layers, theta_h = {d['theta_h']:.4f}, "
            f"{d['steps']} steps, O = {d['pauli']}, full bond 4096. "
            f"qiskit statevector against the dense reference: {d['qiskit_vs_dense']:.1e}.",
            "",
            "| method | relative Frobenius error of U^dag O U |",
            "|---|---|",
        ]
        lines += [f"| {m} | {e:.1e} |" for m, e in d["rel_error"].items()]
        lines += ["", "Stack order:", ""]
        lines += [f"- {k}: {v:.1e}" for k, v in d["stack_order"].items()]
        lines.append("")
    f = c.get("fragment_16q")
    if f:
        lines += [
            f"16-qubit fragment (qubits {f['qubits']}), theta_h = {f['theta_h']:.4f}, "
            f"{f['steps']} steps, O = {f['pauli']}, chi = {f['chi']}: "
            f"<0|O|0> against qiskit ({f['qiskit']:.6f}).",
            "",
            "| method | abs error | max bond |",
            "|---|---|---|",
        ]
        lines += [
            f"| {m} | {e:.1e} | {f['max_bond'][m]} |" for m, e in f["abs_error"].items()
        ]
        lines.append("")
    cl = c.get("clifford")
    if cl:
        lines += ["Clifford points (127 qubits, chi = 16) against stim:", ""]
        for key, v in cl.items():
            if "_is_" in key:
                lines.append(
                    f"- {key}: matches the paper's string up to sign: {v['matches_up_to_sign']}"
                )
        lines += [
            "",
            "| case | stim | method | abs error | max bond |",
            "|---|---|---|---|---|",
        ]
        for key, v in cl.items():
            if "_is_" in key:
                continue
            for m, e in v.items():
                if m == "stim":
                    continue
                extra = (
                    f" (max over steps {e['max_step_error_theta0']:.1e})"
                    if e.get("max_step_error_theta0") is not None
                    else ""
                )
                lines.append(
                    f"| {key} | {v['stim']:g} | {m} | {e['error']:.1e}{extra} | {e['max_bond']} |"
                )
        lines.append("")
    return lines


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    dtype: DType = "complex128",
) -> None:
    """Run a command with every array in ``dtype``.

    Args:
        tokens: The command and its arguments.
        dtype: Precision of the gates, observables, random operators and
            diagnostics; src-method and quimb follow their inputs.
    """
    set_dtype(dtype)
    app(tokens)


if __name__ == "__main__":
    app.meta()

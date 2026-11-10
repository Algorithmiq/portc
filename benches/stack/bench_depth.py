"""One-shot SRC over a stack against sequential pairwise application, by depth.

Three commands:

- ``accuracy``: relative error against the exact dense product on short chains,
  next to a lower bound on the error of any train with the same bond dimension.
- ``timing``: wall time on longer chains, where no dense reference fits.
- ``evolve``: quench dynamics of a mixed-field Ising chain of 50+ qubits, fusing
  ``k`` Trotter steps per SRC sweep, against a reference at a larger bond.

Each row is logged; ``--output`` also writes the table as Markdown.
"""

from __future__ import annotations

import logging
from dataclasses import astuple, dataclass, fields
from functools import reduce
from pathlib import Path
from time import perf_counter

import cyclopts
import numpy as np
from scipy.linalg import expm

from portc import apply, src

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Benchmark one-shot SRC over stacks against pairwise apply.")

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def random_mpo(n_sites: int, bond: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Complex Gaussian MPO, scaled to keep products of order one."""

    def site(*shape: int) -> np.ndarray:
        scale = np.sqrt(2 * bond * 2)
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / scale

    bulk = [site(bond, bond, 2, 2) for _ in range(n_sites - 2)]
    return [site(bond, 2, 2), *bulk, site(bond, 2, 2)]


def random_mps(n_sites: int, bond: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Complex Gaussian MPS."""

    def site(*shape: int) -> np.ndarray:
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(
            2 * bond
        )

    bulk = [site(bond, bond, 2) for _ in range(n_sites - 2)]
    return [site(bond, 2), *bulk, site(bond, 2)]


def trotter_layer(
    n_sites: int, parity: int, dt: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Brickwork layer of ``exp(-i dt h)`` gates on bonds of the given parity.

    ``h`` is a two-site mixed-field Ising term with randomized couplings, so the
    layer has bond dimension at most 4 on its gates and 1 elsewhere.
    """
    sites = [I2[None, None] for _ in range(n_sites)]  # (l, r, u, d)
    for i in range(parity, n_sites - 1, 2):
        coupling = 1.0 + 0.2 * rng.normal()
        local = (0.9 + 0.2 * rng.normal()) * Z + (0.5 + 0.2 * rng.normal()) * X
        h = coupling * np.kron(X, X) + 0.5 * (np.kron(local, I2) + np.kron(I2, local))
        gate = expm(-1j * dt * h).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
        U, S, Vh = np.linalg.svd(gate.reshape(4, 4))
        keep = S > 1e-14
        left = (U[:, keep] * np.sqrt(S[keep])).reshape(2, 2, -1).transpose(2, 0, 1)
        right = (np.sqrt(S[keep])[:, None] * Vh[keep]).reshape(-1, 2, 2)
        sites[i], sites[i + 1] = left[None], right[:, None]
    sites[0], sites[-1] = sites[0][0], sites[-1][:, 0]
    return sites


def make_stack(
    family: str, depth: int, n_sites: int, rng: np.random.Generator
) -> list[list[np.ndarray]]:
    """A stack of ``depth`` trains; ``.mps`` families end with an MPS."""
    n_ops = depth - 1 if family.endswith(".mps") else depth
    if family.startswith("random"):
        ops = [random_mpo(n_sites, 3, rng) for _ in range(n_ops)]
        state = random_mps(n_sites, 4, rng)
    else:
        dt = float(family.removesuffix(".mps").split("-")[1])
        ops = [trotter_layer(n_sites, i % 2, dt, rng) for i in range(n_ops)]
        state = random_mps(n_sites, 2, rng)
    return [*ops, state] if family.endswith(".mps") else ops


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def ising_step(
    n_sites: int, dt: float, J: float = 1.0, hx: float = 1.05, hz: float = 0.5
) -> list[np.ndarray]:
    """First-order Trotter step ``exp(-i dt h) exp(-i dt J sum ZZ)`` as a bond-2 MPO.

    ``h = hx X + hz Z`` on every site; the defaults are the non-integrable point of
    Banuls, Cirac and Hastings, PRL 106, 050405 (2011).
    """
    field = expm(-1j * dt * (hx * X + hz * Z))
    a = J * dt
    # exp(-i a ZZ) = sum_m P_m (x) P_m with P = (sqrt(cos a) I, sqrt(-i sin a) Z).
    P = [np.sqrt(np.cos(a)) * I2, np.sqrt(-1j * np.sin(a)) * Z]
    bulk = np.array([[field @ P[l] @ P[r] for r in range(2)] for l in range(2)])
    first = np.array([field @ p for p in P])
    return [first, *[bulk] * (n_sites - 2), first.copy()]


def product_state(n_sites: int) -> list[np.ndarray]:
    """``|0...0>`` as a bond-1 MPS."""
    up = np.array([1, 0], dtype=complex)
    return [up[None], *[up[None, None]] * (n_sites - 2), up[None]]


def overlap(a: list[np.ndarray], b: list[np.ndarray]) -> complex:
    """``<a|b>`` of two MPS."""
    env = np.einsum("ru,su->rs", a[0].conj(), b[0])
    for x, y in zip(a[1:-1], b[1:-1]):
        env = np.einsum("rs,rtu,svu->tv", env, x.conj(), y, optimize=True)
    return complex(np.einsum("rs,ru,su->", env, a[-1].conj(), b[-1]))


def z_center(psi: list[np.ndarray]) -> float:
    """``<Z>`` on the middle site."""
    j = len(psi) // 2
    z_psi = list(psi)
    z_psi[j] = psi[j] * np.array([1, -1])
    return overlap(psi, z_psi).real / overlap(psi, psi).real


def one_shot(
    stack: list[list[np.ndarray]], chi: int, seed: int, device: str = "cpu"
) -> list[np.ndarray]:
    return src(*stack, chi_out=chi, dtype=np.complex128, seed=seed, device=device)


def sequential(stack: list[list[np.ndarray]], chi: int, seed: int) -> list[np.ndarray]:
    """Right-to-left pairwise `apply`, truncating to ``chi`` after every product."""
    result = stack[-1]
    for i, layer in enumerate(reversed(stack[:-1])):
        result = apply(layer, result, chi, dtype=np.complex128, seed=seed + i)
    return result


# ---------------------------------------------------------------------------
# Dense reference
# ---------------------------------------------------------------------------


def dense(train: list[np.ndarray]) -> np.ndarray:
    """Tensor with site-major legs ``(u_0, d_0, u_1, d_1, ...)``; an MPS has d = 1."""
    if train[0].ndim == 2:
        train = [t[..., None] for t in train]
    T = np.moveaxis(train[0], 0, -1)  # bond last
    for W in train[1:-1]:
        T = np.moveaxis(np.tensordot(T, W, axes=(-1, 0)), -3, -1)
    return np.tensordot(T, train[-1], axes=(-1, 0))


def as_matrix(t: np.ndarray) -> np.ndarray:
    n_sites = t.ndim // 2
    rows = list(range(0, t.ndim, 2))
    cols = list(range(1, t.ndim, 2))
    t = t.transpose(rows + cols)
    return t.reshape(int(np.prod(t.shape[:n_sites])), -1)


def site_major(matrix: np.ndarray, up: list[int], down: list[int]) -> np.ndarray:
    n_sites = len(up)
    t = matrix.reshape(up + down)
    return t.transpose([p for i in range(n_sites) for p in (i, n_sites + i)])


def cut_spectra(t: np.ndarray) -> list[np.ndarray]:
    """Normalized singular values of every left/right cut of a site-major tensor."""
    n_sites = t.ndim // 2
    norm = np.linalg.norm(t)
    return [
        np.linalg.svd(t.reshape(int(np.prod(t.shape[: 2 * cut])), -1), compute_uv=False)
        / norm
        for cut in range(1, n_sites)
    ]


def lower_bound(spectra: list[np.ndarray], chi: int) -> float:
    """Largest best-rank-``chi`` tail over all cuts: no ``chi`` train does better."""
    return max(float(np.sqrt((s[chi:] ** 2).sum())) for s in spectra)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass
class AccuracyRow:
    family: str
    depth: int
    chi: int
    lower_bound: float
    one_shot: float
    sequential: float
    one_shot_wins: float


@dataclass
class TimingRow:
    family: str
    depth: int
    chi: int
    one_shot_s: float
    sequential_s: float
    ratio: float


@dataclass
class EvolveRow:
    n_sites: int
    chi: int
    method: str
    seconds: float
    infidelity: float
    z_error: float


def write_table(rows: list, output: Path | None) -> None:
    """Log every row and, if requested, write them as a Markdown table."""
    for row in rows:
        logger.info("result: %s", vars(row))
    if output is None:
        return
    names = [f.name for f in fields(rows[0])]
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for row in rows:
        cells = [f"{v:.3g}" if isinstance(v, float) else str(v) for v in astuple(row)]
        lines.append("| " + " | ".join(cells) + " |")
    output.write_text("\n".join(lines) + "\n")


def win_fraction(errs_one: list[float], errs_seq: list[float]) -> float:
    """Fraction of instances where one-shot is more accurate, ignoring ties.

    Instances where both errors sit at machine precision, or where both methods
    compute the same thing (depth 2), carry no information and are left out; NaN
    if no instance is informative.
    """
    one, seq = np.array(errs_one), np.array(errs_seq)
    informative = (np.maximum(one, seq) > 1e-12) & (one != seq)
    if not informative.any():
        return float("nan")
    return float(np.mean(one[informative] < seq[informative]))


ACCURACY_CHIS = {
    "random.mps": [4, 8, 16],
    "random": [8, 16, 32],
    "trotter-0.3.mps": [2, 4, 8],
    "trotter-0.3": [4, 8, 16],
    "trotter-0.8.mps": [4, 8, 16],
    "trotter-0.8": [8, 16, 32],
}


@app.command
def accuracy(
    n_sites: int = 10,
    seeds: int = 20,
    depths: tuple[int, ...] = (2, 3, 4),
    families: tuple[str, ...] = tuple(ACCURACY_CHIS),
    output: Path | None = None,
) -> None:
    """Median relative error against the dense product, by depth.

    Args:
        n_sites: Chain length; the dense reference needs ``4**n_sites`` entries.
        seeds: Number of random instances per point.
        depths: Numbers of trains in the stack.
        families: Input families; ``.mps`` ones end with an MPS.
        output: Optional Markdown file for the table.
    """
    rows = []
    for family in families:
        down = 1 if family.endswith(".mps") else 2
        for depth in depths:
            chis = ACCURACY_CHIS[family]
            bound = {chi: [] for chi in chis}
            errs = {(chi, m): [] for chi in chis for m in ("one", "seq")}
            for seed in range(seeds):
                stack = make_stack(family, depth, n_sites, np.random.default_rng(seed))
                ref = reduce(np.matmul, [as_matrix(dense(t)) for t in stack])
                spectra = cut_spectra(site_major(ref, [2] * n_sites, [down] * n_sites))
                norm = np.linalg.norm(ref)
                for chi in chis:
                    bound[chi].append(lower_bound(spectra, chi))
                    for m, method in (("one", one_shot), ("seq", sequential)):
                        got = as_matrix(dense(method(stack, chi, seed)))
                        errs[chi, m].append(np.linalg.norm(got - ref) / norm)
            rows.extend(
                AccuracyRow(
                    family,
                    depth,
                    chi,
                    float(np.median(bound[chi])),
                    float(np.median(errs[chi, "one"])),
                    float(np.median(errs[chi, "seq"])),
                    win_fraction(errs[chi, "one"], errs[chi, "seq"]),
                )
                for chi in chis
            )
    write_table(rows, output)


@app.command
def timing(
    n_sites: int = 30,
    chi: int = 64,
    mps_bond: int = 64,
    reps: int = 3,
    output: Path | None = None,
) -> None:
    """Best-of-``reps`` wall time of one-shot SRC and pairwise apply, by depth.

    Args:
        n_sites: Chain length.
        chi: Output bond dimension.
        mps_bond: Bond dimension of the input MPS.
        reps: Repetitions per point; the minimum is reported.
        output: Optional Markdown file for the table.
    """
    rng = np.random.default_rng(0)
    cases = [
        (
            f"trotter-0.3 x{k} . mps",
            [trotter_layer(n_sites, i % 2, 0.3, rng) for i in range(k)],
        )
        for k in (1, 2, 3, 4)
    ] + [
        (
            f"random D={bond} x{k} . mps",
            [random_mpo(n_sites, bond, rng) for _ in range(k)],
        )
        for bond in (8, 16)
        for k in (1, 2, 3)
    ]
    state = random_mps(n_sites, mps_bond, rng)
    rows = []
    for name, ops in cases:
        stack = [*ops, state]
        times = []
        for method in (one_shot, sequential):
            best = np.inf
            for _ in range(reps):
                start = perf_counter()
                method(stack, chi, 0)
                best = min(best, perf_counter() - start)
            times.append(best)
        rows.append(
            TimingRow(name, len(stack), chi, times[0], times[1], times[0] / times[1])
        )
    write_table(rows, output)


def _evolve_src(
    step: list[np.ndarray], steps: int, k: int, chi: int, device: str
) -> tuple[list[np.ndarray], float]:
    """``steps`` Trotter steps from ``|0...0>``, ``k`` of them per SRC sweep."""
    psi = product_state(len(step))
    start = perf_counter()
    for sweep in range(steps // k):
        psi = one_shot([*[step] * k, psi], chi, sweep, device)
    return psi, perf_counter() - start


def _evolve_quimb(
    step: list[np.ndarray], steps: int, chi: int
) -> tuple[list[np.ndarray], float]:
    """The same evolution with quimb's MPO application and SVD truncation."""
    import quimb.tensor as qtn  # noqa: PLC0415  (only this command needs quimb)

    op = qtn.MatrixProductOperator(step)
    psi = qtn.MatrixProductState(product_state(len(step)))
    op.apply(psi, compress=True, max_bond=chi, cutoff=0.0)  # warm-up
    start = perf_counter()
    for _ in range(steps):
        psi = op.apply(psi, compress=True, max_bond=chi, cutoff=0.0)
    return list(psi.arrays), perf_counter() - start


@app.command
def evolve(
    n_sites: int = 50,
    steps: int = 80,
    dt: float = 0.1,
    chis: tuple[int, ...] = (64, 128, 256, 512),
    steps_per_sweep: tuple[int, ...] = (1, 2, 4),
    quimb_max_chi: int = 256,
    reference_chi: int = 1024,
    device: str = "gpu",
    output: Path | None = None,
) -> None:
    """Quench of a mixed-field Ising chain, fusing several Trotter steps per sweep.

    Every run starts from ``|0...0>`` and applies ``steps`` steps; the reference is
    one step per sweep at ``reference_chi``. Rows report the wall time of
    the evolution, the infidelity and the error of the middle ``<Z>`` against the
    reference at the final time.

    Args:
        n_sites: Number of qubits.
        steps: Number of Trotter steps; a multiple of every ``steps_per_sweep``.
        dt: Trotter time step.
        chis: Bond dimensions of the evolved state.
        steps_per_sweep: Trotter steps fused into one SRC sweep (1 is pairwise).
        quimb_max_chi: Largest ``chi`` also run with quimb on the CPU; 0 skips it.
        reference_chi: Bond dimension of the reference evolution.
        device: Where SRC runs (``cpu`` or ``gpu``).
        output: Optional Markdown file for the table.
    """
    step = ising_step(n_sites, dt)
    if device == "gpu":  # keep CUDA initialisation and kernel compilation out
        small = ising_step(4, dt)
        one_shot([small, small, product_state(4)], 2, 0, device)

    reference, seconds = _evolve_src(step, steps, 1, reference_chi, device)
    ref_norm = overlap(reference, reference).real
    ref_z = z_center(reference)
    logger.info("reference: chi=%d, %.1f s, <Z>=%.6f", reference_chi, seconds, ref_z)

    def row(chi: int, method: str, psi: list[np.ndarray], seconds: float) -> None:
        fidelity = abs(overlap(reference, psi)) ** 2 / (
            ref_norm * overlap(psi, psi).real
        )
        rows.append(
            EvolveRow(
                n_sites, chi, method, seconds, 1 - fidelity, abs(z_center(psi) - ref_z)
            )
        )
        logger.info("done: %s", vars(rows[-1]))

    rows: list[EvolveRow] = []
    for chi in chis:
        for k in steps_per_sweep:
            row(chi, f"src {device} k={k}", *_evolve_src(step, steps, k, chi, device))
        if chi <= quimb_max_chi:
            row(chi, "quimb cpu", *_evolve_quimb(step, steps, chi))
    write_table(rows, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

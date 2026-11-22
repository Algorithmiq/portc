r"""Roofline and time breakdown of the Trotter-fusion quench of ``bench_depth.py``.

Runs the ``evolve`` quench (``k`` first-order Trotter steps per SRC sweep) with the
library patched from here, never edited:

- ``portc._kernels._Contractions.__call__`` is replaced by a copy that times the
  first compilation of each ``(equation, shapes)`` on the host (path planning) and
  every evaluation between two ``cupy.cuda.Event``\\ s. The equation names the
  phase: ``ltr`` is the environment, ``rtl_m`` the sketch, ``rtl_s`` the
  projection and ``first`` the first site.
- ``truncated_qr`` and ``cholesky_qr`` (in ``portc._sweep``) and ``_truncate`` are
  timed with events as the ``qr`` phase.
- ``portc._sites.to_device_async`` is timed with events on its stream (H2D), and
  ``portc._sweep.to_numpy`` on the host after a device synchronize (D2H).
- ``portc._sweep.make_plan`` is timed on the host (planning).

Every SRC call is timed on the host between two device synchronizes; "other" is
that time minus the GPU time of the contractions and the QR.

Flops come from ``opt_einsum.contract_path`` with the library's optimizer,
converted to real flops with a calibration on ``ij,jk->ik`` (8 real flops per
complex multiply-add). Modelled bytes sum, over the pairwise steps of the path,
both inputs and the output at 16 B per element: a lower bound on the DRAM traffic,
so the arithmetic intensity is an upper bound. A complex ``m x n`` QR with explicit
``Q`` (``m >= n``) counts ``16 m n**2`` real flops and ``16 (m n + m n)`` bytes.
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache, partial
from math import prod
from pathlib import Path
from time import perf_counter

import cupy as cp
import cyclopts
import numpy as np
import opt_einsum as oe
from opt_einsum import contract_expression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_depth as bd

from portc import _kernels, _sites, _sweep
from portc._kernels import DEFAULT_OPTIMIZE, equations

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Roofline and time breakdown of the fused Trotter quench.")

PHASES = ("env", "sketch", "proj", "first", "qr")
BYTES = 16  # complex128


@cache
def phase_of(eq: str) -> str:
    """The phase of a sweep equation, whatever the depth."""
    for depth in range(1, 9):
        eqs = equations(depth)
        names = {eqs.ltr: "env", eqs.rtl_m: "sketch", eqs.rtl_s: "proj"}
        names[eqs.first] = "first"
        if eq in names:
            return names[eq]
    return "unknown"


@cache
def oe_cost_per_mac() -> float:
    """opt_einsum's cost of one multiply-add, from a plain matrix product."""
    m, n, k = 37, 41, 43
    _, info = oe.contract_path("ij,jk->ik", (m, k), (k, n), shapes=True)
    return float(info.opt_cost) / (m * n * k)


@cache
def contraction_cost(
    eq: str, shapes: tuple[tuple[int, ...], ...], optimize: str
) -> tuple[float, float, float]:
    """Real flops (opt_einsum, calibrated), real flops (own count) and bytes."""
    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize=optimize)
    sizes = info.size_dict
    macs = n_bytes = 0
    for step in info.contraction_list:
        inputs, output = step[2].split("->")
        terms = inputs.split(",")
        macs += prod(sizes[c] for c in set("".join(terms)))
        elements = sum(prod(sizes[c] for c in t) for t in terms)
        n_bytes += (elements + prod(sizes[c] for c in output)) * BYTES
    flops = float(info.opt_cost) / oe_cost_per_mac() * 8
    return flops, 8.0 * macs, float(n_bytes)


def qr_cost(m: int, n: int) -> tuple[float, float]:
    """Real flops and modelled bytes of an ``m x n`` complex QR with explicit Q."""
    m, n = max(m, n), min(m, n)
    return 16.0 * m * n * n, 2.0 * m * n * BYTES


@dataclass
class Sweep:
    """The measurements of one SRC call."""

    input_bond: int
    host_s: float = 0.0
    # (phase, flops, own flops, bytes, start, stop)
    events: list[tuple] = field(default_factory=list)
    h2d: list[tuple] = field(default_factory=list)  # (bytes, start, stop)
    d2h_s: float = 0.0
    d2h_bytes: int = 0
    plan_s: float = 0.0
    compile_s: float = 0.0
    gpu: dict[str, list[float]] = field(default_factory=dict)  # calls, flops...
    h2d_s: float = 0.0
    h2d_bytes: int = 0

    def resolve(self) -> None:
        """Turn the recorded events into totals; the device must be idle."""
        gpu: dict[str, list[float]] = {p: [0, 0.0, 0.0, 0.0, 0.0] for p in PHASES}
        for phase, flops, own, n_bytes, start, stop in self.events:
            row = gpu.setdefault(phase, [0, 0.0, 0.0, 0.0, 0.0])
            row[0] += 1
            row[1] += flops
            row[2] += own
            row[3] += n_bytes
            row[4] += cp.cuda.get_elapsed_time(start, stop) * 1e-3
        self.gpu = gpu
        for n_bytes, start, stop in self.h2d:
            self.h2d_s += cp.cuda.get_elapsed_time(start, stop) * 1e-3
            self.h2d_bytes += n_bytes
        self.events, self.h2d = [], []


class Recorder:
    """Patches the library and collects one `Sweep` per SRC call."""

    def __init__(self) -> None:
        self.current: Sweep | None = None
        self._saved: list[tuple[object, str, object]] = []

    def _patch(self, owner: object, name: str, new: object) -> None:
        self._saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, new)

    def install(self) -> None:
        """Install every patch."""
        rec = self

        def contract(self: _kernels._Contractions, eq: str, *operands):  # noqa: ANN001, ANN202
            shapes = tuple(op.shape for op in operands)
            expr = self._compiled.get((eq, shapes))
            if expr is None:
                start = perf_counter()
                expr = self._compiled[eq, shapes] = contract_expression(
                    eq, *shapes, optimize=self._optimize
                )
                rec.current.compile_s += perf_counter() - start
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            out = expr(*operands)
            stop.record()
            cost = contraction_cost(eq, shapes, self._optimize)
            rec.current.events.append((phase_of(eq), *cost, start, stop))
            return out

        def gpu_timed(fn, cost):  # noqa: ANN001, ANN202
            def wrapper(*args, **kwargs):  # noqa: ANN202
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                out = fn(*args, **kwargs)
                stop.record()
                rec.current.events.append(("qr", *cost(*args), start, stop))
                return out

            return wrapper

        def matrix_cost(matrix, *_):  # noqa: ANN001, ANN202
            flops, n_bytes = qr_cost(*matrix.shape)
            return flops, flops, n_bytes

        def cholesky_cost(block, rows, *_):  # noqa: ANN001, ANN202
            flops, n_bytes = qr_cost(rows, block.shape[0])
            return flops, flops, n_bytes

        def truncate_cost(sites, *_):  # noqa: ANN001, ANN202
            flops = sum(
                qr_cost(prod(s.shape) // s.shape[1], s.shape[1])[0] for s in sites
            )
            return flops, flops, 0.0

        def to_device_async(host, xp, stream):  # noqa: ANN001, ANN202
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            out = original_h2d(host, xp, stream)
            stop.record(stream)
            rec.current.h2d.append((host.nbytes, start, stop))
            return out

        def to_numpy(arr):  # noqa: ANN001, ANN202
            if not isinstance(arr, cp.ndarray):
                return original_d2h(arr)
            cp.cuda.Device().synchronize()
            start = perf_counter()
            out = original_d2h(arr)
            rec.current.d2h_s += perf_counter() - start
            rec.current.d2h_bytes += arr.nbytes
            return out

        def make_plan(*args, **kwargs):  # noqa: ANN202
            start = perf_counter()
            out = original_plan(*args, **kwargs)
            rec.current.plan_s += perf_counter() - start
            return out

        original_h2d, original_d2h = _sites.to_device_async, _sweep.to_numpy
        original_plan = _sweep.make_plan
        self._patch(_kernels._Contractions, "__call__", contract)
        self._patch(_sweep, "truncated_qr", gpu_timed(_sweep.truncated_qr, matrix_cost))
        self._patch(_sweep, "cholesky_qr", gpu_timed(_sweep.cholesky_qr, cholesky_cost))
        self._patch(_sweep, "_truncate", gpu_timed(_sweep._truncate, truncate_cost))
        self._patch(_sites, "to_device_async", to_device_async)
        self._patch(_sweep, "to_numpy", to_numpy)
        self._patch(_sweep, "make_plan", make_plan)

    def uninstall(self) -> None:
        """Restore the library."""
        while self._saved:
            owner, name, value = self._saved.pop()
            setattr(owner, name, value)

    def one_shot(self, stack: list, chi: int, seed: int, device: str) -> list:
        """`bench_depth.one_shot`, measured."""
        bond = max(max(t.shape[:-1]) for t in stack[-1])
        self.current = Sweep(bond)
        cp.cuda.Device().synchronize()
        start = perf_counter()
        out = bd.one_shot(stack, chi, seed, device)
        cp.cuda.Device().synchronize()
        self.current.host_s = perf_counter() - start
        self.current.resolve()
        return out


def evolve(
    step: list[np.ndarray], steps: int, k: int, chi: int, rec: Recorder | None
) -> tuple[list[np.ndarray], float, list[Sweep]]:
    """The loop of `bench_depth._evolve_src`; measured sweeps when ``rec`` is set."""
    psi = bd.product_state(len(step))
    sweeps = []
    start = perf_counter()
    for sweep in range(steps // k):
        if rec is None:
            psi = bd.one_shot([*[step] * k, psi], chi, sweep, "gpu")
        else:
            psi = rec.one_shot([*[step] * k, psi], chi, sweep, "gpu")
            sweeps.append(rec.current)
    return psi, perf_counter() - start, sweeps


def aggregate(sweeps: list[Sweep]) -> dict[str, float]:
    """Sum the measurements of several sweeps."""
    tot: dict[str, float] = defaultdict(float)
    tot["sweeps"] = len(sweeps)
    for s in sweeps:
        tot["wall_s"] += s.host_s
        for phase, (calls, flops, own, n_bytes, secs) in s.gpu.items():
            tot[f"{phase}_calls"] += calls
            tot[f"{phase}_flops"] += flops
            tot[f"{phase}_own_flops"] += own
            tot[f"{phase}_bytes"] += n_bytes
            tot[f"{phase}_s"] += secs
        tot["h2d_s"] += s.h2d_s
        tot["h2d_bytes"] += s.h2d_bytes
        tot["d2h_s"] += s.d2h_s
        tot["d2h_bytes"] += s.d2h_bytes
        tot["plan_s"] += s.plan_s
        tot["compile_s"] += s.compile_s
    return tot


ROOFLINE_COLUMNS = [
    "n_sites", "chi", "k", "phase", "aggregation", "optimize", "sweeps", "calls", "real_flops",
    "own_real_flops", "modelled_bytes", "gpu_s", "intensity_flop_per_byte_upper",
    "tflops",
]  # fmt: skip
BREAKDOWN_COLUMNS = [
    "n_sites", "chi", "k", "aggregation", "optimize", "sweeps", "wall_s", "env_s", "sketch_s",
    "proj_s", "first_s", "qr_s", "other_s", "h2d_s", "d2h_s", "planning_s",
    "other_rest_s", "h2d_bytes", "d2h_bytes", "uninstrumented_s", "inflation",
]  # fmt: skip


def append_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Append rows, writing the header on a new file."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, columns)
        if new:
            writer.writeheader()
        writer.writerows(rows)


@app.default
def main(
    *,
    n_sites: int = 50,
    chis: tuple[int, ...] = (256, 512, 1024),
    ks: tuple[int, ...] = (1, 2, 4),
    steps: int = 80,
    dt: float = 0.1,
    baseline: bool = True,
    out_dir: Path = Path(),
    optimize: str = DEFAULT_OPTIMIZE,
) -> None:
    """Run the quench instrumented and append to the CSVs in ``out_dir``.

    Args:
        n_sites: Number of qubits.
        chis: Bond dimensions of the evolved state.
        ks: Trotter steps per sweep.
        steps: Number of Trotter steps.
        dt: Trotter time step.
        baseline: Also run each point uninstrumented, before the instrumented run.
        out_dir: Where ``roofline.csv``, ``breakdown.csv`` and ``sweeps.csv`` go.
        optimize: The opt_einsum path optimizer passed to ``src``.
    """
    logger.info(
        "opt_einsum cost per multiply-add: %s (ij,jk->ik); optimizer %s",
        oe_cost_per_mac(),
        optimize,
    )
    if optimize != DEFAULT_OPTIMIZE:
        bd.src = partial(bd.src, optimize=optimize)
    step = bd.ising_step(n_sites, dt)
    small = bd.ising_step(4, dt)
    bd.one_shot([small, small, bd.product_state(4)], 2, 0, "gpu")
    # A few full-size sweeps of every depth, so that no measured run pays a
    # first-call cost.
    for k in ks:
        evolve(step, 4 * k, k, min(chis), None)

    rec = Recorder()
    out_dir.mkdir(parents=True, exist_ok=True)
    for chi in chis:
        for k in ks:
            base_s = float("nan")
            if baseline:
                psi_base, base_s, _ = evolve(step, steps, k, chi, None)
                logger.info(
                    "baseline n=%d chi=%d k=%d: %.2f s", n_sites, chi, k, base_s
                )
            rec.install()
            try:
                psi, loop_s, sweeps = evolve(step, steps, k, chi, rec)
            finally:
                rec.uninstall()
            wall = sum(s.host_s for s in sweeps)
            logger.info(
                "instrumented n=%d chi=%d k=%d: %.2f s in SRC, %.2f s loop",
                n_sites, chi, k, wall, loop_s,
            )  # fmt: skip
            if baseline:
                z_diff = abs(bd.z_center(psi) - bd.z_center(psi_base))
                logger.info("  |<Z> instrumented - baseline| = %.2e", z_diff)
            roof_rows, break_rows, sweep_rows = [], [], []
            groups = {
                "all": sweeps,
                "saturated": [s for s in sweeps if s.input_bond == chi],
            }
            for agg, members in groups.items():
                tot = aggregate(members)
                base = {
                    "n_sites": n_sites,
                    "chi": chi,
                    "k": k,
                    "aggregation": agg,
                    "optimize": optimize,
                }
                for phase in PHASES:
                    secs, flops = tot[f"{phase}_s"], tot[f"{phase}_flops"]
                    n_bytes = tot[f"{phase}_bytes"]
                    roof_rows.append(
                        base
                        | {
                            "phase": phase,
                            "sweeps": len(members),
                            "calls": int(tot[f"{phase}_calls"]),
                            "real_flops": flops,
                            "own_real_flops": tot[f"{phase}_own_flops"],
                            "modelled_bytes": n_bytes,
                            "gpu_s": secs,
                            "intensity_flop_per_byte_upper": flops / n_bytes
                            if n_bytes
                            else float("nan"),
                            "tflops": flops / secs * 1e-12 if secs else float("nan"),
                        }
                    )
                gpu_s = sum(tot[f"{p}_s"] for p in PHASES)
                other = tot["wall_s"] - gpu_s
                planning = tot["plan_s"] + tot["compile_s"]
                break_rows.append(
                    base
                    | {
                        "sweeps": len(members),
                        "wall_s": tot["wall_s"],
                        **{f"{p}_s": tot[f"{p}_s"] for p in PHASES},
                        "other_s": other,
                        "h2d_s": tot["h2d_s"],
                        "d2h_s": tot["d2h_s"],
                        "planning_s": planning,
                        "other_rest_s": other - tot["h2d_s"] - tot["d2h_s"] - planning,
                        "h2d_bytes": tot["h2d_bytes"],
                        "d2h_bytes": tot["d2h_bytes"],
                        # The uninstrumented time is per run; only "all" compares.
                        "uninstrumented_s": base_s if agg == "all" else float("nan"),
                        "inflation": tot["wall_s"] / base_s - 1
                        if agg == "all"
                        else float("nan"),
                    }
                )
            for i, s in enumerate(sweeps):
                for phase, (calls, flops, _, n_bytes, secs) in s.gpu.items():
                    sweep_rows.append(
                        {
                            "n_sites": n_sites,
                            "chi": chi,
                            "k": k,
                            "optimize": optimize,
                            "sweep": i,
                            "input_bond": s.input_bond,
                            "host_s": s.host_s,
                            "phase": phase,
                            "calls": calls,
                            "real_flops": flops,
                            "modelled_bytes": n_bytes,
                            "gpu_s": secs,
                            "h2d_s": s.h2d_s,
                            "d2h_s": s.d2h_s,
                            "planning_s": s.plan_s + s.compile_s,
                        }
                    )
            append_csv(out_dir / "roofline.csv", ROOFLINE_COLUMNS, roof_rows)
            append_csv(out_dir / "breakdown.csv", BREAKDOWN_COLUMNS, break_rows)
            append_csv(out_dir / "sweeps.csv", list(sweep_rows[0]), sweep_rows)
            for row in break_rows:
                logger.info("breakdown: %s", {key: row[key] for key in row})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

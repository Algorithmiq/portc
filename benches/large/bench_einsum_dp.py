"""opt_einsum ``auto`` against ``dp`` paths for the SRC kernels, on the GPU.

Three commands:

- ``kernels``: time `SiteKernels.env`, ``.sketch`` and ``.project`` on random arrays
  of the reference shapes, with both optimizers in one process, and check that
  their outputs agree;
- ``sweep``: run the SRC of a ``generate``-d stack of `bench_large` a few times with
  one optimizer, logging the plan, the pass times and the device peak, and save the
  last output;
- ``distance``: the QR distance between the saved ``auto`` and ``dp`` outputs.

Each command logs its rows and appends them as JSON lines under ``OUT``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
from importlib.metadata import version
from math import prod
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cyclopts
import numpy as np
import opt_einsum as oe

import portc
import portc._kernels as kernels_module
import portc._sweep as sweep_module
from portc import Resources, src
from portc._plan import make_plan

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Measure opt_einsum auto against dp for the SRC kernels.")

HERE = Path(__file__).parent
OUT = Path("einsum_dp")
OPTIMIZERS = ("auto", "dp")
GB = 1e9
# opt_einsum counts a real multiply-add as 2 operations; a complex one is 8 real
# flops, so complex128 rates are 4x the opt_einsum count over the time.
COMPLEX_FLOPS = 4


def provenance(optimize: str | None = None) -> dict[str, Any]:
    """Versions (``portc`` carries the commit), job and GPU of this process."""
    import cupy  # noqa: PLC0415  (GPU only)

    return {
        "optimize": optimize,
        "portc": portc.__version__,
        "numpy": np.__version__,
        "cupy": cupy.__version__,
        "opt_einsum": version("opt_einsum"),
        "python": sys.version.split()[0],
        "gpu": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "gpus": cupy.cuda.runtime.getDeviceCount(),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }


# ---------------------------------------------------------------------------
# Experiment A: kernels
# ---------------------------------------------------------------------------


def _shapes(
    kernel: str, b: int, bonds: tuple[int, ...], phys: int, eta: int
) -> tuple[tuple[int, ...], ...]:
    """Operand shapes of one kernel call, as the planner sizes them."""
    cores = tuple((d, d, phys, phys) for d in bonds)
    return {
        "env": ((b, *bonds), (b, phys, phys), *cores),
        "sketch": ((b, *bonds), *cores, (eta, *bonds)),
        "project": ((b, eta, phys, phys), *cores, (eta, *bonds)),
    }[kernel]


def _random(shape: tuple[int, ...]) -> Any:  # noqa: ANN401  (a cupy array)
    import cupy  # noqa: PLC0415

    scale = 1 / np.sqrt(2 * prod(shape[1:]))
    out = cupy.random.standard_normal(shape) * scale
    return out + 1j * (cupy.random.standard_normal(shape) * scale)


def _timed(fn: Callable[[], Any], warmups: int, runs: int) -> tuple[list[float], Any]:
    """CUDA-event times of ``runs`` calls after ``warmups``, and the last output."""
    import cupy  # noqa: PLC0415

    out = None
    for _ in range(warmups):
        out = fn()
    cupy.cuda.Device().synchronize()
    times = []
    for _ in range(runs):
        del out
        start, end = cupy.cuda.Event(), cupy.cuda.Event()
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(cupy.cuda.get_elapsed_time(start, end) / 1e3)
    return times, out


@app.command
def kernels(
    bonds: tuple[int, ...] = (4, 4, 4000, 4),
    phys: int = 4,
    eta: int = 2000,
    project: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    sketch: tuple[int, ...] = (96, 192, 384),
    env: tuple[int, ...] = (224,),
    limit_gb: float = 50.0,
    warmups: int = 2,
    runs: int = 5,
    output: Path = OUT / "kernels.jsonl",
) -> None:
    """Experiment A: each kernel, batch and optimizer on random reference arrays.

    Both optimizers run in this process, each passed to `SiteKernels` and to
    `peak_elements` as `src` passes it. A (kernel, optimizer, batch) whose planned
    peak exceeds
    ``limit_gb`` is skipped and recorded as such.

    Args:
        bonds: The bond dimensions of the layers.
        phys: The physical leg dimension.
        eta: The sketch size (rows of the projected environment).
        project: Projection batches (rows).
        sketch: Sketch batches (columns).
        env: Environment batches (columns).
        limit_gb: Skip calls whose `peak_elements` exceeds this many GB.
        warmups: Untimed calls before timing.
        runs: Timed calls; the median is reported.
        output: JSON lines file.
    """
    import cupy  # noqa: PLC0415

    pool = cupy.get_default_memory_pool()
    itemsize = np.dtype(np.complex128).itemsize
    output.parent.mkdir(parents=True, exist_ok=True)
    batches = {"env": env, "sketch": sketch, "project": project}
    eqs = kernels_module.equations(len(bonds))
    with output.open("w") as f:
        f.write(json.dumps({"meta": provenance()}) + "\n")
        for kernel, sizes in batches.items():
            eq = {"env": eqs.ltr, "sketch": eqs.rtl_m, "project": eqs.rtl_s}[kernel]
            cores = [_random((d, d, phys, phys)) for d in bonds]
            proj = _random((eta, *bonds)) if kernel != "env" else None
            for b in sizes:
                shapes = _shapes(kernel, b, bonds, phys, eta)
                lead = _random(shapes[0])
                omega = _random(shapes[1]) if kernel == "env" else None
                outputs, args = {}, None
                for opt in OPTIMIZERS:
                    start = perf_counter()
                    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
                    search = perf_counter() - start
                    start = perf_counter()
                    oe.contract_expression(eq, *shapes, optimize=opt)
                    build = perf_counter() - start
                    peak = kernels_module.peak_elements(eq, shapes, opt)
                    row: dict[str, Any] = {
                        "kernel": kernel,
                        "optimizer": opt,
                        "batch": b,
                        "flops": float(info.opt_cost),
                        "largest_intermediate": float(info.largest_intermediate),
                        "peak_elements": peak,
                        "peak_gb": peak * itemsize / GB,
                        "path": [list(p) for p in info.path],
                        "path_search_s": search,
                        "expression_build_s": build,
                    }
                    if peak * itemsize > limit_gb * GB:
                        row["status"] = "skipped: peak_elements above limit"
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        logger.info(
                            "%s b=%d %s: skipped (%.0f GB)",
                            kernel,
                            b,
                            opt,
                            row["peak_gb"],
                        )
                        continue
                    sk = kernels_module.SiteKernels(len(bonds), opt)
                    args = {
                        "env": (lead, omega, cores),
                        "sketch": (lead, cores, proj),
                        "project": (lead, cores, proj),
                    }[kernel]
                    method = getattr(sk, kernel)

                    def call(method=method, args=args) -> Any:  # noqa: ANN401
                        return method(*args)

                    pool.free_all_blocks()
                    held = pool.used_bytes()
                    try:
                        times, out = _timed(call, warmups, runs)
                    except cupy.cuda.memory.OutOfMemoryError as err:
                        row["status"] = f"out of memory: {err}"
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        logger.info("%s b=%d %s: out of memory", kernel, b, opt)
                        pool.free_all_blocks()
                        continue
                    t = median(times)
                    row |= {
                        "status": "ok",
                        "times_s": times,
                        "median_s": t,
                        "per_unit_s": t / b,
                        "tflops": COMPLEX_FLOPS * row["flops"] / t / 1e12,
                        "pool_peak_gb": pool.total_bytes() / GB,
                        "pool_beyond_inputs_gb": (pool.total_bytes() - held) / GB,
                        "inputs_gb": held / GB,
                    }
                    outputs[opt] = out
                    del out
                    if len(outputs) == 2:
                        a, d = outputs["auto"], outputs["dp"]
                        row["rel_diff_vs_auto"] = float(
                            cupy.linalg.norm(a - d) / cupy.linalg.norm(a)
                        )
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    logger.info(
                        "%s b=%d %s: %.4f s (%.3e s/unit), %.2f TFLOP/s, pool +%.1f GB%s",
                        kernel,
                        b,
                        opt,
                        t,
                        t / b,
                        row["tflops"],
                        row["pool_beyond_inputs_gb"],
                        f", diff {row['rel_diff_vs_auto']:.1e}"
                        if "rel_diff_vs_auto" in row
                        else "",
                    )
                del outputs, lead, omega, args
                pool.free_all_blocks()
            del cores, proj
            pool.free_all_blocks()


# ---------------------------------------------------------------------------
# Experiments B and C: sweeps
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    """Collect the pass times and out-of-memory splits the sweep logs."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())

    def passes(self, pattern: str) -> list[float]:
        return [
            float(m.group(1))
            for msg in self.records
            if (m := re.search(pattern + r": ([0-9.]+) s", msg))
        ]


@app.command
def sweep(
    directory: Path,
    *,
    chi_out: int = 500,
    gpu_memory: str = "60GB",
    devices: int = 1,
    repeats: int = 3,
    seed: int = 0,
    save_dir: Path | None = None,
    optimize: str = "dp",
    output: Path = OUT / "sweeps.jsonl",
) -> None:
    """Experiments B and C: ``repeats`` sweeps with one path optimizer.

    Times the expression builds (path search included) separately: `_Contractions`
    plans per `src` call, so every sweep pays them again.

    Args:
        directory: A stack written by `bench_large` ``generate``.
        chi_out: The output bond dimension.
        gpu_memory: GPU budget of each device.
        devices: The number of GPUs.
        repeats: Sweeps to run.
        seed: Seed of the sketch, the same for every sweep.
        save_dir: Where to save the output of the last sweep, as ``<optimizer>.npz``.
        optimize: The opt_einsum path optimizer, passed to `src`.
        output: JSON lines file, appended to.
    """
    import cupy  # noqa: PLC0415

    sys.path.insert(0, str(HERE))
    from bench_large import _load  # noqa: PLC0415

    opt = optimize
    layers = _load(directory)
    capture = _Capture()
    logging.getLogger("portc").addHandler(capture)
    logging.getLogger("portc").setLevel(logging.DEBUG)

    builds: list[float] = []
    real_build = kernels_module.contract_expression

    def timed_build(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        start = perf_counter()
        expr = real_build(*args, **kwargs)
        builds.append(perf_counter() - start)
        return expr

    kernels_module.contract_expression = timed_build
    plans: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        plans.append(make_plan(*args, **kwargs))
        return plans[-1]

    sweep_module.make_plan = spy
    meta = provenance(opt)
    output.parent.mkdir(parents=True, exist_ok=True)
    out = None
    with output.open("a") as f:
        for rep in range(repeats):
            for gpu in range(devices):
                with cupy.cuda.Device(gpu):
                    cupy.get_default_memory_pool().free_all_blocks()
            capture.records.clear()
            builds.clear()
            del out
            start = perf_counter()
            out = src(
                *layers,
                chi_out=chi_out,
                dtype=layers[0][0].dtype,
                seed=seed,
                device="gpu",
                resources=Resources(gpu_memory, None, None, devices=devices),
                optimize=opt,
            )
            seconds = perf_counter() - start
            pools = []
            for gpu in range(devices):
                with cupy.cuda.Device(gpu):
                    pools.append(cupy.get_default_memory_pool().total_bytes())
            plan = plans[-1]
            tiers = [s.tier for s in plan.sites]
            row = {
                "experiment": "B" if devices == 1 else "C",
                "optimizer": opt,
                "devices": devices,
                "repeat": rep,
                "seconds": seconds,
                "left_to_right_s": capture.passes("Left-to-right sweep"),
                "right_to_left_s": capture.passes("Right-to-left sweep"),
                "oom_splits": sum("Out of memory" in m for m in capture.records),
                "expression_builds": len(builds),
                "expression_build_s": sum(builds),
                "batches": sorted(
                    {(s.env_batch, s.sketch_batch, s.project_batch) for s in plan.sites}
                ),
                "batches_per_site": [
                    (s.env_batch, s.sketch_batch, s.project_batch) for s in plan.sites
                ],
                "planned_device_peak_gb": plan.device_peak / GB,
                "pool_gb": [p / GB for p in pools],
                "tiers": {t: tiers.count(t) for t in ("device", "host", "disk")},
                "chi_out": chi_out,
                "gpu_memory": gpu_memory,
                "stack": str(directory),
                "meta": meta,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            logger.info(
                "%s rep %d: %.1f s, L2R %s, R2L %s, builds %d in %.2f s, batches %s, "
                "pool %s GB, planned %.1f GB, oom splits %d",
                opt,
                rep,
                seconds,
                row["left_to_right_s"],
                row["right_to_left_s"],
                len(builds),
                sum(builds),
                row["batches"],
                [round(p, 2) for p in row["pool_gb"]],
                row["planned_device_peak_gb"],
                row["oom_splits"],
            )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez(save_dir / f"{opt}_g{devices}.npz", *out)


@app.command
def distance(
    save_dir: Path,
    devices: int = 1,
    output: Path = OUT / "distances.jsonl",
) -> None:
    """QR distance between the saved ``auto`` and ``dp`` outputs of `sweep`.

    Args:
        save_dir: The ``save_dir`` of both sweeps.
        devices: The GPU count of the sweeps to compare.
        output: JSON lines file, appended to.
    """
    sys.path.insert(0, str(HERE))
    from bench_large import _qr_distance  # noqa: PLC0415

    def load(opt: str) -> list[np.ndarray]:
        with np.load(save_dir / f"{opt}_g{devices}.npz") as z:
            return [z[f"arr_{i}"] for i in range(len(z.files))]

    d = _qr_distance(load("auto"), load("dp"))
    logger.info("QR distance auto-dp on %d GPUs: %.3e", devices, d)
    with output.open("a") as f:
        f.write(json.dumps({"devices": devices, "qr_distance": d}) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    app()

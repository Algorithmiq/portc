"""Measured hardware ceilings of one GPU: ZGEMM, HBM bandwidth and PCIe.

Writes a JSON file with the best of ``reps`` timings after a warm-up:

- ``zgemm_tflops``: ``cupy.matmul`` of two ``n x n`` complex128 matrices (cuBLAS
  ZGEMM, on the FP64 Tensor Cores of an A100), at 8 real flops per complex
  multiply-add.
- ``hbm_copy_gbs``: a device-to-device copy of a ``copy_gb`` array, counting the
  read and the write; ``hbm_triad_gbs``: ``a = b + s * c``, counting two reads and
  one write.
- ``pcie_*_gbs``: host-to-device and device-to-host copies of ``pcie_gb`` from
  page-locked and from pageable memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import cupy as cp
import cyclopts
import numpy as np

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Measure the GPU's compute, HBM and PCIe ceilings.")


def _best_gpu_seconds(fn, reps: int) -> float:  # noqa: ANN001
    """Best GPU time of ``fn`` between two events, after one warm-up call."""
    fn()
    cp.cuda.Device().synchronize()
    best = np.inf
    for _ in range(reps):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(start, stop) * 1e-3)
    return best


def _best_host_seconds(fn, reps: int) -> float:  # noqa: ANN001
    """Best host wall time of a synchronous ``fn``, after one warm-up call."""
    fn()
    best = np.inf
    for _ in range(reps):
        cp.cuda.Device().synchronize()
        start = perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        best = min(best, perf_counter() - start)
    return best


@app.default
def main(
    *,
    output: Path = Path("ceilings.json"),
    n: int = 8192,
    copy_gb: float = 4.0,
    pcie_gb: float = 1.0,
    reps: int = 5,
) -> None:
    """Measure the ceilings and write them to ``output``.

    Args:
        output: The JSON file to write.
        n: Size of the square ZGEMM operands.
        copy_gb: Size in GB of each array of the HBM tests.
        pcie_gb: Size in GB of the host-device copies.
        reps: Timed repetitions; the best is kept.
    """
    out: dict[str, object] = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "reps": reps,
    }

    rng = cp.random.default_rng(0)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    c = cp.empty_like(a)
    seconds = _best_gpu_seconds(lambda: cp.matmul(a, b, out=c), reps)
    out["zgemm"] = {"n": n, "seconds": seconds, "flops": 8 * n**3}
    out["zgemm_tflops"] = 8 * n**3 / seconds * 1e-12
    del a, b, c
    logger.info("ZGEMM %d: %.2f TFLOP/s", n, out["zgemm_tflops"])

    count = int(copy_gb * 1e9) // 16
    x = cp.ones(count, dtype=cp.complex128)
    y = cp.empty_like(x)
    seconds = _best_gpu_seconds(lambda: y.__setitem__(slice(None), x), reps)
    out["hbm_copy_gbs"] = 2 * x.nbytes / seconds * 1e-9
    z = cp.empty_like(x)
    triad = cp.ElementwiseKernel("T b, T c, T s", "T a", "a = b + s * c", "portc_triad")
    s = cp.complex128(0.5)
    seconds = _best_gpu_seconds(lambda: triad(x, y, s, z), reps)
    out["hbm_triad_gbs"] = 3 * x.nbytes / seconds * 1e-9
    out["hbm_bytes_per_array"] = x.nbytes
    del x, y, z
    logger.info(
        "HBM copy %.0f GB/s, triad %.0f GB/s", out["hbm_copy_gbs"], out["hbm_triad_gbs"]
    )

    count = int(pcie_gb * 1e9) // 16
    dev = cp.empty(count, dtype=cp.complex128)
    pageable = np.ones(count, dtype=np.complex128)
    mem = cp.cuda.alloc_pinned_memory(pageable.nbytes)
    pinned = np.frombuffer(mem, np.complex128, count)
    pinned[:] = 1
    nbytes = pageable.nbytes
    for name, host in (("pinned", pinned), ("pageable", pageable)):
        h2d = _best_host_seconds(lambda host=host: dev.set(host), reps)
        d2h = _best_host_seconds(lambda host=host: dev.get(out=host), reps)
        out[f"pcie_h2d_{name}_gbs"] = nbytes / h2d * 1e-9
        out[f"pcie_d2h_{name}_gbs"] = nbytes / d2h * 1e-9
        logger.info(
            "PCIe %s: H2D %.1f GB/s, D2H %.1f GB/s",
            name,
            out[f"pcie_h2d_{name}_gbs"],
            out[f"pcie_d2h_{name}_gbs"],
        )
    # `cupy.ndarray.get()` allocates a fresh pageable array, as `to_numpy` does.
    d2h = _best_host_seconds(dev.get, reps)
    out["pcie_d2h_get_new_gbs"] = nbytes / d2h * 1e-9
    out["pcie_bytes"] = nbytes
    out["ridge_flop_per_byte"] = (
        out["zgemm_tflops"] * 1e12 / (out["hbm_copy_gbs"] * 1e9)
    )
    output.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

"""Complex GEMM throughput and accuracy on one GPU, per cuBLAS compute mode.

Modes: complex64 in plain FP32, in TF32 (CuPy's compute type) and in the BF16x9
FP32 emulation (``CUBLAS_EMULATE_SINGLE_PRECISION``); complex128 in plain FP64
and in the fixed-point FP64 emulation (``CUBLAS_EMULATE_DOUBLE_PRECISION``).
Each mode runs in its own process, since cuBLAS reads its environment once. The
error is the relative Frobenius distance to the product of the same inputs in
plain complex128 (pedantic compute type, never emulated). An ``eager`` strategy
forces the emulation where cuBLAS's default heuristic may decline it. Writes ``nebius/gemm_probe.json``.

    python benches/kicked_ising/nebius/gemm_probe.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
MODES: dict[str, tuple[str, dict[str, str]]] = {
    "c64-fp32": ("complex64", {}),
    "c64-tf32": ("complex64", {}),
    "c64-bf16x9": ("complex64", {"CUBLAS_EMULATE_SINGLE_PRECISION": "1"}),
    "c128-fp64": ("complex128", {}),
    "c128-emulated": ("complex128", {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1"}),
    "c64-bf16x9-eager": (
        "complex64",
        {"CUBLAS_EMULATE_SINGLE_PRECISION": "1", "CUBLAS_EMULATION_STRATEGY": "eager"},
    ),
    "c128-emulated-eager": (
        "complex128",
        {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1", "CUBLAS_EMULATION_STRATEGY": "eager"},
    ),
}
# (m, k, n): square products, and the tall-times-square shape of a sweep at chi 2048.
SHAPES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (65536, 2048, 2048),
]


def measure(mode: str) -> list[dict[str, Any]]:
    """Time and check every shape in ``mode``, in this process."""
    import cupy as cp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from cupy._core import _routines_linalg as rl  # noqa: PLC0415

    dtype = np.dtype(MODES[mode][0])
    rl.set_compute_type(
        dtype, rl.COMPUTE_TYPE_TF32 if mode == "c64-tf32" else rl.COMPUTE_TYPE_DEFAULT
    )
    rng = cp.random.default_rng(0)
    out = []
    for m, k, n in SHAPES:
        a = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))) / k**0.5
        b = rng.standard_normal((k, n)) + 1j * rng.standard_normal((k, n))
        rl.set_compute_type(np.dtype("complex128"), rl.COMPUTE_TYPE_PEDANTIC)
        exact = a @ b
        rl.set_compute_type(np.dtype("complex128"), rl.COMPUTE_TYPE_DEFAULT)
        a, b = a.astype(dtype), b.astype(dtype)
        c = a @ b
        cp.cuda.Device().synchronize()
        reps = max(3, int(2e12 / (8 * m * k * n)))
        t0 = perf_counter()
        for _ in range(reps):
            c = a @ b
        cp.cuda.Device().synchronize()
        seconds = (perf_counter() - t0) / reps
        error = float(
            cp.linalg.norm(c.astype(exact.dtype) - exact) / cp.linalg.norm(exact)
        )
        out.append(
            {
                "shape": [m, k, n],
                "seconds": seconds,
                "tflops": 8 * m * k * n / seconds / 1e12,
                "relative_error": error,
                "compute_type": rl.compute_type_to_str(rl.get_compute_type(dtype)),
            }
        )
        del a, b, c, exact
        cp.get_default_memory_pool().free_all_blocks()
    return out


def main() -> None:
    """Run every mode in a fresh process and write the results."""
    results: dict[str, Any] = {}
    for mode, (_, extra) in MODES.items():
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("CUBLAS_EMULATE")
        }
        env.update(CUPY_TF32="0", NVIDIA_TF32_OVERRIDE="0", **extra)
        proc = subprocess.run(  # noqa: S603
            [sys.executable, __file__, mode],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        results[mode] = {"env": extra, "rows": json.loads(proc.stdout.splitlines()[-1])}
        for row in results[mode]["rows"]:
            logger.info(
                "%s %s: %.1f TFLOP/s, error %.1e",
                mode,
                row["shape"],
                row["tflops"],
                row["relative_error"],
            )
    (HERE / "gemm_probe.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if len(sys.argv) > 1:
        print(json.dumps(measure(sys.argv[1])))  # noqa: T201
    else:
        main()

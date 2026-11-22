"""One saturated sweep of the quench, bracketed for ``nsys`` or ``ncu``.

Evolves ``|0...0>`` uninstrumented until the state bond reaches ``chi``, then runs
one more sweep of ``k`` steps inside ``cudaProfilerStart``/``cudaProfilerStop``:

- ``--scope sweep`` brackets the whole SRC call (for ``nsys profile
  --capture-range=cudaProfilerApi``);
- ``--scope site`` brackets only the ``site``-th call of the environment, sketch
  and projection contractions, one bulk site of each (for ``ncu
  --profile-from-start off``), by patching ``portc._kernels._Contractions``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

import cupy as cp
import cyclopts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_depth as bd

from portc import _kernels
from portc._kernels import equations

logger = logging.getLogger(__name__)
app = cyclopts.App(help="One saturated SRC sweep of the quench, for a profiler.")


@app.default
def main(
    *,
    n_sites: int = 50,
    chi: int = 512,
    k: int = 4,
    dt: float = 0.1,
    scope: str = "sweep",
    site: int = 25,
) -> None:
    """Saturate the bond, then run one profiled sweep.

    Args:
        n_sites: Number of qubits.
        chi: Bond dimension of the state.
        k: Trotter steps in the profiled sweep.
        dt: Trotter time step.
        scope: ``sweep`` or ``site``.
        site: Which call of each phase ``--scope site`` brackets.
    """
    step = bd.ising_step(n_sites, dt)
    psi, sweeps = bd.product_state(n_sites), 0
    while max(max(t.shape[:-1]) for t in psi) < chi:
        psi = bd.one_shot([*[step] * k, psi], chi, sweeps, "gpu")
        sweeps += 1
    logger.info("saturated after %d sweeps of %d steps", sweeps, k)
    stack = [*[step] * k, psi]
    bd.one_shot(stack, chi, sweeps, "gpu")  # warm, unprofiled

    if scope == "site":
        eqs = equations(k + 1)
        targets = {eqs.ltr, eqs.rtl_m, eqs.rtl_s}
        seen: dict[str, int] = {}
        original = _kernels._Contractions.__call__

        def call(self: _kernels._Contractions, eq: str, *operands: object) -> object:
            seen[eq] = seen.get(eq, 0) + 1
            if eq not in targets or seen[eq] != site:
                return original(self, eq, *operands)
            cp.cuda.Device().synchronize()
            cp.cuda.runtime.profilerStart()
            out = original(self, eq, *operands)
            cp.cuda.Device().synchronize()
            cp.cuda.runtime.profilerStop()
            logger.info("profiled %s %s", eq, [o.shape for o in operands])
            return out

        _kernels._Contractions.__call__ = call
        bd.one_shot(stack, chi, sweeps, "gpu")
        return

    cp.cuda.Device().synchronize()
    cp.cuda.runtime.profilerStart()
    start = perf_counter()
    bd.one_shot(stack, chi, sweeps, "gpu")
    cp.cuda.Device().synchronize()
    cp.cuda.runtime.profilerStop()
    logger.info("profiled sweep: %.3f s", perf_counter() - start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

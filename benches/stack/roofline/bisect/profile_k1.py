"""cProfile of the quench at n = 50, chi = 64, k = 1, for one commit's worktree.

Run from the worktree root with ``PYTHONPATH=<worktree>/src``: it imports that
commit's ``benches/stack/bench_depth.py``. After a warm-up of the same evolution,
profiles the 80 steps and writes the top 25 functions by cumulative time.
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import sys
from pathlib import Path

import cyclopts

sys.path.insert(0, str(Path.cwd() / "benches" / "stack"))
import bench_depth as bd

logger = logging.getLogger(__name__)
app = cyclopts.App(help="cProfile of the n=50, chi=64, k=1 quench.")


@app.default
def main(*, out: Path, n_sites: int = 50, chi: int = 64, steps: int = 80) -> None:
    """Profile the evolution and write the top 25 by cumulative time.

    Args:
        out: The text file to write.
        n_sites: Number of qubits.
        chi: Bond dimension.
        steps: Trotter steps, one per sweep.
    """
    step = bd.ising_step(n_sites, 0.1)
    _, warm = bd._evolve_src(step, steps, 1, chi, "gpu")
    profiler = cProfile.Profile()
    profiler.enable()
    _, seconds = bd._evolve_src(step, steps, 1, chi, "gpu")
    profiler.disable()
    text = io.StringIO()
    stats = pstats.Stats(profiler, stream=text).sort_stats("cumulative")
    stats.print_stats(25)
    header = (
        f"n={n_sites} chi={chi} k=1 steps={steps}: warm-up {warm:.2f} s, "
        f"profiled {seconds:.2f} s (under cProfile)\n"
    )
    out.write_text(header + text.getvalue())
    logger.info("%s", header.strip())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

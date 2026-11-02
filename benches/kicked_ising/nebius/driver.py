"""Run kicked Ising bench jobs on the 8-GPU Nebius node, as many at once as fit.

Each line of the job file is ``<gpus> <name> <bench arguments...>``. A job starts
as soon as ``<gpus>`` GPUs are free, with ``CUDA_VISIBLE_DEVICES`` set to them
and its output in ``nebius/logs/<name>.log``. A single-GPU job is pinned to the
24 logical CPUs next to its GPU (GPUs 0-3 on NUMA node 0, 4-7 on node 1) with
as many BLAS threads; a multi-GPU job gets the whole node. Blank lines and lines
starting with ``#`` are skipped. The bench skips finished runs, so a job file
can be rerun after an interruption. An optional second argument restricts the
driver to some GPUs, e.g. ``0,1,3`` while another driver holds GPU 2.

    python benches/kicked_ising/nebius/driver.py jobs.txt [gpus]
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
REPO = HERE.parents[2]
BENCH = HERE.parent / "bench_kicked_ising.py"
PYTHON = REPO / ".venv" / "bin" / "python"
N_GPUS = 8
CPUS_PER_GPU = 24
# RAM disk: the node's only disk is a 0.15 GB/s virtual disk.
SCRATCH = Path("/dev/shm/ki-scratch")  # noqa: S108


def environment(gpus: list[int]) -> dict[str, str]:
    """The environment of one job."""
    env = dict(os.environ)
    env.pop("CUPY_CUDA_PER_THREAD_DEFAULT_STREAM", None)
    threads = str(CPUS_PER_GPU * len(gpus)) if len(gpus) == 1 else str(os.cpu_count())
    env.update(
        KI_PLATFORM="nebius",
        CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
        CUPY_TF32="0",
        NVIDIA_TF32_OVERRIDE="0",
        TMPDIR=str(SCRATCH),
        OMP_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
    )
    return env


def command(gpus: list[int], args: list[str]) -> list[str]:
    """The bench command line, pinned next to its GPU when it has one."""
    cmd = [str(PYTHON), "-W", "ignore::UserWarning", str(BENCH), *args]
    if len(gpus) == 1:
        lo = gpus[0] * CPUS_PER_GPU
        cmd = ["taskset", "-c", f"{lo}-{lo + CPUS_PER_GPU - 1}", *cmd]
    return cmd


def main(job_file: str, gpus: str = "") -> None:
    """Run every job of ``job_file`` on ``gpus`` (comma-separated, default all)."""
    jobs = []
    for line in Path(job_file).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            count, name, *args = shlex.split(line)
            jobs.append((int(count), name, args))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    free = [int(g) for g in gpus.split(",")] if gpus else list(range(N_GPUS))
    running: list[tuple[subprocess.Popen[bytes], list[int], str]] = []
    failed = []
    while jobs or running:
        for proc, gpus, name in list(running):
            if proc.poll() is not None:
                running.remove((proc, gpus, name))
                free = sorted(free + gpus)
                logger.info("done %s (%d)", name, proc.returncode)
                if proc.returncode:
                    failed.append(name)
        while jobs and jobs[0][0] <= len(free):
            n, name, args = jobs.pop(0)
            gpus, free = free[:n], free[n:]
            log = (HERE / "logs" / f"{name}.log").open("w")
            log.write(
                f"# {datetime.now(UTC).isoformat()} gpus {gpus}: {shlex.join(args)}\n"
            )
            log.flush()
            proc = subprocess.Popen(  # noqa: S603
                command(gpus, args),
                cwd=REPO,
                env=environment(gpus),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, gpus, name))
            logger.info("start %s on GPUs %s", name, gpus)
        time.sleep(2)
    logger.info("failed: %s", failed or "none")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    main(*sys.argv[1:])

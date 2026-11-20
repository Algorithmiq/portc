"""Out-of-core SRC of ``N . V . M . U`` with a large ``M``, read from disk.

Four commands:

- ``generate``: write the four MPOs site by site as ``.npy`` files, so that ``M``
  is never whole in memory.
- ``run``: compress the stack with automatic (or given) budgets, on one or several
  GPUs, and report the plan, the wall time, the device pool sizes, the host peak
  and the stall time.
- ``compare``: run twice with two GPU budgets and report the relative distance
  between the two outputs, which checks that the plan does not change the result.
- ``scaling``: run on 1, 2, 4, ... GPUs of the node and report the wall time, the
  parallel efficiency and the distance to the single-GPU output.

The plan, the per-pass times and the stall time are also logged by `portc`
itself at ``DEBUG``; pass ``--debug`` to see them.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import re
import resource
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import cyclopts
import numpy as np

import portc._sweep as sweep_module
from portc import Resources, src
from portc._plan import make_plan

logger = logging.getLogger(__name__)
app = cyclopts.App(help="Benchmark out-of-core SRC of N.V.M.U with a large M.")

LAYERS = ("N", "V", "M", "U")
PHYS = 4  # Pauli transfer matrix legs
Precision = Literal["float32", "float64", "complex64", "complex128"]


def _site_shape(j: int, n_sites: int, bond: int) -> tuple[int, ...]:
    left = () if j == 0 else (bond,)
    right = () if j == n_sites - 1 else (bond,)
    return (*left, *right, PHYS, PHYS)


@app.command
def generate(
    directory: Path,
    *,
    n_sites: int = 50,
    bond: int = 4,
    bond_m: int = 4000,
    dtype: Precision = "complex128",
    seed: int = 0,
) -> None:
    """Write random MPOs ``N``, ``V``, ``M``, ``U`` as one ``.npy`` file per site.

    The draws are made in double precision and rounded to ``dtype``, so a seed gives
    the same stack in every precision, up to that rounding.

    Args:
        directory: Where to write, ideally node-local NVMe.
        n_sites: Number of sites.
        bond: Bond dimension of ``N``, ``V`` and ``U``.
        bond_m: Bond dimension of ``M``.
        dtype: The data type of the sites.
        seed: Seed of the random draws.
    """
    rng = np.random.default_rng(seed)
    kind = np.dtype(dtype)
    for name in LAYERS:
        chi = bond_m if name == "M" else bond
        (directory / name).mkdir(parents=True, exist_ok=True)
        # Every layer, real or complex, draws entries of variance 1 / (chi * PHYS).
        # A sketched environment then grows by 2 per site in norm and the projected
        # one shrinks by about as much, so both stay within single precision for
        # over a hundred sites.
        scale = 1 / np.sqrt(chi * PHYS * (2 if kind.kind == "c" else 1))
        for j in range(n_sites):
            shape = _site_shape(j, n_sites, chi)
            site = np.lib.format.open_memmap(
                directory / name / f"{j:04d}.npy", mode="w+", dtype=kind, shape=shape
            )
            for row in range(shape[0]):  # one left-bond slice at a time
                draw = rng.normal(size=shape[1:])
                if kind.kind == "c":
                    draw = draw + 1j * rng.normal(size=shape[1:])
                site[row] = draw * scale
            site.flush()
            del site
        logger.info("Layer %s written: bond %d", name, chi)


def _load(directory: Path) -> list[list[np.ndarray]]:
    return [
        [
            np.load(path, mmap_mode="r")
            for path in sorted((directory / name).glob("*.npy"))
        ]
        for name in LAYERS
    ]


def _run(
    directory: Path,
    chi_out: int,
    resources: Resources,
    seed: int,
    device: str = "gpu",
) -> tuple[list[np.ndarray], float, dict[str, object]]:
    """Compress the stack once; return the output, the wall time and a run record."""
    layers = _load(directory)
    plans = []

    def spy(*args: object, **kwargs: object) -> object:
        plans.append(make_plan(*args, **kwargs))
        return plans[-1]

    sweep_module.make_plan = spy  # record the plan of the run
    try:
        start = perf_counter()
        out = src(
            *layers,
            chi_out=chi_out,
            dtype=layers[0][0].dtype,
            seed=seed,
            device=device,
            resources=resources,
        )
        seconds = perf_counter() - start
    finally:
        sweep_module.make_plan = make_plan
    (plan,) = plans
    tiers = [site.tier for site in plan.sites]
    pools = []
    if device == "gpu":
        import cupy  # noqa: PLC0415  (optional dependency)

        devices = resources.devices or 1
        for gpu in range(devices) if isinstance(devices, int) else devices:
            with cupy.cuda.Device(gpu):
                pools.append(cupy.get_default_memory_pool().total_bytes())
    record = {
        "seconds": seconds,
        "dtype": str(layers[0][0].dtype),
        "chi_out": chi_out,
        "pools": pools,
        "host_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "planned_device_peak": plan.device_peak,
        "planned_host_peak": plan.host_peak,
        "disk_bytes": plan.disk_bytes,
        "tiers": {tier: tiers.count(tier) for tier in ("device", "host", "disk")},
        "sketch_batches": sorted({site.sketch_batch for site in plan.sites[1:]}),
    }
    logger.info(
        "Run complete: %.1f s, pools %s B, host peak %d B, planned device peak %d B, "
        "planned host peak %d B, disk %d B, tiers %s, sketch batches %s",
        seconds,
        pools,
        record["host_peak"],
        plan.device_peak,
        plan.host_peak,
        plan.disk_bytes,
        record["tiers"],
        record["sketch_batches"],
    )
    return out, seconds, record


def _spill(out: list[np.ndarray], directory: Path) -> list[np.ndarray]:
    """Write the output sites to ``directory`` and return them memory-mapped."""
    for j, site in enumerate(out):
        np.save(directory / f"{j:04d}.npy", site)
    return [np.load(path, mmap_mode="r") for path in sorted(directory.glob("*.npy"))]


def _release(directories: list[Path], device: str) -> None:
    """Return the memory a run leaves behind, so the next run starts on a clean node.

    On the GH200 the GPU memory is also a host NUMA node, so host pages left over
    from a run, including the page cache of the memory-mapped files, take memory
    from the GPUs of the next one.
    """
    gc.collect()
    if device == "gpu":
        import cupy  # noqa: PLC0415  (optional dependency)

        for gpu in range(cupy.cuda.runtime.getDeviceCount()):
            with cupy.cuda.Device(gpu):
                cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    for path in (p for d in directories for p in d.rglob("*.npy")):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _log_free(device: str) -> None:
    """Log the free memory of every GPU and every non-empty NUMA node."""
    gpus = []
    if device == "gpu":
        import cupy  # noqa: PLC0415  (optional dependency)

        for gpu in range(cupy.cuda.runtime.getDeviceCount()):
            with cupy.cuda.Device(gpu):
                gpus.append(cupy.cuda.runtime.memGetInfo()[0])
    nodes = {}
    for meminfo in Path("/sys/devices/system/node").glob("node*/meminfo"):
        match = re.search(r"MemFree:\s+(\d+) kB", meminfo.read_text())
        if match and int(match[1]):
            nodes[meminfo.parent.name] = int(match[1]) * 1024
    logger.info("Free before the run: GPUs %s B, NUMA nodes %s B", gpus, nodes)


def _append(results: Path | None, record: dict[str, object]) -> None:
    if results is not None:
        with results.open("a") as f:
            f.write(json.dumps(record) + "\n")


@app.command
def run(
    directory: Path,
    *,
    chi_out: int = 2000,
    gpu_memory: str | None = None,
    host_memory: str | None = None,
    scratch_dir: Path | None = None,
    devices: int = 1,
    seed: int = 0,
    results: Path | None = None,
    save: Path | None = None,
) -> None:
    """Compress the stack written by ``generate`` on the GPU.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        gpu_memory: GPU budget of each device, e.g. ``36GB``; detected when omitted.
        host_memory: Host budget; detected when omitted.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        devices: The number of GPUs to split the sweep among.
        seed: Seed of the sketch.
        results: A JSON-lines file to append the run record to.
        save: An ``.npz`` file to save the output sites to, ``site_0000``, ...
    """
    resources = Resources(gpu_memory, host_memory, scratch_dir, devices=devices)
    out, _, record = _run(directory, chi_out, resources, seed)
    _append(results, {"gpus": devices, "seed": seed, **record})
    if save is not None:
        np.savez(save, **{f"site_{j:04d}": site for j, site in enumerate(out)})


def _inner(a: list[np.ndarray], b: list[np.ndarray]) -> complex:
    """Frobenius inner product ``<a, b>`` of two MPOs, site by site."""
    env = np.einsum("rud,sud->rs", a[0].conj(), b[0])
    for x, y in zip(a[1:-1], b[1:-1]):
        env = np.einsum("rs,rtud,svud->tv", env, x.conj(), y, optimize=True)
    return complex(np.einsum("rs,rud,sud->", env, a[-1].conj(), b[-1]))


@app.command
def compare(
    directory: Path,
    *,
    chi_out: int = 500,
    small: str = "40GB",
    large: str = "80GB",
    scratch_dir: Path | None = None,
    seed: int = 0,
) -> None:
    """Run with two GPU budgets and report the relative distance of the outputs.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        small: The smaller GPU budget.
        large: The larger GPU budget.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        seed: Seed of the sketch, the same for both runs.
    """
    first, _, _ = _run(directory, chi_out, Resources(small, None, scratch_dir), seed)
    second, _, _ = _run(directory, chi_out, Resources(large, None, scratch_dir), seed)
    logger.info("Relative distance between the runs: %.3e", _distance(first, second))


def _distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Relative Frobenius distance ``|a - b| / |a|`` of two MPOs."""
    aa, bb, ab = _inner(a, a), _inner(b, b), _inner(a, b)
    return float(np.sqrt(max((aa + bb - 2 * ab).real, 0.0) / aa.real))


def _qr_distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Relative distance ``|a - b| / |a|`` from a QR sweep over the MPO ``a - b``.

    `_distance` cancels down to about ``sqrt(eps)``. Here ``a - b`` is a direct sum
    of the two trains, and the QR sweep keeps rounding relative to ``|a|``, which
    resolves differences down to ``eps``. Gauge-free, unlike comparing cores.
    """
    n = len(a)
    carry = np.ones((1, 1), dtype=np.result_type(a[0], b[0]))
    for j, (x, y) in enumerate(zip(a, b)):
        x = x[None] if j == 0 else x[:, None] if j == n - 1 else x
        y = y[None] if j == 0 else y[:, None] if j == n - 1 else y
        if j == 0:
            site = np.concatenate([x, -y], axis=1)
        elif j == n - 1:
            site = np.concatenate([x, y], axis=0)
        else:
            (la, ra, *phys), (lb, rb, _, _) = x.shape, y.shape
            site = np.zeros((la + lb, ra + rb, *phys), dtype=carry.dtype)
            site[:la, :ra], site[la:, ra:] = x, y
        site = np.tensordot(carry, site, axes=(1, 0)).transpose(0, 2, 3, 1)
        carry = np.linalg.qr(site.reshape(-1, site.shape[-1]), mode="r")
    return float(np.linalg.norm(carry) / np.sqrt(_inner(a, a).real))


@app.command
def scaling(
    directory: Path,
    *,
    chi_out: int = 2000,
    devices: Annotated[tuple[int, ...], cyclopts.Parameter(consume_multiple=True)] = (
        1,
        2,
        4,
        8,
    ),
    gpu_memory: str | None = None,
    host_memory: str | None = None,
    scratch_dir: Path | None = None,
    results: Path | None = None,
    seed: int = 0,
    device: str = "gpu",
) -> None:
    """Run on increasing numbers of GPUs and report the strong scaling.

    The first count is the reference: the efficiency of ``G`` GPUs is
    ``T_ref * G_ref / (T_G * G)``, and every output is compared with the
    reference output. The reference output is kept in ``scratch_dir``, and the
    memory of each run is released before the next one.

    Args:
        directory: The directory given to ``generate``.
        chi_out: The output bond dimension.
        devices: The GPU counts, the reference first.
        gpu_memory: GPU budget of each device; detected when omitted.
        host_memory: Host budget of the node; detected when omitted.
        scratch_dir: Where to spill environments; ``$TMPDIR`` when omitted.
        results: A JSON-lines file to append one record per run to.
        seed: Seed of the sketch, the same for every run.
        device: ``gpu``, or ``cpu`` to smoke-test with simulated devices.
    """
    reference: list[np.ndarray] | None = None
    eps = 0.0
    ref_seconds = ref_devices = 0.0
    spill = tempfile.TemporaryDirectory(dir=scratch_dir)
    for count in devices:
        _release([directory, Path(spill.name)], device)
        _log_free(device)
        resources = Resources(gpu_memory, host_memory, scratch_dir, devices=count)
        out, seconds, record = _run(directory, chi_out, resources, seed, device)
        if reference is None:
            # Kept on disk, so it holds no memory during the following runs.
            reference, ref_seconds, ref_devices = (
                _spill(out, Path(spill.name)),
                seconds,
                count,
            )
            eps = float(np.finfo(out[0].dtype).eps)
            distance = qr_distance = 0.0
        else:
            distance = _distance(reference, out)
            qr_distance = _qr_distance(reference, out)
        efficiency = ref_seconds * ref_devices / (seconds * count)
        logger.info(
            "GPUs %d: %.1f s, speed-up %.2f, efficiency %.0f %%, distance %.3e, "
            "QR distance %.3e (eps %.1e)",
            count,
            seconds,
            ref_seconds / seconds,
            100 * efficiency,
            distance,
            qr_distance,
            eps,
        )
        _append(
            results,
            {
                "gpus": count,
                "seed": seed,
                **record,
                "efficiency": efficiency,
                "distance": distance,
                "qr_distance": qr_distance,
                "eps": eps,
            },
        )
        del out
    spill.cleanup()


@app.meta.default
def main(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    debug: bool = False,
) -> None:
    """Configure logging, then dispatch to a command.

    Args:
        tokens: The command and its arguments.
        debug: Also show the `portc` debug log: plan, pass times and stalls.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if debug:
        logging.getLogger("portc").setLevel(logging.DEBUG)
    app(tokens)


if __name__ == "__main__":
    app.meta()

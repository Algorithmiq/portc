"""Trace one `bench_large` run: per-site GPU time of kernels, collectives and QR.

Wraps, without changing what runs, the contractions (`_Contractions.__call__`), the
collectives of `DeviceGroup`, the QR of rank 0, the distributed `cholesky_qr` and `SiteSource.next` with CUDA
events recorded on each rank's stream. Events do not synchronize, so the overlap of
the sweep is kept; the intervals are read once the run is over.

For each rank, pass and site, the GPU time between two consecutive site boundaries
splits into:

- ``kernels``: the contractions, with their real flops along the executed path
  (opt_einsum's count times 4 for complex dtypes, 1 for real ones);
- ``gather``, ``scatter``, ``alltoall``, ``allgather_s``, ``allgather_cores``,
  ``allgather_gram``: from the call to the end of the copies, which includes the
  stream waiting for the other ranks;
- ``qr``: `truncated_qr` on rank 0;
- ``cholesky_qr``: `cholesky_qr` on every rank, ``allgather_gram`` included;
- ``other``: the rest, i.e. uploads, copies of the output core to the host, and the
  stream idling while its host thread waits (barriers, environment reads).

Usage::

    python trace.py STACK --chi-out 2000 --devices 4 --out trace.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from functools import cache
from pathlib import Path

import cupy as cp
import numpy as np
import opt_einsum as oe

sys.path.insert(0, str(Path(__file__).parents[1]))

import bench_large  # noqa: E402

import portc._group as group_module  # noqa: E402
import portc._kernels as kernels_module  # noqa: E402
import portc._sites as sites_module  # noqa: E402
import portc._sweep as sweep_module  # noqa: E402
from portc import Resources  # noqa: E402

_local = threading.local()
# Per device id: (pass, site, label, start event, end event, flops).
_records: dict[int, list[tuple]] = defaultdict(list)


def _event() -> cp.cuda.Event:
    event = cp.cuda.Event()
    event.record(cp.cuda.get_current_stream())
    return event


def _timed(label: str, flops: float = 0.0):  # noqa: ANN202
    """Context manager recording ``label`` on the current device's stream."""

    class _Ctx:
        def __enter__(self) -> None:
            self.start = _event()

        def __exit__(self, *exc: object) -> None:
            _records[cp.cuda.Device().id].append(
                (
                    getattr(_local, "pass_", "?"),
                    getattr(_local, "site", -1),
                    label,
                    self.start,
                    _event(),
                    flops,
                )
            )

    return _Ctx()


@cache
def _flops(eq: str, shapes: tuple, optimize: object, complex_: bool) -> float:
    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize=optimize)
    return float(info.opt_cost) * (4 if complex_ else 1)


def _install() -> None:
    contract = kernels_module._Contractions.__call__
    eqs = kernels_module.equations(4)
    names = {
        eqs.ltr: "env",
        eqs.rtl_m: "sketch",
        eqs.rtl_s: "project",
        eqs.first: "first",
    }

    def traced_contract(self, eq, *operands):  # noqa: ANN001, ANN202
        shapes = tuple(op.shape for op in operands)
        complex_ = np.dtype(operands[0].dtype).kind == "c"
        flops = _flops(eq, shapes, self._optimize, complex_)
        with _timed(names.get(eq, "kernel"), flops):
            return contract(self, eq, *operands)

    kernels_module._Contractions.__call__ = traced_contract

    for name in ("gather", "scatter", "allgather", "alltoall"):
        original = getattr(group_module.DeviceGroup, name)

        def traced(self, *args, _original=original, _name=name, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            label = _name
            if _name == "allgather":
                caller = sys._getframe(1).f_code.co_name  # noqa: SLF001
                labels = {"_upload": "allgather_cores", "gram": "allgather_gram"}
                label = labels.get(caller, "allgather_s")
            with _timed(label):
                return _original(self, *args, **kwargs)

        setattr(group_module.DeviceGroup, name, traced)

    qr = sweep_module.truncated_qr

    def traced_qr(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with _timed("qr"):
            return qr(*args, **kwargs)

    sweep_module.truncated_qr = traced_qr

    cqr = sweep_module.cholesky_qr

    def traced_cqr(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with _timed("cholesky_qr"):
            return cqr(*args, **kwargs)

    sweep_module.cholesky_qr = traced_cqr

    for pass_name in ("_left_to_right", "_right_to_left"):
        original = getattr(sweep_module._Sweep, pass_name)

        def traced_pass(self, *args, _original=original, _pass=pass_name, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            _local.pass_ = _pass.strip("_")
            _local.site = -1
            return _original(self, *args, **kwargs)

        setattr(sweep_module._Sweep, pass_name, traced_pass)

    next_site = sites_module.SiteSource.next

    def traced_next(self, rank):  # noqa: ANN001, ANN202
        cores = next_site(self, rank)
        _local.site = getattr(_local, "site", -1) + 1
        _records[cp.cuda.Device().id].append(
            (
                getattr(_local, "pass_", "?"),
                _local.site,
                "boundary",
                _event(),
                None,
                0.0,
            )
        )
        return cores

    sites_module.SiteSource.next = traced_next


def _summarize() -> dict[str, object]:
    """Per device, pass and site: seconds by label, flops, and the site's GPU span."""
    out: dict[str, object] = {}
    for device, records in sorted(_records.items()):
        with cp.cuda.Device(device):
            cp.cuda.Device(device).synchronize()
            boundaries = [r for r in records if r[2] == "boundary"]
            sites: dict[tuple[str, int], dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            for pass_, site, label, start, end, flops in records:
                if label == "boundary":
                    continue
                sites[pass_, site][label] += cp.cuda.get_elapsed_time(start, end) * 1e-3
                if flops:
                    sites[pass_, site]["flops"] += flops
            # Span of a site: from its boundary to the next one on this device.
            for (pass_, site, _, event, _, _), nxt in zip(boundaries, boundaries[1:]):
                span = cp.cuda.get_elapsed_time(event, nxt[3]) * 1e-3
                sites[pass_, site]["span"] = span
            out[str(device)] = [
                {"pass": p, "site": s, **values} for (p, s), values in sites.items()
            ]
    return out


def main() -> None:
    """Run the traced sweep and write the run record and the trace as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chi-out", type=int, default=2000)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--gpu-memory")
    parser.add_argument("--host-memory")
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reserve", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import logging  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("portc").setLevel(logging.DEBUG)
    _install()
    resources = Resources(
        args.gpu_memory, args.host_memory, args.scratch_dir, devices=args.devices
    )
    _, _, record = bench_large._run(
        args.directory, args.chi_out, resources, args.seed, reserve=args.reserve
    )
    record = {"gpus": args.devices, "seed": args.seed, **record, "trace": _summarize()}
    args.out.write_text(json.dumps(record))


if __name__ == "__main__":
    main()

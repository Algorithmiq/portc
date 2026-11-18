"""Probe a multi-GPU node for the SRC scaling runs: peer access, copy bandwidths, FP32 mode.

Prints one JSON object with:

- ``peer``: ``deviceCanAccessPeer(i, j)`` for every ordered pair;
- ``pair_gbs``: peer-copy bandwidth of every ordered pair, alone;
- ``allgather_gbs``: per-receiver bandwidth when the first ``G`` GPUs each pull a
  block from every other at the same time, one stream per receiver as in
  `portc._group.DeviceGroup.allgather`;
- ``gather_gbs`` / ``scatter_gbs``: the same towards and from GPU 0 alone;
- ``alltoall_gbs``: per-receiver bandwidth when the first ``G`` GPUs each pull a
  ``1/G`` slice of the block from every other at the same time, as in
  `portc._group.DeviceGroup.alltoall`;
- ``h2d_gbs`` / ``d2h_gbs``: pinned host copies per GPU, alone and all at once;
- ``cholesky_qr``: seconds of `portc.utils.linalg.cholesky_qr` on GPU 0 for one
  rank's ``1/G`` of a complex64 ``rows x n`` sketch, without communication, and
  the number of Gram matrices it formed;
- ``matmul``: relative error of a complex64 ``4096 x 4096`` matmul against a
  complex128 reference, and the complex64 and complex128 GEMM rates.

Bandwidths are in GB/s (1e9 bytes), medians over ``--reps`` timed repetitions after
one warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np

from portc.utils.linalg import cholesky_qr

GB = 1e9


def _sync_all(n: int) -> None:
    for d in range(n):
        with cp.cuda.Device(d):
            cp.cuda.Device(d).synchronize()


def _timed(enqueue: object, n_devices: int, reps: int) -> list[float]:
    """Run ``enqueue()`` (which only enqueues work) once to warm up, then time it."""
    times = []
    for rep in range(reps + 1):
        _sync_all(n_devices)
        start = perf_counter()
        enqueue()
        _sync_all(n_devices)
        if rep:
            times.append(perf_counter() - start)
    return times


def _peer_copy(dst: cp.ndarray, src: cp.ndarray, stream: cp.cuda.Stream) -> None:
    cp.cuda.runtime.memcpyPeerAsync(
        dst.data.ptr, dst.device.id, src.data.ptr, src.device.id, dst.nbytes, stream.ptr
    )


def main() -> None:
    """Run every probe and print the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-gb", type=float, default=1.0)
    parser.add_argument("--host-gb", type=float, default=4.0)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--cqr-rows", type=int, default=32000)
    parser.add_argument("--cqr-n", type=int, default=2000)
    args = parser.parse_args()
    n = cp.cuda.runtime.getDeviceCount()
    nbytes = int(args.block_gb * GB)
    out: dict[str, object] = {"block_bytes": nbytes, "reps": args.reps}

    runtime = cp.cuda.runtime
    out["peer"] = [
        [i == j or bool(runtime.deviceCanAccessPeer(i, j)) for j in range(n)]
        for i in range(n)
    ]
    for i in range(n):
        with cp.cuda.Device(i):
            for j in range(n):
                if i != j and runtime.deviceCanAccessPeer(i, j):
                    try:
                        runtime.deviceEnablePeerAccess(j)
                    except runtime.CUDARuntimeError:
                        pass

    # One send and one receive buffer per GPU, and enough receive slots for an
    # allgather of every peer.
    send, recv, streams = [], [], []
    for d in range(n):
        with cp.cuda.Device(d):
            send.append(cp.ones(nbytes, dtype=cp.uint8))
            recv.append(cp.empty((n, nbytes), dtype=cp.uint8))
            streams.append(cp.cuda.Stream(non_blocking=True))

    pair = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            def one(i: int = i, j: int = j) -> None:
                with cp.cuda.Device(j):
                    _peer_copy(recv[j][i], send[i], streams[j])

            pair[i][j] = nbytes / median(_timed(one, n, args.reps)) / GB
    out["pair_gbs"] = pair  # pair[src][dst]

    def collective(g: int, kind: str) -> float:
        def enqueue() -> None:
            for dst in range(g):
                with cp.cuda.Device(dst):
                    for src in range(g):
                        if src == dst:
                            continue
                        if kind == "gather" and dst != 0:
                            continue
                        if kind == "scatter" and src != 0:
                            continue
                        if kind == "alltoall":
                            _peer_copy(
                                recv[dst][src][: nbytes // g],
                                send[src][: nbytes // g],
                                streams[dst],
                            )
                            continue
                        _peer_copy(recv[dst][src], send[src], streams[dst])

        seconds = median(_timed(enqueue, n, args.reps))
        # Bytes per receiver: g - 1 blocks for the allgather and the gather, one
        # for each receiver of the scatter, g - 1 slices of 1/g for the alltoall.
        per_receiver = nbytes * (1 if kind == "scatter" else g - 1)
        if kind == "alltoall":
            per_receiver = nbytes // g * (g - 1)
        return per_receiver / seconds / GB

    counts = [g for g in (2, 4, 8) if g <= n]
    for kind in ("allgather", "gather", "scatter", "alltoall"):
        out[f"{kind}_gbs"] = {g: collective(g, kind) for g in counts}

    # Pinned host copies.
    hbytes = int(args.host_gb * GB)
    host = []
    dev = []
    for d in range(n):
        with cp.cuda.Device(d):
            mem = cp.cuda.alloc_pinned_memory(hbytes)
            arr = np.frombuffer(mem, dtype=np.uint8, count=hbytes)
            arr[:] = 1
            host.append((mem, arr))
            dev.append(cp.empty(hbytes, dtype=cp.uint8))

    def host_copy(devices: list[int], kind: str) -> None:
        for d in devices:
            with cp.cuda.Device(d):
                h = host[d][1].ctypes.data
                if kind == "h2d":
                    runtime.memcpyAsync(
                        dev[d].data.ptr, h, hbytes, runtime.memcpyHostToDevice,
                        streams[d].ptr,
                    )
                else:
                    runtime.memcpyAsync(
                        h, dev[d].data.ptr, hbytes, runtime.memcpyDeviceToHost,
                        streams[d].ptr,
                    )

    for kind in ("h2d", "d2h"):
        alone = [
            hbytes / median(_timed(lambda d=d: host_copy([d], kind), n, args.reps)) / GB
            for d in range(n)
        ]
        together = {}
        for g in counts:
            s = median(_timed(lambda g=g: host_copy(list(range(g)), kind), n, args.reps))
            together[g] = {"per_gpu": hbytes / s / GB, "aggregate": g * hbytes / s / GB}
        out[f"{kind}_gbs"] = {"alone": alone, "together": together}
    del host, dev, send, recv
    _sync_all(n)
    for d in range(n):
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()

    # FP32 math mode: a complex64 matmul against a complex128 reference.
    with cp.cuda.Device(0):
        rng = np.random.default_rng(0)
        size = 4096
        a = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        b = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        a32, b32 = cp.asarray(a, cp.complex64), cp.asarray(b, cp.complex64)
        # The reference uses the rounded inputs, so it isolates the product.
        ref = cp.asarray(a32, cp.complex128) @ cp.asarray(b32, cp.complex128)
        err = float(cp.linalg.norm(a32 @ b32 - ref) / cp.linalg.norm(ref))
        rates = {}
        for dtype in (cp.complex64, cp.complex128):
            big = 8192
            x = cp.ones((big, big), dtype=dtype)
            times = _timed(lambda x=x: x @ x, 1, 3)
            rates[np.dtype(dtype).name] = 8 * big**3 / median(times) / 1e12
        out["matmul"] = {
            "relative_error_complex64": err,
            "eps_complex64": float(np.finfo(np.complex64).eps),
            "tflops": rates,
            "CUPY_TF32": os.environ.get("CUPY_TF32"),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        }
        n_cols = args.cqr_n
        cqr = {}
        for g in [1, *counts]:
            block = cp.asarray(
                rng.normal(size=(n_cols, args.cqr_rows // g))
                + 1j * rng.normal(size=(n_cols, args.cqr_rows // g)),
                cp.complex64,
            )
            calls = []

            def gram(x: cp.ndarray, calls: list[int] = calls) -> cp.ndarray:
                calls.append(1)
                return x

            def factor(block: cp.ndarray = block, gram: object = gram) -> None:
                cholesky_qr(block, args.cqr_rows, 0.0, gram)

            seconds = median(_timed(factor, 1, args.reps))
            cqr[g] = {"seconds": seconds, "grams": len(calls) // (args.reps + 1)}
            del block
        out["cholesky_qr"] = {"rows": args.cqr_rows, "n": n_cols, "G": cqr}
    print(json.dumps(out))


if __name__ == "__main__":
    main()

"""Where the sweep keeps its environments: on the device, in host memory or on disk.

The right-to-left pass reads the environments in the reverse order of the
left-to-right pass that writes them, one batch of sketch columns at a time. Off the
device, every batch moves through a small ring of page-locked buffers: a writer
thread drains device-to-host copies into host arrays or files, and a reader thread
fills buffers ahead of use, so that transfers overlap the kernels.
"""

from __future__ import annotations

import os
import queue
import shutil
import uuid
import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from array_api_compat import array_namespace

from .utils import current_stream, pinned_empty, to_device_async, to_host_async

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path
    from types import ModuleType, TracebackType

    from ._plan import Plan
    from .utils import Array

# Page-locked buffers per direction: one in use, one in flight.
RING_SIZE = 2
# How often a blocked reader checks whether the store is closing.
_POLL_SECONDS = 0.1


class _Closing(Exception):  # noqa: N818  (control flow, not an error)
    """Raised in the reader thread when the store closes under it."""


class EnvironmentStore:
    """Keep the environments ``C_j`` of a sweep, batch by batch.

    Use as a context manager: leaving it waits for pending writes, stops the
    threads and removes the scratch files, also after an exception. The device
    arrays it returns are of the backend of those it was given.

    Args:
        plan: The plan of the sweep, which fixes the tier and batches of each site.
        env_shapes: The shape ``(chi, *left_bonds)`` of ``C_j`` for every site.
        dtype: The data type of the environments.
        scratch_dir: Where the disk tier creates its per-process directory.
        copy_stream: The stream that performs the transfers.
    """

    def __init__(
        self,
        plan: Plan,
        env_shapes: Sequence[tuple[int, ...]],
        dtype: type | np.dtype,
        scratch_dir: Path,
        *,
        copy_stream: Any,  # noqa: ANN401  (a cupy or null stream)
    ) -> None:
        self._plan = plan
        self._tiers = [site.tier for site in plan.sites]
        self._shapes = [tuple(shape) for shape in env_shapes]
        self._dtype = np.dtype(dtype)
        # The namespace of the staged arrays, known from the first staged put.
        self._xp: ModuleType | None = None
        self._copy = copy_stream
        self._device: dict[int, Array] = {}
        self._host: dict[int, np.ndarray] = {}
        self._fds: dict[int, int] = {}
        self._writes: dict[int, list[Future[None]]] = {}
        self._ahead: deque[tuple[int, int, int]] = deque()
        self._inflight: deque[tuple[int, int, int, Future[tuple[Any, np.ndarray]]]] = (
            deque()
        )
        self._closing = False
        self.stall_seconds = 0.0

        self._dir: Path | None = None
        if "disk" in self._tiers:
            self._dir = scratch_dir / f"src-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            self._dir.mkdir(parents=True)
            self._finalizer = weakref.finalize(
                self, shutil.rmtree, self._dir, ignore_errors=True
            )

        staged = [j for j, tier in enumerate(self._tiers) if j > 0 and tier != "device"]
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="src-env-writer")
        self._reader = ThreadPoolExecutor(1, thread_name_prefix="src-env-reader")
        self._out: queue.Queue[Any] = queue.Queue()
        self._in: queue.Queue[tuple[Any, Any]] = queue.Queue()
        # The page-locked rings are allocated by the first staged put.
        self._ring_bytes = max(
            (self._batch_rows(j) * self._row_bytes(j) for j in staged), default=0
        )

    def _row_bytes(self, j: int) -> int:
        return prod(self._shapes[j][1:]) * self._dtype.itemsize

    def _batch_rows(self, j: int) -> int:
        sites = self._plan.sites
        return max(sites[j - 1].env_batch, sites[j].env_batch, sites[j].sketch_batch)

    def __enter__(self) -> Self:
        """Return the store."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the store; see `close`."""
        self.close(wait=exc is None)

    def close(self, *, wait: bool = True) -> None:
        """Stop the threads and remove the scratch files.

        Args:
            wait: Wait for the pending writes and re-raise their first error.
        """
        try:
            if wait:
                for futures in self._writes.values():
                    for future in futures:
                        future.result()
        finally:
            self._closing = True
            self._writer.shutdown(wait=True, cancel_futures=True)
            self._reader.shutdown(wait=True, cancel_futures=True)
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            self._device.clear()
            self._host.clear()
            if self._dir is not None:
                self._finalizer()

    # --- writing ---

    def put(self, j: int, lo: int, hi: int, x: Array) -> None:
        """Store columns ``lo:hi`` of ``C_j``.

        Args:
            j: The site.
            lo: The first column.
            hi: One past the last column.
            x: The columns, ``(hi - lo, *left_bonds)``, on the device.
        """
        if self._tiers[j] == "device":
            if j not in self._device:
                xp = array_namespace(x)
                self._device[j] = xp.empty(self._shapes[j], dtype=self._dtype)
            self._device[j][lo:hi] = x
            return
        xp = self._xp = array_namespace(x)
        self._raise_failed()
        self._open(j)
        if self._ring_bytes:
            for _ in range(RING_SIZE):
                self._out.put(pinned_empty(self._ring_bytes, xp))
                self._in.put((pinned_empty(self._ring_bytes, xp), None))
            self._ring_bytes = 0
        start = perf_counter()
        buffer = self._out.get()
        self.stall_seconds += perf_counter() - start
        host = buffer[: x.nbytes].view(self._dtype).reshape(x.shape)
        # The copy starts once the kernel that wrote x is done.
        self._copy.wait_event(current_stream(xp).record())
        to_host_async(x, host, self._copy)
        done = self._copy.record()
        # x stays referenced by the task until the copy is complete.
        future = self._writer.submit(
            self._write, j, lo, x, host=host, done=done, buffer=buffer
        )
        self._writes.setdefault(j, []).append(future)

    def _open(self, j: int) -> None:
        if self._tiers[j] == "host" and j not in self._host:
            self._host[j] = np.empty(self._shapes[j], dtype=self._dtype)
        elif self._tiers[j] == "disk" and j not in self._fds:
            fd = os.open(self._path(j), os.O_RDWR | os.O_CREAT, 0o600)
            self._fds[j] = fd
            os.ftruncate(fd, prod(self._shapes[j]) * self._dtype.itemsize)

    def _path(self, j: int) -> Path:
        assert self._dir is not None  # noqa: S101  (the disk tier creates it)
        return self._dir / f"env-{j:04d}.bin"

    def _write(
        self,
        j: int,
        lo: int,
        x: Array,
        *,
        host: np.ndarray,
        done: Any,  # noqa: ANN401  (a cupy or null event)
        buffer: Any,  # noqa: ANN401  (a page-locked buffer)
    ) -> None:
        try:
            done.synchronize()
            del x
            if self._tiers[j] == "host":
                np.copyto(self._host[j][lo : lo + host.shape[0]], host)
                return
            offset = lo * self._row_bytes(j)
            try:
                _write_all(self._fds[j], memoryview(buffer[: host.nbytes]), offset)
            except OSError as err:
                msg = (
                    f"Writing environment {j} to {self._path(j)} failed "
                    f"({err.strerror}); the plan spills {self._plan.disk_bytes} bytes "
                    f"to {self._dir}."
                )
                raise OSError(err.errno, msg) from err
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    self._fds[j], offset, host.nbytes, os.POSIX_FADV_DONTNEED
                )
        finally:
            self._out.put(buffer)

    def _raise_failed(self) -> None:
        """Re-raise the first error of a finished write, dropping finished writes."""
        for j, futures in self._writes.items():
            pending = []
            for future in futures:
                if not future.done():
                    pending.append(future)
                elif (error := future.exception()) is not None:
                    raise error
            self._writes[j] = pending

    # --- reading ---

    def prefetch(self, j: int, ranges: Iterable[tuple[int, int]]) -> None:
        """Announce the batches of ``C_j`` that `get` will be asked for, in order.

        Off the device, the reader starts filling buffers with the first ones.

        Args:
            j: The site.
            ranges: The ``(lo, hi)`` column ranges, in the order they will be read.
        """
        if self._tiers[j] == "device":
            return
        self._ahead.extend((j, lo, hi) for lo, hi in ranges)
        self._schedule()

    def _schedule(self) -> None:
        while self._ahead and len(self._inflight) < RING_SIZE:
            j, lo, hi = self._ahead.popleft()
            writes = list(self._writes.get(j, []))
            future = self._reader.submit(self._read, j, lo, hi, writes)
            self._inflight.append((j, lo, hi, future))

    def _read(
        self, j: int, lo: int, hi: int, writes: Sequence[Future[None]]
    ) -> tuple[Any, np.ndarray]:
        for write in writes:
            write.result()
        while True:
            try:
                buffer, event = self._in.get(timeout=_POLL_SECONDS)
                break
            except queue.Empty:
                if self._closing:
                    raise _Closing from None
        if event is not None:
            event.synchronize()
        shape = (hi - lo, *self._shapes[j][1:])
        n_bytes = prod(shape) * self._dtype.itemsize
        host = buffer[:n_bytes].view(self._dtype).reshape(shape)
        if self._tiers[j] == "host":
            np.copyto(host, self._host[j][lo:hi])
        else:
            _read_all(
                self._fds[j], memoryview(buffer[:n_bytes]), lo * self._row_bytes(j)
            )
        return buffer, host

    def get(self, j: int, lo: int, hi: int) -> Array:
        """Return columns ``lo:hi`` of ``C_j`` on the device.

        Args:
            j: The site.
            lo: The first column.
            hi: One past the last column.

        Returns:
            The columns, ``(hi - lo, *left_bonds)``. Kernels on the current stream
            see them once their transfer is done.

        Raises:
            RuntimeError: If the batch is not the next one announced by `prefetch`.
        """
        if self._tiers[j] == "device":
            return self._device[j][lo:hi]
        self._raise_failed()
        if not self._inflight and not self._ahead:
            self.prefetch(j, [(lo, hi)])
        if not self._inflight or self._inflight[0][:3] != (j, lo, hi):
            msg = f"Batch {lo}:{hi} of environment {j} was read out of order."
            raise RuntimeError(msg)
        future = self._inflight.popleft()[3]
        start = perf_counter()
        buffer, host = future.result()
        self.stall_seconds += perf_counter() - start
        # On the current stream, so that the copy is ordered with the kernels that
        # use it and with the ones that freed the memory it reuses.
        xp = self._xp
        assert xp is not None  # noqa: S101  (C_j was put before it is read)
        stream = current_stream(xp)
        device = to_device_async(host, xp, stream)
        self._in.put((buffer, stream.record()))
        self._schedule()
        return device

    def drop(self, j: int) -> None:
        """Release ``C_j``: its memory, or its file."""
        for future in self._writes.pop(j, []):
            future.result()
        self._device.pop(j, None)
        self._host.pop(j, None)
        fd = self._fds.pop(j, None)
        if fd is not None:
            os.close(fd)
            self._path(j).unlink()


def _write_all(fd: int, data: memoryview, offset: int) -> None:
    """Write all of ``data`` at ``offset``; `os.pwrite` may write less."""
    while data:
        written = os.pwrite(fd, data, offset)
        data, offset = data[written:], offset + written


def _read_all(fd: int, data: memoryview, offset: int) -> None:
    """Fill ``data`` from ``offset``; `os.preadv` may read less."""
    while data:
        read = os.preadv(fd, [data], offset)
        if read == 0:
            msg = f"Unexpected end of file at offset {offset}."
            raise OSError(msg)
        data, offset = data[read:], offset + read

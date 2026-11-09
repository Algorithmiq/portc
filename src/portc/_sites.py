"""Lazy, per-site access to the cores of a stack.

A core is read from its source (a NumPy array, an `np.memmap`, a zarr or HDF5
dataset) only when the sweep needs it, and a background thread reads ahead, in
the order the sweep visits the sites. Shapes come from ``.shape`` alone, so
planning never reads any data.

With a group of several devices the cores are read once, into one page-locked
buffer, and shared by the ranks. Where the devices copy directly between them,
each rank uploads a slice of the buffer and gathers the rest from its peers, so the
host link carries every core once rather than once per device.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from time import perf_counter
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np

from ._group import DeviceGroup, GroupAbortedError, blocks
from ._tensor_train import known_kind, pad_site, padded_shape
from .utils import (
    NullEvent,
    current_stream,
    is_host,
    pinned_empty,
    to_device_async,
    to_numpy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from ._tensor_train import Site
    from .utils import Array

# Byte alignment of each core inside a staging buffer.
_ALIGN = 256
# How often a blocked loader checks whether the source is closing.
_POLL_SECONDS = 0.1


def padded_shapes(
    layers: Sequence[Sequence[Site]],
) -> list[tuple[tuple[int, ...], ...]]:
    """Return, for every site, the padded ``(l, r, u, d)`` shape of each layer.

    Args:
        layers: A stack in ket form.

    Returns:
        One tuple of shapes per site, without reading any data.
    """
    kinds = [known_kind(layer) for layer in layers]
    last = len(layers[0]) - 1
    return [
        tuple(
            padded_shape(tuple(layer[j].shape), kind, j, last)
            for layer, kind in zip(layers, kinds)
        )
        for j in range(last + 1)
    ]


def site_bytes(layers: Sequence[Sequence[Site]]) -> list[int]:
    """Return, for every site, the bytes of its cores over all layers."""
    return [
        sum(
            prod(layer[j].shape) * np.dtype(layer[j].dtype).itemsize for layer in layers
        )
        for j in range(len(layers[0]))
    ]


def sweep_order(n_sites: int) -> list[int]:
    """Return the sites in the order the sweep reads them.

    Left to right up to the last but one site, then right to left from the last.
    """
    return [*range(n_sites - 1), *range(n_sites - 1, -1, -1)]


def _aligned(n_bytes: int) -> int:
    return -(-n_bytes // _ALIGN) * _ALIGN


class _Staged(NamedTuple):
    """One site read into memory.

    Attributes:
        cores: Per layer, a host view into ``buffer`` (``None`` for a core given as
            a device array), or the padded host core on the host backend.
        device: Per layer, the core given as a device array, or ``None``.
        offsets: Per layer, the byte offset of its view in ``buffer``.
        n_bytes: The bytes of ``buffer`` in use.
        buffer: The staging buffer, or ``None`` on the host backend.
    """

    cores: list[Any]
    device: list[Any]
    offsets: list[int]
    n_bytes: int
    buffer: Any


class SiteSource:
    """Padded device cores of a stack, one site at a time, in sweep order.

    A loader thread reads the sites in ``order`` into a ring of staging buffers,
    ``depth`` sites ahead of the slowest rank. Each rank takes the sites in the
    same order with `next`; the host-to-device copies are issued on the rank's
    current stream, so they are ordered with the kernels that use them.

    Use as a context manager, so that the loader thread stops.

    Args:
        layers: A stack in ket form, each layer a sequence of array-likes with
            ``shape``, ``dtype``, ``ndim`` and ``np.asarray`` support.
        order: The sites, in the order they will be requested.
        depth: How many sites the loader may read ahead of the one in use.
        group: The devices that share the cores.
    """

    def __init__(
        self,
        layers: Sequence[Sequence[Site]],
        order: Sequence[int],
        *,
        depth: int,
        group: DeviceGroup,
    ) -> None:
        self._layers = layers
        self._kinds = [known_kind(layer) for layer in layers]
        self._last = len(layers[0]) - 1
        self._xp = xp = group.xp
        self._group = group
        self._order = list(order)
        self._staged: list[Future[_Staged]] = [Future() for _ in self._order]
        self._taken = [0] * len(self._order)
        self._events: list[list[Any]] = [[] for _ in self._order]
        self._next = [0] * self._group.size
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self.stall_seconds = [0.0] * self._group.size
        # One slot per site in flight: the one in use and ``depth`` ahead.
        self._slots: queue.Queue[tuple[Any, list[Any]]] = queue.Queue()
        n_bytes = max(
            sum(
                _aligned(prod(layer[j].shape) * np.dtype(layer[j].dtype).itemsize)
                for layer in layers
            )
            for j in range(self._last + 1)
        )
        # A single host device uses the cores as read; simulated devices stage them
        # as a GPU group does, so that the host backend tests the same path.
        staging = not is_host(xp) or self._group.size > 1
        for _ in range(depth + 1):
            buffer = pinned_empty(n_bytes, xp) if staging else None
            self._slots.put((buffer, [NullEvent()]))
        self._group.on_abort(self._closing.set)
        self._loader = ThreadPoolExecutor(1, thread_name_prefix="src-site-loader")
        self._loader.submit(self._load)

    def __len__(self) -> int:
        """Return the number of sites."""
        return self._last + 1

    def __enter__(self) -> Self:
        """Return the source."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the loader thread."""
        self._closing.set()
        self._loader.shutdown(wait=True, cancel_futures=True)

    # --- loading ---

    def _load(self) -> None:
        for k, j in enumerate(self._order):
            try:
                buffer = self._acquire()
                self._staged[k].set_result(self._stage(j, buffer))
            except BaseException as err:  # noqa: BLE001  (handed to the ranks)
                failure = GroupAbortedError() if self._closing.is_set() else err
                for future in self._staged[k:]:
                    future.set_exception(failure)
                return

    def _acquire(self) -> Any:  # noqa: ANN401  (a page-locked buffer, or None)
        while True:
            try:
                buffer, events = self._slots.get(timeout=_POLL_SECONDS)
                break
            except queue.Empty:
                if self._closing.is_set():
                    raise GroupAbortedError from None
        for event in events:
            event.synchronize()
        return buffer

    def _stage(self, j: int, buffer: Any) -> _Staged:  # noqa: ANN401
        """Read site ``j``: padded host cores, or views into the staging buffer."""
        if buffer is None:
            padded = [
                pad_site(_read(layer[j]), kind, j, self._last)
                for layer, kind in zip(self._layers, self._kinds)
            ]
            return _Staged(padded, [None] * len(padded), [], 0, None)
        # A core already on the sweep's device is used as it is by a single device;
        # a group, or a core on another device, stages it like any other.
        keep_device = self._group.size == 1 and not is_host(self._xp)
        views, device, offsets, offset = [], [], [], 0
        for layer in self._layers:
            core = layer[j]
            if (
                keep_device
                and isinstance(core, self._xp.ndarray)
                and core.device.id == self._group.devices[0]
            ):
                views.append(None)
                device.append(core)
                offsets.append(offset)
                continue
            host = np.asarray(to_numpy(core))
            view = buffer[offset : offset + host.nbytes]
            view = view.view(host.dtype).reshape(host.shape)
            np.copyto(view, host)
            views.append(view)
            device.append(None)
            offsets.append(offset)
            offset += _aligned(host.nbytes)
        return _Staged(views, device, offsets, offset, buffer)

    def _release(self, k: int, event: Any) -> None:  # noqa: ANN401
        """Record that a rank is done with its copies of site ``k``."""
        with self._lock:
            self._taken[k] += 1
            self._events[k].append(event)
            if self._taken[k] < self._group.size:
                return
            events, self._events[k] = self._events[k], []
            buffer = self._staged[k].result().buffer
        self._slots.put((buffer, events))

    # --- reading ---

    def next(self, rank: int = 0) -> tuple[Array, ...]:
        """Return the padded cores of the next site in order, on the rank's device.

        Args:
            rank: The calling rank.

        Returns:
            One rank-4 ``(l, r, u, d)`` core per layer.
        """
        k = self._next[rank]
        self._next[rank] += 1
        j = self._order[k]
        future = self._staged[k]
        start = perf_counter()
        while True:
            try:
                staged = future.result(timeout=_POLL_SECONDS)
                break
            except TimeoutError:
                if self._closing.is_set():
                    raise GroupAbortedError from None
        self.stall_seconds[rank] += perf_counter() - start
        if staged.buffer is None:
            self._release(k, NullEvent())
            return tuple(staged.cores)
        cores = self._upload(rank, k, staged)
        return tuple(
            pad_site(core, kind, j, self._last)
            for core, kind in zip(cores, self._kinds)
        )

    def _upload(self, rank: int, k: int, staged: _Staged) -> list[Array]:
        xp, stream = self._xp, current_stream(self._xp)
        if not self._group.split_uploads:
            cores = [
                device if view is None else to_device_async(view, xp, stream)
                for view, device in zip(staged.cores, staged.device)
            ]
            # The buffer is refilled only once these copies are done.
            self._release(k, stream.record())
            return cores
        lo, hi = blocks(staged.n_bytes, self._group.size)[rank]
        local = to_device_async(staged.buffer[lo:hi], xp, stream)
        self._release(k, stream.record())
        flat = self._group.allgather(rank, local)
        del local
        return [
            flat[offset : offset + view.nbytes].view(view.dtype).reshape(view.shape)
            for view, offset in zip(staged.cores, staged.offsets)
        ]


def _read(core: Site) -> np.ndarray:
    """Bring a core into memory; memmaps are read now, not on first touch."""
    if isinstance(core, np.memmap) or not isinstance(core, np.ndarray):
        return np.array(core)
    return core

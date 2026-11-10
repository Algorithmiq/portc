"""The devices of one sweep, one thread per device, and the collectives between them.

A sweep over several devices of a node runs in one process: `DeviceGroup.run`
starts one thread per device, each running the same per-device function on its
*rank* (its position in the group). Collectives are called by every rank in the
same order. They exchange device arrays through peer copies, ordered with the
kernels by events. A rank that fails aborts the group: every other rank waiting in
a collective, or for an input core, raises `GroupAbortedError`, and `run` re-raises the
original error.

On the host backend the devices are simulated: the threads share the host, and the
collectives are plain copies. The result is that of a single device; this mode
exists to test the distributed logic without GPUs.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from ._plan import device_ids
from .utils import (
    copy_into,
    current_device,
    current_stream,
    device_count,
    enable_peer_access,
    get_xp,
    use_device,
    use_stream,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .utils import Array

T = TypeVar("T")


class GroupAbortedError(RuntimeError):
    """Raised on the ranks of a group after another rank failed."""


def blocks(n: int, size: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``size`` contiguous ``(lo, hi)`` blocks, in rank order.

    Block sizes differ by at most one, the larger blocks first. Blocks may be empty
    when ``n < size``.
    """
    base, extra = divmod(n, size)
    out, lo = [], 0
    for rank in range(size):
        hi = lo + base + (rank < extra)
        out.append((lo, hi))
        lo = hi
    return out


class _Part(NamedTuple):
    """What a rank publishes in a collective: an array and the event it is ready at."""

    array: Any
    ready: Any


class DeviceGroup:
    """The devices of one sweep and the collectives between them.

    Args:
        device: ``"cpu"`` or ``"gpu"``, as in `portc.stack.src`.
        devices: A count ``n`` for the first ``n`` visible devices, or the device
            ids in rank order; rank 0 is the root. Defaults to the current device.

    Raises:
        ValueError: If a device does not exist.
    """

    def __init__(self, device: str, devices: int | Sequence[int] | None = None) -> None:
        self.xp = get_xp(device)
        ids = device_ids(devices) or (current_device(self.xp),)
        count = device_count(self.xp)
        if count is not None and max(ids) >= count:
            msg = f"Requested devices {list(ids)}, but only {count} are visible."
            raise ValueError(msg)
        self.devices = ids
        self.size = len(self.devices)
        # Uploading 1/size of each core and gathering the rest pays off only when
        # the devices copy directly between them.
        self.split_uploads = self.size > 1 and enable_peer_access(self.xp, ids)
        self._barrier = threading.Barrier(self.size)
        # Two sets of slots, alternating between consecutive collectives: a rank may
        # publish into the next collective while slower ranks still read this one.
        self._parts: list[list[Any]] = [[None] * self.size, [None] * self.size]
        self._done: list[list[Any]] = [[None] * self.size, [None] * self.size]
        self._calls = [0] * self.size
        self._on_abort: list[Callable[[], None]] = []
        self.aborted = threading.Event()

    def on_abort(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` when a rank fails, to release threads outside the ranks."""
        self._on_abort.append(callback)

    def abort(self) -> None:
        """Wake every rank waiting in a collective or on a registered resource."""
        if self.aborted.is_set():
            return
        self.aborted.set()
        self._barrier.abort()
        for callback in self._on_abort:
            callback()

    def run(self, fn: Callable[[int], T]) -> list[T]:
        """Run ``fn(rank)`` on every device, each in its own thread.

        A group of one runs ``fn(0)`` in the calling thread.

        Args:
            fn: The per-device function.

        Returns:
            The results, in rank order.

        Raises:
            Exception: The first error raised by a rank, after every rank stopped.
        """
        if self.size == 1:
            with use_device(self.xp, self.devices[0]), use_stream(self.xp):
                return [fn(0)]
        first: list[BaseException] = []
        lock = threading.Lock()

        def worker(rank: int) -> T:
            try:
                with use_device(self.xp, self.devices[rank]), use_stream(self.xp):
                    return fn(rank)
            except BaseException as err:
                secondary = isinstance(
                    err, GroupAbortedError | threading.BrokenBarrierError
                )
                with lock:
                    if not first and not secondary:
                        first.append(err)
                self.abort()
                raise

        with ThreadPoolExecutor(self.size, thread_name_prefix="src-device") as pool:
            futures = [pool.submit(worker, rank) for rank in range(self.size)]
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as err:  # noqa: BLE001  (re-raised below)
                    errors.append(err)
        if first:
            raise first[0]
        if errors:
            raise errors[0]
        return results

    # --- collectives ---

    def _wait(self) -> None:
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            raise GroupAbortedError from None

    def _publish(self, rank: int, array: Any) -> tuple[int, list[_Part]]:  # noqa: ANN401
        """Publish ``array`` and return the parts of every rank."""
        parity = self._calls[rank] % 2
        self._calls[rank] += 1
        self._parts[parity][rank] = _Part(array, current_stream(self.xp).record())
        self._wait()
        return parity, list(self._parts[parity])

    def _settle(self, rank: int, parity: int) -> None:
        """Wait, on this rank's stream, until every rank has read what it published.

        Until then the arrays published by this rank must not be overwritten; the
        memory pool reuses freed blocks in stream order, so making the stream wait
        is enough.
        """
        stream = current_stream(self.xp)
        self._done[parity][rank] = stream.record()
        self._wait()
        for event in self._done[parity]:
            stream.wait_event(event)
        self._parts[parity][rank] = None

    def _pull(self, part: _Part, dst: Array) -> None:
        stream = current_stream(self.xp)
        stream.wait_event(part.ready)
        copy_into(dst, part.array, stream)

    def allgather(self, rank: int, local: Array) -> Array:
        """Concatenate the blocks of every rank along the first axis, on every rank.

        Args:
            rank: The calling rank.
            local: This rank's block, C-contiguous on its device; the blocks may
                differ along the first axis only.

        Returns:
            The concatenation in rank order, on this rank's device.
        """
        if self.size == 1:
            return local
        parity, parts = self._publish(rank, local)
        full = self._empty_like_parts(parts)
        lo = 0
        for part in parts:
            hi = lo + part.array.shape[0]
            self._pull(part, full[lo:hi])
            lo = hi
        self._settle(rank, parity)
        return full

    def gather(self, rank: int, local: Array, root: int = 0) -> Array | None:
        """Concatenate the blocks of every rank along the first axis, on ``root``.

        Args:
            rank: The calling rank.
            local: This rank's block, C-contiguous on its device.
            root: The rank that receives the result.

        Returns:
            The concatenation on ``root``, ``None`` on the other ranks.
        """
        if self.size == 1:
            return local
        parity, parts = self._publish(rank, local)
        full = None
        if rank == root:
            full = self._empty_like_parts(parts)
            lo = 0
            for part in parts:
                hi = lo + part.array.shape[0]
                self._pull(part, full[lo:hi])
                lo = hi
        self._settle(rank, parity)
        return full

    def scatter(self, rank: int, full: Array | None, root: int = 0) -> Array:
        """Give every rank its block (see `blocks`) of the first axis of ``full``.

        Args:
            rank: The calling rank.
            full: The array to split, C-contiguous on ``root``; ignored elsewhere.
            root: The rank that holds ``full``.

        Returns:
            This rank's block, on its device.
        """
        if self.size == 1:
            assert full is not None  # noqa: S101  (the root holds it)
            return full
        parity, parts = self._publish(rank, full if rank == root else None)
        source = parts[root].array
        lo, hi = blocks(source.shape[0], self.size)[rank]
        local = self.xp.empty((hi - lo, *source.shape[1:]), dtype=source.dtype)
        self._pull(_Part(source[lo:hi], parts[root].ready), local)
        self._settle(rank, parity)
        return local

    def alltoall(self, rank: int, sends: Sequence[Array]) -> Array:
        """Send ``sends[s]`` to rank ``s``; concatenate what every rank sent here.

        Args:
            rank: The calling rank.
            sends: One C-contiguous block per rank, on this rank's device; the
                blocks sent to one rank may differ along the first axis only.

        Returns:
            The blocks sent to this rank, concatenated along the first axis in rank
            order, on this rank's device.
        """
        if self.size == 1:
            return sends[0]
        parity, parts = self._publish(rank, list(sends))
        mine = [_Part(part.array[rank], part.ready) for part in parts]
        full = self._empty_like_parts(mine)
        lo = 0
        for part in mine:
            hi = lo + part.array.shape[0]
            self._pull(part, full[lo:hi])
            lo = hi
        self._settle(rank, parity)
        return full

    def _empty_like_parts(self, parts: Sequence[_Part]) -> Array:
        first = parts[0].array
        rows = sum(part.array.shape[0] for part in parts)
        return self.xp.empty((rows, *first.shape[1:]), dtype=first.dtype)

"""The seam between the algorithms and the array backends: NumPy and CuPy.

The algorithms speak the array API standard through `array_api_compat`, and take
the namespace from their arrays (``array_namespace(x)``). Arrays stay plain ``numpy.ndarray`` or ``cupy.ndarray``,
so `opt_einsum` dispatches on them as before. What the standard leaves out lives
here: picking the namespace for a ``device``, streams and events, page-locked
memory, CuPy's memory pool, and the host-side random draws. Functions that create
arrays from nothing take the namespace ``xp``; on the host the stream helpers are
synchronous no-ops, so the same algorithm code runs on both backends.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from array_api_compat import numpy as np_compat

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from types import ModuleType

    from numpy.typing import DTypeLike

    from portc._tensor_train import Site, SiteLike

# An array of the active backend: a numpy.ndarray or a cupy.ndarray. CuPy ships no
# type stubs, so to a type checker either is no narrower than Any.
Array: TypeAlias = Any


def get_xp(device: str) -> ModuleType:
    """Return the array API namespace for the requested device.

    Args:
        device: ``"cpu"`` for numpy or ``"gpu"`` for cupy.

    Returns:
        The `array_api_compat` namespace of numpy or cupy.

    Raises:
        ValueError: If ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    if device == "cpu":
        return np_compat
    if device == "gpu":
        import array_api_compat.cupy as xp  # noqa: PLC0415  (optional dependency)

        return xp
    msg = f"Unknown device {device!r}; expected 'cpu' or 'gpu'."
    raise ValueError(msg)


def default_rng(
    seed: int | None,
) -> np.random.Generator:
    """Return a seeded NumPy ``Generator``.

    Always uses NumPy so that the same seed produces identical draws
    regardless of the device, and avoids CuPy ``Generator`` API
    differences (e.g. missing ``.normal()``).
    """
    return np.random.default_rng(seed)


def to_numpy(arr: Array | SiteLike) -> np.ndarray:
    """Bring an array, or a lazily read site, onto the host as a numpy array.

    A no-op for numpy arrays.
    """
    if isinstance(arr, np.ndarray):
        return arr
    # cupy.ndarray exposes .get(); fall back to np.asarray for other dispatchers.
    get = getattr(arr, "get", None)
    return get() if callable(get) else np.asarray(arr)


class NullEvent:
    """Stand-in for a CUDA event on the host backend: already complete."""

    def synchronize(self) -> None:
        """Return at once: host work is synchronous."""


class NullStream:
    """Stand-in for a CUDA stream on the host backend: work runs synchronously."""

    def record(self) -> NullEvent:
        """Return an event that is already complete."""
        return NullEvent()

    def wait_event(self, event: NullEvent) -> None:
        """Return at once: there is nothing to wait for."""

    def synchronize(self) -> None:
        """Return at once: host work is synchronous."""


def is_host(xp: ModuleType) -> bool:
    """Whether ``xp`` is the host backend (numpy)."""
    # An identity check: the stream helpers call this several times per site.
    return xp is np_compat or xp is np


def new_stream(xp: ModuleType) -> Any:  # noqa: ANN401  (a cupy or null stream)
    """Return a non-blocking stream for transfers, or a `NullStream` on the host."""
    if is_host(xp):
        return NullStream()
    return xp.cuda.Stream(non_blocking=True)


def current_stream(xp: ModuleType) -> Any:  # noqa: ANN401  (a cupy or null stream)
    """Return the stream kernels run on, or a `NullStream` on the host."""
    if is_host(xp):
        return NullStream()
    return xp.cuda.get_current_stream()


def pinned_empty(n_bytes: int, xp: ModuleType) -> np.ndarray:
    """Allocate a flat byte buffer, page-locked on the GPU backend.

    Args:
        n_bytes: The size of the buffer.
        xp: The array namespace.

    Returns:
        A ``uint8`` host array of ``n_bytes`` elements.
    """
    if is_host(xp):
        return np.empty(n_bytes, dtype=np.uint8)
    import cupyx  # noqa: PLC0415  (lazy: optional dependency)

    return cupyx.empty_pinned(n_bytes, dtype=np.uint8)


def to_device_async(host: np.ndarray, xp: ModuleType, stream: Any) -> Array:  # noqa: ANN401
    """Copy a host array to a new device array on ``stream``.

    On the host backend this is a plain copy, so the result never aliases a staging
    buffer that is about to be reused.

    Args:
        host: The source; page-locked for the copy to be asynchronous.
        xp: The array namespace of the device.
        stream: The stream that performs the copy.

    Returns:
        The new device array.
    """
    if is_host(xp):
        return host.copy()
    device = xp.empty(host.shape, dtype=host.dtype)
    device.set(host, stream=stream)
    return device


def to_host_async(device: Array, out: np.ndarray, stream: Any) -> None:  # noqa: ANN401
    """Copy a device array into a host array on ``stream``.

    The copy is complete once an event recorded on ``stream`` afterwards is.

    Args:
        device: The source.
        out: The destination, of the same shape and dtype; page-locked for the
            copy to be asynchronous.
        stream: The stream that performs the copy.
    """
    if isinstance(device, np.ndarray):
        np.copyto(out, device)
        return
    device.get(stream=stream, out=out, blocking=False)


def current_device(xp: ModuleType) -> int:
    """Return the id of the current device, or 0 on the host."""
    if is_host(xp):
        return 0
    return xp.cuda.runtime.getDevice()


def device_count(xp: ModuleType) -> int | None:
    """Return the number of visible devices, or ``None`` on the host.

    The host has no limit: its devices are simulated (see `use_device`).
    """
    if is_host(xp):
        return None
    return xp.cuda.runtime.getDeviceCount()


@contextmanager
def use_device(xp: ModuleType, device: int) -> Generator[None]:
    """Make ``device`` current in the calling thread for the duration of the block.

    On the host backend the devices of a group are simulated: they share the host,
    so this is a no-op.

    Args:
        xp: The array namespace.
        device: The device id.

    Yields:
        Nothing.
    """
    if is_host(xp):
        yield
        return
    with xp.cuda.Device(device):
        yield


# One per device, reused across calls: CuPy's pool hands a freed block only to the
# stream that allocated it, so a new stream per call would strand the blocks the
# previous one cached.
_streams: dict[int, Any] = {}


@contextmanager
def use_stream(xp: ModuleType) -> Generator[None]:
    """Run the block on a non-blocking stream of the current device.

    On the legacy null stream, a peer copy into one device serializes with the copy
    in the other direction, which halves the bandwidth of every collective over
    PCIe. The stream first waits for the work already queued on the previous
    current stream, so inputs written there are ready, and is synchronized on a
    normal exit, so whatever the block returns is ready on any stream. On the host
    this is a no-op.

    Args:
        xp: The array namespace.

    Yields:
        Nothing.
    """
    if is_host(xp):
        yield
        return
    device = xp.cuda.Device().id
    stream = _streams.get(device) or _streams.setdefault(
        device, xp.cuda.Stream(non_blocking=True)
    )
    stream.wait_event(xp.cuda.get_current_stream().record())
    with stream:
        yield
    stream.synchronize()


# cudaErrorPeerAccessAlreadyEnabled: another library, or an earlier call, did it.
_PEER_ALREADY_ENABLED = 704


def enable_peer_access(xp: ModuleType, devices: Sequence[int]) -> bool:
    """Enable direct access between every pair of ``devices``, where supported.

    Peer copies work without it, but are then staged through the host.

    Args:
        xp: The array namespace.
        devices: The device ids.

    Returns:
        Whether every pair has direct access; always true on the host.
    """
    if is_host(xp):
        return True
    runtime = xp.cuda.runtime
    complete = True
    for device in devices:
        with xp.cuda.Device(device):
            for peer in devices:
                if peer == device:
                    continue
                if not runtime.deviceCanAccessPeer(device, peer):
                    complete = False
                    continue
                try:
                    runtime.deviceEnablePeerAccess(peer)
                except runtime.CUDARuntimeError as err:
                    if getattr(err, "status", None) != _PEER_ALREADY_ENABLED:
                        raise
    return complete


def copy_into(dst: Array, src: Array, stream: Any) -> None:  # noqa: ANN401
    """Copy ``src`` into ``dst``, possibly on another device, on ``stream``.

    Both arrays must be C-contiguous with the same shape and dtype. Across devices
    the copy is a peer copy: over NVLink or PCIe peer-to-peer when enabled (see
    `enable_peer_access`), staged through the host otherwise. ``stream`` must belong
    to the device of ``dst``, and ``src`` must be ready on it (see
    ``stream.wait_event``).

    Args:
        dst: The destination.
        src: The source.
        stream: The stream that performs the copy.
    """
    if dst.shape != src.shape or dst.dtype != src.dtype:
        msg = f"Cannot copy {src.shape} {src.dtype} into {dst.shape} {dst.dtype}."
        raise ValueError(msg)
    if dst.nbytes == 0:
        return
    if isinstance(dst, np.ndarray):
        np.copyto(dst, src)
        return
    if not (dst.flags.c_contiguous and src.flags.c_contiguous):
        msg = "Device copies need C-contiguous arrays."
        raise ValueError(msg)
    import cupy  # noqa: PLC0415  (optional dependency)

    # CuPy arrays: ``data`` is a memory pointer and ``device`` a CUDA device.
    runtime = cupy.cuda.runtime
    if dst.device.id == src.device.id:
        runtime.memcpyAsync(
            dst.data.ptr,
            src.data.ptr,
            dst.nbytes,
            runtime.memcpyDeviceToDevice,
            stream.ptr,
        )
    else:
        runtime.memcpyPeerAsync(
            dst.data.ptr,
            dst.device.id,
            src.data.ptr,
            src.device.id,
            dst.nbytes,
            stream.ptr,
        )


def solve_triangular(a: Array, b: Array, *, lower: bool) -> Array:
    """Solve ``a @ x = b`` for a triangular ``a``, which the array API standard lacks.

    On the device this is cuBLAS's ``trsm``. The host only runs it with simulated
    devices, so a general solve stands in there.

    Args:
        a: The ``(n, n)`` triangular matrix.
        b: The right-hand sides, ``(n, k)``.
        lower: Whether ``a`` is lower triangular.

    Returns:
        ``x``, ``(n, k)``, on the backend of ``a``.
    """
    if isinstance(a, np.ndarray):
        return np.linalg.solve(a, b)
    from cupyx.scipy.linalg import solve_triangular  # noqa: PLC0415  (optional)

    return solve_triangular(a, b, lower=lower)


def device_memory(xp: ModuleType) -> tuple[int, int]:
    """Return the device bytes available to the sweep and the device total.

    Available bytes are the free device memory plus the bytes cached, but unused,
    by cupy's default memory pool.

    Args:
        xp: The array namespace of cupy.

    Returns:
        ``(available, total)`` in bytes.
    """
    free, total = xp.cuda.runtime.memGetInfo()
    return free + xp.get_default_memory_pool().free_bytes(), total


def device_pool_bytes(xp: ModuleType) -> int:
    """Return the bytes held by cupy's default memory pool, or 0 on the host.

    The pool keeps freed blocks for reuse, so after a sweep this is its high-water
    mark.
    """
    if is_host(xp):
        return 0
    return xp.get_default_memory_pool().total_bytes()


def release_cached_blocks(xp: ModuleType) -> None:
    """Return the free blocks cupy's pool caches on the current device to CUDA.

    Blocks split for smaller allocations stay cached until all their parts are
    free. No-op on the host.
    """
    if not is_host(xp):
        xp.get_default_memory_pool().free_all_blocks()


@contextmanager
def device_pool_limit(xp: ModuleType, budget: int | None) -> Generator[None]:
    """Cap cupy's default memory pool at ``budget`` bytes beyond its current use.

    An allocation past the cap fails at once instead of when another allocation
    runs out. The previous limit is restored on exit. No-op on the host backend or
    without a budget.

    Args:
        xp: The array namespace.
        budget: The bytes the pool may allocate on top of those in use, or
            ``None`` for no cap.

    Yields:
        Nothing.
    """
    if is_host(xp) or budget is None:
        yield
        return
    pool = xp.get_default_memory_pool()
    previous = pool.get_limit()
    pool.set_limit(size=pool.used_bytes() + budget)
    try:
        yield
    finally:
        pool.set_limit(size=previous)


def host_memory_available(meminfo: str = "/proc/meminfo") -> int:
    """Return the host memory available for new allocations, in bytes.

    Reads ``MemAvailable`` on Linux and falls back to the free physical pages.

    Args:
        meminfo: The path of the ``meminfo`` file.

    Returns:
        The available bytes.
    """
    path = Path(meminfo)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def sketch_dtype(dtype: DTypeLike | None, *inputs: Sequence[Site]) -> np.dtype:
    """Resolve the sketch dtype.

    Args:
        dtype: Explicit sketch dtype, or None to follow the inputs.
        *inputs: MPS or MPO tensors.

    Returns:
        The explicit dtype, or the promoted floating dtype of the inputs, so that
        the sketch never changes the precision or the field (real or complex) of
        the result. Integer and boolean inputs give ``float64``.
    """
    if dtype is not None:
        return np.dtype(dtype)
    input_dtype = np.result_type(*(arr.dtype for inpt in inputs for arr in inpt))
    if input_dtype.kind in "fc":
        return input_dtype
    return np.dtype(np.float64)


def gaussian_sketch(
    prng: np.random.Generator, shape: tuple[int, ...], dtype: DTypeLike
) -> np.ndarray:
    """Draw a Gaussian test tensor of the given dtype on the host.

    Real dtypes get i.i.d. standard normal entries. Complex dtypes get the complex
    Ginibre ensemble, ``(x + iy) / sqrt(2)`` with ``x, y`` i.i.d. standard normal,
    whose law is invariant under unitary transformations; a real sketch of a complex
    operator is not, and loses the guarantees of randomized range finding.

    The draw is made in double precision and then cast, so a seed gives the same
    sketch on every device and at every precision.

    Args:
        prng: Host-side random number generator.
        shape: Shape of the tensor.
        dtype: Dtype of the tensor.

    Returns:
        The sketch.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        re, im = prng.normal(size=(2, *shape)) / np.sqrt(2.0)
        return (re + 1j * im).astype(dtype)
    return prng.normal(size=shape).astype(dtype)

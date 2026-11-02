"""Test the host-side behaviour of the backend helpers."""

import numpy as np
import pytest

from src_method.utils import (
    NullEvent,
    NullStream,
    copy_into,
    current_device,
    current_stream,
    device_count,
    device_pool_bytes,
    device_pool_limit,
    enable_peer_access,
    get_xp,
    host_memory_available,
    is_host,
    new_stream,
    pinned_empty,
    to_device_async,
    to_host_async,
    use_device,
    use_stream,
)


def test_host_streams_are_null():
    stream = new_stream(np)

    assert is_host(np)
    assert is_host(get_xp("cpu"))
    assert isinstance(stream, NullStream)
    assert isinstance(current_stream(np), NullStream)
    event = stream.record()
    assert isinstance(event, NullEvent)
    stream.wait_event(event)
    event.synchronize()
    stream.synchronize()


def test_pinned_empty_on_host_is_a_byte_buffer():
    buffer = pinned_empty(24, np)

    assert buffer.dtype == np.uint8
    assert buffer.shape == (24,)


def test_to_device_async_on_host_copies():
    host = np.arange(6.0).reshape(2, 3)

    device = to_device_async(host, np, NullStream())
    host[:] = 0

    np.testing.assert_array_equal(device, np.arange(6.0).reshape(2, 3))


def test_to_host_async_on_host_fills_out():
    out = np.empty((2, 3))

    to_host_async(np.arange(6.0).reshape(2, 3).T.T, out, NullStream())

    np.testing.assert_array_equal(out, np.arange(6.0).reshape(2, 3))


def test_device_pool_limit_is_a_no_op_on_host():
    with device_pool_limit(np, 10):
        pass


def test_device_pool_is_empty_on_host():
    assert device_pool_bytes(np) == 0


def test_host_memory_reads_mem_available(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable:    2048 kB\n")

    assert host_memory_available(str(meminfo)) == 2048 * 1024


def test_host_memory_falls_back_without_meminfo(tmp_path):
    assert host_memory_available(str(tmp_path / "missing")) > 0


def test_host_devices_are_simulated():
    assert device_count(np) is None
    assert current_device(np) == 0
    assert enable_peer_access(np, [0, 1, 2])
    with use_device(np, 5):
        assert current_device(np) == 0


def test_host_use_stream_is_a_no_op():
    with use_stream(np):
        assert isinstance(current_stream(np), NullStream)


def test_copy_into_on_host():
    dst = np.empty((2, 3), dtype=complex)

    copy_into(dst, np.arange(6.0).reshape(2, 3).astype(complex), NullStream())

    np.testing.assert_array_equal(dst, np.arange(6.0).reshape(2, 3))


def test_copy_into_rejects_mismatches():
    with pytest.raises(ValueError, match="Cannot copy"):
        copy_into(np.empty(3), np.empty(4), NullStream())
    with pytest.raises(ValueError, match="Cannot copy"):
        copy_into(np.empty(3), np.empty(3, dtype=np.float32), NullStream())

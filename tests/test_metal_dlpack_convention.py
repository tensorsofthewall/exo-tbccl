"""MLX's Metal DLPack export: the bridge must hand libtbccl the CPU-visible address of an array's first byte whichever convention this MLX uses.

Older MLX builds export the MTLBuffer's contents address in ``data``; newer ones export the ``id<MTLBuffer>`` handle (DLPack's own convention for the Metal device)
with the array's offset in ``byte_offset``. Reading a handle as if it were the contents corrupted memory (segmentation faults in the pipeline). These checks
read the bytes at the borrowed address and compare them with the array, for offsets and every element width the pipeline uses.
"""
import ctypes
import sys

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Metal arrays exist only on macOS")

from exo_tbccl import bridge  # noqa: E402


def _bytes_at(b):
    return ctypes.string_at(b.ptr, b.nbytes)


def _expected(arr):
    import numpy as np

    uint = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[arr.dtype.size]
    return np.array(arr.view(uint)).tobytes()


@pytest.mark.parametrize("dtype", [mx.uint8, mx.float16, mx.bfloat16, mx.float32, mx.int32])
def test_borrowed_address_holds_the_arrays_bytes(dtype):
    a = (mx.arange(257 * 3).reshape(257, 3) % 251).astype(dtype)
    mx.eval(a)
    b = bridge.borrow(a)
    try:
        assert b.kind == bridge.MEMORY_METAL_SHARED
        assert _bytes_at(b) == _expected(a)
    finally:
        b.release()


@pytest.mark.parametrize("start,stop", [(0, 5), (3, 11), (100, 257)])
def test_slices_with_a_byte_offset(start, stop):
    base = (mx.arange(257 * 8).reshape(257, 8) % 241).astype(mx.float32)
    mx.eval(base)
    a = base[start:stop]
    mx.eval(a)
    b = bridge.borrow(a)
    try:
        assert _bytes_at(b) == _expected(a)
    finally:
        b.release()


def test_writing_through_the_borrowed_address_changes_the_array():
    a = mx.zeros((64,), dtype=mx.uint8)
    mx.eval(a)
    b = bridge.borrow(a, writable=True)
    try:
        ctypes.memset(b.ptr, 0x5A, b.nbytes)
    finally:
        b.release()
    assert bytes(bytearray(__import__("numpy").array(a).tobytes())) == b"\x5a" * 64


def test_the_convention_is_detected_once_and_is_consistent():
    first = bridge._metal_uses_buffer_handles()
    assert bridge._metal_uses_buffer_handles() is first

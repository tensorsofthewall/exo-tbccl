"""borrow(view_on_cpu=True) (EXO_TBCCL_ALLOC_STREAM=cpu on Metal) evaluates the metadata-only view on the CPU stream: same address, same bytes, same ownership as the default path."""
import pytest

mx = pytest.importorskip("mlx.core")
metal_only = pytest.mark.skipif(not mx.metal.is_available(), reason="the CPU-stream view is a Metal-only policy")

from exo_tbccl import bridge
from exo_tbccl.config import FastPathConfig
from exo_tbccl.group import TbcclPipelineComm, _metal_available


def test_the_default_path_does_not_use_the_cpu_stream_view():
    import inspect

    assert inspect.signature(bridge.borrow).parameters["view_on_cpu"].default is False
    assert FastPathConfig().alloc_cpu is False
    assert TbcclPipelineComm(object(), 0, 1, FastPathConfig())._view_cpu is False


def test_the_option_only_takes_effect_on_metal():
    comm = TbcclPipelineComm(object(), 0, 1, FastPathConfig(alloc_cpu=True))
    assert comm._view_cpu is _metal_available()  # CUDA and CPU-only hosts keep the default path


@metal_only
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32, mx.int32])
def test_cpu_stream_view_borrows_the_same_storage(dtype):
    a = (mx.arange(64).astype(dtype) + mx.zeros((64,), dtype=dtype))
    mx.eval(a)
    default, on_cpu = bridge.borrow(a), bridge.borrow(a, view_on_cpu=True)
    try:
        assert (on_cpu.ptr, on_cpu.nbytes, on_cpu.kind, on_cpu.label) == (default.ptr, default.nbytes, default.kind, default.label)
    finally:
        default.release()
        on_cpu.release()


@metal_only
def test_cpu_stream_view_of_a_writable_destination_is_written_through():
    import ctypes

    dest = mx.zeros((16,), dtype=mx.uint8)
    mx.eval(dest)
    b = bridge.borrow(dest, writable=True, view_on_cpu=True)
    try:
        ctypes.memset(b.ptr, 0x5A, b.nbytes)
    finally:
        b.release()
    assert mx.all(dest == 0x5A).item()


@metal_only
def test_non_contiguous_input_is_materialized_and_counted():
    a = mx.arange(32).reshape(4, 8).astype(mx.float32)[:, ::2]
    stats = bridge.CopyStats()
    b = bridge.borrow(a, stats, view_on_cpu=True)
    b.release()
    assert stats.materialized_copies == 1

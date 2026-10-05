"""Phase 59: the CPU-stream same-width view (EXO_TBCCL_VIEW_STREAM=cpu) must alias the original storage: same pointer, same bytes, no payload copy, for every
dtype the bridge sees. Metal-only experiment: skipped where Metal is unavailable (CUDA never takes this path: bridge.borrow ignores the setting there)."""

import pytest

mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal-only experiment")


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int8", "uint8"])
@pytest.mark.parametrize("n", [1, 7, 1024, 4097])
def test_cpu_stream_view_aliases_the_original_storage(dtype, n):
    from exo_tbccl import _native as native

    dt = getattr(mx, dtype)
    uint = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[dt.size]
    a = (mx.arange(n) % 100).astype(dt)
    mx.eval(a)
    v_gpu = a.view(uint)
    v_cpu = a.view(uint, stream=mx.cpu)
    mx.eval(v_gpu, v_cpu)
    e_gpu, e_cpu = native.Export(v_gpu), native.Export(v_cpu)
    try:
        assert e_cpu.ptr == e_gpu.ptr, "the CPU-stream view must share the storage the GPU-stream view shares"
        assert e_cpu.nbytes == e_gpu.nbytes == a.nbytes
        assert bool(mx.array_equal(v_cpu, v_gpu))
    finally:
        e_gpu.release()
        e_cpu.release()
    if dt in (mx.float32, mx.float16):  # a dtype __dlpack__ can export directly: the view's pointer is the array's own
        e_orig = native.Export(a)
        try:
            e_v = native.Export(v_cpu)
            assert e_v.ptr == e_orig.ptr
            e_v.release()
        finally:
            e_orig.release()

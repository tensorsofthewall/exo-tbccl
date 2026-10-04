"""The MLX CUDA export is real managed memory and the driver reports what the host-direct policy needs (skips off CUDA)."""

import pytest


def _worker(rank, world, ex):
    import mlx.core as mx

    from exo_tbccl import _cuda_caps
    from exo_tbccl._loader import native

    x = (mx.arange(1024, dtype=mx.float32) * 3).view(mx.uint32)
    mx.eval(x)
    dev = x.__dlpack_device__()
    if dev[0] != native.DL_CUDA_MANAGED:
        return None
    exp = native.Export(x)
    try:
        info = _cuda_caps.pointer_info(exp.ptr)
        caps = _cuda_caps.device_caps(dev[1])
    finally:
        exp.release()
    return dev, info, caps


def test_mlx_cuda_pointer_is_managed_and_caps_queryable():
    from tests.harness import run_world

    (res,) = run_world(1, _worker)
    if res is None:
        pytest.skip("not an MLX CUDA (managed) system")
    dev, info, caps = res
    assert info is not None and info.is_managed and info.device_ordinal == dev[1]
    assert caps is not None and caps.attrs["managed_memory"] in (0, 1) and caps.attrs["concurrent_managed_access"] in (0, 1)

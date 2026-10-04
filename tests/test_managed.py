"""CUDA managed-memory host-direct mode (the latency-attribution work): forced mode correctness with GPU producers and immediate GPU consumers, on loopback."""

import pytest

from tests.harness import run_world
from tests.managed_worker import pingpong

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("mlx") is None, reason="needs MLX"
)


def _is_cuda_managed():
    import mlx.core as mx

    return mx.default_device() == mx.gpu and mx.__version__ is not None


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("nbytes", [2048, 65536])
def test_forced_host_direct_roundtrip(dtype, nbytes):
    env = {"EXO_TBCCL_CUDA_MANAGED_MODE": "host"}
    res = run_world(2, pingpong, env, dtype, nbytes, 60)
    for r in res:
        assert r["copies"] == (0, 0)
        assert set(r["labels"]) <= {"managed-host", "host", "metal-direct"}, r["labels"]

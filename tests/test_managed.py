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


def _fake_comm(monkeypatch, attrs, mode="auto", **cfg):
    from exo_tbccl import _cuda_caps, group
    from exo_tbccl.config import FastPathConfig

    monkeypatch.setattr(_cuda_caps, "device_caps", lambda ordinal: None if attrs is None else _cuda_caps.DeviceCaps(ordinal, attrs))
    c = group.TbcclPipelineComm.__new__(group.TbcclPipelineComm)
    c.config = FastPathConfig(managed_mode=mode, **cfg)
    c._managed_verified = {}
    return c


PROVEN = {"managed_memory": 1, "concurrent_managed_access": 1, "pageable_memory_access": 1, "pageable_memory_access_uses_host_page_tables": 0,
          "direct_managed_mem_access_from_host": 0, "integrated": 0}


class _Arr:
    def __init__(self, nbytes, device=(13, 0)):
        self.nbytes, self._d = nbytes, device

    def __dlpack_device__(self):
        return self._d


def test_auto_uses_host_only_for_a_proven_signature_small_payload_and_managed_storage(monkeypatch):
    assert _fake_comm(monkeypatch, PROVEN)._managed_as_host(True, False, _Arr(2048)) is True
    assert _fake_comm(monkeypatch, PROVEN)._managed_as_host(True, False, _Arr(1 << 20)) is False  # over the size gate
    assert _fake_comm(monkeypatch, PROVEN)._managed_as_host(True, False, _Arr(2048, (2, 0))) is False  # plain CUDA device memory
    assert _fake_comm(monkeypatch, PROVEN)._managed_as_host(True, False, _Arr(2048, (8, 0))) is False  # Metal


@pytest.mark.parametrize("change", [{"concurrent_managed_access": 0}, {"managed_memory": 0}, {"integrated": 1}, {"direct_managed_mem_access_from_host": 1},
                                    {"pageable_memory_access": 0}])
def test_auto_falls_back_to_cuda_on_unproven_or_unknown_capabilities(monkeypatch, change):
    assert _fake_comm(monkeypatch, PROVEN | change)._managed_as_host(True, False, _Arr(2048)) is False


def test_auto_falls_back_when_the_driver_cannot_be_queried_and_cuda_mode_never_maps(monkeypatch):
    assert _fake_comm(monkeypatch, None)._managed_as_host(True, False, _Arr(2048)) is False
    assert _fake_comm(monkeypatch, PROVEN, mode="cuda")._managed_as_host(True, False, _Arr(2048)) is False


def test_forced_host_respects_direction_flags(monkeypatch):
    c = _fake_comm(monkeypatch, None, mode="host", managed_recv=False)
    assert c._managed_as_host(True, False, _Arr(1 << 20)) is True
    assert c._managed_as_host(False, True, _Arr(1 << 20)) is False

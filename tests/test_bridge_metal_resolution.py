"""Platform-independent checks of how the bridge turns a DLPack export into an address (the convention probe itself needs a Mac: test_metal_dlpack_convention.py)."""
from types import SimpleNamespace

import pytest

from exo_tbccl import bridge
from exo_tbccl._loader import native
from exo_tbccl.errors import TbcclUnsupportedError


def _exp(ptr=0x1000, data=0x2000, offset=64):
    return SimpleNamespace(ptr=ptr, data=data, byte_offset=offset)


def test_non_metal_exports_use_the_exported_address_unchanged():
    assert bridge._export_address(_exp(), native.DL_CUDA) == 0x1000
    assert bridge._export_address(_exp(), native.DL_CPU) == 0x1000


def test_metal_contents_convention_uses_the_exported_address(monkeypatch):
    monkeypatch.setattr(bridge, "_metal_uses_buffer_handles", lambda: False)
    assert bridge._export_address(_exp(), native.DL_METAL) == 0x1000


def test_metal_buffer_handle_convention_resolves_contents_plus_offset(monkeypatch):
    monkeypatch.setattr(bridge, "_metal_uses_buffer_handles", lambda: True)
    seen = []
    monkeypatch.setattr(bridge, "_objc_contents", lambda handle: seen.append(handle) or 0x9000)
    assert bridge._export_address(_exp(data=0x2000, offset=64), native.DL_METAL) == 0x9000 + 64
    assert seen == [0x2000]


def test_a_private_buffer_is_refused_not_guessed(monkeypatch):
    monkeypatch.setattr(bridge, "_metal_uses_buffer_handles", lambda: True)
    monkeypatch.setattr(bridge, "_objc_contents", lambda handle: 0)
    with pytest.raises(TbcclUnsupportedError, match="not CPU-visible"):
        bridge._export_address(_exp(), native.DL_METAL)

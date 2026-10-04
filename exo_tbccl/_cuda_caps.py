"""CUDA managed-memory capability inspection through the CUDA driver API (ctypes; no CUDA build dependency, internal to exo_tbccl).

kDLCUDAManaged says only that storage came from cudaMallocManaged. Whether the CPU may touch it while the GPU is active, and what that costs, is
a property of the system, so the adapter asks the driver instead of inferring it from the OS or the GPU generation.
"""

from __future__ import annotations

import ctypes
import functools
from dataclasses import dataclass

_PTR_CONTEXT, _PTR_MEMORY_TYPE, _PTR_DEVICE_POINTER, _PTR_HOST_POINTER, _PTR_IS_MANAGED, _PTR_DEVICE_ORDINAL = 1, 2, 3, 4, 8, 9
_DEV_ATTRS = {
    "compute_major": 75,
    "compute_minor": 76,
    "unified_addressing": 41,
    "integrated": 18,
    "managed_memory": 83,
    "pageable_memory_access": 88,
    "concurrent_managed_access": 89,
    "pageable_memory_access_uses_host_page_tables": 100,
    "direct_managed_mem_access_from_host": 101,
}


@dataclass(frozen=True)
class DeviceCaps:
    ordinal: int
    attrs: dict[str, int]

    @property
    def cpu_may_access_managed_while_gpu_active(self) -> bool:
        return self.attrs.get("managed_memory") == 1 and self.attrs.get("concurrent_managed_access") == 1


@dataclass(frozen=True)
class PointerInfo:
    is_managed: bool
    memory_type: int
    device_ordinal: int
    host_pointer: int
    device_pointer: int


@functools.cache
def _driver() -> ctypes.CDLL | None:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        return lib if lib.cuInit(0) == 0 else None
    except OSError:
        return None


def device_caps(ordinal: int) -> DeviceCaps | None:
    """None when the driver cannot be queried (the caller must then treat managed memory as CUDA memory)."""
    return _device_caps(ordinal)


@functools.cache
def _device_caps(ordinal: int) -> DeviceCaps | None:
    cu = _driver()
    if cu is None:
        return None
    dev = ctypes.c_int()
    if cu.cuDeviceGet(ctypes.byref(dev), ordinal) != 0:
        return None
    attrs: dict[str, int] = {}
    for name, code in _DEV_ATTRS.items():
        v = ctypes.c_int(-1)
        attrs[name] = v.value if cu.cuDeviceGetAttribute(ctypes.byref(v), code, dev.value) == 0 else -1
    return DeviceCaps(ordinal, attrs)


def pointer_info(ptr: int) -> PointerInfo | None:
    cu = _driver()
    if cu is None:
        return None

    def attr(code: int, ctype: type) -> int | None:
        v = ctype(0)
        return v.value if cu.cuPointerGetAttribute(ctypes.byref(v), code, ctypes.c_uint64(ptr)) == 0 else None

    managed = attr(_PTR_IS_MANAGED, ctypes.c_int)
    ordinal = attr(_PTR_DEVICE_ORDINAL, ctypes.c_int)
    if managed is None or ordinal is None:
        return None
    return PointerInfo(
        bool(managed),
        attr(_PTR_MEMORY_TYPE, ctypes.c_int) or 0,
        ordinal,
        attr(_PTR_HOST_POINTER, ctypes.c_uint64) or 0,
        attr(_PTR_DEVICE_POINTER, ctypes.c_uint64) or 0,
    )

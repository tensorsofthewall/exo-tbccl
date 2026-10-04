"""MLX array <-> TBCCL buffer bridge (DLPack, no PyTorch, no MLX C++ internals).

Findings that shape this module (docs/mlx_dlpack_bridge.md): MLX reports the real device only through ``__dlpack_device__`` (CUDA arrays are
CUDA *managed* memory, Metal arrays are Metal), its capsule always says CPU and cannot carry bfloat16, and ``array.view(mx.uint8)`` is a
zero-copy alias on both platforms. A borrowed buffer is therefore a same-width unsigned view of an evaluated, row-contiguous array.

A ``Borrow`` keeps the MLX array, its view and the DLPack export alive until ``release()``; the caller releases it only after the TBCCL Work
is terminal.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import TbcclInvalidArgumentError, TbcclUnsupportedError
from ._loader import native

if TYPE_CHECKING:
    import mlx.core as mx

log = logging.getLogger("exo_tbccl")

MEMORY_HOST = native.TBCCL_MEMORY_HOST
MEMORY_CUDA = native.TBCCL_MEMORY_CUDA
MEMORY_METAL_SHARED = native.TBCCL_MEMORY_METAL_SHARED

# DLPack device type -> (TBCCL memory kind, label reported by trace/stats)
_KIND_BY_DEVICE: dict[int, tuple[int, str]] = {
    native.DL_CPU: (MEMORY_HOST, "host"),
    native.DL_CUDA_HOST: (MEMORY_HOST, "host"),
    native.DL_CUDA: (MEMORY_CUDA, "cuda-direct"),
    native.DL_CUDA_MANAGED: (MEMORY_CUDA, "cuda-direct"),
    native.DL_METAL: (MEMORY_METAL_SHARED, "metal-direct"),
}


@dataclass
class CopyStats:
    """Every payload-sized copy the adapter itself makes is counted here (it must stay 0 for the strong-success paths)."""

    direct_ops: dict[str, int] = field(default_factory=dict)
    direct_bytes: int = 0
    materialized_copies: int = 0
    staged_fallback_copies: int = 0

    def note_direct(self, label: str, nbytes: int) -> None:
        self.direct_ops[label] = self.direct_ops.get(label, 0) + 1
        self.direct_bytes += nbytes


TRACE = os.environ.get("EXO_TBCCL_TRACE", "") not in ("", "0")


class Borrow:
    """A TBCCL-ready view of an array's storage. Must outlive the Work that uses it."""

    __slots__ = ("owner", "view", "export", "ptr", "nbytes", "kind", "device", "label")

    def __init__(self, owner: object, view: object, export: object | None, ptr: int, nbytes: int, kind: int, device: int, label: str):
        self.owner = owner
        self.view = view
        self.export = export
        self.ptr = ptr
        self.nbytes = nbytes
        self.kind = kind
        self.device = device
        self.label = label

    def release(self) -> None:
        export = self.export
        self.export = None
        if export is not None:
            export.release()  # pyright: ignore[reportAttributeAccessIssue]
        self.view = None
        self.owner = None


def _kind_for(device_type: int, device_id: int) -> tuple[int, int, str]:
    try:
        kind, label = _KIND_BY_DEVICE[device_type]
    except KeyError:
        raise TbcclUnsupportedError(native.TBCCL_UNSUPPORTED, "bridge", f"unsupported DLPack device type {device_type}") from None
    return kind, (device_id if kind == MEMORY_CUDA else -1), label


def _is_mlx(array: object) -> bool:
    return type(array).__module__.startswith("mlx.")


def borrow(array: object, stats: CopyStats | None = None, *, writable: bool = False) -> Borrow:
    """Borrow the bytes of an MLX array (or any DLPack producer with a C-contiguous layout).

    For MLX the array is evaluated (the same ``mx.eval`` boundary exo's pipeline already requires) and made row-contiguous when it is not;
    that materialization is a payload-sized copy and is counted in ``stats.materialized_copies``.
    """
    if _is_mlx(array):
        import mlx.core as mx

        arr: mx.array = array  # pyright: ignore[reportAssignmentType]
        mx.eval(arr)
        device_type, device_id = arr.__dlpack_device__()
        kind, device, label = _kind_for(device_type, device_id)
        if arr.size == 0:
            return Borrow(arr, None, None, 0, 0, kind, device, label)
        uint = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[arr.dtype.size]
        # A same-width unsigned view is metadata-only and keeps the strides, so the native consumer can tell whether `arr` is row-contiguous
        # and hands back the array's real storage pointer (bfloat16 cannot go through __dlpack__ directly; its uint16 alias can).
        owner: object = arr
        probe = arr.view(uint)
        mx.eval(probe)
        try:
            exp = native.Export(probe)
        except native.NotContiguousError:
            if writable:
                raise TbcclInvalidArgumentError(native.TBCCL_INVALID_ARGUMENT, "bridge", "receive destination must be row-contiguous") from None
            if stats is not None:
                stats.materialized_copies += 1
            owner = mx.contiguous(arr)
            mx.eval(owner)
            probe = owner.view(uint)
            mx.eval(probe)
            exp = native.Export(probe)
        if exp.device != (device_type, device_id):
            exp.release()
            raise TbcclInvalidArgumentError(native.TBCCL_INVALID_ARGUMENT, "bridge", f"device changed while borrowing: {exp.device} vs {(device_type, device_id)}")
        if exp.nbytes != arr.nbytes:
            exp.release()
            raise TbcclInvalidArgumentError(native.TBCCL_INVALID_ARGUMENT, "bridge", f"byte count mismatch: view {exp.nbytes} vs array {arr.nbytes}")
        return Borrow((arr, owner), probe, exp, exp.ptr, exp.nbytes, kind, device, label)

    exp = native.Export(array)
    device_type, device_id = exp.device
    try:
        kind, device, label = _kind_for(device_type, device_id)
    except TbcclUnsupportedError:
        exp.release()
        raise
    return Borrow(array, None, exp, exp.ptr, exp.nbytes, kind, device, label)

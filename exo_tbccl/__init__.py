"""exo-tbccl: TBCCL as a heterogeneous pipeline data plane for exo (no PyTorch, no MLX C++ internals)."""

from . import errors
from .errors import (
    TbcclAbortedError,
    TbcclDeviceError,
    TbcclError,
    TbcclTimeoutError,
    TbcclTransportError,
    TbcclUnsupportedError,
)

__version__ = "0.3.0rc1"


def is_available() -> tuple[bool, str]:
    """(True, "") when the native binding and a compatible TBCCL (C ABI 1) can be loaded."""
    try:
        from ._loader import native

        abi = native.abi_version()
    except Exception as e:  # noqa: BLE001 - any load failure means "unavailable"
        return False, f"{type(e).__name__}: {e}"
    if abi != native.TBCCL_C_ABI_VERSION:
        return False, f"TBCCL C ABI {abi} does not match the ABI this build targets ({native.TBCCL_C_ABI_VERSION})"
    return True, ""


__all__ = [
    "TbcclAbortedError",
    "TbcclDeviceError",
    "TbcclError",
    "TbcclTimeoutError",
    "TbcclTransportError",
    "TbcclUnsupportedError",
    "errors",
    "is_available",
    "__version__",
]

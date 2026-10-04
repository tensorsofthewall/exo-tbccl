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

__version__ = "0.1.0"

__all__ = [
    "TbcclAbortedError",
    "TbcclDeviceError",
    "TbcclError",
    "TbcclTimeoutError",
    "TbcclTransportError",
    "TbcclUnsupportedError",
    "errors",
    "__version__",
]

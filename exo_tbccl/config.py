"""Fast-path policy for TbcclPipelineComm (Phase 54). Defaults reproduce Phase 53 behavior exactly; every optimization is opt-in.

Environment overrides (benchmarking/debugging, read once at ``FastPathConfig.from_env()``):

``EXO_TBCCL_CUDA_MANAGED_MODE``  cuda | host | auto      how a kDLCUDAManaged export is described to TBCCL (default cuda)
``EXO_TBCCL_MANAGED_DIRS``       send,recv | send | recv  which directions the managed mapping applies to (default send,recv)
``EXO_TBCCL_MANAGED_MAX_BYTES``   integer                   largest payload ``auto`` maps to host (default 16384; ``host`` ignores it)
``EXO_TBCCL_RECV``               fresh | reuse             receive destination policy (default fresh)
``EXO_TBCCL_ASYNC_SEND``         0 | 1                     decode sends are not waited for immediately (default 0)
``EXO_TBCCL_ALLOC_STREAM``        gpu | cpu                 stream that allocates fresh receive/all_gather destinations (Phase 57 experiment, default gpu)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MANAGED_CUDA = "cuda"
MANAGED_HOST = "host"
MANAGED_AUTO = "auto"


@dataclass(frozen=True)
class FastPathConfig:
    managed_mode: str = MANAGED_CUDA
    managed_send: bool = True
    managed_recv: bool = True
    managed_max_bytes: int = 16384
    recv_reuse: bool = False
    async_send: bool = False
    alloc_cpu: bool = False

    @classmethod
    def from_env(cls) -> "FastPathConfig":
        env = os.environ
        mode = env.get("EXO_TBCCL_CUDA_MANAGED_MODE", MANAGED_CUDA).lower()
        if mode not in (MANAGED_CUDA, MANAGED_HOST, MANAGED_AUTO):
            raise ValueError(f"EXO_TBCCL_CUDA_MANAGED_MODE must be cuda|host|auto, got {mode!r}")
        dirs = {d for d in env.get("EXO_TBCCL_MANAGED_DIRS", "send,recv").split(",") if d}
        if not dirs <= {"send", "recv"}:
            raise ValueError(f"EXO_TBCCL_MANAGED_DIRS must be a subset of send,recv, got {sorted(dirs)}")
        recv = env.get("EXO_TBCCL_RECV", "fresh").lower()
        if recv not in ("fresh", "reuse"):
            raise ValueError(f"EXO_TBCCL_RECV must be fresh|reuse, got {recv!r}")
        return cls(mode, "send" in dirs, "recv" in dirs, int(env.get("EXO_TBCCL_MANAGED_MAX_BYTES", "16384")), recv == "reuse", env.get("EXO_TBCCL_ASYNC_SEND", "0") not in ("", "0"), env.get("EXO_TBCCL_ALLOC_STREAM", "gpu").lower() == "cpu")

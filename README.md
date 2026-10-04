# exo-tbccl

TBCCL as a heterogeneous **pipeline-parallel data plane for exo**: MLX activations move between pipeline stages (Mac Metal, Linux CUDA, or
CPU) over TBCCL's stable C ABI v1. Optional, separate from exo and from TBCCL; no PyTorch, no MLX C++ internals, no copy of TBCCL.

```
exo Master / placement / topology
            |
            v
   PipelineShardMetadata
            |
            v
      exo PipelineComm
        /          \
MlxPipelineComm   TbcclPipelineComm   <- this package
     |                 |
mx.distributed    DLPack bridge
                       |
                  TBCCL C ABI v1
                       |
                    libtbccl
```

Scope: pipeline parallelism for text generation only. Not tensor parallelism, not image/CFG models. exo owns discovery,
topology, placement and shard assignment; TBCCL never discovers anything.

## Install

Needs an installed TBCCL >= 0.5 (C ABI 1). A CUDA-enabled install on Linux/NVIDIA, a host/Metal install on macOS.

```sh
TBCCL_ROOT=<tbccl prefix> uv pip install --python <exo venv>/bin/python -e .
```

The extension links only `TBCCL::tbccl_c`. Use `uv` for package management. `uv sync` in exo removes it; reinstall afterwards.
exo works without this package: selecting `MlxTbccl` without it is a clear placement error (`exo_tbccl.is_available()` explains why).

## Use (what exo does)

```python
from exo_tbccl.group import TbcclPipelineComm
comm = TbcclPipelineComm.create(rank, world_size, exchange, bind_host=host, advertise_host=host)
# exchange(purpose, payload) -> list[bytes]: the host application's all-gather of opaque bytes (exo's runner byte exchange)
x = comm.recv_like(template, src)      # MLX array in, MLX array out; Work waited
comm.send(array, dst)
comm.flush_sends([(a, dst), ...])      # submits every send first, then waits as a group
y = comm.all_gather(array)             # rank-order concatenation along axis 0
comm.barrier(); comm.any_true(flag); comm.close()   # close() is explicit; no finalizer dependence
```

Rank and world size come from exo's `PipelineShardMetadata`, never from node ordering. Every error is a typed exception built from TBCCL's
structured result code (`TbcclTransportError`, `TbcclAbortedError`, `TbcclTimeoutError`, `TbcclDeviceError`, ...) carrying rank, peer and
operation. Set `EXO_TBCCL_TRACE=1` to log the path of every operation (host, cuda-direct, metal-direct); `comm.stats` counts any
adapter copy (it must stay 0).

## Tests

```sh
<exo venv>/bin/python -m pytest -p no:asyncio tests          # 27 tests, process-per-rank, loopback
<exo venv>/bin/python examples/link_probe.py ...             # two-host correctness probe (see its docstring)
<exo venv>/bin/python benchmarks/bridge_overhead.py          # loopback bridge cost
```

See `docs/architecture.md`, `docs/mlx_dlpack_bridge.md`, `docs/bootstrap.md`
and

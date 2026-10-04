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

Scope (Phase 53, extended in Phase 54): pipeline parallelism for text generation only. Not tensor parallelism, not image/CFG models. exo owns discovery,
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

## Fast paths (Phase 54, all off by default)

`FastPathConfig` (or the environment, read when a communicator is created) selects three opt-in optimizations; the defaults are exactly the Phase 53 behavior.

| setting | values | what it does | measured |
|---|---|---|---|
| `EXO_TBCCL_CUDA_MANAGED_MODE` | `cuda` (default), `host`, `auto` | describe CUDA-managed MLX storage to TBCCL as host memory (`auto`: only a proven capability signature and <= 16 KiB) | wins on loopback at decode sizes, **loses on the real TB4 link**; leave at `cuda` |
| `EXO_TBCCL_RECV` | `fresh` (default), `reuse` | serve `recv_like` destinations from a bounded pool, released at exo's `step_complete()` | -60% bridge round trip on Metal; neutral end to end |
| `EXO_TBCCL_ASYNC_SEND` | `0` (default), `1` | decode sends submit and return; Work/Borrow are tracked and reaped, failures surface at the next communication point | neutral end to end |

`kDLCUDAManaged` stays authoritative for what storage is; see `docs/cuda_managed_memory.md`, `docs/receive_buffer_pool.md`, `docs/async_send.md` and `docs/phase54_results.md`.
Reuse is audited for Qwen3 (KVCache) and the synthetic model only; audit other cache families with `EXO_TBCCL_RECV_POISON=0xA5` before enabling it.

## Tests

```sh
<exo venv>/bin/python -m pytest -p no:asyncio tests          # 58 tests, process-per-rank, loopback
<exo venv>/bin/python examples/link_probe.py ...             # two-host correctness probe (see its docstring)
<exo venv>/bin/python benchmarks/bridge_overhead.py          # loopback bridge cost
```

Two-host probes: `examples/two_host_fastpath.py`, `examples/two_host_ring_chain.py`, `benchmarks/real_model_two_host.py` (see their docstrings; AER-gate every real-link run).

See `docs/architecture.md`, `docs/mlx_dlpack_bridge.md`, `docs/bootstrap.md`, `docs/phase53_memory_bridge.md`, `docs/phase53_exo_audit.md`
`docs/phase53_results.md` and `docs/phase54_results.md`.

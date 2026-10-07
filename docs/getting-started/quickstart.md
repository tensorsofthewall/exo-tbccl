# Quickstart

You normally do not call exo-tbccl yourself: exo's placement chooses it, and exo's runners create the communicator. The API below shows what exo does.

```python
from exo_tbccl.group import TbcclPipelineComm

comm = TbcclPipelineComm.create(rank, world_size, exchange, bind_host=host, advertise_host=host)
# exchange(purpose, payload) -> list[bytes]: the host application's all-gather of opaque bytes
#   (exo supplies its runner byte exchange).

x = comm.recv_like(template, src)      # MLX array in, MLX array out; the Work is waited
comm.send(array, dst)
comm.flush_sends([(a, dst), ...])      # submits every send first, then waits as a group
y = comm.all_gather(array)             # rank-order concatenation along axis 0
comm.barrier()
comm.any_true(flag)
comm.close()                           # explicit; no dependence on finalizers
```

Rank and world size come from exo's `PipelineShardMetadata`, never from node ordering. Errors are typed exceptions built from TBCCL's structured result codes (`TbcclTransportError`, `TbcclAbortedError`, `TbcclTimeoutError`, `TbcclDeviceError`, and so on) and carry rank, peer and operation. Set `EXO_TBCCL_TRACE=1` to log the path of every operation (host, CUDA-direct, Metal-direct); `comm.stats` counts any adapter copy, which must stay 0.

See [architecture](../concepts/architecture.md) for how exo uses it.

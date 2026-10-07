# Architecture

exo-tbccl is a pipeline-parallel data plane for exo: MLX activations move between pipeline stages (Mac Metal, Linux CUDA, or CPU) over TBCCL's stable C ABI. It is separate from exo and from TBCCL, uses no PyTorch and no MLX C++ internals, and contains no copy of TBCCL.

```
exo Master / placement / topology
        |
 PipelineShardMetadata
        |
   exo PipelineComm
     /          \
MlxPipelineComm   TbcclPipelineComm   <- this package
  mx.distributed      DLPack bridge
                          |
                  TBCCL C ABI v1 -> libtbccl
```

## Who owns what

| Project | Owns |
|---|---|
| exo | discovery, topology, placement, shard assignment, model loading, scheduling, the `PipelineComm` seam, the generic runner byte exchange |
| exo-tbccl | the MLX to TBCCL bridge, the C-ABI binding, `TbcclPipelineComm`, typed errors |
| libtbccl | bytes, collectives, transports, memory providers, work and failure semantics |

Scope: pipeline parallelism for text generation only. Not tensor parallelism, not image or classifier-free-guidance models. TBCCL never discovers anything ({doc}`../adr/0001-c-abi-only`).

## Layers inside exo-tbccl

- `src/native/exo_tbccl_native.c`: a CPython binding of the C ABI (`Bootstrap`, `Comm`, `Work`) plus a DLPack consumer. Blocking calls release the GIL; errors are built from the result code.
- `exo_tbccl/bridge.py`: `borrow(array)` evaluates an MLX array, takes a same-width unsigned view, consumes it through DLPack, maps the device to a TBCCL memory kind and returns a `Borrow` that keeps the array, the view and the export alive.
- `exo_tbccl/group.py`: `TbcclPipelineComm`, which tracks every in-flight transfer so a borrow outlives its Work.
- `exo_tbccl/errors.py`: the exception hierarchy keyed by result code.

## How exo uses it

1. **Placement.** `InstanceMeta.MlxTbccl` is available only if `exo_tbccl` imports; it accepts pipeline sharding and text models, reuses `PipelineShardMetadata`, and uses one address per node.
2. **Runner.** `init_tbccl_pipeline_comm` creates the communicator; the unique id and the endpoint blobs travel through exo's runner byte exchange ({doc}`bootstrap`).
3. **Generation.** The runner threads a `CommGroup` (an `mx.distributed.Group` or a `PipelineComm`) through the pipeline layers, the prefill queue and the runner-level agreements (barrier, any, task agreement, KV-cache pressure).
4. **Lifetime boundary.** `PipelineLastLayer` calls `comm.step_complete()` right after the stage output has been evaluated; `TbcclPipelineComm` uses it to recycle receive destinations when reuse is enabled (a no-op for `MlxPipelineComm`).
5. **Shutdown.** The engine releases the communicator before model state is destroyed.

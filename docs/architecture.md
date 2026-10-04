# Architecture

Three codebases, one-way dependencies:

| repo | owns |
|---|---|
| exo | discovery, topology, placement, shard assignment, model loading, scheduling; the `PipelineComm` seam; `MlxTbcclInstance`; the generic runner byte exchange |
| exo-tbccl | the MLX <-> TBCCL bridge, the C-ABI binding (`_native`), `TbcclPipelineComm`, typed errors |
| libtbccl | bytes, collectives, transports, memory providers, Work/failure semantics |

## Layers inside exo-tbccl

- `src/native/exo_tbccl_native.c`: CPython binding of the C ABI v1 (`Bootstrap`, `Comm`, `Work`) plus a DLPack **consumer** (`Export`).
  Blocking calls release the GIL. Errors go through a Python factory from the result code. The consumer renames the capsule to
  `used_dltensor` and runs the producer's deleter on `release()` (never with an exception pending).
- `exo_tbccl/bridge.py`: `borrow(array)` evaluates an MLX array, takes a same-width unsigned view (metadata only, keeps strides), consumes it
  through DLPack, maps `__dlpack_device__` to a TBCCL memory kind and returns a `Borrow` that keeps the array, the view and the export alive.
- `exo_tbccl/group.py`: `TbcclPipelineComm`. Tracks every in-flight transfer so a borrow outlives its Work, whatever happens to the Python
  handle. `close()` aborts only when work is in flight, drains, releases borrows, then destroys the communicator.
- `exo_tbccl/errors.py`: exception hierarchy keyed by result code.

## How exo uses it

1. Placement (`exo.master.placement`): `InstanceMeta.MlxTbccl` is available only if `exo_tbccl` imports; it accepts Pipeline sharding and text
   models, reuses the existing `PipelineShardMetadata`, and picks one address per node that every peer can reach (thunderbolt preferred,
   same rule as the ring backend). Per-peer addresses are rejected: TBCCL advertises one endpoint per rank.
2. Runner (`MlxBuilder.connect`): `init_tbccl_pipeline_comm` creates the communicator. The UniqueId and the 256-byte endpoint blobs travel
   through exo's runner byte exchange as opaque bytes (see `bootstrap.md`).
3. Generation: the runner threads a `CommGroup` (an `mx.distributed.Group` or a `PipelineComm`) through the pipeline layers, the prefill queue and
   the runner-level agreements (barrier, any, task agreement, KV-cache pressure). Tensor-parallel code still takes an `mx.distributed.Group`.
4. Lifetime boundary (Phase 54): `PipelineLastLayer` calls `comm.step_complete()` right after the stage output has been evaluated; `TbcclPipelineComm` uses it to recycle receive
   destinations when `EXO_TBCCL_RECV=reuse` (a no-op for `MlxPipelineComm`).
5. Shutdown: the engine releases the communicator before model state is destroyed.

## Fast-path modules (Phase 54)

- `exo_tbccl/config.py`: `FastPathConfig`, environment overrides. `exo_tbccl/_cuda_caps.py`: driver capability inspection (ctypes, internal). `exo_tbccl/recv_pool.py`: bounded receive pool.
  Asynchronous sends reuse the `Transfer` pending table in `group.py` (`detached` transfers).

## Failure model

Peer death surfaces as a structured transport error on the surviving rank within about a second; the runner fails, exo reports an error chunk to
the client and the instance can be deleted. Cancellation never fails the instance. A bootstrap that cannot complete ends at its timeout (120 s) with
the ranks it was still waiting for. A failed asynchronous send is held and raised at the next communication point; every collective drains outstanding sends.

# Overview

## What it provides

exo splits a model into pipeline stages and places each stage on a machine; between stages, activations have to move. exo-tbccl adds a pipeline data plane for that: an exo **instance type** called `MlxTbccl`. With it, a Mac (Metal) and a Linux machine (CUDA) can run one pipeline over a Thunderbolt 4 cable or any TCP network, using TBCCL's explicit device handling (CUDA staged through pinned host memory, Metal unified memory read in place) and its structured failure semantics. It is an alternative to exo's own MLX backends, not a replacement: `MlxRing` is the default and also runs across a Linux and a Mac, and nothing changes unless you select `MlxTbccl`.

```text
exo Master / placement / topology
        |
 PipelineShardMetadata
        |
   exo PipelineComm
     /          \
MlxPipelineComm   TbcclPipelineComm   <- exo-tbccl
  mx.distributed      MLX/DLPack bridge
                          |
                  TBCCL C ABI v1 -> libtbccl (statically linked)
```

exo still owns discovery, topology, placement, shard assignment, model loading and scheduling. exo-tbccl only moves bytes: it turns MLX arrays into TBCCL buffers (zero-copy where the memory allows), sends and receives them, and maps TBCCL errors to typed exceptions.

## How exo selects it

You choose `MlxTbccl` when you place an instance: `"instance_meta": "MlxTbccl"` together with `"sharding": "Pipeline"` in the place-instance request. `MlxTbccl` is offered only if `exo_tbccl` imports in exo's environment; selecting it without the package is a clear placement error. exo then runs the same layer allocation as its other runtimes, advertises one address per node, builds a TBCCL communicator in each runner and runs generation through it. See [the quickstart](quickstart.md).

## Supported arrangement

| | |
|---|---|
| Stages | a Mac with Metal and a Linux host with CUDA (also CPU hosts); validated with two stages |
| Rank mapping | exo assigns each node a device rank from its shard metadata; pipeline stage `k` is rank `k`. Activations flow from rank `k` to rank `k + 1` |
| Workload | pipeline parallelism for text generation; not tensor parallelism, not image models |
| Link | a direct Thunderbolt 4 link is preferred (exo picks the address every peer can reach); any IPv4 TCP path works |

The device path is the usual TBCCL one: CUDA arrays are staged through pinned host memory by libtbccl, Metal arrays are CPU-visible unified memory handed to the transport directly. There is no direct CUDA-to-Metal DMA.

## Requirements to know before installing

exo-tbccl needs an exo that contains the `PipelineComm` seam, the `MlxTbccl` instance type and the runner byte exchange. These are not in upstream exo yet; they are published as the immutable tag `tbccl-integration-v3` of the maintained fork (v1 remains available) `tensorsofthewall/exo-fork` ([compatibility](../reference/compatibility.md)).

## Next steps

1. [Install](install.md) the wheel into exo's environment.
2. Form the cluster and run a model ([quickstart](quickstart.md), [two hosts](../guides/two-host.md)).
3. Confirm `MlxTbccl` is active ([verify](../guides/verify-active.md)).

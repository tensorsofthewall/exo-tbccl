# Where should this change go?

The TBCCL ecosystem is four repositories with strict layering. Putting a change in the right place keeps the architecture clean. **If the missing thing is generic (every adapter would want it), it belongs in tbccl. If it mentions a framework, it belongs in that framework's adapter.**

| The change is about... | It belongs in |
|---|---|
| transport, collectives, algorithms, bootstrap, the wire protocol, the C ABI, CUDA staging, Metal-shared handling, work and failure semantics, `tbccl-info`, release archives | **tbccl** |
| PyTorch `ProcessGroup` / `Work` behavior, c10d backend registration, dtype and `ReduceOp` mapping, the MPS dispatch registration, DDP behavior, the torch-tbccl wheel | **torch-tbccl** |
| the vLLM platform plugin, the device communicator, worker subclasses, the bundled c10d backend, the vllm-metal transport seam and its patch, the vllm-tbccl wheel | **vllm-tbccl** |
| the exo `MlxTbccl` instance type, the MLX/DLPack bridge, `TbcclPipelineComm`, typed errors, the exo-tbccl wheel | **exo-tbccl** |
| exo discovery, topology, placement, the `PipelineComm` seam, the runner byte exchange | the exo integration (the exo fork / upstream exo), not exo-tbccl |
| vLLM scheduling, the multi-node executor, worker-death detection | upstream **vLLM** |
| the vllm-metal model runner | upstream **vllm-metal** (vllm-tbccl carries one generic patch for the pipeline-transport seam) |
| `torch.distributed` internals, the MPS backend, Metal assertions | upstream **PyTorch** |
| MLX array behavior and device backends | upstream **MLX** |

## Layering rules

```text
libtbccl must not depend on any framework.
Adapters may depend on libtbccl.
Adapters must not require other adapters.
```

- A transport, algorithm, chunk-scheduling or staging change never goes into an adapter. If an adapter seems to need one, identify the missing generic libtbccl API instead.
- An adapter is not patched around a libtbccl defect: reproduce it outside the framework, fix it in tbccl as its own change, then update the adapter.
- A framework concept (a tensor type, a model name, a process-group class) never goes into tbccl.

## Examples

| Request | Where | Why |
|---|---|---|
| "all_reduce is slow for 8 MiB tensors" | **tbccl** | algorithm selection and chunking are runtime behavior |
| "Exo does not find the other node" | the exo integration or the network, not exo-tbccl | discovery is Exo's (IPv6 link-local multicast); see [discovery troubleshooting](../guides/discovery-troubleshooting.md) |
| "`MlxTbccl` should avoid a copy of a Metal array" | this adapter (the bridge) | the DLPack bridge and the receive pool live here |
| "make `MlxTbccl` the default backend" | the exo integration (placement), not exo-tbccl | the adapter provides `MlxTbccl`; exo chooses the instance type |
| "a cluster-wide barrier on node join" | Exo | cluster membership is not a pipeline communication |
| "use torch-tbccl's process group here" | not allowed | adapters must not require each other (the package declares no dependencies in `pyproject.toml`) |

## What does not belong in the core runtime

An Exo, MLX, model, layer or placement concept; discovery; scheduling; and framework error types. Generic behavior that every adapter would want does go to tbccl.

## How this adapter depends on TBCCL without depending on other adapters

The wheel contains a statically linked libtbccl reached through a small C extension over the C ABI. It does not import torch-tbccl or vllm-tbccl, and installing it installs neither. The shared contract is the C ABI version and the wire protocol version; a peer with a different wire protocol version is rejected at connect time ([versioning](https://tbccl.tensorsofthewall.com/en/latest/reference/versioning.html)). The optional extra of the Exo integration tag only installs this one package.

## Documentation

Document a feature in the repository that owns it. The ecosystem overview and the compatibility matrix live in tbccl and are linked from the adapters; do not copy them. When you change a version, an interface or a supported configuration, update that repository's `compatibility.json` and the pages that quote it in the same pull request.

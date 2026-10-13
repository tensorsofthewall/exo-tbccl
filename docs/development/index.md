# Development

This repository, **exo-tbccl**, is the exo adapter. It owns the MLX bridge, the C-ABI binding and `TbcclPipelineComm`, and nothing of the transport or of exo's orchestration.

| Belongs here | Does not belong here |
|---|---|
| the MLX to TBCCL bridge (DLPack, memory-kind mapping) | communication algorithms, TCP code, staging (these are TBCCL) |
| the CPython binding of the TBCCL C ABI | exo discovery, placement, topology, shard assignment (these are exo) |
| `TbcclPipelineComm`, typed errors, optional fast paths | a dependency on torch-tbccl, vllm-tbccl or PyTorch |
| tests and probes for the above | model loading and scheduling |

Start with [contributing](contributing.md) and [where should this change go?](where-should-this-change-go.md). The rules for the code base are in `AGENTS.md`.

```{toctree}
:maxdepth: 1

contributing
where-should-this-change-go
testing
building-docs
release-process
documentation-versions
```

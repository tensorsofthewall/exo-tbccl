# The TBCCL ecosystem

TBCCL is a family of four projects that let machines with different operating systems and accelerators work together over TCP, including a direct Thunderbolt 4 link. One runtime does the communication; three independent adapters connect it to PyTorch, vLLM and exo.

```text
 Applications / frameworks
          |
          +-- PyTorch ------- torch-tbccl --+
          |                                 |
          +-- vLLM ---------- vllm-tbccl ---+--->  libtbccl  --->  TCP / Thunderbolt 4
          |                                 |
          +-- exo ----------- exo-tbccl ----+
```

| Project | What it is | Use it when | Documentation | Source |
|---|---|---|---|---|
| tbccl | the runtime (libtbccl): C++17 library, stable C ABI v1, TCP transport, point-to-point and collectives, CUDA and Metal-shared memory providers | you program against TBCCL from C or C++ | [docs](https://tbccl.tensorsofthewall.com/en/latest/) | [GitHub](https://github.com/tensorsofthewall/tbccl) |
| torch-tbccl | a PyTorch `torch.distributed` backend named `"tbccl"` | you run PyTorch collectives, point-to-point or DDP across machines | [docs](https://torch-tbccl.tensorsofthewall.com/en/latest/) | [GitHub](https://github.com/tensorsofthewall/torch-tbccl) |
| vllm-tbccl | a vLLM platform plugin with its own bundled backend | you serve a vLLM model as a pipeline across a CUDA node and a Mac (or CPU node) | [docs](https://vllm-tbccl.tensorsofthewall.com/en/latest/) | [GitHub](https://github.com/tensorsofthewall/vllm-tbccl) |
| exo-tbccl (this project) | an exo pipeline data plane for MLX activations | you run exo pipeline-parallel generation across a Mac and a Linux CUDA host | [docs](https://exo-tbccl.tensorsofthewall.com/en/latest/) | [GitHub](https://github.com/tensorsofthewall/exo-tbccl) |

## How the pieces fit

- **libtbccl never depends on a framework.** It knows nothing about models, placement or machine-learning libraries.
- **Adapters depend on libtbccl, and only on libtbccl.** Each adapter statically links the TBCCL it was built with, so the wheels you install contain the runtime. You do not install tbccl separately to use an adapter.
- **Adapters are independent of each other.** torch-tbccl, vllm-tbccl and exo-tbccl do not import, link or require one another. vllm-tbccl ships its own minimal c10d backend over the C ABI; it does not use torch-tbccl. (vllm-tbccl has an optional compatibility mode, `VLLM_TBCCL_PROCESS_GROUP=torch-tbccl`, that uses a separately installed torch-tbccl instead; it is never the default and never required.)
- **Each adapter owns only its framework glue**: dtype and device mapping, framework lifecycle hooks and error mapping. Transport, collectives, staging and failure semantics belong to libtbccl.

## Device data path

CUDA buffers are staged through pinned host memory; Metal shared (unified) memory is handed to the transport directly. There is no direct CUDA-to-Metal DMA, and the transport is TCP, not RDMA. See the [core architecture](https://tbccl.tensorsofthewall.com/en/latest/concepts/architecture.html).

## Current versions

The current stable versions, tags and framework pins of all four projects are on the [canonical ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html), maintained in the TBCCL documentation. They are not repeated here, because documentation built from a release tag is frozen and would go stale when another project releases. This project's own release-specific compatibility is in its [compatibility reference](../reference/compatibility.md). All four share **C ABI 1** and **wire protocol 4**; ranks that speak different wire protocols cannot communicate.

## Hosted documentation

Each project has its own site. `latest` is development documentation built from `main`; released versions appear as numbered versions (for example `v0.6.0`) and `stable` is the latest final release of that project.

## Where a change belongs

See [where should this change go?](../development/where-should-this-change-go.md)

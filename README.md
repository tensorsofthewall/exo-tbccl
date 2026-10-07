# exo-tbccl

exo-tbccl is an optional package that lets [exo](https://github.com/exo-explore/exo) (an MLX-based distributed inference system) run **pipeline-parallel text generation** across machines of different kinds, for example a Mac with Metal and a Linux host with CUDA, with [TBCCL](https://github.com/tensorsofthewall/tbccl) moving the activations between stages over its stable C ABI. It is separate from exo and from TBCCL, and uses no PyTorch and no MLX C++ internals.

> **Status:** development version 0.2.1, experimental, no release published.

## What you can use it for

- Running an exo pipeline (the `MlxTbccl` instance type) over a Thunderbolt 4 link or any TCP network between a Mac (Metal), a Linux host (CUDA) or CPU hosts.
- Pipeline parallelism and text models only; not tensor parallelism and not image models. exo owns discovery, topology, placement and shard assignment.

## Install

Into exo's virtual environment, against an installed TBCCL 0.5 or newer (C ABI 1):

```sh
TBCCL_ROOT=<tbccl prefix> uv pip install --python <exo venv>/bin/python -e .
```

exo works without this package; selecting `MlxTbccl` without it is a clear placement error and `exo_tbccl.is_available()` explains why. More: [installing](docs/getting-started/install.md).

## Minimal example

exo calls it for you; the API it uses is:

```python
from exo_tbccl.group import TbcclPipelineComm
comm = TbcclPipelineComm.create(rank, world_size, exchange, bind_host=host, advertise_host=host)
x = comm.recv_like(template, src)      # MLX array in, MLX array out
comm.send(array, dst)
y = comm.all_gather(array)
comm.close()
```

Rank and world size come from exo's `PipelineShardMetadata`. Errors are typed exceptions built from TBCCL's structured result codes. See the [quickstart](docs/getting-started/quickstart.md).

## Supported configurations

| | Validated |
|---|---|
| Stages | Mac Metal and Linux CUDA over Thunderbolt 4 (also CPU and loopback) |
| Models | Qwen3-0.6B-8bit (loopback) and `Qwen3.8-27B-4bit` (two hosts, real exo) |
| TBCCL | C ABI 1; physical runs recorded at an earlier wire protocol, loopback suites at wire protocol 4 |

Optional fast paths and Metal activity policies are off by default ([fast paths](docs/concepts/fast-paths.md)). See [compatibility](docs/reference/compatibility.md) and the draft [validation](docs/validation/0.2.1.md).

## Documentation

The documentation is in `docs/` and builds with `make docs`: [getting started](docs/getting-started/index.md), [guides](docs/guides/index.md), [concepts](docs/concepts/index.md), [reference](docs/reference/index.md). Contributing: `CONTRIBUTING.md` and `AGENTS.md`.

## License

No license file has been published yet.

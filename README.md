# exo-tbccl

[![docs](https://github.com/tensorsofthewall/exo-tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/exo-tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/exo-tbccl?include_prereleases&label=pre-release)](https://github.com/tensorsofthewall/exo-tbccl/releases)

exo-tbccl is an optional package that lets [exo](https://github.com/exo-explore/exo) (an MLX-based distributed inference system) run **pipeline-parallel text generation** across machines of different kinds, for example a Mac with Metal and a Linux host with CUDA, with [TBCCL](https://github.com/tensorsofthewall/tbccl) moving the activations between stages over its stable C ABI. It is separate from exo and from TBCCL, and uses no PyTorch and no MLX C++ internals.

> **Status:** release candidate **0.3.0rc1** (pre-release, experimental; not production-ready). The final 0.3.0 has not been released. It is one of three independent adapters over TBCCL (with [torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl) and [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl)) and depends on neither.

## What is supported

- Running an exo pipeline (the `MlxTbccl` instance type) over a Thunderbolt 4 link or any TCP network between a Mac (Metal), a Linux host (CUDA) or CPU hosts.
- Pipeline parallelism and text models only; not tensor parallelism and not image models. exo owns discovery, topology, placement and shard assignment; exo-tbccl only moves the activations.

```text
exo (discovery, placement)  ->  exo-tbccl (MLX bridge + TbcclPipelineComm)  ->  libtbccl (statically linked, C ABI v1)  ->  TCP / Thunderbolt 4
```

## Requirements

- **exo with the `PipelineComm` seam.** exo-tbccl needs an exo build that contains the integration (the `MlxTbccl` instance type, the `PipelineComm` seam and the runner byte exchange). They are a short series of commits over upstream exo `75d8ad2b` (2026-10-12) that are not part of upstream exo and have **not been published yet** ([compatibility](docs/reference/compatibility.md)). Until they are, a working cluster from public sources is not possible.
- Python 3.13 or newer, MLX 0.32 (CUDA 13 on Linux, Metal on macOS; exo's pinned macOS MLX, a 0.32.4 development build, is supported: see [compatibility](docs/reference/compatibility.md)).
- Nothing else of TBCCL to install: the wheel contains libtbccl.

## Quick start

Install the release-candidate wheel into **exo's** virtual environment:

```sh
# Linux (CUDA):
uv pip install --python <exo venv>/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0rc1/exo_tbccl-0.3.0rc1-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# macOS (Apple silicon):
uv pip install --python <exo venv>/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0rc1/exo_tbccl-0.3.0rc1-cp313-cp313-macosx_14_0_arm64.whl
<exo venv>/bin/python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"    # 0.3.0rc1 (True, '')
```

Start exo on both hosts (see [running across two hosts](docs/guides/two-host.md): on the Mac, launch exo from a live `ssh -tt` session, not from detached tmux or nohup), then select the pipeline type:

```sh
curl -X POST localhost:52415/place_instance -H 'content-type: application/json' \
  -d '{"model_id": "mlx-community/Qwen3-0.6B-8bit", "sharding": "Pipeline", "instance_meta": "MlxTbccl", "min_nodes": 2}'
```

exo calls exo-tbccl for you; there is no separate API to learn. Confirm it is active: the instance in `GET /state` is an `MlxTbcclInstance` and the runner logs `TBCCL communicator ready` ([how do I know TBCCL is being used?](docs/guides/verify-active.md)).

## Minimal example (what exo does)

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
| Models | Qwen3-0.6B (two hosts, release-candidate wheels, three full start-to-shutdown cycles) and `Qwen3.8-27B-4bit` (two hosts, wire protocol 4, both pipeline orders) |
| TBCCL | C ABI 1, wire protocol 4 (bundled) |

Optional fast paths and Metal activity policies are off by default ([fast paths](docs/concepts/fast-paths.md)). See [compatibility](docs/reference/compatibility.md), the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html) and [validated configurations](docs/project/validated-configurations.md).

## Documentation

Hosted documentation: https://exo-tbccl.tensorsofthewall.com/ (`latest` is development documentation built from `main`; release-candidate versions appear as numbered versions; `stable` does not exist until the final release). Build it locally with `make docs`. Start with [getting started](docs/getting-started/index.md), [running across two hosts](docs/guides/two-host.md), [discovery troubleshooting](docs/guides/discovery-troubleshooting.md) and [known limitations](docs/project/limitations.md).

## Known limitations

Pipeline sharding and text generation only; one address per node; the exo integration is not yet published; on macOS the exo node must be started from a live session and no other program may listen on UDP port 52413 (a validated operational workaround, not a Thunderbolt requirement). See [known limitations](docs/project/limitations.md).

## Development

`CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base; the [development documentation](docs/development/index.md) covers building, testing and where a change belongs. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

exo-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.

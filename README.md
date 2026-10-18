# exo-tbccl

[![docs](https://github.com/tensorsofthewall/exo-tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/exo-tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/exo-tbccl?label=release)](https://github.com/tensorsofthewall/exo-tbccl/releases)

exo-tbccl is an optional package that lets [exo](https://github.com/exo-explore/exo) (an MLX-based distributed inference system) run **pipeline-parallel text generation** across machines of different kinds, for example a Mac with Metal and a Linux host with CUDA, with [TBCCL](https://github.com/tensorsofthewall/tbccl) moving the activations between stages over its stable C ABI. It is separate from exo and from TBCCL, and uses no PyTorch and no MLX C++ internals.

> **Status:** **0.3.1** (patch release of 0.3.0: adds the public Exo benchmark provider; experimental). It is one of three independent adapters over TBCCL (with [torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl) and [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl)) and depends on neither.

## What is supported

- Running an exo pipeline (the `MlxTbccl` instance type) over a Thunderbolt 4 link or any TCP network between a Mac (Metal), a Linux host (CUDA) or CPU hosts.
- Pipeline parallelism and text models only; not tensor parallelism and not image models. exo owns discovery, topology, placement and shard assignment; exo-tbccl only moves the activations.

```text
exo (discovery, placement)  ->  exo-tbccl (MLX bridge + TbcclPipelineComm)  ->  libtbccl (statically linked, C ABI v1)  ->  TCP / Thunderbolt 4
```

## Supported configurations

| | Validated |
|---|---|
| Stages | Mac Metal and Linux CUDA over Thunderbolt 4 (also CPU and loopback) |
| Models | Qwen3-0.6B (two hosts, release-candidate wheels, three full start-to-shutdown cycles) and `Qwen3.8-27B-4bit` (two hosts, wire protocol 4, both pipeline orders) |
| exo | the fork [`tensorsofthewall/exo-fork`](https://github.com/tensorsofthewall/exo-fork) at the immutable tag `tbccl-integration-v2` (upstream exo `bc412e22` plus the integration; `tbccl-integration-v1` remains available) |
| Python / MLX | CPython 3.13 (the wheels are `cp313`); MLX 0.32 from exo's own environment (CUDA 13 on Linux, Metal on macOS) |
| Platforms | Linux x86-64 (CUDA, CPU), macOS arm64 (Metal, CPU) |
| TBCCL | C ABI 1, wire protocol 4 (bundled in the wheel) |

Optional fast paths and Metal activity policies are off by default ([fast paths](docs/concepts/fast-paths.md)). See [compatibility](docs/reference/compatibility.md), the [current ecosystem status](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html) (a living page), the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html) and [validated configurations](docs/project/validated-configurations.md).

## Installation

exo-tbccl needs an exo that contains the TBCCL integration, which upstream exo does not: the immutable tag `tbccl-integration-v2` of [exo-fork](https://github.com/tensorsofthewall/exo-fork). You do not install TBCCL or a separate libtbccl: the wheel contains the native runtime (C ABI 1, wire protocol 4; TBCCL 0.6.1 in the 0.3.1 wheel, TBCCL 0.6.0 in the 0.3.0 wheel that the `tbccl` extra installs). exo-tbccl stays an optional dependency of exo: exo works without it. Install on **every** node, in the checkout's own uv environment (Python 3.13, uv, node and Rust as for any exo source install).

The tag's own README predates this installation method; follow the commands here. The `tbccl` extra installs the 0.3.0 release wheel on purpose, not because the pin is stale: the tag is immutable, 0.3.1 has the same adapter code and only adds the benchmark provider, so the extra keeps resolving to 0.3.0 (install the 0.3.1 wheel by hand to have 0.3.1 in the environment). It applies on Linux x86-64 and macOS arm64 only; on any other platform it installs nothing.

```sh
git clone --depth 1 --branch tbccl-integration-v2 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
# Linux x86-64 (CUDA 13):
uv sync --no-dev --locked --extra mlx-cuda13 --extra tbccl
# macOS arm64 (Metal), instead of the line above:
#   uv sync --no-dev --locked --extra mlx --extra tbccl
uv run --no-sync python -c "import exo_tbccl; print(exo_tbccl.is_available())"    # (True, '')
```

**Legacy: `tbccl-integration-v1` and a manual wheel.** The earlier integration tag has no extra; the adapter wheel is installed by hand. It stays available and unchanged:

```sh
git clone --branch tbccl-integration-v1 --depth 1 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
# Linux x86-64 (CUDA 13):
uv sync --no-dev --extra mlx-cuda13
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.1/exo_tbccl-0.3.1-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# macOS arm64 (Metal), instead of the two lines above:
#   uv sync --no-dev --extra mlx
#   uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.1/exo_tbccl-0.3.1-cp313-cp313-macosx_14_0_arm64.whl
```

With v1, a plain `uv sync` removes the manually installed wheel: later syncs use `uv sync --inexact ...`, or reinstall the wheel. [Installing](docs/getting-started/install.md) explains both paths.

**Optional check and sync helper.** The development line of exo-fork (`main`, not the `tbccl-integration-v2` tag) has `tools/setup_tbccl.py`: `check` reports, without changing anything, whether the tools are present and prints the exact `uv sync` command for your platform, `sync` runs that locked sync, and `doctor` prints the installed exo, adapter and TBCCL runtime versions, the model directories and whether the discovery and API ports are free. It installs nothing system-wide and changes no network setting. Future integration tags will contain it; the commands above remain the supported path for the tag.

## Verify the installation

```sh
uv run --no-sync python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"    # 0.3.0 (True, '') with the tbccl extra of tbccl-integration-v2; 0.3.1 (True, '') after installing the 0.3.1 wheel by hand
```

`(False, reason)` says what is missing. Run it on every node.

## Quick start: a two-node pipeline

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

## Documentation

Hosted documentation: https://exo-tbccl.tensorsofthewall.com/ (`stable` is the latest final release, numbered versions are earlier releases, and `latest` is development documentation built from `main`). Build it locally with `make docs`. Start with [getting started](docs/getting-started/index.md), [running across two hosts](docs/guides/two-host.md), [discovery troubleshooting](docs/guides/discovery-troubleshooting.md) and [known limitations](docs/project/limitations.md).

Questions, feedback and what to include in a report: [getting help](https://tbccl.tensorsofthewall.com/en/latest/project/getting-help.html). The projects collect [no telemetry](https://tbccl.tensorsofthewall.com/en/latest/project/privacy.html).

## Benchmarks

Run the Exo pipeline benchmarks (placement and backend verification, TTFT, TPOT, throughput, CPU cost, MlxRing against MlxTbccl) on your own hardware, standalone or through the TBCCL benchmark runner: [benchmarks](docs/benchmarks/index.md). Reference measurements from one machine pair, with their limits and the exploratory results labelled, are in the [results](docs/benchmarks/results.md).

## Limitations

Pipeline sharding and text generation only; one address per node; the exo integration is a fork tag, not in upstream exo; on macOS the exo node must be started from a live session and no other program may listen on UDP port 52413 (a validated operational workaround, not a Thunderbolt requirement). See [known limitations](docs/project/limitations.md).

## Development

[MAINTENANCE.md](MAINTENANCE.md) records the maintenance policy and what needs physical requalification. `CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base; the [development documentation](docs/development/index.md) covers building, testing and where a change belongs. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

exo-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.

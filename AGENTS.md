# AGENTS.md

Technical guidance for contributors and coding agents working in this repository. User documentation is in `README.md`; design notes are in `docs/concepts/architecture.md`, `docs/concepts/bridge.md` and `docs/concepts/bootstrap.md`. Contribution workflow is in `CONTRIBUTING.md`.

## Purpose

`exo-tbccl` is an optional package that lets exo (an MLX-based distributed inference system) run pipeline-parallel text generation across machines, for example Mac Metal and Linux CUDA, with TBCCL moving the activations.

```
exo (placement, shards, runners) -> exo PipelineComm -> TbcclPipelineComm (this repo) -> DLPack bridge -> TBCCL C ABI v1 -> libtbccl
```

## Layout

| Path | Role |
|---|---|
| `exo_tbccl/` | Python package: `TbcclPipelineComm`, configuration, group handling, DLPack bridge |
| `src/native/` | C extension over the TBCCL C ABI |
| `tests/`, `examples/`, `benchmarks/` | Process-per-rank tests, probes, benchmarks and measurement tools |

## Architecture boundaries

- Link only the stable C ABI (`TBCCL::tbccl_c`, an installed TBCCL >= 0.5). No TBCCL C++ header, source-tree include or private symbol.
- No PyTorch in the bridge and no MLX C++ internals. Arrays enter through the Python DLPack protocol and the DLPack C structs.
- exo owns discovery, topology, placement, shard assignment, model loading and scheduling. This package never rediscovers interfaces and never derives rank from node ordering: rank is the pipeline shard's device rank and world size comes from exo.
- TBCCL core is not changed for exo's convenience. Fix generic TBCCL defects in the TBCCL repository.
- Pipeline parallelism and text models only. No tensor parallelism, no image or CFG models.

## Critical invariants

- Never hide a copy. Any payload-sized copy made by the adapter is counted in `CopyStats` and visible in trace mode (`EXO_TBCCL_TRACE=1`). The zero-copy paths (Metal, CUDA) must report 0.
- Lifetime: a borrowed MLX array, its view and its DLPack export stay alive until the TBCCL `Work` is terminal. No correctness may depend on `__del__` or interpreter shutdown; `close()` is explicit.
- Errors come from structured result codes, never from parsing strings; keep rank, peer and operation in the exception.
- Blocking calls release the GIL.
- Closing a communicator with work in flight aborts it, and an abort fails a peer that is still completing its last collective. Quiesce all ranks before closing.
- Do not `fork` a process that already imported MLX or CUDA; use `spawn`.
- Optional fast paths are off by default and opt-in (see `README.md`); a default-path change needs the same evidence as a new feature.

## Build and test

```sh
TBCCL_ROOT=<installed tbccl prefix> uv pip install --python <exo venv>/bin/python -e .
<exo venv>/bin/python -m pytest -q -p no:asyncio tests    # loopback, process-per-rank
```

- The package is installed into exo's virtual environment; a dependency sync in exo can uninstall it, so reinstall afterwards. Rebuild the extension after editing `src/native/`.
- ASan/UBSan can run the non-MLX tests only; MLX/CUDA tests cannot run under ASan.
- Real two-host runs need explicit link-health checks and are not part of ordinary testing. Never download model weights.

## Code ownership expectations

Changes to `src/native/`, the DLPack bridge and the default (non-opt-in) communication path need maintainer review.

## Conventions

- Match the surrounding code. Comment only non-obvious constraints.
- Do not commit absolute machine paths or model locations.
- Add explicit paths when staging files.

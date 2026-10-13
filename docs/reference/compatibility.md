# Compatibility

| | Value |
|---|---|
| Package version | 0.3.0rc1 (release candidate; no final release has been published); the planned release is 0.3.0 and is **unreleased** |
| Python | 3.13 or newer (`requires-python`); the wheels are `cp313` |
| TBCCL | bundled and statically linked: C ABI 1, wire protocol 4 (minimum TBCCL package 0.6.0 when building from source) |
| exo | the exo integration described below |
| MLX | 0.32 (Linux CUDA 13, macOS Metal) |
| Platforms | Linux x86-64 (CUDA, CPU), macOS arm64 (Metal, CPU) |

Core transport compatibility (platforms, CUDA, world sizes, the C ABI and wire protocol) is defined by the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html). exo-tbccl depends only on exo and MLX at run time (libtbccl is bundled); it does not depend on torch-tbccl or vllm-tbccl.

## Wheels

| Wheel | Platform |
|---|---|
| `exo_tbccl-0.3.0rc1-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl` | Linux x86-64 |
| `exo_tbccl-0.3.0rc1-cp313-cp313-macosx_14_0_arm64.whl` | macOS 14 or newer, Apple silicon |

## The required exo

exo-tbccl needs an exo that contains four things: the `PipelineComm` protocol seam, the `MlxTbccl` instance type and its placement rules, the generic runner byte all-gather (`RunnerByteExchange`), and the fix that keeps draining the byte-exchange inbox after the exchange completes. None is in upstream exo yet.

The validated integration is a short series of commits over upstream exo `75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6` (2026-10-12; upstream `main` at the time), re-derived on that base from the earlier series over `21a54c5e`: upstream had meanwhile stopped rebuilding the `Worker` on master election (`announce_to_master`), moved its macOS MLX pin to a 0.32.4 development build and reworked placement, the KV prefix cache and the runner bootstrap, so the seams were ported rather than rebased mechanically. The series is maintained as a local branch of the maintainers and **has not been published**: there is no public repository or upstream pull request to install exo with this integration from yet. This is tracked for the final release (publish the integration, or propose it upstream). Until then, exo-tbccl can be exercised only with that branch.

## MLX versions

MLX changed how a Metal array is exported through DLPack between its 0.32.0 development builds and the 0.32.4 development build that upstream exo now pins on macOS: the older builds put the buffer's CPU-visible address in `data`, the newer ones the `id<MTLBuffer>` handle (DLPack's own convention) with the array's offset in `byte_offset`. exo-tbccl 0.2 era code read the handle as an address, which crashed the ranks with SIGSEGV/SIGBUS. The bridge now establishes the convention once per process with an exact-content probe and refuses an export it cannot interpret; it works with both. Validated: macOS MLX `0.32.4.dev20261013+0e3ff3643` (exo's pin) and `0.32.0.dev20261004+cc3f3e60` (the earlier pin), Linux CUDA MLX 0.32.0.

## Validated configurations

See [validated configurations](../project/validated-configurations.md) and [validation](../validation/0.3.0.md). The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `pyproject.toml` and the extension's C ABI constant.

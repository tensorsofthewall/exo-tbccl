# Compatibility

| | Value |
|---|---|
| Package version | 0.4.0rc1 |
| Python | `requires-python` is 3.13 or newer; the published wheels are `cp313` only, and no other interpreter has a validated prebuilt wheel |
| TBCCL | bundled and statically linked: C ABI 1, wire protocol 4. The 0.4.0rc1 wheel bundles TBCCL 0.7.0rc1 and the 0.3.0 wheel (installed by the `tbccl` extra of `tbccl-integration-v2`) bundles TBCCL 0.6.0; the minimum TBCCL package when building from source is 0.6.0 |
| exo | the exo integration described below |
| MLX | 0.32 (Linux CUDA 13, macOS Metal) |
| Platforms | Linux x86-64 (CUDA, CPU), macOS arm64 (Metal, CPU) |

Core transport compatibility (platforms, CUDA, world sizes, the C ABI and wire protocol) is defined by the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html). exo-tbccl depends only on exo and MLX at run time (libtbccl is bundled); it does not depend on torch-tbccl or vllm-tbccl.

## Wheels

| Wheel | Platform |
|---|---|
| `exo_tbccl-0.4.0rc1-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl` | Linux x86-64 |
| `exo_tbccl-0.4.0rc1-cp313-cp313-macosx_14_0_arm64.whl` | macOS 14 or newer, Apple silicon |

## The required exo

exo-tbccl needs an exo that contains four things: the `PipelineComm` protocol seam, the `MlxTbccl` instance type and its placement rules, the generic runner byte all-gather (`RunnerByteExchange`), and the fix that keeps draining the byte-exchange inbox after the exchange completes. None is in upstream exo yet.

They are published as the maintained fork `tensorsofthewall/exo-fork` at the immutable tag `tbccl-integration-v2` (commit `d1497b28b89618172d273e70e9ac7992811635f5`, upstream exo `bc412e22` plus the TBCCL integration). The tag is annotated and immutable; install exo from it, never from the fork's `main` (which tracks upstream and carries the integration forward but is not the compatibility anchor).

```sh
git clone --depth 1 --branch tbccl-integration-v2 https://github.com/tensorsofthewall/exo-fork.git
```

The tag is upstream exo `bc412e2230f395e652c23426a5eb88dfa515769b` (2026-10-13) plus the unchanged integration and an optional `tbccl` dependency extra (platform-gated 0.3.0 wheel sources, a verified `uv.lock`), so that `uv sync --locked --extra mlx-cuda13 --extra tbccl` (Linux) or `--extra mlx --extra tbccl` (macOS) installs the adapter. exo-tbccl 0.3.0 is the adapter version of both integrations: the wheel did not change. exo-tbccl 0.4.0rc1 adds opt-in Metal options and bundles TBCCL 0.7.0rc1; the `tbccl` extra of this immutable tag keeps resolving to the 0.3.0 wheel, and the 0.4.0rc1 wheel is installed by hand over it. The tag's own README predates this installation method. Physical qualification of v2: one two-rank `MlxTbccl` pipeline (Mac Metal and Linux CUDA, Thunderbolt 4, `Qwen3-0.6B-8bit`) completed a short generation with no new AER errors or link instability; no performance, scalability, recovery or other-model claim.

**Legacy integration.** `tbccl-integration-v1` (commit `1416807b46efd87c3601c69ce5784f51c9dd4c75`, upstream exo `75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6`, 2026-10-12) remains available and unchanged, installed with the 0.3.0 wheel by hand ([installing](../getting-started/install.md)). It was the integration re-derived on that base: the seams were ported rather than rebased mechanically over the earlier series on `21a54c5e`, because upstream had meanwhile stopped rebuilding the `Worker` on master election, moved its macOS MLX pin to a 0.32.4 development build and reworked placement, the KV prefix cache and the runner bootstrap. The integration is meant to be proposed upstream; until then the tag is the anchor.

## MLX versions

MLX changed how a Metal array is exported through DLPack between its 0.32.0 development builds and the 0.32.4 development build that upstream exo now pins on macOS: the older builds put the buffer's CPU-visible address in `data`, the newer ones the `id<MTLBuffer>` handle (DLPack's own convention) with the array's offset in `byte_offset`. exo-tbccl 0.2 era code read the handle as an address, which crashed the ranks with SIGSEGV/SIGBUS. The bridge now establishes the convention once per process with an exact-content probe and refuses an export it cannot interpret; it works with both. Validated: macOS MLX `0.32.4.dev20261013+0e3ff3643` (exo's pin) and `0.32.0.dev20261004+cc3f3e60` (the earlier pin), Linux CUDA MLX 0.32.0.

## Validated configurations

See [validated configurations](../project/validated-configurations.md) and [validation](../validation/0.3.0.md). The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `pyproject.toml` and the extension's C ABI constant.

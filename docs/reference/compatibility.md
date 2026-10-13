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

The validated integration is **14 commits over upstream exo `21a54c5e`** (2026-08-25, "docs: request the mlx extra in the documented setup commands (#2245)"); its last commit is "Keep draining the runner byte-exchange inbox after the exchange completes". The commits are maintained as a local branch of the maintainers and **have not been published**: there is no public repository or upstream pull request to install exo with this integration from yet. This is tracked for the final release (publish the integration, or propose it upstream). Until then, exo-tbccl 0.3.0rc1 can be exercised only with that branch.

## Validated configurations

See [validated configurations](../project/validated-configurations.md) and [validation](../validation/0.3.0.md). The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `pyproject.toml` and the extension's C ABI constant.

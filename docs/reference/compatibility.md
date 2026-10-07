# Compatibility

| | Value |
|---|---|
| Package version | 0.2.1 (development; no release has been published); the next release is planned as 0.3.0 and is **unreleased** |
| Python | 3.13 or newer (`requires-python`) |
| TBCCL | 0.5 or newer, C ABI 1 (the extension links only `TBCCL::tbccl_c`); rebuilt against a wire protocol 4 prefix for the current development tree |
| exo | the exo integration branch with the `PipelineComm` seam, the runner byte exchange and the glue fix that drains the byte-exchange inbox |
| MLX | 0.32 (Linux CUDA 13, macOS Metal) |
| Platforms | Linux x86-64 (CUDA), macOS arm64 (Metal), CPU |

Validated configurations are listed in [validation](../validation/0.2.1.md). The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `pyproject.toml` and the extension's C ABI constant.

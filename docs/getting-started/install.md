# Installing exo-tbccl

exo-tbccl is an optional package that is installed into **exo's** virtual environment. exo works without it: selecting the `MlxTbccl` instance type without the package is a clear placement error, and `exo_tbccl.is_available()` explains why.

## Requirements

- Python 3.13 or newer and a C compiler (the package builds a small native extension with scikit-build-core).
- An installed [TBCCL](https://github.com/tensorsofthewall/tbccl) 0.5 or newer (C ABI 1): a CUDA-enabled install on Linux with an NVIDIA GPU, a host or Metal install on macOS. The extension links only `TBCCL::tbccl_c`.
- exo with its MLX environment: `mx-cuda` on Linux or `mlx` on macOS.

## Install

```sh
TBCCL_ROOT=<tbccl prefix> uv pip install --python <exo venv>/bin/python -e .
```

A dependency sync inside exo can uninstall the package; reinstall afterwards. Rebuild after editing `src/native/`.

## Check

```python
import exo_tbccl
print(exo_tbccl.is_available())    # (True, "") when the native binding loads and the TBCCL C ABI matches
```

After installing a TBCCL with a different wire protocol version, rebuild the extension against the new prefix; ranks built against different wire versions cannot connect ([TBCCL versioning](https://github.com/tensorsofthewall/tbccl/blob/main/docs/reference/versioning.md)).

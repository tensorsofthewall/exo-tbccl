# Installing exo-tbccl

exo-tbccl is installed into **exo's** virtual environment. exo works without it: selecting the `MlxTbccl` instance type without the package is a clear placement error, and `exo_tbccl.is_available()` explains why. The wheel contains libtbccl (statically linked); you do not install tbccl separately.

## Requirements

| | |
|---|---|
| Python | 3.13 or newer (the wheels are `cp313`) |
| exo | an exo with the `PipelineComm` seam, `MlxTbccl` and the runner byte exchange: the integration series described in [compatibility](../reference/compatibility.md) (commits over upstream exo `75d8ad2b`; not yet published) |
| MLX | 0.32.x (`mlx-cuda13` extra on Linux with CUDA 13, `mlx` on macOS), from exo's own environment; validated with 0.32.0 (Linux CUDA) and the 0.32.4 development build `0e3ff3643` that upstream exo pins on macOS |
| Platforms | Linux x86-64 (CUDA, CPU) and macOS arm64 (Metal, CPU); wheels `manylinux_2_24`/`2_28` and `macosx_14_0_arm64` |
| TBCCL | bundled: C ABI 1, wire protocol 4 |

## Install the release-candidate wheel

The 0.3.0rc1 wheels are attached to the GitHub pre-release [`v0.3.0rc1`](https://github.com/tensorsofthewall/exo-tbccl/releases/tag/v0.3.0rc1). They are not published on PyPI or TestPyPI. The wheel does **not** need an editable install.

```sh
# Linux:
uv pip install --python <exo venv>/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0rc1/exo_tbccl-0.3.0rc1-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# macOS:
uv pip install --python <exo venv>/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0rc1/exo_tbccl-0.3.0rc1-cp313-cp313-macosx_14_0_arm64.whl
```

Install it on **every** node, with the same version. `--no-deps` keeps the installer from touching exo's MLX environment. A dependency sync inside exo (`uv sync`) can uninstall the package; reinstall afterwards.

## Check

```python
import exo_tbccl
print(exo_tbccl.__version__, exo_tbccl.is_available())    # 0.3.0rc1 (True, '')
```

`is_available()` returns `(True, "")` when the native binding loads and the bundled TBCCL C ABI matches; otherwise `(False, reason)`. Run it with exo's Python, on every node.

## Build from source

Only needed without a matching wheel or to change the code. Build against an installed TBCCL prefix (a [release archive](https://tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) works), 0.6.0 or newer:

```sh
TBCCL_ROOT=<tbccl prefix> uv pip install --python <exo venv>/bin/python -e .
```

A C compiler builds the small native extension (scikit-build-core). Rebuild after editing `src/native/`.

## After upgrading TBCCL

Because the wheel bundles TBCCL, upgrading means installing a newer exo-tbccl wheel on **every** node. Ranks built against different wire protocols cannot connect ([TBCCL versioning](https://tbccl.tensorsofthewall.com/en/latest/reference/versioning.html)).

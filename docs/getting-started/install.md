# Installing exo-tbccl

exo-tbccl is installed into **exo's** virtual environment. exo works without it: selecting the `MlxTbccl` instance type without the package is a clear placement error, and `exo_tbccl.is_available()` explains why. The wheel contains libtbccl (statically linked); you do not install tbccl separately.

## Requirements

| | |
|---|---|
| Python | 3.13 or newer (the wheels are `cp313`) |
| exo | the exo fork with the TBCCL integration: `tensorsofthewall/exo-fork` at the immutable tag `tbccl-integration-v1` (commit `1416807b46efd87c3601c69ce5784f51c9dd4c75`, upstream exo `75d8ad2b` plus the TBCCL integration) ([compatibility](../reference/compatibility.md)) |
| MLX | 0.32.x (`mlx-cuda13` extra on Linux with CUDA 13, `mlx` on macOS), from exo's own environment; validated with 0.32.0 (Linux CUDA) and the 0.32.4 development build `0e3ff3643` that upstream exo pins on macOS |
| Platforms | Linux x86-64 (CUDA, CPU) and macOS arm64 (Metal, CPU); wheels `manylinux_2_24`/`2_28` and `macosx_14_0_arm64` |
| TBCCL | bundled: C ABI 1, wire protocol 4 |

## Install

exo-tbccl goes into the environment of an exo that carries the integration. There are two paths; do not mix their requirements.

### Path A: the supported tag and a manual wheel (use this today)

The supported exo is the immutable tag `tbccl-integration-v1` of [exo-fork](https://github.com/tensorsofthewall/exo-fork). The 0.3.0 wheels are attached to the GitHub release [`v0.3.0`](https://github.com/tensorsofthewall/exo-tbccl/releases/tag/v0.3.0). They are not published on PyPI or TestPyPI. The wheel does **not** need an editable install, and nobody has to build libtbccl or the adapter from source to use it. Use exo's own uv environment (`.venv` in the checkout), named explicitly in each command:

```sh
git clone --branch tbccl-integration-v1 --depth 1 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
# Linux x86-64 (CUDA 13):
uv sync --no-dev --extra mlx-cuda13
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0/exo_tbccl-0.3.0-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# macOS arm64 (Metal):
uv sync --no-dev --extra mlx
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0/exo_tbccl-0.3.0-cp313-cp313-macosx_14_0_arm64.whl
```

Install it on **every** node, with the same version. `--no-deps` keeps the installer from touching exo's MLX environment. `uv sync` is an exact sync and removes the wheel again: after installing it, synchronise with `uv sync --inexact --no-dev --extra ...` (verified to keep it) or reinstall the wheel. An exo that lives somewhere else works the same way with its own `.venv/bin/python`.

### Path B: the optional extra (pending the next integration tag)

A development revision of exo-fork declares `exo-tbccl` as an optional dependency, resolved to the published release wheel on Linux x86-64 and macOS arm64 (and doing nothing elsewhere), with the lock recording both wheels and their hashes:

```sh
uv sync --no-dev --extra mlx-cuda13 --extra tbccl     # Linux CUDA 13
uv sync --no-dev --extra mlx --extra tbccl            # macOS
```

On Linux x86-64 (CUDA 13) and on macOS arm64 (Metal) this was verified to install exactly one extra package (`exo-tbccl 0.3.0`) compared with the same sync without `--extra tbccl`, to keep it across a repeated sync, and to leave the MLX and PyTorch pins unchanged. On macOS the first sync builds MLX from its pinned git revision (Xcode and the Metal toolchain are needed, and it takes several minutes); a later sync may reinstall that date-stamped MLX build, which is existing behaviour and leaves `exo-tbccl` installed. On any other platform (for example Linux aarch64 or Intel macOS) no adapter wheel exists, the extra installs nothing, and the same diagnostic as without the extra appears: `exo_tbccl is not installed (No module named 'exo_tbccl')`, and selecting `MlxTbccl` fails placement with `MlxTbccl requires the optional exo-tbccl package: <that reason>`. It is **pending**: it is not in `tbccl-integration-v1`, which is never changed, and becomes the supported path when a new integration tag with the extra is released and qualified. exo without the extra behaves as before.

## Check

```python
import exo_tbccl
print(exo_tbccl.__version__, exo_tbccl.is_available())    # 0.3.0 (True, '')
```

`is_available()` returns `(True, "")` when the native binding loads and the bundled TBCCL C ABI matches; otherwise `(False, reason)`. Run it with exo's Python (`.venv/bin/python`), on every node.

## Build from source

For developers only: needed without a matching wheel or to change the code, and not part of installing the wheel. Build against an installed TBCCL prefix (a [release archive](https://tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) works), 0.6.0 or newer:

```sh
TBCCL_ROOT=<tbccl prefix> uv pip install --python .venv/bin/python -e .
```

A C compiler builds the small native extension (scikit-build-core). Rebuild after editing `src/native/`.

## After upgrading TBCCL

Because the wheel bundles TBCCL, upgrading means installing a newer exo-tbccl wheel on **every** node. Ranks built against different wire protocols cannot connect ([TBCCL versioning](https://tbccl.tensorsofthewall.com/en/latest/reference/versioning.html)).

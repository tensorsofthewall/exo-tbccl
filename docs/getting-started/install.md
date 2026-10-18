# Installing exo-tbccl

exo-tbccl is installed into **exo's** virtual environment. exo works without it: selecting the `MlxTbccl` instance type without the package is a clear placement error, and `exo_tbccl.is_available()` explains why. The wheel contains libtbccl (statically linked); you do not install tbccl separately.

## Which integration do I have, and which do I want?

exo-tbccl installs into the environment of an **exo that carries the TBCCL integration**. Two immutable tags of [exo-fork](https://github.com/tensorsofthewall/exo-fork) exist, with different install paths. Read the tag of your checkout (read-only):

```sh
git -C <exo-fork checkout> describe --tags --exact-match      # prints tbccl-integration-v2 or tbccl-integration-v1
```

| Tag | How exo-tbccl gets in | exo-tbccl version | Bundled TBCCL | Choose it when |
|---|---|---|---|---|
| `tbccl-integration-v2` | the `tbccl` optional extra of `uv sync` (see [Default](#default-the-optional-extra-of-tbccl-integration-v2)) | **0.3.0** | 0.6.0 | you are starting now: this is the supported path |
| `tbccl-integration-v1` | a manually installed wheel (see [Legacy](#legacy-tbccl-integration-v1-and-a-manual-wheel)) | **0.3.1** | 0.6.1 | you already run the earlier integration |

These versions are intentional: the tag is immutable and 0.3.0 carries the same adapter code as 0.3.1. Do not mix the two paths' requirements, and do not "upgrade" one to the other's version. Every node must use the same integration and the same exo-tbccl version. The verification order after installing is in [how do I know TBCCL is being used?](../guides/verify-active.md): integration, adapter, matching nodes, discovery, placement, then transport.

## Requirements

| | |
|---|---|
| Python | the package metadata allows 3.13 or newer; the published wheels are `cp313` only, so CPython 3.13 is the only interpreter with a validated prebuilt wheel (anything else would need a [source build](#build-from-source), which is not validated) |
| exo | the exo fork with the TBCCL integration: `tensorsofthewall/exo-fork` at the immutable tag `tbccl-integration-v2` (commit `d1497b28b89618172d273e70e9ac7992811635f5`, upstream exo `bc412e22` plus the TBCCL integration; the earlier `tbccl-integration-v1` remains available) ([compatibility](../reference/compatibility.md)) |
| MLX | 0.32.x (`mlx-cuda13` extra on Linux with CUDA 13, `mlx` on macOS), from exo's own environment; validated with 0.32.0 (Linux CUDA) and the 0.32.4 development build `0e3ff3643` that upstream exo pins on macOS |
| Platforms | Linux x86-64 (CUDA, CPU) and macOS arm64 (Metal, CPU); wheels `manylinux_2_24`/`2_28` and `macosx_14_0_arm64` |
| TBCCL | bundled: C ABI 1, wire protocol 4. The 0.3.1 wheel bundles TBCCL 0.6.1; the 0.3.0 wheel that the `tbccl` extra installs bundles TBCCL 0.6.0 |

## Install

exo-tbccl goes into the environment of an exo that carries the integration. There are two paths; do not mix their requirements. The default is the optional extra of `tbccl-integration-v2`; the manual wheel belongs to `tbccl-integration-v1`.

### Default: the optional extra of `tbccl-integration-v2`

The immutable tag `tbccl-integration-v2` of [exo-fork](https://github.com/tensorsofthewall/exo-fork) declares `exo-tbccl` as an optional dependency (`tbccl` extra), resolved to the published 0.3.0 release wheel (bundling TBCCL 0.6.0) on purpose, because the tag is immutable and 0.3.0 has the same adapter code as the current 0.3.1, on Linux x86-64 and macOS arm64 (and doing nothing elsewhere), with the lock recording both wheels and their hashes. The wheel bundles the native TBCCL runtime: nothing else is installed or built, and exo works without the extra. The tag's own README predates this installation method; follow these commands:

```sh
git clone --depth 1 --branch tbccl-integration-v2 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
uv sync --no-dev --locked --extra mlx-cuda13 --extra tbccl     # Linux x86-64, CUDA 13
uv sync --no-dev --locked --extra mlx --extra tbccl            # macOS arm64, Metal
uv run --no-sync python -c "import exo_tbccl; print(exo_tbccl.is_available())"
```

Run it on **every** node. It installs exactly one extra package (`exo-tbccl 0.3.0`) compared with the same sync without `--extra tbccl`, keeps it across a repeated sync, and leaves the MLX and PyTorch pins unchanged. On macOS the first sync builds MLX from its pinned git revision (Xcode and the Metal toolchain are needed, and it takes several minutes); a later sync may reinstall that date-stamped MLX build, which is existing behaviour and leaves `exo-tbccl` installed. On any other platform (for example Linux aarch64 or Intel macOS) no adapter wheel exists, the extra installs nothing, and the diagnostic is `exo_tbccl is not installed (No module named 'exo_tbccl')`; selecting `MlxTbccl` fails placement with `MlxTbccl requires the optional exo-tbccl package: <that reason>`. This configuration was qualified on a Mac Metal host and a Linux CUDA host over Thunderbolt 4 with `Qwen3-0.6B-8bit` (one short generation); see [validated configurations](../project/validated-configurations.md).

### Legacy: `tbccl-integration-v1` and a manual wheel

The earlier immutable tag `tbccl-integration-v1` has no extra. The 0.3.1 wheels are attached to the GitHub release [`v0.3.1`](https://github.com/tensorsofthewall/exo-tbccl/releases/tag/v0.3.1). They are not published on PyPI or TestPyPI. The wheel does **not** need an editable install, and nobody has to build libtbccl or the adapter from source to use it. Use exo's own uv environment (`.venv` in the checkout), named explicitly in each command:

```sh
git clone --branch tbccl-integration-v1 --depth 1 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
# Linux x86-64 (CUDA 13):
uv sync --no-dev --extra mlx-cuda13
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.1/exo_tbccl-0.3.1-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
# macOS arm64 (Metal):
uv sync --no-dev --extra mlx
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.1/exo_tbccl-0.3.1-cp313-cp313-macosx_14_0_arm64.whl
```

Install it on **every** node, with the same version. `--no-deps` keeps the installer from touching exo's MLX environment. `uv sync` is an exact sync and removes the wheel again: after installing it, synchronise with `uv sync --inexact --no-dev --extra ...` (verified to keep it) or reinstall the wheel. An exo that lives somewhere else works the same way with its own `.venv/bin/python`.

## Check

```python
import exo_tbccl
print(exo_tbccl.__version__, exo_tbccl.is_available())    # 0.3.1 (True, '') for the 0.3.1 wheel installed by hand (legacy path); 0.3.0 (True, '') with the tbccl extra, which is expected
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

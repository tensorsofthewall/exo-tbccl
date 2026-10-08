#!/usr/bin/env bash
# Build a publishable exo-tbccl wheel against an installed TBCCL prefix.
#   TBCCL_ROOT=<prefix> tools/build_release_wheel.sh <out-dir>
# Linux: run inside a manylinux_2_28 environment with the CUDA 13 toolkit headers (the TBCCL CUDA archive links the static CUDA runtime, so the wheel has no
# libcudart dependency); the wheel is repaired with auditwheel and must come out as a genuine manylinux wheel.
# macOS: build on arm64 with MACOSX_DEPLOYMENT_TARGET set (the wheel tag follows it); delocate lists the dependencies, which must all be system libraries.
# Either way the wheel is then inspected (tag, licence, version, no private paths, no development files, no run-time dependency on TBCCL or CUDA).
set -euo pipefail
OUT=${1:?output directory}
: "${TBCCL_ROOT:?set TBCCL_ROOT to an installed TBCCL prefix}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
RAW=$(mktemp -d "${TMPDIR:-/tmp}/exo-tbccl-wheel.XXXXXX")
trap 'rm -rf "$RAW"' EXIT
python -m build --wheel --no-isolation -o "$RAW" "$HERE"
case "$(uname -s)" in
    Linux)
        auditwheel show "$RAW"/*.whl
        auditwheel repair --plat manylinux_2_28_x86_64 -w "$OUT" "$RAW"/*.whl ;;
    Darwin)
        : "${MACOSX_DEPLOYMENT_TARGET:?set MACOSX_DEPLOYMENT_TARGET (for example 14.0) so the wheel tag is deliberate}"
        delocate-listdeps --all "$RAW"/*.whl
        cp "$RAW"/*.whl "$OUT"/ ;;
    *) echo "unsupported platform" >&2; exit 1 ;;
esac
python "$HERE/tools/inspect_wheel.py" "$OUT"/*.whl > "$OUT/inspect.json"
echo "release wheel(s) in $OUT:"; ls "$OUT"/*.whl

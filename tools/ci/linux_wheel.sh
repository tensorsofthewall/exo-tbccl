#!/usr/bin/env bash
# Runs INSIDE the manylinux_2_28 container: tools/ci/linux_wheel.sh <out-dir>
# Expects /src/exo-tbccl (this repository) and /src/tbccl (the TBCCL core at the commit being released against). Builds the CUDA 13 core archive (its static CUDA
# runtime means the wheel has no libcudart dependency), then builds the wheel twice with the release recipe (build, auditwheel repair against manylinux_2_28,
# inspect) and compares the two; one wheel is left in <out-dir>.
set -euo pipefail
OUT=${1:?output directory}
SRC=/src
git config --global --add safe.directory '*'
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y -q cuda-nvcc-13-0-13.0.88-1 cuda-cudart-devel-13-0-13.0.96-1 cuda-driver-devel-13-0-13.0.96-1
export PATH=/usr/local/cuda-13.0/bin:/opt/python/cp313-cp313/bin:$PATH
pip install -q "build==1.6.1" "auditwheel==6.8.2" "patchelf==0.17.2.4" "scikit-build-core==0.11.6" ninja
mkdir -p "$OUT/inputs" /w
pip freeze > "$OUT/build-environment.txt"; rpm -qa 'cuda*' 'libnv*' | sort >> "$OUT/build-environment.txt"
(cd "$SRC/tbccl" && scripts/package_native.sh --cuda --out /w/core)
cp /w/core/tbccl-*.tar.gz "$OUT/inputs/"
mkdir -p /w/pfx && tar -xzf /w/core/tbccl-*.tar.gz -C /w/pfx
export TBCCL_ROOT=$(ls -d /w/pfx/tbccl-*)
for n in 1 2; do
    rm -rf "$SRC/exo-tbccl/build"
    (cd "$SRC/exo-tbccl" && tools/build_release_wheel.sh /w/wheel$n)
done
W1=$(ls /w/wheel1/*.whl); W2=$(ls /w/wheel2/*.whl)
[ "$(basename "$W1")" = "$(basename "$W2")" ] || { echo "the two builds produced different file names" >&2; exit 1; }
python "$SRC/exo-tbccl/tools/release_metadata.py" compare "$W1" "$W2" | tee "$OUT/reproducibility.txt"
cp "$W1" "$OUT/"
cp /w/wheel1/inspect.json "$OUT/inspect.json"

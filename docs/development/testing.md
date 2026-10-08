# Building and testing

Copy-pasteable commands. Run them from the repository root, using exo's virtual environment (`<exo venv>`) which holds MLX.

## Build and test

```sh
TBCCL_ROOT=<installed tbccl prefix> uv pip install --python <exo venv>/bin/python -e .
<exo venv>/bin/python -m pytest -q -p no:asyncio tests      # loopback, process-per-rank; Linux and macOS
<exo venv>/bin/python -m pytest -q -p no:asyncio tests/test_docs_consistency.py     # the docs gates: environment variables, versions, independence
<exo venv>/bin/python benchmarks/bridge_overhead.py          # loopback cost of the bridge
```

The package is installed into exo's environment; syncing exo's dependencies can uninstall it. Rebuild after editing `src/native/`.

## Wheel build and inspection

```sh
TBCCL_ROOT=<prefix> python -m build --wheel --no-isolation -o dist .
python tools/inspect_wheel.py dist/exo_tbccl-*.whl --allow-local-platform    # members, metadata, no libtbccl or libcudart dependency, no embedded machine paths
```

A publishable wheel is built in a manylinux environment (Linux) or with `MACOSX_DEPLOYMENT_TARGET` set (macOS) by `TBCCL_ROOT=<prefix> tools/build_release_wheel.sh <out-dir>`, which repairs and then inspects it; CI does this for releases. Maintainers verify published wheels with `SHA256SUMS`, the SPDX SBOM and `gh attestation verify` as described in `RELEASE.md`.

## Documentation and manifest checks

```sh
make docs              # builds docs/_build/html with warnings as errors (creates .venv-docs)
make docs-linkcheck    # checks local links and anchors
python tools/check_compatibility_manifest.py    # compatibility.json against pyproject.toml and the extension's C ABI constant
```

## Notes

- Tests run one process per rank and use `spawn`, never `fork`.
- AddressSanitizer and UndefinedBehaviorSanitizer can run only the non-MLX tests (MLX CUDA tests cannot run under ASan). Build `src/native` with the sanitizer flags, uninstall the editable install (it shadows `PYTHONPATH`), and run with the sanitizer libraries preloaded.
- Two real-model loopback ranks plus a reference rank do not fit an 8 GiB GPU at long prompts; `benchmarks/real_model_loopback.py` shows the settings (cache limit, small chunks, one reference rank).
- Two-host probes are opt-in. Read the link's PCIe error counters before and after every real-link run.

## Benchmark environment variables

These variables apply to the scripts under `benchmarks/` only; none of them is read by the package.

| Variable | Used by | Meaning |
|---|---|---|
| `EXO_TBCCL_BENCH_MODEL` | `real_model_loopback.py`, `tp_graph.py`, `tp_compute_bench.py` | Directory of the local model; default `~/models/Qwen3-0.6B-8bit`. Nothing is downloaded. |
| `EXO_TBCCL_BENCH_SYNC_RECORD` | `real_model_loopback.py`, `emu_run.sh` | Path prefix: record every evaluation and communication call with a semantic label (measurement only). |
| `EXO_TBCCL_BENCH_CADENCE` | `real_model_loopback.py` | Path prefix: record the communication cadence (measurement only). |
| `EXO_TBCCL_BENCH_LAYERS` | `real_model_loopback.py` | With the sync recorder: per-layer host (graph-build) timing. |
| `EXO_TBCCL_BENCH_ACTIVITY` | `real_model_loopback.py` | Test-only control: an external CPU-activity thread (see `activity_thread.py`). |
| `EXO_TBCCL_BENCH_QOS` | `real_model_loopback.py` | Test-only control: set a macOS QoS class on the main thread. |
| `EXO_TBCCL_BENCH_EXT_SPIN` | `emu_run.sh` | Control: a CPU-burning process outside the pipeline process. |
| `EXO_TBCCL_BENCH_DIE_AT` | `real_model_loopback.py` | Failure injection: `<rank>:prefill:<chunk index>` or `<rank>:decode:<step>`. |

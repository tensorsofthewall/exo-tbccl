# Testing

```sh
<exo venv>/bin/python -m pytest -q -p no:asyncio tests      # loopback, process-per-rank; Linux and macOS
<exo venv>/bin/python benchmarks/bridge_overhead.py          # loopback cost of the bridge
```

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

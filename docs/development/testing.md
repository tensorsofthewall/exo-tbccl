# Testing

```sh
<exo venv>/bin/python -m pytest -q -p no:asyncio tests      # loopback, process-per-rank; Linux and macOS
<exo venv>/bin/python benchmarks/bridge_overhead.py          # loopback cost of the bridge
```

- Tests run one process per rank and use `spawn`, never `fork`.
- AddressSanitizer and UndefinedBehaviorSanitizer can run only the non-MLX tests (MLX CUDA tests cannot run under ASan). Build `src/native` with the sanitizer flags, uninstall the editable install (it shadows `PYTHONPATH`), and run with the sanitizer libraries preloaded.
- Two real-model loopback ranks plus a reference rank do not fit an 8 GiB GPU at long prompts; `benchmarks/real_model_loopback.py` shows the settings (cache limit, small chunks, one reference rank).
- Two-host probes are opt-in. Read the link's PCIe error counters before and after every real-link run.

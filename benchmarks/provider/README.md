# Exo benchmark provider

Benchmark workloads for real Exo running a two-node pipeline with `MlxRing` or `MlxTbccl`: `client.py` (a standard-library lifecycle and streaming client: placement and backend verification, load timing, latency, throughput, correctness and cancellation phases), `sampler.py` (a passive per-host resource sampler), `metrics.py` and `validation.py`, a trace hook, and `provider.py` (the entry point of the TBCCL benchmark provider protocol 1). `stream_metrics.py` and `process_sampler.py` are this provider's own copies of the framework-neutral stream metrics and process sampler, so it depends on no other adapter.

```sh
python benchmarks/provider/provider.py describe                       # what this environment can run (no Exo or MLX import)
python benchmarks/provider/client.py --spec spec.json --out-dir out   # measure a cluster you started
python -m unittest discover -s benchmarks/provider/tests -p 'test_*.py'
```

Documentation: [docs/benchmarks](../../docs/benchmarks/index.md). The older diagnostic tools in `benchmarks/` (for example `real_exo_e2e.py`, `distributed_timeline.py`) are unchanged and are not the supported benchmark path.

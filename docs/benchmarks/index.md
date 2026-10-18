# Benchmarks

exo-tbccl ships its Exo benchmark workloads in `benchmarks/provider/`. They measure real Exo (cluster discovery, placement, runners, HTTP API and streaming) running a two-node pipeline on **your** hardware, with either Exo's own `MlxRing` backend or `MlxTbccl` (exo-tbccl over TBCCL): placement and backend verification, time to first token, time per output token, end-to-end latency, throughput under concurrency, CPU cost, and correctness against a single-device reference.

```{toctree}
:maxdepth: 1

getting-started
ring-vs-tbccl
workloads
results
experimental
methodology
```

| I want to... | Read |
|---|---|
| see reference measurements with charts | [Results](results.md) |
| compare `MlxRing` with `MlxTbccl` correctly | [MlxRing versus MlxTbccl](ring-vs-tbccl.md) |
| understand what a number does and does not mean | [Methodology and limitations](methodology.md) |
| check my environment and run a first measurement | [Getting started](getting-started.md) |
| know what each workload and metric means | [Inference workloads](workloads.md) |
| understand the optional, exploratory settings | [Experimental settings](experimental.md) |
| generate a report from my run, or compare two runs | the TBCCL runner: [report generation](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/reports.html) and [comparing runs](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/comparing.html) |
| find out why a run failed | [Getting started](getting-started.md) for the client, [runner troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/troubleshooting.html) for the runner |

Two ways to run the same workloads:

* **Standalone.** Clone this repository, install Exo with exo-tbccl as described in the [two-host guide](../guides/two-host.md), start the nodes yourself, and point `benchmarks/provider/client.py` at the API. The client uses only the Python standard library. Nothing else needs to be installed: no torch-tbccl, no vllm-tbccl, no TBCCL benchmark suite.
* **Through the TBCCL benchmark runner.** The [TBCCL benchmark runner](https://tbccl.tensorsofthewall.com/en/latest/development/benchmark-suite.html) starts a node on each host with an isolated state directory, a sampler and the client, supervises and cleans up every process, checks the data path and health, stores results in one schema and compares runs. exo-tbccl is a *provider* for that runner.

The default benchmarks use the released, default configuration. Experimental settings are separate, explicitly enabled, and labelled wherever they appear.

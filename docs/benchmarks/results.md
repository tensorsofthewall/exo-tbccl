# Results

These are reference measurements from one pair of machines. They show what the benchmarks produce and how large the effects were there; they are not a statement about other hardware, other link types, other models or other Exo versions. Run the benchmarks on your own systems for numbers that apply to them.

## Performance at a glance

Default configuration only; the one experimental result is shown separately [below](#experimental-metal-activity-step-exploratory) and is not part of this summary.

| What was measured | Result on this pair of machines |
|---|---|
| Time per output token, Mac-first 19/9 | MlxRing 10.22 ms, default MlxTbccl 15.54 ms (6 / 9 lifecycles): classified *regression* |
| Time per output token, Linux-first 9/19, interleaved controls | MlxRing 11.44 ms, default MlxTbccl 14.44 ms (4 / 4 lifecycles): classified *regression* |
| Output tokens per second, concurrency 1 to 8 | MlxTbccl below MlxRing in the Linux-first order, 19% at concurrency 1 and 4.4% at concurrency 8; exploratory (one and two lifecycles) |
| 27B 4-bit model, short prompt | 127.1 ms (MlxRing) against 138.0 ms (MlxTbccl): a limited demonstration, not a scaling result |

**The default MlxTbccl integration was slower than MlxRing in the configurations measured here.** That is a comparison of two integrations of the same pipeline, not of a transport. These numbers come from one pair of machines joined by one Thunderbolt 4 cable. Each chart below is drawn from the table next to it; the table is the record and the chart only makes its shape easier to see. The charts need no scripts and follow the light and dark theme.

## What was measured

| | Linux host | macOS host |
|---|---|---|
| machine | Intel Core i7-12800HX laptop, NVIDIA RTX 3070 Ti Laptop GPU (8 GiB), CUDA 13, EndeavourOS, Linux 6.18 LTS | Apple M4 Mac mini, 16 GiB unified memory, macOS 26.5.1 |
| link | one direct Thunderbolt 4 cable, IP over Thunderbolt, MTU 9000, TCP/IPv4 | |
| software | Exo 0.3.70 (maintained fork, tag `tbccl-integration-v2`), exo-tbccl 0.3.0, TBCCL 0.6.0 (C ABI 1, wire protocol 4), MLX 0.32.0 (CUDA 13), mlx-lm 0.31.3 | the same Exo commit, MLX 0.32.4.dev |
| models | Qwen3-0.6B-8bit (28 layers) and, for the large-model run, Qwen3.8-27B-4bit; file hashes identical on both machines | |

**Hardware health.** The Linux host was booted with `pci=noaer`, so PCIe Advanced Error Reporting was disabled for the whole campaign. Do not read these results as "zero PCIe errors". The statement that holds is: *AER unavailable; no detected link-state or interface-health regression under the available checks.*

A stage order is written *Mac-first 19/9* (Mac runs stage 0 with 19 layers, the Linux host stage 1 with 9) or *Linux-first 9/19*. Exo chose the order at startup; each row compares only runs with the same order and split. "6 / 9" means six and nine independent lifecycles. Prompt of about 128 tokens and 32 generated tokens unless stated.

## MlxRing against default MlxTbccl

| Comparison | MlxRing | MlxTbccl (default) | Change | Exact test p | Lifecycles | Class |
|---|---|---|---|---|---|---|
| Mac-first 19/9, TPOT | 10.22 ms | 15.54 ms | +52.0% | 0.0004 | 6 / 9 | regression |
| Mac-first 19/9, end-to-end | 478.7 ms | 643.2 ms | +34.4% | 0.0004 | 6 / 9 | regression |
| Linux-first 9/19, TPOT, interleaved controls | 11.44 ms | 14.44 ms | +26.3% | 0.0286 | 4 / 4 | regression |
| Linux-first 9/19, TPOT, earlier sessions | 11.70 ms | 12.82 ms | | | 4 / 1 | insufficient evidence |

"Regression" is the comparison tool's label for a statistically separated increase over the baseline. Per-lifecycle TPOT in the Mac-first comparison: MlxRing 10.38, 9.76, 10.09, 10.49, 10.17, 10.28 ms; MlxTbccl 15.54, 13.35, 15.73, 15.71, 15.58, 15.53, 15.59, 15.54, 15.42 ms. The MlxTbccl lifecycles are not all alike.

**How to read this.** This is an application-level difference between two integrations of the same pipeline (see [MlxRing versus MlxTbccl](ring-vs-tbccl.md)), not the cost of a transport protocol. The same data moved over the link per token in both (about 10 kB); the difference is in how the steps wait and synchronise. The fourth row has a single MlxTbccl lifecycle and shows why lifecycle counts matter: it supports no conclusion.

```{raw} html
:file: charts/ring-vs-tbccl.html
```

The chart keeps the lifecycle counts and the class of each comparison in its labels. The first two comparisons are separated statistically; the third has one MlxTbccl lifecycle and carries no conclusion, and it is shown for that reason. The bars are an application-level comparison of the two backends and say nothing about where the extra time of a MlxTbccl step goes.

## Experimental: Metal activity-step (exploratory)

| Linux-first 9/19, TPOT | default MlxTbccl | activity-step | Change | Lifecycles | Class |
|---|---|---|---|---|---|
| descriptive only | 14.44 ms | 9.48 ms | -34.4% | 4 / 2 | insufficient evidence |

Per-lifecycle TPOT: activity-step 9.46 and 9.50 ms; default 14.39, 14.43, 14.46, 14.47 ms; MlxRing 11.34, 11.43, 11.45, 12.06 ms. The two experimental lifecycles are completely separated from the four controls, but two lifecycles cannot reach p ≤ 0.05 in an exact test. This is **an exploratory observation, not the default result and not a validated optimisation**; it costs about one busy Mac core. See [Experimental settings](experimental.md).

```{raw} html
:file: charts/activity-step-experimental.html
```

The chart is separate from the default comparison on purpose and says so in its title. The two experimental lifecycles are fewer than an exact test needs, so no improvement is confirmed.

## Concurrency

Closed loop, output tokens per second (passes pooled per backend and stratum; passes inside a lifecycle are not independent):

| Stage order, backend | lifecycles / passes | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|---|
| Linux-first 9/19, MlxRing | 1 / 3 | 57.4 | 90.7 | 124.8 | 141.1 |
| Mac-first 19/9, MlxRing | 1 / 3 | 63.3 | 92.6 | 122.5 | 137.6 |
| Linux-first 9/19, MlxTbccl | 2 / 6 | 46.3 | 76.9 | 112.9 | 134.9 |

At concurrency 1 default MlxTbccl was 19% below MlxRing in the Linux-first order and 4.4% below at concurrency 8: the per-step cost is amortised by batching. This is **exploratory**: one MlxRing lifecycle and two MlxTbccl lifecycles, and the first pass of a lifecycle was often the highest.

```{raw} html
:file: charts/concurrency-throughput.html
```

The lines are descriptive. They come from one or two lifecycles per line and from passes that are not independent of each other, so they show a trend, not a statistically established difference.

## CPU cost per generated token

Mac-first 19/9, CPU seconds of the Exo process tree per generated token, median over lifecycles:

| Backend | Linux host | Mac |
|---|---|---|
| MlxRing | 0.0309 (n = 6) | 0.0106 (n = 6) |
| MlxTbccl | 0.0272 (n = 9) | 0.0147 (n = 9) |

MlxTbccl used less CPU on the Linux host (it blocks while waiting) and more on the Mac. This is a correlation across runs; it does not by itself explain the latency difference.

## Large model, short context

Qwen3.8-27B-4bit (Mac 43 layers, Linux 21), a short prompt:

| | TPOT |
|---|---|
| MlxRing | 127.1 ms and 127.0 ms (two lifecycles, one of them the reference run) |
| MlxTbccl, Mac-first | 138.0 ms (one lifecycle); 137.8 ms Linux-first |

MlxTbccl was 8.6% above MlxRing in this demonstration. **Exploratory**: two and two lifecycles, one Mac-first MlxTbccl run. A prompt of 138 tokens ran out of CUDA memory in prefill on the 8 GiB GPU in this benchmark run; longer prompts are not demonstrated. (The earlier validation of the 27B pipeline, with exo's memory override, recorded a 138-token prefill that completed: see [validated configurations](../project/validated-configurations.md). The two runs were not compared setting for setting, so read neither as a general memory limit.) A 27B model at a short prompt and a 0.6B model at a long prompt are not a model-size scaling comparison.

## Layer split and tracing

One lifecycle per split and order gave TPOT between 13.28 and 15.65 ms across all tried splits and both orders: no dependence on the split was visible, which suggests, but does not show, a fixed per-step cost for this small model. A diagnostic trace added 5% and 6% to the default MlxTbccl TPOT (one traced lifecycle each); traced runs are never pooled with untraced ones.

## What was not measured

Aligned per-stage timing inside the full Exo server was not available: the diagnostic hook records that a communication call happened, not how long it took, and clocks of different machines are never subtracted. Where the extra time of a MlxTbccl step goes is therefore not measured.

## Limitations of these results

* One machine pair, one cable, one Exo version, two models. Nothing applies to other Thunderbolt generations, GPUs, operating systems or models.
* Four to nine lifecycles in the main comparisons, one or two in the exploratory ones.
* The Metal activity-step result and the 27B result are exploratory.
* AER was unavailable (see above).

## Reproducing and reading these results

* What a lifecycle is, what may be compared and how classifications are assigned: [Methodology and limitations](methodology.md) and [MlxRing versus MlxTbccl](ring-vs-tbccl.md).
* What each workload and metric means: [Inference workloads](workloads.md).
* The optional, off-by-default settings behind the experimental chart: [Experimental settings](experimental.md).
* To run the workloads yourself, start with [Getting started](getting-started.md).

## How the charts are made

The chart inputs are in `docs/benchmarks/chart-data.json`, copied by hand from the tables on this page. They are published aggregates, not raw records, and carry the exact text of each cell. `python3 tools/benchmark_charts.py` regenerates the figures in `docs/benchmarks/charts/` deterministically and `--check` fails when they are out of date; a test fails when a chart input no longer appears in the table it cites.

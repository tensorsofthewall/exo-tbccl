# Externally written storage -> first GPU consumer (Metal and CUDA)

Question (Phase 57, part G/H): does MLX storage that was written by the CPU / TBCCL, not by a GPU kernel, make the **first GPU computation that reads it** slower or unsafe? Tool: `benchmarks/recv_visibility.py` (two ranks over loopback; data in `docs/data/phase57/recv_visibility_{mac,linux}.txt`).

## Method

Rank 1 sends a 1x1024 bfloat16 payload whose every element depends on the iteration (`((7*i + j) % 253)`); rank 0 receives it and immediately runs a GPU computation consuming it, `y = x.astype(float32) * 2 + 1`, and compares `y` with the exact expected array **every iteration** (stale or half-written data cannot pass). Timed from the moment the receive returned: host graph build, then `mx.eval(y)`. Five variants, three interleaved rounds of 1000 iterations each (3000 per variant per machine), alternating order:

| variant | what it does |
|---|---|
| `fresh+eval` | current path: `recv_like` (fresh `mx.zeros` destination) then exo's post-receive `mx.eval(x)`, then the consumer |
| `fresh` | the same without the post-receive `mx.eval` (the consumer's own eval does the work) |
| `pool` | receive-buffer reuse (`EXO_TBCCL_RECV=reuse`) then the consumer |
| `direct` | **no TBCCL**: a host `memmove` into a fresh evaluated MLX array's storage (borrowed writable), then the consumer |
| `gpu` | control: the same data written by a GPU computation (`mx.full`-style) and evaluated, then the consumer |

## Results (microseconds, median; every one of the 3000 results per variant exact on both machines)

| variant | Mac Metal consumer (p25..p75) | Mac since-recv | Linux CUDA consumer (p25..p75) | Linux since-recv |
|---|---:|---:|---:|---:|
| fresh+eval | 162.6 (150.0..174.3) | 164.9 | 167.4 (30.5..179.4) | 173.8 |
| fresh | 163.0 (147.0..177.4) | 165.2 | 31.4 (30.2..171.7) | 33.5 |
| pool | 166.0 (156.2..182.8) | 168.1 | 31.3 (30.5..162.6) | 33.5 |
| direct (no TBCCL) | 153.8 (143.2..168.0) | 155.5 | 31.2 (29.9..131.8) | 33.4 |
| gpu (control) | 159.8 (154.0..171.2) | 162.0 | 28.5 (28.0..29.0) | 30.3 |

* **Metal: no external-write penalty.** All five variants, including the GPU-written control, sit at 154-166 us for this tiny consumer: that is the cost of a Metal command-buffer round trip, not of visibility. TBCCL-written (`fresh`/`pool`) and CPU-memmove-written (`direct`) storage behave the same as GPU-written storage. Unified memory needs no flush: nothing in MLX or in exo-tbccl is required to make the write visible beyond the existing evaluate-before-borrow, receive-before-use ordering.
* **CUDA: a bimodal tail for externally written storage.** The median consumer is 31 us against 28.5 us for GPU-written data (+3 us), but the externally written variants have a second mode near 130-180 us in roughly a quarter to a half of iterations (p75 130-180 us vs a flat 29.0 for the control). The CUDA path allocates managed memory, so host writes leave pages resident on the host and the next GPU read can migrate them. The `fresh+eval` row's median is in the slow mode (167 us); the other external variants' medians are in the fast mode, so which mode wins an iteration is noise-sensitive. Correctness is unaffected (0 wrong results in 9000).
* The first-consumer effect is therefore a **CUDA** phenomenon (tens to ~150 us, intermittent), not a Metal one. The plan's hypothesis that the Metal side pays a large first-consumer cost for externally written shared memory is **refuted** by this measurement.

## What this does and does not show

It shows the per-consumer cost and correctness for a 2 KiB activation under loopback with a GPU that is otherwise idle. It does not show the cost under a busy GPU queue (a real stage: ~5 ms of kernels) or across the TB4 link (where the receive arrives later but the write-then-read ordering is the same). The correctness conclusion (no stale reads in 18,000 iterations) is the part that carries over.

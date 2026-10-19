# Decode timeline diagnostics

An opt-in diagnostic that records, for every decode step on every pipeline rank, where the time goes: waiting for the upstream stage, building the graph, stage compute, sending, the all-gather, and the work between steps. It works for `MlxRing` and `MlxTbccl` with the same definitions, so the two can be compared interval by interval. It is a diagnostic, not a benchmark: use it to explain a difference, never to produce a headline number.

**It is off by default.** Nothing is wrapped and nothing is written unless both `TBCCL_B6_TRACE_DIR` (the output directory) and `TBCCL_TRACE_TIMELINE=1` are set in the environment of the Exo node. The code lives under `benchmarks/provider/`; the `exo_tbccl` package does not import it.

## Turning it on

The hook is a `sitecustomize` on `PYTHONPATH` of the Exo node (the runner processes inherit it), exactly like the existing trace hook:

```sh
PYTHONPATH=benchmarks/provider/trace_hook \
TBCCL_B6_TRACE_DIR=/path/for/traces \
TBCCL_TRACE_TIMELINE=1 \
exo ...
```

With the benchmark suite, set `trace: true` on the Exo workload and override the per-operation logging, which would put file writes in the decode path:

```yaml
options:
  trace: true
  env:
    EXO_TBCCL_TRACE: "0"
    TBCCL_TRACE_TIMELINE: "1"
```

Run traced and untraced lifecycles as separate, interleaved arms and report the difference; never pool them.

## What is recorded

Each wrapped call appends one integer (monotonic nanoseconds and an event code) to an in-memory array; a background thread appends it to `timeline-<pid>.bin` about twice a second and writes `timeline-<pid>.json` (provenance). There is no formatting, locking or file I/O in the decode path. Process CPU time and context-switch counters are read once per step.

| Boundary | Events |
|---|---|
| first pipeline layer | `FL_ENTER`, `FL_EXIT`; `recv_like`: `RECV_ENTER`, `RECV_EXIT` |
| last pipeline layer | `LL_ENTER`, `STEP_COMPLETE`, `send`: `SEND_ENTER`, `SEND_EXIT`; `all_gather`: `GATHER_ENTER`, `GATHER_EXIT`; `LL_EXIT` (`LL_EXIT_PREFILL` for prefill passes) |
| `MlxTbccl` internals | destination allocation, borrow/export, submit, blocking wait, drain, reap (enter and exit) |

## Reading it

```sh
python benchmarks/provider/timeline_report.py <trace-dir> [<trace-dir> ...] \
    [--json report.json] [--markdown report.md] [--require-complete]
```

One trace directory per host and lifecycle. Intervals (microseconds) are medians over the steady decode steps of the most common request length; `period`, `post`, `build`, `compute`, `gather_call`, `other` and the others are defined in the docstring of `timeline_report.py`. `other` is the residual of a measured partition, not an estimate. A small synthetic example:

```text
example-host pid 100 rank 1 backend tbccl [COMPLETE]: 3 requests of 24 steps, 69 steady decode steps
  period       median        7.0 us
  post         median        1.0 us
  recv_call    median        0.3 us
  build        median        0.2 us
  compute      median        5.0 us
  gather_call  median        0.3 us
  other        median        0.0 us
```

The JSON report (`tbccl-decode-timeline-report/1`) carries, per process, its host, rank, backend, status, accounting of excluded passes, the tracing configuration (`TBCCL_TRACE_TIMELINE`, every `EXO_TBCCL_*` variable) and software versions (`exo-tbccl`, `mlx`, `exo`), so a result always says how it was produced.

## What it does not tell you

* Timestamps come from each process's own monotonic clock. **No interval is ever computed between two machines**, and the report keeps ranks and hosts separate. A wait on one rank contains the peer's work; the two ranks' intervals are not additive.
* Between `MlxRing` and `MlxTbccl` the *boundaries* match but the work inside them differs: `MlxRing` performs a transfer inside the next `mx.eval`, `MlxTbccl` inside the communicator call.
* GPU idle time is not measured; `compute` is wall time between two CPU-side events.
* Tracing perturbs the run slightly. Measure it on your hardware by interleaving traced and untraced arms.

## Partial, failed and cancelled runs

A process is `complete` only if its recorder closed cleanly, wrote without errors and did not reach its event cap (`TBCCL_TRACE_TIMELINE_MAX_EVENTS`, default 8,000,000 events, 64 MiB). A killed, crashed or still-running process, a truncated trace, trailing partial writes, unknown event codes and a pass left unfinished at the end are reported as `partial` with the reason, and `--require-complete` exits with status 2. Passes with missing, unbalanced or out-of-order events are counted, excluded and never repaired. Streams that were cancelled early have a different length from the steady requests and are not part of the statistics (`request_lengths` shows them).

## Tests

`benchmarks/provider/tests/test_exo_timeline.py` runs offline against stand-in `exo` and `exo_tbccl` packages: default-off, activation, exact durations, missing and malformed events, unfinished and killed processes, rank attribution, deterministic output, several lifecycles, the event cap and a record-then-report round trip. No GPU, second machine or model is needed.

# Inference workloads

## Phases of the client

`client.py` places one instance with Exo's normal placement, waits for the model to load, and then runs an ordered list of phases, each writing the raw per-request event timeline.

| Phase kind | What it does |
|---|---|
| `warmup` | untimed sequential requests (kept, flagged) |
| `rounds` | closed loop in rounds of `conc` simultaneous requests |
| `continuous` | closed loop with `conc` workers issuing their next request when the previous one ends, for `requests` requests in total |
| `reference` | greedy generations stored with the top-1/top-2 log-probability margin of every position (run on a single-device instance) |
| `check` | requests compared with a stored reference (exact, or a divergence at a near tie) |
| `cancel` | streams dropped after a few tokens, then a follow-up request compared with the reference (Exo cancels on disconnect) |
| `recreate` | deletes the instance, places it again and checks the new placement |

Model loading and instance deletion are timed on their own and are never part of a steady-state decode figure.

## Metrics

All are computed from the raw event timelines.

* **TTFT**: first token event minus request submission. **E2E**: last event minus submission.
* **TPOT**: (last event minus first event) divided by the tokens that followed the first. Undefined, never zero, for a single-token reply.
* **ITL**: intervals between successive events.
* **Throughput**: the tokens the server reported over the phase's wall time.
* **CPU per generated token**: CPU seconds of the Exo process tree on each host (from a passive sampler) divided by generated tokens. It is read together with the latency, not instead of it.
* Prefill time is never inferred from TTFT, because the stream carries no separate prefill event.

Every duration uses one machine's monotonic clock. Timestamps of different machines are never subtracted.

## The workloads the runner defines

| Runner benchmark | What it measures |
|---|---|
| `exo_correctness` | greedy token streams against a single-device reference, repeated requests, instance recreation and cancellation |
| `exo_latency` | one request at a time: TTFT, TPOT, ITL and E2E |
| `exo_throughput` | output-token and request throughput at a fixed closed-loop concurrency |
| `exo_transport_ab` | `MlxRing` against `MlxTbccl` with the same model, placement and workload, interleaved |
| `exo_stage_profile` | per-stage diagnostic timing and resource cost (traced runs are never mixed with throughput runs) |
| `exo_large_model` | a model that needs the combined device memory of both machines |
| `exo_lifecycle` | cluster start, placement, model load, first request, instance deletion and shutdown |

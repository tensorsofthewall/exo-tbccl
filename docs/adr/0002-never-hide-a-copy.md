# ADR 0002: Never hide a copy

- Status: accepted
- Date: 2026-10-04

## Context

Activations are moved between GPUs and hosts on every decode token, so an unnoticed payload-sized copy directly costs latency, and lifetimes of borrowed MLX arrays must be exact because TBCCL buffers are non-owning.

## Decision

Any payload-sized copy the adapter makes is counted in `CopyStats` and visible in trace mode; the zero-copy paths (Metal and CUDA) must report zero copies. A borrowed MLX array, its view and its DLPack export stay alive until the TBCCL `Work` is terminal; no correctness depends on `__del__` or interpreter shutdown, and `close()` is explicit.

## Consequences

- Tests assert zero adapter copies on the zero-copy paths.
- Optional optimizations that change lifetimes (receive reuse, asynchronous sends) release resources only at explicit boundaries (`step_complete()`, terminal `Work`) and ship with negative controls.

# Receive-buffer reuse (Phase 54)

`recv_like` used to allocate and evaluate a fresh MLX destination (`mx.zeros` + `mx.eval`) for every receive: ~130-165 us on Metal, ~170-280 us on CUDA, plus, on CUDA,
MLX's first-export `move_to_unified_memory` copy for the fresh array. `EXO_TBCCL_RECV=reuse` (`FastPathConfig.recv_reuse`) serves destinations from a small pool instead. Default: `fresh`.

## The lifetime audit (why reuse can be safe)

Where the activation lives, in exo (`auto_parallel.py`):

```
PipelineFirstLayer (rank > 0):  mx.eval(x); x = comm.recv_like(x, r-1); mx.eval(x)
  -> this shard's layers (lazy MLX graph rooted at x)
PipelineLastLayer:              output = layers(x); mx.eval(output); [step_complete]; send / queue; (decode) all_gather
```

- `mx.eval(output)` is a blocking evaluation of the whole graph reachable from `x`; when it returns, every kernel that reads `x` has finished. exo only ever builds the
  next step's graph after it has the sampled token, so no later graph reads the old `x` by design.
- No cache aliases `x`: an attention cache stores arrays *computed* from `x` (key/value projections), not `x`. The one way an array can alias `x` is an identity/view stage
  (a shard whose output is `x` or a view of it); the pool pins any slot whose storage overlaps an outgoing send (`pin_aliased`) so it is never recycled.
- `PipelineComm.step_complete()` (new, called right after that `mx.eval`) is the explicit boundary; `MlxPipelineComm` implements it as a no-op. Nothing relies on Python
  reference counts or `__del__`.
- Not audited: model families other than Qwen3/KVCache and the synthetic model (hybrid/SSM states, MLA `CacheList`, rotating caches). Reuse is therefore opt-in; the poison
  control below is the tool to audit another family.

## Negative control (mutation test)

`EXO_TBCCL_RECV_POISON=0xA5` overwrites a slot's bytes with a recognizable pattern at release. Evidence:

- `tests/test_fastpath.py::test_poison_control_detects_a_premature_release_boundary`: releasing *before* the consumer is evaluated changes the output (the control is sensitive).
- `::test_leased_buffer_is_never_overwritten_before_step_complete`: a second receive without `step_complete` never reuses the leased buffer.
- Synthetic pipeline through exo's real path (N=2/3/4, float32/bfloat16, prefill chunks + decode, poison on): outputs and every stage-boundary hash bit-exact.
- Real Qwen3-0.6B-8bit (`benchmarks/real_model_loopback.py`, split 21/7 and 7/21, prompts of 49, 577 and ~7k tokens, poison on): token ids identical to the unsharded greedy
  reference, and the KV cache prefix snapshot taken mid-run is byte-identical at the end (earlier KV entries unchanged by later poisoning). Cross-run KV digests are *not*
  comparable (CUDA kernels are not bit-deterministic run to run); the within-run prefix check is.

## Pool design

Keyed by `(dtype, shape)`, never byte count. Slot states: FREE -> RECEIVING (a TBCCL recv is writing it) -> LEASED (handed to MLX) -> FREE at `step_complete`. A failed receive
discards the slot. Bounds: 2 FREE slots per key, 64 MiB total, at most 4 LEASED/RECEIVING (beyond that a slot is allocated untracked and freed by MLX). Variable
shapes (decode, each prefill chunk, a different prompt length) simply use different keys; prefill-size slots are evicted LRU when the byte bound is reached.
`ReceivePool.stats` reports hits, misses, evictions, untracked, peak bytes/slots.

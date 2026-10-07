# Optional fast paths

All fast paths are **off by default**: the defaults are exactly the behavior without them. `FastPathConfig` (or the environment, read when a communicator is created) selects them.

## Receive-buffer reuse (`EXO_TBCCL_RECV=reuse`)

By default `recv_like` allocates and evaluates a fresh MLX destination for every receive (about 130 to 165 microseconds on Metal, 170 to 280 on CUDA, plus an MLX-internal unified-memory copy on first export on CUDA). With reuse, destinations come from a small pool and are released only at exo's `step_complete()` boundary, never by reference counting.

Why this can be safe: when `mx.eval(output)` returns, every kernel that reads the received activation has finished, exo builds the next step's graph only after it has the sampled token, and attention caches store arrays *computed from* the activation, not the activation itself. The pool also pins any slot whose storage overlaps an outgoing send. Reuse is **audited only for Qwen3 with a standard KV cache and the synthetic model**; audit another model family first. `EXO_TBCCL_RECV_POISON=0xA5` overwrites a slot's bytes at release with a recognizable pattern as a negative control: tests show that releasing before the consumer is evaluated changes the output, that a second receive never reuses a leased buffer without `step_complete`, and that a poisoned synthetic pipeline and a real Qwen3-0.6B-8bit run stay bit-exact.

The pool is keyed by (dtype, shape), holds at most two free slots per key, 64 MiB in total and four leased or receiving slots (beyond that a slot is allocated untracked). Measured effect: about 60% less bridge round-trip time on Metal, neutral end to end.

## Asynchronous decode sends (`EXO_TBCCL_ASYNC_SEND=1`)

A decode send submits, keeps the `Work` and its borrows in the pending table and returns at once, instead of waiting. Every communication entry point first reaps finished transfers; a failed send is never dropped, its error is raised at the next communication point, and every collective drains outstanding sends, so a send error surfaces no later than the next synchronizing pipeline point. Submission order per rank is unchanged, so FIFO and collective order are unchanged; an intentionally invalid send/all_gather order is detected as a protocol mismatch by TBCCL, not silently corrupted. Measured effect: neutral end to end.

## CUDA managed memory (`EXO_TBCCL_CUDA_MANAGED_MODE`)

MLX's CUDA backend reports `kDLCUDAManaged`. The default `cuda` describes it to TBCCL as CUDA memory, which goes through TBCCL's CUDA provider (stream, staging, event completion); `host` describes managed storage as host memory (measurement only); `auto` does so only when the array is managed, the driver-reported attributes match a proven signature, and the payload is at most `EXO_TBCCL_MANAGED_MAX_BYTES` (default 16384). `EXO_TBCCL_MANAGED_DIRS=send|recv|send,recv` restricts the directions. The device type reported by the bridge never changes, and the export's device is re-checked on every borrow.

Host mapping wins on loopback at decode sizes but **loses on the real Thunderbolt link** (a 2 KiB ping-pong round trip was 18% to 70% slower), because a CPU read of GPU-resident managed pages migrates them. **Leave the default at `cuda`.**

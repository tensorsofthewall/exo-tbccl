# Environment variables

exo-tbccl needs no configuration: every variable below is an optional benchmarking or debugging override, read when a communicator is created. Settings can also be given programmatically through `FastPathConfig`. The variables are per process (set them on each node that should use them). exo's own settings (`EXO_ZENOH_NAMESPACE`, `EXO_HOME`, `EXO_OFFLINE`, `OVERRIDE_MEMORY_MB`) are documented in exo and in [running across two hosts](../guides/two-host.md).

| Variable | Values (default first) | Effect |
|---|---|---|
| `EXO_TBCCL_TRACE` | `0`, `1` | logs the path of every operation (host, CUDA-direct, Metal-direct) |
| `EXO_TBCCL_CUDA_MANAGED_MODE` | `cuda`, `host`, `auto` | how a `kDLCUDAManaged` export is described to TBCCL ([fast paths](../concepts/fast-paths.md)) |
| `EXO_TBCCL_MANAGED_DIRS` | `send,recv`, `send`, `recv` | which directions the managed mapping applies to |
| `EXO_TBCCL_MANAGED_MAX_BYTES` | 16384 | largest payload `auto` maps to host memory (`host` ignores it) |
| `EXO_TBCCL_RECV` | `fresh`, `reuse` | receive destination policy |
| `EXO_TBCCL_RECV_POISON` | unset, e.g. `0xA5` | overwrites released pool slots with a pattern (negative control) |
| `EXO_TBCCL_ASYNC_SEND` | `0`, `1` | decode sends are not waited for immediately |
| `EXO_TBCCL_ALLOC_STREAM` | `gpu` (default), `cpu` | stream that allocates fresh receive and all_gather destinations. On Metal, `cpu` also evaluates the metadata-only borrow views on the CPU stream, avoiding unnecessary GPU-stream queueing for those views; producer evaluation and required synchronization may still wait for GPU work, so not every GPU wait is removed. Experimental; Metal only, ignored on CUDA, default behavior unchanged |
| `EXO_TBCCL_WAIT_SPIN_MS` | `0` (block) or milliseconds | poll a pending `Work` before blocking (Metal only) |
| `EXO_TBCCL_PREALLOC_GATHER` | `0` (default), `1` | while a posted receive is in flight, prepare the next `all_gather` destination so its allocation and evaluation leave the critical path of the stage that waits for its input. The destination is the same fresh zeroed array as without the setting. Experimental; Metal only, ignored on CUDA and CPU. On the two-host Qwen3-0.6B pipeline it lowered the per-token time by about 0.3 ms (about 3%) when the Mac is the receiving stage and made no difference in the other orientation; it is not expected to matter for large models |
| `EXO_TBCCL_ACTIVITY_MODE`, `EXO_TBCCL_STEP_ACTIVITY` | `off`, `step`; alias `1` | Metal activity policy ([experimental](../concepts/mac-activity.md)) |
| `EXO_TBCCL_ACTIVITY_DUTY` | 1.0 (0 < d <= 1) | fraction of each millisecond the helper burns inside an open window |
| `EXO_TBCCL_ACTIVITY_MAX_MS` | 100 | hard bound of one activity window |

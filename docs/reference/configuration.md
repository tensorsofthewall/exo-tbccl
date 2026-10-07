# Configuration

Read when a communicator is created (benchmarking and debugging overrides; nothing is required). Settings can also be given programmatically through `FastPathConfig`.

| Variable | Values (default first) | Effect |
|---|---|---|
| `EXO_TBCCL_TRACE` | `0`, `1` | logs the path of every operation (host, CUDA-direct, Metal-direct) |
| `EXO_TBCCL_CUDA_MANAGED_MODE` | `cuda`, `host`, `auto` | how a `kDLCUDAManaged` export is described to TBCCL ([fast paths](../concepts/fast-paths.md)) |
| `EXO_TBCCL_MANAGED_DIRS` | `send,recv`, `send`, `recv` | which directions the managed mapping applies to |
| `EXO_TBCCL_MANAGED_MAX_BYTES` | 16384 | largest payload `auto` maps to host memory (`host` ignores it) |
| `EXO_TBCCL_RECV` | `fresh`, `reuse` | receive destination policy |
| `EXO_TBCCL_RECV_POISON` | unset, e.g. `0xA5` | overwrites released pool slots with a pattern (negative control) |
| `EXO_TBCCL_ASYNC_SEND` | `0`, `1` | decode sends are not waited for immediately |
| `EXO_TBCCL_ALLOC_STREAM` | `gpu`, `cpu` | stream that allocates fresh receive and all_gather destinations (experiment) |
| `EXO_TBCCL_WAIT_SPIN_MS` | `0` (block) or milliseconds | poll a pending `Work` before blocking (Metal only) |
| `EXO_TBCCL_ACTIVITY_MODE`, `EXO_TBCCL_STEP_ACTIVITY` | `off`, `step`; alias `1` | Metal activity policy ([experimental](../concepts/mac-activity.md)) |
| `EXO_TBCCL_ACTIVITY_DUTY` | 1.0 (0 < d <= 1) | fraction of each millisecond the helper burns inside an open window |
| `EXO_TBCCL_ACTIVITY_MAX_MS` | 100 | hard bound of one activity window |

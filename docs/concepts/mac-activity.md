# Metal activity policies (experimental)

On the real Mac and Linux link, a blocking TBCCL wait can leave the Mac's pipeline thread on slow cores, so the Mac stage takes longer than under exo's MlxRing backend, whose busy-polling worker keeps the Mac fast. Two opt-in, Metal-only mitigations exist. Both are **experimental and off by default**; they are ignored on CUDA and CPU.

| Setting | What it does |
|---|---|
| `EXO_TBCCL_ACTIVITY_MODE=step` (alias `EXO_TBCCL_STEP_ACTIVITY=1`), `EXO_TBCCL_ACTIVITY_DUTY`, `EXO_TBCCL_ACTIVITY_MAX_MS` | a helper thread burns CPU in `memset` calls (GIL released, a private buffer, no tensor, `Work` or communicator access) while the pipeline waits for its input and computes. The window opens when a receive is posted (or, in a pipeline whose previous step had no receive, when an `all_gather` completes) and closes at the next `all_gather` submission, `barrier`, `any_true`, abort or close; a watchdog ends any window after `MAX_MS` (default 100) |
| `EXO_TBCCL_WAIT_SPIN_MS=<ms>` | while a call waits for its `Work`, the calling thread polls it (GIL released per poll) for up to this budget before blocking; only while a `Work` is outstanding |

Both use no CPU when idle, have hard bounds, and leave failure paths unchanged. Neither is the default because their effect depends on the orientation of the pipeline and costs CPU (about one busy core during decode for the activity helper): in a real 27-billion-parameter two-node run, the default TBCCL decode was about 5% slower than MlxRing, and with the activity policy it was 1 to 4 ms per token faster. Lowering the duty below 1.0 loses the effect on the real link.

# Metal activity policies (experimental)

On the real Mac and Linux link, a blocking TBCCL wait can leave the Mac's pipeline thread on slow cores, so the Mac stage takes longer than under exo's MlxRing backend, whose busy-polling worker keeps the Mac fast. Three opt-in, Metal-only options exist. Both are **experimental and off by default**; they are ignored on CUDA and CPU.

| Setting | What it does |
|---|---|
| `EXO_TBCCL_ACTIVITY_MODE=step` (alias `EXO_TBCCL_STEP_ACTIVITY=1`), `EXO_TBCCL_ACTIVITY_DUTY`, `EXO_TBCCL_ACTIVITY_MAX_MS` | a helper thread burns CPU in `memset` calls (GIL released, a private buffer, no tensor, `Work` or communicator access) while the pipeline waits for its input and computes. The window opens when a receive is posted (or, in a pipeline whose previous step had no receive, when an `all_gather` completes) and closes at the next `all_gather` submission, `barrier`, `any_true`, abort or close; a watchdog ends any window after `MAX_MS` (default 100) |
| `EXO_TBCCL_WAIT_SPIN_MS=<ms>` | while a call waits for its `Work`, the calling thread polls it (GIL released per poll) for up to this budget before blocking; only while a `Work` is outstanding |
| `EXO_TBCCL_PREALLOC_GATHER=1` | while a posted receive is in flight, the next `all_gather` destination (a fresh zeroed array) is built, so its allocation and evaluation leave the critical path; it is used only when its shape and dtype match, and an unused one is dropped on close. It uses no extra thread and no extra CPU |

Both use no CPU when idle, have hard bounds, and leave failure paths unchanged. Neither is the default because their effect depends on the orientation of the pipeline and costs CPU (about one busy core during decode for the activity helper): in a real 27-billion-parameter two-node run, the default TBCCL decode was about 5% slower than MlxRing, and with the activity policy it was 1 to 4 ms per token faster. Lowering the duty below 1.0 loses the effect on the real link.

## Measured effect on a small model (0.4.0 release candidate)

Qwen3-0.6B (8-bit), two stages, a Linux CUDA host and an Apple-silicon Mac over a direct Thunderbolt 4 link, 48 decode tokens, five fresh process pairs per arm, median per-token time in milliseconds. "A" has the Linux host as the first stage; "B" has the Mac as the first stage. Output tokens and the KV prefix were identical in every arm.

| Arm | A | B |
|---|---|---|
| MlxRing (exo's own backend) | 6.91 | 8.71 |
| TBCCL default | 9.57 | 11.87 |
| `EXO_TBCCL_ACTIVITY_MODE=step` | 8.53 | 9.63 |
| step + `EXO_TBCCL_WAIT_SPIN_MS=8` | 8.05 | 8.73 |
| `EXO_TBCCL_PREALLOC_GATHER=1` | 9.46 | 11.27 |
| step + `EXO_TBCCL_PREALLOC_GATHER=1` | 7.75 | 9.55 |

What to take from it:

- The step helper closed about a third to a half of the gap to MlxRing here, more in B than in A. An earlier, shorter measurement closed nearly all of the gap in A; that did not reproduce on the current stack, so treat the effect as dependent on orientation, model and software versions.
- Step plus the wait-spin budget was the best arm in B (close to MlxRing); in A it was the second best.
- Preallocation helped about 0.3 ms per token when the Mac is the receiving stage (paired difference across 35 process pairs, standard error 0.06 ms) and made no measurable difference in the other orientation. On a large model, where token time is dominated by compute, it is not expected to matter.
- The cost of the step helper is CPU: the Mac process used about 15% more CPU seconds per run (about 0.8 of a core during decode). Preallocation did not change CPU use.
- These are measurements on one machine pair with a small model, not a promise for other pairs. None of the options is on by default.

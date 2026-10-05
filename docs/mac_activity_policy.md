# Metal activity policy (`EXO_TBCCL_ACTIVITY_MODE=step`, experimental, opt-in, default off)

Config: `EXO_TBCCL_ACTIVITY_MODE=off|step` (default `off`; `EXO_TBCCL_STEP_ACTIVITY=1` is the Phase 63 alias), `EXO_TBCCL_ACTIVITY_DUTY` (0 < d <= 1, default 1.0; **below 1.0 loses the effect on the real link, see `phase64_duty_minimization.md`**), `EXO_TBCCL_ACTIVITY_MAX_MS` (watchdog, default 100). `TbcclPipelineComm.activity_stats()` returns `activity_windows, activity_us, helper_cpu_us, duty, fallback_timeouts` (zeros when off or inert); no per-token logging.

## Why

On the real Mac<->Linux link a blocking TBCCL wait leaves the Mac's pipeline thread on slow cores and the Mac stage takes ~2x as long as under MlxRing, whose busy-polling worker keeps the Mac fast. A helper thread that burns CPU while the pipeline thread is working/waiting for its input restores the speed (physical A: TPOT 11.8 -> 6.8-7.0 ms, Ring 7.2-7.5). Causality was established in Phase 63; Phase 64 minimised the window and the duty.

## When it starts and stops (communicator events only; no model, rank, split or orientation input)

- **Opens** when `recv_like` is called (a receiving stage: the wait for its input plus the local compute that follows, no later: opening when the receive *completes* fails physically), or, in a pipeline whose previous step had no receive, when an `all_gather` completes (observed pattern, not a rank test: `_step_had_recv`). Windows are armed only after the first `all_gather` has completed, so prefill is untouched.
- **Closes** at the next `all_gather` submission, at `barrier` / `any_true` / `abort` / `close`, and by the **watchdog**: any single window ends by itself after `MAX_MS` (a lost end event, an exception, peer death or an idle gap between requests cannot leave a core burning). Decode steps here are 7-12 ms; windows are ~4-6 ms.
- State machine: IDLE (helper blocked on an event, zero CPU) -> window open (helper burns in libc `memset` calls with the GIL released, optionally duty-cycled) -> IDLE. The helper starts lazily at the first window and is joined by `close()`; it never touches tensors, Work, borrows or communicator state.

## B waits and `WAIT_SPIN_MS`

In a pipeline whose stage has no receive (orientation B: the Mac sends then waits at the AllGather for the slow remote stage) the window opens at the previous AllGather completion and closes at the next submit, so the Mac's own compute is covered but the AllGather wait is not. Local B emulator (10 pairs, `docs/data/phase64/local_B/`): TBCCL +3.96 ms vs Ring; spin8 +0.13; activity +0.12; both +0.25 (worse than either alone); activity at duty 0.5 +0.72. The activity policy alone matches `WAIT_SPIN_MS=8` in B and the combination adds nothing, so `WAIT_SPIN_MS` is **redundant** with the policy on every measured case; it is kept unchanged for comparison and marked for Phase 65 cleanup.

## Cost (physical, Orientation A)

Helper ~4.3 ms of CPU per ~6.8 ms step (about 0.6 core); process 0.72-0.77 cores (baseline 0.58); CPU power 1.0-1.2 W (baseline ~0.1 W, benchmark `stepwork` helper 2.7 W, MlxRing 5.4-6.2 W). Ring's own process CPU reads 0.4 cores in the external sampler, which under-reports a spinning thread.

## Metal only / safety

Inert unless `mx.metal.is_available()` (no helper on Linux CUDA or CPU-only hosts; stats zero). Lifecycle tests: idle zero CPU, repeated create/start/stop/shutdown with no leaked thread, close during an open window, peer death with an open window, an exception inside a window ended by the watchdog, concurrent state transitions while shutting down, exact `all_gather` results with the flag on and off. The helper is pure Python + libc `memset` (no native runtime code), so TSan/ASan/UBSan do not apply to it; the concurrency stress test stands in for the focused TSan the plan asks for.

## Status

Opt-in, not default (single physical runs per configuration so far; Phase 65 needs repeats, both orientations on the real link, and the decision about `WAIT_SPIN_MS`). Supersedes `docs/mac_step_activity.md` (the Phase 63 whole-step window).

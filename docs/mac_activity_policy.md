# Metal activity policy (`EXO_TBCCL_ACTIVITY_MODE=step`, experimental, opt-in, default off)

Config: `EXO_TBCCL_ACTIVITY_MODE=off|step` (default `off`; `EXO_TBCCL_STEP_ACTIVITY=1` is the Phase 63 alias), `EXO_TBCCL_ACTIVITY_DUTY` (0 < d <= 1, default 1.0; **below 1.0 loses the effect on the real link, see `phase64_duty_minimization.md`**), `EXO_TBCCL_ACTIVITY_MAX_MS` (watchdog, default 100). `TbcclPipelineComm.activity_stats()` returns `activity_windows, activity_us, helper_cpu_us, duty, fallback_timeouts` (zeros when off or inert); no per-token logging.

## Why

On the real Mac<->Linux link a blocking TBCCL wait leaves the Mac's pipeline thread on slow cores and the Mac stage takes ~2x as long as under MlxRing, whose busy-polling worker keeps the Mac fast. A helper thread that burns CPU while the pipeline thread is working/waiting for its input restores the speed (physical A: TPOT 11.8 -> 6.8-7.0 ms, Ring 7.2-7.5). Causality was established in Phase 63; Phase 64 minimised the window and the duty.

## When it starts and stops (communicator events only; no model, rank, split or orientation input)

- **Opens** when `recv_like` is called (a receiving stage: the wait for its input plus the local compute that follows, no later: opening when the receive *completes* fails physically), or, in a pipeline whose previous step had no receive, when an `all_gather` completes (observed pattern, not a rank test: `_step_had_recv`). Windows are armed only after the first `all_gather` has completed, so prefill is untouched.
- **Closes** at the next `all_gather` submission, at `barrier` / `any_true` / `abort` / `close`, and by the **watchdog**: any single window ends by itself after `MAX_MS` (a lost end event, an exception, peer death or an idle gap between requests cannot leave a core burning). Decode steps here are 7-12 ms; windows are ~4-6 ms.
- State machine: IDLE (helper blocked on an event, zero CPU) -> window open (helper burns in libc `memset` calls with the GIL released, optionally duty-cycled) -> IDLE. The helper starts lazily at the first window and is joined by `close()`; it never touches tensors, Work, borrows or communicator state.

## B waits and `WAIT_SPIN_MS` (corrected by the Phase 65 physical Orientation-B validation)

In a pipeline whose stage has no receive (orientation B: the Mac sends then waits at the AllGather for the slow remote stage) the window opens at the previous AllGather completion and closes at the next submit. The local B emulator predicted a full fix (policy +0.12..+0.20 ms vs Ring, spin8 +0.10..+0.13, both +0.18..+0.25), but **the real link did not agree**: 3 interleaved repetitions closed only 37-41 % of the baseline-to-Ring gap (policy 11.58 ms vs baseline 12.99 vs Ring 9.33; `docs/phase65_final_validation.md`), while Phase 60 measured `WAIT_SPIN_MS=8` closing 56-59 % in one run. The two mechanisms are therefore **not interchangeable**: the activity policy is validated for the receiving-stage case (orientation A), `WAIT_SPIN_MS` remains the only mechanism with physical evidence of helping B. `WAIT_SPIN_MS` is kept supported and not deprecated; there is no automatic selection by rank or orientation (the plan forbids it); using both together was not measured on the link.

## Cost (physical, repeated in Phase 65)

Orientation A, 3 repetitions: helper ~3.9 ms of CPU per ~6.8 ms step (~0.57 core); process 0.77 cores (baseline 0.58, Ring ~0.4 by the external sampler); CPU power ~0.4 W (run-level mean; Phase 64 measured 1.0-1.2 W; baseline ~0.15 W, MlxRing 5.4-5.6 W). Orientation B: ~1 core of helper CPU with only 39 % of the gap closed (not recommended there). See `docs/phase65_power_efficiency.md`.

## Metal only / safety

Inert unless `mx.metal.is_available()` (no helper on Linux CUDA or CPU-only hosts; stats zero). Lifecycle tests: idle zero CPU, repeated create/start/stop/shutdown with no leaked thread, close during an open window, peer death with an open window, an exception inside a window ended by the watchdog, concurrent state transitions while shutting down, exact `all_gather` results with the flag on and off. The helper is pure Python + libc `memset` (no native runtime code), so TSan/ASan/UBSan do not apply to it; the concurrency stress test stands in for the focused TSan the plan asks for.

## Status (Phase 65)

**Supported experimental, opt-in, not default, not renamed.** Orientation A repeats cleanly (3/3 repetitions: 109-118 % of the gap closed, 0.60 ms below Ring, first-use and stage at or better than Ring, ~0.4 W CPU power, tokens identical); orientation B fails the plan's >= 70 % gate on the real link (37-41 %). Per the plan, a policy that fails B is not default-enabled, is not promoted to a supported config name (`EXO_TBCCL_ACTIVITY_MODE`/`_DUTY`/`_MAX_MS` stay as they are; duty 1.0 is the only validated value), and does not displace `WAIT_SPIN_MS`. Recommended use: a Metal rank that *receives* its stage input (the Mac as a downstream pipeline stage), duty 1.0. Not recommended: a Mac stage that only sends and waits (B). Supported statement: same-process activity during the relevant pipeline window restores execution performance; the exact scheduler or frequency mechanism is not established.

Supersedes `docs/mac_step_activity.md` (the Phase 63 whole-step window).

## Status (Phase 66)

Orientation B physical, 3 repetitions: STEP 31 % of the gap, WAIT_SPIN alone -3 %, STEP + `EXO_TBCCL_WAIT_SPIN_MS=8` 64 % (53-82 %; 0.70 W, 1.18 cores vs Ring ~6 W). The two mechanisms do not overlap in time (0.04 ms/step) and together cover ~99.6 % of the step, so the combination is the best known opt-in setting for a Mac stage that mostly waits, but it misses the 80 % gate and is 1.3 ms above Ring. No unified state machine, no default, no auto-selection. See `docs/phase66_results.md`.

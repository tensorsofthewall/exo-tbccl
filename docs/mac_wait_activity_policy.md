# Mac wait-activity policy (`EXO_TBCCL_WAIT_SPIN_MS`): status after Phase 60

## The policy as implemented (unchanged in Phase 60)

*Fixed bounded wait.* When a `TbcclPipelineComm` call blocks on a Work (`wait(t)` with no timeout), **on Metal only** and only if `EXO_TBCCL_WAIT_SPIN_MS` > 0, the calling thread polls the Work with the native `wait(0)` (GIL released at every poll) until it completes or the budget expires, then falls into the normal blocking native wait. Properties:

* **Only while a Work is outstanding.** Nothing runs when no Work exists: no background thread, no timer. An idle communicator used 0.0 busy cores over 5 s with the policy on (`tests/test_wait_spin.py::test_idle_communicator_uses_no_cpu_with_the_policy_on`; the 30 s variant of the plan was shortened to 5 s because the check is a ratio).
* **Hard bound.** Each wait spins at most the budget; a 600 ms wait with an 8 ms budget spun for under 40 ms (the test's assertion limit) and used < 0.2 s CPU in total, the rest being a blocking wait (`test_a_long_wait_spins_only_up_to_the_budget_then_blocks`). Prefill-scale and failure-scale waits therefore fall back to blocking.
* **Failure paths unchanged.** The poll returns as soon as the Work is terminal, including a failed Work: peer death while polling surfaces the structured error promptly (`test_peer_death_while_the_caller_polls_surfaces_promptly`), an abort from another thread ends the wait within 3 s (`test_abort_from_another_thread_ends_the_polling_wait`), and `close()` with a Work outstanding is bounded (`test_close_with_outstanding_work_is_bounded`), each with a 1000 ms budget; the full suite passes on the Mac with the budget at 0, 8 and 1000.
* **Inert off Metal.** `_metal_available()` is checked once at construction; on Linux/CUDA `comm._spin_s == 0` with any value set (`test_metal_only_experiments_are_ignored_off_metal`; Linux suite identical with the variable at 8).
* **Counters, no logging.** `comm.wait_stats`: `waits_total`, `waits_spun`, `waits_completed_during_spin`, `waits_fell_back`, `total_spin_us`, `max_spin_us` (per decode-like run: 20/20 waits spun, >= 18 completed during the spin, test `test_decode_like_waits_complete_during_the_spin_and_are_counted`).
* **No exo change.** exo does not see the budget; `FastPathConfig` + env only.

## Causal evidence (Phase 59, local) and what Phase 60 added (physical)

* Local emulator (real Mac stage, real backends): in-process CPU activity while a communication call is outstanding removes the Mac slowdown (orientation B +4.1 -> +0.2 ms vs MlxRing; a 6-8 ms budget +0.1-0.3 ms; 2 ms +0.95; 0.5 ms none); a 50%-duty thread removes half; a spinning **separate process** nothing; QoS nothing; the synthetic Metal stage shows the same effect (generic, not Qwen-specific). Allowed claim: *Mac performance depends causally on in-process CPU activity during the wait.* Not established: the macOS scheduler/thread-group/DVFS mechanism.
* **Physical (Phase 60, spin 8, one run per cell, AER 2/0/0):** orientation B closes **56-59%** of the TBCCL-vs-Ring gap (sampler 94%, AllGather 71%, stage compute 56%, but resume/graph build only 20-24%); orientation A closes **7-14%** and no Mac row moves. The Mac in A is on the critical path and almost never has a Work outstanding (recv wait 53 us), so a wait-gated activity has nothing to act on.

## Decision

**The policy is not made default.** The plan's default-enablement gate needs both physical orientations to improve significantly and the Mac rows to move toward Ring; A fails and B is partial. The setting stays an **opt-in Metal-only experiment** (`EXO_TBCCL_WAIT_SPIN_MS=<ms>`, 0 = blocking = default); a recommended value for orientation-B-like pipelines (the Mac waits for a slower remote stage) is 6-8 ms, with no physical evidence beyond B's 56-59%. No minimum-budget refinement, operation-class filtering or adaptive variant was built or tested (the plan's Part G-M is conditional on both orientations validating), and no budget was tuned over the link. exo-tbccl stays **0.2.0**.

## CPU cost

Mac process CPU over the 16-token decode window (physical): TBCCL baseline 0.27-0.28 s in 2.8-3.0 s wall (0.09-0.10 cores), TBCCL spin 8 0.30 s (0.10-0.11 cores: the spin adds ~0.02 s), MlxRing 2.5-2.8 s (0.95-0.96 cores: it keeps a core active throughout). Local emulator decode windows (Phase 59): blocking TBCCL 0.44 s CPU in 1.04 s, spin 0.24-0.27 s in 0.72-0.77 s, MlxRing 0.51 s in 0.70 s: a spin that shortens the stalled run does not raise total CPU. Power was not measured (`powermetrics` needs root; not invoked).

## Open question for the next phase

If the slowdown in orientation A occurs in the Mac's own work between waits, the right activity window is **the Mac's compute window, not the communication wait**; whether continuous (or step-scoped) in-process activity closes physical A, at what CPU cost, is untested and would need its own local emulator that actually reproduces A (the current one reproduces about a quarter of A's gap) before any physical run.

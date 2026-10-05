# `EXO_TBCCL_STEP_ACTIVITY` (experimental, opt-in, Metal only, default off)

`EXO_TBCCL_STEP_ACTIVITY=1` (and `EXO_TBCCL_STEP_ACTIVITY_MAX_MS`, default 250) makes `TbcclPipelineComm` run one helper thread (`exo_tbccl/step_activity.py`) that burns CPU in `memset` calls (GIL released, private buffer, no tensor/Work/communicator access) **between two AllGathers**:

- opens when `all_gather` completes; closes when the next `all_gather` is submitted, at `barrier`, `any_true`, `abort`, and `close` (which joins the thread);
- hard bound: any single window ends by itself after `MAX_MS` (a lost event, an exception or an idle gap between requests cannot leave a core burning);
- started lazily at the first window; idle it blocks on an event (zero CPU, tested); ignored on CUDA/CPU (`_metal_available()`), no model/rank/split/orientation knowledge.

Evidence: physical Orientation A, TPOT 11.1 -> 7.19 ms (Ring 7.33), Mac first-use 1.06 -> 0.19 ms, stage 4.2 -> 2.5 ms (one run, `docs/phase63_pcore_causality.md`); local Orientation B emulator +3.93 -> +0.15 ms vs Ring. Cost: about one busy core during decode (1.15 cores for the process, CPU power ~0.1 -> ~2.8 W in the nearest measurement vs Ring's ~5.7 W); `WAIT_SPIN_MS` is independent and the combination adds nothing. Not a default: the CPU/power cost is the price of the latency and one run per configuration is thin evidence. A narrower window (`compute`, 35 % of a step locally) or a duty below 100 % could cut the cost but was not measured physically.

Tests: `tests/test_step_activity.py` (default off/env parse, idle -> busy -> idle, hard bound, 100 start/stop/shutdown-while-active cycles with no leaked thread, exact `all_gather` results with the flag on and off, close during an open window, peer death with an open window); Linux (CUDA/CPU host) is inert (`comm._act is None`).

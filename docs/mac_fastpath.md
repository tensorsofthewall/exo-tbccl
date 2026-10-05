# Mac fast path (experimental, opt-in): `EXO_TBCCL_WAIT_SPIN_MS`

**What it is.** While a `TbcclPipelineComm` call waits for its Work, the calling thread polls the Work (native `wait(0)`, GIL released at every poll) for up to this many milliseconds before blocking in the native wait. It imitates the one thing MlxRing does that TBCCL does not: MlxRing's non-blocking socket worker keeps the process's CPU active while a transfer is pending. **Metal only** (`_metal_available()` is checked once at construction; on CUDA the setting is ignored and `comm._spin_s == 0`, asserted by `tests/test_fastpath.py::test_metal_only_experiments_are_ignored_off_metal` and by the Linux suite passing unchanged with the variable set). Default `0` = off. No TBCCL, wire or ABI change; no new thread (so no new TSan surface); `config.py` + a few lines in `group.py::wait`.

**Why.** Phase 59 (`docs/phase59_mac_scheduling.md`) reproduced the physical Mac slowdown locally and showed it depends causally on CPU activity inside the pipeline process while it waits: Ring-like activity removes it, a 50% duty cycle half of it, activity in another process none. Every Mac-executed row that was inflated under TBCCL (sampler, Python graph build, stage compute, first use) returns to MlxRing's.

**Measured (Mac, real Qwen stage against the remote-peer emulator, interleaved pairs).**

| budget | orientation B: TBCCL - Ring | orientation A |
|---|---:|---:|
| off (baseline) | +4.38 +/- 0.08 ms | +0.84 +/- 0.05 ms |
| 0.5 ms | +4.52 +/- 0.11 | |
| 2 ms | +0.95 +/- 0.03 | |
| **6 ms** | **+0.27 +/- 0.05** | **+0.14 +/- 0.02** |
| 30 ms | +0.11 +/- 0.02 | +0.13 +/- 0.02 |

Real model on ordinary Mac loopback (both ranks real, where the Mac is never idle): `spin6` 11.07 vs 11.31 ms (**-0.24 +/- 0.03 ms**, 12 pairs; one 18.9 ms outlier in a first set was a one-off); tokens identical in every run. CPU: over the decode window the spinning process used 0.24-0.27 s CPU in 0.72-0.77 s wall, against 0.44 s in 1.04 s for blocking TBCCL and 0.51 s in 0.70 s for MlxRing (the slowed blocking state costs more CPU than the spin).

**How to size it.** The budget must cover the peer's compute between this rank's communication calls (Phase 58: ~2.7-3.4 ms for the Linux 21-layer stage), so 6-10 ms is the recommended range; longer waits (prefill chunks, 100+ ms) exceed the budget and fall back to the blocking wait, bounding the extra CPU.

**Status.** Not enabled by default and **not validated on the physical link**: the emulator reproduces the physical slowdown fully in orientation B and about a quarter of it in A, and has loopback transit. A physical A/B is the remaining gate (`docs/phase59_results.md`). exo-tbccl stays 0.2.0 because no default changed.

**Safety checks done.** Bit-exact synthetic chains N=2/3/4, Float32/BFloat16 (`test_wait_spin_is_bit_exact_vs_blocking_wait`); full exo-tbccl suite on the Mac with the variable set to 6 and to 1000 (peer death, cancellation, close with work outstanding, dropped references and repeated create/destroy are in the suite): 72 passed, 1 skipped, identical to the default; Linux 73 passed, 20 Metal-only skips, identical with every experiment variable set.

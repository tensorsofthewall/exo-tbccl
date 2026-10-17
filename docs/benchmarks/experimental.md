# Experimental settings

The default benchmark runs the released, default configuration of `MlxTbccl`. exo-tbccl also has optional, off-by-default settings, described in [Metal activity policies](../concepts/mac-activity.md) and the [configuration reference](../reference/configuration.md). The benchmarks never turn them on by themselves:

* A baseline workload that sets any of `EXO_TBCCL_RECV`, `EXO_TBCCL_ASYNC_SEND`, `EXO_TBCCL_ACTIVITY_MODE`, `EXO_TBCCL_MANAGED` or `EXO_TBCCL_ACTIVITY_THREAD` is **refused**. A separate workload must declare `tier: fastpath`, and its results carry that tier and the exact environment in every record and every report.
* Results from the experimental tier are never pooled with the default, and the comparator treats the setting as part of the experiment's identity.
* The provider's `validate` operation rejects an activity policy unless `tier` is `fastpath`.

## The Metal activity-step result is exploratory

On the reference machines, with the Linux host first, the `step` activity policy (a helper thread that burns CPU on the Mac while the pipeline waits and computes) measured **9.48 ms TPOT against 14.44 ms for default `MlxTbccl`** (descriptive difference of -34%). That is a measurement of two lifecycles with the policy against four default controls, all completely separated; with only two lifecycles an exact test cannot reach p ≤ 0.05 (its floor is 0.133), so by the benchmark runner's rule the result is *insufficient evidence*, not a confirmed improvement.

* It is **not** the default result and not a validated production optimisation.
* It costs about one busy Mac core (CPU 94% against 72% of one core in the sampled runs).
* It was not tested for other models, a loaded Mac, battery power or other thermal states, and in the Mac-first orientation there was no control inside the same campaign.
* Receive-buffer reuse and asynchronous send, measured the same way, were indistinguishable from the default (14.41 to 14.55 ms). A wait-spin budget gave 13.47 ms in a single lifecycle, which is not enough to conclude anything.

If you reproduce it, report the number of lifecycles, the orientation, the control, and the CPU cost together.

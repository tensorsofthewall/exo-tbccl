# Methodology and limitations

* **Independent unit.** The replication unit is an independent cluster lifecycle: nodes start fresh, the instance is placed, the model is loaded, and clean-up is verified on every host. Requests and passes inside a lifecycle are samples of that lifecycle, not replicates. Tests and lifecycle counts use lifecycle medians.
* **Actual placement.** The stage order and layer split Exo chose are read back and recorded; only identical placements are compared, and a placement that does not match the experiment stops the run.
* **Actual tokens.** Token counts come from the server-reported usage, cross-checked against the number of stream events. Events with empty text are tokens. Exo exposes no token ids, so correctness means token-text equality with a single-device reference.
* **Correctness outside the timer.** Reference and check gates run separately from timed phases. A failed gate means no measurement.
* **Model loading is separate.** Load time and deletion time are timed on their own and are not part of decode figures.
* **Clocks.** Each interval uses one machine's monotonic clock; timestamps of different machines are never subtracted.
* **Default versus experimental.** The default benchmark never enables an optional fast path or activity policy. Experimental runs are a separate tier and are never pooled with the default.
* **Observed difference versus cause.** A difference between MlxRing and MlxTbccl is a measured difference between two integrations. Naming a cause needs an experiment that isolates it.
* **Hardware observability.** Link state and interface counters are checked around each lifecycle. Where PCIe AER is disabled the result says AER was unavailable instead of reporting zero errors.
* **Not covered.** Model quality, long prompts on the 8 GiB GPU, more than two nodes, other model families, and the physical link rate (these benchmarks run over TCP).

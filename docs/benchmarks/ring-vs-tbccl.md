# MlxRing versus MlxTbccl

`MlxRing` is Exo's own pipeline backend. `MlxTbccl` is the backend exo-tbccl adds, which moves the pipeline's tensors with TBCCL. A comparison of the two is a comparison of **two integrations of the same pipeline**, including how often each synchronises, builds graphs and waits. It is not a measurement of a transport protocol, and the benchmarks label such a result `ab_transport` for that reason.

## What a valid comparison fixes

* **The backend that actually ran.** Requesting `MlxTbccl` must produce an `MlxTbcclInstance` whose communicator is ready; requesting `MlxRing` must produce an `MlxRingInstance`. The client reads back what Exo placed and stops if it differs. A run that silently fell back to the other backend has no measurement.
* **The placement that actually happened.** Exo may assign the two stages to the machines in either order at startup. The client records the stage order and the number of layers on each stage, and the runner compares only runs with the same order and the same layer split. A `MlxRing` run with the Mac first and 19 layers must never be pooled with a `MlxTbccl` run with the Linux host first and 9 layers.
* **The same model, quantisation, prompts, output lengths and concurrency.**
* **Independent lifecycles.** Each lifecycle is a fresh cluster start. Requests inside a lifecycle are not replicates.
* **Interleaving.** Where the cluster state could drift (thermal state, other load), run the backends interleaved, not one after the other.

## What is verified in the output

* Exo streams one event per generated token, and a token can decode to an empty string, so an event with empty text still counts as a token. The token count of a request is the server-reported `usage.completion_tokens`, cross-checked against the number of events. Exo exposes **token text, not token ids**, so correctness is token-text equality with a single-device reference; the benchmarks do not claim token-id equality.
* A divergence from the reference is *exact* (none), a *near tie* when the reference's own top-1/top-2 margin at the first difference is below the declared tie margin (0.251), and a failure otherwise. A failed gate means the run records no measurements.

## Reading the result

Compare per-lifecycle medians, count lifecycles, and look at CPU cost as well as latency. A difference between the backends can come from where each spends its time (for example a blocking wait versus a busy-polling worker); the benchmark that measures the difference does not by itself say which part of the step is responsible. See [Results](results.md) for what was and was not isolated.

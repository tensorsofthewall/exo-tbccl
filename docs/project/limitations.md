# Known limitations

Each limitation is labelled with the project that owns it. The same categories are used in the other TBCCL projects' pages. Core transport limits (TCP only, no RDMA, no direct CUDA-to-Metal DMA, no encryption) are in the [TBCCL limitations](https://tbccl.tensorsofthewall.com/en/latest/project/limitations.html).

## Unsupported

| Limitation | Owner | Detail |
|---|---|---|
| Parallelism | exo-tbccl | pipeline parallelism for text generation only. Tensor parallelism is rejected at placement; image and classifier-free-guidance models are rejected |
| Addresses | exo-tbccl / exo | one advertised address per node; topologies that need different addresses for different peers are rejected |
| Python | exo-tbccl | CPython 3.13 or newer; the wheels are `cp313` |
| The required exo | exo integration | exo-tbccl needs the `PipelineComm` seam, `MlxTbccl` and the runner byte exchange, which are not in upstream exo; they are published as the fork tag `tensorsofthewall/exo-fork@tbccl-integration-v2`, based on upstream exo `bc412e22` (`tbccl-integration-v1`, upstream `75d8ad2b`, remains available) ([compatibility](../reference/compatibility.md)) |
| Reductions beyond `Sum`, recovery after a communicator failure | TBCCL | see the TBCCL limitations |

## Supported but not validated

| Item | State |
|---|---|
| More than two stages on hardware | synthetic pipelines of 2 to 4 stages ran on loopback; the physical validation used two stages |
| Models other than Qwen3 | receive-buffer reuse is audited only for Qwen3 with a standard KV cache and the synthetic model; the default (no reuse) is model-agnostic but only Qwen3-0.6B and a 27B 4-bit Qwen3 model were run on two hosts |
| Larger models | `Qwen3.8-27B-4bit` was run on the two-host pair at wire protocol 4 (both orientations, short prompts only: 138 tokens on the 8 GB GPU); other large models are not validated |
| The Metal activity policies | experimental, off by default |
| Peer failure on a physical link | validated on loopback; do not kill a physical peer to test it |

## Known upstream issues

| Issue | Owner | Detail |
|---|---|---|
| Prompt length on an 8 GB GPU | MLX / model | prompts were limited to 138 tokens for the 27B model on the 8 GB GPU because of the model's CUDA gated-delta fallback memory use (the same with `MlxRing`) |
| `uv sync` removes exo-tbccl | exo tooling | a dependency sync inside exo can uninstall the package; reinstall it |

## Performance

- At default settings on the 27B run, TBCCL's decode median was about 5% slower than exo's `MlxRing` backend (about 146 versus 139 ms per token); with the opt-in Metal activity policy it was 1 to 4 ms per token faster. exo-tbccl does not claim to be faster than `MlxRing`.
- CUDA activations are staged through pinned host memory; Metal arrays are read in place.
- Optional fast paths (receive-buffer reuse, asynchronous decode sends, managed-memory mapping) are off by default and measured as neutral end to end ([fast paths](../concepts/fast-paths.md)).

## Operational workarounds

These are findings from the validated environment, not architectural requirements.

| Situation | Workaround |
|---|---|
| a Mac exo node started from a detached tmux session or a nohup job never joins the cluster (its discovery sends fail with `No route to host`) | start the Mac node from a live Terminal or `ssh -tt` session ([running across two hosts](../guides/two-host.md)) |
| another program listens on UDP port 52413 on the same host | stop it; on macOS it can take the unicast discovery replies from exo |
| exo advertises host RAM, not GPU memory, for a CUDA node | set `OVERRIDE_MEMORY_MB` so the layer split fits |
| macOS firewall prompt | answer it once from a local session; never script firewall changes |

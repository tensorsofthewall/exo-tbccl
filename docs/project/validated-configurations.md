# Validated configurations

Configurations that were exercised, not a benchmark report. Evidence is in the [validation report](../validation/0.3.0.md). "Release-candidate wheels" means the published 0.3.0rc1 wheels were used. Versions and platforms are in the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html).

| Configuration | What was run | Result |
|---|---|---|
| Linux CUDA host and Mac Metal host, Thunderbolt 4, release-candidate wheels, a real exo cluster | Qwen3-0.6B placed with `MlxTbccl` (min 2 nodes); three complete cycles of cluster start, 60 s idle stability, placement, generation (short, 374-token and 7,390-token prompts), shutdown; the layer split differed between cycles | 3 of 3 discovered, stable (no peer expirations, one master), placed on both hosts, generated; outputs match the earlier single-device, ring and TBCCL reference runs; both ranks ran `send`/`recv`, `all_gather`, `barrier` and `any_true`; clean shutdown on both hosts; link health unchanged |
| The same hosts, standalone transport probe | the exo-tbccl communicator between a CUDA rank and a Metal rank, five fast-path modes, 30 verified iterations each | passed bit for bit |
| Linux CUDA host and Mac Metal host, Thunderbolt 4, a real exo cluster | `mlx-community/Qwen3.8-27B-4bit` (64 layers) as a two-node pipeline, at an earlier TBCCL wire protocol (3) | passed; not repeated at wire 4 |
| Loopback, two ranks | Qwen3-0.6B-8bit split 21/7 and 7/21, prompts of 49, 577 and about 7,000 tokens | token ids identical to the unsharded greedy reference |
| Synthetic pipelines, 2 to 4 stages, loopback | the pipeline driver | bit-identical boundaries to `MlxRing` |
| A Mac exo node started from a detached tmux session | observed: discovery sends failed and the node never joined the cluster | see [known limitations](limitations.md) (operational workaround) |

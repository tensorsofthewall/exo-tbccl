# How do I know TBCCL is being used?

A silent fall-back to exo's default MLX backends (for example `MlxRing`) or to a single-node placement would invalidate a benchmark. Use this page when the backend does not seem to be active (placement fails, or the instance is not `MlxTbcclInstance`), and to show that the pipeline runs on exo-tbccl and TBCCL.

## The verification order

Go down this list on **every** node, in order. Each row checks one layer; discovery (which finds the other machine) and transport (which moves the activations) are different layers and are verified separately. Stop at the first row that fails.

```{table}
:class: tb-stack tb-stack-ladder

| # | Question | Layer | Check | A pass looks like |
|---|---|---|---|---|
| 1 | Which Exo integration is this? | exo-fork | `git -C <exo-fork checkout> describe --tags --exact-match` | `tbccl-integration-v2` (or `-v1` for the legacy path) |
| 2 | Is the adapter available? | exo-tbccl | [step 1](#1-the-package-is-usable-on-every-node) below | `(True, '')` with the version for your path (0.3.0 for the extra, 0.3.1 for the manual wheel) |
| 3 | Are both nodes' environments compatible? | exo-tbccl and exo | run row 2 on both: the same version; both on the same exo tag; the same `EXO_ZENOH_NAMESPACE` | identical output |
| 4 | Did the cluster discover both hosts? | **Exo** (IPv6 link-local discovery) | `/state` shows the nodes ([discovery troubleshooting](discovery-troubleshooting.md)) | `2 nodes`; this says nothing yet about TBCCL |
| 5 | Was `MlxTbccl` selected? | Exo placement | [step 2](#2-the-instance-type-is-mlxtbccl) below | `['MlxTbcclInstance']` with layers on both nodes |
| 6 | Does the pipeline communicate through TBCCL? | **TBCCL** (TCP over IPv4) | [step 3](#3-the-runners-built-a-tbccl-communicator) and [step 4](#4-operation-level-evidence) below | `TBCCL communicator ready` on both ranks, on the link addresses you expect |
```

A cluster that forms (row 4) does not prove that TBCCL carries the pipeline, and a working TBCCL path does not prove that discovery works: the first needs IPv6 link-local multicast, the second IPv4 TCP.

## 1. The package is usable on every node

```sh
.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"      # 0.3.0 (True, '') with the tbccl extra of tbccl-integration-v2; 0.3.1 (True, '') with the 0.3.1 wheel installed by hand
```

## 2. The instance type is `MlxTbccl`

```sh
curl -s localhost:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print([next(iter(i)) for i in d['instances'].values()])"
# ['MlxTbcclInstance']
```

Any other instance type (for example `MlxRingInstance`) is a different backend. In the same JSON, `instances[...]['MlxTbcclInstance']['shardAssignments']` lists each runner's shard (`deviceRank`, `startLayer`, `endLayer`): both nodes must hold layers. A single shard means a single-node placement.

## 3. The runners built a TBCCL communicator

The exo log of each node contains, once the instance starts:

```text
rank 0/2 TBCCL bootstrap on 192.168.3.1
rank 0/2 TBCCL communicator ready (instance <id>)
```

on one node and `rank 1/2 ... on 192.168.3.2` on the other: the rank, the world size and the address exo chose for that node. This is the evidence of world size, rank and peer mapping; the addresses should be the link addresses you expect.

## 4. Operation-level evidence

`EXO_TBCCL_TRACE=1` makes the package log the path of every operation (host, CUDA-direct or Metal-direct) with its peer and size through the standard-logging logger `exo_tbccl` at INFO level. In the validated runs these lines did **not** appear in exo's own log, because exo does not attach a handler to that logger; to see them, add a logging handler for `exo_tbccl` in the runner process (for example from a `sitecustomize` on `PYTHONPATH`). `comm.stats` counts any adapter copy, which must stay 0 on the paths where the adapter makes no copy.

Counting calls from outside exo is possible with the same kind of wrapper around `TbcclPipelineComm`. A generation across two nodes performs `send`/`recv` pairs for every decode step plus `all_gather`, `any_true` and `barrier` calls for the runner-level agreements, on **both** ranks (in the validated Qwen3-0.6B runs: 209 `all_gather`, 14 `any_true`, 9 `barrier` on each rank, and 128 receives on the later stage matching the sends of the earlier one).

## 5. For benchmarking

Record the instance type, the layer split and the exo-tbccl version next to your results. To compare with exo's default backend, place the same model and prompt with `"instance_meta": "MlxRing"` and the same node set.

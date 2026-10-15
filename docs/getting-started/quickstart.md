# Quickstart

Form a two-node exo cluster, place a model with `MlxTbccl`, and generate. This is the flow validated with the published wheels (a Linux CUDA host and a Mac, joined by Thunderbolt 4). Install exo and exo-tbccl on both hosts first ([installing](install.md)), connect the machines with [Two machines over Thunderbolt](https://tbccl.tensorsofthewall.com/en/latest/getting-started/two-machine-thunderbolt.html), and read [Before you start](../guides/two-host.md): the model must already be on both machines and IPv6 link-local must work on the link.

## 1. Start exo on both hosts

Use the same namespace on both nodes so they join one cluster, and give each its own exo state directory. On Linux:

```sh
cd <exo-fork checkout at tag tbccl-integration-v2>
EXO_HOME=$HOME/.exo EXO_ZENOH_NAMESPACE=mycluster uv run --no-sync exo
```

On the Mac, start exo from a **live Terminal or `ssh -tt` session** (not a detached tmux/nohup job; see [running across two hosts](../guides/two-host.md)):

```sh
ssh -tt <mac> 'export PATH=/opt/homebrew/bin:$PATH EXO_HOME=$HOME/.exo EXO_ZENOH_NAMESPACE=mycluster; cd <exo-fork checkout at tag tbccl-integration-v2> && exec uv run --no-sync exo'
```

`EXO_ZENOH_NAMESPACE` (or `--namespace`) must be the same on every node of the cluster. Add `EXO_OFFLINE=true` (or `--offline`) when the models are pre-staged locally and nothing should be downloaded; this guide assumes the model is already in each machine's models directory ([where exo looks](../guides/two-host.md#before-you-start)). Within seconds of the second node starting, both should see each other.

## 2. Check the cluster

```sh
curl -s localhost:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['topology']['nodes']), 'nodes;', {k[:6]: v['friendlyName'] for k, v in d['nodeIdentities'].items()})"
```

Expect `2 nodes`. Exactly one node acts as master (the other logs `elected master - demoting self` once, right after discovery). If you do not see two nodes, use [discovery troubleshooting](../guides/discovery-troubleshooting.md).

## 3. Place a model with `MlxTbccl`

```sh
curl -X POST localhost:52415/place_instance -H 'content-type: application/json' \
  -d '{"model_id": "mlx-community/Qwen3-0.6B-8bit", "sharding": "Pipeline", "instance_meta": "MlxTbccl", "min_nodes": 2}'
```

`min_nodes: 2` makes placement refuse a single-node placement. exo assigns the layers (the split depends on each node's available memory; for a 28-layer model it may be, for example, 7 layers on the Mac and 21 on Linux) and each node's runner loads its share. Wait until every runner reports ready in `/state`.

## 4. Generate

```sh
curl -s localhost:52415/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "mlx-community/Qwen3-0.6B-8bit", "messages": [{"role": "user", "content": "Count from one to five in words."}], "temperature": 0, "max_tokens": 32}'
```

## 5. Confirm `MlxTbccl` is active and stop

```sh
curl -s localhost:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print([next(iter(i)) for i in d['instances'].values()])"     # ['MlxTbcclInstance']
```

See [how do I know TBCCL is being used?](../guides/verify-active.md). Delete the instance with `curl -X DELETE localhost:52415/instance/<instance id>` and stop each exo node with Ctrl-C (or `kill -TERM`). `kill -9` is a last resort.

## Using the API directly

You normally do not call exo-tbccl yourself; this is what exo does:

```python
from exo_tbccl.group import TbcclPipelineComm

comm = TbcclPipelineComm.create(rank, world_size, exchange, bind_host=host, advertise_host=host)
# exchange(purpose, payload) -> list[bytes]: the host application's all-gather of opaque bytes
#   (exo supplies its runner byte exchange).

x = comm.recv_like(template, src)      # MLX array in, MLX array out; the Work is waited
comm.send(array, dst)
comm.flush_sends([(a, dst), ...])      # submits every send first, then waits as a group
y = comm.all_gather(array)             # rank-order concatenation along axis 0
comm.barrier()
comm.any_true(flag)
comm.close()                           # explicit; no dependence on finalizers
```

Rank and world size come from exo's `PipelineShardMetadata`, never from node ordering. Errors are typed exceptions built from TBCCL's structured result codes (`TbcclTransportError`, `TbcclAbortedError`, `TbcclTimeoutError`, `TbcclDeviceError`, and so on) and carry rank, peer and operation. `comm.stats` counts any adapter copy, which must stay 0. See [architecture](../concepts/architecture.md) for how exo uses it.

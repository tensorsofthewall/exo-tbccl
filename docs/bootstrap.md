# Bootstrap

TBCCL does not discover nodes. exo supplies membership (shard assignments), rank (`device_rank`), world size and one address per node
(`MlxTbcclInstance.hosts_by_node`, chosen at placement from exo's topology).

## Flow inside each runner

```
UniqueId exchange (all-gather; rank 0's id is used) -> tbcclBootstrapBegin(rank, world, id, bind=host, advertise=host)
 -> endpoint blob (256 bytes, opaque) -> all-gather of blobs -> tbcclBootstrapComplete -> communicator
```

Each all-gather is `RunnerByteExchange.all_gather(namespace, payload, timeout_s)`.

## The exo primitive (generic, not TBCCL-specific)

A runner only talks to its own Worker, so exo gained one small primitive: **all-gather of opaque bytes between the runners of an instance**.
runner -> Worker (mp channel) -> a new zenoh topic `runner_bytes` -> peer Workers -> peer runners. Messages carry
`(instance_id, namespace, rank, world_size, payload)`:

- Stale blobs cannot match a new communicator: the instance id is part of the key, and a restarted instance has a new id.
- The pub/sub layer is best effort, so each rank republishes until the gather completes and keeps answering late peers for a bounded time; a Worker
  buffers (bounded) messages that arrive before its runner exists.
- Bounded by a timeout; cancellation and a closed inbox unblock it.
- It performs no discovery and leaves no service behind. It could carry other backends' bootstrap data.

Why not a side channel in exo-tbccl: exo already has a Worker/Router control plane that reaches every node; reusing it avoids a second
rendezvous mechanism.

## Operational note

The Worker is built at startup and again after a master election; both must be given the runner-byte channels (they are required
constructor arguments for that reason).

# Bootstrap

TBCCL does not discover nodes. exo supplies membership (shard assignments), rank (`device_rank`), world size and one address per node.

```
UniqueId exchange (all-gather; rank 0's id is used)
 -> tbcclBootstrapBegin(rank, world, id, bind=host, advertise=host)
 -> 256-byte opaque endpoint blob -> all-gather of the blobs
 -> tbcclBootstrapComplete -> communicator
```

Each all-gather is `RunnerByteExchange.all_gather(namespace, payload, timeout_s)`, a small generic primitive added to exo: an all-gather of opaque bytes between the runners of an instance. A runner talks only to its own worker, so the bytes go runner, worker, a pub/sub topic, peer workers, peer runners. Messages carry the instance id, a namespace, the rank, the world size and the payload, so a stale blob can never match a new communicator. The pub/sub layer is best effort, so each rank republishes until the gather completes and keeps answering late peers for a bounded time; a worker buffers (bounded) messages that arrive before its runner exists. The exchange is bounded by a timeout (120 seconds for bootstrap), performs no discovery and leaves no service behind.

Reusing exo's existing control plane avoids a second rendezvous mechanism. The exo worker is built at startup and again after a master election; both must be given the runner-byte channels.

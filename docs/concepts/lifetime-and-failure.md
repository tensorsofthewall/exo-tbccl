# Lifetime and failure

- **Lifetime.** A borrowed MLX array, its view and its DLPack export stay alive until the TBCCL `Work` is terminal, whatever happens to the Python handle. No correctness depends on `__del__` or on interpreter shutdown; `close()` is explicit. `close()` aborts only when work is in flight, drains, releases borrows, then destroys the communicator. Because an abort fails a peer that is still completing its last collective, quiesce all ranks before closing.
- **Errors.** Errors come from TBCCL's structured result codes, never from parsing strings, and keep rank, peer and operation. Blocking calls release the GIL.
- **Peer death.** A dead peer surfaces as a structured transport error on the surviving rank within about a second; the runner fails, exo reports an error chunk to the client, and the instance can be deleted. Cancellation never fails the instance.
- **No hidden copies.** Any payload-sized copy the adapter makes is counted in `CopyStats` and visible in trace mode; the zero-copy paths (Metal, CUDA) must report 0.
- **Bootstrap.** A bootstrap that cannot complete ends at its timeout with the ranks it was still waiting for.

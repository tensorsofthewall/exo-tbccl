# Asynchronous decode sends (Phase 54)

TBCCL submission is nonblocking since Phase 52, but `TbcclPipelineComm.send` waited for the Work immediately. With `EXO_TBCCL_ASYNC_SEND=1`
(`FastPathConfig.async_send`) a decode send submits, keeps `{Work, Borrow}` in the pending table and returns the original array at once. Default: synchronous.
Prefill's grouped `flush_sends` (submit all, then wait as a group) is unchanged.

## Ownership and lifetime

The existing `Transfer` (Work + Borrows) in the `_pending` set is the only ownership mechanism; an async send is a `Transfer` with `detached=True`. The array, its uint view and
the DLPack export stay alive until the Work is terminal, even if the caller drops the returned array (`test_dropped_send_result_keeps_storage_alive_until_terminal`).

## Reaping and errors

- Every communication entry point first runs a nonblocking `_reap()` (Work `test()`), releasing finished transfers. The pending table therefore stays at the actual
  communication depth (tests: <= 3 over 300 decode steps at N=2/3/4; `async_send_submitted == async_send_reaped`).
- A failed detached send is never dropped: its structured error is held and raised at the next communication point (`_reap()` at entry of send/recv/all_gather/barrier/any_true).
- After every collective (`all_gather`, `barrier`, `any_true`) `_drain_async_sends()` waits (GIL released) for any still-outstanding detached send and raises a held failure. A
  send error therefore surfaces no later than the next synchronizing pipeline point; it cannot stay hidden across requests.
- `close()` is unchanged: abort only if work is in flight, drain, release borrows, destroy.

## Ordering with collectives (decode = send, then all_gather)

Submission order on each rank is the program order: rank r submits `send(r+1)` then `all_gather`; rank r+1 submits `recv(r)` then its own `all_gather`. Async only removes
the sender's *wait* between the two submissions; the descriptor order per rank is identical to the synchronous run, so per-lane FIFO and collective descriptor order are
unchanged. Verified: N=2/3/4 chains (recv -> compute -> send -> all_gather, 300 iterations, changing token dimension, float32/bfloat16) produce bit-identical gathered output
to the synchronous run. An intentionally invalid order (rank 0 send-then-all_gather, rank 1 all_gather-then-recv) is *detected* by TBCCL
(`TbcclProtocolMismatchError`), not silently corrupted (`test_invalid_p2p_collective_order_is_detected_or_correct`).

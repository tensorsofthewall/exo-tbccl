"""Worker for managed-memory fast-path tests and benchmarks: GPU-produced varying payloads, immediate GPU consumers, exact verification."""

from __future__ import annotations

import time

ADV = "127.0.0.1"


def make(mx, dt, n, i, salt):
    return ((mx.arange(n) * (salt + 1) + i * 7) % 251).astype(dt)


def pingpong(rank, world, ex, env, dtype, nbytes, iters, direction="both", verify=True):
    """Alternating GPU-produced sends in both directions. Receiver consumes the destination on the GPU at once and checks the result."""
    import os

    os.environ.update(env)
    import mlx.core as mx
    import numpy as np

    from exo_tbccl.group import TbcclPipelineComm

    dt = getattr(mx, dtype)
    n = nbytes // dt.size
    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=30000)
    consume_us, recv_us, send_us, rtt_us = [], [], [], []
    try:
        peer = 1 - rank
        for i in range(iters):
            ti = time.perf_counter()
            for sender in (0, 1):
                if direction == "0to1" and sender == 1:
                    continue
                if sender == rank:
                    x = make(mx, dt, n, i, sender)  # produced on the GPU, evaluated by borrow()
                    t0 = time.perf_counter()
                    comm.send(x, peer)
                    send_us.append((time.perf_counter() - t0) * 1e6)
                else:
                    dest = mx.zeros((n,), dtype=dt)
                    mx.eval(dest)
                    t0 = time.perf_counter()
                    comm.wait(comm.recv_into_async(dest, peer))
                    t1 = time.perf_counter()
                    y = dest.astype(mx.float32) * 2 + 1  # GPU consumer of the CPU/socket-written storage
                    mx.eval(y)
                    t2 = time.perf_counter()
                    if not verify:
                        recv_us.append((t1 - t0) * 1e6)
                        consume_us.append((t2 - t1) * 1e6)
                        continue
                    exp = make(mx, dt, n, i, sender).astype(mx.float32) * 2 + 1
                    mx.eval(exp)
                    assert bool(mx.array_equal(y, exp)), f"GPU consumer mismatch iter {i} {dtype} {nbytes}"
                    host = np.array(dest.view(mx.uint8 if dt.size == 1 else {2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[dt.size]))
                    expb = np.array(make(mx, dt, n, i, sender).view({1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[dt.size]))
                    assert host.tobytes() == expb.tobytes(), f"byte mismatch iter {i}"
                    recv_us.append((t1 - t0) * 1e6)
                    consume_us.append((t2 - t1) * 1e6)
            rtt_us.append((time.perf_counter() - ti) * 1e6)
        comm.barrier()
        s = comm.stats
        return {"labels": dict(s.direct_ops), "copies": (s.materialized_copies, s.staged_fallback_copies), "recv_us": recv_us, "consume_us": consume_us, "send_us": send_us, "rtt_us": rtt_us}
    finally:
        comm.close()

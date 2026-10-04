"""Loopback ping-pong: raw C ABI binding vs TbcclPipelineComm (MLX arrays), plus a per-stage breakdown of the bridge. No physical link.

    python benchmarks/bridge_overhead.py [--iters 300]

Prints one JSON object. Round trip = rank0 send -> rank1 recv -> rank1 send -> rank0 recv; times are per round trip in microseconds (median
and p90). The bridge breakdown is measured on rank 0's send/recv_like calls: export (evaluate + same-width view + DLPack consume), submit
(the native Send call), wait (Work wait), alloc (recv_like's zeros + eval).
"""

import argparse
import ctypes
import json
import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("/benchmarks", 1)[0])
from tests.harness import run_world  # noqa: E402

SIZES = [2 * 1024, 8 * 1024, 64 * 1024, 1024 * 1024]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def worker(rank, world, ex, iters):
    import mlx.core as mx

    from exo_tbccl import bridge
    from exo_tbccl._loader import native
    from exo_tbccl.group import TbcclPipelineComm

    comm = TbcclPipelineComm.create(rank, world, ex, bind_host="127.0.0.1", advertise_host="127.0.0.1", timeout_ms=30000)
    nat = comm._comm
    peer = 1 - rank
    out = {}
    try:
        for size in SIZES:
            warm = 30
            # --- raw binding on host buffers (ctypes), the C-ABI floor
            sbuf = (ctypes.c_uint8 * size)()
            rbuf = (ctypes.c_uint8 * size)()
            sp, rp = ctypes.addressof(sbuf), ctypes.addressof(rbuf)
            raw = []
            for i in range(iters + warm):
                t0 = time.perf_counter()
                if rank == 0:
                    nat.send(sp, size, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                    nat.recv(rp, size, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                else:
                    nat.recv(rp, size, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                    nat.send(sp, size, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                if i >= warm:
                    raw.append((time.perf_counter() - t0) * 1e6)
            # --- native binding on the MLX array's own storage, borrowed once (the TBCCL memory-provider cost without any per-op bridge work)
            x = mx.zeros((size // 2,), dtype=mx.bfloat16)
            mx.eval(x)
            y0 = mx.zeros((size // 2,), dtype=mx.bfloat16)
            mx.eval(y0)
            bs, br = bridge.borrow(x, comm.stats), bridge.borrow(y0, comm.stats, writable=True)
            nat_dev = []
            for i in range(iters + warm):
                t0 = time.perf_counter()
                if rank == 0:
                    nat.send(bs.ptr, bs.nbytes, bs.kind, bs.device, peer).wait()
                    nat.recv(br.ptr, br.nbytes, br.kind, br.device, peer).wait()
                else:
                    nat.recv(br.ptr, br.nbytes, br.kind, br.device, peer).wait()
                    nat.send(bs.ptr, bs.nbytes, bs.kind, bs.device, peer).wait()
                if i >= warm:
                    nat_dev.append((time.perf_counter() - t0) * 1e6)
            as_host = []
            if bs.kind != native.TBCCL_MEMORY_HOST:  # diagnostic only: the same managed/shared storage declared as plain host memory
                for i in range(iters + warm):
                    t0 = time.perf_counter()
                    if rank == 0:
                        nat.send(bs.ptr, bs.nbytes, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                        nat.recv(br.ptr, br.nbytes, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                    else:
                        nat.recv(br.ptr, br.nbytes, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                        nat.send(bs.ptr, bs.nbytes, native.TBCCL_MEMORY_HOST, -1, peer).wait()
                    if i >= warm:
                        as_host.append((time.perf_counter() - t0) * 1e6)
            label = bs.label
            bs.release()
            br.release()
            # --- TbcclPipelineComm on MLX arrays
            via = []
            stages = {"export": [], "submit": [], "wait": [], "alloc": []}
            for i in range(iters + warm):
                t0 = time.perf_counter()
                if rank == 0:
                    t = time.perf_counter()
                    b = bridge.borrow(x, comm.stats)
                    t1 = time.perf_counter()
                    work = nat.send(b.ptr, b.nbytes, b.kind, b.device, peer)
                    t2 = time.perf_counter()
                    work.wait()
                    b.release()
                    t3 = time.perf_counter()
                    y = comm.recv_like(x, peer)
                    t4 = time.perf_counter()
                    if i >= warm:
                        stages["export"].append((t1 - t) * 1e6)
                        stages["submit"].append((t2 - t1) * 1e6)
                        stages["wait"].append((t3 - t2) * 1e6)
                        stages["alloc"].append(0.0)
                else:
                    y = comm.recv_like(x, peer)
                    comm.send(x, peer)
                if i >= warm:
                    via.append((time.perf_counter() - t0) * 1e6)
            # recv_like's allocation cost alone (mx.zeros + eval)
            allocs = []
            for _ in range(iters):
                t = time.perf_counter()
                z = mx.zeros((size // 2,), dtype=mx.bfloat16)
                mx.eval(z)
                allocs.append((time.perf_counter() - t) * 1e6)
            out[str(size)] = {
                "raw_rtt_us": [statistics.median(raw), q(raw, 0.9)],
                "native_on_mlx_storage_rtt_us": [statistics.median(nat_dev), q(nat_dev, 0.9)],
                "same_storage_declared_host_rtt_us": [statistics.median(as_host), q(as_host, 0.9)] if as_host else None,
                "path": label,
                "pipelinecomm_rtt_us": [statistics.median(via), q(via, 0.9)],
                "bridge_export_us": statistics.median(stages["export"]) if stages["export"] else None,
                "native_submit_us": statistics.median(stages["submit"]) if stages["submit"] else None,
                "work_wait_us": statistics.median(stages["wait"]) if stages["wait"] else None,
                "recv_alloc_eval_us": statistics.median(allocs),
            }
        ex("quiesce", b"")
        out["device"] = str(mx.default_device())
    finally:
        comm.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    res = run_world(2, worker, a.iters, timeout=900)
    print(json.dumps(res[0]))


if __name__ == "__main__":
    main()

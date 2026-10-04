"""Two-host fast-path probe (the latency-attribution work): TbcclPipelineComm between Linux CUDA and Mac Metal over the TB4 link, one rank per host.

    python examples/two_host_fastpath.py --rank R --host <my TB ip> --peer <peer TB ip> [--port 29555] [--modes baseline,managed,recv,async,combined]
                                         [--sizes 2048,8192,16384] [--iters 150] [--out results.json]

Rank 0 listens on --host:--port for the bootstrap exchange (the application's all-gather: only opaque endpoint bytes cross it), rank 1 connects to
--peer:--port. For every mode both ranks build a fresh communicator and run, per dtype/size: (1) a ping-pong of GPU-produced, per-iteration varying
payloads, verified bit for bit and consumed on the GPU at once, and (2) a decode chain (send -> all_gather -> step_complete) checking every gathered
row. Payloads are small (<= 64 KiB) and iteration counts modest; this is a correctness-first probe, not a throughput test. Mac ranks must run inside a
live ssh session. Check AER before/after (docs/mac_thunderbolt_access.md).
"""

import argparse
import json
import socket
import statistics
import struct
import sys
import time

import mlx.core as mx

from exo_tbccl.config import FastPathConfig
from exo_tbccl.group import TbcclPipelineComm

MODES = {
    "baseline": FastPathConfig(),
    "managed": FastPathConfig(managed_mode="auto"),
    "recv": FastPathConfig(recv_reuse=True),
    "managed+recv": FastPathConfig(managed_mode="auto", recv_reuse=True),
    "async": FastPathConfig(async_send=True),
    "combined": FastPathConfig(managed_mode="auto", recv_reuse=True, async_send=True),
}


def _recvn(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("control channel closed")
        buf += chunk
    return buf


class TcpExchange:
    def __init__(self, rank, host, peer, port, timeout_s=120):
        if rank == 0:
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            srv.settimeout(timeout_s)
            self.sock, _ = srv.accept()
            srv.close()
        else:
            deadline = time.time() + timeout_s
            while True:
                try:
                    self.sock = socket.create_connection((peer, port), timeout=5)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.5)
        self.sock.settimeout(timeout_s)
        self.rank = rank

    def __call__(self, purpose, payload):
        frame = struct.pack("!I", len(payload)) + payload
        if self.rank == 0:
            theirs = _recvn(self.sock, struct.unpack("!I", _recvn(self.sock, 4))[0])
            self.sock.sendall(frame)
            return [payload, theirs]
        self.sock.sendall(frame)
        theirs = _recvn(self.sock, struct.unpack("!I", _recvn(self.sock, 4))[0])
        return [theirs, payload]


def produce(dt, n, i, sender):
    """GPU-produced, per-iteration varying payload whose exact values exist in every dtype."""
    return ((mx.arange(n) * (sender + 1) + i * 7) % 251).astype(dt)


def u8(a):
    return a.reshape(-1).view(mx.uint8)


def pingpong(comm, rank, dt, nbytes, iters, verify=True):
    n = nbytes // dt.size
    peer = 1 - rank
    rtt, consume = [], []
    for i in range(iters):
        t0 = time.perf_counter()
        for sender in (0, 1):
            if sender == rank:
                comm.send(produce(dt, n, i, sender), peer)
            else:
                dest = comm.recv_like(mx.zeros((n,), dtype=dt), peer)
                tc = time.perf_counter()
                y = dest.astype(mx.float32) * 2 + 1  # immediate GPU consumer
                mx.eval(y)
                consume.append((time.perf_counter() - tc) * 1e6)
                if verify:
                    exp = produce(dt, n, i, sender)
                    assert bool(mx.array_equal(u8(dest), u8(exp))), f"byte mismatch iter {i} {dt} {nbytes}"
                    assert bool(mx.array_equal(y, exp.astype(mx.float32) * 2 + 1)), f"GPU consumer mismatch iter {i}"
                comm.step_complete()
        rtt.append((time.perf_counter() - t0) * 1e6)
    comm.barrier()
    return rtt, consume


def decode_chain(comm, rank, dt, nbytes, iters):
    """Rank 0 produces, sends to rank 1; rank 1 computes and replies by all_gather only (the decode shape of exo's pipeline)."""
    n = nbytes // dt.size
    t = []
    for i in range(iters):
        t0 = time.perf_counter()
        if rank == 0:
            x = produce(dt, n, i, 0)
            comm.send(x, 1)
            y = x
        else:
            x = comm.recv_like(mx.zeros((n,), dtype=dt), 0)
            y = ((x.astype(mx.float32) * 3 + 1) % 251).astype(dt)
            mx.eval(y)
            comm.step_complete()
        g = comm.all_gather(y)
        mx.eval(g)
        exp0 = produce(dt, n, i, 0)
        exp1 = ((exp0.astype(mx.float32) * 3 + 1) % 251).astype(dt)
        assert bool(mx.array_equal(u8(g[:n]), u8(exp0))) and bool(mx.array_equal(u8(g[n:]), u8(exp1))), f"gather mismatch iter {i}"
        t.append((time.perf_counter() - t0) * 1e6)
    comm.barrier()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=29555)
    ap.add_argument("--modes", default="baseline,managed,recv,async,combined")
    ap.add_argument("--sizes", default="2048,8192,16384")
    ap.add_argument("--dtypes", default="bfloat16")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ex = TcpExchange(a.rank, a.host, a.peer, a.port)
    results = {"device": str(mx.default_device()), "rank": a.rank, "modes": {}}
    for mode in a.modes.split(","):
        comm = TbcclPipelineComm.create(a.rank, 2, ex, bind_host=a.host, advertise_host=a.host, timeout_ms=60000, config=MODES[mode])
        m = results["modes"][mode] = {}
        try:
            for dtn in a.dtypes.split(","):
                dt = getattr(mx, dtn)
                for size in (int(x) for x in a.sizes.split(",")):
                    pingpong(comm, a.rank, dt, size, 20)  # warm-up (verified)
                    rtt, cons = pingpong(comm, a.rank, dt, size, a.iters)
                    ch = decode_chain(comm, a.rank, dt, size, a.iters)
                    m[f"{dtn}/{size}"] = {
                        "pingpong_rtt_us_median": round(statistics.median(rtt), 1),
                        "first_gpu_consume_us_median": round(statistics.median(cons), 1),
                        "decode_chain_us_median": round(statistics.median(ch), 1),
                        "verified_iters": a.iters,
                    }
            s = comm.stats
            m["stats"] = {"labels": dict(s.direct_ops), "copies": [s.materialized_copies, s.staged_fallback_copies], "async": [s.async_send_submitted, s.async_send_reaped],
                          "pool": {"hits": comm.pool.stats.hits, "misses": comm.pool.stats.misses, "peak_bytes": comm.pool.stats.peak_bytes}, "pending_end": len(comm._pending)}
            print(mode, json.dumps(m), flush=True)
        finally:
            comm.close()
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

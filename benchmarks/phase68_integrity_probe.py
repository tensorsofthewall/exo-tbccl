"""Bit-exact transfer-integrity probe for the exact tensors a 5120-wide exo pipeline moves, between two ranks (loopback or Linux CUDA <-> Mac Metal over TB4).

    python benchmarks/phase68_integrity_probe.py --rank R --host <my ip> --peer <peer ip> [--port 29700] [--rows 1,138,2048] [--iters 1000,100,10] [--jitter-ms 15] [--out FILE]

Why: in the real-model runs the first generated token flips between two bf16 candidates whose logits tie (a one-ulp difference), more often with default TBCCL than with Ring or the
STEP policy. A one-ulp perturbation could be ordinary run-to-run noise or a transfer-visibility/ordering fault (a stale element). This probe isolates the transport: payloads are
produced ON THE DEVICE by integer arithmetic (exact in bf16, identical on Metal and CUDA), the receiver rebuilds the expected tensor on ITS device and compares it bit for bit
(uint16 view) immediately after `recv_like` returns (consumed on the GPU at once), with random compute-like gaps between operations (the Mac is left in whatever scheduling state the
default configuration gives it: no activity policy). Patterns: (1) ping-pong with the send posted before the peer posts its receive and the receive evaluated immediately,
(2) the exo decode chain `send -> all_gather -> step_complete` checking the received row and every gathered row, (3) queued prefill sends (all chunks submitted, then received).
A mismatch is reported with its count and first position; any mismatch is a transport-integrity failure.
"""
import argparse
import json
import random
import socket
import struct
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/examples")
from two_host_fastpath import TcpExchange  # noqa: E402

from exo_tbccl.group import TbcclPipelineComm  # noqa: E402

HIDDEN = 5120


def payload(rows, it, rank):
    n = rows * HIDDEN
    v = (mx.arange(n, dtype=mx.int32) * 7 + it * 13 + rank * 101 + rows) % 251
    return v.astype(mx.bfloat16).reshape(1, rows, HIDDEN)


def same(a, b):
    """Bit-exact comparison evaluated on the device; returns (ok, mismatching elements)."""
    ne = mx.sum(a.view(mx.uint16) != b.view(mx.uint16))
    return int(ne.item()) == 0, int(ne.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=29700)
    ap.add_argument("--rows", default="1,138,2048")
    ap.add_argument("--iters", default="1000,100,10")
    ap.add_argument("--jitter-ms", type=float, default=15.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows_l, iters_l = [int(x) for x in a.rows.split(",")], [int(x) for x in a.iters.split(",")]
    ex = TcpExchange(a.rank, a.host, a.peer, a.port, timeout_s=600)
    comm = TbcclPipelineComm.create(a.rank, 2, ex, bind_host=a.host, advertise_host=a.host, timeout_ms=600000)
    rng = random.Random(1234 + a.rank)
    peer = 1 - a.rank
    res = {"rank": a.rank, "host": socket.gethostname(), "patterns": {}}

    def gap():
        if a.jitter_ms > 0:
            time.sleep(rng.random() * a.jitter_ms / 1e3)

    bad_total = 0
    try:
        for rows, iters in zip(rows_l, iters_l):
            nbytes = rows * HIDDEN * 2
            stats = {"bytes": nbytes, "iters": iters, "pingpong_bad": 0, "chain_bad": 0, "chain_gather_bad": 0, "queued_bad": 0, "elements_bad": 0, "first_bad_iter": None}
            for it in range(iters):  # (1) ping-pong, rank 0 first
                mine, theirs = payload(rows, it, a.rank), payload(rows, it, peer)
                mx.eval(mine, theirs)
                gap()
                if a.rank == 0:
                    comm.send(mine, peer)
                    got = comm.recv_like(mine, peer)
                else:
                    got = comm.recv_like(mine, peer)
                    comm.send(mine, peer)
                ok, n = same(got, theirs)
                if not ok:
                    stats["pingpong_bad"] += 1
                    stats["elements_bad"] += n
                    stats["first_bad_iter"] = stats["first_bad_iter"] if stats["first_bad_iter"] is not None else it
            if rows <= 138:
                for it in range(iters):  # (2) exo decode chain: rank 0 sends, both all_gather, rank 1 receives then gathers
                    mine, theirs = payload(rows, 10_000 + it, a.rank), payload(rows, 10_000 + it, peer)
                    mx.eval(mine, theirs)
                    gap()
                    if a.rank == 0:
                        comm.send(mine, 1)
                        got = None
                    else:
                        got = comm.recv_like(mine, 0)
                        ok, n = same(got, theirs)
                        stats["chain_bad"] += 0 if ok else 1
                        stats["elements_bad"] += n
                    comm.step_complete()
                    g = comm.all_gather(mine.reshape(rows, HIDDEN))
                    expect = mx.concatenate([payload(rows, 10_000 + it, r).reshape(rows, HIDDEN) for r in range(2)], axis=0)
                    ok, n = same(g, expect)
                    stats["chain_gather_bad"] += 0 if ok else 1
                    stats["elements_bad"] += n
            if rows >= 138:
                k = 4
                for it in range(max(1, iters // 4)):  # (3) queued prefill sends: submit k chunks back to back, receive them in order
                    chunks = [payload(rows, 20_000 + it * k + j, a.rank) for j in range(k)]
                    wants = [payload(rows, 20_000 + it * k + j, peer) for j in range(k)]
                    mx.eval(*chunks, *wants)
                    gap()
                    if a.rank == 0:
                        ts = [comm.send_async(c, peer) for c in chunks]
                        outs = [comm.recv_like(chunks[0], peer) for _ in range(k)]
                        comm.wait_all(ts)
                    else:
                        outs = [comm.recv_like(chunks[0], peer) for _ in range(k)]
                        ts = [comm.send_async(c, peer) for c in chunks]
                        comm.wait_all(ts)
                    for o, w in zip(outs, wants):
                        ok, n = same(o, w)
                        stats["queued_bad"] += 0 if ok else 1
                        stats["elements_bad"] += n
            bad = stats["pingpong_bad"] + stats["chain_bad"] + stats["chain_gather_bad"] + stats["queued_bad"]
            bad_total += bad
            res["patterns"][f"rows{rows}"] = stats
            print(json.dumps({"rank": a.rank, **stats}), flush=True)
        comm.barrier()
        res["copies"] = comm.stats.materialized_copies
        res["bad_total"] = bad_total
    finally:
        comm.close()
    print("RESULT", json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()

"""Decode-chain latency over exo's MlxRing between two hosts (same chain as two_host_fastpath.decode_chain, for attributing the MlxRing/MlxTbccl gap).

    python examples/two_host_ring_chain.py --rank R --ips <rank0 ip>,<rank1 ip> [--sizes 2048,16384] [--iters 150]
"""

import argparse
import json
import os
import statistics
import tempfile
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--ips", required=True)
    ap.add_argument("--sizes", default="2048,16384")
    ap.add_argument("--iters", type=int, default=150)
    a = ap.parse_args()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([f"{ip}:{29830 + i}" for i, ip in enumerate(a.ips.split(","))], f)
    os.environ["MLX_HOSTFILE"], os.environ["MLX_RANK"] = f.name, str(a.rank)
    from exo.worker.engines.mlx.pipeline_comm import MlxPipelineComm

    comm = MlxPipelineComm(mx.distributed.init(backend="ring", strict=True))
    dt = mx.bfloat16
    out = {}
    for size in (int(x) for x in a.sizes.split(",")):
        n = size // 2
        ts = []
        for i in range(a.iters + 20):
            t0 = time.perf_counter()
            if a.rank == 0:
                x = ((mx.arange(n) + i * 7) % 251).astype(dt)
                mx.eval(x)
                mx.eval(comm.send(x, 1))
                y = x
            else:
                x = comm.recv_like(mx.zeros((n,), dtype=dt), 0)
                mx.eval(x)
                y = ((x.astype(mx.float32) * 3 + 1) % 251).astype(dt)
                mx.eval(y)
            g = comm.all_gather(y)
            mx.eval(g)
            if i >= 20:
                ts.append((time.perf_counter() - t0) * 1e6)
        out[size] = round(statistics.median(ts), 1)
    mx.eval(comm.all_gather(mx.zeros((1,))))
    print("RING_CHAIN_US", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

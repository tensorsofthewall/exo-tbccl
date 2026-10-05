"""One rank of the real-model pipeline across two hosts (Linux CUDA <-> Mac Metal), local Qwen3-0.6B-8bit through exo's pipeline_auto_parallel.

    python benchmarks/real_model_two_host.py --rank R --host <my TB ip> --peer <peer TB ip> --split <layers on rank 0> --prompt medium --tokens 48 --chunk 512

Run one per host (Mac inside a live ssh session; fast-path modes via EXO_TBCCL_* env on each host). Prints one JSON line. Rank 0 computes the unsharded
greedy reference on its own device; compare rank 0's and rank 1's `all_tokens` and the reference afterwards (cross-device numerics can legitimately
diverge late in a long generation; the first tokens must agree).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, HERE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=29556)
    ap.add_argument("--split", type=int, default=21)
    ap.add_argument("--prompt", default="medium")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--backend", default="tbccl", choices=["tbccl", "ring", "null"])
    ap.add_argument("--ring-ips", default="192.168.3.2,192.168.3.1", help="rank-ordered TB ips for the ring hostfile")
    a = ap.parse_args()
    from real_model_loopback import worker
    from two_host_fastpath import TcpExchange

    ex = None
    if a.backend == "ring":
        import tempfile

        ips = a.ring_ips.split(",")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([f"{ip}:{29810 + i}" for i, ip in enumerate(ips)], f)
        os.environ["MLX_HOSTFILE"], os.environ["MLX_RANK"] = f.name, str(a.rank)
    else:
        ex = TcpExchange(a.rank, a.host, a.peer, a.port, timeout_s=1800)
    res = worker(a.rank, 2, ex, {}, a.split, a.prompt, a.tokens, a.chunk, a.reps, a.host, a.backend)
    print("RESULT", json.dumps(res))

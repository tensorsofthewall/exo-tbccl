"""Externally written storage -> first GPU consumer, on Metal (Mac) or CUDA (Linux) — cost AND correctness.

    python benchmarks/recv_visibility.py [--iters 1000] [--rounds 3] [--host 127.0.0.1]

Two ranks over loopback. Rank 1 sends a deterministic bfloat16 payload (a 1x1024 ramp keyed by the iteration); rank 0 receives it into an MLX array and
immediately runs a GPU computation that consumes it (y = x.astype(float32) * 2 + 1), then compares y with the exact expected value EVERY iteration
(a stale or half-written read cannot pass: the payload differs per iteration and per element).

Variants (rank 0), timed from the moment the receive returned:
  fresh+eval   current path: recv_like (fresh mx.zeros destination) then exo's post-receive mx.eval(x), then the consumer
  fresh        the same without the post-receive mx.eval (the consumer's own eval does the work)
  pool         receive-buffer reuse (EXO_TBCCL_RECV=reuse semantics) then the consumer
  direct       NO TBCCL: a host memmove into a fresh evaluated MLX array's storage (borrowed writable), then the consumer
  gpu          control: the same array written by a GPU computation (mx.full) and evaluated, then the consumer
Per iteration three spans: receive_return -> graph built (host), graph built -> mx.eval(y) returned (GPU consumer incl. any pending work), total.
"""
import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("/benchmarks", 1)[0])
from tests.harness import run_world  # noqa: E402

N = 1024


def payload(i):
    import numpy as np

    return ((i * 7 + np.arange(N)) % 253).astype(np.float32)


def worker(rank, world, ex, iters, rounds, host):
    import ctypes

    import mlx.core as mx
    import numpy as np

    from exo_tbccl import bridge
    from exo_tbccl.config import FastPathConfig
    from exo_tbccl.group import TbcclPipelineComm

    comms = {
        "fresh+eval": TbcclPipelineComm.create(rank, world, ex, bind_host=host, advertise_host=host, timeout_ms=60000),
        "pool": TbcclPipelineComm.create(rank, world, ex, bind_host=host, advertise_host=host, timeout_ms=60000, config=FastPathConfig(recv_reuse=True)),
    }
    comms["fresh"] = comms["fresh+eval"]
    order = ["fresh+eval", "fresh", "pool", "direct", "gpu"]
    out = {v: {"recv_return_to_graph_us": [], "consumer_us": [], "since_recv_us": [], "bad": 0} for v in order}
    template = mx.zeros((1, N), dtype=mx.bfloat16)
    mx.eval(template)
    stats = bridge.CopyStats()
    try:
        for rnd in range(rounds):
            for v in (order if rnd % 2 == 0 else order[::-1]):
                comm = comms.get(v)
                for it in range(iters + 20):
                    i = rnd * 100000 + it
                    exp = payload(i) * 2 + 1
                    if rank == 1:
                        if comm is not None:
                            x = mx.array(payload(i)).astype(mx.bfloat16).reshape(1, N)
                            mx.eval(x)
                            comm.send(x, 0)
                            if v == "pool":
                                comm.step_complete()
                        continue
                    if v in ("fresh+eval", "fresh", "pool"):
                        x = comm.recv_like(template, 1)
                        t_recv = time.perf_counter_ns()
                        if v == "fresh+eval":
                            mx.eval(x)
                    elif v == "direct":
                        x = mx.zeros((1, N), dtype=mx.bfloat16)
                        mx.eval(x)
                        host_bf16 = mx.array(payload(i)).astype(mx.bfloat16)
                        hb = np.asarray(host_bf16.view(mx.uint16)).tobytes()  # host-side copy of the bytes to write
                        b = bridge.borrow(x, stats, writable=True)
                        ctypes.memmove(b.ptr, hb, len(hb))
                        b.release()
                        t_recv = time.perf_counter_ns()
                    else:
                        x = mx.array(payload(i)).astype(mx.bfloat16).reshape(1, N)
                        mx.eval(x)
                        t_recv = time.perf_counter_ns()
                    y = x.astype(mx.float32) * 2.0 + 1.0
                    t_graph = time.perf_counter_ns()
                    mx.eval(y)
                    t_done = time.perf_counter_ns()
                    ok = np.array_equal(np.array(y).reshape(-1), exp)
                    if v == "pool":
                        comm.step_complete()
                    if it >= 20:
                        out[v]["recv_return_to_graph_us"].append((t_graph - t_recv) / 1000)
                        out[v]["consumer_us"].append((t_done - t_graph) / 1000)
                        out[v]["since_recv_us"].append((t_done - t_recv) / 1000)
                        out[v]["bad"] += 0 if ok else 1
        if rank == 1:
            return None
        return out
    finally:
        for c in {id(c): c for c in comms.values()}.values():
            c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    res = run_world(2, worker, a.iters, a.rounds, a.host, timeout=1800)[0]
    import mlx.core as mx

    print(f"{mx.default_device()}: {a.iters} iterations x {a.rounds} rounds per variant; microseconds, median (p25..p75); every iteration's GPU result checked exactly")
    summary = {}
    for v, d in res.items():
        row = {}
        for k in ("recv_return_to_graph_us", "consumer_us", "since_recv_us"):
            s = sorted(d[k])
            row[k] = (statistics.median(s), s[len(s) // 4], s[3 * len(s) // 4])
        row["bad"] = d["bad"]
        row["n"] = len(d["consumer_us"])
        summary[v] = row
        print(f"  {v:11s} graph {row['recv_return_to_graph_us'][0]:7.1f}   consumer {row['consumer_us'][0]:7.1f} ({row['consumer_us'][1]:.1f}..{row['consumer_us'][2]:.1f})   "
              f"since-recv {row['since_recv_us'][0]:7.1f}   wrong results {row['bad']}/{row['n']}")
    print("JSON", json.dumps(summary))


if __name__ == "__main__":
    main()

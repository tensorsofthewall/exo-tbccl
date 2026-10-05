"""Phase 59 control: the Mac side of orientation B (rank 0) with a SYNTHETIC Metal stage instead of Qwen, against the same remote-peer emulator.

    python benchmarks/synthetic_mac_stage.py --backend tbccl|ring --stage-ms 2 --sampler-ms 2 --tokens 48 --host 127.0.0.1 --peer 127.0.0.1 --port N --out PREFIX

Per decode step it does exactly the pipeline's rank-0 sequence with no model, no KV cache and no sampler graph: [sampler-like GPU chain] [graph build] [stage-like GPU
chain] step_complete send(1,1,1024) all_gather(1,1,1024), over the real communication backend, so that "Metal compute separated by communication gaps" can be
compared with and without Qwen's graph structure. The chains are 1024x1024 bfloat16 matmuls sized to the requested duration (calibrated at start).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import mlx.core as mx  # noqa: E402

from sync_recorder import SyncRecorder  # noqa: E402

HIDDEN = 1024


def sampler_eval(y):
    mx.eval(y)


def stage_eval(y):
    mx.eval(y)


def do_send(comm, x, dst):
    out = comm.send(x, dst)
    mx.eval(out)
    return out


def do_gather(comm, x):
    g = comm.all_gather(x)
    mx.eval(g)
    return g


def chain(ws, x, n):
    y = x
    for i in range(n):
        y = mx.tanh(y @ ws[i % len(ws)])
    return y


def calibrate(ws, x, target_ms):
    """Number of chained matmuls whose WARM eval takes about target_ms (the cold GPU will take longer; that is the point of the experiment)."""
    for n in range(1, 400):
        t = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            mx.eval(chain(ws, x, n))
            t.append((time.perf_counter_ns() - t0) / 1e6)
        if sorted(t)[2] >= target_ms:
            return n
    return 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tbccl", "ring"], required=True)
    ap.add_argument("--stage-ms", type=float, default=2.0)
    ap.add_argument("--sampler-ms", type=float, default=2.0)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--peer", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29900)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=577)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ws = [(mx.random.normal((HIDDEN, HIDDEN)) * 0.03).astype(mx.bfloat16) for _ in range(8)]
    x0 = mx.random.normal((1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(ws, x0)
    n_stage, n_samp = calibrate(ws, x0, a.stage_ms), calibrate(ws, x0, a.sampler_ms)
    from remote_peer_emulator import make_comm

    class Args:
        pass

    ar = Args()
    ar.backend, ar.host, ar.peer, ar.port = a.backend, a.host, a.peer, a.port
    comm = make_comm(ar, 0)
    rec = SyncRecorder(0, a.backend)
    comm = rec.install(comm)
    body = a.prompt_tokens - 1
    out = mx.zeros((1, 1, HIDDEN), dtype=mx.bfloat16)
    try:
        for i in range(0, body, a.chunk):
            n = min(a.chunk, body - i)
            time.sleep(0.15)
            comm.flush_sends([(mx.zeros((1, n, HIDDEN), dtype=mx.bfloat16), 1)])
        comm.phase = "decode"
        rec.mark_decode_start()
        t_all0 = time.perf_counter_ns()
        x = out.reshape(1, HIDDEN)
        for step in range(a.tokens + 1):
            y = chain(ws, x0, n_samp)  # sampler-like: runs right after the previous all_gather
            if step > 0:
                sampler_eval(y)
            z = chain(ws, y if step > 0 else x0, n_stage)
            stage_eval(z)
            comm.step_complete()
            do_send(comm, z.reshape(1, 1, HIDDEN), 1)
            do_gather(comm, z.reshape(1, 1, HIDDEN))
        total = (time.perf_counter_ns() - t_all0) / 1e6
        comm.barrier()
    finally:
        rec.dump(f"{a.out}.rank0.json")
        rec.uninstall()
        comm.close() if hasattr(comm, "close") else None
    print("SYNTH", json.dumps({"backend": a.backend, "stage_layers": n_stage, "sampler_layers": n_samp, "tpot_ms": total / (a.tokens + 1)}))


if __name__ == "__main__":
    main()

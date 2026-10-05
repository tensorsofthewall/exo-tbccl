"""The cross-host timeline work hypothesis check: does a stage computation run slower right after the host thread BLOCKED (idle) than after it SPUN for the same time?

    python benchmarks/idle_vs_spin_compute.py [--layers 7] [--iters 300] [--gaps-ms 1,3,5]

For each gap it alternates, per iteration, three ways of spending the gap before the same GPU computation (a chain of 1024x1024 bfloat16 matmuls on a 1x1024
activation, a decode stage's shape): `sleep` (time.sleep: the thread is descheduled, cores may drop to a low-power state), `block` (the thread blocks in
recv() on a socket that a helper thread feeds after the gap: what TbcclPipelineComm's native wait does), `spin` (a busy loop: what MlxRing's non-blocking
socket worker does while a transfer is pending, on a different thread but on the same machine) and `none` (back-to-back). Reports the median
eval time and the median host graph-build time per mode. The question is only whether the modes differ; absolute numbers are machine specific.
"""
import argparse
import socket
import statistics
import threading
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--gaps-ms", default="1,3,5")
    a = ap.parse_args()
    ws = [(mx.random.normal((1024, 1024)) * 0.03).astype(mx.bfloat16) for _ in range(a.layers)]
    mx.eval(ws)
    x0 = mx.random.normal((1, 1024)).astype(mx.bfloat16)
    mx.eval(x0)
    s1, s2 = socket.socketpair()

    def feeder(gap_s, stop):
        while not stop.is_set():
            ev.wait()
            ev.clear()
            if stop.is_set():
                return
            time.sleep(gap_s)
            s2.send(b"x")

    def compute(i):
        t0 = time.perf_counter_ns()
        y = x0 + mx.array(i % 5, dtype=mx.bfloat16)
        for w in ws:
            y = mx.tanh(y @ w)
        t1 = time.perf_counter_ns()
        mx.eval(y)
        t2 = time.perf_counter_ns()
        return (t1 - t0) / 1000, (t2 - t1) / 1000

    print(f"{mx.default_device()}, {a.layers} layers, {a.iters} iterations per mode; microseconds median (p25..p75): host graph build | eval")
    for gap_ms in [float(g) for g in a.gaps_ms.split(",")]:
        res = {m: ([], []) for m in ("none", "sleep", "block", "spin")}
        ev, stop = threading.Event(), threading.Event()
        th = threading.Thread(target=feeder, args=(gap_ms / 1000, stop), daemon=True)
        th.start()
        for it in range(a.iters + 20):
            for mode in ("none", "sleep", "block", "spin"):
                if mode == "sleep":
                    time.sleep(gap_ms / 1000)
                elif mode == "block":
                    ev.set()
                    s1.recv(1)
                elif mode == "spin":
                    end = time.perf_counter_ns() + int(gap_ms * 1e6)
                    while time.perf_counter_ns() < end:
                        pass
                b, e = compute(it)
                if it >= 20:
                    res[mode][0].append(b)
                    res[mode][1].append(e)
        stop.set(); ev.set()
        for mode, (bs, es) in res.items():
            f = lambda v: f"{statistics.median(v):7.0f} ({sorted(v)[len(v)//4]:.0f}..{sorted(v)[3*len(v)//4]:.0f})"
            print(f"  gap {gap_ms:4.1f} ms  {mode:6s} build {f(bs)} | eval {f(es)}")


if __name__ == "__main__":
    main()

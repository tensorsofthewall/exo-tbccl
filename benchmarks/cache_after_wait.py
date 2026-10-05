"""Phase 59 mechanism test: is host code slower after a fully idle wait only when it touches a lot of memory (cold caches / power-gated cluster), and does a spinning
thread elsewhere in the process prevent it?

Phase 58's cpu_after_wait.py timed a tiny CPU-bound loop and saw no effect. The emulator runs show the PYTHON graph build of Qwen's seven transformer blocks going from
~114 us (MlxRing) to ~785 us (TbcclPipelineComm blocking wait), and back to ~117 us with a spinner. A graph build touches many Python/C++ objects; this times three
workloads right after the same wait modes: `tiny` (a small arithmetic loop), `touch` (a pointer-chasing walk over a few MB of Python objects) and `graph` (building
~300 lazy MLX ops, no eval).
  none | block | block+spinner (a separate thread burning CPU while this one is blocked) | spin (this thread busy-waits the gap)
    python benchmarks/cache_after_wait.py [--gap-ms 3] [--iters 400]
"""
import argparse
import os
import random
import socket
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402

from activity_thread import Activity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-ms", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    nodes = [[i, None] for i in range(40000)]
    order = list(range(len(nodes)))
    random.Random(1).shuffle(order)
    for i, j in zip(order, order[1:] + order[:1]):
        nodes[i][1] = nodes[j]
    ws = [(mx.random.normal((64, 64)) * 0.1) for _ in range(8)]
    x0 = mx.random.normal((1, 64))
    mx.eval(ws, x0)

    def tiny():
        s = 0
        for i in range(30000):
            s += i * i
        return s

    def touch():
        n = nodes[0]
        for _ in range(40000):
            n = n[1]
        return n

    def graph():
        y = x0
        for k in range(300):
            y = mx.tanh(y @ ws[k % 8]) + 0.01
        return y

    s1, s2 = socket.socketpair()
    ev, stop = threading.Event(), threading.Event()
    gap = a.gap_ms / 1000

    def feeder():
        while not stop.is_set():
            ev.wait(); ev.clear()
            if stop.is_set():
                return
            time.sleep(gap); s2.send(b"x")

    threading.Thread(target=feeder, daemon=True).start()
    act = Activity("comm")
    modes = ("none", "block", "block+spinner", "spin")
    work = {"tiny": tiny, "touch": touch, "graph": graph}
    res = {(m, w): [] for m in modes for w in work}
    for it in range(a.iters + 20):
        for m in modes:
            for w, fn in work.items():
                if m == "block":
                    ev.set(); s1.recv(1)
                elif m == "block+spinner":
                    act.comm_begin(); ev.set(); s1.recv(1); act.comm_end()
                elif m == "spin":
                    end = time.perf_counter_ns() + int(a.gap_ms * 1e6)
                    while time.perf_counter_ns() < end:
                        pass
                t0 = time.perf_counter_ns()
                fn()
                d = (time.perf_counter_ns() - t0) / 1000
                if it >= 20:
                    res[(m, w)].append(d)
    stop.set(); ev.set(); act.close()
    print(f"gap {a.gap_ms} ms, {a.iters} iterations; microseconds (median) of each workload right after the wait")
    print(f"  {'mode':14}" + "".join(f"{w:>10}" for w in work))
    for m in modes:
        print(f"  {m:14}" + "".join(f"{statistics.median(res[(m, w)]):10.0f}" for w in work))


if __name__ == "__main__":
    main()

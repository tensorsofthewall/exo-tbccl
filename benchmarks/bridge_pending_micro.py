"""Phase 57 microbenchmark: what does borrow() cost when its input is PENDING (not yet evaluated) vs already evaluated, split into
mx.eval / the same-width uint view (+ its eval) / __dlpack__ (native Export) / the whole borrow().

    python benchmarks/bridge_pending_micro.py [--iters 2000] [--layers 8] [--dtype bfloat16]

Each iteration builds a fresh GPU computation (a chain of `layers` 1024x1024 matmuls on a 1x1024 activation, the shape of a decode step's stage) so
nothing is cached, then takes ONE of the measurements below on that computation (rotating so every variant sees equal numbers of cold and warm GPU).
Compare TOTAL producer-to-send-ready time (compute call -> borrow returned), not the borrow alone: a pending borrow absorbs the GPU wait that a
preceding mx.eval would have absorbed.
"""
import argparse
import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("/benchmarks", 1)[0])
import mlx.core as mx  # noqa: E402

from exo_tbccl import bridge  # noqa: E402
from exo_tbccl import _native as native  # noqa: E402


def us(a, b):
    return (b - a) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    dt = getattr(mx, a.dtype)
    ws = [mx.random.normal((1024, 1024)).astype(dt) * 0.03 for _ in range(a.layers)]
    mx.eval(ws)
    x0 = mx.random.normal((1, 1024)).astype(dt)
    mx.eval(x0)
    uint = {2: mx.uint16, 4: mx.uint32}[x0.dtype.size]
    stats = bridge.CopyStats()

    def compute(i):
        y = x0 + mx.array(i % 7, dtype=dt)  # a fresh graph each iteration
        for w in ws:
            y = mx.tanh(y @ w)
        return y

    res = {k: [] for k in (
        "A eval(pending)", "A' compute->eval done",
        "B borrow(pending)", "B' compute->borrow done",
        "C1 eval(pending) then borrow(evaluated)", "C1' compute->borrow done",
        "D view(uint) of evaluated", "D eval(view) of evaluated", "E __dlpack__ via native.Export (view evaluated)", "F borrow(evaluated) total",
        "G pending view: eval(view)", "H null: eval of an already evaluated array",
    )}
    for it in range(a.iters + 50):
        phase = it % 5
        t0 = time.perf_counter_ns()
        y = compute(it)
        if phase == 0:  # A: explicit eval of the pending result
            t1 = time.perf_counter_ns(); mx.eval(y); t2 = time.perf_counter_ns()
            rec = {"A eval(pending)": us(t1, t2), "A' compute->eval done": us(t0, t2)}
        elif phase == 1:  # B: borrow the pending result directly
            t1 = time.perf_counter_ns(); b = bridge.borrow(y, stats); t2 = time.perf_counter_ns(); b.release()
            rec = {"B borrow(pending)": us(t1, t2), "B' compute->borrow done": us(t0, t2)}
        elif phase == 2:  # C1: eval, then borrow the evaluated array
            mx.eval(y); t1 = time.perf_counter_ns(); b = bridge.borrow(y, stats); t2 = time.perf_counter_ns(); b.release()
            rec = {"C1 eval(pending) then borrow(evaluated)": us(t1, t2), "C1' compute->borrow done": us(t0, t2)}
        elif phase == 3:  # D/E/F on an evaluated array, pieces separately
            mx.eval(y)
            t1 = time.perf_counter_ns(); v = y.view(uint); t2 = time.perf_counter_ns(); mx.eval(v); t3 = time.perf_counter_ns()
            e = native.Export(v); t4 = time.perf_counter_ns(); e.release()
            t5 = time.perf_counter_ns(); b = bridge.borrow(y, stats); t6 = time.perf_counter_ns(); b.release()
            t7 = time.perf_counter_ns(); mx.eval(y); t8 = time.perf_counter_ns()
            rec = {"D view(uint) of evaluated": us(t1, t2), "D eval(view) of evaluated": us(t2, t3), "E __dlpack__ via native.Export (view evaluated)": us(t3, t4),
                   "F borrow(evaluated) total": us(t5, t6), "H null: eval of an already evaluated array": us(t7, t8)}
        else:  # G: pending view: view first, eval the view (graph = compute + view)
            v = y.view(uint); t1 = time.perf_counter_ns(); mx.eval(v); t2 = time.perf_counter_ns()
            rec = {"G pending view: eval(view)": us(t1, t2)}
        if it >= 50:
            for k, v_ in rec.items():
                res[k].append(v_)
    print(f"{a.layers} matmul layers, {a.dtype}, {a.iters} iterations, {mx.default_device()}; microseconds, median (p25..p75)")
    for k, v in res.items():
        if v:
            s = sorted(v)
            print(f"  {k:48s} {statistics.median(v):9.1f}   ({s[len(s)//4]:.1f}..{s[3*len(s)//4]:.1f})   n={len(v)}")


if __name__ == "__main__":
    main()

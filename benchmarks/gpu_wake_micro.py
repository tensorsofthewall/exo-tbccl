"""Does a small GPU operation issued DURING an idle period make the next real stage computation slower (Metal)?

Under TbcclPipelineComm the Mac issues several tiny GPU evals around each communication (fresh destination zeros+eval, three same-width views) while the GPU
is otherwise idle waiting for the peer; MlxRing issues none. Physical traces showed the Mac's big evals 1.5x slower in the TBCCL runs. Sequences, each followed
by the same stage-like computation (a chain of 1024x1024 bfloat16 matmuls), compared on total time and on the stage's own eval time:
  idle           sleep for the full gap, then the stage                                       (ring-like)
  idle+tiny      sleep gap/2, tiny evals (zeros + 3 views), sleep gap/2, the stage            (tbccl-like)
  idle+tiny-cpu  the same tiny evals allocated on the CPU stream (zeros only; the views stay on the GPU)
  idle+external-input  sleep the gap, then a stage whose input is a fresh borrowed buffer written by the host (the TBCCL receive destination)
  busy           the stage back-to-back with the previous one                                 (GPU kept awake)
    python benchmarks/gpu_wake_micro.py [--layers 7] [--gap-ms 3] [--iters 300]
"""
import argparse
import statistics
import time

import mlx.core as mx
import sys
sys.path.insert(0, __file__.rsplit('/benchmarks', 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument("--gap-ms", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    ws = [(mx.random.normal((1024, 1024)) * 0.03).astype(mx.bfloat16) for _ in range(a.layers)]
    x0 = mx.random.normal((1, 1024)).astype(mx.bfloat16)
    mx.eval(ws, x0)

    import ctypes

    from exo_tbccl import bridge

    stats = bridge.CopyStats()
    host = (x0 * 1.0).view(mx.uint16)
    mx.eval(host)
    host_bytes = bytes(memoryview(__import__("numpy").array(host)))

    def external(i):
        """A TbcclPipelineComm-style receive destination: fresh zeros, evaluated, bytes written by the host through the borrowed pointer."""
        d = mx.zeros((1, 1024), dtype=mx.bfloat16)
        mx.eval(d)
        b = bridge.borrow(d, stats, writable=True)
        ctypes.memmove(b.ptr, host_bytes, len(host_bytes))
        b.release()
        return d

    def stage(i, ext=False):
        y = (external(i) if ext else x0) + mx.array(i % 5, dtype=mx.bfloat16)
        for w in ws:
            y = mx.tanh(y @ w)
        t = time.perf_counter_ns()
        mx.eval(y)
        return (time.perf_counter_ns() - t) / 1000

    def tiny(cpu):
        z = mx.zeros((1, 1024), dtype=mx.bfloat16, stream=mx.cpu) if cpu else mx.zeros((1, 1024), dtype=mx.bfloat16)
        mx.eval(z)
        for _ in range(3):
            v = x0.view(mx.uint16)
            mx.eval(v)

    modes = ("idle", "idle+tiny", "idle+tiny-cpu", "idle+external-input", "busy")
    res = {m: ([], []) for m in modes}
    g = a.gap_ms / 1000
    for it in range(a.iters + 20):
        for m in modes:
            t0 = time.perf_counter_ns()
            if m == "idle":
                time.sleep(g)
            elif m in ("idle+tiny", "idle+tiny-cpu"):
                time.sleep(g / 2); tiny(m.endswith("cpu")); time.sleep(g / 2)
            if m == "idle+external-input":
                time.sleep(g)
            e = stage(it, ext=(m == "idle+external-input"))
            tot = (time.perf_counter_ns() - t0) / 1000
            if it >= 20:
                res[m][0].append(e)
                res[m][1].append(tot)
    print(f"{mx.default_device()}, {a.layers} layers, gap {a.gap_ms} ms, {a.iters} iterations; microseconds median (p25..p75)")
    for m, (e, t) in res.items():
        f = lambda v: f"{statistics.median(v):7.0f} ({sorted(v)[len(v)//4]:.0f}..{sorted(v)[3*len(v)//4]:.0f})"
        print(f"  {m:14s} stage eval {f(e)}   gap+tiny+stage total {f(t)}")


if __name__ == "__main__":
    main()

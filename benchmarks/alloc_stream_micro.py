"""The per-token timeline work candidate preview: cost of the receive-destination allocation (mx.zeros + mx.eval) and of the borrow's same-width view + eval on the GPU stream
versus the CPU stream. Arrays have no device in MLX; a stream only decides where an op runs, so a destination created on the CPU stream is ordinary
unified/managed storage that any GPU op may read afterwards (verified below by running a GPU consumer on it and comparing the result).

    python benchmarks/alloc_stream_micro.py [--iters 3000] [--nbytes 2048]
"""
import argparse
import statistics
import time

import mlx.core as mx


def med(xs):
    s = sorted(xs)
    return f"{statistics.median(s):7.1f} ({s[len(s)//4]:.1f}..{s[3*len(s)//4]:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--nbytes", type=int, default=2048)
    a = ap.parse_args()
    n = a.nbytes // 2
    res = {k: [] for k in ("zeros+eval gpu", "zeros+eval cpu", "view+eval gpu (evaluated src)", "view+eval cpu (evaluated src)",
                           "TOTAL alloc(gpu)+consumer", "TOTAL alloc(cpu)+consumer", "consumer of gpu-allocated", "consumer of cpu-allocated")}
    bad = 0
    src = mx.random.normal((1, n)).astype(mx.bfloat16)
    mx.eval(src)
    for it in range(a.iters + 100):
        t0 = time.perf_counter_ns(); z = mx.zeros((1, n), dtype=mx.bfloat16); mx.eval(z); t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns(); zc = mx.zeros((1, n), dtype=mx.bfloat16, stream=mx.cpu); mx.eval(zc); t3 = time.perf_counter_ns()
        t4 = time.perf_counter_ns(); v = src.view(mx.uint16); mx.eval(v); t5 = time.perf_counter_ns()
        t6 = time.perf_counter_ns(); vc = src.view(mx.uint16, stream=mx.cpu); mx.eval(vc); t7 = time.perf_counter_ns()
        t8 = time.perf_counter_ns(); yg = z.astype(mx.float32) + 3.0; mx.eval(yg); t9 = time.perf_counter_ns()  # a GPU consumer of the GPU-stream allocation
        t10 = time.perf_counter_ns(); y = zc.astype(mx.float32) + 3.0; mx.eval(y); t11 = time.perf_counter_ns()  # ... and of the CPU-stream allocation
        bad += 0 if float(y.sum().item()) == 3.0 * n else 1
        bad += 0 if bool((vc == v).all().item()) else 1
        if it >= 100:
            for k, (x, y_) in zip(res, ((t0, t1), (t2, t3), (t4, t5), (t6, t7))):
                res[k].append((y_ - x) / 1000)
            res["TOTAL alloc(gpu)+consumer"].append(((t1 - t0) + (t9 - t8)) / 1000)
            res["TOTAL alloc(cpu)+consumer"].append(((t3 - t2) + (t11 - t10)) / 1000)
            res["consumer of gpu-allocated"].append((t9 - t8) / 1000)
            res["consumer of cpu-allocated"].append((t11 - t10) / 1000)
    print(f"{mx.default_device()}, {a.nbytes} B, {a.iters} iterations; microseconds median (p25..p75); GPU consumer of CPU-stream allocation wrong results: {bad}")
    for k, v in res.items():
        print(f"  {k:34s} {med(v)}")


if __name__ == "__main__":
    main()

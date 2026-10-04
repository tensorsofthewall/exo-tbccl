"""Phase 54 prefetch experiment: cost of CPU-writing an MLX CUDA array and consuming it on the GPU at once, with and without an explicit blocking
managed-memory prefetch toward the GPU in between. Single process, no TBCCL. Throwaway measurement tool; the package does not use prefetch.

    python benchmarks/prefetch_experiment.py
"""

import ctypes
import statistics
import time

import mlx.core as mx

cu = ctypes.CDLL("libcuda.so.1")
assert cu.cuInit(0) == 0
ctx = ctypes.c_void_p()
dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0)
assert cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0
assert cu.cuCtxSetCurrent(ctx) == 0


class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


def prefetch(ptr, n, to_device):
    loc = Loc(1 if to_device else 2, dev.value if to_device else 0)
    r = cu.cuMemPrefetchAsync_v2(ctypes.c_uint64(ptr), ctypes.c_size_t(n), loc, 0, None)
    if r != 0:
        raise RuntimeError(f"cuMemPrefetchAsync -> {r}")
    assert cu.cuStreamSynchronize(None) == 0


def trial(nbytes, mode, iters=200):
    from exo_tbccl._loader import native

    n = nbytes // 4
    out = []
    for i in range(iters + 20):
        a = mx.zeros((n,), dtype=mx.uint32)
        mx.eval(a)
        e = native.Export(a)
        ptr, nb = e.ptr, e.nbytes
        mx.eval(mx.sum(a))  # touch on the GPU so the pages are GPU-resident, as a receive destination that was just consumed would be
        t0 = time.perf_counter()
        if mode == "prefetch-cpu+gpu":
            prefetch(ptr, nb, False)
        ctypes.memset(ptr, 0x5A, nb)  # the CPU/socket write of a receive
        if mode in ("prefetch-gpu", "prefetch-cpu+gpu"):
            prefetch(ptr, nb, True)
        t1 = time.perf_counter()
        y = mx.sum(a)  # immediate GPU consumer
        mx.eval(y)
        t2 = time.perf_counter()
        e.release()
        if i >= 20:
            out.append(((t1 - t0) * 1e6, (t2 - t1) * 1e6))
    w = statistics.median(x[0] for x in out)
    c = statistics.median(x[1] for x in out)
    return round(w, 1), round(c, 1), round(w + c, 1)


if __name__ == "__main__":
    print("size  mode  (cpu_write_us, first_gpu_consume_us, total_us)")
    for nbytes in (2048, 8192, 16384, 65536, 1048576):
        for mode in ("none", "prefetch-gpu", "prefetch-cpu+gpu"):
            try:
                print(nbytes, mode, trial(nbytes, mode), flush=True)
            except RuntimeError as ex:
                print(nbytes, mode, "ERR", ex)

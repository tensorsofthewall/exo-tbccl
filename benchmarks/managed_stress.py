"""Forced managed host-direct stress (the latency-attribution work): thousands of GPU-produced, per-iteration varying payloads sent and received in host-direct mode, each received
buffer consumed by a GPU kernel at once and verified bit for bit, for FP32/FP16/BF16 at 2 KiB, 8 KiB, 64 KiB and 1 MiB.

    python benchmarks/managed_stress.py [iters-per-direction-per-config, default 1500]   (1500 -> 36,000 verified receives)
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/benchmarks", 1)[0])
from tests.harness import run_world  # noqa: E402
from tests.managed_worker import pingpong  # noqa: E402

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    total = 0
    for dt in ("float32", "float16", "bfloat16"):
        for nbytes in (2048, 8192, 65536, 1048576):
            t = time.time()
            res = run_world(2, pingpong, {"EXO_TBCCL_CUDA_MANAGED_MODE": "host"}, dt, nbytes, iters, timeout=900)
            n = sum(len(r["consume_us"]) for r in res)
            total += n
            print(dt, nbytes, "ok", n, "verified receives", res[0]["labels"], f"{time.time() - t:.1f}s", flush=True)
    print("TOTAL verified GPU-consumed receives:", total)

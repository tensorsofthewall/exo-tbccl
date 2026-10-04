"""Loopback A/B of managed-memory modes (Phase 54): per-direction send/recv/first-GPU-consumer times with GPU-produced varying data.

    python benchmarks/managed_fastpath.py [--iters 400] [--rounds 3] [--dtype bfloat16]

Modes are interleaved round by round. Prints medians in microseconds per mode and size. Correctness is asserted inside the worker (exact bytes and
GPU-consumer result on every iteration).
"""

import argparse
import json
import statistics
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/benchmarks", 1)[0])
from tests.harness import run_world  # noqa: E402
from tests.managed_worker import pingpong  # noqa: E402

SIZES = [2048, 8192, 65536, 1048576]
MODES = {
    "cuda": {"EXO_TBCCL_CUDA_MANAGED_MODE": "cuda"},
    "host-send": {"EXO_TBCCL_CUDA_MANAGED_MODE": "host", "EXO_TBCCL_MANAGED_DIRS": "send"},
    "host-recv": {"EXO_TBCCL_CUDA_MANAGED_MODE": "host", "EXO_TBCCL_MANAGED_DIRS": "recv"},
    "host-both": {"EXO_TBCCL_CUDA_MANAGED_MODE": "host", "EXO_TBCCL_MANAGED_DIRS": "send,recv"},
}


def temp():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]).split()[0])
    except Exception:  # noqa: BLE001
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--sizes", default="")
    a = ap.parse_args()
    acc = {}
    for rnd in range(a.rounds):
        order = list(MODES) if rnd % 2 == 0 else list(reversed(MODES))
        for size in ([int(x) for x in a.sizes.split(",")] if a.sizes else SIZES):
            for mode in order:
                res = run_world(2, pingpong, MODES[mode], a.dtype, size, a.iters, "both", False, timeout=600)
                d = acc.setdefault((mode, size), {"rtt": [], "send": [], "consume": [], "temp": []})
                for r in res:
                    d["send"] += r["send_us"][10:]
                    d["rtt"] += r["rtt_us"][10:] if r is res[0] else []
                    d["consume"] += r["consume_us"][10:]
                d["temp"].append(temp())
    out = {}
    for (mode, size), d in acc.items():
        out.setdefault(str(size), {})[mode] = {k: round(statistics.median(v), 1) for k, v in d.items() if k != "temp"} | {"gpu_temp": (min(d["temp"]), max(d["temp"]))}
    for size, modes in out.items():
        for mode, m in modes.items():
            print(f"{size:>8} {mode:10} rtt={m['rtt']:>8} send={m['send']:>8} first_gpu_consume={m['consume']:>8} temp={m['gpu_temp']}")


if __name__ == "__main__":
    main()

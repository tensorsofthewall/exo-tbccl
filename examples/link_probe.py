"""Two-host correctness probe for TbcclPipelineComm with real MLX arrays (Linux CUDA <-> Mac Metal). Correctness only, payloads <= 1 MiB.

    python examples/link_probe.py --rank R --world 2 --host <this host's reachable IP> --dir <exchange dir> [--timeout-s 60]

The application exchange here is files in --dir (`<purpose>.<rank>`): the operator (or a sync loop) copies them between hosts. Each rank
verifies every received payload against a deterministically regenerated reference, bit for bit (compared as raw bytes).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

from exo_tbccl.group import TbcclPipelineComm


def pattern(shape, dtype, seed):
    """Deterministic bytes as an MLX array of `dtype` (any bit pattern, NaNs included)."""
    nbytes = mx.array(1, dtype=dtype).nbytes
    n = 1
    for d in shape:
        n *= d
    raw = ((mx.arange(n * nbytes, dtype=mx.uint32) * 131 + seed * 17) % 251).astype(mx.uint8)
    return raw.reshape(*shape[:-1], shape[-1] * nbytes).view(dtype).reshape(shape)


def same_bytes(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(mx.array_equal(a.reshape(-1).view(mx.uint8), b.reshape(-1).view(mx.uint8)).item())


class FileExchange:
    def __init__(self, directory, rank, world, timeout_s):
        self.dir, self.rank, self.world, self.timeout_s = Path(directory), rank, world, timeout_s

    def __call__(self, purpose, payload):
        mine = self.dir / f"{purpose}.{self.rank}"
        tmp = self.dir / f".{purpose}.{self.rank}.tmp"
        tmp.write_bytes(payload)
        os.replace(tmp, mine)
        out = []
        deadline = time.time() + self.timeout_s
        for r in range(self.world):
            p = self.dir / f"{purpose}.{r}"
            while True:
                if p.exists() and p.stat().st_size == len(payload):
                    out.append(p.read_bytes())
                    break
                if time.time() > deadline:
                    raise TimeoutError(f"exchange {purpose}: no file from rank {r}")
                time.sleep(0.1)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--host", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--timeout-s", type=float, default=60)
    a = ap.parse_args()
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    rank, world = a.rank, a.world
    peer = 1 - rank
    comm = TbcclPipelineComm.create(rank, world, FileExchange(a.dir, rank, world, a.timeout_s), bind_host=a.host, advertise_host=a.host, timeout_ms=int(a.timeout_s * 1000))
    report = {"rank": rank, "device": str(mx.default_device()), "checks": {}}
    ok = True

    def check(name, cond):
        nonlocal ok
        report["checks"][name] = bool(cond)
        ok = ok and bool(cond)

    try:
        cases = [("f32_2KiB", (1, 512), mx.float32), ("bf16_32KiB", (8, 2048), mx.bfloat16), ("f16_512KiB", (256, 1024), mx.float16), ("f32_1MiB", (256, 1024), mx.float32)]
        for i, (name, shape, dt) in enumerate(cases):
            mine, theirs = pattern(shape, dt, 10 + rank * 100 + i), pattern(shape, dt, 10 + peer * 100 + i)
            mx.eval(mine, theirs)
            if rank == 0:
                comm.send(mine, peer)
                got = comm.recv_like(mine, peer)
            else:
                got = comm.recv_like(mine, peer)
                comm.send(mine, peer)
            mx.eval(got)
            check(f"p2p_{name}", same_bytes(got, theirs))
        for n in (7, 1001):
            mine, theirs = pattern((n,), mx.uint8, 900 + rank), pattern((n,), mx.uint8, 900 + peer)
            mx.eval(mine, theirs)
            if rank == 0:
                comm.send(mine, peer)
                got = comm.recv_like(mine, peer)
            else:
                got = comm.recv_like(mine, peer)
                comm.send(mine, peer)
            check(f"p2p_odd_{n}B", same_bytes(got, theirs))
        for i, dt in enumerate((mx.float32, mx.float16, mx.bfloat16)):
            mine = pattern((2, 64), dt, 50 + rank * 7 + i)
            g = comm.all_gather(mine)
            ref = mx.concatenate([pattern((2, 64), dt, 50 + r * 7 + i) for r in range(world)], axis=0)
            check(f"all_gather_{dt}", same_bytes(g, ref))
        comm.barrier()
        check("barrier", True)
        check("any_true", comm.any_true(rank == 1) and not comm.any_true(False))
        # prefill-style: queue 8 sends, flush (submit all, then wait), peer receives in order
        n = 64 * 1024
        sends = [(pattern((n,), mx.uint8, 300 + i), peer) for i in range(8)]
        if rank == 0:
            comm.flush_sends(sends)
            ok_flush = True
        else:
            ok_flush = all(same_bytes(comm.recv_like(sends[i][0], 0), pattern((n,), mx.uint8, 300 + i)) for i in range(8))
        check("flush_8x64KiB", ok_flush)
        report["stats"] = {"direct_ops": comm.stats.direct_ops, "direct_bytes": comm.stats.direct_bytes, "materialized_copies": comm.stats.materialized_copies, "staged_fallback_copies": comm.stats.staged_fallback_copies}
        check("no_adapter_copies", comm.stats.materialized_copies == 0 and comm.stats.staged_fallback_copies == 0)
        FileExchange(a.dir, rank, world, a.timeout_s)("quiesce", b"x")
    finally:
        comm.close()
    report["ok"] = ok
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

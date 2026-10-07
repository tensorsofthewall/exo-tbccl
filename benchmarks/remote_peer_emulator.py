"""A Mac-local emulator of the remote Linux pipeline peer, driven by the cross-host timeline physical traces.

    python benchmarks/remote_peer_emulator.py --orientation A --backend tbccl --profile benchmarks/emulator_profiles/profile_A_tbccl.json \
        --host 127.0.0.1 --peer 127.0.0.1 --port 29900 --out /tmp/emu

The real Mac stage (benchmarks/real_model_two_host.py, Qwen3-0.6B-8bit, the latency-attribution prompt) runs as the other rank over loopback with the SAME communication
backend: TbcclPipelineComm (the real native binding) or MlxPipelineComm over the real MlxRing. Nothing is replaced by queues, pipes or shared memory, so each
backend's own worker/thread behaviour on the Mac is preserved. The emulator itself does no model work and no GPU work (default device = CPU); it plays the
Linux rank's HOST timeline:

  orientation A (emulator = rank 0, Mac = rank 1 with 7 layers):
      prefill: send a (1,512,1024) and a (1,64,1024) bfloat16 chunk, then per decode step
      [wait resume+sampler+graph+compute+prep from the profile] step_complete  send (1,1,1024)  all_gather (1,1,1024)
  orientation B (emulator = rank 1, Mac = rank 0 with 7 layers):
      prefill: receive the two chunks, then per decode step
      [wait the profile's sampler+pre-recv delay] recv (1,1,1024) [wait first-use+compute+pre-gather] step_complete  all_gather (1,1,1024)

All delays are Linux-local intervals recorded from physical traces (the profiles in benchmarks/emulator_profiles), replayed with calibrated sleeps plus a short spin (macOS timers run ~1.5x
long: the cold-progress work). Requested and achieved delays are recorded for every step. The payloads are small deterministic bfloat16 values, so the Mac's tokens are NOT
those of the real run (the activations are not the real model's): the experiment measures timing and communication behaviour, not text.

Differences from the real Linux peer (also.md): the wire is loopback, the peer's CPU/GPU is the Mac's (idle except for the
timers and, with MlxRing, the ring worker's own busy-polling while a transfer is pending, exactly as on Linux but now competing for the Mac's cores), and no
Linux GPU work exists.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from sync_recorder import SyncRecorder  # noqa: E402

HIDDEN = 1024


def now():
    return time.perf_counter_ns()


class Clock:
    """Calibrated waiting: coarse sleep (scaled by the measured timer ratio) until ~300 us before the deadline, then spin."""

    def __init__(self):
        total = 0.0
        for _ in range(15):
            t0 = now()
            time.sleep(0.002)
            total += (now() - t0) / 2e6
        self.ratio = max(1.0, total / 15) if total / 15 > 1.15 else 1.0
        self.log: list[tuple[str, float, float]] = []

    def wait_until(self, deadline_ns: int) -> None:
        rem = deadline_ns - now()
        if os.environ.get("EMU_NOSPIN") and rem > 0:  # No busy tail (the emulator then leaves the P-cluster idle between events); timing error is a few hundred us
            time.sleep(rem / 1e9 / self.ratio)
            return
        if rem > 600_000:
            time.sleep(max(0.0, (rem - 400_000) / 1e9 / self.ratio))
        while now() < deadline_ns:
            pass


def payload(n, seed):
    base = (mx.arange(n * HIDDEN) % 251).astype(mx.float32) * 0.001 + float(seed % 7) * 0.01
    return base.reshape(1, n, HIDDEN).astype(mx.bfloat16)


# one mx.eval per helper: the recorder names them like the real pipeline's evals (sync_recorder.SITES)
def do_send(comm, x, dst):
    out = comm.send(x, dst)
    mx.eval(out)
    return out


def do_recv(comm, template, src):
    r = comm.recv_like(template, src)
    mx.eval(r)
    return r


def do_gather(comm, x):
    g = comm.all_gather(x)
    mx.eval(g)
    return g


def make_comm(args, rank):
    if args.backend == "ring":
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(["127.0.0.1:29810", "127.0.0.1:29811"], f)
        os.environ["MLX_HOSTFILE"], os.environ["MLX_RANK"] = f.name, str(rank)
        from exo.worker.engines.mlx.pipeline_comm import MlxPipelineComm

        return MlxPipelineComm(mx.distributed.init(backend="ring", strict=True))
    from two_host_fastpath import TcpExchange

    from exo_tbccl.group import TbcclPipelineComm

    ex = TcpExchange(rank, args.host, args.peer, args.port, timeout_s=1800)
    return TbcclPipelineComm.create(rank, 2, ex, bind_host=args.host, advertise_host=args.host, timeout_ms=1800000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orientation", choices=["A", "B"], required=True)
    ap.add_argument("--backend", choices=["tbccl", "ring"], required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--peer", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29900)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--prompt-tokens", type=int, default=577)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--prefill-ms", type=float, default=150.0)
    ap.add_argument("--advance-us", type=float, default=0.0, help="orientation A: shorten the emulated Linux stage compute by this much so its send is ready BEFORE the Mac posts its receive (the Mac stays the critical path; 0 = Phase 59 behaviour)")
    ap.add_argument("--out", required=True, help="prefix for <out>.rank<R>.json (the recorder dump) and <out>.fidelity.json")
    a = ap.parse_args()
    prof = json.load(open(a.profile))
    rank = prof["emulated_rank"]
    steps = prof["steps"]
    clock = Clock()
    comm = make_comm(a, rank)
    rec = SyncRecorder(rank, a.backend)
    comm = rec.install(comm)
    body = a.prompt_tokens - 1
    chunks = [min(a.chunk, body - i) for i in range(0, body, a.chunk)]
    fidelity, bad = [], 0
    peer = 1 - rank
    try:
        # ---- prefill (timing is not part of the decode comparison) ----
        for n in chunks:
            if rank == 0:
                clock.wait_until(now() + int(a.prefill_ms * 1e6))
                comm.flush_sends([(payload(n, 1), 1)])
            else:
                r = do_recv(comm, mx.zeros((1, n, HIDDEN), dtype=mx.bfloat16), 0)
                bad += 0 if tuple(r.shape) == (1, n, HIDDEN) else 1
                clock.wait_until(now() + int(a.prefill_ms * 1e6))
        comm.phase = "decode"
        t_ref = now()  # end of the previous gather (the last prefill step ends the same way)
        x = payload(1, 3)
        for i in range(a.tokens + 1):
            st = steps[i % len(steps)]
            if rank == 0:  # A
                req = [("resume", st["resume_us"]), ("sampler", st["sampler_us"]), ("graph", st["graph_us"]), ("compute", max(0.0, st["compute_us"] - a.advance_us)), ("prep", st["prep_us"])]
                t = t_ref
                for name, us in req:
                    t0 = t
                    t += int(us * 1000)
                    clock.wait_until(t)
                    ach = (now() - t0) / 1000.0
                    fidelity.append((name, us, ach))
                    if name == "sampler" and i > 0:
                        rec.add("eval", "real_model_loopback.py:worker#4", t0, now())
                    if name == "compute":
                        rec.add("eval", "model_output_eval", t0, now())
                comm.step_complete()
                do_send(comm, x, 1)
                g = do_gather(comm, x)
            else:  # B
                t = t_ref
                pre = int(st["post_us"] * 1000)
                clock.wait_until(t + pre)
                fidelity.append(("post-gather-to-recv-posted", st["post_us"], (now() - t) / 1000.0))
                if i > 0:
                    rec.add("eval", "real_model_loopback.py:worker#4", t + int(st["resume_us"] * 1000), t + int(st["resume_us"] * 1000) + int(st["sampler_us"] * 1000))
                r = do_recv(comm, mx.zeros((1, 1, HIDDEN), dtype=mx.bfloat16), 0)
                t_rc = now()
                bad += 0 if tuple(r.shape) == (1, 1, HIDDEN) and bool(mx.all(mx.isfinite(r.astype(mx.float32)))) else 1
                clock.wait_until(t_rc + int(st["compute_total_us"] * 1000))
                fidelity.append(("recv-complete-to-gather", st["compute_total_us"], (now() - t_rc) / 1000.0))
                rec.add("eval", "model_output_eval", t_rc + int(st["first_use_us"] * 1000), t_rc + int((st["first_use_us"] + st["compute_us"]) * 1000))
                comm.step_complete()
                g = do_gather(comm, x)
            bad += 0 if tuple(g.shape) == (2, 1, HIDDEN) else 1
            t_ref = now()
        comm.barrier()
    finally:
        rec.dump(f"{a.out}.rank{rank}.json")
        rec.uninstall()
        comm.close() if hasattr(comm, "close") else None
    errs = [abs(ach - req) for _, req, ach in fidelity]
    per = {}
    for name, req, ach in fidelity:
        per.setdefault(name, []).append((req, ach))
    summary = {"orientation": a.orientation, "backend": a.backend, "timer_ratio": clock.ratio, "steps": a.tokens + 1, "wrong_shape_or_nonfinite": bad,
               "median_abs_error_us": statistics.median(errs), "p95_abs_error_us": sorted(errs)[int(0.95 * len(errs))],
               "components": {k: {"requested_median_us": statistics.median(r for r, _ in v), "achieved_median_us": statistics.median(x for _, x in v),
                                  "median_error_us": statistics.median(x - r for r, x in v)} for k, v in per.items()}}
    json.dump(summary, open(f"{a.out}.fidelity.json", "w"), indent=1)
    print("EMULATOR", json.dumps(summary))


if __name__ == "__main__":
    main()

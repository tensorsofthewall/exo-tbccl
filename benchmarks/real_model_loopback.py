"""Real-model (local Qwen3-0.6B-8bit, never downloaded) pipeline over loopback TbcclPipelineComm through exo's pipeline_auto_parallel.

    python benchmarks/real_model_loopback.py --split 21 --prompt medium [--tokens 48]   (run with exo's venv; mode via EXO_TBCCL_* env)

Greedy decode on one rank unsharded gives the reference tokens; the two ranks then run the pipelined model (prefill with queued sends, decode with
the final all_gather) and must produce identical token ids. Also prints a digest of each rank's KV cache so modes can be compared bit for bit
(the receive-buffer poison control must not change tokens or cache state). Not a performance measurement: both ranks share one GPU.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, __file__.rsplit("/benchmarks", 1)[0])
from tests.harness import run_world  # noqa: E402

MODEL = os.path.expanduser("~/.exo_p53/local_models/Qwen3-0.6B-8bit")
TEXT = (
    "The history of distributed computing spans decades of work on how separate machines can cooperate. "
    "Early systems exchanged messages over slow serial lines; later ones built shared file systems, remote procedure calls, and eventually "
    "collective communication libraries that move tensors between accelerators. "
)


def worker(rank, world, ex, env, split, prompt_kind, ntok, chunk, reps_override=0, host="127.0.0.1", backend="tbccl"):
    os.environ.update(env)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from exo.shared.models.model_cards import ModelCard, ModelTask
    from exo.shared.types.backends import Backend
    from exo.shared.types.common import ModelId
    from exo.shared.types.memory import Memory
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.worker.engines.mlx.auto_parallel import (
        flush_prefill_sends,
        pipeline_auto_parallel,
        set_pipeline_prefill,
        set_pipeline_queue_sends,
    )
    from exo_tbccl.group import TbcclPipelineComm

    mx.set_cache_limit(128 << 20)  # two ranks and a reference share one small GPU
    model, tok = load(MODEL)
    reps = reps_override or {"short": 1, "medium": 12, "long": 150}[prompt_kind]
    prompt = tok.encode(TEXT * reps)
    p = mx.array(prompt)

    def greedy_reference():
        cache = make_prompt_cache(model)
        for i in range(0, p.size - 1, chunk):  # same chunking and split as the pipelined run (prefill the prompt without its last token, then decode it)
            mx.eval(model(p[:-1][i : i + chunk][None], cache=cache))
        logits = model(p[-1:].reshape(1, 1), cache=cache)
        out = []
        t = mx.argmax(logits[0, -1])
        for _ in range(ntok):
            mx.eval(t)
            out.append(int(t))
            logits = model(t.reshape(1, 1), cache=cache)
            t = mx.argmax(logits[0, -1])
        return out

    ref = greedy_reference() if rank == 0 else None  # one reference at a time: two unsharded long prefills do not fit an 8 GiB GPU
    mx.clear_cache()

    if backend == "ring":  # exo's MlxRing: the hostfile is a JSON list of "ip:port" in rank order (MLX_HOSTFILE / MLX_RANK already set)
        from exo.worker.engines.mlx.pipeline_comm import MlxPipelineComm

        comm = MlxPipelineComm(mx.distributed.init(backend="ring", strict=True))
    else:
        comm = TbcclPipelineComm.create(rank, world, ex, bind_host=host, advertise_host=host, timeout_ms=1800000)
    sync_prefix = os.environ.get("EXO_P57_SYNC")  # Phase 57: record every eval / communication call with a semantic label (path prefix); measurement only
    sync_rec = None
    if sync_prefix:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sync_recorder import SyncRecorder

        sync_rec = SyncRecorder(rank, backend)
        comm = sync_rec.install(comm)
    cadence = os.environ.get("EXO_P56_CADENCE")  # Phase 56: record the communication cadence (path prefix); measurement only
    if cadence:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cadence_recorder import CadenceRecorder

        comm = CadenceRecorder(comm, rank)
    try:
        n_layers = len(model.layers)
        bounds = [(0, split), (split, n_layers)]
        card = ModelCard(
            model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"), storage_size=Memory.from_kb(1), n_layers=n_layers, hidden_size=1024,
            supports_tensor=False, tasks=[ModelTask.TextGeneration], backends=[Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
        )
        shard = PipelineShardMetadata(model_card=card, device_rank=rank, world_size=world, start_layer=bounds[rank][0], end_layer=bounds[rank][1], n_layers=n_layers)
        gen = pipeline_auto_parallel(model, comm, shard)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            model = stop.value
        cache = make_prompt_cache(model)

        def prefix_digest(n):
            h = hashlib.sha256()
            for c in cache:
                if getattr(c, "keys", None) is not None:
                    for a in (c.keys[..., :n, :], c.values[..., :n, :]):
                        mx.eval(a)
                        h.update(bytes(memoryview(__import__("numpy").array(a.astype(mx.float32)))))
            return h.hexdigest()

        snaps = {}
        t0 = time.perf_counter()
        set_pipeline_prefill(model, True)
        set_pipeline_queue_sends(model, True)
        body = p[:-1]
        die = os.environ.get("EXO_P54_DIE_AT", "")  # "<rank>:prefill:<chunk index>" or "<rank>:decode:<step>" (failure-injection test)
        for i in range(0, body.size, chunk):
            if die == f"{rank}:prefill:{i // chunk}":
                os._exit(0)
            out = model(body[i : i + chunk][None], cache=cache)
            mx.eval(out)
            flush_prefill_sends()
            del out
            mx.clear_cache()
        set_pipeline_queue_sends(model, False)
        set_pipeline_prefill(model, False)
        if cadence or sync_rec:
            comm.phase = "decode"
        logits = model(p[-1:].reshape(1, 1), cache=cache)
        t = mx.argmax(logits[0, -1])
        ttft = time.perf_counter() - t0
        toks, step_s = [], []
        def resources():
            rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20 if os.path.exists("/proc/self/statm") else -1
            return {"rss_mb": round(rss, 1), "threads": __import__("threading").active_count(), "fds": len(os.listdir("/dev/fd")), "pending": len(getattr(comm, "_pending", ())),
                    "detached": len(getattr(comm, "_async_sends", ())), "pool_slots": comm.pool.slot_count if hasattr(comm, "pool") else 0, "pool_bytes": comm.pool.cached_bytes if hasattr(comm, "pool") else 0}

        samples = {}
        for k in range(ntok):
            if die == f"{rank}:decode:{k}":
                os._exit(0)
            if k in (200, ntok // 2, ntok - 1):
                samples[k] = resources()
            if k in (0, ntok // 2):
                n = next(c.offset for c in cache if getattr(c, "keys", None) is not None)
                snaps[n] = prefix_digest(n)
            ts = time.perf_counter()
            mx.eval(t)
            toks.append(int(t))
            logits = model(t.reshape(1, 1), cache=cache)
            t = mx.argmax(logits[0, -1])
            step_s.append(time.perf_counter() - ts)
        stable = all(prefix_digest(n) == d for n, d in snaps.items())  # earlier KV entries unchanged by everything that followed
        comm.barrier()
        if cadence:
            comm.dump(f"{cadence}.rank{rank}.json")
        if sync_rec:
            sync_rec.dump(f"{sync_prefix}.rank{rank}.json")
            sync_rec.uninstall()
        if backend == "ring":
            return {"match_ref": (toks == ref) if rank == 0 else None, "all_tokens": toks, "ref_tokens": ref, "tokens": toks[:8], "prompt_tokens": len(prompt),
                    "ttft_s": round(ttft, 3), "tpot_ms": round(1e3 * sorted(step_s)[len(step_s) // 2], 3), "kv_prefix_stable": stable,
                    "copies": (0, 0), "labels": {"ring": 1}, "pool": {}, "async": (0, 0), "pending_end": 0}
        s, ps = comm.stats, comm.pool.stats
        return {
            "match_ref": (toks == ref) if rank == 0 else None, "all_tokens": toks, "ref_tokens": ref, "tokens": toks[:8], "prompt_tokens": len(prompt), "ttft_s": round(ttft, 3),
            "tpot_ms": round(1e3 * sorted(step_s)[len(step_s) // 2], 3), "kv_prefix_stable": stable, "resources": samples,
            "copies": (s.materialized_copies, s.staged_fallback_copies), "labels": dict(s.direct_ops),
            "pool": {"hits": ps.hits, "misses": ps.misses, "peak_bytes": ps.peak_bytes, "untracked": ps.untracked, "evictions": ps.evictions},
            "async": (s.async_send_submitted, s.async_send_reaped), "pending_end": len(comm._pending),
        }
    finally:
        comm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=21)
    ap.add_argument("--prompt", default="medium")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=0)
    a = ap.parse_args()
    res = run_world(2, worker, {}, a.split, a.prompt, a.tokens, a.chunk, a.reps, timeout=1800)
    ref = res[0]["ref_tokens"]
    for r in res:
        r["match_ref"] = res[0]["match_ref"] and r["all_tokens"] == ref  # rank 1 is checked against rank 0's reference
        del r["all_tokens"], r["ref_tokens"]
    print(json.dumps(res))

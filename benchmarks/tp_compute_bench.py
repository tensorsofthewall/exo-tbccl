"""Phase 67: warm, synchronized per-part compute timings of the local Qwen3-0.6B-8bit for single-device and TP2-shard execution (measurement only; nothing is distributed).

    python benchmarks/tp_compute_bench.py --out docs/data/phase67/compute_<host>.json [--warmup 10 --iters 50] [--prefill-len 577]

Runs on whichever MLX device the host has (Metal on the Mac, CUDA on the RTX 3070 Ti: the same quantized kernels and dtypes as the pipeline runs). Every timing includes
building the lazy graph and `mx.eval` of the result (real completion, not an enqueue). The unit of measurement is exactly what a TP2 layer executes between two
collectives:
  attn(f)   input_layernorm -> q/k/v (column shard) -> q/k norm -> rope -> KV update -> attention core -> o_proj (row shard): the partial sum an AllReduce would reduce
  mlp(f)    post_attention_layernorm -> gate/up (column shard) -> swiglu -> down_proj (row shard): the partial sum an AllReduce would reduce
for a fraction f of the heads / intermediate columns (f = 1.0 is the unsharded layer half). Shards are slices of the real quantized weights (group-aligned), so the work
and memory traffic are the real ones; the outputs are not used. Also isolated op timings (QKV, attention core, o_proj, gate/up, activation, down; each its own eval, so
they include per-eval overhead and are an upper bound on the fused cost), the final norm + LM head for the vocabulary shard, and the full single-device decode step.
"""
import argparse
import json
import os
import socket
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

MODEL = os.path.expanduser("~/.exo_p53/local_models/Qwen3-0.6B-8bit")
KV_FRACTIONS = {0.25: 2, 0.375: 3, 0.5: 4, 0.625: 5, 0.75: 6, 1.0: 8}  # kv heads (of 8) per fraction of the attention work; q heads = 2 x kv heads


def stats(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"median_us": statistics.median(xs) * 1e6, "p25_us": q(0.25) * 1e6, "p75_us": q(0.75) * 1e6, "p95_us": q(0.95) * 1e6, "n": len(xs)}


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return stats(out)


def slice_q(lin, rows=None, cols=None, bits=8, gs=64):
    """A QuantizedLinear holding rows [r0:r1) of the output dim and/or columns [c0:c1) of the input dim (group aligned)."""
    w, s, b = lin.weight, lin.scales, lin.biases
    if rows is not None:
        w, s, b = w[rows[0]:rows[1]], s[rows[0]:rows[1]], b[rows[0]:rows[1]]
    if cols is not None:
        pack = 32 // bits
        w, s, b = w[:, cols[0] // pack:cols[1] // pack], s[:, cols[0] // gs:cols[1] // gs], b[:, cols[0] // gs:cols[1] // gs]
    out = nn.QuantizedLinear(gs, 1, bias=False, group_size=gs, bits=bits)
    out.weight, out.scales, out.biases = mx.contiguous(w), mx.contiguous(s), mx.contiguous(b)  # a column slice is a strided view; the CUDA quantized matmul needs dense row-major weights
    mx.eval(out.weight, out.scales, out.biases)
    return out


class Shard:
    """One transformer block's TP shard for a fraction of the attention heads (kv_heads of 8) and of the MLP inner dimension."""

    def __init__(self, blk, args, kv_heads, inner, kv_len, seq=1):
        a = blk.self_attn
        hd, self.kv_heads, self.q_heads = args.head_dim, kv_heads, 2 * kv_heads
        self.blk, self.seq, self.scale = blk, seq, hd ** -0.5
        # column shard of q/k/v (heads), row shard of o_proj (the same heads' input columns): first `heads` heads, a representative contiguous slice
        self.q = slice_q(a.q_proj, (0, self.q_heads * hd))
        self.k = slice_q(a.k_proj, (0, kv_heads * hd))
        self.v = slice_q(a.v_proj, (0, kv_heads * hd))
        self.o = slice_q(a.o_proj, cols=(0, self.q_heads * hd))
        self.gate = slice_q(blk.mlp.gate_proj, (0, inner))
        self.up = slice_q(blk.mlp.up_proj, (0, inner))
        self.down = slice_q(blk.mlp.down_proj, cols=(0, inner))
        self.cache = KVCache()
        self.cache.update_and_fetch(mx.random.normal((1, kv_heads, kv_len, hd)).astype(mx.bfloat16), mx.random.normal((1, kv_heads, kv_len, hd)).astype(mx.bfloat16))
        mx.eval(self.cache.keys, self.cache.values)
        self.kv_len = kv_len
        self.x = mx.random.normal((1, seq, args.hidden_size)).astype(mx.bfloat16)
        self.hd = hd
        self.mask = "causal" if seq > 1 else None
        mx.eval(self.x)

    def _qkv(self, h):
        B, L, _ = h.shape
        a = self.blk.self_attn
        q, k, v = self.q(h), self.k(h), self.v(h)
        q = a.q_norm(q.reshape(B, L, self.q_heads, -1)).transpose(0, 2, 1, 3)
        k = a.k_norm(k.reshape(B, L, self.kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.kv_heads, -1).transpose(0, 2, 1, 3)
        off = self.kv_len
        return a.rope(q, offset=off), a.rope(k, offset=off), v

    def attn(self):
        h = self.blk.input_layernorm(self.x)
        q, k, v = self._qkv(h)
        keys = mx.concatenate([self.cache.keys[..., :self.kv_len, :], k], axis=2)  # the cache update + fetch of a real decode/prefill step (no in-place state change)
        values = mx.concatenate([self.cache.values[..., :self.kv_len, :], v], axis=2)
        o = scaled_dot_product_attention(q, keys, values, cache=None, scale=self.scale, mask=self.mask)
        o = o.transpose(0, 2, 1, 3).reshape(self.x.shape[0], self.x.shape[1], -1)
        mx.eval(self.o(o))

    def mlp(self):
        h = self.blk.post_attention_layernorm(self.x)
        mx.eval(self.down(nn.silu(self.gate(h)) * self.up(h)))

    def ops(self, warmup, iters):
        """Isolated, individually evaluated ops (each includes one eval): an upper bound on their fused cost."""
        out = {}
        a = self.blk.self_attn
        h = self.blk.input_layernorm(self.x)
        mx.eval(h)
        out["rmsnorm"] = timeit(lambda: mx.eval(self.blk.input_layernorm(self.x)), warmup, iters)
        out["qkv_proj"] = timeit(lambda: mx.eval(self.q(h), self.k(h), self.v(h)), warmup, iters)
        q, k, v = self._qkv(h)
        keys = mx.concatenate([self.cache.keys[..., :self.kv_len, :], k], axis=2)
        values = mx.concatenate([self.cache.values[..., :self.kv_len, :], v], axis=2)
        mx.eval(q, keys, values)
        out["attn_core"] = timeit(lambda: mx.eval(scaled_dot_product_attention(q, keys, values, cache=None, scale=self.scale, mask=self.mask)), warmup, iters)
        o = scaled_dot_product_attention(q, keys, values, cache=None, scale=self.scale, mask=self.mask).transpose(0, 2, 1, 3).reshape(self.x.shape[0], self.x.shape[1], -1)
        mx.eval(o)
        out["o_proj"] = timeit(lambda: mx.eval(self.o(o)), warmup, iters)
        h2 = self.blk.post_attention_layernorm(self.x)
        mx.eval(h2)
        out["gate_up"] = timeit(lambda: mx.eval(self.gate(h2), self.up(h2)), warmup, iters)
        g, u = self.gate(h2), self.up(h2)
        mx.eval(g, u)
        out["activation"] = timeit(lambda: mx.eval(nn.silu(g) * u), warmup, iters)
        act = nn.silu(g) * u
        mx.eval(act)
        out["down_proj"] = timeit(lambda: mx.eval(self.down(act)), warmup, iters)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--kv-len", type=int, default=577)
    ap.add_argument("--prefill-len", type=int, default=577)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--activity", action="store_true", help="diagnostic control: keep the existing same-process activity helper (exo_tbccl.step_activity) running during the whole run (Mac slow-state check)")
    a = ap.parse_args()
    act = None
    if a.activity:
        from exo_tbccl.step_activity import StepActivity

        act = StepActivity(max_window_ms=3.6e6, duty=1.0)
        act.open()
    model, tok = load(MODEL)
    args = model.args if hasattr(model, "args") else model.model.args
    inner_full = args.intermediate_size
    res = {"activity_helper": bool(a.activity), "host": socket.gethostname(), "device": str(mx.default_device()), "mlx": mx.__version__, "kv_len": a.kv_len, "layer": a.layer, "decode": {}, "prefill": {}, "ops": {}}
    blk = model.model.layers[a.layer]
    for f, kvh in KV_FRACTIONS.items():
        inner = int(round(inner_full * (kvh / 8) / 64)) * 64
        sh = Shard(blk, args, kvh, inner, a.kv_len)
        res["decode"][str(f)] = {"kv_heads": kvh, "inner": inner, "attn": timeit(sh.attn, a.warmup, a.iters), "mlp": timeit(sh.mlp, a.warmup, a.iters)}
        print(f"decode f={f}: attn {res['decode'][str(f)]['attn']['median_us']:.0f} us, mlp {res['decode'][str(f)]['mlp']['median_us']:.0f} us", flush=True)
        if f in (0.5, 1.0):
            res["ops"][str(f)] = sh.ops(a.warmup, a.iters)
        del sh
    pw, pi = 3, 10
    for f, kvh in KV_FRACTIONS.items():
        inner = int(round(inner_full * (kvh / 8) / 64)) * 64
        sh = Shard(blk, args, kvh, inner, 0, seq=a.prefill_len)
        res["prefill"][str(f)] = {"attn": timeit(sh.attn, pw, pi), "mlp": timeit(sh.mlp, pw, pi)}
        print(f"prefill f={f}: attn {res['prefill'][str(f)]['attn']['median_us']:.0f} us, mlp {res['prefill'][str(f)]['mlp']['median_us']:.0f} us", flush=True)
        del sh
    # final norm + LM head (tied quantized embedding as a linear): full and half vocabulary shard
    emb = model.model.embed_tokens
    x = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    res["lm_head"] = {"full": timeit(lambda: mx.eval(emb.as_linear(model.model.norm(x))), a.warmup, a.iters)}
    half = nn.QuantizedLinear(64, 1, bias=False, group_size=64, bits=8)
    half.weight, half.scales, half.biases = (mx.contiguous(t[: args.vocab_size // 2]) for t in (emb.weight, emb.scales, emb.biases))
    mx.eval(half.weight, half.scales, half.biases)
    res["lm_head"]["half"] = timeit(lambda: mx.eval(half(model.model.norm(x))), a.warmup, a.iters)
    res["lm_head"]["argmax_half"] = timeit(lambda: mx.eval(mx.argmax(half(model.model.norm(x)), axis=-1)), a.warmup, a.iters)
    # full single-device decode step: 28 layers + head + greedy sample, KV length kv_len
    from mlx_lm.models.cache import make_prompt_cache

    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 60)[: a.kv_len]
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    mx.eval(model(mx.array([ids]), cache=cache))
    res["prefill_full_s"] = time.perf_counter() - t0  # includes first-use kernel compile; a second, warm prefill follows
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    mx.eval(model(mx.array([ids]), cache=cache))
    res["prefill_full_warm_s"] = time.perf_counter() - t0
    steps = []
    tokid = ids[-1]
    for _ in range(a.warmup + 40):
        t = time.perf_counter()
        y = mx.argmax(model(mx.array([[tokid]]), cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)
        tokid = int(y.item())
        steps.append(time.perf_counter() - t)
    res["decode_full_step"] = stats(steps[a.warmup:])
    print("full decode step median us", res["decode_full_step"]["median_us"], "warm prefill s", res["prefill_full_warm_s"], flush=True)
    if act is not None:
        act.shutdown()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

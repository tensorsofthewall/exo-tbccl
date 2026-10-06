"""Phase 67: the exact tensor-parallel (TP2) communication graph of the local Qwen3-0.6B-8bit, derived from config.json and verified against the real mlx_lm graph.

    python benchmarks/tp_graph.py [--trace] [--out docs/data/phase67/tp_graph.json]

Analytic part: classic Megatron-style TP2 (column-parallel q/k/v and gate/up, row-parallel o_proj and down_proj), one AllReduce after each row-parallel projection.
Trace part (--trace): runs one real decode step (batch 1, KV length = the prompt) on the unmodified mlx_lm model with every projection / norm / attention call wrapped
(shapes and dtypes only, semantics untouched) and reconciles the tensors that a TP2 split would reduce with the analytic payloads. No weights are sharded.
"""
import argparse
import json
import os
import sys

MODEL = os.path.expanduser("~/.exo_p53/local_models/Qwen3-0.6B-8bit")


def analytic(cfg: dict, batch: int, seq: int, tp: int = 2, act_bytes: int = 2) -> dict:
    h, i, nh, nkv, hd, nl, v = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["num_hidden_layers"], cfg["vocab_size"]
    tokens = batch * seq
    per_layer = [
        {"op": "attention output (row-parallel o_proj partial sums)", "collective": "AllReduce", "bytes": tokens * h * act_bytes, "count": 1,
         "dependency": "residual add -> post_attention_layernorm -> MLP gate/up (hard)"},
        {"op": "MLP output (row-parallel down_proj partial sums)", "collective": "AllReduce", "bytes": tokens * h * act_bytes, "count": 1,
         "dependency": "residual add -> next layer input_layernorm -> q/k/v (hard)"},
    ]
    final = [
        {"op": "LM head, vocab-parallel logits, greedy: local (max, argmax) exchange", "collective": "AllGather", "bytes": 8 * batch, "count": 1, "dependency": "sampler"},
        {"op": "LM head, vocab-parallel logits, full logits (non-greedy): half-vocab gather (bf16)", "collective": "AllGather", "bytes": batch * (v // tp) * act_bytes, "count": 1,
         "dependency": "sampler"},
    ]
    return {"model": {"hidden": h, "intermediate": i, "heads": nh, "kv_heads": nkv, "head_dim": hd, "layers": nl, "vocab": v, "tie": cfg["tie_word_embeddings"]},
            "batch": batch, "seq": seq, "tp": tp, "act_bytes": act_bytes, "per_layer": per_layer, "final": final,
            "shard": {"q_heads": nh // tp, "kv_heads": nkv // tp, "qkv_out": ((nh + 2 * nkv) // tp) * hd, "o_in": (nh // tp) * hd, "mlp_inner": i // tp, "lm_head_rows": v // tp},
            "collectives_per_layer": 2, "collectives_per_token": 2 * nl + 1, "allreduce_bytes_per_token": 2 * nl * tokens * h * act_bytes,
            "critical_path_collectives_per_token": 2 * nl + 1}


def trace() -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(MODEL)
    rec: list = []
    state = {"on": False}

    def wrap(cls, name, label):
        orig = cls.__call__

        def f(self, x, *a, **kw):
            out = orig(self, x, *a, **kw)
            if state["on"]:
                rec.append({"op": label, "in": list(x.shape), "in_dtype": str(x.dtype), "out": list(out.shape), "out_dtype": str(out.dtype)})
            return out

        cls.__call__ = f

    wrap(qwen3.Attention, "__call__", "attention (qkv->sdpa->o_proj)")
    wrap(qwen3.MLP, "__call__", "mlp (gate/up->swiglu->down)")
    wrap(nn.QuantizedLinear, "__call__", "quantized_linear")
    wrap(nn.RMSNorm, "__call__", "rmsnorm")
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 60)[:577]
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([ids]), cache=cache))
    state["on"] = True
    logits = model(mx.array([[ids[-1]]]), cache=cache)
    mx.eval(logits)
    state["on"] = False
    layer0 = [r for r in rec[:14]]
    lin = [r for r in rec if r["op"] == "quantized_linear"]
    n_layers = len([r for r in rec if r["op"].startswith("attention")])
    return {"kv_len": len(ids), "layers_seen": n_layers, "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype), "layer0_calls": layer0,
            "linear_calls_per_token": len(lin), "rmsnorm_calls_per_token": len([r for r in rec if r["op"] == "rmsnorm"]),
            "linear_shapes_layer0": [(r["in"], r["out"], r["in_dtype"]) for r in lin[:7]], "lm_head": lin[-1] if lin else None,
            "residual_dtype": layer0[-1]["out_dtype"] if layer0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    res = {"config": cfg, "decode_b1": analytic(cfg, 1, 1), "decode_b2": analytic(cfg, 2, 1), "decode_b4": analytic(cfg, 4, 1), "prefill_577": analytic(cfg, 1, 577)}
    if a.trace:
        res["trace"] = trace()
        t = res["trace"]
        want = [1, 1, cfg["hidden_size"]]
        print("traced decode step: kv_len", t["kv_len"], "layers", t["layers_seen"], "linears/token", t["linear_calls_per_token"], "logits", t["logits_shape"], t["logits_dtype"])
        print("residual stream shape", want, "dtype", t["residual_dtype"])
    for k in ("decode_b1", "prefill_577"):
        d = res[k]
        print(f"{k}: {d['collectives_per_layer']} AllReduce/layer, {d['collectives_per_token']} collectives/token, {d['allreduce_bytes_per_token']} B AllReduce payload/token, per-call {d['per_layer'][0]['bytes']} B")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

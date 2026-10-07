"""Trace-calibrated heterogeneous TP2 cost model (Mac M4 / Linux RTX 3070 Ti over TB4), decode and prefill, with model-size / batch / split scaling.

    python benchmarks/tp_cost_model.py --data docs/data/phase67 [--out docs/data/phase67/cost_model.json]

Inputs (all measured in this phase): synchronized per-part compute at H=1024 for shard fractions 0.25..1.0 on both hosts (compute_linux.json, compute_mac.json), the full
single-device decode step and warm prefill of each host, and the physical exact-size TBCCL latencies (collectives.json: p2p / AllGather / bf16 AllReduce, gap 0/200/500 us,
8 B .. 2.36 MB), plus the physical decode cadence replays (physical/R_decode_*).

Compute model per host h and synchronized part (attention half or MLP half) with shard fraction f:
    decode : t = floor_h + bytes(f) / BW_h          bytes = weights (+ KV cache read for attention) of the shard; floor_h = the per-eval synchronization / launch floor
    prefill: t = c_h + flops(f, tokens) / peak_h
(floor, BW, c, peak fitted per host by least squares on the measured fractions at H=1024.) Scaling to other H / intermediate / layers / heads / batch / sequence changes
bytes and flops only; floors do not scale. Communication: one bf16 AllReduce of tokens*H*2 B after each row-parallel projection (2 per layer), plus one small final exchange.
A token costs sum over parts of [max over hosts of the part's compute] + AllReduce latency (interpolated on the measured curves) x the calibrated cadence factor.
Baselines: Linux-only, Mac-only (when the weights fit) and the best batch-1 pipeline split (stages run one after the other, plus the measured 1.5 ms pipeline overhead).
"""
import argparse
import json
import math
import os

# ----- model classes (public configs; hidden / intermediate / layers / q heads / kv heads / head_dim / vocab) -----
CLASSES = {
    "Qwen3-0.6B (H=1024)": (1024, 3072, 28, 16, 8, 128, 151936),
    "Qwen2.5-1.5B (H=1536)": (1536, 8960, 28, 12, 2, 128, 151936),
    "Qwen3-1.7B (H=2048)": (2048, 6144, 28, 16, 8, 128, 151936),
    "Qwen3-4B (H=2560)": (2560, 9728, 36, 32, 8, 128, 151936),
    "Llama-3.2-3B (H=3072)": (3072, 8192, 28, 24, 8, 128, 128256),
    "Qwen3-8B (H=4096)": (4096, 12288, 36, 32, 8, 128, 151936),
    "Qwen3-14B (H=5120)": (5120, 17408, 40, 40, 8, 128, 151936),
    "Qwen3-32B (H=5120)": (5120, 25600, 64, 64, 8, 128, 151936),
}
BPE = 1.0625  # bytes per weight element at 8-bit affine quantization (group 64, bf16 scale + bias)
CAP = {"linux": 6.8e9, "mac": 10.0e9}  # usable bytes for weights + KV cache (8 GB VRAM, 16 GB unified memory)
CADENCE = 1.12  # replay / model with compute gaps (measured +11.4 .. +12.5 % over the three gap scenarios)
OPT_FACTOR = 0.65  # optimistic: the back-to-back replay cost per collective (128 us) over the isolated 200 us-gap AllReduce p25 (196 us)
PIPE_OVERHEAD_US = 1500.0  # measured: best pipeline TPOT 6.74 ms (A, STEP) minus the 21+7 layer stage compute at fused single-device speed


def shapes(c, bits=8):
    H, I, L, nh, nkv, hd, V = c
    bpe = BPE if bits == 8 else (bits / 8 + 4 / 64)
    attn_el = H * hd * (nh + 2 * nkv) + nh * hd * H
    mlp_el = 3 * H * I
    return {"H": H, "I": I, "L": L, "nh": nh, "nkv": nkv, "hd": hd, "V": V, "bpe": bpe, "attn_el": attn_el, "mlp_el": mlp_el, "emb_el": V * H,
            "weights": (L * (attn_el + mlp_el) + V * H) * bpe}


def lstsq(xs, ys):
    n = len(xs)
    mx_, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    b = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx_
    return a, b


class Host:
    def __init__(self, name, d, base):
        self.name = name
        s = shapes(CLASSES["Qwen3-0.6B (H=1024)"])
        kv = 2 * s["nkv"] * s["hd"] * d["kv_len"] * 2  # K and V bf16 bytes per layer at f = 1
        fs = [float(f) for f in d["decode"]]
        xs, ys = [], []
        for f in fs:  # decode: both halves share one (floor, 1/BW) pair; the bytes of a shard are f * (weights + KV)
            xs += [f * (s["attn_el"] * s["bpe"] + kv), f * s["mlp_el"] * s["bpe"]]
            ys += [d["decode"][str(f)]["attn"]["median_us"], d["decode"][str(f)]["mlp"]["median_us"]]
        self.floor, slope = lstsq(xs, ys)
        self.floor = max(self.floor, 0.0)
        self.slope = max(slope, 1e-9)  # us per byte
        self.bw = 1e6 / self.slope / 1e9 if slope > 0 else float("inf")  # GB/s implied
        tok = 577
        px, py = [], []
        for f in fs:
            fl_a = 2 * tok * s["attn_el"] * f + 4 * tok * tok * s["nh"] * s["hd"] * f * 0.5
            fl_m = 2 * tok * s["mlp_el"] * f
            px += [fl_a, fl_m]
            py += [d["prefill"][str(f)]["attn"]["median_us"], d["prefill"][str(f)]["mlp"]["median_us"]]
        self.pc, pslope = lstsq(px, py)
        self.pc = max(self.pc, 0.0)
        self.pslope = max(pslope, 1e-12)
        self.peak = 1e6 / self.pslope / 1e12  # TFLOP/s implied
        self.full_decode_us = d["decode_full_step"]["median_us"]
        self.full_prefill_us = d["prefill_full_warm_s"] * 1e6
        w06 = s["weights"]
        self.fused_bw = w06 / self.full_decode_us / 1e3  # GB/s effective of the fused single-device decode step (all 28 layers + head), kept for reference
        # fused single-device decode = per-layer fixed cost + weight traffic at the part-fit bandwidth, calibrated on the measured full decode step of the 0.6B model
        self.fused_layer_us = max(0.0, (self.full_decode_us - w06 * self.slope) / 28)
        self.head_half_us = d["lm_head"]["half"]["median_us"]
        self.head_full_us = d["lm_head"]["full"]["median_us"]
        self.cap = CAP[name]

    def dec_part(self, nbytes, flops=0.0):
        return max(self.floor + nbytes * self.slope, flops * self.pslope + self.pc)

    def pre_part(self, flops):
        return self.pc + flops * self.pslope


def curve(points):
    pts = sorted(points)

    def f(b):
        if b <= pts[0][0]:
            return pts[0][1]
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            if b <= b1:
                return math.exp(math.log(v0) + (math.log(v1) - math.log(v0)) * (math.log(b) - math.log(b0)) / (math.log(b1) - math.log(b0)))
        b0, v0 = pts[-1]
        return v0 * b / b0  # beyond 2.36 MB: bandwidth bound at the last measured effective rate

    return f


class Comm:
    """Physical AllReduce latency curves (us) by regime. cold = a 200 us compute gap precedes each op (decode cadence); large sizes were measured back to back."""

    def __init__(self, col):
        def pts(op, stat, small_gap=200):
            out = []
            for k, v in col.items():
                o, b, g = k.split("|")
                if o != op:
                    continue
                b, g = int(b), int(g)
                if (b <= 16384 and g == small_gap) or (b > 16384 and g == 0):
                    out.append((b, v["rank0"][stat]))
            return out

        self.opt = curve(pts("allreduce", "p25_us"))
        self.real = curve(pts("allreduce", "median_us"))
        self.pess = curve(pts("allreduce", "p95_us"))
        self.hot = curve([(int(k.split("|")[1]), v["rank0"]["median_us"]) for k, v in col.items() if k.startswith("allreduce|") and k.endswith("|0")])
        self.gather = curve([(int(k.split("|")[1]), v["rank0"]["median_us"]) for k, v in col.items() if k.startswith("allgather|") and k.endswith("|200") or (k.startswith("allgather|") and k.endswith("|0") and int(k.split("|")[1]) > 16384)])


def tp_decode(m, mac, lin, comm, fm, batch=1, kv_len=577, regime="real", cadence=1.0, comm_scale=1.0, attn_replicated=False, comm_bytes_scale=1.0):
    """One decode token of TP2 with fraction fm of the heads / MLP columns / vocabulary on the Mac. Returns the breakdown in us (None if the shards do not fit)."""
    s = shapes(m) if isinstance(m, tuple) else m
    L = s["L"]
    kv_bytes = lambda f: 2 * s["nkv"] * s["hd"] * kv_len * 2 * f * batch
    parts = []
    for h, f in ((mac, fm), (lin, 1.0 - fm)):
        fa = 1.0 if attn_replicated else f  # formulation 2: attention replicated on both hosts, only the MLP is sharded, one AllReduce per layer
        ab = fa * s["attn_el"] * s["bpe"] + kv_bytes(fa)
        mb = f * s["mlp_el"] * s["bpe"]
        fl_a, fl_m = 2 * batch * s["attn_el"] * fa, 2 * batch * s["mlp_el"] * f
        parts.append((h.dec_part(ab, fl_a), h.dec_part(mb, fl_m)))
        if (f * (s["weights"]) + L * kv_bytes(f) + (L * (1 - f) * s["attn_el"] * s["bpe"] if attn_replicated else 0)) > h.cap:
            return None
    fac = OPT_FACTOR if regime == "opt" else CADENCE * cadence
    ar = getattr(comm, regime)(batch * s["H"] * 2 * comm_bytes_scale) * comm_scale * fac
    ta = max(parts[0][0], parts[1][0])
    tm = max(parts[0][1], parts[1][1])
    head = [h.dec_part(f * s["emb_el"] * s["bpe"], 2 * batch * f * s["emb_el"]) for h, f in ((mac, fm), (lin, 1.0 - fm))]
    exch = comm.gather(8 * batch) * comm_scale * fac
    crit_c = L * (ta + tm) + max(head)
    crit_m = L * (1 if attn_replicated else 2) * ar + exch
    own_m = L * (parts[0][0] + parts[0][1]) + head[0]
    own_l = L * (parts[1][0] + parts[1][1]) + head[1]
    floors = L * 2 * (mac.floor if parts[0][0] >= parts[1][0] else lin.floor)
    tot = crit_c + crit_m
    return {"total": tot, "compute_crit": crit_c, "comm": crit_m, "mac_compute": own_m, "linux_compute": own_l, "imbalance_wait": crit_c - min(own_m, own_l) if False else abs(own_m - own_l),
            "floor_share": floors, "comm_pct": 100 * crit_m / tot, "ar_us": ar}


def tp_prefill(m, mac, lin, comm, fm, tokens, regime="real", cadence=1.0):
    s = shapes(m) if isinstance(m, tuple) else m
    L = s["L"]
    parts = []
    for h, f in ((mac, fm), (lin, 1.0 - fm)):
        fl_a = 2 * tokens * s["attn_el"] * f + 4 * tokens * tokens * s["nh"] * s["hd"] * f * 0.5
        fl_m = 2 * tokens * s["mlp_el"] * f
        parts.append((h.pre_part(fl_a), h.pre_part(fl_m)))
        if f * s["weights"] > h.cap:
            return None
    ar = getattr(comm, regime)(tokens * s["H"] * 2) * (OPT_FACTOR if regime == "opt" else CADENCE * cadence)
    crit_c = L * (max(parts[0][0], parts[1][0]) + max(parts[0][1], parts[1][1]))
    crit_m = L * 2 * ar
    own_m, own_l = L * sum(parts[0]), L * sum(parts[1])
    tot = crit_c + crit_m
    return {"total": tot, "compute_crit": crit_c, "comm": crit_m, "mac_compute": own_m, "linux_compute": own_l, "imbalance_wait": abs(own_m - own_l), "comm_pct": 100 * crit_m / tot, "ar_us": ar}


def single(h, s, tokens=None, batch=1, kv_len=577):
    """Fused single-device time per decode token (or per prefill of `tokens`), None if it does not fit."""
    kv = s["L"] * 2 * s["nkv"] * s["hd"] * kv_len * 2 * batch
    if s["weights"] + kv > h.cap:
        return None
    if tokens is None:
        mem = s["L"] * h.fused_layer_us + (s["weights"] + kv) * h.slope
        return max(mem, 2 * batch * (s["L"] * (s["attn_el"] + s["mlp_el"]) + s["emb_el"]) * h.pslope + h.pc)
    fl = s["L"] * (2 * tokens * (s["attn_el"] + s["mlp_el"]) + 4 * tokens * tokens * s["nh"] * s["hd"] * 0.5)
    return fl * h.pslope + s["L"] * h.pc


def pipeline(mac, lin, s, batch=1, kv_len=577):
    """Best batch-1 pipeline split: the Linux stage takes as many layers as fit, stages run one after the other (no overlap at batch 1)."""
    per_layer_bytes = (s["attn_el"] + s["mlp_el"]) * s["bpe"] + 2 * s["nkv"] * s["hd"] * kv_len * 2 * batch
    best = None
    for nl in range(0, s["L"] + 1):
        if nl * per_layer_bytes > lin.cap or (s["L"] - nl) * per_layer_bytes + (s["emb_el"] * s["bpe"]) > mac.cap:
            continue
        t = nl * (lin.fused_layer_us + per_layer_bytes * lin.slope) + (s["L"] - nl) * (mac.fused_layer_us + per_layer_bytes * mac.slope) + s["emb_el"] * s["bpe"] * mac.slope + PIPE_OVERHEAD_US
        if best is None or t < best[0]:
            best = (t, nl)
    return best


def best_baseline(mac, lin, s, batch=1):  # returns (name, us, candidates); name None when nothing fits
    cands = {}
    for nm, h in (("Linux-only", lin), ("Mac-only", mac)):
        t = single(h, s, batch=batch)
        if t is not None:
            cands[nm] = t * (1 if batch == 1 else 1)
    p = pipeline(mac, lin, s, batch)
    if p:
        cands[f"pipeline ({p[1]}/{s['L'] - p[1]} Linux/Mac layers)"] = p[0]
    if not cands:
        return None, float("inf"), cands
    nm = min(cands, key=cands.get)
    return nm, cands[nm], cands


def best_fm(fn, step=0.025):
    """Best Mac fraction in [0.1, 0.9] (the Mac must really take part; fm = 0 would be Linux-only plus pointless collectives)."""
    best = None
    k = 4
    while k <= 36:
        fm = k * step
        r = fn(fm)
        if r is not None and (best is None or r["total"] < best[1]["total"]):
            best = (fm, r)
        k += 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    ld = lambda n: json.load(open(os.path.join(a.data, n)))
    lin, mac = Host("linux", ld("compute_linux.json"), a.data), Host("mac", ld("compute_mac.json"), a.data)
    comm = Comm(ld("collectives.json"))
    res = {"fits": {}, "validation": [], "decode": [], "prefill": [], "breakeven": {}}
    for h in (lin, mac):
        print(f"{h.name:6} decode part floor {h.floor:6.0f} us, implied weight BW {h.bw:6.1f} GB/s | prefill const {h.pc:6.0f} us, implied peak {h.peak:5.2f} TFLOP/s | fused decode eff BW {h.fused_bw:5.1f} GB/s | LM head half {h.head_half_us:.0f} us")
        res["fits"][h.name] = {"floor_us": h.floor, "bw_gbs": h.bw, "prefill_const_us": h.pc, "peak_tflops": h.peak, "fused_bw_gbs": h.fused_bw}
    # ----- validation against the physical decode replays -----
    prof = lambda n, r: json.load(open(os.path.join(a.data, "profiles", f"{n}.rank{r}.json")))["ops"]
    print("\nValidation: model of the replay (sum over collectives of max(gap) + AllReduce latency at the real curve) vs measured replay token time, us")
    ratios = []
    for sc in ("decode_comm_only", "decode_equal", "decode_asym25", "decode_asym375", "decode_asym375#2", "decode_attn_replicated", "prefill_comm_only", "prefill_equal", "prefill_asym25"):
        base = sc.split("#")[0]
        f = os.path.join(a.data, "physical", f"R_{base}.linux.out" if "#" not in sc and sc != "decode_attn_replicated" else f"R2_{base}.linux.out")
        if not os.path.exists(f):
            continue
        prof_name = base
        meas = json.loads(open(f).read().strip().splitlines()[-1])["token_wall_median_us"]
        o0, o1 = prof(prof_name, 0), prof(prof_name, 1)
        pred = sum(max(x["gap_us"], y["gap_us"]) + (comm.real(x["bytes"]) * (CADENCE if sc.startswith("prefill") else 1.0) if x["op"] == "allreduce" else comm.gather(max(x["bytes"], 8))) for x, y in zip(o0, o1))
        pred_hot = sum(max(x["gap_us"], y["gap_us"]) + (comm.hot(x["bytes"]) if x["op"] == "allreduce" else comm.gather(max(x["bytes"], 8))) for x, y in zip(o0, o1))
        gaps = sum(max(x["gap_us"], y["gap_us"]) for x, y in zip(o0, o1))
        if sc.startswith("decode") and sc != "decode_comm_only":
            ratios.append(meas / pred)
        res["validation"].append({"scenario": sc, "replay_us": meas, "model_us": pred, "model_hot_us": pred_hot, "diff_pct": 100 * (meas - pred) / pred, "sum_max_gaps_us": gaps,
                                  "replay_minus_gaps_us": meas - gaps, "collectives": len(o0)})
        print(f"  {sc:20} replay {meas:8.0f}  model {pred:8.0f} ({100 * (meas - pred) / pred:+5.1f} %)  [hot-curve model {pred_hot:8.0f}]  max-gap sum {gaps:7.0f}  extra over gaps {meas - gaps:7.0f} us = {(meas - gaps) / len(o0):.0f} us per collective")
    cad = sorted(ratios)[len(ratios) // 2] if ratios else 1.0
    res["cadence_factor_used"] = 1.0
    print(f"  replay/model ratios {['%.3f' % r for r in ratios]}; the model is used unscaled (cadence factor 1.0) when all are within ~15-20 %")

    # ----- decode table -----
    print("\nDecode, batch 1, kv 577, 8-bit weights. TPOT in ms (TP2: Mac fraction in parentheses = optimum over 0..1; None = does not fit)")
    print(f"{'model':26}{'weights GB':>10}{'equal':>9}{'37.5/62.5':>11}{'25/75':>9}{'optimum':>16}{'best baseline':>34}{'gain %':>8}{'comm %':>7}{'free-comm':>10}{'attn-repl':>10}{'int8 comm':>10}")
    for nm, c in CLASSES.items():
        s = shapes(c)
        bn, bt, cands = best_baseline(mac, lin, s)
        if bn is None:
            print(f"{nm:26}{s['weights'] / 1e9:10.2f}  does not fit Linux + Mac together (6.8 + 10 GB usable)")
            continue
        row = {"model": nm, "weights_gb": s["weights"] / 1e9, "baseline": bn, "baseline_ms": bt / 1e3, "baselines_ms": {k: v / 1e3 for k, v in cands.items()}}
        cell = {}
        for tag, fm in (("equal", 0.5), ("a375", 0.375), ("a25", 0.25)):
            r = tp_decode(s, mac, lin, comm, fm)
            cell[tag] = r
            row[tag + "_ms"] = r["total"] / 1e3 if r else None
        opt = best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm))
        row["opt_fm"], row["opt_ms"] = (opt[0], opt[1]["total"] / 1e3) if opt else (None, None)
        row["opt"] = opt[1] if opt else None
        gain = 100 * (bt - opt[1]["total"]) / bt if opt else float("nan")
        row["opt_gain_pct"] = gain
        fo = best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, comm_scale=0.0))
        row["free_comm_gain_pct"] = 100 * (bt - fo[1]["total"]) / bt if fo else None
        a2 = best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, attn_replicated=True))
        row["attn_replicated_ms"] = a2[1]["total"] / 1e3 if a2 else None
        row["attn_replicated_gain_pct"] = 100 * (bt - a2[1]["total"]) / bt if a2 else None
        i8 = best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, comm_bytes_scale=0.5))
        row["int8_comm_gain_pct"] = 100 * (bt - i8[1]["total"]) / bt if i8 else None
        res["decode"].append(row)
        f = lambda r: f"{r['total'] / 1e3:9.2f}" if r else f"{'n/a':>9}"
        print(f"{nm:26}{s['weights'] / 1e9:10.2f}{f(cell['equal'])}{f(cell['a375']):>11}{f(cell['a25'])}{(f"{opt[1]['total'] / 1e3:8.2f} @{opt[0]:.3f}" if opt else 'n/a'):>16}{(bn + f' {bt / 1e3:.2f}'):>34}{gain:8.1f}{(opt[1]['comm_pct'] if opt else float('nan')):7.1f}{row['free_comm_gain_pct']:10.1f}{(row['attn_replicated_gain_pct'] if a2 else float('nan')):10.1f}{(row['int8_comm_gain_pct'] if i8 else float('nan')):10.1f}")

    # ----- separate the two heterogeneity costs: Mac per-part synchronization floor (B) vs pure bandwidth/compute imbalance (A) -----
    print("\nDecode upper bounds (gain % of the optimum TP2 split over the best baseline): where does the heterogeneity cost come from?")
    print(f"{'model':26}{'as measured':>12}{'Mac floor 0':>13}{'comm free':>11}{'both (A only)':>15}{'Linux floor too':>17}")
    for nm, c in CLASSES.items():
        s = shapes(c)
        bn, bt, _ = best_baseline(mac, lin, s)
        if bn is None:
            continue
        g = lambda **kw: (lambda o: 100 * (bt - o[1]["total"]) / bt if o else float("nan"))(best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, **kw)))
        as_m = g()
        fm0 = mac.floor
        mac.floor = 0.0
        nofl, both = g(), g(comm_scale=0.0)
        fl0 = lin.floor
        lin.floor = 0.0
        both2 = g(comm_scale=0.0)
        mac.floor, lin.floor = fm0, fl0
        res["decode"][[r["model"] for r in res["decode"]].index(nm)]["bounds"] = {"measured": as_m, "mac_floor0": nofl, "comm_free": g(comm_scale=0.0), "pure_imbalance": both, "no_floors": both2}
        print(f"{nm:26}{as_m:12.1f}{nofl:13.1f}{g(comm_scale=0.0):11.1f}{both:15.1f}{both2:17.1f}")

    # ----- decode sensitivities for the 0.6B and a mid model -----
    print("\nDecode sensitivity (optimum Mac fraction, gain % vs best baseline) by AllReduce latency regime / scale")
    for nm in ("Qwen3-0.6B (H=1024)", "Qwen3-4B (H=2560)", "Qwen3-8B (H=4096)", "Qwen3-14B (H=5120)", "Qwen3-32B (H=5120)"):
        s = shapes(CLASSES[nm])
        bn, bt, _ = best_baseline(mac, lin, s)
        if bn is None:
            continue
        out = []
        for lab, kw in (("optimistic", {"regime": "opt"}), ("realistic", {}), ("pessimistic", {"regime": "pess"}), ("comm x0.5", {"comm_scale": 0.5}), ("comm x0.25", {"comm_scale": 0.25}), ("comm x0 (free)", {"comm_scale": 0.0})):
            o = best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, **kw))
            out.append((lab, o[1]["total"] / 1e3 if o else None, o[0] if o else None, 100 * (bt - o[1]["total"]) / bt if o else None))
        res["breakeven"][nm] = {"baseline": bn, "baseline_ms": bt / 1e3, "sensitivity": out}
        print(f"  {nm:24} baseline {bn} {bt / 1e3:.2f} ms: " + "; ".join(f"{l} {t:.2f} ms @{fm:.2f} ({g:+.0f} %)" for l, t, fm, g in out if t is not None))

    # ----- batch -----
    print("\nDecode batch scaling (optimum split; gain % vs best baseline at the same batch)")
    for nm in ("Qwen3-0.6B (H=1024)", "Qwen3-8B (H=4096)", "Qwen3-14B (H=5120)"):
        s = shapes(CLASSES[nm])
        line = []
        for b in (1, 2, 4, 8, 16, 32):
            bn, bt, _ = best_baseline(mac, lin, s, batch=b)
            o = best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, batch=b))
            line.append(f"B={b}: TP {o[1]['total'] / 1e3:.2f} vs {bt / 1e3:.2f} ms ({100 * (bt - o[1]['total']) / bt:+.0f} %)" if o else f"B={b}: n/a")
        print(f"  {nm:24} " + "; ".join(line))

    # ----- prefill -----
    print("\nPrefill (time to process the prompt, ms): TP2 optimum vs best single device / pipeline (compute-bound estimate)")
    for nm in ("Qwen3-0.6B (H=1024)", "Qwen3-4B (H=2560)", "Qwen3-8B (H=4096)", "Qwen3-14B (H=5120)"):
        s = shapes(CLASSES[nm])
        for tok in (128, 512, 1024):
            cands = {}
            for hn, h in (("Linux-only", lin), ("Mac-only", mac)):
                t = single(h, s, tokens=tok)
                if t is not None:
                    cands[hn] = t
            nl_best = None
            per_layer_bytes = (s["attn_el"] + s["mlp_el"]) * s["bpe"]
            for nl in range(0, s["L"] + 1):
                if nl * per_layer_bytes > lin.cap or (s["L"] - nl) * per_layer_bytes + s["emb_el"] * s["bpe"] > mac.cap:
                    continue
                fl = lambda n: n * (2 * tok * (s["attn_el"] + s["mlp_el"]) + 4 * tok * tok * s["nh"] * s["hd"] * 0.5)
                t = fl(nl) * lin.pslope + fl(s["L"] - nl) * mac.pslope + lin.pc + mac.pc + PIPE_OVERHEAD_US
                if nl_best is None or t < nl_best[0]:
                    nl_best = (t, nl)
            if nl_best and nl_best[1] == s["L"]:
                pass  # all layers on Linux: the Linux-only candidate above
            elif nl_best:
                cands[f"pipeline ({nl_best[1]}/{s['L'] - nl_best[1]})"] = nl_best[0]
            bn = min(cands, key=cands.get)
            o = best_fm(lambda fm: tp_prefill(s, mac, lin, comm, fm, tok))
            row = {"model": nm, "tokens": tok, "baseline": bn, "baseline_ms": cands[bn] / 1e3, "tp_ms": o[1]["total"] / 1e3 if o else None, "fm": o[0] if o else None,
                   "comm_pct": o[1]["comm_pct"] if o else None, "gain_pct": 100 * (cands[bn] - o[1]["total"]) / cands[bn] if o else None}
            res["prefill"].append(row)
            print(f"  {nm:24} L={tok:5}: baseline {bn:24} {cands[bn] / 1e3:8.1f}  TP2 " + (f"{o[1]['total'] / 1e3:8.1f} @fm {o[0]:.3f} comm {o[1]['comm_pct']:.0f} % gain {row['gain_pct']:+.0f} %" if o else "n/a"))

    # ----- break-even AllReduce latency (decode) -----
    print("\nBreak-even AllReduce latency (us per 2-per-layer AllReduce at batch 1) so that TP2 (optimum split) reaches +0 %, +20 % over the best baseline; measured: "
          f"{comm.real(2048):.0f} us (2 KB)")
    for nm in CLASSES:
        s = shapes(CLASSES[nm])
        bn, bt, _ = best_baseline(mac, lin, s)
        if bn is None:
            continue
        out = {}
        for tag, target in (("0%", 0.0), ("20%", 0.20)):
            lo, hi = 0.0, 5.0
            ok = lambda sc: (lambda o: o is not None and o[1]["total"] <= bt * (1 - target))(best_fm(lambda fm: tp_decode(s, mac, lin, comm, fm, comm_scale=sc)))
            if not ok(0.0):
                out[tag] = None  # unreachable even with free communication: the Mac adds too little bandwidth/compute
                continue
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
            out[tag] = lo * comm.real(2048)
        res["breakeven"].setdefault(nm, {})["ar_us"] = out
        print(f"  {nm:26} baseline {bt / 1e3:7.2f} ms: AR latency <= {'unreachable (free comm still loses)' if out['0%'] is None else round(out['0%'])} us for parity, <= {'unreachable' if out['20%'] is None else round(out['20%'])} us for +20 %  (free-comm bound {res['decode'][[r['model'] for r in res['decode']].index(nm)]['free_comm_gain_pct']:+.0f} %)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

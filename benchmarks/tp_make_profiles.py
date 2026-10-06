"""Per-rank TP cadence-replay profiles for tp_replay.cpp from the measured compute JSONs (benchmarks/tp_compute_bench.py).

    python benchmarks/tp_make_profiles.py --linux <data dir> --mac <data dir> --out <data dir>

One decode token (or one prefill pass) of TP2 = per layer: [shard attention compute] AllReduce [shard MLP compute] AllReduce, then [LM-head shard + argmax] AllGather(8 B, greedy
(max, argmax) exchange). The gap before each collective is that rank's own measured median compute for its shard (the whole part incl. its eval); rank 0 = Linux, rank 1 = Mac.
Scenarios: equal (50/50), asym25 (Mac 25 % / Linux 75 %), asym375 (Mac 37.5 % / Linux 62.5 %), comm-only (all gaps 0), attention-replicated (MLP-only TP, one AllReduce per layer), prefill.
"""
import argparse
import json
import os

LAYERS, HIDDEN = 28, 1024


def part(d, kind, f, phase="decode"):
    return d[phase][str(f)][kind]["median_us"]


def prof(rank, d, scen, phase, layers=LAYERS):
    fm, ar_bytes = scen["mac"], (HIDDEN * 2 if phase == "decode" else 577 * HIDDEN * 2)
    f = fm if rank == 1 else 1.0 - fm
    ops = []
    for _ in range(layers):
        if scen.get("attn_replicated"):
            g = 0 if scen["comm_only"] else round(part(d, "attn", 1.0, phase) + part(d, "mlp", f, phase))
            ops.append({"op": "allreduce", "bytes": ar_bytes, "gap_us": g})
            continue
        ga = 0 if scen["comm_only"] else round(part(d, "attn", f, phase))
        gm = 0 if scen["comm_only"] else round(part(d, "mlp", f, phase))
        ops.append({"op": "allreduce", "bytes": ar_bytes, "gap_us": ga})
        ops.append({"op": "allreduce", "bytes": ar_bytes, "gap_us": gm})
    head = d["lm_head"]
    gh = 0 if scen["comm_only"] else round(head["half"]["median_us"] * (f / 0.5) if phase == "decode" else 0)
    ops.append({"op": "all_gather", "bytes": 8, "gap_us": gh})
    return {"rank": rank, "ops": ops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linux", required=True)
    ap.add_argument("--mac", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hosts = {0: json.load(open(a.linux)), 1: json.load(open(a.mac))}
    scens = {
        "decode_equal": ({"mac": 0.5, "comm_only": False}, "decode"),
        "decode_asym25": ({"mac": 0.25, "comm_only": False}, "decode"),
        "decode_asym375": ({"mac": 0.375, "comm_only": False}, "decode"),
        "decode_comm_only": ({"mac": 0.5, "comm_only": True}, "decode"),
        "decode_attn_replicated": ({"mac": 0.5, "comm_only": False, "attn_replicated": True}, "decode"),
        "prefill_equal": ({"mac": 0.5, "comm_only": False}, "prefill"),
        "prefill_asym25": ({"mac": 0.25, "comm_only": False}, "prefill"),
        "prefill_comm_only": ({"mac": 0.5, "comm_only": True}, "prefill"),
    }
    for name, (scen, phase) in scens.items():
        for rank in (0, 1):
            p = prof(rank, hosts[rank], scen, phase)
            json.dump(p, open(f"{a.out}/{name}.rank{rank}.json", "w"), separators=(",", ":"))
        tot = [sum(o["gap_us"] for o in prof(r, hosts[r], scen, phase)["ops"]) for r in (0, 1)]
        print(f"{name:26} ops/token {len(p['ops'])}  gap sum per token: Linux {tot[0]} us, Mac {tot[1]} us")


if __name__ == "__main__":
    main()

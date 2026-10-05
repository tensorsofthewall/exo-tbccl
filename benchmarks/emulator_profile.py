"""Phase 59: turn the Linux side of the Phase 58 physical traces into replay profiles for benchmarks/remote_peer_emulator.py.

    python benchmarks/emulator_profile.py --out docs/data/phase59/profiles [--data docs/data/phase58/physical]

Every delay is an interval on the LINUX host's own clock (so no cross-host alignment is involved):
  orientation A (the emulator plays Linux rank 0): per step, the time from the end of its previous all_gather to the start of its send, split into
      resume | sampler | graph build (pre-compute) | stage compute | send prep
  orientation B (the emulator plays Linux rank 1): the time from the end of its previous all_gather to posting its receive (sampler + pre-recv), and from
      the receive's completion to entering the all_gather (first use + stage compute + pre-gather)
One profile per (orientation, backend) from that backend's physical runs (their Linux sides are statistically identical; each backend's own is used for fidelity).
Steps 0-1 and steps containing the driver's KV-digest evals are dropped.
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distributed_timeline as dt  # noqa: E402


def steps_for(path, role):
    d, ev = dt.load(path)
    st = dt.by_step(ev, d["backend"])
    out = []
    for s in sorted(st):
        if s < 2 or s - 1 not in st:
            continue
        a, p = st[s], st[s - 1]
        if role == "rank0":  # Linux = rank 0 (orientation A)
            need = [p["ag_end"], p["sampler"], a["model"], a["send_begin"]]
            if any(x is None for x in need):
                continue
            us = lambda x: x / 1000.0
            out.append({
                "resume_us": us(p["sampler"]["t0"] - p["ag_end"]),
                "sampler_us": us(p["sampler"]["t1"] - p["sampler"]["t0"]),
                "graph_us": us(a["model"]["t0"] - p["sampler"]["t1"]),
                "compute_us": us(a["model"]["t1"] - a["model"]["t0"]),
                "prep_us": us(a["send_begin"] - a["model"]["t1"]),
                "total_us": us(a["send_begin"] - p["ag_end"]),
            })
        else:  # Linux = rank 1 (orientation B)
            need = [p["ag_end"], p["sampler"], a["pre_recv"], a["recv_begin"], a["recv_complete"], a["model"], a["ag_begin"]]
            if any(x is None for x in need):
                continue
            us = lambda x: x / 1000.0
            out.append({
                "resume_us": us(p["sampler"]["t0"] - p["ag_end"]),
                "sampler_us": us(p["sampler"]["t1"] - p["sampler"]["t0"]),
                "pre_recv_us": us(a["recv_begin"] - p["sampler"]["t1"]),  # graph build + pre-recv eval until the receive is posted
                "post_us": us(a["recv_begin"] - p["ag_end"]),
                "first_use_us": us(a["model"]["t0"] - a["recv_complete"]),
                "compute_us": us(a["model"]["t1"] - a["model"]["t0"]),
                "pregather_us": us(a["ag_begin"] - a["model"]["t1"]),
                "compute_total_us": us(a["ag_begin"] - a["recv_complete"]),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data", "phase58", "physical"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = {"A": ("phys_A_{b}*.rank0.json", "rank0"), "B": ("phys_B7_{b}*.rank1.json", "rank1")}
    if os.environ.get("PHASE") == "60":  # Phase 61 A-v2: the Phase 60 physical A runs (A1 = TBCCL baseline, A3 = Ring)
        a.data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data", "phase60", "physical")
        spec = {"A": ("phys_A{n}_{b}.rank0.json", "rank0")}
    for orient, (pat, role) in spec.items():
        for backend in ("tbccl", "ring"):
            files = sorted(f for f in glob.glob(os.path.join(a.data, pat.format(b=backend, n="1" if backend == "tbccl" else "3"))) if not f.endswith("jsonl"))
            steps = [s for f in files for s in steps_for(f, role)]
            med = {k: round(statistics.median(x[k] for x in steps), 1) for k in steps[0]}
            prof = {"orientation": orient, "backend": backend, "emulated_rank": 0 if orient == "A" else 1, "source": [os.path.basename(f) for f in files],
                    "n_steps": len(steps), "median_us": med, "steps": [{k: round(v, 1) for k, v in x.items()} for x in steps]}
            path = os.path.join(a.out, f"profile_{orient}_{backend}.json" if os.environ.get("PHASE") != "60" else f"phase60_A_physical_v2_{backend}.json")
            json.dump(prof, open(path, "w"), indent=1)
            print(path, len(steps), "steps; median", {k: v for k, v in med.items() if k.endswith("total_us") or k in ("post_us",)})


if __name__ == "__main__":
    main()

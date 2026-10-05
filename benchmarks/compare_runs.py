"""Ring vs TBCCL critical-path difference table for one orientation, from physical (or loopback) traces.

    python benchmarks/compare_runs.py --ring r0a.json r1a.json [r0b.json r1b.json ...] --tbccl t0a.json t1a.json [...] [--host0 Linux --host1 Mac]

Each backend's runs are paired by rank files in order; per component the per-step values of all runs are pooled and the median is reported, and the paired
(step i of a TBCCL run vs step i of a Ring run, same run index) mean difference and its standard error are computed. Rows are the plan's table plus the
host that executes the component; the sum of the rows equals the step period exactly (the decomposition telescopes), so the 'explained' fraction of the
TPOT difference is the sum of the rows that are attributed to a cause, shown in the notes of
"""
import argparse
import math
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import distributed_timeline as dt  # noqa: E402

ROWS = [  # (row label, components summed, executing host rank)
    ("rank 0 sampling (token eval after the gather)", ["sampler"], 0),
    ("rank 0 resume (gather return -> next step) + pre-compute (graph build)", ["resume", "pre-compute"], 0),
    ("rank 0 stage compute", ["compute0"], 0),
    ("send-side sync (step_complete, dependency evals)", ["send-prep"], 0),
    ("rank 1 not yet ready when the send began (peer-late)", ["peer-late"], 1),
    ("send -> recv complete (bridge, wire, wake)", ["transit"], None),
    ("rank 1 first use (recv complete -> compute begin)", ["first-use"], 1),
    ("rank 1 stage compute", ["compute1"], 1),
    ("rank 1 pre-gather", ["pre-gather"], 1),
    ("AllGather entry -> rank 0 completion", ["ag-xfer"], None),
    ("TOTAL step period", ["period"], None),
]


def steps(r0path, r1path, skip=2):
    d0, e0 = dt.load(r0path)
    d1, e1 = dt.load(r1path)
    al = dt.align(d0.get("clock", {}))
    s0, s1 = dt.by_step(e0, d0["backend"]), dt.by_step(e1, d1["backend"])
    out = {}
    for s in sorted(s0):
        if s >= skip:
            c = dt.components(s0, s1, al, s)
            if c:
                out[s] = c
    return out, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", nargs="+", required=True)
    ap.add_argument("--tbccl", nargs="+", required=True)
    ap.add_argument("--host0", default="rank0")
    ap.add_argument("--host1", default="rank1")
    a = ap.parse_args()
    hosts = {0: a.host0, 1: a.host1, None: "link/both"}
    R = [steps(a.ring[i], a.ring[i + 1]) for i in range(0, len(a.ring), 2)]
    T = [steps(a.tbccl[i], a.tbccl[i + 1]) for i in range(0, len(a.tbccl), 2)]
    unc = max(al["unc"] for _, al in R + T) / 1000
    print(f"{len(R)} Ring run(s), {len(T)} TBCCL run(s); clock-offset uncertainty up to +/- {unc:.0f} us; microseconds, medians over pooled steps (p25..p75) and paired mean difference +/- s.e.")
    print(f"  {'component':68}{'host':>10}{'Ring':>8}{'TBCCL':>8}{'TBCCL-Ring':>12}{'paired diff':>16}")
    tot = None
    for label, comps, h in ROWS:
        get = lambda runs: [sum(c[k] for k in comps) for run, _ in runs for c in run.values()]
        rv, tv = get(R), get(T)
        pd = []
        for (rr, _), (tt, _) in zip(R, T):
            for s in set(rr) & set(tt):
                pd.append(sum(tt[s][k] for k in comps) - sum(rr[s][k] for k in comps))
        mr, mt = statistics.median(rv), statistics.median(tv)
        se = statistics.stdev(pd) / math.sqrt(len(pd)) if len(pd) > 1 else 0
        print(f"  {label:68}{hosts[h]:>10}{mr:8.0f}{mt:8.0f}{mt - mr:+12.0f}{statistics.mean(pd):+10.0f} +/- {se:.0f}")
    for label, comps in (("(not in sum) AllGather entry skew: rank 1 enters later than rank 0 by", ["ag-entry-skew"]), ("(not in sum) send call duration", ["send-call"])):
        get = lambda runs: [c[comps[0]] for run, _ in runs for c in run.values()]
        print(f"  {label:68}{'':>10}{statistics.median(get(R)):8.0f}{statistics.median(get(T)):8.0f}{statistics.median(get(T)) - statistics.median(get(R)):+12.0f}")


if __name__ == "__main__":
    main()

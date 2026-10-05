"""Phase 58: reconstruct one decode step across two hosts on a common clock and attribute its time.

    python benchmarks/distributed_timeline.py rank0.json rank1.json [--steps 3,4] [--skip 3] [--csv out.csv]

Inputs are sync_recorder dumps (rank 0's holds the clock-alignment results of benchmarks/clock_sync.py). Rank 1's timestamps are mapped onto rank 0's clock with
offset(t) = pre-offset + drift * (t - t_pre) (linear between the pre- and post-run measurements); the uncertainty printed is the larger of the two measurements'.

One decode step, per rank (exo's pipeline; identical for both backends):
    rank 0:  sampler(s-1) | compute(s) | send(s) | all_gather(s)      (sampler(s-1) = the token eval that follows all_gather(s-1))
    rank 1:  sampler(s-1) | pre_recv eval | recv(s) | compute(s) | all_gather(s)
Boundaries used (the practical start/end of each operation, the same semantic point for both backends):
    send        tbccl: begin/end of the `send` call (bridge + native submit + Work wait);  ring: begin/end of the eval that EXECUTES the lazy Send (send_dependency_eval)
    recv        tbccl: begin of `recv_like`, complete = end of the native wait;            ring: begin/end of the eval that executes the lazy Recv (post_recv_eval)
    all_gather  tbccl: begin/end of the `all_gather` call;                                 ring: begin/end of the eval that executes it (post_allgather_eval)
The step period seen from rank 0 (all_gather end to all_gather end) telescopes exactly into:
    resume      rank 0 all_gather end -> its sampler begin            (completion wake: the eval/call returned, the next step's graph starts)
    sampler     the driver's token eval (lm_head + argmax; every rank does it)
    pre-compute rank 0 sampler end -> compute begin
    compute0    rank 0 stage compute (model_output_eval)
    send-prep   compute end -> send begin (step_complete, dependency evals)
    peer-late   how long rank 1 had NOT yet posted its receive when the send began (rank 1 still in its sampler / pre-recv eval): max(0, recv_begin - send_begin)
    transit     max(send_begin, recv_begin) -> recv complete on rank 1 (bridge on both sides + wire + wake on rank 1)
    first-use   recv complete -> rank 1 compute begin
    compute1    rank 1 stage compute
    pre-gather  rank 1 compute end -> its all_gather begin
    ag-xfer     rank 1 all_gather begin -> rank 0 all_gather end (the data movement and completion once both ranks are in it)
and the entry skew (rank 1 all_gather begin - rank 0 all_gather begin) is reported separately: it is how long rank 0 waits in the collective for rank 1's compute.
"""
import argparse
import json
import statistics

DIGEST = "prefix_digest"


def load(path):
    d = json.load(open(path))
    ev = [dict(t0=a, t1=b, kind=k, label=l, depth=dp, step=st, op=op) for a, b, k, l, dp, st, op, _tid in d["events"]]
    return d, ev


def align(clock):
    pre, post = clock.get("pre") or {}, clock.get("post") or {}
    if not pre:
        return dict(offset=0.0, drift=0.0, t0=0, unc=float("inf"), shift=0.0)
    unc = max(pre["uncertainty_ns"], post.get("uncertainty_ns", 0.0))
    if not post:
        return dict(offset=pre["offset_ns"], drift=0.0, t0=pre["t_mid_ns"], unc=unc, shift=0.0)
    dt = (post["t_mid_ns"] - pre["t_mid_ns"]) / 1e9
    drift = (post["offset_ns"] - pre["offset_ns"]) / dt if dt > 0 else 0.0
    return dict(offset=pre["offset_ns"], drift=drift, t0=pre["t_mid_ns"], unc=unc, shift=post["offset_ns"] - pre["offset_ns"], interval=dt)


def to_rank0(al, t1):
    # t0 is in rank 0's clock frame, so the drift is applied to the elapsed rank-0 time (Phase 62: the earlier t1 - t0 mixed frames, an error of
    # drift x |offset| that was ~0.1 ms when the hosts' monotonic clocks differed by ~30 s, and tens of ms when they differ by thousands of seconds)
    t = t1 - al["offset"]
    return t - al["drift"] * (t - al["t0"]) / 1e9


def pick(evs, kind, label):
    for e in evs:
        if e["kind"] == kind and e["label"] == label and e["depth"] == (0 if kind == "comm" or kind == "eval" else e["depth"]):
            return e
    return None


def by_step(ev, backend):
    steps = {}
    for e in ev:
        if e["step"] < 0 or e["label"].startswith("prefill:"):
            continue
        steps.setdefault(e["step"], []).append(e)
    out = {}
    for s, evs in steps.items():
        if any(DIGEST in e["label"] for e in evs):
            continue  # the driver's KV-digest steps are not decode
        d = {"evs": evs}
        top = [e for e in evs if e["depth"] == 0]
        g = lambda kind, label: next((e for e in top if e["kind"] == kind and e["label"] == label), None)
        d["model"] = g("eval", "model_output_eval")
        d["sampler"] = g("eval", "real_model_loopback.py:worker#4")
        d["pre_recv"] = g("eval", "pre_recv_template_eval")
        if backend == "tbccl":
            d["send"], d["recv"], d["ag"] = g("comm", "send"), g("comm", "recv_like"), g("comm", "all_gather")
            d["recv_begin"] = d["recv"]["t0"] if d["recv"] else None
            wr = [e for e in evs if e["kind"] == "tbccl_wait" and e["label"] == "wait:recv"]
            d["recv_complete"] = wr[0]["t1"] if wr else None
            d["send_begin"], d["send_end"] = (d["send"]["t0"], d["send"]["t1"]) if d["send"] else (None, None)
            d["ag_begin"], d["ag_end"] = (d["ag"]["t0"], d["ag"]["t1"]) if d["ag"] else (None, None)
        else:
            sd, pr, pa = g("eval", "send_dependency_eval"), g("eval", "post_recv_eval"), g("eval", "post_allgather_eval")
            d["send_begin"], d["send_end"] = (sd["t0"], sd["t1"]) if sd else (None, None)
            d["recv_begin"], d["recv_complete"] = (pr["t0"], pr["t1"]) if pr else (None, None)
            d["ag_begin"], d["ag_end"] = (pa["t0"], pa["t1"]) if pa else (None, None)
        out[s] = d
    return out


def components(r0, r1, al, s):
    """Telescoping decomposition of rank 0's step period for step s, microseconds, or None when an event is missing."""
    a, b = r0.get(s), r1.get(s)
    prev = r0.get(s - 1)
    if not (a and b and prev):
        return None
    need0 = [a["model"], prev["sampler"], a["send_begin"], a["ag_begin"], a["ag_end"], prev["ag_end"]]
    need1 = [b["model"], b["recv_complete"], b["recv_begin"], b["ag_begin"]]
    if any(x is None for x in need0 + need1):
        return None
    t = lambda x: to_rank0(al, x)
    send_begin = a["send_begin"]
    recv_begin, recv_done = t(b["recv_begin"]), t(b["recv_complete"])
    c1b, c1e, g1b = t(b["model"]["t0"]), t(b["model"]["t1"]), t(b["ag_begin"])
    comp = {
        "resume": prev["sampler"]["t0"] - prev["ag_end"],
        "sampler": prev["sampler"]["t1"] - prev["sampler"]["t0"],
        "pre-compute": a["model"]["t0"] - prev["sampler"]["t1"],
        "compute0": a["model"]["t1"] - a["model"]["t0"],
        "send-prep": send_begin - a["model"]["t1"],
        "peer-late": max(0.0, recv_begin - send_begin),
        "transit": recv_done - max(send_begin, recv_begin),
        "first-use": c1b - recv_done,
        "compute1": c1e - c1b,
        "pre-gather": g1b - c1e,
        "ag-xfer": a["ag_end"] - g1b,
    }
    comp = {k: v / 1000.0 for k, v in comp.items()}
    period = (a["ag_end"] - prev["ag_end"]) / 1000.0
    comp["period"] = period
    comp["unattributed"] = period - sum(v for k, v in comp.items() if k != "period")
    comp["ag-entry-skew"] = (g1b - a["ag_begin"]) / 1000.0
    comp["send-call"] = (a["send_end"] - a["send_begin"]) / 1000.0
    return comp


def rank_profile(r0, r1, al, skip=3):
    """Per-rank medians (us) used to compose an expected cross-host step from two single-machine runs (see compose_expected.py)."""
    out = {k: [] for k in ("resume0", "sampler0", "pre_compute0", "compute0", "send_prep0", "resume1", "sampler1", "pre_recv1", "first_use1", "compute1", "pre_gather1", "transit", "ag_xfer", "period")}
    for s_ in sorted(r0):
        if s_ < skip or s_ not in r1 or s_ - 1 not in r0 or s_ - 1 not in r1:
            continue
        a, b, pa, pb = r0[s_], r1[s_], r0[s_ - 1], r1[s_ - 1]
        c = components(r0, r1, al, s_)
        if not c or not (pb["sampler"] and b["pre_recv"] and pb["ag_end"]):
            continue
        t = lambda x: to_rank0(al, x)
        out["resume0"].append(c["resume"]); out["sampler0"].append(c["sampler"]); out["pre_compute0"].append(c["pre-compute"]); out["compute0"].append(c["compute0"])
        out["send_prep0"].append(c["send-prep"]); out["first_use1"].append(c["first-use"]); out["compute1"].append(c["compute1"]); out["pre_gather1"].append(c["pre-gather"])
        out["transit"].append(c["transit"] + c["peer-late"]); out["ag_xfer"].append(c["ag-xfer"]); out["period"].append(c["period"])
        out["resume1"].append((pb["sampler"]["t0"] - pb["ag_end"]) / 1000.0 if "ag_end" in pb and pb["ag_end"] else 0.0)
        out["sampler1"].append((pb["sampler"]["t1"] - pb["sampler"]["t0"]) / 1000.0)
        out["pre_recv1"].append((b["pre_recv"]["t1"] - pb["sampler"]["t1"]) / 1000.0)  # sampler end -> receive posted (graph build + pre-recv eval)
    return {k: statistics.median(v) for k, v in out.items() if v}


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rank0")
    ap.add_argument("rank1")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--steps", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--profile", default="", help="write the per-rank medians (for compose_expected.py) to this JSON file")
    a = ap.parse_args()
    d0, e0 = load(a.rank0)
    d1, e1 = load(a.rank1)
    al = align(d0.get("clock", {}))
    r0, r1 = by_step(e0, d0["backend"]), by_step(e1, d1["backend"])
    rows = []
    for s in sorted(r0):
        if s < a.skip:
            continue
        c = components(r0, r1, al, s)
        if c:
            c["step"] = s
            rows.append(c)
    print(f"{a.rank0.rsplit('/',1)[-1]} + {a.rank1.rsplit('/',1)[-1]}: backend {d0['backend']}, hosts {d0.get('host')} (rank 0) / {d1.get('host')} (rank 1), {len(rows)} steps")
    print(f"clock alignment: offset(pre) {al['offset']/1000:.1f} us, pre->post shift {al.get('shift', 0)/1000:.1f} us over {al.get('interval', 0):.1f} s (drift {al['drift']/1000:.2f} us/s), "
          f"uncertainty +/- {al['unc']/1000:.1f} us (smaller components are not resolved)")
    if not rows:
        print("no complete steps")
        return
    keys = ["resume", "sampler", "pre-compute", "compute0", "send-prep", "peer-late", "transit", "first-use", "compute1", "pre-gather", "ag-xfer", "unattributed", "period"]
    print(f"  {'component':14}{'median':>9}{'p25':>9}{'p75':>9}{'p95':>9}   (microseconds, per step)")
    for k in keys + ["ag-entry-skew", "send-call"]:
        v = [r[k] for r in rows]
        print(f"  {k:14}{statistics.median(v):9.0f}{q(v, .25):9.0f}{q(v, .75):9.0f}{q(v, .95):9.0f}" + ("   <- not part of the sum" if k in ("ag-entry-skew", "send-call") else ""))
    if a.profile:
        json.dump(rank_profile(r0, r1, al, a.skip), open(a.profile, "w"), indent=1)
    if a.steps:
        for s in [int(x) for x in a.steps.split(",")]:
            r = next((r for r in rows if r["step"] == s), None)
            if r:
                print(f"  step {s}: " + ", ".join(f"{k} {r[k]:.0f}" for k in keys))
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step"] + keys + ["ag-entry-skew", "send-call"])
            w.writeheader()
            for r in rows:
                w.writerow({k: round(r[k], 1) for k in w.fieldnames})


if __name__ == "__main__":
    main()

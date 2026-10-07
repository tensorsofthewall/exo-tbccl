"""Per-run metrics and repeated-run statistics for the final physical validation (both orientations).

    python benchmarks/phase65_stats.py --manifest docs/data/phase65/manifest.json [--out docs/data/phase65/stats.json]

manifest: {"runs": [{"name": "A1_base", "orient": "A", "mode": "base|policy|ring", "rep": 1, "dir": "physical", "mac_mtime_ns": N, "power": "power_a.txt.gz"}, ...]}
Per run (files <dir>/phys_<name>.rank{0,1}.json, .res.json.gz, <name>.linux.out): TPOT and match_ref, component medians (us) from the distributed trace (sampler,
resume + graph build, rank-0 stage, first-use, rank-1 stage, AllGather completion, step period with p25/p75), Mac process CPU per step, Mac main-thread CPU per step, the
runtime policy's helper CPU and window per step (watchdog windows removed), and CPU power / P-E residency / GPU from powermetrics over the decode steps.
Per (orientation, mode): mean, median, sd, min, max over repetitions; per repetition gap closure (baseline - policy) / (baseline - Ring) from the same repetition's runs.
"""
import argparse
import bisect
import gzip
import json
import math
import re
import statistics as st
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import compare_runs as cr  # noqa: E402
import distributed_timeline as dt  # noqa: E402
import powermetrics_report as pm  # noqa: E402

KEYS = ["sampler", "resume", "pre-compute", "compute0", "first-use", "compute1", "pre-gather", "ag-xfer", "peer-late", "transit"]


def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def interp(ts, vals, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return vals[0]
    if i >= len(ts):
        return vals[-1]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return vals[i - 1] + f * (vals[i] - vals[i - 1])


def run_metrics(r, base, powers):
    d = f"{base}/{r['dir']}/phys_{r['name']}"
    out = {"name": r["name"], "orient": r["orient"], "mode": r["mode"], "rep": r["rep"]}
    txts = [open(f"{base}/{r['dir']}/{r['name']}.{h}.out").read() for h in ("linux", "mac")]  # the reference check runs on the rank that holds it (Linux in A, the Mac in B)
    out["tpot_ms"] = float(re.search(r'"tpot_ms": ([\d.]+)', txts[0])[1])
    out["match_ref"] = any('"match_ref": true' in t for t in txts) and not any('"match_ref": false' in t for t in txts)
    steps, al = cr.steps(d + ".rank0.json", d + ".rank1.json")
    out["clock_unc_us"] = al["unc"] / 1000
    med = lambda k: st.median(c[k] for c in steps.values())
    for k in KEYS + ["period"]:
        out[k + "_us"] = med(k)
    per = sorted(c["period"] for c in steps.values())
    out["period_p25_us"], out["period_p75_us"] = per[len(per) // 4], per[(3 * len(per)) // 4]
    mac_rank = 1 if r["orient"] == "A" else 0
    rec, resp = f"{d}.rank{mac_rank}.json", f"{d}.res.json.gz"
    dd, ev = dt.load(rec)
    rr = json.load(_open(resp))
    out["sampler_gap_ms"] = {k: rr[k] for k in ("interval_ms", "gap_ms_median", "gap_ms_p95", "gap_ms_max")}
    rows = rr["rows"]
    ts = [x[0] for x in rows]
    cpu = [x[1] + x[2] for x in rows]
    main = min({t for x in rows for t in x[6]}, key=int)
    mcpu = [x[6].get(main, [0, 0])[0] + x[6].get(main, [0, 0])[1] for x in rows]
    by = {}
    for e in ev:
        if e["step"] >= 2 and not e["label"].startswith("prefill:") and e["label"] != "barrier" and e["depth"] == 0 and e["kind"] != "activity":
            by.setdefault(e["step"], []).append(e)
    ivs = [(min(e["t0"] for e in v), max(e["t1"] for e in v)) for s, v in sorted(by.items()) if not any(dt.DIGEST in e["label"] for e in v)]
    ivs = [(a, b) for a, b in ivs if b - a < 30e6]
    out["decode_steps"] = len(ivs)
    out["proc_cpu_ms_per_step"] = sum(interp(ts, cpu, b) - interp(ts, cpu, a) for a, b in ivs) / len(ivs) * 1e3
    out["main_cpu_ms_per_step"] = sum(interp(ts, mcpu, b) - interp(ts, mcpu, a) for a, b in ivs) / len(ivs) * 1e3
    out["step_wall_ms"] = sum(b - a for a, b in ivs) / len(ivs) / 1e6
    out["proc_cores"] = out["proc_cpu_ms_per_step"] / out["step_wall_ms"]
    ra = dd.get("runtime_activity")
    if ra and ra.get("activity_windows"):
        tmo = ra["fallback_timeouts"]
        w = max(1, ra["activity_windows"] - tmo)
        out["helper_cpu_ms_per_step"] = (ra["helper_cpu_us"] - tmo * 100000.0 * ra["duty"]) / 1e3 / w
        out["window_ms"] = (ra["activity_us"] - tmo * 100000.0) / w / 1e3
        out["windows"], out["watchdog_windows"] = ra["activity_windows"], tmo
    p = powers.get(r.get("power"))
    if p is not None and r.get("mac_mtime_ns"):
        s = pm.summarise(p, pm.decode_window(rec, resp, int(r["mac_mtime_ns"])))
        if s:
            out["power"] = {k: s[k] for k in ("p_active_pct", "p_freq_mhz", "e_active_pct", "cpu_mw", "gpu_mw", "gpu_mhz", "gpu_res_pct", "samples")}
            out["power"]["thermal"] = s["thermal"]
    return out


def agg(xs):
    return {"n": len(xs), "mean": st.mean(xs), "median": st.median(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    base = a.manifest.rsplit("/", 1)[0]
    man = json.load(open(a.manifest))
    powers = {}
    for r in man["runs"]:
        pth = r.get("power")
        if pth and pth not in powers:
            powers[pth] = pm.parse(f"{base}/{pth}")[0]
    runs = [run_metrics(r, base, powers) for r in man["runs"]]
    res = {"runs": runs, "summary": {}}
    for o in sorted({r["orient"] for r in runs}):
        print(f"\n=== Orientation {o} ===")
        print(f"{'run':14}{'TPOT':>8}{'first-use':>10}{'rk0 stg':>9}{'rk1 stg':>9}{'AG':>7}{'period':>8}{'proc ms':>8}{'cores':>6}{'helper':>7}{'CPU mW':>8}{'P%':>6}{'tok':>5}")
        for r in [x for x in runs if x["orient"] == o]:
            pw = r.get("power", {})
            print(f"{r['name']:14}{r['tpot_ms']:8.3f}{r['first-use_us']:10.0f}{r['compute0_us']:9.0f}{r['compute1_us']:9.0f}{r['ag-xfer_us']:7.0f}{r['period_us']:8.0f}{r['proc_cpu_ms_per_step']:8.2f}{r['proc_cores']:6.2f}{r.get('helper_cpu_ms_per_step', 0):7.2f}{pw.get('cpu_mw', float('nan')):8.0f}{pw.get('p_active_pct', float('nan')):6.1f}{'ok' if r['match_ref'] else 'BAD':>5}")
        for m in ("base", "policy", "ring"):
            xs = [r for r in runs if r["orient"] == o and r["mode"] == m]
            if not xs:
                continue
            s = {k: agg([r[k] for r in xs]) for k in ("tpot_ms", "first-use_us", "compute0_us", "compute1_us", "ag-xfer_us", "sampler_us", "resume_us", "pre-compute_us", "period_us", "proc_cpu_ms_per_step", "proc_cores")}
            if all("power" in r for r in xs):
                s["cpu_mw"] = agg([r["power"]["cpu_mw"] for r in xs])
            if all("helper_cpu_ms_per_step" in r for r in xs):
                s["helper_cpu_ms_per_step"] = agg([r["helper_cpu_ms_per_step"] for r in xs])
            res["summary"][f"{o}_{m}"] = s
            t = s["tpot_ms"]
            print(f"  {m:7} TPOT mean {t['mean']:.3f} median {t['median']:.3f} sd {t['sd']:.3f} min {t['min']:.3f} max {t['max']:.3f} (n={t['n']})")
        reps = sorted({r["rep"] for r in runs if r["orient"] == o})
        gc = []
        for k in reps:
            g = {r["mode"]: r["tpot_ms"] for r in runs if r["orient"] == o and r["rep"] == k}
            if all(m in g for m in ("base", "policy", "ring")):
                gc.append((g["base"] - g["policy"]) / (g["base"] - g["ring"]))
                print(f"  rep {k}: gap closed {100 * gc[-1]:.0f} %  policy - Ring {g['policy'] - g['ring']:+.3f} ms")
        if gc:
            res["summary"][f"{o}_gap_closed"] = agg(gc)
            pol, ring = [r["tpot_ms"] for r in runs if r["orient"] == o and r["mode"] == "policy"], [r["tpot_ms"] for r in runs if r["orient"] == o and r["mode"] == "ring"]
            res["summary"][f"{o}_policy_minus_ring_median_ms"] = st.median(pol) - st.median(ring)
            print(f"  median policy - median Ring {st.median(pol) - st.median(ring):+.3f} ms; gap closed mean {100 * st.mean(gc):.0f} % (min {100 * min(gc):.0f} %)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Physical Orientation-B composition statistics (BASE, STEP, SPIN, BOTH, Ring), per run and over repetitions.

    python benchmarks/phase66_stats.py --manifest docs/data/phase66/manifest.json [--out docs/data/phase66/stats.json]

manifest: {"runs": [{"name": "B1_base", "orient": "B", "mode": "base|step|spin|both|ring", "rep": 1, "dir": "physical", "mac_mtime_ns": N, "power": "power.txt.gz"}, ...]}
Per run (files <dir>/phys_<name>.rank{0,1}.json, .res.json.gz, <name>.linux.out / .mac.out): everything phase65_stats.py reports (component medians from the distributed trace,
Mac process CPU per step, helper CPU, powermetrics) plus the Mac AllGather Work wait (the wait:all_gather call for TBCCL; the post_allgather_eval for Ring), the caller's WAIT_SPIN
time per step, the runtime helper's open time per step and their overlap (the instants both burn a core), from the recorder's `helper` and `wait_spin` events.
Per mode: mean, median, sd, min, max; per repetition gap closure (BASE - mode) / (BASE - Ring) from the same repetition and paired TPOT differences BOTH-STEP and BOTH-SPIN.
"""
import argparse
import json
import re
import statistics as st
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import compare_runs as cr  # noqa: E402
import distributed_timeline as dt  # noqa: E402
import phase65_stats as p5  # noqa: E402
import powermetrics_report as pm  # noqa: E402

AUX = ("activity", "helper", "wait_spin")
MODES = ("base", "step", "spin", "both", "ring")


def _clip(ivs, a, b):
    return [(max(s, a), min(e, b)) for s, e in ivs if min(e, b) > max(s, a)]


def _total(ivs):
    return sum(e - s for s, e in ivs)


def _intersect(x, y):
    out = []
    for s1, e1 in x:
        for s2, e2 in y:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return out


def helper_intervals(ev, cap_ns):
    """Open instants -> next close instant (an open while already open changes nothing; a window never lasts longer than the runtime watchdog)."""
    out, start = [], None
    for e in sorted((e for e in ev if e["kind"] == "helper"), key=lambda e: e["t0"]):
        if e["label"] == "open" and start is None:
            start = e["t0"]
        elif e["label"] == "close_window" and start is not None:
            out.append((start, min(e["t0"], start + cap_ns)))
            start = None
    return out


def run_metrics(r, base, powers, max_ms=100.0):
    d = f"{base}/{r['dir']}/phys_{r['name']}"
    out = {"name": r["name"], "orient": r["orient"], "mode": r["mode"], "rep": r["rep"]}
    txts = [open(f"{base}/{r['dir']}/{r['name']}.{h}.out").read() for h in ("linux", "mac")]
    out["tpot_ms"] = float(re.search(r'"tpot_ms": ([\d.]+)', txts[0])[1])
    out["match_ref"] = any('"match_ref": true' in t for t in txts) and not any('"match_ref": false' in t for t in txts)
    toks = [re.search(r'"all_tokens": (\[[^\]]*\])', t) for t in txts]
    out["tokens_agree"] = bool(toks[0] and toks[1] and toks[0][1] == toks[1][1])
    steps, al = cr.steps(d + ".rank0.json", d + ".rank1.json")
    out["clock_unc_us"] = al["unc"] / 1000
    med = lambda k: st.median(c[k] for c in steps.values())
    for k in p5.KEYS + ["period", "send-prep", "compute0", "compute1", "ag-xfer"]:
        out[k + "_us"] = med(k)
    per = sorted(c["period"] for c in steps.values())
    out["period_p25_us"], out["period_p75_us"] = per[len(per) // 4], per[(3 * len(per)) // 4]
    mac_rank = 1 if r["orient"] == "A" else 0
    rec, resp = f"{d}.rank{mac_rank}.json", f"{d}.res.json.gz"
    dd, ev = dt.load(rec)
    rr = json.load(p5._open(resp))
    out["sampler_gap_ms"] = {k: rr[k] for k in ("interval_ms", "gap_ms_median", "gap_ms_p95", "gap_ms_max")}
    rows = rr["rows"]
    ts = [x[0] for x in rows]
    cpu = [x[1] + x[2] for x in rows]
    main = min({t for x in rows for t in x[6]}, key=int)
    mcpu = [x[6].get(main, [0, 0])[0] + x[6].get(main, [0, 0])[1] for x in rows]
    by = {}
    for e in ev:
        if e["step"] >= 2 and not e["label"].startswith("prefill:") and e["label"] != "barrier" and e["depth"] == 0 and e["kind"] not in AUX:
            by.setdefault(e["step"], []).append(e)
    sel = [(s, (min(e["t0"] for e in v), max(e["t1"] for e in v))) for s, v in sorted(by.items()) if not any(dt.DIGEST in e["label"] for e in v)]
    sel = [(s, iv) for s, iv in sel if iv[1] - iv[0] < 30e6]
    ivs = [iv for _, iv in sel]
    out["decode_steps"] = len(ivs)
    out["proc_cpu_ms_per_step"] = sum(p5.interp(ts, cpu, b) - p5.interp(ts, cpu, a) for a, b in ivs) / len(ivs) * 1e3
    out["main_cpu_ms_per_step"] = sum(p5.interp(ts, mcpu, b) - p5.interp(ts, mcpu, a) for a, b in ivs) / len(ivs) * 1e3
    out["step_wall_ms"] = sum(b - a for a, b in ivs) / len(ivs) / 1e6
    out["proc_cores"] = out["proc_cpu_ms_per_step"] / out["step_wall_ms"]
    # Mac AllGather Work wait
    if dd["backend"] == "ring":
        w = [e for e in ev if e["kind"] == "eval" and e["label"] == "post_allgather_eval" and e["depth"] == 0 and e["step"] >= 2]
    else:
        w = [e for e in ev if e["kind"] == "tbccl_wait" and e["label"] == "wait:all_gather" and e["step"] >= 2]
    out["ag_wait_us"] = st.median((e["t1"] - e["t0"]) / 1000 for e in w) if w else float("nan")
    # helper / spin activity on the Mac
    hi = helper_intervals(ev, max_ms * 1e6)
    sp = sorted(((e["t0"], e["t1"]) for e in ev if e["kind"] == "wait_spin"))
    tot_h = tot_s = tot_o = tot_w = 0.0
    for a, b in ivs:
        h, s = _clip(hi, a, b), _clip(sp, a, b)
        tot_h, tot_s, tot_o, tot_w = tot_h + _total(h), tot_s + _total(s), tot_o + _total(_intersect(h, s)), tot_w + (b - a)
    n = len(ivs)
    out["helper_open_ms_per_step"], out["spin_ms_per_step"], out["overlap_ms_per_step"] = tot_h / n / 1e6, tot_s / n / 1e6, tot_o / n / 1e6
    out["helper_only_ms"], out["spin_only_ms"] = (tot_h - tot_o) / n / 1e6, (tot_s - tot_o) / n / 1e6
    out["idle_ms"] = (tot_w - (tot_h + tot_s - tot_o)) / n / 1e6
    out["helper_cover_pct"], out["spin_cover_pct"], out["union_cover_pct"] = 100 * tot_h / tot_w, 100 * tot_s / tot_w, 100 * (tot_h + tot_s - tot_o) / tot_w
    ra = dd.get("runtime_activity")
    if ra and ra.get("activity_windows"):
        tmo = ra["fallback_timeouts"]
        wn = max(1, ra["activity_windows"] - tmo)
        out["helper_cpu_ms_per_step"] = (ra["helper_cpu_us"] - tmo * max_ms * 1000.0 * ra["duty"]) / 1e3 / wn
        out["window_ms"] = (ra["activity_us"] - tmo * max_ms * 1000.0) / wn / 1e3
        out["windows"], out["watchdog_windows"] = ra["activity_windows"], tmo
    p = powers.get(r.get("power"))
    if p is not None and r.get("mac_mtime_ns"):
        s = pm.summarise(p, pm.decode_window(rec, resp, int(r["mac_mtime_ns"])))
        if s:
            out["power"] = {k: s[k] for k in ("p_active_pct", "p_freq_mhz", "e_active_pct", "cpu_mw", "gpu_mw", "gpu_mhz", "gpu_res_pct", "samples")}
            out["power"]["thermal"] = s["thermal"]
    return out


SUMMARY_KEYS = ("tpot_ms", "sampler_us", "resume_us", "pre-compute_us", "compute0_us", "send-prep_us", "first-use_us", "compute1_us", "ag-xfer_us", "ag_wait_us", "period_us",
                "proc_cpu_ms_per_step", "proc_cores", "helper_open_ms_per_step", "spin_ms_per_step", "overlap_ms_per_step", "helper_only_ms", "spin_only_ms", "idle_ms",
                "union_cover_pct", "helper_cpu_ms_per_step")


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
    print(f"{'run':10}{'TPOT':>8}{'sampler':>8}{'resume':>7}{'graph':>7}{'stage':>7}{'AGwait':>8}{'AGxfer':>7}{'period':>8}{'proc ms':>8}{'cores':>6}{'helper':>7}{'spin':>6}{'ovlp':>6}{'CPU mW':>8}{'tok':>5}")
    for r in runs:
        pw = r.get("power", {})
        print(f"{r['name']:10}{r['tpot_ms']:8.3f}{r['sampler_us']:8.0f}{r['resume_us']:7.0f}{r['pre-compute_us']:7.0f}{r['compute0_us']:7.0f}{r['ag_wait_us']:8.0f}{r['ag-xfer_us']:7.0f}{r['period_us']:8.0f}"
              f"{r['proc_cpu_ms_per_step']:8.2f}{r['proc_cores']:6.2f}{r['helper_open_ms_per_step']:7.2f}{r['spin_ms_per_step']:6.2f}{r['overlap_ms_per_step']:6.2f}{pw.get('cpu_mw', float('nan')):8.0f}"
              f"{'ok' if r['match_ref'] and r['tokens_agree'] else 'BAD':>5}")
    for m in MODES:
        xs = [r for r in runs if r["mode"] == m]
        if not xs:
            continue
        s = {k: p5.agg([r[k] for r in xs]) for k in SUMMARY_KEYS if all(k in r for r in xs)}
        if all("power" in r for r in xs):
            s["cpu_mw"] = p5.agg([r["power"]["cpu_mw"] for r in xs])
        res["summary"][m] = s
        t = s["tpot_ms"]
        print(f"  {m:5} TPOT mean {t['mean']:.3f} median {t['median']:.3f} sd {t['sd']:.3f} min {t['min']:.3f} max {t['max']:.3f} (n={t['n']})")
    reps = sorted({r["rep"] for r in runs})
    closure = {m: [] for m in ("step", "spin", "both")}
    paired = {"both-step": [], "both-spin": [], "both-ring": []}
    for k in reps:
        g = {r["mode"]: r["tpot_ms"] for r in runs if r["rep"] == k}
        if not all(m in g for m in ("base", "ring")):
            continue
        for m in closure:
            if m in g:
                closure[m].append((g["base"] - g[m]) / (g["base"] - g["ring"]))
        if "both" in g and "step" in g:
            paired["both-step"].append(g["both"] - g["step"])
        if "both" in g and "spin" in g:
            paired["both-spin"].append(g["both"] - g["spin"])
        if "both" in g:
            paired["both-ring"].append(g["both"] - g["ring"])
        print(f"  rep {k}: closure " + ", ".join(f"{m} {100 * (g['base'] - g[m]) / (g['base'] - g['ring']):.0f} %" for m in closure if m in g) + f"   (BASE {g['base']:.2f}, Ring {g['ring']:.2f})")
    for m, xs in closure.items():
        if xs:
            res["summary"][f"closure_{m}"] = p5.agg(xs)
            print(f"  gap closed {m}: mean {100 * st.mean(xs):.0f} % median {100 * st.median(xs):.0f} % min {100 * min(xs):.0f} % max {100 * max(xs):.0f} %")
    for k, xs in paired.items():
        if xs:
            res["summary"][f"paired_{k}_ms"] = p5.agg(xs)
            print(f"  paired TPOT {k}: mean {st.mean(xs):+.3f} ms, sd {st.stdev(xs) if len(xs) > 1 else 0:.3f}, per rep " + ", ".join(f"{x:+.2f}" for x in xs))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

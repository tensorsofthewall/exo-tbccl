"""Phase 62: per-token-window CPU activity from a recorder dump plus a mac_resource_sampler dump (same machine, same perf_counter_ns clock).

    python benchmarks/resource_report.py --rec <run.rank1.json> --res <run.res.json> [--label X] [--json out.json]

Per decode step (the first two skipped, KV-digest steps excluded) four windows on the Mac rank:
  A  previous post-AllGather eval end -> post_recv_eval end (recv complete; sampler + graph build + wait for the peer)
  B  recv complete -> model_output_eval begin           (first use)
  C  model_output_eval                                   (the Mac stage)
  D  model_output_eval end -> post_allgather_eval end    (pre-gather + AllGather + its eval)
For each window: process CPU cores (CPU seconds / wall seconds, interpolated between samples), the Python main thread (lowest thread id)'s cores, the sum of all other threads'
cores, involuntary context switches per ms, and the number of sampler rows inside. Medians over steps. Sampling limits are in the dump header (gap_ms_*); windows
shorter than ~2x the sampling gap are noisy and are flagged by their row counts.
"""
import argparse
import gzip
import bisect
import json
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import distributed_timeline as dt  # noqa: E402

WINDOWS = ["A prev-AG->recv", "B first-use", "C stage", "D post-stage->AG"]


def interp(ts, vals, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return vals[0]
    if i >= len(ts):
        return vals[-1]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return vals[i - 1] + f * (vals[i] - vals[i - 1])


def analyse(rec_path, res_path):
    d, ev = dt.load(rec_path)
    r = json.load(gzip.open(res_path, "rt") if res_path.endswith(".gz") else open(res_path))
    rows = r["rows"]
    ts = [x[0] for x in rows]
    proc = [x[1] + x[2] for x in rows]
    inv = [x[4] for x in rows]
    vol = [x[3] for x in rows]
    tids = sorted({t for x in rows for t in x[6]})
    thr = {t: [(x[6].get(t, [0, 0])[0] + x[6].get(t, [0, 0])[1]) for x in rows] for t in tids}
    msys = [x[6].get(min(tids, key=int), [0, 0])[1] for x in rows]
    tot = {t: thr[t][-1] - thr[t][0] for t in tids}
    main = min(tids, key=int)  # the Python main thread is the first (lowest) thread id; the Ring worker / TBCCL workers are later threads
    steps = {}
    for e in ev:
        if e["step"] < 0 or e["label"].startswith("prefill:") or e["depth"] != 0:
            continue
        if dt.DIGEST in e["label"]:
            continue
        steps.setdefault(e["step"], {})[(e["kind"], e["label"])] = e
    order = sorted(steps)
    per = {w: [] for w in WINDOWS}
    prev_ag = None
    for s in order:
        g = steps[s]
        rc, mo, ag = g.get(("eval", "post_recv_eval")), g.get(("eval", "model_output_eval")), g.get(("eval", "post_allgather_eval"))
        if not (rc and mo and ag):
            prev_ag = ag["t1"] if ag else prev_ag
            continue
        win = {WINDOWS[1]: (rc["t1"], mo["t0"]), WINDOWS[2]: (mo["t0"], mo["t1"]), WINDOWS[3]: (mo["t1"], ag["t1"])}
        if prev_ag is not None:
            win[WINDOWS[0]] = (prev_ag, rc["t1"])
        prev_ag = ag["t1"]
        if s < 2:
            continue
        for w, (a, b) in win.items():
            if b <= a:
                continue
            wall = (b - a) / 1e9
            pc = (interp(ts, proc, b) - interp(ts, proc, a)) / wall
            mc = (interp(ts, thr[main], b) - interp(ts, thr[main], a)) / wall
            iv = (interp(ts, inv, b) - interp(ts, inv, a)) / ((b - a) / 1e6)
            vv = (interp(ts, vol, b) - interp(ts, vol, a)) / ((b - a) / 1e6)
            i0, i1 = bisect.bisect_left(ts, a), bisect.bisect_right(ts, b)
            n = i1 - i0
            mr = statistics.mean((rows[i][6].get(main, [0, 0, 0])[2] == 1) for i in range(i0, i1)) if n else float("nan")
            orun = statistics.mean(sum(1 for t, v in rows[i][6].items() if t != main and v[2] == 1) for i in range(i0, i1)) if n else float("nan")
            ms = (interp(ts, msys, b) - interp(ts, msys, a)) / wall
            per[w].append((wall * 1e6, pc, mc, pc - mc, iv, vv, n, mr, orun, ms))
    out = {"backend": d["backend"], "main_tid": main, "threads_cpu_s": {f"{t}:{r.get('thread_names', {}).get(t, '')}": round(v, 4) for t, v in tot.items() if v > 0.002},
           "sampler": {k: r[k] for k in ("interval_ms", "n", "gap_ms_median", "gap_ms_p95", "gap_ms_max")}, "windows": {}}
    for w, v in per.items():
        if v:
            med = [statistics.median(x[i] for x in v) for i in range(10)]
            out["windows"][w] = dict(steps=len(v), wall_us=med[0], proc_cores=med[1], main_cores=med[2], other_cores=med[3], invcs_per_ms=med[4], volcs_per_ms=med[5], rows=med[6], main_running=med[7], other_running=med[8], main_sys_cores=med[9])
    act = d.get("activity")
    if act:  # Phase 63: the benchmark-only activity helper's own accounting, restricted to the decode steps (>=2, no digest steps, < 30 ms)
        ivs = []
        for st, v in steps.items():
            es = [e for e in v.values() if e["kind"] != "activity"]
            if st >= 2 and es and not any(dt.DIGEST in k[1] for k in v):
                a0, a1 = min(e["t0"] for e in es), max(e["t1"] for e in es)
                if a1 - a0 < 30e6:
                    ivs.append((a0, a1))
        ov = lambda b0, b1: sum(max(0, min(b1, y) - max(b0, x)) for x, y in ivs)
        n = max(1, len(ivs))
        step_ns = sum(y - x for x, y in ivs)
        act_ns = sum(ov(b[0], b[1]) for b in act["bursts"])
        cpu_ns = sum(b[2] * ov(b[0], b[1]) / max(1, b[1] - b[0]) for b in act["bursts"])
        hs = [x[6].get(str(act["native_id"]), [0, 0])[0] + x[6].get(str(act["native_id"]), [0, 0])[1] for x in rows]
        out["activity"] = {"mode": act["mode"], "duty": act["duty"], "steps": len(ivs), "active_ms_per_step": act_ns / n / 1e6, "helper_cpu_ms_per_step": cpu_ns / n / 1e6,
                           "active_fraction_of_step_time": act_ns / max(1, step_ns), "helper_cpu_s_total_run": sum(b[2] for b in act["bursts"]) / 1e9,
                           "sampler_helper_thread_cpu_s": (max(hs) - next((h for h in hs if h > 0), 0)) if hs else None}
    ra = d.get("runtime_activity")
    if ra and ra.get("activity_windows"):
        tmo = ra["fallback_timeouts"]
        mx = 100000.0  # the default watchdog (us): KV-digest windows of the benchmark driver run into it and are removed from the per-step figures
        w = max(1, ra["activity_windows"] - tmo)
        out["runtime_activity"] = dict(ra, per_window_ms=(ra["activity_us"] - tmo * mx) / w / 1e3, helper_cpu_ms_per_window=ra["helper_cpu_us"] / max(1, ra["activity_windows"]) / 1e3)
    tw = r["rows"][-1][0] - r["rows"][0][0]
    out["run"] = {"wall_s": tw / 1e9, "cpu_s": (proc[-1] - proc[0]), "invcs": inv[-1] - inv[0], "volcs": vol[-1] - vol[0]}
    return out


def show(o, label):
    print(f"{label}: backend {o['backend']} main tid {o['main_tid']} sampler gap median {o['sampler']['gap_ms_median']:.2f} ms p95 {o['sampler']['gap_ms_p95']:.2f} max {o['sampler']['gap_ms_max']:.1f}")
    print(f"  threads with CPU (s over the whole sampled run): {o['threads_cpu_s']}")
    print(f"  {'window':20}{'steps':>6}{'wall us':>9}{'proc':>7}{'main':>7}{'other':>7}{'inv/ms':>8}{'vol/ms':>8}{'rows':>6}{'mainRun':>8}{'othRun':>7}{'mainSys':>8}")
    if o.get("activity"):
        a = o["activity"]
        print(f"  activity {a['mode']} duty {a['duty']}: {a['steps']} steps, active {a['active_ms_per_step']:.2f} ms/step ({100 * a['active_fraction_of_step_time']:.0f} % of step time), helper CPU {a['helper_cpu_ms_per_step']:.2f} ms/step, run total {a['helper_cpu_s_total_run']:.2f} s (sampler view of the helper thread {a['sampler_helper_thread_cpu_s']:.2f} s)")
    if o.get("runtime_activity"):
        a = o["runtime_activity"]
        print(f"  runtime activity: {a['activity_windows']} windows ({a['fallback_timeouts']} watchdog), duty {a['duty']}, ~{a['per_window_ms']:.2f} ms/window (watchdog windows removed), helper CPU total {a['helper_cpu_us'] / 1e6:.2f} s")
    for w, v in o["windows"].items():
        print(f"  {w:20}{v['steps']:>6}{v['wall_us']:>9.0f}{v['proc_cores']:>7.2f}{v['main_cores']:>7.2f}{v['other_cores']:>7.2f}{v['invcs_per_ms']:>8.2f}{v['volcs_per_ms']:>8.2f}{v['rows']:>6.0f}{v['main_running']:>8.2f}{v['other_running']:>7.2f}{v['main_sys_cores']:>8.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    o = analyse(a.rec, a.res)
    show(o, a.label or a.rec)
    if a.json:
        json.dump(o, open(a.json, "w"), indent=1)

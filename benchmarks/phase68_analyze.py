"""Phase 68: aggregate the real-exo sessions (docs/data/phase68/raw/<label>/) into the comparison tables.

    python benchmarks/phase68_analyze.py docs/data/phase68 [--out docs/data/phase68/analysis.json]

Per session: orientation (A = Linux rank 0 / Mac rank 1, B = Mac rank 0 / Linux rank 1) and exo's layer split, load time, TTFT per repetition (rep 1 is cold: its prompt is not in the
KV prefix cache; later repetitions hit it), prefill duration from the runner log (Starting prefill -> KV cache added), decode TPOT from the streamed token arrival times (pooled over
the repetitions, first 3 tokens of each dropped; median / p25 / p75 / p95), token identity across repetitions, peak Linux VRAM / process RSS, minimum Mac available memory and swap,
and the process-tree CPU cores over the generation window (needs the absolute timestamps of the later sessions). Sessions are then grouped by (orientation, configuration).
"""
import json
import os
import re
import statistics as st
import sys


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def tree_cpu(samples, t0, t1):
    """Cores used by the process tree between t0 and t1: sum over pids present at both window ends of the CPU-seconds delta / wall."""
    if not samples:
        return None
    a = min(samples, key=lambda s: abs(s["t"] - t0))
    b = min(samples, key=lambda s: abs(s["t"] - t1))
    ca = {p["pid"]: p["cpu_s"] for p in a["procs"]}
    d = sum(p["cpu_s"] - ca[p["pid"]] for p in b["procs"] if p["pid"] in ca)
    return d / max(b["t"] - a["t"], 1e-6)


def prefill_durations(log_path):
    """Seconds between 'Starting prefill' and 'KV cache added' for each request, from a runner log (ANSI stripped)."""
    if not os.path.exists(log_path):
        return []
    from datetime import datetime

    start, out = None, []
    for line in open(log_path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)", line)
        if not m:
            continue
        t = datetime.strptime(m[1], "%H:%M:%S.%f").timestamp()
        if "Starting prefill" in line:
            start = t
        elif "KV cache added" in line and start is not None:
            out.append(t - start)
            start = None
    return out


def session(d, label):
    r = json.load(open(f"{d}/{label}/result.json"))
    lay = [(s["device_rank"], s["n_layers"]) for s in r["layout"]]
    orient = "A" if lay[0][1] < lay[1][1] else "B"
    runs = next(iter(r["runs"].values()))
    name = next(iter(r["runs"]))
    gaps = []
    for x in runs:
        tt = x["token_times"]
        gaps += [b - a for a, b in zip(tt[3:], tt[4:])]
    cfg = "ring" if label.startswith("ring") else ("tbccl_both" if "both" in label else "tbccl_step" if "step" in label else "tbccl")
    out = {"label": label, "cfg": cfg, "orient": orient, "layers": lay, "load_s": r.get("load_s"), "prompt": name, "ttft": [x["ttft"] for x in runs], "errors": [x["error"] for x in runs if x["error"]],
           "usage": [x["usage"] for x in runs], "finish": [x["finish"] for x in runs], "tokens_identical_across_reps": all(x["tokens"] == runs[0]["tokens"] for x in runs),
           "tpot_median_ms": 1e3 * st.median(gaps) if gaps else None, "tpot_p25_ms": 1e3 * q(gaps, 0.25) if gaps else None, "tpot_p75_ms": 1e3 * q(gaps, 0.75) if gaps else None,
           "tpot_p95_ms": 1e3 * q(gaps, 0.95) if gaps else None, "n_gaps": len(gaps), "text": runs[0]["text"], "tokens": runs[0]["tokens"]}
    ml, mm = jl(f"{d}/{label}/mon_linux.jsonl"), jl(f"{d}/{label}/mon_mac.jsonl")
    if ml:
        out["linux_gpu_peak_mib"] = max(x.get("gpu", {}).get("used_mib", 0) for x in ml)
        out["linux_gpu_loaded_mib"] = st.median(x["gpu"]["used_mib"] for x in ml if x.get("gpu") and x["gpu"]["used_mib"] > 3000) if any(x.get("gpu") and x["gpu"]["used_mib"] > 3000 for x in ml) else None
        out["linux_rss_peak_gb"] = max(sum(p["rss"] for p in x["procs"]) for x in ml) / 1e9
        out["linux_gpu_temp_max"] = max(x.get("gpu", {}).get("temp_c", 0) for x in ml)
    if mm:
        out["mac_min_avail_gb"] = min(x["avail"] for x in mm) / 1e9
        out["mac_swap_max_gb"] = max(x["swap_used"] for x in mm) / 1e9
        out["mac_rss_peak_gb"] = max(sum(p["rss"] for p in x["procs"]) for x in mm) / 1e9
    if "t_abs_ready" in r:
        t0, t1 = runs[0]["t_abs_start"], runs[-1]["t_abs_end"]
        out["linux_cores"], out["mac_cores"] = tree_cpu(ml, t0, t1), tree_cpu(mm, t0, t1)
        gw = [x["gpu"]["util_pct"] for x in ml if t0 <= x["t"] <= t1 and x.get("gpu")]
        out["linux_gpu_util_mean"] = st.mean(gw) if gw else None
    pf = prefill_durations(f"{d}/{label}/mac_exo.log") or prefill_durations(f"{d}/{label}/mac_stdout.log")
    out["prefill_s_per_request_mac_log"] = [round(x, 2) for x in pf]
    b, a = (open(f"{d}/{label}/aer_{k}.txt").read() for k in ("before", "after"))
    g = lambda t, k: int(re.search(k + r"=(\d+)", t)[1])
    out["aer_delta"] = {"timeout": g(a, "Timeout") - g(b, "Timeout"), "correctable": g(a, "cor_total") - g(b, "cor_total"), "nonfatal": g(a, "nonfatal") - g(b, "nonfatal"), "fatal": g(a, "fatal") - g(b, "fatal")}
    return out


def main():
    d = sys.argv[1] + "/raw"
    labels = sorted(x for x in os.listdir(d) if re.match(r"(ring|tbccl)(_step|_both)?_p\d+$", x) and os.path.exists(f"{d}/{x}/result.json"))
    S = [session(d, x) for x in labels]
    print(f"{'session':16}{'or':>3}{'layers':>12}{'load s':>7}{'TTFT cold/warm s':>20}{'TPOT med/p25/p75/p95 ms':>28}{'GPU MiB':>8}{'Mac swap':>9}{'LinCPU':>7}{'MacCPU':>7} identical errs")
    for s in S:
        ttft = "/".join(f"{x:.1f}" for x in s["ttft"] if x)
        print(f"{s['label']:16}{s['orient']:>3}{str(tuple(l[1] for l in s['layers'])):>12}{s['load_s']:7.0f}{ttft:>20}{s['tpot_median_ms']:9.1f}{s['tpot_p25_ms']:7.1f}{s['tpot_p75_ms']:7.1f}{s['tpot_p95_ms']:7.1f}"
              f"{s.get('linux_gpu_peak_mib', 0):8}{s.get('mac_swap_max_gb', 0):9.2f}{(s.get('linux_cores') or float('nan')):7.2f}{(s.get('mac_cores') or float('nan')):7.2f}  {s['tokens_identical_across_reps']}  {len(s['errors'])}")
    groups = {}
    for s in S:
        groups.setdefault((s["orient"], s["cfg"]), []).append(s)
    print("\nBy orientation and configuration (sessions pooled; TPOT = median of the per-session medians, with the range):")
    print(f"{'orient':7}{'config':12}{'n':>3}{'TPOT med ms':>12}{'range':>16}{'p95 med':>9}{'TTFT cold':>10}{'prefill s':>10}{'Linux GPU MiB':>14}{'Mac min avail GB':>17}")
    res = {}
    for k in sorted(groups):
        g = groups[k]
        tp = [s["tpot_median_ms"] for s in g]
        res[f"{k[0]}_{k[1]}"] = {"n": len(g), "tpot_median_ms": st.median(tp), "tpot_min": min(tp), "tpot_max": max(tp), "tpot_p95_ms": st.median(s["tpot_p95_ms"] for s in g), "sessions": [s["label"] for s in g],
                                  "ttft_cold_s": st.median(s["ttft"][0] for s in g), "linux_gpu_peak_mib": max(s.get("linux_gpu_peak_mib", 0) for s in g), "mac_min_avail_gb": min(s.get("mac_min_avail_gb", 99) for s in g)}
        pf = [x for s in g for x in s["prefill_s_per_request_mac_log"][:1]]
        print(f"{k[0]:7}{k[1]:12}{len(g):3}{st.median(tp):12.1f}{min(tp):8.1f}-{max(tp):.1f}{st.median(s['tpot_p95_ms'] for s in g):9.1f}{st.median(s['ttft'][0] for s in g):10.1f}{(st.median(pf) if pf else float('nan')):10.2f}"
              f"{max(s.get('linux_gpu_peak_mib', 0) for s in g):14}{min(s.get('mac_min_avail_gb', 99) for s in g):17.2f}")
    for o in "AB":
        base = res.get(f"{o}_ring")
        if not base:
            continue
        for c in ("tbccl", "tbccl_step", "tbccl_both"):
            x = res.get(f"{o}_{c}")
            if x:
                print(f"  orientation {o}: {c} - ring = {x['tpot_median_ms'] - base['tpot_median_ms']:+.1f} ms ({100 * (x['tpot_median_ms'] / base['tpot_median_ms'] - 1):+.1f} %)")
    if "--out" in sys.argv:
        json.dump({"sessions": S, "groups": res}, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

"""Phase 62: parse a `powermetrics --samplers cpu_power,gpu_power,thermal -i 100` text capture and summarise it over each run's DECODE window.

    python benchmarks/powermetrics_report.py --power p62_power.txt --run <label> <rank1 recorder.json> <res.json> <res_mtime_ns> [--run ...] [--tz-hours 5.5]

Clock alignment (no sub-second wall clock is printed by powermetrics): sample k ends at T0 + cumulative elapsed time; T0 is fitted so that every printed
timestamp second equals floor(end of its sample) (a constraint set that pins T0 to a few ms when the capture crosses many second boundaries). A run's decode window
is taken from the Mac recorder (first decode step begin -> last decode step end, perf_counter ns) and put on the wall clock through the sampler dump's mtime (the
dump is written right after the target exits, i.e. at the sampler's last row). Accuracy ~ one sample (100 ms) -- a 16-token decode window is only ~2-3 samples, so
these are run-level frequency/power states, not per-step values. Samples are weighted by their overlap with the window.
"""
import argparse
import calendar
import json
import re
import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import distributed_timeline as dt  # noqa: E402

HDR = re.compile(r"\*\*\* Sampled system activity \((\w+ \w+ +\d+ \d+:\d+:\d+ \d+) ([+-]\d{4})\) \(([\d.]+)ms elapsed\)")


def parse(path):
    txt = open(path, errors="replace").read()
    parts = HDR.split(txt)
    out = []
    for i in range(1, len(parts), 4):
        stamp, tz, el, body = parts[i], parts[i + 1], float(parts[i + 2]), parts[i + 3]
        t = time.strptime(stamp, "%a %b %d %H:%M:%S %Y")
        sign = 1 if tz[0] == "+" else -1
        off = sign * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
        sec = calendar.timegm(t) - off
        s = dict(sec=sec, elapsed_ms=el)
        cpus = {}
        for m in re.finditer(r"CPU (\d+) frequency: ([\d.]+) MHz\nCPU \1 active residency:\s+([\d.]+)%", body):
            cpus[int(m[1])] = (float(m[2]), float(m[3]))
        s["cpus"] = cpus
        for name, pat in (("cpu_mw", r"CPU Power: ([\d.]+) mW"), ("gpu_mw", r"GPU Power: ([\d.]+) mW"), ("gpu_mhz", r"GPU HW active frequency: ([\d.]+) MHz"),
                          ("gpu_res", r"GPU HW active residency:\s+([\d.]+)%"), ("pcl_res", r"P-Cluster HW active residency:\s+([\d.]+)%")):
            m = re.search(pat, body)
            s[name] = float(m[1]) if m else None
        m = re.search(r"Current pressure level: (\w+)", body)
        s["thermal"] = m[1] if m else None
        m = re.search(r"GPU SW requested state: \(([^)]*)\)", body)
        s["gpu_sw"] = {k: float(v) for k, v in re.findall(r"(P\d+) :\s+([\d.]+)%", m[1])} if m else {}
        out.append(s)
    # fit T0 (seconds since epoch of the capture start): sample k ends at T0 + cum_k; printed second = floor(end)
    cum, lo, hi = 0.0, -1e18, 1e18
    for s in out:
        cum += s["elapsed_ms"] / 1000.0
        s["cum"] = cum
        lo = max(lo, s["sec"] - cum)
        hi = min(hi, s["sec"] + 1 - cum)
    t0 = (lo + hi) / 2 if lo < hi else lo
    for s in out:
        s["end"] = t0 + s["cum"]
        s["start"] = s["end"] - s["elapsed_ms"] / 1000.0
    return out, (lo, hi)


def decode_window(rec_path, res_path, mtime_ns):
    d, ev = dt.load(rec_path)
    r = json.load(open(res_path))
    t_last = r["rows"][-1][0]
    by = {}
    for e in ev:
        if e["step"] >= 2 and not e["label"].startswith("prefill:") and e["label"] != "barrier" and e["depth"] == 0:
            by.setdefault(e["step"], []).append(e)
    wall = lambda t: (mtime_ns - (t_last - t)) / 1e9
    # one interval per decode step (KV-digest steps and the idle gaps between steps stay out of the window)
    return [(wall(min(e["t0"] for e in v)), wall(max(e["t1"] for e in v))) for st, v in sorted(by.items()) if not any(dt.DIGEST in e["label"] for e in v) and max(e["t1"] for e in v) - min(e["t0"] for e in v) < 30e6]


def summarise(samples, ivs):
    w = []
    for s in samples:
        ov = sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for a, b in ivs)
        if ov > 0:
            w.append((ov, s))
    b, a = sum(y - x for x, y in ivs), 0.0
    if not w:
        return None
    tot = sum(o for o, _ in w)
    wavg = lambda f: sum(o * f(s) for o, s in w) / tot
    pcpu = [c for c in range(6, 10)]
    pf = lambda s: (sum(s["cpus"][c][0] * s["cpus"][c][1] for c in pcpu if c in s["cpus"]) / max(1e-9, sum(s["cpus"][c][1] for c in pcpu if c in s["cpus"])))
    cpu_act = {c: wavg(lambda s, c=c: s["cpus"][c][1]) for c in sorted(samples[0]["cpus"])}
    cpu_mhz = {c: (sum(o * s["cpus"][c][0] * s["cpus"][c][1] for o, s in w) / max(1e-9, sum(o * s["cpus"][c][1] for o, s in w))) for c in cpu_act}
    return dict(cpu_active_pct=cpu_act, cpu_active_mhz=cpu_mhz, window_s=b - a, samples=len(w), p_freq_mhz=wavg(pf), p_active_pct=wavg(lambda s: statistics.mean(s["cpus"][c][1] for c in pcpu if c in s["cpus"])),
                e_active_pct=wavg(lambda s: statistics.mean(v[1] for c, v in s["cpus"].items() if c < 6)), cpu_mw=wavg(lambda s: s["cpu_mw"] or 0), gpu_mw=wavg(lambda s: s["gpu_mw"] or 0),
                gpu_mhz=wavg(lambda s: s["gpu_mhz"] or 0), gpu_res_pct=wavg(lambda s: s["gpu_res"] or 0), thermal=sorted({s["thermal"] for _, s in w}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", required=True)
    ap.add_argument("--run", nargs=4, action="append", metavar=("LABEL", "REC", "RES", "MTIME_NS"), required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    samples, (lo, hi) = parse(a.power)
    print(f"{len(samples)} samples; T0 fit interval {lo:.3f}..{hi:.3f} ({'consistent' if lo < hi else 'INCONSISTENT'}), median elapsed {statistics.median(s['elapsed_ms'] for s in samples):.0f} ms")
    res = {}
    print(f"  {'run':22}{'win s':>7}{'smp':>4}{'Pfreq':>7}{'Pact%':>7}{'Eact%':>7}{'CPU mW':>8}{'GPU MHz':>8}{'GPUres%':>8}{'GPU mW':>8}  thermal")
    for label, rec, rs, mt in a.run:
        o = summarise(samples, decode_window(rec, rs, int(mt)))
        res[label] = o
        if o:
            print(f"  {label:22}{o['window_s']:7.2f}{o['samples']:4d}{o['p_freq_mhz']:7.0f}{o['p_active_pct']:7.1f}{o['e_active_pct']:7.1f}{o['cpu_mw']:8.0f}{o['gpu_mhz']:8.0f}{o['gpu_res_pct']:8.1f}{o['gpu_mw']:8.0f}  {','.join(o['thermal'])}")
            print("      per-CPU active% (E0-5, P6-9): " + " ".join(f"{c}:{v:.0f}@{o['cpu_active_mhz'][c]:.0f}" for c, v in o["cpu_active_pct"].items()))
        else:
            print(f"  {label:22} no overlapping powermetrics sample")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)

"""Phase 62: read-only external process/thread resource sampler (macOS, psutil), aligned with the recorder clock.

    python benchmarks/mac_resource_sampler.py --match <substring of the target argv> --out <file.json> [--interval-ms 2]

Runs as a SEPARATE process (no GIL contention with the pipeline, no change to the target's behaviour). It finds the target process by an argv substring (never
itself), then samples until the target exits: time.perf_counter_ns() (the recorder's clock, same machine), process user/system CPU, voluntary/involuntary
context switches, RSS, thread count and per-thread user/system CPU (stable thread ids; macOS exposes no thread names or per-thread context switches through
psutil). Nothing is written to the target and no privileged tool is used. The dump is a list of rows [t_ns, utime, stime, nvcsw, nivcsw, rss, {tid: [u, s]}]
plus the achieved sampling statistics.
"""
import argparse
import json
import os
import sys
import time

import psutil


def find(match, also, deadline_s):
    me = os.getpid()
    t_end = time.time() + deadline_s
    while time.time() < t_end:
        for p in psutil.process_iter(["pid", "cmdline"]):
            c = p.info["cmdline"] or []
            if p.info["pid"] != me and any("python" in a for a in c[:1]) and any(match in a for a in c) and any(also in a for a in c):
                return psutil.Process(p.info["pid"])
        time.sleep(0.01)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--also", default="", help="a second argv substring that must also be present (e.g. the port)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=float, default=2.0)
    ap.add_argument("--find-timeout-s", type=float, default=600.0)
    a = ap.parse_args()
    p = find(a.match, a.also, a.find_timeout_s)
    if p is None:
        sys.exit("target not found")
    rows, gap = [], []
    iv = a.interval_ms / 1000.0
    prev = time.perf_counter_ns()
    try:
        while True:
            t = time.perf_counter_ns()
            th = {str(x.id): [x.user_time, x.system_time] for x in p.threads()}
            ct = p.cpu_times()
            cs = p.num_ctx_switches()
            rows.append([t, ct.user, ct.system, cs.voluntary, cs.involuntary, p.memory_info().rss, th])
            gap.append((t - prev) / 1e6)
            prev = t
            time.sleep(iv)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass
    gs = sorted(gap[1:]) or [0.0]
    json.dump({"pid": p.pid, "interval_ms": a.interval_ms, "n": len(rows), "gap_ms_median": gs[len(gs) // 2], "gap_ms_p95": gs[int(0.95 * len(gs))], "gap_ms_max": gs[-1],
               "rows": rows}, open(a.out, "w"))


if __name__ == "__main__":
    main()

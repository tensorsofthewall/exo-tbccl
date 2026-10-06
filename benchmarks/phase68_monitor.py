"""Phase 68: external resource monitor for an exo node (read-only): every INTERVAL s it records, for the exo process tree (exo + its runner processes),
RSS, CPU seconds and thread count, plus the host's available memory and, on Linux, the GPU's used memory / utilization / temperature (nvidia-smi).

    python benchmarks/phase68_monitor.py --pid-file <exo.pid> --out <file.jsonl> [--interval 1.0]      # stop with SIGTERM / Ctrl-C
"""
import argparse
import json
import shutil
import signal
import subprocess
import sys
import time

import psutil

stop = False


def _sig(*_):
    global stop
    stop = True


def gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"], timeout=5).decode().split(",")
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], timeout=5).decode().strip().splitlines()
        return {"used_mib": int(o[0]), "total_mib": int(o[1]), "util_pct": int(o[2]), "temp_c": int(o[3]), "sm_mhz": int(o[4]), "power_w": float(o[5]), "apps": [a.strip() for a in apps]}
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    root = int(open(a.pid_file).read().strip())
    with open(a.out, "w") as f:
        while not stop:
            t = time.time()
            rec = {"t": t}
            try:
                p = psutil.Process(root)
                procs = [p] + p.children(recursive=True)
                rec["procs"] = []
                for q in procs:
                    try:
                        mi = q.memory_info()
                        ct = q.cpu_times()
                        rec["procs"].append({"pid": q.pid, "rss": mi.rss, "cpu_s": ct.user + ct.system, "threads": q.num_threads(), "cmd": " ".join(q.cmdline())[:80]})
                    except psutil.Error:
                        pass
            except psutil.Error:
                rec["procs"] = []
            vm = psutil.virtual_memory()
            rec["avail"] = vm.available
            rec["used"] = vm.used
            rec["swap_used"] = psutil.swap_memory().used
            g = gpu()
            if g:
                rec["gpu"] = g
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()

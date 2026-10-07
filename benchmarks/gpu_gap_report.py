"""Classify every GPU eval of a recorded decode as warm or cold by the idle gap before it, per label.

    python benchmarks/gpu_gap_report.py dump.json [dump2.json ...] [--skip 3]

Source-level only (no powermetrics): from a sync_recorder dump, take the evals that run GPU work (stage compute, sampler, the bridge's destination allocation and
view evals, cache dependency) and, for each, the idle time since the previous such eval ended on this process. Gap buckets: warm < 300 us, mid 0.3-1 ms, cold > 1 ms.
Prints per label and bucket the number of calls per step and the median duration, plus the per-step count and total duration of the TINY evals.
"""
import argparse
import json
import statistics

GPU_LABELS = {"model_output_eval": "stage", "real_model_loopback.py:worker#4": "sampler", "destination_alloc_eval": "tiny", "borrow_view_eval": "tiny",
              "borrow_array_eval": "status", "cache_dependency_eval": "small", "allgather_destination_eval": "status", "recv_destination_eval": "status",
              "pre_recv_template_eval": "small"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--skip", type=int, default=3)
    a = ap.parse_args()
    for path in a.files:
        d = json.load(open(path))
        ev = sorted((e for e in d["events"] if e[2] == "eval" and not e[3].startswith("prefill:") and e[3] in GPU_LABELS and e[5] >= a.skip), key=lambda e: e[0])
        # only evals that can wait (>3 us): status checks on evaluated arrays are not GPU work
        ev = [e for e in ev if (e[1] - e[0]) / 1000 > 3]
        rows = {}
        last_end = None
        per_step_tiny = {}
        for t0, t1, kind, label, depth, step, op, tid in ev:
            gap = (t0 - last_end) / 1000 if last_end else 0.0
            last_end = t1
            b = "warm" if gap < 300 else "mid" if gap < 1000 else "cold"
            rows.setdefault((label, b), []).append((t1 - t0) / 1000)
            if GPU_LABELS[label] == "tiny":
                per_step_tiny.setdefault(step, []).append((t1 - t0) / 1000)
        print(f"{path}: rank {d['rank']} {d['backend']}; per decode step: tiny GPU evals {statistics.median(len(v) for v in per_step_tiny.values()) if per_step_tiny else 0:.0f}, "
              f"their total {statistics.median(sum(v) for v in per_step_tiny.values()) if per_step_tiny else 0:.0f} us")
        print(f"  {'label':34}{'gap':>6}{'n':>6}{'median us':>11}")
        for (label, b), v in sorted(rows.items()):
            print(f"  {label:34}{b:>6}{len(v):6d}{statistics.median(v):11.0f}")


if __name__ == "__main__":
    main()

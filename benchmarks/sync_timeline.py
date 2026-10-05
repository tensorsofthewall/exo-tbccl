"""Per-token timeline and synchronization counts from sync_recorder dumps.

    python benchmarks/sync_timeline.py rank0.json rank1.json [...] [--token N] [--md]

A decode token is the span between two `step_complete` calls (exo calls it once per forward step, right after the stage output is evaluated), so a
token starts at one step_complete and ends at the next. For each rank prints, over the steady-state tokens (the first 3 dropped):
  * counts per token: eval calls, "blocking" evals (> BLOCK_US), DLPack exports, native waits;
  * per label: calls/token and median inclusive microseconds, with the time spent inside nested events removed for `exclusive`;
  * one representative token (the median-length one) as an ordered list of top-level events.
"""
import json
import statistics
import sys

BLOCK_US = 20.0


def load(path):
    d = json.load(open(path))
    ev = [dict(t0=a, t1=b, kind=k, label=l, depth=dp) for a, b, k, l, dp in d["events"]]
    return d["rank"], d["backend"], ev


def tokens(ev):
    marks = [i for i, e in enumerate(ev) if e["kind"] == "comm" and e["label"] == "step_complete" and not e["label"].startswith("prefill:")]
    # step_complete events recorded in decode have their label unchanged; prefill ones are prefixed
    spans = []
    for a, b in zip(marks, marks[1:]):
        spans.append(ev[a: b + 1])
    return spans


def exclusive(span):
    """exclusive time of each event: its duration minus the durations of events nested directly inside it (depth+1, contained)."""
    out = []
    for i, e in enumerate(span):
        inner = sum(c["t1"] - c["t0"] for c in span if c["depth"] == e["depth"] + 1 and c["t0"] >= e["t0"] and c["t1"] <= e["t1"])
        out.append(((e["t1"] - e["t0"]) - inner) / 1000.0)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for path in args:
        rank, backend, ev = load(path)
        dec = [e for e in ev if not e["label"].startswith("prefill:")]
        spans = tokens(dec)[3:]
        if not spans:
            print(f"{path}: no decode tokens"); continue
        n = len(spans)
        per_label = {}
        counts = {"eval": [], "blocking_eval": [], "dlpack": [], "wait": [], "async_eval": []}
        for sp in spans:
            c = {k: 0 for k in counts}
            seen = {}
            ex = exclusive(sp)
            for e, x in zip(sp, ex):
                dur = (e["t1"] - e["t0"]) / 1000.0
                if e["kind"] == "eval":
                    c["eval"] += 1
                    if dur > BLOCK_US: c["blocking_eval"] += 1
                elif e["kind"] == "async_eval": c["async_eval"] += 1
                elif e["kind"] == "dlpack_export": c["dlpack"] += 1
                elif e["kind"] == "tbccl_wait": c["wait"] += 1
                key = (e["kind"], e["label"])
                s = seen.setdefault(key, [0, 0.0, 0.0]); s[0] += 1; s[1] += dur; s[2] += x
            for k in counts: counts[k].append(c[k])
            for key, s in seen.items():
                per_label.setdefault(key, []).append(s)
        print(f"\n=== {path}: rank {rank} backend {backend}, {n} steady-state tokens")
        print("  per token: " + ", ".join(f"{k}={statistics.median(v):.0f}" for k, v in counts.items()))
        print(f"  {'kind':13}{'label':34}{'calls/tok':>10}{'incl us':>10}{'excl us':>10}")
        for key in sorted(per_label, key=lambda k: -statistics.median(s[2] for s in per_label[k])):
            v = per_label[key]
            print(f"  {key[0]:13}{key[1]:34}{statistics.median(s[0] for s in v):10.1f}{statistics.median(s[1] for s in v):10.1f}{statistics.median(s[2] for s in v):10.1f}")
        mid = sorted(spans, key=lambda s: s[-1]["t1"] - s[0]["t0"])[len(spans) // 2]
        t0 = mid[0]["t0"]
        print(f"  representative token ({(mid[-1]['t1'] - t0) / 1000:.0f} us), top-level events:")
        for e in mid:
            if e["depth"] == 0:
                print(f"    +{(e['t0'] - t0) / 1000:8.1f}  {(e['t1'] - e['t0']) / 1000:8.1f} us  {e['kind']:13}{e['label']}")


if __name__ == "__main__":
    main()

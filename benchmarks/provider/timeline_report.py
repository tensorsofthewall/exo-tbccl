"""Decode the per-step timeline written by ``trace_hook/step_timeline.py`` into critical-path intervals (standard library only).

    python timeline_report.py <trace-dir> [<trace-dir> ...] [--json out.json] [--markdown out.md] [--require-complete]

Every interval is a *measured* difference of two events on the same process's monotonic clock; nothing is derived across hosts and nothing is estimated. Intervals of a decode step
(microseconds; a stage without a receive has no recv intervals, the last stage no send intervals)::

    post         previous LL_EXIT      -> FL_ENTER       sampling, token handling and the Python loop between two forward passes
    pre_recv     FL_ENTER              -> RECV_ENTER     evaluation of the receive template
    recv_call    RECV_ENTER            -> RECV_EXIT      MlxTbccl: allocation + borrow + submit + the blocking wait; MlxRing: building a lazy node
    recv_eval    RECV_EXIT             -> FL_EXIT        MlxRing: the mx.eval that performs the receive (the wait for the upstream stage); MlxTbccl: a status check
    build        FL_EXIT               -> LL_ENTER       Python graph construction of the stage's layers (the GPU is idle)
    compute      LL_ENTER              -> STEP_COMPLETE  last layer construction plus the mx.eval of the whole stage graph (GPU execution and synchronization)
    send_call    SEND_ENTER            -> SEND_EXIT
    send_eval    SEND_EXIT (or STEP_COMPLETE) -> GATHER_ENTER   MlxRing: the mx.eval that performs the send; MlxTbccl: a status check
    gather_call  GATHER_ENTER          -> GATHER_EXIT
    gather_eval  GATHER_EXIT           -> LL_EXIT        MlxRing: the mx.eval that performs the all_gather (the wait for the peer); MlxTbccl: a status check
    period       previous LL_EXIT      -> LL_EXIT        token to token on this rank (decode only)

MlxTbccl only, summed over the step and *contained in* the call intervals above: ``alloc`` (destination allocation and mx.eval), ``borrow`` (view evaluation + DLPack export),
``submit`` (the native call that queues the Work), ``wait`` (the blocking wait including any bounded spin), ``drain`` and ``reap``. ``wait`` is the only interval in which the thread
can be blocked on the peer; it contains the peer's compute, so it is not a cost of the transport by itself. ``other`` is the residual of the measured partition, not an estimate.
Communicator calls made before FL_ENTER (for example exo's per-token task agreement) are reported as ``x_gather`` and ``x_alloc``/``x_wait``/``x_borrow``/``x_submit``; they are contained in ``post``.

Validity. A process is ``complete`` only if its recorder closed cleanly (``final``), wrote without errors and did not hit its event cap. A killed, crashed, cancelled or still-running process
is ``partial`` and says why; its statistics are labelled partial and ``--require-complete`` exits non-zero. Forward passes that are malformed (missing, unbalanced or out-of-order events) or
unfinished are counted and excluded from every statistic, never repaired. Ranks and hosts are kept separate: no interval is ever computed between two processes.
"""
from __future__ import annotations

import argparse
import array
import glob
import json
import os
import statistics
import sys

SCHEMA = "tbccl-decode-timeline-report/1"
SHIFT = 6
MASK = (1 << SHIFT) - 1
INTERVALS = ("post", "pre_recv", "recv_call", "recv_eval", "build", "compute", "send_call", "send_eval", "gather_call", "gather_eval")
INTERNAL = ("other", "alloc", "borrow", "submit", "wait", "drain", "reap", "x_gather", "x_alloc", "x_wait", "x_borrow", "x_submit", "x_gathers")
PAIRS = {"alloc": ("ALLOC_ENTER", "ALLOC_EXIT"), "borrow": ("BORROW_ENTER", "BORROW_EXIT"), "submit": ("SUBMIT_ENTER", "SUBMIT_EXIT"), "wait": ("WAIT_ENTER", "WAIT_EXIT"),
         "drain": ("DRAIN_ENTER", "DRAIN_EXIT"), "reap": ("REAP_ENTER", "REAP_EXIT")}
BALANCED = {"RECV": ("RECV_ENTER", "RECV_EXIT"), "SEND": ("SEND_ENTER", "SEND_EXIT"), "GATHER": ("GATHER_ENTER", "GATHER_EXIT"), **{k.upper(): v for k, v in PAIRS.items()}}
REQUIRED = ("FL_ENTER", "FL_EXIT", "LL_ENTER", "STEP_COMPLETE")
COUNTERS = ("CPU_NS", "NVCSW", "NIVCSW")
COUNT_KEYS = ("cpu_us", "d_nvcsw", "d_nivcsw", "x_gathers")


def load(directory: str):
    """Yield ``(meta, events, problems)`` for every process in a trace directory; ``events`` is a list of ``(t_ns_or_value, name)`` and ``problems`` lists what could not be read."""
    for meta_path in sorted(glob.glob(os.path.join(directory, "timeline-*.json"))):
        problems = []
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            names = {v: k for k, v in meta["events"].items()}
        except (OSError, ValueError, KeyError) as e:
            yield {"pid": None, "path": meta_path}, [], [f"unreadable metadata: {e}"]
            continue
        bin_path = meta_path[:-5] + ".bin"
        if not os.path.exists(bin_path):
            continue  # a process of the node that ran no pipeline layers (the hook is loaded in every Python process of the node)
        with open(bin_path, "rb") as f:
            data = f.read()
        if len(data) % 8:
            problems.append(f"{len(data) % 8} trailing bytes ignored (a partial last write)")
        raw = array.array("q")
        raw.frombytes(data[: len(data) // 8 * 8])
        events, unknown = [], 0
        for v in raw:
            name = names.get(v & MASK)
            if name is None:
                unknown += 1
                continue
            events.append((v >> SHIFT, name))
        if unknown:
            problems.append(f"{unknown} events with an unknown code ignored")
        yield meta, events, problems


def steps_of(events):
    """Split an event list into forward passes ending at LL_EXIT / LL_EXIT_PREFILL and attach the counters recorded right after. Returns ``(steps, tail)``; ``tail`` are the events after the last pass."""
    steps, cur = [], []
    for value, name in events:
        if name in COUNTERS:
            if steps:
                steps[-1][{"CPU_NS": "cpu_ns", "NVCSW": "nvcsw", "NIVCSW": "nivcsw"}[name]] = value
            continue
        cur.append((value, name))
        if name in ("LL_EXIT", "LL_EXIT_PREFILL"):
            steps.append({"events": cur, "prefill": name == "LL_EXIT_PREFILL"})
            cur = []
    return steps, cur


def step_problem(ev):
    """Why a forward pass cannot be used, or None. Checks required events, enter/exit balance per pair and monotonic time."""
    names = [n for _, n in ev]
    for need in REQUIRED:
        if need not in names:
            return f"missing {need}"
    if names.count("FL_ENTER") > 1 or names.count("LL_ENTER") > 1:
        return "repeated layer events"
    last = None
    for v, _ in ev:
        if last is not None and v < last:
            return "timestamps go backwards"
        last = v
    for label, (a, b) in BALANCED.items():
        depth = 0
        for n in names:
            depth += (n == a) - (n == b)
            if depth < 0:
                return f"{b} without {a}"
        if depth:
            return f"{a} without {b}"
    if names.index("FL_EXIT") < names.index("FL_ENTER") or names.index("STEP_COMPLETE") < names.index("LL_ENTER"):
        return "events out of order"
    return None


def decompose(step, prev_exit_ns):
    """Intervals of one forward pass. Communicator calls made *before* the pass starts (FL_ENTER) belong to the loop around it, not to the pipeline layers (see the module docstring)."""
    ev = step["events"]
    start = max((i for i, (_, n) in enumerate(ev) if n == "FL_ENTER"), default=0)
    pre, ev = ev[:start], ev[start:]
    extra = {}
    if pre:
        for kind, (a, b) in (("x_gather", ("GATHER_ENTER", "GATHER_EXIT")), ("x_alloc", PAIRS["alloc"]), ("x_wait", PAIRS["wait"]), ("x_borrow", PAIRS["borrow"]), ("x_submit", PAIRS["submit"])):
            total, seen, begin = 0, False, None
            for v, name in pre:
                if name == a and begin is None:
                    begin = v
                elif name == b and begin is not None:
                    total += v - begin
                    seen, begin = True, None
            if seen:
                extra[kind] = total
        extra["x_gathers"] = sum(1 for _, n in pre if n == "GATHER_ENTER")
    t = {}
    for v, name in ev:
        t.setdefault(name, v)
    out = {}
    fl, ll_exit = t["FL_ENTER"], ev[-1][0]
    if prev_exit_ns is not None:
        out["post"] = fl - prev_exit_ns
    if "RECV_ENTER" in t:
        out["pre_recv"] = t["RECV_ENTER"] - fl
        out["recv_call"] = t["RECV_EXIT"] - t["RECV_ENTER"]
        out["recv_eval"] = t["FL_EXIT"] - t["RECV_EXIT"]
    out["build"] = t["LL_ENTER"] - t["FL_EXIT"]
    out["compute"] = t["STEP_COMPLETE"] - t["LL_ENTER"]
    if "SEND_ENTER" in t:
        out["send_call"] = t["SEND_EXIT"] - t["SEND_ENTER"]
    if "GATHER_ENTER" in t:
        out["send_eval"] = t["GATHER_ENTER"] - t.get("SEND_EXIT", t["STEP_COMPLETE"])
        out["gather_call"] = t["GATHER_EXIT"] - t["GATHER_ENTER"]
        out["gather_eval"] = ll_exit - t["GATHER_EXIT"]
    if prev_exit_ns is not None:
        out["period"] = ll_exit - prev_exit_ns
    for kind, (a, b) in PAIRS.items():
        depth_start, total, seen = [], 0, False
        for v, name in ev:
            if name == a:
                depth_start.append(v)
            elif name == b and depth_start:
                s = depth_start.pop()
                if not depth_start:
                    total += v - s
                    seen = True
        if seen:
            out[kind] = total
    out.update(extra)
    if "period" in out:
        out["other"] = out["period"] - sum(out.get(k, 0) for k in INTERVALS)
    return out


def process_summary(meta, events):
    """Rows (one per usable decode step) and the accounting of everything that was excluded."""
    steps, tail = steps_of(events)
    rows, prev_exit, request, k, last_was_decode = [], None, -1, 0, True
    acct = {"forward_passes": len(steps), "prefill_passes": 0, "decode_steps_used": 0, "malformed_passes": 0, "malformed_reasons": {}, "unfinished_passes": 0}
    for step in steps:
        problem = step_problem(step["events"])
        if problem:
            acct["malformed_passes"] += 1
            acct["malformed_reasons"][problem] = acct["malformed_reasons"].get(problem, 0) + 1
            prev_exit, last_was_decode = None, True  # a broken pass breaks the chain: the next period would span it
            continue
        if step["prefill"]:
            acct["prefill_passes"] += 1
            if last_was_decode:
                request += 1
                k = 0
            last_was_decode = False
            prev_exit = step["events"][-1][0]
            continue
        d = decompose(step, prev_exit)
        d.update(request=request, k=k, cpu_ns=step.get("cpu_ns"), nvcsw=step.get("nvcsw"), nivcsw=step.get("nivcsw"))
        rows.append(d)
        prev_exit = step["events"][-1][0]
        last_was_decode = True
        k += 1
    if any(n == "FL_ENTER" for _, n in tail):
        acct["unfinished_passes"] = 1  # a pass started and never reached LL_EXIT (cancelled, failed or killed mid-step)
    acct["decode_steps_used"] = len(rows)
    for a, b in zip(rows, rows[1:]):
        if a["request"] == b["request"] and a.get("cpu_ns") is not None and b.get("cpu_ns") is not None:
            b["cpu_us"] = (b["cpu_ns"] - a["cpu_ns"]) / 1e3
            b["d_nvcsw"] = b["nvcsw"] - a["nvcsw"]
            b["d_nivcsw"] = b["nivcsw"] - a["nivcsw"]
    return rows, acct


def summarize(rows, length=None):
    """Median of each interval over the steady decode steps (k >= 1) of the requests with the most common decode length of at least 20 steps (the o32 latency rounds, which dominate a
    lifecycle); ``length`` overrides that choice. Requests of any other length (cancelled streams, warmups, other workloads) are not part of the statistics."""
    by_request = {}
    for r in rows:
        by_request.setdefault(r["request"], []).append(r)
    lengths = {}
    for rs in by_request.values():
        lengths[len(rs)] = lengths.get(len(rs), 0) + 1
    if length is None:
        counts = {n: c for n, c in lengths.items() if n >= 20}
        length = min(n for n, c in counts.items() if c == max(counts.values())) if counts else 0
    chosen = [r for rs in by_request.values() if len(rs) == length for r in rs if r["k"] >= 1]
    res = {"decode_steps": len(chosen), "requests": sum(1 for rs in by_request.values() if len(rs) == length), "decode_length": length,
           "request_lengths": {str(n): c for n, c in sorted(lengths.items())}}
    for key in INTERVALS + ("period",) + INTERNAL + ("cpu_us", "d_nvcsw", "d_nivcsw"):
        vals = [r[key] for r in chosen if r.get(key) is not None]
        if vals:
            scale = 1.0 if key in COUNT_KEYS else 1e-3
            res[key] = {"median": statistics.median(vals) * scale, "mean": statistics.fmean(vals) * scale, "n": len(vals)}
    return res


def process_status(meta, problems, acct):
    reasons = list(problems)
    if not meta.get("final"):
        reasons.append("the recorder was not closed (process killed, crashed or still running)")
    if meta.get("truncated"):
        reasons.append(f"event cap of {meta.get('max_events')} reached; later events were dropped")
    if meta.get("write_errors"):
        reasons.append(f"{meta['write_errors']} failed writes")
    if acct["unfinished_passes"]:
        reasons.append("a forward pass was unfinished at the end of the trace")
    return ("partial" if reasons else "complete"), reasons


def build_report(dirs):
    processes = []
    for d in dirs:
        for meta, events, problems in load(d):
            if meta.get("pid") is None:
                processes.append({"dir": d, "status": "partial", "reasons": problems, "pid": None})
                continue
            if not events:
                continue
            rows, acct = process_summary(meta, events)
            status, reasons = process_status(meta, problems, acct)
            entry = {"dir": d, "host": meta.get("host"), "pid": meta["pid"], "backend": meta.get("backend"), "rank": meta.get("rank"), "world_size": meta.get("world_size"),
                     "status": status, "reasons": reasons, "accounting": acct, "statistics": summarize(rows),
                     "provenance": {"schema": meta.get("schema"), "tracing": meta.get("tracing"), "software": meta.get("software"), "python": meta.get("python"), "platform": meta.get("platform"),
                                    "events_written": meta.get("events_written"), "comms": meta.get("comms")}}
            processes.append(entry)
    processes.sort(key=lambda p: (str(p["dir"]), str(p.get("host")), p["rank"] if isinstance(p.get("rank"), int) else -1, p["pid"] or 0))
    return {"schema": SCHEMA, "processes": processes}


def render_text(report):
    lines = []
    for p in report["processes"]:
        if p["pid"] is None:
            lines.append(f"{p['dir']}: UNREADABLE ({'; '.join(p['reasons'])})")
            continue
        s = p["statistics"]
        lines.append(f"{p['host']} pid {p['pid']} rank {p['rank']} backend {p['backend']} [{p['status'].upper()}]: {s['requests']} requests of {s['decode_length']} steps, {s['decode_steps']} steady decode steps")
        for r in p["reasons"]:
            lines.append(f"  ! {r}")
        for k, v in p["accounting"]["malformed_reasons"].items():
            lines.append(f"  ! {v} malformed pass(es) excluded: {k}")
        for key in ("period",) + INTERVALS + INTERNAL + ("cpu_us", "d_nvcsw", "d_nivcsw"):
            if key in s:
                unit = "" if key in ("d_nvcsw", "d_nivcsw", "x_gathers") else " us"
                lines.append(f"  {key:12s} median {s[key]['median']:10.1f}{unit}")
    return "\n".join(lines) + "\n"


def render_markdown(report):
    out = ["# Decode timeline report", "", f"Schema `{report['schema']}`. Intervals are medians in microseconds over steady decode steps of one process; no value spans two processes.", ""]
    cols = ("period",) + INTERVALS + ("other",)
    out += ["| host | rank | backend | status | steps | " + " | ".join(cols) + " |", "|---|---|---|---|---|" + "---|" * len(cols)]
    for p in report["processes"]:
        if p["pid"] is None:
            out.append("| unreadable | | | partial | | " + " | ".join("" for _ in cols) + " |")
            continue
        s = p["statistics"]
        out.append(f"| {p['host']} | {p['rank']} | {p['backend']} | {p['status']} | {s['decode_steps']} | " + " | ".join(f"{s[c]['median']:.1f}" if c in s else "-" for c in cols) + " |")
    for p in report["processes"]:
        if p["reasons"]:
            out += ["", f"Process {p.get('host')} pid {p['pid']}: " + "; ".join(p["reasons"])]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", help="write the report as JSON (keys sorted, deterministic)")
    ap.add_argument("--markdown", help="write the report as a Markdown table")
    ap.add_argument("--require-complete", action="store_true", help="exit with status 2 unless at least one process was read and every process is complete")
    args = ap.parse_args(argv)
    report = build_report(args.dirs)
    sys.stdout.write(render_text(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
            f.write("\n")
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(render_markdown(report))
    if args.require_complete and (not report["processes"] or any(p["status"] != "complete" for p in report["processes"])):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

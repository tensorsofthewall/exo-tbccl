"""The optional decode timeline (trace_hook/step_timeline.py + timeline_report.py), offline: stand-in exo and exo_tbccl packages, no GPU, no second machine, no model.

Covers opt-in activation, default-off, event order and exact durations, missing/malformed/incomplete events, rank attribution, deterministic output, several lifecycles, cancelled and
killed processes, the event cap, and a record-then-report round trip through the real hook."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import common  # noqa: F401  (puts the provider directory on sys.path)

import timeline_report

HOOK = str(Path(__file__).resolve().parents[1] / "trace_hook")
sys.path.insert(0, HOOK)
import step_timeline  # noqa: E402

AUTO_PARALLEL = '''
class PipelineFirstLayer:
    def __init__(self, comm, r): self.comm, self.r = comm, r
    def __call__(self, x):
        if self.r != 0:
            x = self.comm.recv_like(x, self.r - 1)
        return x
class PipelineLastLayer:
    def __init__(self, comm, r, s): self.comm, self.r, self.s, self.is_prefill = comm, r, s, False
    def __call__(self, x):
        self.comm.step_complete()
        if self.r != self.s - 1:
            x = self.comm.send(x, self.r + 1)
        if not self.is_prefill:
            x = self.comm.all_gather(x)
        return x
'''
PIPELINE_COMM = '''
class MlxPipelineComm:
    def recv_like(self, t, src): return t
    def send(self, a, dst): return a
    def all_gather(self, a): return a
    def step_complete(self): pass
'''
GROUP = '''
import time
class _Stats:
    materialized_copies = 0
    staged_fallback_copies = 0
    direct_bytes = 0
class _Cfg: wait_spin_ms = 0.0
def borrow(a, *x, **k): return a
class TbcclPipelineComm:
    def __init__(self, rank=1):
        self._rank, self.stats, self.config = rank, _Stats(), _Cfg()
    def activity_stats(self): return {"activity_windows": 0}
    def _alloc(self, *a): return 0
    def _submit(self, *a): return 0
    def wait(self, t, timeout_ms=None): time.sleep(0.002)
    def _drain_async_sends(self): pass
    def _reap(self): pass
    def recv_like(self, t, src):
        self._alloc(); borrow(t); self._submit(); self.wait(0); return t
    def send(self, a, dst):
        borrow(a); self._submit(); self.wait(0); return a
    def all_gather(self, a):
        self._alloc(); borrow(a); borrow(a); self._submit(); self.wait(0); self._drain_async_sends(); return a
    def step_complete(self): pass
'''
SCRIPT = '''
import exo.worker.engines.mlx.auto_parallel as ap
import exo.worker.engines.mlx.pipeline_comm as pc
import time, os
%s
first, last = ap.PipelineFirstLayer(comm, 1), ap.PipelineLastLayer(comm, 1, 3)
last.is_prefill = True
first(1); last(1)
last.is_prefill = False
for _ in range(%d):
    first(1); time.sleep(float(os.environ.get('STEP_SLEEP', '0.001'))); last(1)
%s
print("done")
'''
TBCCL_BODY = "import exo_tbccl.group as g\ncomm = g.TbcclPipelineComm(1)"
RING_BODY = "comm = pc.MlxPipelineComm()"


def write_trace(directory, events, pid=100, final=True, host="hostA", rank=0, world=2, backend="tbccl", **meta_extra):
    """A trace written the way the recorder writes it, from [(t_ns, name)] and optional extra raw words."""
    os.makedirs(directory, exist_ok=True)
    words = []
    for t, name in events:
        words.append(t << step_timeline.SHIFT | step_timeline.EVENTS[name] if isinstance(name, str) else t << step_timeline.SHIFT | name)
    with open(os.path.join(directory, f"timeline-{pid}.bin"), "wb") as f:
        import array

        array.array("q", words).tofile(f)
    meta = {"schema": step_timeline.SCHEMA, "pid": pid, "host": host, "platform": "test", "python": "3", "backend": backend, "rank": rank, "world_size": world, "events": step_timeline.EVENTS,
            "shift": step_timeline.SHIFT, "events_written": len(words), "max_events": 8_000_000, "truncated": False, "write_errors": 0, "final": final,
            "tracing": {"TBCCL_TRACE_TIMELINE": "1"}, "software": {"exo_tbccl": "0.0"}, "comms": []}
    meta.update(meta_extra)
    with open(os.path.join(directory, f"timeline-{pid}.json"), "w") as f:
        json.dump(meta, f)


def step(t0, gap_before=1000, compute=5000, with_recv=False, prefill=False, cpu=None):
    """One forward pass starting at t0 (ns): returns (events, end_time). Durations are fixed so tests can assert exact intervals."""
    t = t0 + gap_before
    ev = [(t, "FL_ENTER")]
    if with_recv:
        ev += [(t + 100, "RECV_ENTER"), (t + 400, "RECV_EXIT")]
    ev += [(t + 500, "FL_EXIT"), (t + 700, "LL_ENTER"), (t + 700 + compute, "STEP_COMPLETE")]
    t = t + 700 + compute
    if not prefill:
        ev += [(t + 10, "GATHER_ENTER"), (t + 310, "GATHER_EXIT")]
        t += 310
    ev += [(t + 20, "LL_EXIT_PREFILL" if prefill else "LL_EXIT")]
    t += 20
    if cpu is not None:
        ev += [(cpu[0], "CPU_NS"), (cpu[1], "NVCSW"), (cpu[2], "NIVCSW")]
    return ev, t


def drop_nth(events, name, n):
    seen = 0
    out = []
    for e in events:
        if e[1] == name:
            seen += 1
            if seen == n:
                continue
        out.append(e)
    return out


def request(t0, n_decode, cpu0=0):
    """One prefill pass then n_decode decode steps, as the recorder sees a request."""
    ev, t = step(t0, prefill=True)
    for i in range(n_decode):
        e, t = step(t, with_recv=True, cpu=(cpu0 + 1000 * (i + 1), 10 * (i + 1), i))
        ev += e
    return ev, t + 50_000


class Hook(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        for rel, text in {"exo/__init__.py": "", "exo/worker/__init__.py": "", "exo/worker/engines/__init__.py": "", "exo/worker/engines/mlx/__init__.py": "",
                          "exo/worker/engines/mlx/auto_parallel.py": AUTO_PARALLEL, "exo/worker/engines/mlx/pipeline_comm.py": PIPELINE_COMM,
                          "exo_tbccl/__init__.py": "", "exo_tbccl/group.py": GROUP}.items():
            path = self.tmp / "pkgs" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def run_script(self, body, timeline=True, steps=4, tail="", extra_env=None, trace_dir=True):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_") and k != "PYTHONPATH"}
        env["PYTHONPATH"] = os.pathsep.join([HOOK, str(self.tmp / "pkgs")])
        if trace_dir:
            env["TBCCL_B6_TRACE_DIR"] = str(self.tmp / "trace")
        if timeline:
            env["TBCCL_TRACE_TIMELINE"] = "1"
        env.update(extra_env or {})
        return subprocess.run([sys.executable, "-c", SCRIPT % (body, steps, tail)], env=env, cwd=str(self.tmp), capture_output=True, text=True, timeout=60)

    def test_nothing_is_recorded_unless_both_variables_are_set(self):
        for kwargs in ({"timeline": False}, {"trace_dir": False}):
            out = self.run_script(TBCCL_BODY, **kwargs)
            self.assertEqual((out.returncode, out.stdout.strip()), (0, "done"), out.stderr)
            self.assertEqual(list((self.tmp / "trace").glob("timeline-*")) if (self.tmp / "trace").exists() else [], [])

    def test_a_recorded_tbccl_process_is_final_and_carries_its_provenance(self):
        out = self.run_script(TBCCL_BODY)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "done"), out.stderr)
        (meta, events, problems), = list(timeline_report.load(str(self.tmp / "trace")))
        self.assertEqual((meta["backend"], meta["final"], meta["truncated"], meta["write_errors"], problems), ("tbccl", True, False, 0, []))
        self.assertEqual((meta["rank"], meta["world_size"], meta["schema"]), (1, 3, step_timeline.SCHEMA))
        self.assertEqual(meta["tracing"]["TBCCL_TRACE_TIMELINE"], "1")
        self.assertIn("exo_tbccl", meta["software"])
        rows, acct = timeline_report.process_summary(meta, events)
        self.assertEqual((len(rows), acct["malformed_passes"], acct["unfinished_passes"]), (4, 0, 0))
        row = rows[2]
        for key in ("post", "pre_recv", "recv_call", "recv_eval", "build", "compute", "send_call", "send_eval", "gather_call", "gather_eval", "period", "alloc", "borrow", "submit", "wait", "drain"):
            self.assertIn(key, row, key)
        self.assertGreaterEqual(row["wait"], 3 * 2_000_000)  # recv + send + gather each sleep 2 ms in wait()
        parts = sum(row[k] for k in ("pre_recv", "recv_call", "recv_eval", "build", "compute", "send_call", "send_eval", "gather_call", "gather_eval"))
        self.assertEqual(row["other"], row["period"] - row["post"] - parts)
        self.assertIn("cpu_us", rows[2])

    def test_ring_uses_the_same_events_without_tbccl_internals(self):
        out = self.run_script(RING_BODY)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "done"), out.stderr)
        (meta, events, _), = list(timeline_report.load(str(self.tmp / "trace")))
        self.assertEqual(meta["backend"], "ring")
        rows, _ = timeline_report.process_summary(meta, events)
        self.assertIn("gather_call", rows[2])
        self.assertNotIn("wait", rows[2])

    def test_the_event_cap_bounds_the_output_and_marks_the_process_partial(self):
        out = self.run_script(TBCCL_BODY, steps=200, extra_env={"TBCCL_TRACE_TIMELINE_MAX_EVENTS": "300", "STEP_SLEEP": "0.01"}, tail="time.sleep(0.6)")
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "done"), out.stderr)
        report = timeline_report.build_report([str(self.tmp / "trace")])
        (proc,) = report["processes"]
        size = (self.tmp / "trace").glob("timeline-*.bin")
        self.assertLess(next(size).stat().st_size, 8 * 4000)  # an uncapped trace of these 200 steps holds about 7,000 events; the cap is enforced at the next flush (0.5 s)
        self.assertEqual(proc["status"], "partial")
        self.assertTrue(any("event cap" in r for r in proc["reasons"]), proc["reasons"])

    def test_report_round_trip_through_the_cli(self):
        self.assertEqual(self.run_script(TBCCL_BODY).returncode, 0)
        j, m = self.tmp / "r.json", self.tmp / "r.md"
        for _ in range(2):
            out = subprocess.run([sys.executable, str(Path(HOOK).parent / "timeline_report.py"), str(self.tmp / "trace"), "--json", str(j), "--markdown", str(m), "--require-complete"],
                                 capture_output=True, text=True, cwd=str(self.tmp))
            self.assertEqual(out.returncode, 0, out.stderr + out.stdout)
        data = json.loads(j.read_text())
        self.assertEqual(data["schema"], timeline_report.SCHEMA)
        self.assertEqual(data["processes"][0]["status"], "complete")
        self.assertIn("| host | rank |", m.read_text())


class Report(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def trace(self, name="a", **kw):
        d = self.tmp / name
        events, t = [], 10_000
        for n in kw.pop("lengths", (24, 24, 24)):
            e, t = request(t, n)
            events += e
        events += kw.pop("extra", [])
        write_trace(str(d), events, **kw)
        return str(d)

    def stats(self, d, **kw):
        (p,) = timeline_report.build_report([d])["processes"]
        return p

    def test_durations_are_exact_and_ordered(self):
        p = self.stats(self.trace())
        s = p["statistics"]
        self.assertEqual(p["status"], "complete")
        # recv call 300 ns, FL_EXIT-RECV_EXIT 100 ns, build 200 ns, compute 5000 ns, gather_call 300 ns, gather_eval 20 ns -> medians in microseconds
        self.assertAlmostEqual(s["recv_call"]["median"], 0.3)
        self.assertAlmostEqual(s["recv_eval"]["median"], 0.1)
        self.assertAlmostEqual(s["build"]["median"], 0.2)
        self.assertAlmostEqual(s["compute"]["median"], 5.0)
        self.assertAlmostEqual(s["gather_call"]["median"], 0.3)
        self.assertAlmostEqual(s["gather_eval"]["median"], 0.02)
        self.assertAlmostEqual(s["post"]["median"], 1.0)
        self.assertEqual(s["decode_length"], 24)
        self.assertEqual(s["requests"], 3)
        self.assertEqual(s["decode_steps"], 3 * 23)  # the first decode step of each request is not steady state
        self.assertAlmostEqual(s["other"]["median"], 0.0)

    def test_missing_events_make_a_pass_malformed_and_it_is_excluded_not_repaired(self):
        ev, t = request(10_000, 24)
        broken = drop_nth(ev, "STEP_COMPLETE", 10)  # drop one STEP_COMPLETE in the middle
        write_trace(str(self.tmp / "m"), broken)
        p = self.stats(str(self.tmp / "m"))
        self.assertEqual(p["accounting"]["malformed_passes"], 1)
        self.assertIn("missing STEP_COMPLETE", p["accounting"]["malformed_reasons"])
        self.assertEqual(p["accounting"]["decode_steps_used"], 23)

    def test_unbalanced_pairs_unknown_codes_backwards_time_and_trailing_bytes(self):
        ev, _ = request(10_000, 24)
        unbalanced = drop_nth(ev, "GATHER_EXIT", 10)
        write_trace(str(self.tmp / "u"), unbalanced)
        p = self.stats(str(self.tmp / "u"))
        self.assertTrue(any("without" in r for r in p["accounting"]["malformed_reasons"]), p["accounting"])
        # a word with an unknown event code and a trailing partial write
        d = self.tmp / "w"
        write_trace(str(d), ev)
        with open(d / "timeline-100.bin", "ab") as f:
            f.write((5 << step_timeline.SHIFT | 63).to_bytes(8, "little", signed=True) + b"\x01\x02\x03")
        p = self.stats(str(d))
        self.assertEqual(p["status"], "partial")
        self.assertTrue(any("unknown code" in r for r in p["reasons"]) and any("trailing bytes" in r for r in p["reasons"]), p["reasons"])
        # time going backwards inside a pass
        ev2 = list(ev)
        i = [i for i, e in enumerate(ev2) if e[1] == "LL_ENTER"][10]
        ev2[i] = (ev2[i][0] - 10_000_000, "LL_ENTER")
        write_trace(str(self.tmp / "b"), ev2)
        self.assertIn("timestamps go backwards", self.stats(str(self.tmp / "b"))["accounting"]["malformed_reasons"])

    def test_an_unfinished_pass_at_the_end_makes_the_trace_partial(self):
        ev, t = request(10_000, 24)
        ev += [(t + 100, "FL_ENTER"), (t + 600, "RECV_ENTER")]  # cancelled or killed in the middle of a pass
        write_trace(str(self.tmp / "i"), ev)
        p = self.stats(str(self.tmp / "i"))
        self.assertEqual((p["status"], p["accounting"]["unfinished_passes"]), ("partial", 1))

    def test_a_process_that_never_closed_is_partial_and_require_complete_fails(self):
        d = self.trace("k", final=False)
        p = self.stats(d)
        self.assertEqual(p["status"], "partial")
        self.assertTrue(any("not closed" in r for r in p["reasons"]))
        self.assertEqual(timeline_report.main([d, "--require-complete"]), 2)
        self.assertEqual(timeline_report.main([self.trace("c")]), 0)
        self.assertEqual(timeline_report.main([self.trace("c2"), "--require-complete"]), 0)

    def test_cancelled_requests_are_not_part_of_the_steady_statistics(self):
        d = self.trace("x", lengths=(24, 24, 3, 24, 5))  # two requests dropped after a few tokens
        s = self.stats(d)["statistics"]
        self.assertEqual((s["requests"], s["decode_length"], s["decode_steps"]), (3, 24, 69))
        self.assertEqual(s["request_lengths"], {"3": 1, "5": 1, "24": 3})

    def test_ranks_are_attributed_and_never_combined_and_output_is_deterministic(self):
        a = self.trace("r0", pid=11, host="hostA", rank=0)
        b = self.trace("r1", pid=22, host="hostB", rank=1)
        r1 = timeline_report.build_report([b, a])
        r2 = timeline_report.build_report([a, b])
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))
        self.assertEqual([(p["host"], p["rank"]) for p in r1["processes"]], [("hostA", 0), ("hostB", 1)])
        self.assertEqual(timeline_report.render_text(r1), timeline_report.render_text(r2))
        self.assertEqual(timeline_report.render_markdown(r1), timeline_report.render_markdown(r2))

    def test_several_lifecycles_are_reported_separately(self):
        dirs = [self.trace(f"l{i}", pid=100 + i) for i in range(3)]
        report = timeline_report.build_report(dirs)
        self.assertEqual([p["status"] for p in report["processes"]], ["complete"] * 3)
        self.assertEqual([p["dir"] for p in report["processes"]], sorted(dirs))

    def test_unreadable_metadata_is_reported_not_raised(self):
        d = self.tmp / "bad"
        d.mkdir()
        (d / "timeline-7.json").write_text("{not json")
        (d / "timeline-7.bin").write_bytes(b"")
        (p,) = timeline_report.build_report([str(d)])["processes"]
        self.assertEqual(p["status"], "partial")
        self.assertEqual(timeline_report.main([str(d), "--require-complete"]), 2)

    def test_an_empty_directory_fails_require_complete(self):
        self.assertEqual(timeline_report.main([str(self.tmp), "--require-complete"]), 2)

    def test_the_first_decode_step_after_prefill_and_prefill_are_not_steady_state(self):
        s = self.stats(self.trace("p", lengths=(24,)))["statistics"]
        self.assertEqual(s["decode_steps"], 23)


if __name__ == "__main__":
    unittest.main()

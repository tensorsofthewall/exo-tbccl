"""Phase 57 measurement-only recorder: every mx.eval / mx.async_eval, every communication call and, for TbcclPipelineComm, every borrow, DLPack export
and native wait, with entry/exit timestamps (time.perf_counter_ns) and a SEMANTIC label.

Not part of the library and not imported by it. Enable it from a driver with `install(comm, rank, backend)`; it patches mlx.core.eval/async_eval (exo's
pipeline calls them through the module, so the patch sees every call), wraps the comm, and for a TbcclPipelineComm also patches exo_tbccl.group.borrow,
the native Export constructor and TbcclPipelineComm.wait. Nothing it patches changes behaviour; it only timestamps and delegates.

An eval's label is derived from the call site, not from the call: (file, qualified function, n-th `mx.eval(` line of that function) is looked up in
`SITES`, so two textually identical `mx.eval(x)` lines in one function get different semantic names. Unknown sites keep `file:function`.
"""

import functools
import inspect
import json
import sys
import threading
import time

import mlx.core as mx

# (file basename, qualified function) -> labels by ordinal of the `mx.eval(` / `mx.async_eval(` lines in that function
SITES: dict[tuple[str, str], list[str]] = {
    ("auto_parallel.py", "PipelineFirstLayer.__call__"): ["pre_recv_template_eval", "post_recv_eval"],
    ("auto_parallel.py", "PipelineLastLayer.__call__"): ["model_output_eval", "send_dependency_eval", "cache_dependency_eval", "post_allgather_eval"],
    ("bridge.py", "borrow"): ["borrow_array_eval", "borrow_view_eval", "borrow_contiguous_eval", "borrow_contiguous_view_eval"],
    ("group.py", "recv_like"): ["recv_destination_eval"],
    ("group.py", "all_gather"): ["allgather_destination_eval"],
    ("group.py", "TbcclPipelineComm._alloc"): ["destination_alloc_eval"],
    ("pipeline_comm.py", "MlxPipelineComm.flush_sends"): ["flush_async_eval"],
    # Phase 59 remote-peer emulator: one eval per helper, named like the real pipeline's evals so distributed_timeline.py reads both
    ("synthetic_mac_stage.py", "sampler_eval"): ["real_model_loopback.py:worker#4"],
    ("synthetic_mac_stage.py", "stage_eval"): ["model_output_eval"],
    ("synthetic_mac_stage.py", "do_send"): ["send_dependency_eval"],
    ("synthetic_mac_stage.py", "do_gather"): ["post_allgather_eval"],
    ("remote_peer_emulator.py", "do_send"): ["send_dependency_eval"],
    ("remote_peer_emulator.py", "do_recv"): ["post_recv_eval"],
    ("remote_peer_emulator.py", "do_gather"): ["post_allgather_eval"],
}

_site_cache: dict[object, dict[int, str]] = {}


def _label_for(frame) -> str:
    code = frame.f_code
    base = code.co_filename.rsplit("/", 1)[-1]
    qual = getattr(code, "co_qualname", code.co_name)
    table = _site_cache.get(code)
    if table is None:
        table = {}
        labels = SITES.get((base, qual))
        try:
            lines, first = inspect.getsourcelines(code)
        except (OSError, TypeError):
            lines, first = [], 0
        ordinals = [first + i for i, l in enumerate(lines) if "mx.eval(" in l or "mx.async_eval(" in l]
        for n, lineno in enumerate(ordinals):
            table[lineno] = labels[n] if labels and n < len(labels) else f"{base}:{qual}#{n}"
        _site_cache[code] = table
    return table.get(frame.f_lineno, f"{base}:{qual}:{frame.f_lineno}")


class SyncRecorder:
    def __init__(self, rank: int, backend: str):
        self.rank = rank
        self.backend = backend
        # (t0, t1, kind, label, depth, step, op, tid): step = decode forward-step index (-1 outside decode), op = per-rank communication-call number (-1 outside one)
        self.events: list[tuple[int, int, str, str, int, int, int, int]] = []
        self.phase = "prefill"
        self.clock: dict = {}  # filled by the driver (benchmarks/clock_sync.py results)
        self._tl = threading.local()
        self._orig: list[tuple[object, str, object]] = []
        self._completed = 0  # decode step_complete calls finished so far
        self._res0 = None
        self.activity = None  # benchmarks/activity_thread.Activity (Phase 59 control), set by the driver
        self._op = 0

    def _depth(self) -> int:
        return getattr(self._tl, "d", 0)

    def _step(self, kind: str, label: str) -> int:
        """The forward step an event belongs to. exo's step is [compute][step_complete][send][all_gather] on rank 0 and
        [pre_recv eval][recv][compute][step_complete][all_gather] on rank 1, followed by the sampler eval: events before the step's step_complete
        belong to step `completed`, events after it (and the sampler, which samples that step's logits) to step `completed - 1`."""
        if self.phase != "decode":
            return -1
        cur = getattr(self._tl, "op", None)
        if cur == "recv_like" or label in ("pre_recv_template_eval", "post_recv_eval", "model_output_eval", "recv_like", "wait:recv"):
            return self._completed
        return self._completed - 1

    def timed(self, kind: str, label: str, fn, *args, **kw):
        d = self._depth()
        self._tl.d = d + 1
        outer_op = getattr(self._tl, "op", None)
        op = getattr(self._tl, "opid", -1)
        if kind == "comm" and d == 0 and label in ("send", "recv_like", "all_gather", "barrier", "any_true"):
            self._tl.op = label
            self._op += 1
            op = self._tl.opid = self._op
        act = self.activity
        if act is not None and act.window is not None and self.phase == "decode":
            act.on_event("begin", kind, label)
        t0 = time.perf_counter_ns()
        try:
            if kind == "comm" and label == "step_complete" and self.phase == "decode":
                self._completed += 1
            return fn(*args, **kw)
        finally:
            t1 = time.perf_counter_ns()
            if act is not None and act.window is not None and self.phase == "decode":
                act.on_event("end", kind, label)
            self._tl.d = d
            step = self._step(kind, label)
            if kind == "comm" and d == 0 and label in ("send", "recv_like", "all_gather", "barrier", "any_true"):
                self._tl.op, self._tl.opid = outer_op, -1
            self.events.append((t0, t1, kind, label if self.phase == "decode" else "prefill:" + label, d, step, op, threading.get_ident()))

    def instrument_model(self, model) -> None:
        """Phase 59 first-use/graph-build breakdown: time every layer __call__ (host graph construction; lazy MLX returns before any GPU work) so evals nested in a
        layer call, if any, are visible as depth>0 eval events inside it. Patches the layer CLASSES of the pipelined model; undone by uninstall()."""
        rec = self
        seen = set()
        for layer in model.layers:
            cls = type(layer)
            if cls in seen or not hasattr(cls, "__call__"):
                continue
            seen.add(cls)
            orig = cls.__call__

            def make(orig, name):
                def call(self_, *a, **kw):
                    return rec.timed("layer", name, orig, self_, *a, **kw)
                return call

            self._patch(cls, "__call__", make(orig, f"layer:{cls.__name__}"))

    def add(self, kind: str, label: str, t0: int, t1: int, depth: int = 0) -> None:
        """Record a synthetic event (the emulator's modelled compute/sampler intervals) with the step the real pipeline would give it."""
        self.events.append((t0, t1, kind, label if self.phase == "decode" else "prefill:" + label, depth, self._step(kind, label), -1, threading.get_ident()))

    def _patch(self, obj, name, new):
        self._orig.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)

    @staticmethod
    def _resources():
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        out = {"utime_s": r.ru_utime, "stime_s": r.ru_stime, "nvcsw": r.ru_nvcsw, "nivcsw": r.ru_nivcsw, "t_ns": time.perf_counter_ns()}
        try:
            import psutil

            out["threads"] = {str(t.id): [t.user_time, t.system_time] for t in psutil.Process().threads()}
        except Exception:  # noqa: BLE001
            pass
        return out

    def mark_decode_start(self):
        """Snapshot process CPU time / context switches / per-thread CPU at the start of decode (driver calls it); dump() adds the deltas."""
        self._res0 = self._resources()

    def install(self, comm) -> object:
        orig_eval, orig_async = mx.eval, mx.async_eval
        rec = self

        @functools.wraps(orig_eval)
        def eval_(*args):
            lab = _label_for(sys._getframe(1))
            act = getattr(rec, "activity", None)
            if act is not None and lab in ("send_dependency_eval", "post_recv_eval", "post_allgather_eval"):
                act.comm_begin()  # MlxRing executes its lazy transfers inside these evals: the transfer is outstanding here
                try:
                    return rec.timed("eval", lab, orig_eval, *args)
                finally:
                    act.comm_end()
            return rec.timed("eval", lab, orig_eval, *args)

        @functools.wraps(orig_async)
        def async_eval_(*args):
            return rec.timed("async_eval", _label_for(sys._getframe(1)), orig_async, *args)

        self._patch(mx, "eval", eval_)
        self._patch(mx, "async_eval", async_eval_)
        if type(comm).__name__ == "TbcclPipelineComm":
            import exo_tbccl.bridge as bridge
            import exo_tbccl.group as group

            orig_borrow = group.borrow

            def borrow_(array, *a, **kw):
                return rec.timed("borrow", "borrow", orig_borrow, array, *a, **kw)

            self._patch(group, "borrow", borrow_)
            orig_export = bridge.native.Export

            def export_(probe):
                return rec.timed("dlpack_export", "native.Export(__dlpack__)", orig_export, probe)

            self._patch(bridge.native, "Export", export_)
            orig_wait = group.TbcclPipelineComm.wait

            def wait_(self_, t, *a, **kw):
                return rec.timed("tbccl_wait", f"wait:{t.op}", orig_wait, self_, t, *a, **kw)

            self._patch(group.TbcclPipelineComm, "wait", wait_)
            orig_submit = group.TbcclPipelineComm._submit

            def submit_(self_, op, peer, borrows, call):
                return rec.timed("tbccl_submit", f"submit:{op}", orig_submit, self_, op, peer, borrows, call)

            self._patch(group.TbcclPipelineComm, "_submit", submit_)
        return _CommProxy(comm, self)

    def uninstall(self) -> None:
        for obj, name, orig in reversed(self._orig):
            setattr(obj, name, orig)
        self._orig.clear()

    def dump(self, path: str) -> None:
        import socket

        with open(path, "w") as f:
            res = None
            if self._res0 is not None:
                r1 = self._resources()
                res = {k: r1[k] - self._res0[k] for k in ("utime_s", "stime_s", "nvcsw", "nivcsw", "t_ns")}
                if "threads" in r1:
                    res["threads"] = {tid: [round(v[0] - self._res0.get("threads", {}).get(tid, [0, 0])[0], 3), round(v[1] - self._res0.get("threads", {}).get(tid, [0, 0])[1], 3)]
                                      for tid, v in r1["threads"].items()}
            qos = None
            try:
                import ctypes

                lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                cls, rel = ctypes.c_int(0), ctypes.c_int(0)
                lib.pthread_self.restype = ctypes.c_void_p
                lib.pthread_get_qos_class_np(ctypes.c_void_p(lib.pthread_self()), ctypes.byref(cls), ctypes.byref(rel))
                qos = {"main_thread_qos_class": cls.value}  # 0x21 interactive, 0x19 user-initiated, 0x15 default, 0x11 utility, 0x09 background
            except Exception:  # noqa: BLE001
                pass
            json.dump({"rank": self.rank, "backend": self.backend, "host": socket.gethostname(), "pid": __import__("os").getpid(), "clock": self.clock,
                       "resources_decode": res, "qos": qos, "activity_cpu_s": getattr(self.activity, "cpu_s", None), "events": self.events}, f)
        with open(path[:-5] + ".jsonl" if path.endswith(".json") else path + ".jsonl", "w") as f:  # the same events, one JSON object per line
            f.write(json.dumps({"meta": {"rank": self.rank, "backend": self.backend, "host": socket.gethostname(), "pid": __import__("os").getpid(), "clock": self.clock}}) + "\n")
            for t0, t1, kind, label, depth, step, op, tid in self.events:
                f.write(json.dumps({"timestamp_ns": t0, "end_ns": t1, "host": socket.gethostname(), "rank": self.rank, "pid": __import__("os").getpid(), "thread_id": tid,
                                    "token_id": step, "operation_id": op, "event": f"{kind}:{label}", "kind": kind, "label": label, "depth": depth, "backend": self.backend}) + "\n")


_COMM_OPS = ("send", "recv_like", "all_gather", "barrier", "any_true", "flush_sends", "step_complete")


class _CommProxy:
    """Delegating wrapper: comm calls become `comm` events; everything else (including attribute reads like `.stats`) passes through."""

    def __init__(self, comm, rec: SyncRecorder):
        object.__setattr__(self, "_comm", comm)
        object.__setattr__(self, "_rec", rec)

    def __getattr__(self, name):
        attr = getattr(self._comm, name)
        if name in _COMM_OPS and callable(attr):
            act = getattr(self._rec, "activity", None)
            if act is not None and name in ("send", "recv_like", "all_gather", "barrier", "any_true", "flush_sends"):
                def with_activity(*a, **kw):
                    act.comm_begin()
                    try:
                        return self._rec.timed("comm", name, attr, *a, **kw)
                    finally:
                        act.comm_end()
                return with_activity
            return lambda *a, **kw: self._rec.timed("comm", name, attr, *a, **kw)
        return attr

    def __setattr__(self, name, value):
        if name == "phase":
            self._rec.phase = value
        else:
            setattr(self._comm, name, value)

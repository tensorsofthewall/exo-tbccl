"""The per-token timeline work measurement-only recorder: every mx.eval / mx.async_eval, every communication call and, for TbcclPipelineComm, every borrow, DLPack export
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
        self.events: list[tuple[int, int, str, str, int]] = []  # (t0, t1, kind, label, depth)
        self.phase = "prefill"
        self._tl = threading.local()
        self._orig: list[tuple[object, str, object]] = []

    def _depth(self) -> int:
        return getattr(self._tl, "d", 0)

    def timed(self, kind: str, label: str, fn, *args, **kw):
        d = self._depth()
        self._tl.d = d + 1
        t0 = time.perf_counter_ns()
        try:
            return fn(*args, **kw)
        finally:
            t1 = time.perf_counter_ns()
            self._tl.d = d
            self.events.append((t0, t1, kind, label, d))
            self.events[-1] = (t0, t1, kind, label if self.phase == "decode" else "prefill:" + label, d)

    def _patch(self, obj, name, new):
        self._orig.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)

    def install(self, comm) -> object:
        orig_eval, orig_async = mx.eval, mx.async_eval
        rec = self

        @functools.wraps(orig_eval)
        def eval_(*args):
            return rec.timed("eval", _label_for(sys._getframe(1)), orig_eval, *args)

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
        return _CommProxy(comm, self)

    def uninstall(self) -> None:
        for obj, name, orig in reversed(self._orig):
            setattr(obj, name, orig)
        self._orig.clear()

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"rank": self.rank, "backend": self.backend, "events": self.events}, f)


_COMM_OPS = ("send", "recv_like", "all_gather", "barrier", "any_true", "flush_sends", "step_complete")


class _CommProxy:
    """Delegating wrapper: comm calls become `comm` events; everything else (including attribute reads like `.stats`) passes through."""

    def __init__(self, comm, rec: SyncRecorder):
        object.__setattr__(self, "_comm", comm)
        object.__setattr__(self, "_rec", rec)

    def __getattr__(self, name):
        attr = getattr(self._comm, name)
        if name in _COMM_OPS and callable(attr):
            return lambda *a, **kw: self._rec.timed("comm", name, attr, *a, **kw)
        return attr

    def __setattr__(self, name, value):
        if name == "phase":
            self._rec.phase = value
        else:
            setattr(self._comm, name, value)

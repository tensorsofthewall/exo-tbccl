"""Per-step pipeline timeline for the diagnostic trace hook (``TBCCL_TRACE_TIMELINE=1`` together with ``TBCCL_B6_TRACE_DIR``; off otherwise).

The hook wraps the points of exo's pipeline step that are the same for ``MlxRing`` and ``MlxTbccl`` (the first and last pipeline layer and the four ``PipelineComm`` methods the
layers call) and, for ``MlxTbccl`` only, the internal stages of ``TbcclPipelineComm`` (destination allocation, borrow/export, submission, the blocking wait). A wrapped call appends
one integer (``perf_counter_ns << 6 | event``) to an in-memory ``array('q')``: no formatting, no I/O and no lock in the decode path. A background thread appends the new slice to
``timeline-<pid>.bin`` about twice a second, so a runner that is killed loses at most the last half second. Timestamps come from the local monotonic clock only and are never
compared across hosts. The tool imports nothing from the production ``exo_tbccl`` runtime at import time; it wraps the classes after exo imports them.

Bounds: at most ``TBCCL_TRACE_TIMELINE_MAX_EVENTS`` events (default 8,000,000, 64 MiB on disk) are recorded per process; later events are dropped, counted, and the process is marked
``truncated``. ``timeline-<pid>.json`` carries ``final: true`` only after the recorder was closed at interpreter exit, so a killed or crashed process is reported as partial.

Events per decode step on one rank, in program order (a stage without a receive has no RECV events, the last stage no SEND events)::

    FL_ENTER [RECV_ENTER ... RECV_EXIT] FL_EXIT   LL_ENTER  STEP_COMPLETE  [SEND_ENTER SEND_EXIT]  GATHER_ENTER ... GATHER_EXIT  LL_EXIT (or LL_EXIT_PREFILL)

``MlxRing`` builds lazy graph nodes in the four comm methods and performs the transfer inside the next ``mx.eval``, so the intervals *between* these events are what is comparable
across backends (``benchmarks/provider/timeline_report.py`` defines them). At every ``LL_EXIT`` the process CPU time and the voluntary/involuntary context-switch counters
(``getrusage``) are recorded as extra events. Traced runs are never mixed with performance runs; measure the overhead by interleaving traced and untraced lifecycles.
"""
from __future__ import annotations

import array
import json
import os
import resource
import socket
import sys
import threading
import time

SCHEMA = "tbccl-decode-timeline/1"
EVENTS = {
    "FL_ENTER": 1, "FL_EXIT": 2, "RECV_ENTER": 3, "RECV_EXIT": 4, "LL_ENTER": 5, "STEP_COMPLETE": 6, "SEND_ENTER": 7, "SEND_EXIT": 8, "GATHER_ENTER": 9, "GATHER_EXIT": 10,
    "LL_EXIT": 11, "LL_EXIT_PREFILL": 12, "CPU_NS": 13, "NVCSW": 14, "NIVCSW": 15,
    "ALLOC_ENTER": 20, "ALLOC_EXIT": 21, "BORROW_ENTER": 22, "BORROW_EXIT": 23, "SUBMIT_ENTER": 24, "SUBMIT_EXIT": 25, "WAIT_ENTER": 26, "WAIT_EXIT": 27,
    "DRAIN_ENTER": 28, "DRAIN_EXIT": 29, "REAP_ENTER": 30, "REAP_EXIT": 31,
}
SHIFT = 6
FLUSH_S = 0.5
DEFAULT_MAX_EVENTS = 8_000_000
_E = EVENTS


def _version(distribution: str):
    try:
        import importlib.metadata

        return importlib.metadata.version(distribution)
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


class Recorder:
    def __init__(self, directory: str):
        self.dir = directory
        self.pid = os.getpid()
        self.buf = array.array("q")
        self.written = 0
        self.max_events = int(os.environ.get("TBCCL_TRACE_TIMELINE_MAX_EVENTS", DEFAULT_MAX_EVENTS) or DEFAULT_MAX_EVENTS)
        self.full = False
        self.backend: str | None = None
        self.rank: int | None = None
        self.world_size: int | None = None
        self.threads_seen: set[int] = set()
        self.comms: list = []
        self.write_errors = 0
        self.final = False
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.t0_ns = time.perf_counter_ns()
        self.wall0_ns = time.time_ns()
        self.bin_path = os.path.join(directory, f"timeline-{self.pid}.bin")
        self.meta_path = os.path.join(directory, f"timeline-{self.pid}.json")
        self.thread = threading.Thread(target=self._loop, name="trace-timeline-flush", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.wait(FLUSH_S):
            self.flush()

    def flush(self) -> None:
        """Append the new events and rewrite the metadata. Never raises: a diagnostic must not break the runner, a failed write is counted and makes the process partial."""
        with self.lock:
            n = len(self.buf)
            try:
                if n > self.written:
                    with open(self.bin_path, "ab") as f:
                        self.buf[self.written:n].tofile(f)
                    self.written = n
                if self.written >= self.max_events:
                    self.full = True
                meta = self._meta()
                tmp = self.meta_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(meta, f, sort_keys=True)
                os.replace(tmp, self.meta_path)
            except OSError:
                self.write_errors += 1

    def _meta(self) -> dict:
        meta = {
            "schema": SCHEMA, "pid": self.pid, "host": socket.gethostname(), "platform": sys.platform, "python": sys.version.split()[0],
            "backend": self.backend, "rank": self.rank, "world_size": self.world_size, "events": EVENTS, "shift": SHIFT,
            "perf_ns_at_start": self.t0_ns, "wall_ns_at_start": self.wall0_ns, "perf_ns_at_flush": time.perf_counter_ns(), "wall_ns_at_flush": time.time_ns(),
            "threads_seen": sorted(self.threads_seen), "events_written": self.written, "max_events": self.max_events, "truncated": self.full, "write_errors": self.write_errors,
            "final": self.final,
            "tracing": {"TBCCL_TRACE_TIMELINE": os.environ.get("TBCCL_TRACE_TIMELINE"), "EXO_TBCCL_TRACE": os.environ.get("EXO_TBCCL_TRACE"),
                        "exo_tbccl_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("EXO_TBCCL_")}},
            "software": {"exo_tbccl": _version("exo-tbccl"), "mlx": _version("mlx"), "exo": _version("exo")},
            "comms": [],
        }
        for c in list(self.comms):
            entry = {"rank": getattr(c, "_rank", None)}
            try:
                ws = getattr(c, "wait_stats", None)
                if ws is not None:
                    entry["wait_stats"] = dict(vars(ws))
                entry["activity"] = c.activity_stats()
                entry["config"] = dict(vars(c.config)) if hasattr(c, "config") else None
                ps = getattr(getattr(c, "pool", None), "stats", None)
                if ps is not None:
                    entry["pool"] = dict(vars(ps))
                entry["copy"] = {"materialized_copies": c.stats.materialized_copies, "staged_fallback_copies": c.stats.staged_fallback_copies, "direct_bytes": c.stats.direct_bytes}
            except Exception as e:  # noqa: BLE001 - a diagnostic must never raise into the runner
                entry["error"] = repr(e)
            meta["comms"].append(entry)
        return meta

    def close(self) -> None:
        self.stop.set()
        self.final = True
        self.flush()


def install(directory: str):
    """Create the recorder and return ``(recorder, after_import)``: ``after_import(module_name, module)`` wraps whatever that module offers."""
    os.makedirs(directory, exist_ok=True)
    rec = Recorder(directory)
    append = rec.buf.append
    now = time.perf_counter_ns
    getrusage = resource.getrusage
    RUSAGE_SELF = resource.RUSAGE_SELF
    ident = threading.get_ident

    def timed(fn, enter, exit_):
        def wrapper(*a, **k):
            if rec.full:
                return fn(*a, **k)
            append(now() << SHIFT | enter)
            try:
                return fn(*a, **k)
            finally:
                append(now() << SHIFT | exit_)
        wrapper.__wrapped__ = fn
        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        return wrapper

    def stamp(fn, code):
        def wrapper(*a, **k):
            if not rec.full:
                append(now() << SHIFT | code)
            return fn(*a, **k)
        wrapper.__wrapped__ = fn
        return wrapper

    def first_layer(fn):
        def wrapper(self, *a, **k):
            if rec.full:
                return fn(self, *a, **k)
            rec.threads_seen.add(ident())
            if rec.rank is None:
                rec.rank, rec.world_size = getattr(self, "r", None), None
            append(now() << SHIFT | _E["FL_ENTER"])
            try:
                return fn(self, *a, **k)
            finally:
                append(now() << SHIFT | _E["FL_EXIT"])
        wrapper.__wrapped__ = fn
        return wrapper

    def last_layer(fn):
        def wrapper(self, *a, **k):
            if rec.full:
                return fn(self, *a, **k)
            if rec.world_size is None:
                rec.world_size = getattr(self, "s", None)
            append(now() << SHIFT | _E["LL_ENTER"])
            try:
                return fn(self, *a, **k)
            finally:
                code = _E["LL_EXIT_PREFILL"] if getattr(self, "is_prefill", False) else _E["LL_EXIT"]
                append(now() << SHIFT | code)
                ru = getrusage(RUSAGE_SELF)
                append(int((ru.ru_utime + ru.ru_stime) * 1e9) << SHIFT | _E["CPU_NS"])
                append(ru.ru_nvcsw << SHIFT | _E["NVCSW"])
                append(ru.ru_nivcsw << SHIFT | _E["NIVCSW"])
        wrapper.__wrapped__ = fn
        return wrapper

    def comm_methods(cls):
        cls.recv_like = timed(cls.recv_like, _E["RECV_ENTER"], _E["RECV_EXIT"])
        cls.send = timed(cls.send, _E["SEND_ENTER"], _E["SEND_EXIT"])
        cls.all_gather = timed(cls.all_gather, _E["GATHER_ENTER"], _E["GATHER_EXIT"])
        cls.step_complete = stamp(cls.step_complete, _E["STEP_COMPLETE"])

    def after_import(name: str, module) -> None:
        if name == "exo.worker.engines.mlx.auto_parallel":
            module.PipelineFirstLayer.__call__ = first_layer(module.PipelineFirstLayer.__call__)
            module.PipelineLastLayer.__call__ = last_layer(module.PipelineLastLayer.__call__)
        elif name == "exo.worker.engines.mlx.pipeline_comm":
            if rec.backend is None:
                rec.backend = "ring"
            comm_methods(module.MlxPipelineComm)
        elif name == "exo_tbccl.group":
            rec.backend = "tbccl"
            cls = module.TbcclPipelineComm
            comm_methods(cls)
            cls._alloc = timed(cls._alloc, _E["ALLOC_ENTER"], _E["ALLOC_EXIT"])
            cls._submit = timed(cls._submit, _E["SUBMIT_ENTER"], _E["SUBMIT_EXIT"])
            cls.wait = timed(cls.wait, _E["WAIT_ENTER"], _E["WAIT_EXIT"])
            cls._drain_async_sends = timed(cls._drain_async_sends, _E["DRAIN_ENTER"], _E["DRAIN_EXIT"])
            cls._reap = timed(cls._reap, _E["REAP_ENTER"], _E["REAP_EXIT"])
            module.borrow = timed(module.borrow, _E["BORROW_ENTER"], _E["BORROW_EXIT"])
            original_init = cls.__init__

            def init(self, *a, **k):
                original_init(self, *a, **k)
                rec.comms.append(self)
            cls.__init__ = init

    rec.start()
    return rec, after_import

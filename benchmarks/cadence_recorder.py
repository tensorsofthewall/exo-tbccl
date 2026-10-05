"""Phase 56 measurement wrapper: records entry/exit timestamps (time.perf_counter_ns) of every communication call a pipeline comm receives.

Not part of the library. Wrap a PipelineComm (MlxPipelineComm or TbcclPipelineComm) before handing it to exo's pipeline_auto_parallel; set `.phase`
("prefill" / "decode") from the driver; call `dump(path)` at the end. Delegates everything else, so the wrapped comm behaves unchanged.
"""

import json
import time

_OPS = ("send", "recv_like", "all_gather", "barrier", "any_true", "flush_sends", "send_async", "recv_into_async", "wait", "wait_all", "step_complete")


class CadenceRecorder:
    def __init__(self, comm, rank):
        object.__setattr__(self, "_comm", comm)
        object.__setattr__(self, "_rank", rank)
        object.__setattr__(self, "_events", [])
        object.__setattr__(self, "phase", "prefill")

    def _wrap(self, name, fn):
        events = self._events

        def call(*args, **kw):
            nbytes = 0
            for a in args:
                n = getattr(a, "nbytes", None)
                if isinstance(n, int):
                    nbytes = n
                    break
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kw)
            finally:
                events.append((t0, time.perf_counter_ns(), name, nbytes, self.phase))

        return call

    def __getattr__(self, name):
        attr = getattr(self._comm, name)
        return self._wrap(name, attr) if name in _OPS and callable(attr) else attr

    def dump(self, path):
        with open(path, "w") as f:
            json.dump({"rank": self._rank, "events": self._events}, f)

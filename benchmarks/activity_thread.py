"""Phase 59 TEST-ONLY control: an external CPU-activity thread that imitates what MlxRing's communication worker does on the Mac without touching the
communication backend. MlxRing sets its sockets non-blocking and busy-polls recv/send for as long as a transfer is pending (about 0.5 core on the Mac);
TbcclPipelineComm blocks. Enable with EXO_P59_ACTIVITY (read by benchmarks/real_model_loopback.py):

    spin          a thread that burns CPU continuously (without holding the GIL for long: it spins in libc memset calls)
    comm          the same spinning, but ONLY while a communication call is outstanding (the recorder proxy raises/clears the flag): Ring-like
    duty:<pct>    spins <pct>% of every 1 ms period (a controlled average CPU: duty:50 ~ 0.5 core)
    off / unset   nothing
    win:<name>[@<duty>]   Phase 61 step windows, gated by recorder events (benchmarks/sync_recorder.py calls on_event for every recorded call/eval/layer):
                  step       from the end of an all_gather to the begin of the next one (everything the Mac does between collectives)
                  compute    from the end of recv_like to the begin of the next all_gather (orientation A: recv complete -> AllGather submission)
                  graphstage from the begin of the first TransformerBlock call of a step to the end of model_output_eval (graph build + Metal stage)
                  graph      only while a TransformerBlock.__call__ runs (needs EXO_P59_LAYERS=1)
                  stage      only around mx.eval(model_output_eval)
                  sampler    only around the token eval (lm_head + argmax)
                  <duty>     optional percent of each 1 ms period spent spinning while the window is open (default 100)

It never changes what the pipeline computes or communicates. The thread is joined at the end of the run.
"""
import ctypes
import ctypes.util
import threading
import time

_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_BUF = ctypes.create_string_buffer(1 << 18)


def _burn(until_ns=None):
    memset = _libc.memset
    while until_ns is None or time.perf_counter_ns() < until_ns:
        memset(_BUF, 0, len(_BUF))  # releases the GIL for the duration of the call
        if until_ns is None:
            return


WINDOWS = {
    # name: (open events, close events); an event is (phase, kind, label-or-prefix)
    "step": ([("end", "comm", "all_gather")], [("begin", "comm", "all_gather")]),
    "compute": ([("end", "comm", "recv_like")], [("begin", "comm", "all_gather")]),
    "graphstage": ([("begin", "layer", "layer:TransformerBlock")], [("end", "eval", "model_output_eval")]),
    "graph": ([("begin", "layer", "layer:TransformerBlock")], [("end", "layer", "layer:TransformerBlock")]),
    "stage": ([("begin", "eval", "model_output_eval")], [("end", "eval", "model_output_eval")]),
    "sampler": ([("begin", "eval", "real_model_loopback.py:worker#4")], [("end", "eval", "real_model_loopback.py:worker#4")]),
    # Phase 63 continuous positive control: from the first decode receive until the final barrier (spans the KV-digest windows too)
    "decode": ([("begin", "comm", "recv_like")], [("begin", "comm", "barrier")]),
}


class Activity:
    def __init__(self, mode: str):
        self.mode = mode or "off"
        self.window = None
        self.duty = 1.0
        if self.mode.startswith("win:"):
            spec = self.mode[4:]
            name, _, duty = spec.partition("@")
            self.window = WINDOWS[name]
            self.duty = float(duty) / 100.0 if duty else 1.0
            self.mode = "window"
        self.burst_s = 0.0
        self.stop = threading.Event()
        self.outstanding = threading.Event()
        self.cpu_s = 0.0
        self.native_id = None  # the helper's OS thread id (the libproc id the external sampler reports)
        self.events = []  # (perf_counter_ns, "activity_begin" | "activity_end") as seen by the pipeline thread (Phase 63)
        self.bursts = []  # (begin_ns, end_ns, helper thread CPU ns) per active period, measured by the helper itself
        self.thread = None
        if self.mode != "off":
            self.thread = threading.Thread(target=self._run, daemon=True, name="p59-activity")
            self.thread.start()

    def _run(self):
        self.native_id = threading.get_native_id()
        t_start = time.thread_time()
        try:
            if self.mode == "spin":
                while not self.stop.is_set():
                    _burn()
            elif self.mode == "comm":
                while not self.stop.is_set():
                    if self.outstanding.is_set():
                        _burn()
                    else:
                        self.outstanding.wait(0.02)
            elif self.mode == "window":
                burst = None
                while not self.stop.is_set():
                    if self.outstanding.is_set():
                        if burst is None:
                            burst = (time.perf_counter_ns(), time.thread_time_ns())
                        if self.duty >= 1.0:
                            _burn()
                        else:
                            t0 = time.perf_counter_ns()
                            _burn(t0 + int(self.duty * 1e6))
                            time.sleep(max(0.0, (1e6 - (time.perf_counter_ns() - t0)) / 1e9))
                    else:
                        if burst is not None:
                            self.bursts.append((burst[0], time.perf_counter_ns(), time.thread_time_ns() - burst[1]))
                            burst = None
                        self.outstanding.wait(0.02)
            elif self.mode.startswith("duty:"):
                duty = float(self.mode.split(":")[1]) / 100.0
                while not self.stop.is_set():
                    t0 = time.perf_counter_ns()
                    _burn(t0 + int(duty * 1e6))
                    time.sleep(max(0.0, (1e6 - (time.perf_counter_ns() - t0)) / 1e9) if duty < 1 else 0)
        finally:
            self.cpu_s = time.thread_time() - t_start

    def on_event(self, phase: str, kind: str, label: str):
        """Recorder hook (window modes): open or close the window at the configured events. Costs two comparisons per recorded event. Returns "activity_begin" /
        "activity_end" when the state changed (the recorder stores it as an event with the step id), else None."""
        if self.window is None:
            return None
        opens, closes = self.window
        key = (phase, kind, label)
        if key in closes or (phase, kind, label.split(":prefill")[0]) in closes:
            if self.outstanding.is_set():
                self.outstanding.clear()
                self.events.append((time.perf_counter_ns(), "activity_end"))
                return "activity_end"
        elif key in opens and not self.outstanding.is_set():
            self.outstanding.set()
            self.events.append((time.perf_counter_ns(), "activity_begin"))
            return "activity_begin"
        return None

    def comm_begin(self):
        if self.mode == "comm":  # window modes are driven only by on_event (Phase 63: comm_end used to clear an open window)
            self.outstanding.set()

    def comm_end(self):
        if self.mode == "comm":
            self.outstanding.clear()

    def close(self):
        self.stop.set()
        self.outstanding.set()
        if self.thread:
            self.thread.join(timeout=2)

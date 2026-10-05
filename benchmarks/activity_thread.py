"""TEST-ONLY control: an external CPU-activity thread that imitates what MlxRing's communication worker does on the Mac without touching the
communication backend. MlxRing sets its sockets non-blocking and busy-polls recv/send for as long as a transfer is pending (about 0.5 core on the Mac);
TbcclPipelineComm blocks. Enable with EXO_TBCCL_BENCH_ACTIVITY (read by benchmarks/real_model_loopback.py):

    spin          a thread that burns CPU continuously (without holding the GIL for long: it spins in libc memset calls)
    comm          the same spinning, but ONLY while a communication call is outstanding (the recorder proxy raises/clears the flag): Ring-like
    duty:<pct>    spins <pct>% of every 1 ms period (a controlled average CPU: duty:50 ~ 0.5 core)
    off / unset   nothing

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


class Activity:
    def __init__(self, mode: str):
        self.mode = mode or "off"
        self.stop = threading.Event()
        self.outstanding = threading.Event()
        self.cpu_s = 0.0
        self.thread = None
        if self.mode != "off":
            self.thread = threading.Thread(target=self._run, daemon=True, name="activity")
            self.thread.start()

    def _run(self):
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
            elif self.mode.startswith("duty:"):
                duty = float(self.mode.split(":")[1]) / 100.0
                while not self.stop.is_set():
                    t0 = time.perf_counter_ns()
                    _burn(t0 + int(duty * 1e6))
                    time.sleep(max(0.0, (1e6 - (time.perf_counter_ns() - t0)) / 1e9) if duty < 1 else 0)
        finally:
            self.cpu_s = time.thread_time() - t_start

    def comm_begin(self):
        self.outstanding.set()

    def comm_end(self):
        self.outstanding.clear()

    def close(self):
        self.stop.set()
        self.outstanding.set()
        if self.thread:
            self.thread.join(timeout=2)

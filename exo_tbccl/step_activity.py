"""opt-in experiment (``EXO_TBCCL_STEP_ACTIVITY=1``, Metal only, default off): same-process CPU activity between two collectives.

measured on the real Linux<->Mac link (orientation A) that TBCCL's blocking Mac pipeline thread is scheduled on the efficiency cores (~1.2 GHz,
P-cluster ~1 % active) while MlxRing's busy-polling worker keeps the P-cluster busy. A helper thread that only burns CPU while the pipeline is between
two AllGathers moved the Mac onto the performance cores (first-use 1.1 ms -> 0.13 ms, stage 4.3 -> 2.2 ms, TPOT 12.0 -> 6.6 ms, one run) at about half the
CPU of a continuous spinner. The window is defined only by communicator events (no model, rank, split or orientation knowledge):

    opens when an ``all_gather`` completes          (the pipeline step that follows is the process' own work)
    closes when the next ``all_gather`` is submitted, or at barrier / any_true / abort / close

A hard bound (``EXO_TBCCL_STEP_ACTIVITY_MAX_MS``, default 250) ends any single window by itself, so a lost event, an exception or an idle gap between requests
cannot leave a core burning. The helper never touches tensors, Work or communicator state; it only calls libc ``memset`` on a private buffer (the GIL is
released for the duration of every call). Idle, it blocks on an event (zero CPU). It starts lazily at the first window and is joined by ``close()``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
import time

_BUF_BYTES = 1 << 18


class StepActivity:
    def __init__(self, max_window_ms: float = 250.0):
        self._max_s = max_window_ms / 1000.0
        self._go = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._deadline = 0.0
        self.windows = 0
        self.timed_out = 0
        self.cpu_s = 0.0  # the helper's own CPU, accumulated when it parks

    def open(self) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="exo-tbccl-step-activity", daemon=True)
                self._thread.start()
            self._deadline = time.perf_counter() + self._max_s
            self.windows += 1
            self._go.set()

    def close_window(self) -> None:
        self._go.clear()

    def shutdown(self) -> None:
        self._stop.set()
        self._go.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        memset = ctypes.CDLL(ctypes.util.find_library("c")).memset
        buf = ctypes.create_string_buffer(_BUF_BYTES)
        while not self._stop.is_set():
            self._go.wait()
            if self._stop.is_set():
                return
            t0 = time.thread_time()
            while self._go.is_set() and not self._stop.is_set():
                if time.perf_counter() >= self._deadline:
                    self.timed_out += 1
                    self._go.clear()
                    break
                memset(buf, 0, _BUF_BYTES)  # one ~15 us call with the GIL released
            self.cpu_s += time.thread_time() - t0

"""Metal activity policy (``EXO_TBCCL_ACTIVITY_MODE=step``; Phase 63 experiment, Phase 64 policy): same-process CPU activity during the pipeline's own local work.

Phase 62/63 measured on the real Linux<->Mac link (orientation A) that TBCCL's blocking Mac pipeline thread is scheduled on slow (efficiency) cores while MlxRing's
busy-polling worker keeps the Mac fast; a helper thread that burns CPU while the pipeline does its own work restored the speed (first-use 1.1 -> 0.13 ms, stage
4.3 -> 2.2 ms, TPOT 12.0 -> 6.6 ms). See docs/mac_activity_policy.md. The window is defined only by communicator events (no model, rank, split or orientation input):

    opens   when a ``recv`` completes (the receiving rank's local compute follows), or, in a pipeline whose previous step had no receive, when an ``all_gather`` completes
    closes  when the next ``all_gather`` is submitted, or at barrier / any_true / abort / close

Windows only open after the first ``all_gather`` has completed (the decode loop), so prefill is untouched. A hard bound (``EXO_TBCCL_ACTIVITY_MAX_MS``, default 100)
ends any single window by itself (a lost event, an exception or an idle gap cannot leave a core burning). ``EXO_TBCCL_ACTIVITY_DUTY`` (0 < d <= 1) burns for d of every
1 ms period inside an open window. The helper never touches tensors, Work or communicator state: it calls libc ``memset`` on a private buffer (the GIL is released for
every call). Idle, it blocks on an event (zero CPU). It starts lazily at the first window and is joined by ``close()``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
import time

_BUF_BYTES = 1 << 18
_PERIOD_S = 0.001


class StepActivity:
    """The helper thread and its per-window accounting."""

    def __init__(self, max_window_ms: float = 100.0, duty: float = 1.0):
        self._max_s = max_window_ms / 1000.0
        self._duty = min(1.0, max(0.01, duty))
        self._go = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._deadline = 0.0
        self._opened_at = 0.0
        self.windows = 0
        self.timed_out = 0
        self.active_s = 0.0  # wall time windows were open (until closed or timed out)
        self.cpu_s = 0.0  # the helper's own CPU, accumulated when it parks

    @property
    def is_open(self) -> bool:
        return self._go.is_set()

    def open(self) -> None:
        if self._stop.is_set() or self._go.is_set():
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="exo-tbccl-activity", daemon=True)
                self._thread.start()
            now = time.perf_counter()
            self._opened_at = now
            self._deadline = now + self._max_s
            self.windows += 1
            self._go.set()

    def close_window(self) -> None:
        if self._go.is_set():
            self._go.clear()
            self.active_s += min(time.perf_counter(), self._deadline) - self._opened_at

    def shutdown(self) -> None:
        self.close_window()
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
        duty = self._duty
        while not self._stop.is_set():
            self._go.wait()
            if self._stop.is_set():
                return
            t0 = time.thread_time()
            while self._go.is_set() and not self._stop.is_set():
                now = time.perf_counter()
                if now >= self._deadline:
                    self.timed_out += 1
                    self._go.clear()
                    self.active_s += self._deadline - self._opened_at
                    break
                if duty >= 1.0:
                    memset(buf, 0, _BUF_BYTES)  # one ~15 us call with the GIL released
                else:
                    until = now + duty * _PERIOD_S
                    while time.perf_counter() < until and self._go.is_set():
                        memset(buf, 0, _BUF_BYTES)
                    time.sleep(max(0.0, (1.0 - duty) * _PERIOD_S))
            self.cpu_s += time.thread_time() - t0


class MetalActivityPolicy:
    """The window state machine driven by communicator events (see the module docstring)."""

    def __init__(self, max_window_ms: float, duty: float):
        self.activity = StepActivity(max_window_ms, duty)
        self.duty = duty
        self._armed = False  # at least one all_gather completed: the decode loop has started
        self._recv_since_gather = False
        self._step_had_recv = False  # what the previous step looked like (a receiving pipeline stage opens its window at the receive)

    def on_recv_complete(self) -> None:
        self._recv_since_gather = True
        if self._armed:
            self.activity.open()

    def on_gather_submit(self) -> None:
        self.activity.close_window()
        self._step_had_recv = self._recv_since_gather
        self._recv_since_gather = False

    def on_gather_complete(self) -> None:
        self._armed = True
        if not self._step_had_recv:
            self.activity.open()

    def on_boundary(self) -> None:
        self.activity.close_window()

    def shutdown(self) -> None:
        self.activity.shutdown()

    def stats(self) -> dict:
        a = self.activity
        return {"activity_windows": a.windows, "activity_us": a.active_s * 1e6, "helper_cpu_us": a.cpu_s * 1e6, "duty": self.duty, "fallback_timeouts": a.timed_out}

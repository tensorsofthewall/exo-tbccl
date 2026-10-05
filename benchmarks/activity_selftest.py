"""Phase 63: lifecycle / CPU-cost self test of benchmarks/activity_thread.py (run on the Mac; no model, no communication).

    python benchmarks/activity_selftest.py

1. functional: a window opened for ~100 ms makes the process ~1 core busy; closed, it returns to idle (CPU measured with resource.getrusage).
2. lifecycle: 100 create/open/close cycles leave no thread behind.
3. CPU cost: continuous (decode window), and step-like windows of 5 ms and 10 ms in an 11.5 ms step: wall, helper CPU (the helper's own thread_time), process CPU, active time.
4. stop bound: close() while the window is open returns within the join timeout.
"""
import resource
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from activity_thread import Activity  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def sleep_until(t_ns):
    while True:
        rem = t_ns - time.perf_counter_ns()
        if rem <= 0:
            return
        time.sleep(min(rem / 1e9, 0.0005) if rem > 700_000 else 0)


def main():
    a = Activity("win:step")
    c0, t0 = cpu(), time.perf_counter()
    time.sleep(0.2)
    idle = (cpu() - c0) / (time.perf_counter() - t0)
    a.on_event("end", "comm", "all_gather")
    c0, t0 = cpu(), time.perf_counter()
    time.sleep(0.2)
    busy = (cpu() - c0) / (time.perf_counter() - t0)
    a.on_event("begin", "comm", "all_gather")
    c0, t0 = cpu(), time.perf_counter()
    time.sleep(0.2)
    back = (cpu() - c0) / (time.perf_counter() - t0)
    a.close()
    print(f"functional: idle {idle:.3f} cores, window open {busy:.3f}, closed again {back:.3f}")

    n0 = threading.active_count()
    for _ in range(100):
        b = Activity("win:step")
        b.on_event("end", "comm", "all_gather")
        time.sleep(0.002)
        b.on_event("begin", "comm", "all_gather")
        b.close()
        assert not b.thread.is_alive()
    print(f"lifecycle: 100 cycles, threads before {n0} after {threading.active_count()}")

    a = Activity("win:decode")
    a.on_event("begin", "comm", "recv_like")
    t0, c0 = time.perf_counter(), cpu()
    time.sleep(1.0)
    t1 = time.perf_counter()
    a.on_event("begin", "comm", "barrier")
    time.sleep(0.05)
    print(f"continuous (decode window): wall {t1 - t0:.3f}s process CPU {cpu() - c0:.3f}s -> {(cpu() - c0) / (t1 - t0):.2f} cores; helper CPU {sum(b[2] for b in a.bursts) / 1e9:.3f}s over {len(a.bursts)} burst(s)")
    a.close()

    for active_ms in (5, 10):
        a = Activity("win:step")
        steps, period = 200, 11_500_000
        c0, t0 = cpu(), time.perf_counter()
        nxt = time.perf_counter_ns()
        for _ in range(steps):
            a.on_event("end", "comm", "all_gather")
            sleep_until(time.perf_counter_ns() + active_ms * 1_000_000)
            a.on_event("begin", "comm", "all_gather")
            nxt += period
            sleep_until(nxt)
        t1, c1 = time.perf_counter(), cpu()
        a.close()
        helper = sum(b[2] for b in a.bursts) / 1e9
        act = sum(b[1] - b[0] for b in a.bursts) / 1e9
        print(f"step-like {active_ms} ms of {period / 1e6:.1f} ms: wall {t1 - t0:.3f}s process CPU {c1 - c0:.3f}s ({(c1 - c0) / (t1 - t0):.2f} cores), helper CPU {helper:.3f}s "
              f"({helper / (t1 - t0):.2f} cores, {1e3 * helper / steps:.2f} ms/step), active {1e3 * act / steps:.2f} ms/step, bursts {len(a.bursts)}")

    a = Activity("win:step")
    a.on_event("end", "comm", "all_gather")
    time.sleep(0.05)
    t0 = time.perf_counter()
    a.close()
    print(f"close while active: {1e3 * (time.perf_counter() - t0):.2f} ms, alive={a.thread.is_alive()}")


if __name__ == "__main__":
    main()

"""How fast does plain host (Python) code run right after the thread was blocked, versus after other ways of spending the same gap?

Physical Linux<->Mac traces showed the Mac's host-side work between communication calls (return from a call, graph building, command encoding) 4-15x slower
under TbcclPipelineComm (blocking native wait) than under MlxRing (whose non-blocking socket worker busy-polls on another thread while a transfer is pending).
This isolates the mechanism with no communication library: a CPU-bound Python loop (~200 us hot) timed right after each wait mode:
  none        no wait
  block       thread blocks in recv() on a socketpair a helper thread feeds after the gap (what a native Work wait does)
  block+spinner  the same, plus a helper thread busy-polling a non-blocking socket during the gap (what MlxRing's worker does)
  spin        the thread itself busy-waits the gap
  block+qos   block, with the thread's QoS raised to user-interactive first (macOS only)
    python benchmarks/cpu_after_wait.py [--gaps-ms 1,3,5] [--iters 400]
"""
import argparse
import ctypes
import platform
import socket
import statistics
import threading
import time


def work():
    t0 = time.perf_counter_ns()
    s = 0
    for i in range(30000):
        s += i * i
    return (time.perf_counter_ns() - t0) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps-ms", default="1,3,5")
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    s1, s2 = socket.socketpair()
    spin_sock, spin_peer = socket.socketpair()
    spin_sock.setblocking(False)
    ev, stop, spin_on = threading.Event(), threading.Event(), threading.Event()
    gap = {"s": 0.003}

    def feeder():
        while not stop.is_set():
            ev.wait(); ev.clear()
            if stop.is_set():
                return
            time.sleep(gap["s"]); s2.send(b"x")

    def spinner():
        while not stop.is_set():
            if not spin_on.is_set():
                spin_on.wait(0.05); continue
            try:
                spin_sock.recv(1)
            except BlockingIOError:
                pass

    threading.Thread(target=feeder, daemon=True).start()
    threading.Thread(target=spinner, daemon=True).start()
    qos = None
    if platform.system() == "Darwin":
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        qos = lambda cls: lib.pthread_set_qos_class_self_np(cls, 0)
    for g in [float(x) for x in a.gaps_ms.split(",")]:
        gap["s"] = g / 1000
        res = {m: [] for m in ("none", "block", "block+spinner", "spin", "block+qos")}
        for it in range(a.iters + 20):
            for mode in res:
                if mode == "block+qos" and qos is None:
                    continue
                if mode == "none":
                    pass
                elif mode in ("block", "block+spinner", "block+qos"):
                    if mode == "block+spinner":
                        spin_on.set()
                    if mode == "block+qos":
                        qos(0x21)  # QOS_CLASS_USER_INTERACTIVE
                    ev.set(); s1.recv(1)
                    spin_on.clear()
                    if mode == "block+qos":
                        qos(0x15)  # back to QOS_CLASS_USER_INITIATED
                else:
                    end = time.perf_counter_ns() + int(g * 1e6)
                    while time.perf_counter_ns() < end:
                        pass
                d = work()
                if it >= 20:
                    res[mode].append(d)
        print(f"{platform.system()} gap {g:3.1f} ms: " + "  ".join(f"{m} {statistics.median(v):6.0f}us" for m, v in res.items() if v))
    stop.set(); ev.set()


if __name__ == "__main__":
    main()

"""The cross-host timeline work measurement-only cross-host clock alignment (NTP-style ping-pong over one plain TCP connection).

The recorder timestamps with time.perf_counter_ns() on each host; those clocks are unrelated. Rank 0 plays the requester (L0 send, L3 receive), rank 1 the
responder (M1 receive, M2 send, taken immediately around the echo). For one exchange, assuming a symmetric path,
    offset = ((M1 + M2) - (L0 + L3)) / 2        # rank-1 clock minus rank-0 clock
    rtt    = (L3 - L0) - (M2 - M1)
`run()` takes N exchanges, keeps the lowest-RTT fraction and reports the median offset, its dispersion and an uncertainty (the larger of the dispersion
and half the minimum RTT, which bounds an asymmetric path). It uses its OWN socket (host/port given by the driver), never the model communication, the
bootstrap exchange or exo's control channel, so it cannot change runner state. Call it twice (before and after the traced region) to estimate drift.

    result = {"offset_ns", "rtt_min_ns", "rtt_median_ns", "dispersion_ns", "uncertainty_ns", "n", "n_used", "t_mid_ns"}   (t_mid_ns: rank-0 clock midpoint)
"""
import json
import socket
import statistics
import struct
import time

N_DEFAULT = 200
KEEP = 0.2


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("clock sync peer closed")
        buf += chunk
    return buf


def _estimate(l0, l3, m1, m2):
    pairs = []
    for a, d, b, c in zip(l0, l3, m1, m2):
        rtt = (d - a) - (c - b)
        off = ((b + c) - (a + d)) / 2
        pairs.append((rtt, off, (a + d) // 2))
    pairs.sort()
    used = pairs[: max(5, int(len(pairs) * KEEP))]
    offs = [o for _, o, _ in used]
    disp = statistics.pstdev(offs) if len(offs) > 1 else 0.0
    rtt_min = pairs[0][0]
    return {
        "offset_ns": statistics.median(offs),
        "rtt_min_ns": rtt_min,
        "rtt_median_ns": statistics.median(p[0] for p in pairs),
        "dispersion_ns": disp,
        "uncertainty_ns": max(disp, rtt_min / 2),
        "n": len(pairs),
        "n_used": len(used),
        "t_mid_ns": statistics.median(t for _, _, t in used),
    }


def serve(rank1_sock, n):
    """Rank 1: echo n requests, recording (M1, M2); then send the list to rank 0."""
    m1, m2 = [], []
    for _ in range(n):
        req = _recvn(rank1_sock, 8)
        a = time.perf_counter_ns()
        rank1_sock.sendall(req)
        b = time.perf_counter_ns()
        m1.append(a)
        m2.append(b)
    rank1_sock.sendall(struct.pack("!I", len(m1)) + b"".join(struct.pack("!qq", a, b) for a, b in zip(m1, m2)))
    return m1, m2


def request(rank0_sock, n):
    """Rank 0: n requests, recording (L0, L3); then read rank 1's (M1, M2) and estimate."""
    l0, l3 = [], []
    for i in range(n):
        a = time.perf_counter_ns()
        rank0_sock.sendall(struct.pack("!q", i))
        _recvn(rank0_sock, 8)
        d = time.perf_counter_ns()
        l0.append(a)
        l3.append(d)
    k = struct.unpack("!I", _recvn(rank0_sock, 4))[0]
    raw = _recvn(rank0_sock, 16 * k)
    m = [struct.unpack("!qq", raw[16 * i: 16 * i + 16]) for i in range(k)]
    return _estimate(l0, l3, [x[0] for x in m], [x[1] for x in m])


def connect(rank, host, peer, port, timeout_s=120):
    """Rank 0 listens on (host, port); rank 1 connects to (peer, port). Returns a TCP_NODELAY socket."""
    if rank == 0:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        srv.settimeout(timeout_s)
        sock, _ = srv.accept()
        srv.close()
    else:
        deadline = time.time() + timeout_s
        while True:
            try:
                sock = socket.create_connection((peer, port), timeout=5)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(timeout_s)
    return sock


class ClockSync:
    """Holds the sync socket for a run: `measure(tag)` is called by BOTH ranks at the same program points (before and after the traced region)."""

    def __init__(self, rank, host, peer, port, n=N_DEFAULT):
        self.rank, self.n = rank, n
        self.sock = connect(rank, host, peer, port)
        self.results: dict[str, dict] = {}

    def measure(self, tag):
        if self.rank == 0:
            self.results[tag] = request(self.sock, self.n)
        else:
            serve(self.sock, self.n)
            self.results[tag] = {}
        return self.results[tag]

    def close(self):
        self.sock.close()


def alignment(clock: dict) -> dict:
    """Linear offset(t) (rank-1 clock minus rank-0 clock as a function of the rank-0 clock) from the pre/post measurements in rank 0's `clock` dict."""
    pre, post = clock.get("pre"), clock.get("post")
    if not pre:
        return {"offset_ns": 0.0, "drift_ns_per_s": 0.0, "uncertainty_ns": float("inf"), "t0_ns": 0}
    if not post:
        return {"offset_ns": pre["offset_ns"], "drift_ns_per_s": 0.0, "uncertainty_ns": pre["uncertainty_ns"], "t0_ns": pre["t_mid_ns"]}
    dt = (post["t_mid_ns"] - pre["t_mid_ns"]) / 1e9
    drift = (post["offset_ns"] - pre["offset_ns"]) / dt if dt > 0 else 0.0
    return {"offset_ns": pre["offset_ns"], "drift_ns_per_s": drift, "uncertainty_ns": max(pre["uncertainty_ns"], post["uncertainty_ns"]), "t0_ns": pre["t_mid_ns"],
            "pre_post_shift_ns": post["offset_ns"] - pre["offset_ns"], "interval_s": dt}


def offset_at(al: dict, t_rank0_ns: float) -> float:
    return al["offset_ns"] + al["drift_ns_per_s"] * (t_rank0_ns - al["t0_ns"]) / 1e9


if __name__ == "__main__":  # standalone check: python benchmarks/clock_sync.py --rank R --host H --peer P --port N
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--port", type=int, default=29600)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    a = ap.parse_args()
    cs = ClockSync(a.rank, a.host, a.peer, a.port, a.n)
    print(json.dumps({"pre": cs.measure("pre")}))
    time.sleep(2)
    print(json.dumps({"post": cs.measure("post")}))

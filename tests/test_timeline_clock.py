"""Phase 63: regression for the cross-host clock alignment (benchmarks/distributed_timeline.py).

Phase 62 found that to_rank0 applied the drift to (t1 - t0) with t1 in rank 1's clock and t0 in rank 0's, an error of drift x |offset| that only became visible when
the hosts' monotonic clocks differed by thousands of seconds (a rebooted Linux host).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import distributed_timeline as dt  # noqa: E402


def clock(offset_ns, drift_ns_per_s, t_mid_ns, dt_s):
    """pre/post sync records as clock_sync produces them: offset = rank1 - rank0 (ns), t_mid in rank 0's clock."""
    return {"pre": {"offset_ns": offset_ns, "t_mid_ns": t_mid_ns, "uncertainty_ns": 40000.0},
            "post": {"offset_ns": offset_ns + drift_ns_per_s * dt_s, "t_mid_ns": t_mid_ns + dt_s * 1e9, "uncertainty_ns": 40000.0}}


def check(offset_ns):
    drift, t_mid = -3800.0, 6.6e12
    al = dt.align(clock(offset_ns, drift, t_mid, 3.0))
    for t0 in (t_mid + 0.1e9, t_mid + 1.5e9, t_mid + 2.9e9):  # a rank-0 time, and the same instant on rank 1's clock
        elapsed_s = (t0 - t_mid) / 1e9
        t1 = t0 + offset_ns + drift * elapsed_s
        assert abs(dt.to_rank0(al, t1) - t0) < 50  # ns


def test_small_offset():
    check(32.5e9)


def test_large_offset_from_a_rebooted_host():
    check(8.0e12)


def test_negative_large_offset():
    check(-8.0e12)

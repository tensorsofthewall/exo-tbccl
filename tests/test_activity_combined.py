"""The combined configuration (EXO_TBCCL_ACTIVITY_MODE=step + EXO_TBCCL_WAIT_SPIN_MS=8) on real communicators.

Both mechanisms are Metal-only: on a host without Metal (Linux/CUDA, CPU) the helper must never start and no wait may spin, whatever the variables say.
"""

import os
import threading
import time

import pytest

from tests.harness import run_world
from tests.test_group import ADV

mx = pytest.importorskip("mlx.core")
METAL = bool(mx.metal.is_available())


def _create(rank, world, ex, spin="8"):
    os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "step"
    os.environ["EXO_TBCCL_ACTIVITY_DUTY"] = "1.0"
    os.environ["EXO_TBCCL_WAIT_SPIN_MS"] = spin
    from exo_tbccl.group import TbcclPipelineComm

    return TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)


def _gather(comm):
    import mlx.core as mx_
    import numpy as np

    g = comm.all_gather(mx_.array(np.ones((1, 64), dtype=np.float32)))
    mx_.eval(g)
    return g


def _w_decode_like(rank, world, ex):
    """Rank 0: send, all_gather (never receives); rank 1: recv, all_gather. Several steps: both mechanisms engage on Metal, neither anywhere else."""
    import mlx.core as mx_
    import numpy as np

    comm = _create(rank, world, ex)
    x = mx_.array(np.ones((1, 256), dtype=np.float32))
    for _ in range(30):
        if rank == 0:
            comm.send(x, 1)
        else:
            mx_.eval(comm.recv_like(x, 0))
        _gather(comm)
    a, w = comm.activity_stats(), comm.wait_stats
    out = (a["activity_windows"], a["fallback_timeouts"], w.waits_spun, w.total_spin_us)
    ex("quiesce", b"")
    act = comm._act
    comm.close()
    return out, (act.activity.thread_alive if act else False)


def test_combined_stats_are_active_on_metal_and_zero_elsewhere():
    res = run_world(2, _w_decode_like)
    for (windows, timeouts, spun, spin_us), alive in res:
        assert not alive
        if METAL:
            assert windows > 0 and spun > 0 and spin_us > 0, res
        else:
            assert windows == 0 and spun == 0 and spin_us == 0, res


def _w_peer_dies_in_spin_and_window(rank, world, ex):
    import numpy as np

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, spin="1000")
    _gather(comm)
    if rank == 1:
        time.sleep(0.4)
        os._exit(0)
    t0, name = time.time(), "no error"
    try:
        comm.wait(comm.recv_into_async(np.zeros(1024, dtype=np.uint8), 1))  # window open (rank 0 never receives), the caller polls
    except TbcclError as e:
        name = type(e).__name__
    act = comm._act
    comm.close()
    return name, time.time() - t0, (act.activity.thread_alive if act else False)


def test_peer_death_with_window_open_and_caller_polling_is_bounded():
    name, dt, alive = run_world(2, _w_peer_dies_in_spin_and_window, timeout=120, tolerate_exit=(1,))[0]
    assert name != "no error" and dt < 30 and not alive


def _w_abort_in_spin_and_window(rank, world, ex):
    import numpy as np

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, spin="1000")
    _gather(comm)
    if rank == 1:
        time.sleep(1.5)
        comm.close()
        return None
    threading.Timer(0.2, lambda: comm.abort("test abort")).start()
    t0, name = time.time(), "no error"
    try:
        comm.wait(comm.recv_into_async(np.zeros(1024, dtype=np.uint8), 1))
    except TbcclError as e:
        name = type(e).__name__
    act = comm._act
    comm.close()
    return name, time.time() - t0, (act.activity.thread_alive if act else False)


def test_abort_ends_the_polling_wait_and_the_window():
    name, dt, alive = run_world(2, _w_abort_in_spin_and_window, timeout=120)[0]
    assert name != "no error" and dt < 10 and not alive


def _w_cancellation_round(rank, world, ex):
    comm = _create(rank, world, ex)
    _gather(comm)
    agreed = comm.any_true(rank == 0)
    still = bool(comm._act and comm._act.activity.is_open)
    ex("quiesce", b"")
    comm.close()
    return agreed, still


def test_cancellation_agreement_round_closes_the_window_in_the_combined_config():
    for agreed, still in run_world(2, _w_cancellation_round):
        assert agreed and not still


def _w_long_wait(rank, world, ex):
    comm = _create(rank, world, ex)
    _gather(comm)
    if rank == 1:
        time.sleep(3.0)  # the peer is late: far beyond the 8 ms spin budget and the window watchdog
    import resource

    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    _gather(comm)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    dt = time.perf_counter() - t0
    cores = (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / dt
    w = comm.wait_stats
    comm.close()
    return cores, dt, w.max_spin_us, w.waits_fell_back


def test_a_long_collective_wait_costs_a_bounded_burst_then_blocks():
    res = run_world(2, _w_long_wait, timeout=120)
    cores0, dt0, max_spin, fell = res[0]
    assert dt0 > 2.5
    if METAL:
        assert max_spin < 20000, res  # the caller's poll never exceeds the 8 ms budget (plus a scheduling quantum)
        assert cores0 < 0.2, res  # 8 ms of polling over a 3 s wait, nothing else running (the window closed at submit)
    else:
        assert max_spin == 0 and fell == 0, res


def _w_hundred_cycles(rank, world, ex, cycles):
    import gc

    import psutil

    proc = psutil.Process()
    c = _create(rank, world, ex)
    _gather(c)
    ex("quiesce", b"")
    c.close()
    gc.collect()
    base_fds, base_py, base_os = proc.num_fds(), threading.active_count(), proc.num_threads()
    for _ in range(cycles):
        c = _create(rank, world, ex)
        _gather(c)
        ex("quiesce", b"")
        c.close()
        assert c._act is None or not c._act.activity.thread_alive
    gc.collect()
    time.sleep(0.5)
    return proc.num_fds() - base_fds, threading.active_count() - base_py, proc.num_threads() - base_os


def test_hundred_create_destroy_cycles_in_the_combined_config_leak_nothing():
    for fds, py_threads, os_threads in run_world(2, _w_hundred_cycles, 100, timeout=900):
        assert fds == 0 and py_threads == 0 and os_threads <= 0


def _w_idle(rank, world, ex):
    comm = _create(rank, world, ex)
    _gather(comm)
    time.sleep(0.3)
    import resource

    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    time.sleep(20.0)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cores = (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / (time.perf_counter() - t0)
    ex("quiesce", b"")
    comm.close()
    return cores


def test_idle_communicator_in_the_combined_config_uses_no_cpu_beyond_tbccl_itself():
    res = run_world(2, _w_idle, timeout=120)
    ref = run_world(2, _w_idle_off, timeout=120)
    for on, off in zip(res, ref):
        assert on < off + 0.05, (res, ref)


def _w_idle_off(rank, world, ex):
    os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "off"
    os.environ["EXO_TBCCL_WAIT_SPIN_MS"] = "0"
    from exo_tbccl.group import TbcclPipelineComm

    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)
    _gather(comm)
    time.sleep(0.3)
    import resource

    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    time.sleep(20.0)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cores = (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / (time.perf_counter() - t0)
    ex("quiesce", b"")
    comm.close()
    return cores


def test_lost_wakeup_regression_150_adversarial_shutdowns():
    """The repeated physical-validation work's StepActivity close_window/shutdown race: shutdown must never be undone by a concurrent close_window, so the helper always exits and joins."""
    from exo_tbccl.step_activity import StepActivity

    base = threading.active_count()
    for i in range(150):
        a = StepActivity(max_window_ms=50, duty=1.0)
        stop = threading.Event()

        def hammer(f):
            while not stop.is_set():
                f()
                time.sleep(0.0002)  # four GIL-hot Python loops with no pause starve the helper's own GIL re-acquisition for seconds; the transitions still race at ~5 kHz

        ts = [threading.Thread(target=hammer, args=(f,)) for f in (a.open, a.close_window, a.open, a.close_window)]
        for t in ts:
            t.start()
        time.sleep(0.001 * (i % 5))
        a.shutdown()
        stop.set()
        for t in ts:
            t.join()
        assert not a.thread_alive, i
    assert threading.active_count() == base


def _w_exact(rank, world, ex, combined):
    import mlx.core as mx_
    import numpy as np

    from exo_tbccl.group import TbcclPipelineComm

    os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "step" if combined else "off"
    os.environ["EXO_TBCCL_WAIT_SPIN_MS"] = "8" if combined else "0"
    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=30000)
    out = {}
    for name, dt in (("float32", mx_.float32), ("bfloat16", mx_.bfloat16)):
        for i in range(4):
            vals = (np.arange(256, dtype=np.float32) * 0.37 + rank * 11.0 + i) % 97.0
            x = mx_.array(vals).astype(dt).reshape(1, 256)
            g = comm.all_gather(x)
            mx_.eval(g)
            out[f"{name}_gather_{i}"] = np.array(g.astype(mx_.float32)).tobytes()
            nxt, prv = (rank + 1) % world, (rank - 1) % world
            ts = [comm.send_async(x, nxt)]
            got = comm.recv_like(x, prv)
            mx_.eval(got)
            comm.wait_all(ts)
            out[f"{name}_ring_{i}"] = np.array(got.astype(mx_.float32)).tobytes()
            time.sleep(0.002)
    comm.close()
    return out


@pytest.mark.parametrize("world", [2, 3, 4])
def test_collectives_are_bit_exact_with_both_mechanisms_on(world):
    off = run_world(world, _w_exact, False, timeout=300)
    on = run_world(world, _w_exact, True, timeout=300)
    for rank in range(world):
        assert off[rank].keys() == on[rank].keys()
        for k, v in on[rank].items():
            assert v == off[rank][k], (world, rank, k)

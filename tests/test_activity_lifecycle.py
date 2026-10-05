"""Phase 65: lifecycle, failure and threading validation of the Metal activity policy (EXO_TBCCL_ACTIVITY_MODE=step) on real communicators.

Every test runs with the policy requested; on a host without Metal (Linux/CUDA, CPU) it must be inert (no helper thread, zero counters) and the same assertions on leaks hold.
"""

import threading
import time

import pytest

from tests.harness import run_world
from tests.test_group import ADV

mx = pytest.importorskip("mlx.core")
METAL = bool(mx.metal.is_available())


def _create(rank, world, ex, duty="1.0"):
    import os

    os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "step"
    os.environ["EXO_TBCCL_ACTIVITY_DUTY"] = duty
    from exo_tbccl.group import TbcclPipelineComm

    return TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)


def _gather(comm):
    import mlx.core as mx_
    import numpy as np

    g = comm.all_gather(mx_.array(np.ones((1, 64), dtype=np.float32)))
    mx_.eval(g)
    return g


def _w_hundred_cycles(rank, world, ex, cycles):
    import gc

    import psutil

    proc = psutil.Process()
    c = _create(rank, world, ex)
    _gather(c)
    ex("quiesce", b"")
    c.close()
    gc.collect()
    base_fds, base_py, base_os, base_rss = proc.num_fds(), threading.active_count(), proc.num_threads(), proc.memory_info().rss
    helper_seen = 0
    for _ in range(cycles):
        c = _create(rank, world, ex)
        _gather(c)  # arms the policy and (for a stage that never receives) opens a window
        helper_seen += c._act is not None and c._act.activity.thread_alive
        ex("quiesce", b"")
        c.close()  # closed while a window is open
        assert c._act is None or not c._act.activity.thread_alive
    gc.collect()
    time.sleep(0.5)
    return proc.num_fds() - base_fds, threading.active_count() - base_py, proc.num_threads() - base_os, (proc.memory_info().rss - base_rss) / 1e6, helper_seen


def test_hundred_create_destroy_cycles_with_activity_leak_nothing():
    res = run_world(2, _w_hundred_cycles, 100, timeout=900)
    for fds, py_threads, os_threads, rss_mb, helper_seen in res:
        assert fds == 0 and py_threads == 0 and os_threads <= 0, res
        assert rss_mb < 60, res  # no material growth across 100 lifetimes
        assert helper_seen == (100 if METAL else 0), res  # the helper ran on every cycle on Metal, never elsewhere


def _w_peer_dies(rank, world, ex, point):
    import os as _os

    import numpy as np

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex)
    _gather(comm)  # window open on rank 0 (it never receives)
    if rank == 1:
        if point == "before_recv":
            _os._exit(0)
        if point == "during_local_step":
            time.sleep(0.3)
            _os._exit(0)
        time.sleep(0.3)  # during_allgather: rank 1 never joins the next AllGather
        _os._exit(0)
    t0, name = time.time(), "no error"
    try:
        if point == "before_recv":
            comm.wait(comm.recv_into_async(np.zeros(1024, dtype=np.uint8), 1))
        elif point == "during_local_step":
            time.sleep(0.3)  # local compute with the window open while the peer dies
            _gather(comm)
        else:
            _gather(comm)
    except TbcclError as e:
        name = type(e).__name__
    act = comm._act
    comm.close()
    return name, time.time() - t0, (act.activity.thread_alive if act else False), threading.active_count()


@pytest.mark.parametrize("point", ["before_recv", "during_local_step", "during_allgather"])
def test_peer_death_with_activity_is_bounded_and_leaves_no_helper(point):
    res = run_world(2, _w_peer_dies, point, timeout=120, tolerate_exit=(1,))
    name, dt, alive, _ = res[0]
    assert name != "no error" and dt < 60 and not alive, res


def _w_abort_while_active(rank, world, ex):
    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex)
    _gather(comm)
    if rank == 1:
        time.sleep(1.5)
        comm.close()
        return None
    was_open = bool(comm._act and comm._act.activity.is_open)
    threading.Timer(0.2, lambda: comm.abort("test abort")).start()
    time.sleep(0.4)  # the window is open (or was ended by the watchdog) when the abort lands
    try:
        _gather(comm)
        name = "no error"
    except TbcclError as e:
        name = type(e).__name__
    still_open = bool(comm._act and comm._act.activity.is_open)
    act = comm._act
    comm.close()
    return name, was_open, still_open, (act.activity.thread_alive if act else False)


def test_abort_from_another_thread_while_the_window_is_active():
    res = run_world(2, _w_abort_while_active, timeout=120)
    name, was_open, still_open, alive = res[0]
    assert name != "no error" and not still_open and not alive, res
    assert was_open == METAL, res


def _w_any_true_closes_window(rank, world, ex):
    comm = _create(rank, world, ex)
    _gather(comm)
    opened = bool(comm._act and comm._act.activity.is_open)
    agreed = comm.any_true(rank == 0)  # exo's agreement/cancellation round: a boundary, closes the window
    closed = not (comm._act and comm._act.activity.is_open)
    ex("quiesce", b"")
    comm.close()
    return opened, closed, agreed


def test_agreement_round_is_a_boundary_that_closes_the_window():
    res = run_world(2, _w_any_true_closes_window)
    for opened, closed, agreed in res:
        assert closed and agreed
        assert opened == METAL, res


def _idle_cores(seconds):
    import resource

    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    time.sleep(seconds)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / (time.perf_counter() - t0)


def _w_idle_thirty_seconds(rank, world, ex):
    import os

    comm = _create(rank, world, ex)
    _gather(comm)
    time.sleep(0.3)  # let the watchdog end the first window
    h0 = comm.activity_stats()["helper_cpu_us"]
    on = _idle_cores(30.0)
    stats = comm.activity_stats()
    ex("quiesce", b"")
    comm.close()
    os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "off"  # the same communicator without the policy: TBCCL's own idle CPU is the reference (it is not zero on every host)
    from exo_tbccl.group import TbcclPipelineComm

    c2 = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)
    _gather(c2)
    time.sleep(0.3)
    off = _idle_cores(5.0)
    ex("quiesce", b"")
    c2.close()
    return on, off, (stats["helper_cpu_us"] - h0) / 1e3, stats["activity_windows"]


def test_thirty_seconds_idle_costs_no_extra_cpu():
    res = run_world(2, _w_idle_thirty_seconds, timeout=300)
    for on, off, helper_ms, windows in res:
        assert on < off + 0.05, res  # the policy adds no idle CPU
        assert helper_ms < 5.0, res  # the helper's own CPU over 30 s of idleness
        assert windows == (1 if METAL else 0), res


def test_enable_disable_abort_stress_leaves_threads_at_baseline():
    from exo_tbccl.step_activity import MetalActivityPolicy

    base = threading.active_count()
    for i in range(40):
        p = MetalActivityPolicy(max_window_ms=20, duty=1.0 if i % 2 else 0.5)
        stop = threading.Event()
        errs = []

        def hammer(fn):
            try:
                while not stop.is_set():
                    fn()
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        ts = [threading.Thread(target=hammer, args=(f,)) for f in (p.on_gather_complete, p.on_gather_submit, p.on_recv_begin, p.on_boundary)]
        for t in ts:
            t.start()
        time.sleep(0.02)
        p.shutdown()  # "disable/abort" while the transitions race
        time.sleep(0.005)
        stop.set()
        for t in ts:
            t.join()
        assert not errs and not p.activity.thread_alive
    assert threading.active_count() == base


def _w_exact(rank, world, ex, mode):
    import os as _os

    import mlx.core as mx_
    import numpy as np

    _os.environ["EXO_TBCCL_ACTIVITY_MODE"] = mode
    from exo_tbccl.group import TbcclPipelineComm

    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=30000)
    out = {}
    for name, dt in (("float32", mx_.float32), ("bfloat16", mx_.bfloat16)):
        for i in range(4):  # several steps so windows open and close between the collectives
            vals = (np.arange(256, dtype=np.float32) * 0.37 + rank * 11.0 + i) % 97.0
            x = mx_.array(vals).astype(dt).reshape(1, 256)
            g = comm.all_gather(x)
            mx_.eval(g)
            out[f"{name}_gather_{i}"] = np.array(g.astype(mx_.float32)).tobytes()
            if world > 1:  # a point-to-point ring pass as well (rank r -> r+1), bit-exact
                nxt, prv = (rank + 1) % world, (rank - 1) % world
                ts = [comm.send_async(x, nxt)]
                got = comm.recv_like(x, prv)
                mx_.eval(got)
                comm.wait_all(ts)
                out[f"{name}_ring_{i}"] = np.array(got.astype(mx_.float32)).tobytes()
            time.sleep(0.002)
    act = comm._act
    comm.close()
    return out, (act.activity.windows if act else 0)


@pytest.mark.parametrize("world", [2, 3, 4])
def test_collectives_are_bit_exact_and_identical_with_activity_on_and_off(world):
    import numpy as np

    off = run_world(world, _w_exact, "off", timeout=300)
    on = run_world(world, _w_exact, "step", timeout=300)
    for rank in range(world):
        assert off[rank][0].keys() == on[rank][0].keys()
        for k, v in on[rank][0].items():
            assert v == off[rank][0][k], (world, rank, k)  # activity never changes a single byte
            if "_gather_" in k:
                i = int(k.rsplit("_", 1)[1])
                dt = np.float32
                expect = np.concatenate([((np.arange(256, dtype=np.float32) * 0.37 + r * 11.0 + i) % 97.0).astype(dt).reshape(1, 256) for r in range(world)])
                got = np.frombuffer(v, dtype=np.float32).reshape(world, 256)
                if k.startswith("float32"):
                    assert np.array_equal(got, expect), (world, rank, k)
        assert (on[rank][1] > 0) if METAL else (on[rank][1] == 0)  # windows opened on Metal, none elsewhere

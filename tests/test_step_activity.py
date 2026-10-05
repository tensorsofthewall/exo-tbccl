"""EXO_TBCCL_STEP_ACTIVITY (the step-activity work opt-in experiment): helper lifecycle, bounds, inertness off Metal, exactness, failure paths."""

import os
import resource
import threading
import time

import pytest

from exo_tbccl.config import FastPathConfig
from exo_tbccl.step_activity import MetalActivityPolicy, StepActivity
from tests.harness import run_world
from tests.test_group import ADV

mx = pytest.importorskip("mlx.core")
METAL = bool(mx.metal.is_available())


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _cores(seconds):
    c0, t0 = _cpu(), time.perf_counter()
    time.sleep(seconds)
    return (_cpu() - c0) / (time.perf_counter() - t0)


def test_default_off_and_env_parse(monkeypatch):
    for k in ("EXO_TBCCL_ACTIVITY_MODE", "EXO_TBCCL_ACTIVITY_DUTY", "EXO_TBCCL_ACTIVITY_MAX_MS", "EXO_TBCCL_STEP_ACTIVITY"):
        monkeypatch.delenv(k, raising=False)
    c = FastPathConfig.from_env()
    assert (c.activity_mode, c.activity_duty, c.activity_max_ms) == ("off", 1.0, 100.0)
    monkeypatch.setenv("EXO_TBCCL_ACTIVITY_MODE", "step")
    monkeypatch.setenv("EXO_TBCCL_ACTIVITY_DUTY", "0.5")
    monkeypatch.setenv("EXO_TBCCL_ACTIVITY_MAX_MS", "40")
    c = FastPathConfig.from_env()
    assert (c.activity_mode, c.activity_duty, c.activity_max_ms) == ("step", 0.5, 40.0)
    monkeypatch.delenv("EXO_TBCCL_ACTIVITY_MODE")
    monkeypatch.setenv("EXO_TBCCL_STEP_ACTIVITY", "1")  # the step-activity alias
    assert FastPathConfig.from_env().activity_mode == "step"
    monkeypatch.setenv("EXO_TBCCL_ACTIVITY_MODE", "bogus")
    with pytest.raises(ValueError):
        FastPathConfig.from_env()
    monkeypatch.setenv("EXO_TBCCL_ACTIVITY_MODE", "step")
    monkeypatch.setenv("EXO_TBCCL_ACTIVITY_DUTY", "0")
    with pytest.raises(ValueError):
        FastPathConfig.from_env()


def test_helper_idle_busy_idle_and_joined():
    base = _cores(0.2)  # other tests of a full run can leave threads behind: compare against the process' own baseline
    a = StepActivity(max_window_ms=5000)
    assert not a.thread_alive and _cores(0.2) < base + 0.15  # lazily started: no thread before the first window
    a.open()
    assert _cores(0.3) > base + 0.8  # ~1 core while the window is open
    a.close_window()
    time.sleep(0.05)
    assert _cores(0.3) < base + 0.15  # parked on the event: idle again
    a.shutdown()
    assert not a.thread_alive


def test_hard_bound_ends_a_window_by_itself():
    base = _cores(0.2)
    a = StepActivity(max_window_ms=60)
    a.open()
    time.sleep(0.3)
    assert a.timed_out == 1
    assert _cores(0.3) < base + 0.15  # the lost-event case: no core left burning
    a.shutdown()


def test_repeated_windows_and_shutdown_while_active_leak_no_thread():
    n0 = threading.active_count()
    for _ in range(100):
        a = StepActivity(max_window_ms=1000)
        a.open()
        time.sleep(0.001)
        a.close_window()
        a.open()
        a.shutdown()
        assert not a.thread_alive
        a.open()  # after shutdown: ignored, no new thread
        assert not a.thread_alive
    assert threading.active_count() == n0


def test_policy_windows_follow_the_communicator_pattern():
    p = MetalActivityPolicy(max_window_ms=5000, duty=1.0)
    a = p.activity
    p.on_recv_complete()  # prefill receive: not armed, no window
    assert a.windows == 0
    p.on_gather_submit()
    p.on_gather_complete()  # first decode gather: the previous step had a receive -> no window at the gather
    assert a.windows == 0 and p._armed
    p.on_recv_complete()  # a receiving stage opens its window at the receive
    assert a.windows == 1 and a.is_open
    p.on_gather_submit()
    assert not a.is_open
    p.on_gather_complete()
    assert a.windows == 1
    p.shutdown()
    q = MetalActivityPolicy(max_window_ms=5000, duty=1.0)  # a stage that never receives opens at the gather completion
    q.on_gather_submit()
    q.on_gather_complete()
    assert q.activity.windows == 1 and q.activity.is_open
    q.on_boundary()
    assert not q.activity.is_open
    q.shutdown()
    assert q.stats()["activity_windows"] == 1 and not q.activity.thread_alive


def test_duty_scales_helper_cpu_and_stats_report_it():
    cpu = {}
    for duty in (1.0, 0.5, 0.25):
        a = StepActivity(max_window_ms=5000, duty=duty)
        a.open()
        time.sleep(0.4)
        a.close_window()
        time.sleep(0.05)
        a.shutdown()
        cpu[duty] = a.cpu_s
        assert abs(a.active_s - 0.4) < 0.15 and a.windows == 1
    assert 0.3 < cpu[1.0] / 0.4 < 1.15
    assert 0.35 < cpu[0.5] / cpu[1.0] < 0.7
    assert 0.12 < cpu[0.25] / cpu[1.0] < 0.4


def _w_gathers(rank, world, ex, enabled):
    import os as _os

    import mlx.core as mx_
    import numpy as np

    _os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "step" if enabled else "off"
    from exo_tbccl.group import TbcclPipelineComm

    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)
    ok = True
    for i in range(20):
        x = mx_.array(np.full((1, 512), rank * 100 + i, dtype=np.float32))
        g = comm.all_gather(x)
        mx_.eval(g)
        ok &= bool(np.array_equal(np.array(g), np.concatenate([np.full((1, 512), r * 100 + i, dtype=np.float32) for r in range(world)])))
        time.sleep(0.005)
    act = comm._act
    info = (act is not None, act.activity.windows if act else 0, act.activity.thread_alive if act else False)
    stats = comm.activity_stats()
    comm.close()  # closed while a window is open
    return ok, info, (act.activity.thread_alive if act else False), stats


@pytest.mark.parametrize("enabled", [True, False])
def test_all_gather_exact_and_helper_follows_the_communicator(enabled):
    res = run_world(2, _w_gathers, enabled)
    for ok, (has_act, windows, alive), alive_after_close, stats in res:
        assert ok
        assert has_act == (enabled and METAL)  # inert off Metal and when disabled
        if has_act:
            assert windows >= 19 and alive and stats["activity_windows"] == windows
        else:
            assert stats["activity_windows"] == 0 and stats["helper_cpu_us"] == 0 and stats["activity_us"] == 0
        assert not alive_after_close


def _w_peer_death_in_window(rank, world, ex):
    import os as _os

    import mlx.core as mx_
    import numpy as np

    _os.environ["EXO_TBCCL_ACTIVITY_MODE"] = "step"
    from exo_tbccl.errors import TbcclError
    from exo_tbccl.group import TbcclPipelineComm

    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)
    g = comm.all_gather(mx_.array(np.ones((1, 64), dtype=np.float32)))
    mx_.eval(g)
    if rank == 1:
        _os._exit(0)  # dies with rank 0's window open
    got = np.zeros(1024, dtype=np.uint8)
    t0 = time.time()
    try:
        comm.wait(comm.recv_into_async(got, 1))
        name = "no error"
    except TbcclError as e:
        name = type(e).__name__
    act = comm._act
    comm.close()
    return name, time.time() - t0, (act.activity.thread_alive if act else False), (act is not None)


def test_peer_death_with_an_open_window_still_fails_promptly_and_leaves_no_thread():
    res = run_world(2, _w_peer_death_in_window, tolerate_exit=(1,))
    name, dt, alive, has = res[0]
    assert name != "no error", res
    assert dt < 20 and not alive, res

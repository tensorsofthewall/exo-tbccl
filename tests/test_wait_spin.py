"""EXO_TBCCL_WAIT_SPIN_MS failure / lifecycle / bound tests (the wait-policy work). The setting is honoured on Metal only; on CUDA/CPU hosts every test must also pass with the policy
inert (the counters then show waits but no spinning). Process-per-rank over loopback."""

import pytest

from tests.harness import run_world
from tests.test_group import ADV

mx = pytest.importorskip("mlx.core")
METAL = bool(mx.metal.is_available())


def _create(rank, world, ex, spin_ms):
    import os

    os.environ["EXO_TBCCL_WAIT_SPIN_MS"] = str(spin_ms)
    from exo_tbccl.group import TbcclPipelineComm

    return TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)


def _w_peer_death_while_spinning(rank, world, ex, spin_ms):
    import os
    import time

    import numpy as np

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, spin_ms)
    if rank == 1:
        time.sleep(0.5)
        os._exit(0)
    got = np.zeros(1024, dtype=np.uint8)
    t0 = time.time()
    try:
        comm.wait(comm.recv_into_async(got, 1))
        return ("no error", time.time() - t0, comm.wait_stats.waits_spun)
    except TbcclError as e:
        return (type(e).__name__, time.time() - t0, comm.wait_stats.waits_spun)
    finally:
        comm.close()


def test_peer_death_while_the_caller_polls_surfaces_promptly():
    res = run_world(2, _w_peer_death_while_spinning, 1000, tolerate_exit=(1,))
    name, dt, spun = res[0]
    assert name != "no error", res
    assert dt < 20, res  # a 1000 ms budget must not delay the structured error beyond the normal failure path
    assert spun == (1 if METAL else 0), res


def _w_abort_while_spinning(rank, world, ex, spin_ms):
    import threading
    import time

    import numpy as np

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, spin_ms)
    try:
        if rank == 0:
            threading.Timer(0.3, lambda: comm.abort("test abort")).start()
            got = np.zeros(1024, dtype=np.uint8)
            t0 = time.time()
            try:
                comm.wait(comm.recv_into_async(got, 1))
                return ("no error", time.time() - t0)
            except TbcclError as e:
                return (type(e).__name__, time.time() - t0)
        time.sleep(1.5)
        return None
    finally:
        comm.close()


def test_abort_from_another_thread_ends_the_polling_wait():
    res = run_world(2, _w_abort_while_spinning, 1000)
    name, dt = res[0]
    assert name != "no error", res
    assert dt < 3.0, res


def _w_close_with_outstanding_work(rank, world, ex, spin_ms):
    import threading
    import time

    import numpy as np

    comm = _create(rank, world, ex, spin_ms)
    if rank == 1:
        time.sleep(1.0)
        comm.close()
        return None
    got = np.zeros(1024, dtype=np.uint8)
    t = comm.recv_into_async(got, 1)
    th = threading.Thread(target=lambda: _swallow(comm, t))
    th.start()
    time.sleep(0.2)
    t0 = time.time()
    comm.close()
    th.join(timeout=10)
    return (time.time() - t0, th.is_alive())


def _swallow(comm, t):
    try:
        comm.wait(t)
    except Exception:  # noqa: BLE001
        pass


def test_close_with_outstanding_work_is_bounded():
    res = run_world(2, _w_close_with_outstanding_work, 1000)
    dt, alive = res[0]
    assert dt < 10 and not alive, res


def _w_long_wait_cpu(rank, world, ex, spin_ms):
    import time

    import numpy as np

    comm = _create(rank, world, ex, spin_ms)
    try:
        if rank == 1:
            time.sleep(0.6)
            comm.wait(comm.send_async(np.arange(256, dtype=np.uint8), 0))
            return None
        got = np.zeros(256, dtype=np.uint8)
        c0, t0 = time.process_time(), time.perf_counter()
        comm.wait(comm.recv_into_async(got, 1))
        cpu, wall = time.process_time() - c0, time.perf_counter() - t0
        assert bytes(got) == bytes(np.arange(256, dtype=np.uint8))
        s = comm.wait_stats
        return (cpu, wall, s.waits_spun, s.waits_fell_back, s.max_spin_us)
    finally:
        comm.close()


def test_a_long_wait_spins_only_up_to_the_budget_then_blocks():
    cpu, wall, spun, fell_back, max_spin_us = run_world(2, _w_long_wait_cpu, 8)[0]
    assert wall > 0.4
    if METAL:
        assert spun == 1 and fell_back == 1
        assert max_spin_us < 40_000, max_spin_us  # ~8 ms budget (+ scheduling slack), never the 600 ms wait
        assert cpu < 0.2, (cpu, wall)  # the other ~590 ms were a blocking wait
    else:
        assert spun == 0 and fell_back == 0


def _w_idle_cpu(rank, world, ex, spin_ms, seconds):
    import time

    comm = _create(rank, world, ex, spin_ms)
    try:
        c0, t0 = time.process_time(), time.perf_counter()
        time.sleep(seconds)
        return (time.process_time() - c0) / (time.perf_counter() - t0)
    finally:
        comm.close()


def test_idle_communicator_uses_no_cpu_with_the_policy_on():
    res = run_world(2, _w_idle_cpu, 8, 5.0)
    assert all(r < 0.02 for r in res), res


def _w_stats_decode_like(rank, world, ex, spin_ms):
    import time

    import numpy as np

    comm = _create(rank, world, ex, spin_ms)
    try:
        for _ in range(20):
            if rank == 0:
                time.sleep(0.003)
                comm.wait(comm.send_async(np.zeros(2048, dtype=np.uint8), 1))
            else:
                comm.wait(comm.recv_into_async(np.zeros(2048, dtype=np.uint8), 0))
        s = comm.wait_stats
        return (s.waits_total, s.waits_spun, s.waits_completed_during_spin, s.waits_fell_back, s.total_spin_us)
    finally:
        comm.close()


def test_decode_like_waits_complete_during_the_spin_and_are_counted():
    total, spun, completed, fell_back, spin_us = run_world(2, _w_stats_decode_like, 8)[1]
    assert total == 20
    if METAL:
        assert spun == 20 and completed >= 18 and fell_back <= 2
        assert spin_us > 0
    else:
        assert spun == 0 and completed == 0

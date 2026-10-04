"""Lifecycle, failure and threading behaviour of TbcclPipelineComm."""

import pytest

from tests.harness import run_world
from tests.test_group import ADV, _create


def _w_cycles(rank, world, ex, cycles):
    import threading

    import psutil

    proc = psutil.Process()
    # warm-up so lazy one-time allocations are not counted
    c = _create(rank, world, ex)
    c.barrier()
    ex("quiesce", b"")
    c.close()
    base_fds, base_threads = proc.num_fds(), threading.active_count()
    base_os_threads = proc.num_threads()
    for _ in range(cycles):
        c = _create(rank, world, ex)
        c.barrier()
        assert c.any_true(rank == 0)
        ex("quiesce", b"")  # every rank finished its collectives before anyone destroys its side
        c.close()
    import gc
    import time

    gc.collect()
    time.sleep(0.5)
    return proc.num_fds() - base_fds, proc.num_threads() - base_os_threads, threading.active_count() - base_threads


def test_twenty_create_destroy_cycles_do_not_leak_fds_or_threads():
    res = run_world(2, _w_cycles, 20, timeout=300)
    for fds, os_threads, py_threads in res:
        assert fds == 0 and os_threads == 0 and py_threads == 0, res


def _w_flush_and_gil(rank, world, ex):
    import threading
    import time

    import numpy as np

    comm = _create(rank, world, ex)
    try:
        n = 64 * 1024
        if rank == 0:
            sends = [(np.full(n, i, dtype=np.uint8), 1) for i in range(40)]
            ticks = {"n": 0}
            stop = threading.Event()

            def spin():
                while not stop.is_set():
                    ticks["n"] += 1
                    time.sleep(0.001)

            th = threading.Thread(target=spin)
            th.start()
            time.sleep(1.0)  # the receiver delays its receives: the flush wait must not hold the GIL
            before = ticks["n"]
            comm.flush_sends(sends)
            grew = ticks["n"] - before
            stop.set()
            th.join()
            return grew
        time.sleep(1.5)
        for i in range(40):
            got = np.zeros(n, dtype=np.uint8)
            comm.wait(comm.recv_into_async(got, 0))
            assert bytes(got) == bytes([i]) * n
        return 0
    finally:
        comm.close()


def test_flush_submits_all_sends_before_waiting_and_releases_the_gil():
    res = run_world(2, _w_flush_and_gil)
    assert res[0] > 100, res  # ~1.5 s of a 1 kHz spinner continued while rank 0 waited


def _w_peer_death(rank, world, ex, mode):
    import os
    import time

    import numpy as np

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex)
    if rank == 1:
        time.sleep(0.5)
        os._exit(0)
    got = np.zeros(1024, dtype=np.uint8)
    t0 = time.time()
    try:
        if mode == "recv":
            comm.wait(comm.recv_into_async(got, 1))
        else:
            sends = [(np.zeros(1 << 20, dtype=np.uint8), 1) for _ in range(64)]
            comm.flush_sends(sends)
            comm.barrier()
        return ("no error", time.time() - t0)
    except TbcclError as e:
        return (type(e).__name__, time.time() - t0, e.rank, e.peer, e.op)
    finally:
        comm.close()


@pytest.mark.parametrize("mode", ["recv", "flush"])
def test_peer_death_surfaces_a_structured_error_and_close_does_not_hang(mode):
    res = run_world(2, _w_peer_death, mode, timeout=120, tolerate_exit=[1])
    assert res[0][0] in ("TbcclTransportError", "TbcclAbortedError", "TbcclTimeoutError"), res
    assert res[0][1] < 60, res
    assert res[0][2] == 0 and res[0][3] == 1, res  # the error names this rank and the peer


def _w_abort_blocked_recv(rank, world, ex):
    import threading
    import time

    import numpy as np

    from exo_tbccl.errors import TbcclAbortedError

    comm = _create(rank, world, ex)
    try:
        if rank == 1:
            time.sleep(2.0)
            return "idle"
        got = np.zeros(1 << 20, dtype=np.uint8)
        t = comm.recv_into_async(got, 1)
        threading.Timer(0.3, lambda: comm.abort("test abort")).start()
        try:
            comm.wait(t)
        except TbcclAbortedError as e:
            return "aborted" if e.rank == 0 and e.peer == 1 else f"bad context {e.rank} {e.peer}"
        return "no error"
    finally:
        comm.close()


def test_abort_unblocks_a_waiting_recv_with_an_aborted_error():
    res = run_world(2, _w_abort_blocked_recv)
    assert res[0] == "aborted", res


def _w_close_with_inflight(rank, world, ex):
    import numpy as np

    comm = _create(rank, world, ex)
    if rank == 1:
        import time

        time.sleep(1.0)
        comm.close()
        return "peer closed"
    ts = [comm.send_async(np.zeros(1 << 20, dtype=np.uint8), 1) for _ in range(50)]
    comm.close()  # abort + drain + release every borrow; must not hang or crash
    return all(t.terminal for t in ts)


def test_close_drains_inflight_work_and_releases_borrows():
    res = run_world(2, _w_close_with_inflight)
    assert res[0] is True

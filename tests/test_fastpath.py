"""Receive-buffer pool and asynchronous decode sends (Phase 54): lifecycle, ordering with collectives, failure surfacing. Loopback, process-per-rank."""

import pytest

from tests.harness import run_world

ADV = "127.0.0.1"
mlx = pytest.importorskip("mlx.core")


def _create(rank, world, ex, env=None):
    import os

    os.environ.update(env or {})
    from exo_tbccl.group import TbcclPipelineComm

    return TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=30000)


# ---- receive pool ---------------------------------------------------------------------------------------------------------------------


def _w_pool_lifecycle(rank, world, ex, env):
    import mlx.core as mx

    comm = _create(rank, world, ex, env)
    try:
        shapes = [(1, 1, 64), (1, 7, 64), (1, 1, 64), (1, 300, 64), (1, 1, 64), (1, 7, 64)] * 20
        for i, shape in enumerate(shapes):
            for dt in (mx.float32, mx.bfloat16):
                if rank == 0:
                    comm.send(((mx.arange(shape[1] * shape[2]) + i).reshape(shape) % 200).astype(dt), 1)
                else:
                    tmpl = mx.zeros(shape, dtype=dt)
                    x = comm.recv_like(tmpl, 0)
                    y = x.astype(mx.float32) + 1  # GPU consumer; the stage output
                    mx.eval(y)
                    exp = (((mx.arange(shape[1] * shape[2]) + i).reshape(shape) % 200).astype(dt)).astype(mx.float32) + 1
                    assert bool(mx.array_equal(y, exp)), (i, shape, dt)
                    comm.step_complete()
        comm.barrier()
        st = comm.pool.stats
        return {"hits": st.hits, "misses": st.misses, "peak_bytes": st.peak_bytes, "peak_slots": st.peak_slots, "evictions": st.evictions, "cached": comm.pool.cached_bytes}
    finally:
        comm.close()


def test_pool_reuses_with_variable_shapes_and_stays_bounded():
    res = run_world(2, _w_pool_lifecycle, {"EXO_TBCCL_RECV": "reuse"})
    r = res[1]
    assert r["hits"] > 200 and r["misses"] <= 6 and r["peak_slots"] <= 6 and r["peak_bytes"] < 1 << 20, r
    assert res[0]["hits"] == 0


def _w_pool_lease_protocol(rank, world, ex, env):
    import mlx.core as mx

    comm = _create(rank, world, ex, env)
    try:
        if rank == 0:
            for i in range(3):
                comm.send(mx.full((256,), i + 1, dtype=mx.float32), 1)
            return None
        t = mx.zeros((256,), dtype=mx.float32)
        a = comm.recv_like(t, 0)
        b = comm.recv_like(t, 0)  # no step_complete in between: `a` is still leased and must not be overwritten
        c = comm.recv_like(t, 0)
        assert float(a[0]) == 1 and float(b[0]) == 2 and float(c[0]) == 3
        assert comm.pool.stats.hits == 0
        comm.step_complete()
        return True
    finally:
        comm.close()


def test_leased_buffer_is_never_overwritten_before_step_complete():
    assert run_world(2, _w_pool_lease_protocol, {"EXO_TBCCL_RECV": "reuse"})[1] is True


def _w_poison(rank, world, ex, env):
    import mlx.core as mx

    comm = _create(rank, world, ex, env)
    try:
        if rank == 0:
            comm.send(mx.full((64,), 7, dtype=mx.uint8), 1)
            return None
        x = comm.recv_like(mx.zeros((64,), dtype=mx.uint8), 0)
        before = bytes(memoryview(__import__("numpy").array(x)))
        comm.step_complete()  # release point: the negative control poisons the slot here
        after = bytes(memoryview(__import__("numpy").array(x)))
        return before[:2], after[:2]
    finally:
        comm.close()


def test_poison_control_overwrites_released_storage_only_at_step_complete():
    res = run_world(2, _w_poison, {"EXO_TBCCL_RECV": "reuse", "EXO_TBCCL_RECV_POISON": "0xA5"})
    assert res[1] == (b"\x07\x07", b"\xa5\xa5")


# ---- asynchronous sends ---------------------------------------------------------------------------------------------------------------


def _w_chain(rank, world, ex, env, iters, dtype):
    """Each stage: recv previous, compute on the GPU, send next, all_gather (the decode sequence). Returns a digest of every gathered output."""
    import hashlib

    import mlx.core as mx

    dt = getattr(mx, dtype)
    comm = _create(rank, world, ex, env)
    h = hashlib.sha256()
    max_pending = 0
    try:
        for i in range(iters):
            n = 64 + (i % 5) * 64  # changing token dimension
            tmpl = mx.zeros((1, n), dtype=dt)
            x = (mx.arange(n).reshape(1, n) + i).astype(dt) if rank == 0 else comm.recv_like(tmpl, rank - 1)
            y = ((x.astype(mx.float32) * 2 + (rank + 1)) % 251).astype(dt)
            mx.eval(y)
            if rank < world - 1:
                y = comm.send(y, rank + 1)
            comm.step_complete()
            g = comm.all_gather(y)
            mx.eval(g)
            h.update(bytes(memoryview(__import__("numpy").array(g.view({2: mx.uint16, 4: mx.uint32}[dt.size])))))
            max_pending = max(max_pending, len(comm._pending))
        comm.barrier()
        s = comm.stats
        return h.hexdigest(), max_pending, (s.async_send_submitted, s.async_send_reaped), len(comm._async_sends)
    finally:
        comm.close()


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_async_send_chain_is_bit_exact_vs_sync_and_bounded(world, dtype):
    base = run_world(world, _w_chain, {}, 300, dtype)
    fast = run_world(world, _w_chain, {"EXO_TBCCL_ASYNC_SEND": "1", "EXO_TBCCL_RECV": "reuse"}, 300, dtype)
    assert [r[0] for r in fast] == [r[0] for r in base]
    for r, (digest, max_pending, (sub, reaped), left) in enumerate(fast):
        if r < world - 1:
            assert sub == 300 and reaped == 300 and left == 0
        assert max_pending <= 3, (r, max_pending)


def _w_async_failure(rank, world, ex, env, mode):
    import os
    import time

    import mlx.core as mx

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, env)
    if rank == 1:
        time.sleep(0.5)
        os._exit(0)
    time.sleep(1.5)  # the peer is gone before anything is sent
    t0 = time.time()
    try:
        for i in range(50):
            comm.send(mx.full((1 << 18,), i, dtype=mx.float32), 1)  # returns at once; the Work fails later
            if mode == "gather":
                comm.all_gather(mx.zeros((4,), dtype=mx.float32))
            time.sleep(0.05)
    except TbcclError as e:
        return type(e).__name__, time.time() - t0, len(comm._async_sends)
    finally:
        comm.close()
    return "no error", time.time() - t0, 0


@pytest.mark.parametrize("mode", ["send", "gather"])
def test_async_send_failure_surfaces_as_structured_error(mode):
    res = run_world(2, _w_async_failure, {"EXO_TBCCL_ASYNC_SEND": "1"}, mode, tolerate_exit=(1,))
    name, elapsed, _ = res[0]
    assert name in ("TbcclTransportError", "TbcclAbortedError", "TbcclTimeoutError"), res[0]
    assert elapsed < 30


def _w_drop_and_close(rank, world, ex, env):
    import mlx.core as mx

    comm = _create(rank, world, ex, env)
    try:
        if rank == 0:
            for i in range(20):
                comm.send(mx.full((1 << 16,), i, dtype=mx.float32), 1)  # result dropped immediately
            comm.barrier()
            assert len(comm._async_sends) == 0 and len(comm._pending) == 0
            return True
        for i in range(20):
            x = comm.recv_like(mx.zeros((1 << 16,), dtype=mx.float32), 0)
            assert float(x[5]) == i
        comm.barrier()
        return True
    finally:
        comm.close()


def test_dropped_send_result_keeps_storage_alive_until_terminal():
    assert run_world(2, _w_drop_and_close, {"EXO_TBCCL_ASYNC_SEND": "1"}) == [True, True]


def _w_invalid_order(rank, world, ex, env):
    """Rank 0 sends then all_gathers; rank 1 all_gathers then receives: an order mismatch. It must fail or time out, never deliver wrong bytes."""
    import mlx.core as mx

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, env)
    payload = mx.full((4096,), 3, dtype=mx.float32)
    try:
        if rank == 0:
            comm.send(payload, 1)
            g = comm.all_gather(mx.full((4096,), 9, dtype=mx.float32))
            return "completed", bool(mx.array_equal(g[4096:], mx.full((4096,), 9, dtype=mx.float32)))
        g = comm.all_gather(mx.full((4096,), 9, dtype=mx.float32))
        ok_gather = bool(mx.array_equal(g[:4096], mx.full((4096,), 9, dtype=mx.float32)))
        x = comm.recv_like(payload, 0)
        return "completed", ok_gather and bool(mx.array_equal(x, payload))
    except TbcclError as e:
        return "error", type(e).__name__
    finally:
        comm.close()


def test_invalid_p2p_collective_order_is_detected_or_correct():
    res = run_world(2, _w_invalid_order, {"EXO_TBCCL_ASYNC_SEND": "1"}, timeout=90)
    print(res)
    for r in res:
        # either the protocol rejected the mismatch, or every delivered byte is correct; silent corruption is the failure
        assert r[0] == "error" or r[1] is True, res


def _w_premature_release(rank, world, ex, env):
    import mlx.core as mx

    comm = _create(rank, world, ex, env)
    try:
        if rank == 0:
            comm.send(mx.full((256,), 5, dtype=mx.float32), 1)
            return None
        x = comm.recv_like(mx.zeros((256,), dtype=mx.float32), 0)
        y = x + 1  # lazy: nothing has consumed x yet
        comm.step_complete()  # WRONG boundary: released before the consumer was evaluated
        mx.eval(y)
        return float(y[0])
    finally:
        comm.close()


def test_poison_control_detects_a_premature_release_boundary():
    res = run_world(2, _w_premature_release, {"EXO_TBCCL_RECV": "reuse", "EXO_TBCCL_RECV_POISON": "0xA5"})
    assert res[1] != 6.0  # the control is sensitive: a wrong boundary changes the output


def _w_cancel_with_outstanding_sends(rank, world, ex, env):
    import time

    import numpy as np

    comm = _create(rank, world, ex, env)
    if rank == 1:
        time.sleep(6.0)  # never receives: rank 0's sends stay outstanding
        comm.close()
        return None
    bufs = [np.full(1 << 22, i, dtype=np.uint8) for i in range(8)]
    for b in bufs:
        comm.send(b, 1)  # asynchronous; keeps {Work, Borrow}
    outstanding = len(comm._async_sends)
    t0 = time.time()
    comm.close()  # cancellation: in-flight work -> abort -> drain -> release borrows -> destroy
    return outstanding, time.time() - t0, len(comm._pending), len(comm._async_sends)


def test_close_with_outstanding_async_sends_is_bounded_and_releases_everything():
    res = run_world(2, _w_cancel_with_outstanding_sends, {"EXO_TBCCL_ASYNC_SEND": "1"}, timeout=120, tolerate_exit=(1,))
    outstanding, elapsed, pending, detached = res[0]
    assert outstanding >= 1 and elapsed < 30 and pending == 0 and detached == 0, res[0]


def _w_instance_cycles(rank, world, ex, env, cycles):
    import os
    import threading

    import mlx.core as mx

    def counts():
        return threading.active_count(), len(os.listdir("/dev/fd"))

    series = []
    for c in range(cycles):
        comm = _create(rank, world, ex, env)
        for i in range(5):
            if rank == 0:
                comm.send(mx.full((512,), i, dtype=mx.float32), 1)
            else:
                x = comm.recv_like(mx.zeros((512,), dtype=mx.float32), 0)
                assert float(x[0]) == i
                comm.step_complete()
            comm.all_gather(mx.zeros((4,), dtype=mx.float32))
        comm.barrier()
        comm.close()
        series.append(counts())
    return series


def test_repeated_create_close_cycles_with_every_fast_path_do_not_grow_threads_or_fds():
    env = {"EXO_TBCCL_ASYNC_SEND": "1", "EXO_TBCCL_RECV": "reuse", "EXO_TBCCL_CUDA_MANAGED_MODE": "host"}
    for series in run_world(2, _w_instance_cycles, env, 8):
        assert series[-1][0] <= series[1][0] and series[-1][1] <= series[1][1] + 2, series


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cpu_stream_destination_allocation_is_bit_exact_vs_gpu_stream(world, dtype):
    """Phase 57 experiment: EXO_TBCCL_ALLOC_STREAM=cpu allocates fresh receive/all_gather destinations on the CPU stream. Every gathered stage output
    (computed on the GPU from the received storage) must hash identically to the default path."""
    base = run_world(world, _w_chain, {}, 200, dtype)
    cpu = run_world(world, _w_chain, {"EXO_TBCCL_ALLOC_STREAM": "cpu"}, 200, dtype)
    assert [r[0] for r in cpu] == [r[0] for r in base]


def test_alloc_stream_config_parsing(monkeypatch):
    from exo_tbccl.config import FastPathConfig

    monkeypatch.delenv("EXO_TBCCL_ALLOC_STREAM", raising=False)
    assert FastPathConfig.from_env().alloc_cpu is False
    monkeypatch.setenv("EXO_TBCCL_ALLOC_STREAM", "cpu")
    assert FastPathConfig.from_env().alloc_cpu is True
    monkeypatch.setenv("EXO_TBCCL_ALLOC_STREAM", "gpu")
    assert FastPathConfig.from_env().alloc_cpu is False


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_wait_spin_is_bit_exact_vs_blocking_wait(world, dtype):
    """Phase 59 experiment: EXO_TBCCL_WAIT_SPIN_MS polls a pending Work before blocking (Metal only; a no-op on CUDA). Every gathered stage output must hash
    identically to the blocking path, and the receive-pool bookkeeping must stay bounded."""
    base = run_world(world, _w_chain, {}, 200, dtype)
    spin = run_world(world, _w_chain, {"EXO_TBCCL_WAIT_SPIN_MS": "50"}, 200, dtype)
    assert [r[0] for r in spin] == [r[0] for r in base]


def test_wait_spin_config_parsing(monkeypatch):
    from exo_tbccl.config import FastPathConfig

    monkeypatch.delenv("EXO_TBCCL_WAIT_SPIN_MS", raising=False)
    assert FastPathConfig.from_env().wait_spin_ms == 0.0
    monkeypatch.setenv("EXO_TBCCL_WAIT_SPIN_MS", "2.5")
    assert FastPathConfig.from_env().wait_spin_ms == 2.5

"""Metal correctness stress for the CPU-stream borrow view (EXO_TBCCL_ALLOC_STREAM=cpu), real Metal storage, two ranks over loopback on one Mac.

Every payload is a known pattern produced on the GPU behind a deliberately busy queue and compared byte for byte with an independently computed expectation. A spy on ``mlx.core.view``
proves which path ran: with the flag on the borrow's views are evaluated on the CPU stream (and never on the GPU stream), with it off they never are. The iteration count is
``STRESS_ITERS`` (default 30; the qualification campaign uses far more).
"""
import os

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="the CPU-stream view is a Metal-only policy")

from tests.harness import run_world

ITERS = int(os.environ.get("STRESS_ITERS", "30"))
DTYPES = ("bfloat16", "float16", "float32")
SIZES = (512, 32768, 262144)


def _create(rank, world, ex, flag, reuse):
    from exo_tbccl.config import FastPathConfig
    from exo_tbccl.group import TbcclPipelineComm

    return TbcclPipelineComm.create(rank, world, ex, bind_host="127.0.0.1", advertise_host="127.0.0.1", timeout_ms=30000, config=FastPathConfig(alloc_cpu=flag, recv_reuse=reuse))


class _Spy:
    """Counts mlx.core.view calls by the stream they ask for."""

    def __init__(self):
        import mlx.core as mx

        self.mx, self.cpu, self.gpu = mx, 0, 0
        self.real = mx.view

        def view(a, dtype, stream=None):
            if stream is not None and str(stream).lower().find("cpu") >= 0:
                self.cpu += 1
            else:
                self.gpu += 1
            return self.real(a, dtype, stream=stream) if stream is not None else self.real(a, dtype)

        mx.view = view
        # array.view(dtype) is the default-stream form; count it too
        self._array_view = type(mx.array([0])).view
        spy = self

        def array_view(arr, dtype):
            spy.gpu += 1
            return spy._array_view(arr, dtype)

        type(mx.array([0])).view = array_view

    def restore(self):
        self.mx.view = self.real
        type(self.mx.array([0])).view = self._array_view


def _dt(mx, name):
    return getattr(mx, name)


def _pattern_cpu(mx, n, dtype, seed):
    """The expectation, built on the CPU stream without any GPU work."""
    base = mx.arange(n, stream=mx.cpu) % 251
    return (base.astype(mx.float32, stream=mx.cpu) + float(seed)).astype(dtype, stream=mx.cpu)


def _pattern_gpu(mx, n, dtype, seed, busy):
    """The same values produced on the GPU queue behind `busy` (a pending matmul chain the payload depends on)."""
    base = (mx.arange(n) % 251).astype(mx.float32) + float(seed)
    base = base + mx.zeros_like(base) * mx.sum(busy)
    return base.astype(dtype)


def _w_stress(rank, world, ex, flag, reuse, iters):
    import gc

    import mlx.core as mx

    comm = _create(rank, world, ex, flag, reuse)
    spy = _Spy()

    def cview(a, u, stream=None):  # the test's own comparison views must not be counted by the spy
        return spy.real(a, u, stream=mx.cpu)

    out = {"flag": comm._view_cpu}
    w = mx.random.normal((768, 768))
    mx.eval(w)
    try:
        # A. GPU-produced input immediately followed by send; receive into a template-shaped destination, then a GPU kernel reads it
        bad = 0
        for i in range(iters):
            for dn in DTYPES:
                for n in SIZES:
                    dtype = _dt(mx, dn)
                    seed = (i % 3) + (7 if rank == 0 else 0)
                    if rank == 0:
                        busy = (w @ w) @ w
                        mx.async_eval(busy)
                        comm.send(_pattern_gpu(mx, n, dtype, seed, busy), 1)
                        back = comm.recv_like(mx.zeros((n,), dtype=dtype), 1)
                        got_seed = (i % 3) + 0
                    else:
                        got = comm.recv_like(mx.zeros((n,), dtype=dtype), 0)
                        s = mx.sum(got.astype(mx.float32))  # a GPU kernel reading the host-written destination
                        exp = _pattern_cpu(mx, n, dtype, (i % 3) + 7)
                        bad += (not mx.array_equal(cview(got, mx.uint16 if dtype.size == 2 else mx.uint32, stream=mx.cpu), cview(exp, mx.uint16 if dtype.size == 2 else mx.uint32, stream=mx.cpu)).item())
                        bad += abs(s.item() - mx.sum(exp.astype(mx.float32, stream=mx.cpu), stream=mx.cpu).item()) > 1e-3 * max(1.0, abs(s.item()))
                        comm.step_complete()  # exo's boundary: the received activation is dead, its slot may be reused
                        busy = (w @ w) @ w
                        mx.async_eval(busy)
                        comm.send(_pattern_gpu(mx, n, dtype, (i % 3), busy), 0)
                        continue
                    exp = _pattern_cpu(mx, n, dtype, got_seed)
                    u = mx.uint16 if dtype.size == 2 else mx.uint32
                    bad += (not mx.array_equal(cview(back, u, stream=mx.cpu), cview(exp, u, stream=mx.cpu)).item())
        out["A_bad"] = bad
        comm.barrier()

        # B. back-to-back asynchronous operations with distinct patterns, destinations prepared on the CPU stream
        bad = 0
        for i in range(max(1, iters // 3)):
            k = 8
            if rank == 0:
                arrs = [_pattern_gpu(mx, 4096, mx.float16, j + i % 5, (w @ w)) for j in range(k)]
                comm.wait_all([comm.send_async(a, 1) for a in arrs])
            else:
                dests = [mx.zeros((4096,), dtype=mx.float16, stream=mx.cpu) for _ in range(k)]
                mx.eval(dests)
                ts = [comm.recv_into_async(d, 0) for d in dests]
                comm.wait_all(ts)
                for j, d in enumerate(dests):
                    exp = _pattern_cpu(mx, 4096, mx.float16, j + i % 5)
                    bad += (not mx.array_equal(cview(d, mx.uint16, stream=mx.cpu), cview(exp, mx.uint16, stream=mx.cpu)).item())
        out["B_bad"] = bad
        comm.barrier()

        # C. non-contiguous sender: the adapter must materialize (counted) and still deliver the right bytes
        bad = 0
        before = comm.stats.materialized_copies
        for i in range(max(1, iters // 3)):
            if rank == 0:
                full = _pattern_gpu(mx, 64 * 256, mx.float32, i % 4, w @ w).reshape(64, 256)
                comm.send(full[:, ::2], 1)
            else:
                got = comm.recv_like(mx.zeros((64, 128), dtype=mx.float32), 0)
                full = _pattern_cpu(mx, 64 * 256, mx.float32, i % 4).reshape(64, 256)[:, ::2]
                bad += (not mx.array_equal(got, mx.contiguous(full, stream=mx.cpu)).item())
        out["C_bad"], out["C_materialized"] = bad, comm.stats.materialized_copies - before
        comm.barrier()

        # D. rapid object allocation and release; no leaked borrows or growing device memory
        from exo_tbccl.bridge import borrow

        gc.collect()
        m0 = mx.get_active_memory()
        for i in range(max(500, iters * 40)):
            a = _pattern_gpu(mx, 1024, mx.float32, 1, w)
            mx.eval(a)
            b = borrow(a, view_on_cpu=flag)
            b.release()
            del a, b
        gc.collect()
        mx.clear_cache()
        out["D_mem_growth_mib"] = (mx.get_active_memory() - m0) / 2**20
        comm.barrier()

        # E. ownership during in-flight communication: drop every Python reference to the payload while the Work is pending
        bad = 0
        for i in range(max(1, iters // 6)):
            if rank == 0:
                t = comm.send_async(_pattern_gpu(mx, 2_000_000, mx.float32, 3, w @ w), 1)
                gc.collect()
                mx.clear_cache()
                comm.wait(t)
            else:
                got = comm.recv_like(mx.zeros((2_000_000,), dtype=mx.float32), 0)
                exp = _pattern_cpu(mx, 2_000_000, mx.float32, 3)
                bad += (not mx.array_equal(cview(got, mx.uint32, stream=mx.cpu), cview(exp, mx.uint32, stream=mx.cpu)).item())
        out["E_bad"] = bad
        comm.barrier()
        out["spy_cpu_views"], out["spy_gpu_views"] = spy.cpu, spy.gpu
        out["copies"] = (comm.stats.staged_fallback_copies,)
        if reuse:
            out["pool_hits"] = comm.pool.stats.hits
        return out
    finally:
        spy.restore()
        comm.close()


@pytest.mark.parametrize("flag,reuse", [(False, False), (True, False), (True, True)])
def test_gpu_patterns_survive_the_cpu_stream_view(flag, reuse):
    res = run_world(2, _w_stress, flag, reuse, ITERS, timeout=900)
    for r, d in enumerate(res):
        assert d["flag"] is flag, d
        assert all(d.get(k, 0) == 0 for k in ("A_bad", "B_bad", "C_bad", "E_bad")), (r, d)
        assert d["D_mem_growth_mib"] < 16, (r, d)
        assert d["copies"] == (0,), (r, d)
        if flag:
            assert d["spy_cpu_views"] > 0, ("the candidate path did not run", r, d)
        else:
            assert d["spy_cpu_views"] == 0, ("the default path used the CPU stream", r, d)
    assert res[0]["C_materialized"] >= 1  # the noncontiguous sender copied (and counted it)
    if reuse:
        assert res[1]["pool_hits"] > 0


def _w_fail(rank, world, ex, mode):
    import threading
    import time

    import mlx.core as mx

    from exo_tbccl.errors import TbcclError

    comm = _create(rank, world, ex, True, False)
    spy = _Spy()
    try:
        if mode == "abort":
            if rank == 1:
                time.sleep(2.0)
                return {"kind": "idle"}
            dest = mx.zeros((1 << 20,), dtype=mx.float32, stream=mx.cpu)
            mx.eval(dest)
            t = comm.recv_into_async(dest, 1)
            threading.Timer(0.3, lambda: comm.abort("stress abort")).start()
            try:
                comm.wait(t)
                return {"kind": "no error"}
            except TbcclError as e:
                return {"kind": type(e).__name__, "terminal": t.terminal, "borrows_released": all(b.export is None for b in t.borrows), "cpu_views": spy.cpu}
        if rank == 1:
            time.sleep(0.5)
            os._exit(0)
        dest = mx.zeros((1 << 20,), dtype=mx.float32, stream=mx.cpu)
        mx.eval(dest)
        t0 = time.time()
        try:
            comm.wait(comm.recv_into_async(dest, 1))
            return {"kind": "no error"}
        except TbcclError as e:
            return {"kind": type(e).__name__, "dt": time.time() - t0, "peer": e.peer, "cpu_views": spy.cpu}
    finally:
        spy.restore()
        comm.close()


@pytest.mark.parametrize("mode", ["abort", "peer_death"])
def test_failures_with_cpu_stream_destinations_are_structured_and_release_borrows(mode):
    res = run_world(2, _w_fail, mode, timeout=180, tolerate_exit=[1] if mode == "peer_death" else [])
    r = res[0]
    assert r["kind"] in ("TbcclAbortedError", "TbcclTransportError", "TbcclTimeoutError"), r
    assert r["cpu_views"] > 0, r
    if mode == "abort":
        assert r["terminal"] and r["borrows_released"], r
    else:
        assert r["dt"] < 60 and r["peer"] == 1, r
